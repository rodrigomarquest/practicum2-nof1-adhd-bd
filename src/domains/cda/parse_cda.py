from __future__ import annotations

from pathlib import Path
from typing import Optional, Dict
import xml.etree.ElementTree as ET
import os
import json
import csv
from datetime import datetime, timezone
import logging

# optional tqdm
try:
    from tqdm import tqdm as _tqdm  # type: ignore
except Exception:  # pragma: no cover
    def _tqdm(it, *a, **k):
        return it


def _strip_ns(tag: str) -> str:
    if '}' in tag:
        return tag.split('}', 1)[1]
    return tag


def cda_probe(path: Path, home_tz: Optional[str] = None) -> Dict:
    """Streaming probe of export_cda.xml. Returns summary dict.

    Behaviors:
      - Try strict parse using xml.etree.ElementTree.iterparse (streaming).
      - On ParseError, attempt recovery using lxml with recover=True (if available).
      - If lxml not available or fails, attempt a light salvage: strip UTF-8 BOM and keep first
        </ClinicalDocument> block, then parse that.
      - Never raise; always return a dict with counts and optional 'recover' or 'error' keys.

    summary: {"n_section": int, "n_observation": int, "codes": {code: count}}
    """
    logger = logging.getLogger("etl.extract")

    def _count_from_iterable(itable):
        n_s = 0
        n_o = 0
        cs: Dict[str, int] = {}
        for elem in itable:
            tag = _strip_ns(elem.tag) if getattr(elem, 'tag', None) else ''
            if tag.lower().endswith('section'):
                n_s += 1
            if tag.lower().endswith('observation'):
                n_o += 1
                code_elem = None
                for child in elem:
                    t = _strip_ns(child.tag)
                    if t.lower() == 'code':
                        code_elem = child
                        break
                if code_elem is not None:
                    code = code_elem.attrib.get('code') or code_elem.attrib.get('displayName') or 'unknown'
                    cs[code] = cs.get(code, 0) + 1
        return n_s, n_o, cs

    disable = not (os.getenv("ETL_TQDM") == "1" and not os.getenv("CI"))

    # 1) strict streaming parse
    try:
        it = _tqdm(ET.iterparse(str(path), events=("end",)), desc="CDA export_cda.xml", disable=disable)
        n_section, n_observation, codes = _count_from_iterable((elem for _, elem in it))
        return {
            "n_section": n_section,
            "n_observation": n_observation,
            "codes": codes,
            "recover": {"used": False, "method": None, "notes": None},
            "error": None,
            "error_type": None,
            "error_msg": None,
        }
    except ET.ParseError as pe:
        logger.warning("CDA strict parse failed (%s); attempting recovery", pe)

        # 2) try lxml recover if available
        try:
            from lxml import etree as LET  # type: ignore

            with open(path, 'rb') as fh:
                data = fh.read()
            try:
                parser = LET.XMLParser(recover=True)
                root = LET.fromstring(data, parser=parser)
            except Exception as e_lxml:
                logger.warning("lxml recovery failed: %s", e_lxml)
                root = None
            if root is not None:
                # iterate over elements
                n_s = 0
                n_o = 0
                cs: Dict[str, int] = {}
                for elem in root.iter():
                    tag = _strip_ns(elem.tag) if getattr(elem, 'tag', None) else ''
                    if tag.lower().endswith('section'):
                        n_s += 1
                    if tag.lower().endswith('observation'):
                        n_o += 1
                        code_elem = None
                        for child in elem:
                            t = _strip_ns(child.tag)
                            if t.lower() == 'code':
                                code_elem = child
                                break
                        if code_elem is not None:
                            code = code_elem.attrib.get('code') or code_elem.attrib.get('displayName') or 'unknown'
                            cs[code] = cs.get(code, 0) + 1
                logger.warning("CDA recovered using lxml.recover: sections=%d obs=%d", n_s, n_o)
                return {
                    "n_section": n_s,
                    "n_observation": n_o,
                    "codes": cs,
                    "recover": {"used": True, "method": "lxml", "notes": "recover=True used"},
                    "error": None,
                    "error_type": None,
                    "error_msg": None,
                }
        except Exception:
            # lxml not available; fallthrough to salvage
            pass

        # 3) salvage: strip BOM and keep first ClinicalDocument block
        try:
            raw = Path(path).read_bytes()
            if raw.startswith(b'\xef\xbb\xbf'):
                raw = raw[3:]
            text = raw.decode('utf-8', errors='replace')
            end_tag = '</ClinicalDocument>'
            idx = text.find(end_tag)
            if idx != -1:
                salvaged = text[: idx + len(end_tag)]
            else:
                salvaged = text
            try:
                root = ET.fromstring(salvaged)
            except Exception as e_salv:
                logger.warning("Salvage parse failed: %s", e_salv)
                # indicate salvage was attempted but failed
                return {
                    "n_section": 0,
                    "n_observation": 0,
                    "codes": {},
                    "recover": {"used": True, "method": "salvage_first_root", "notes": f"failed: {e_salv}"},
                    "error": str(pe),
                    "error_type": type(pe).__name__,
                    "error_msg": str(pe),
                }
            n_s, n_o, cs = _count_from_iterable(root.iter())
            logger.warning("CDA recovered using salvage_first_root: sections=%d obs=%d", n_s, n_o)
            return {
                "n_section": n_s,
                "n_observation": n_o,
                "codes": cs,
                "recover": {"used": True, "method": "salvage_first_root", "notes": "kept first </ClinicalDocument> block"},
                "error": None,
                "error_type": None,
                "error_msg": None,
            }
        except Exception as e_final:
            logger.warning("CDA recovery failed: %s", e_final)
            return {
                "n_section": 0,
                "n_observation": 0,
                "codes": {},
                "recover": {"used": True, "method": None, "notes": str(e_final)},
                "error": str(pe),
                "error_type": type(pe).__name__,
                "error_msg": str(pe),
            }


def write_cda_qc(out_dir: Path, summary: Dict) -> None:
    # Ensure snapshot qc dir exists and enrich summary with metadata
    qc_dir = out_dir / "qc"
    qc_dir.mkdir(parents=True, exist_ok=True)
    json_path = qc_dir / "cda_summary.json"
    csv_path = qc_dir / "cda_summary.csv"

    # Enrich summary with standard fields
    enriched = dict(summary or {})
    enriched.setdefault("snapshot_dir", str(out_dir))
    enriched.setdefault("source_path", enriched.get("source_path", ""))
    enriched.setdefault("home_tz", enriched.get("home_tz", "unknown") or "unknown")
    # Normalize status
    n_obs = int(enriched.get("n_observation", 0) or 0)
    if "status" not in enriched:
        enriched["status"] = "ok" if n_obs > 0 else "empty"
    # error fields
    if enriched.get("status") == "error":
        enriched.setdefault("error_type", enriched.get("error_type", enriched.get("error", {}).get("type") if isinstance(enriched.get("error"), dict) else None))
        enriched.setdefault("error_msg", enriched.get("error_msg", enriched.get("error", "") if enriched.get("error") else ""))
    else:
        enriched.setdefault("error_type", None)
        enriched.setdefault("error_msg", None)

    enriched.setdefault("n_section", int(enriched.get("n_section", 0) or 0))
    enriched.setdefault("n_observation", n_obs)
    enriched.setdefault("codes", enriched.get("codes", {}) or {})
    # normalize recover block
    rec = enriched.get("recover") or {}
    enriched["recover"] = {
        "used": bool(rec.get("used", False)),
        "method": rec.get("method") if rec.get("method") is not None else None,
        "notes": rec.get("notes") if rec.get("notes") is not None else None,
    }
    # timezone-aware UTC timestamp, drop microseconds and normalize +00:00 -> Z
    enriched["ts_utc"] = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")

    # write JSON
    try:
        with json_path.open("w", encoding="utf-8") as fh:
            json.dump(enriched, fh, indent=2, ensure_ascii=False)
    except Exception:
        pass

    # write CSV with two blocks: key,value lines then code,count
    try:
        with csv_path.open("w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            # header
            w.writerow(["key", "value"])
            # core keys in deterministic order
            core_keys = [
                "snapshot_dir",
                "source_path",
                "home_tz",
                "status",
                "error_type",
                "error_msg",
                "n_section",
                "n_observation",
                # recovery flat fields
                "recover_used",
                "recover_method",
                "recover_notes",
                "ts_utc",
            ]
            for k in core_keys:
                if k == "recover_used":
                    v = str(bool(enriched.get("recover", {}).get("used", False))).lower()
                elif k == "recover_method":
                    v = enriched.get("recover", {}).get("method") or ""
                elif k == "recover_notes":
                    v = enriched.get("recover", {}).get("notes") or ""
                else:
                    v = enriched.get(k, "")
                if v is None:
                    v = ""
                w.writerow([k, v])
            # blank separator
            w.writerow([])
            w.writerow(["code", "count"])
            codes = enriched.get("codes", {}) or {}
            for code, cnt in sorted(codes.items(), key=lambda x: (-x[1], x[0])):
                w.writerow([code, cnt])
    except Exception:
        pass
