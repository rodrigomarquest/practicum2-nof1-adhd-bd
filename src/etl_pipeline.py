# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path
import re
import json
import tempfile
from datetime import datetime, timezone
from typing import Optional, Iterable, Tuple, Any, Dict, Union
import xml.etree.ElementTree as ET
import traceback
import zipfile
import shutil
import time
import os
import sys
import math
import warnings
import argparse

# third-party
import pandas as pd
import numpy as np

# local
from src.etl.common.progress import Timer, progress_bar
from src.domains.common.io import migrate_from_data_ai_if_present, etl_snapshot_root
from lib.io_guards import ensure_joined_snapshot, write_joined_features
from src.domains.enriched.pre import enrich_prejoin_run


# ----------------------------------------------------------------------
# Atomic write helpers
# ----------------------------------------------------------------------
def _write_atomic_csv(df: "pd.DataFrame", out_path: str | os.PathLike[str]):
    """Prefer canonical atomic writer from lib.io_guards, fallback to local atomic rename."""
    try:
        from lib.io_guards import atomic_backup_write  # type: ignore

        atomic_backup_write(df, Path(out_path))
        return
    except Exception:
        # fallback
        d = os.path.dirname(str(out_path)) or "."
        fd, tmp = tempfile.mkstemp(prefix=".tmp_", dir=d, suffix=".csv")
        os.close(fd)
        try:
            # Use pandas to write the DataFrame to the temporary file, then atomically replace
            df.to_csv(tmp, index=False)
            os.replace(tmp, str(out_path))
        finally:
            if os.path.exists(tmp):
                try:
                    os.remove(tmp)
                except Exception:
                    pass


def _write_atomic_json(obj: dict, out_path: str | os.PathLike[str]):
    d = os.path.dirname(out_path) or "."
    fd, tmp = tempfile.mkstemp(prefix=".tmp_", dir=d, suffix=".json")
    os.close(fd)
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
        os.replace(tmp, out_path)
    finally:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except Exception:
                pass


# ----------------------------------------------------------------------
# Paths e snapshots
# ----------------------------------------------------------------------
RAW_ROOT = Path("data_etl")
AI_ROOT = Path("data_ai")  # kept for migration detection only; new outputs go to data/etl via etl_snapshot_root

# common filenames used in paths/manifests
EXPORT_XML = "export.xml"
VERSION_RAW = "version_raw.csv"
VERSION_LOG = "version_log_enriched.csv"

# common help strings
HELP_SNAPSHOT_DIR = "Path data/etl/<PID>/<YYYY-MM-DD>"
HELP_PARTICIPANT = "PID (usar com --snapshot)"
HELP_SNAPSHOT_ID = "Snapshot id (YYYY-MM-DD ou YYYYMMDD)"
USAGE_SNAPSHOT_OR_PARTICIPANT = "Use --snapshot_dir OU (--participant e --snapshot)."
SNAPSHOT_NOT_EXISTS = "WARNING: snapshot_dir não existe:"

_SNAP_ISO = re.compile(r"^20\d{2}-[01]\d-[0-3]\d$")
_SNAP_8DIG = re.compile(r"^20\d{6}$")  # YYYYMMDD


def canon_snap_id(name: str) -> str:
    name = name.strip()
    if _SNAP_ISO.match(name):
        return name
    if _SNAP_8DIG.match(name):
        return f"{name[:4]}-{name[4:6]}-{name[6:8]}"
    raise ValueError(f"Invalid snapshot '{name}'. Expected YYYY-MM-DD or YYYYMMDD.")


def canon_pid(pid: str) -> str:
    """Normalize participant ID to canonical form 'P' + zero-padded 6-digit number.

    Examples:
      P1 -> P000001
      p0001 -> P000001
      1 -> P000001
    If pid does not contain any digits, return it upper-cased unchanged.
    """
    if not pid:
        return pid
    s = str(pid).strip()
    # extract digits
    m = re.search(r"(\d+)", s)
    if not m:
        return s.upper()
    num = int(m.group(1))
    return f"P{num:06d}"


def ensure_ai_outdir(pid: str, snap: str) -> Path:
    """Return the canonical snapshot output dir.

    Historically this returned a path under `data_ai/`. To preserve
    compatibility with existing callers we keep the function name but
    make it return the canonical `data/etl/<pid>/<snap>` using
    `etl_snapshot_root` so callers receive the correct write location.
    """
    snap_iso = canon_snap_id(snap)
    outdir = etl_snapshot_root(pid, snap_iso)
    outdir.mkdir(parents=True, exist_ok=True)
    return outdir


def find_export_xml(pid: str, snap: str) -> Optional[Path]:
    snap_iso = canon_snap_id(snap)
    cands = [
        RAW_ROOT / pid / "snapshots" / snap_iso / EXPORT_XML,
        RAW_ROOT / pid / "snapshots" / snap_iso.replace("-", "") / EXPORT_XML,
        RAW_ROOT / pid / snap_iso / EXPORT_XML,
        RAW_ROOT / pid / snap_iso.replace("-", "") / EXPORT_XML,
    ]
    for p in cands:
        if p.exists():
            return p
    return None


def _sha256_file(path: str | os.PathLike[str]) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


# ----------------------------------------------------------------------
# Helpers for auto-discovery, zip inspection and sha handling
# ----------------------------------------------------------------------
RAW_ARCHIVE = Path("data/raw")  # raw device zips per participant (discovery base)
ETL_ROOT = Path("data/etl")  # target for extracted device content


def find_latest_zip(root: Path) -> Optional[Path]:
    """Find the latest ZIP file under `root` (root is a directory to search).
    Sorting: mtime desc, size desc, name desc.
    Returns Path or None.
    """
    if not root.exists():
        return None
    cands = list(root.rglob("*.zip"))
    if not cands:
        return None

    def _key(p: Path):
        try:
            st = p.stat()
            return (st.st_mtime, st.st_size, str(p))
        except Exception:
            return (0, 0, str(p))

    cands.sort(key=_key, reverse=True)
    return cands[0]


def find_latest_health_zip(pid: str) -> Path:
    """Find the latest Apple Health ZIP under data/raw/<pid>/apple.

    Raises FileNotFoundError if no zip is present.
    """
    root = RAW_ARCHIVE / pid / "apple"
    z = find_latest_zip(root)
    if not z:
        raise FileNotFoundError(f"No apple_health_export_*.zip under {root}")
    # safety: ensure zip is inside the participant raw dir
    _ensure_within_base(z, RAW_ARCHIVE / pid)
    return z


def snapshot_from_zip_file_date(zip_path: Path) -> str:
    """Derive a YYYY-MM-DD snapshot id from the zip filename or mtime.

    Tries to parse an ISO or YYYYMMDD date from the filename. Falls back to
    the file mtime when no date token is present.
    """
    name = zip_path.name
    m = re.search(r"(20\d{2}-\d{2}-\d{2})", name)
    if m:
        return canon_snap_id(m.group(1))
    m = re.search(r"(20\d{6})", name)
    if m:
        ymd = m.group(1)
        return canon_snap_id(f"{ymd[:4]}-{ymd[4:6]}-{ymd[6:8]}")
    # fallback: use mtime
    try:
        ts = zip_path.stat().st_mtime
        return datetime.fromtimestamp(ts).date().isoformat()
    except Exception:
        # last-resort: today's date
        return date.today().isoformat()


# ----------------------------------------------------------------------
# New: snapshot resolution, discovery and unified extract_run
# ----------------------------------------------------------------------
def resolve_snapshot(snapshot_arg: str, pid: str) -> str:
    """Resolve the canonical snapshot id for a run.

    - If snapshot_arg != 'auto' -> return canonicalized snapshot
    - If 'auto' -> read data/config/<PID>_profile.json for home_tz and use
      local today() in that timezone. Fallback to UTC.
    """
    if snapshot_arg and snapshot_arg != "auto":
        return canon_snap_id(snapshot_arg)

    # auto: inspect profile
    profile_path = Path("data") / "config" / f"{pid}_profile.json"
    tz_name = None
    try:
        if profile_path.exists():
            d = json.loads(profile_path.read_text(encoding="utf-8"))
            tz_name = d.get("home_tz")
    except Exception:
        tz_name = None

    if tz_name:
        try:
            tz = get_tz(tz_name)
            snap = datetime.now(tz).date().isoformat()
            return snap
        except Exception:
            pass

    # fallback to UTC
    return datetime.utcnow().date().isoformat()


def discover_sources(pid: str, auto_zip: bool = True) -> list[dict]:
    """Discover latest candidate zips for apple (variants) and zepp.

    Returns list of dicts: {vendor, variant, zip_path}
    """
    results: list[dict] = []
    base = RAW_ARCHIVE / pid
    if not base.exists():
        return results

    # Apple: search under data/raw/<pid>/apple/export or data/raw/<pid>/apple/**
    apple_root = base / "apple"
    apple_candidates = []
    if apple_root.exists():
        # first prefer export subfolder
        exp = apple_root / "export"
        search_root = exp if exp.exists() else apple_root
        for p in search_root.rglob("*.zip"):
            apple_candidates.append(p)

    # classify apple zips into variants
    variants = {"inapp": [], "itunes": [], "autoextract": []}
    for z in apple_candidates:
        try:
            if zip_contains(z, ["apple_health_export/export.xml"]):
                variants["inapp"].append(z)
                continue
            if zip_contains(z, ["manifest.db"]):
                variants["itunes"].append(z)
                continue
            if zip_contains(z, ["autoexport", "auto-extract", "autoextract"]):
                variants["autoextract"].append(z)
                continue
            # fallback: if a zip contains export.xml anywhere, treat as inapp
            if zip_contains(z, ["export.xml"]):
                variants["inapp"].append(z)
        except Exception:
            continue

    def pick_latest(lst: list[Path]) -> Optional[Path]:
        if not lst:
            return None
        lst.sort(key=lambda p: (p.stat().st_mtime, p.stat().st_size), reverse=True)
        return lst[0]

    for var, lst in variants.items():
        z = pick_latest(lst)
        if z:
            results.append({"vendor": "apple", "variant": var, "zip_path": z})
        else:
            # no candidate -> skip entry
            pass

    # Zepp: any zip under data/raw/<pid>/zepp/export or zepp/**
    zepp_root = base / "zepp"
    zepp_candidates = []
    if zepp_root.exists():
        exp = zepp_root / "export"
        search_root = exp if exp.exists() else zepp_root
        for p in search_root.rglob("*.zip"):
            zepp_candidates.append(p)
    z = pick_latest(zepp_candidates)
    if z:
        results.append({"vendor": "zepp", "variant": "cloud", "zip_path": z})

    return results


def _ensure_outdir(pid: str, snapshot: str, vendor: str, variant: str) -> Path:
    snap_iso = canon_snap_id(snapshot)
    out = etl_snapshot_root(pid, snap_iso) / "extracted" / vendor / variant
    out.mkdir(parents=True, exist_ok=True)
    return out


def _write_qc_rows(qc_rows: list[dict], qc_dir: Path) -> None:
    qc_dir.mkdir(parents=True, exist_ok=True)
    csv_path = qc_dir / "extract_qc.csv"
    # ensure deterministic column order
    cols = ["vendor", "variant", "found_zip", "parsed_xml", "parsed_cda", "rows_out", "seconds", "status"]
    # convert to DataFrame (pandas available)
    try:
        df = pd.DataFrame(qc_rows)
        # ensure columns
        for c in cols:
            if c not in df.columns:
                df[c] = None
        _write_atomic_csv(df[cols], str(csv_path))
    except Exception:
        # fallback simple CSV
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write(",".join(cols) + "\n")
            for r in qc_rows:
                f.write(",".join(str(r.get(c, "")) for c in cols) + "\n")


def _write_manifest(manifest: dict, qc_dir: Path) -> None:
    qc_dir.mkdir(parents=True, exist_ok=True)
    path = qc_dir / "extract_manifest.json"
    _write_atomic_json(manifest, str(path))


class CDAParseWarning(Exception):
    pass


def _parse_export_xml(xml_path: Path) -> int:
    """Lightweight parse of export.xml that counts top-level Record/Workout nodes.

    Returns number of records found (approximation)."""
    try:
        # use a streaming iterparse to avoid loading whole file
        count = 0
        for event, elem in ET.iterparse(str(xml_path), events=("end",)):
            if elem.tag.endswith("Record") or elem.tag.endswith("Workout"):
                count += 1
            # free memory
            elem.clear()
        return count
    except Exception:
        return 0


def _parse_export_cda(xml_path: Path) -> tuple[int, list[str]]:
    """Attempt a tolerant streaming parse of export_cda.xml.

    Returns (rows_out, warnings)
    """
    warnings: list[str] = []
    rows = 0
    try:
        # iterparse but guard against very large files
        for event, elem in ET.iterparse(str(xml_path), events=("end",)):
            if elem.tag.endswith("entry") or elem.tag.endswith("Section"):
                rows += 1
            elem.clear()
            # yield time slice to avoid blocking UI
        return rows, warnings
    except Exception as e:
        warnings.append(f"failed to parse CDA XML: {xml_path}")
        return 0, warnings


def extract_run(pid: str, snapshot_arg: str = "auto", auto_zip: bool = True, dry_run: bool = False) -> int:
    """Unified extract run that processes Apple variants and Zepp under one canonical snapshot.

    Returns exit code: 0 if any source produced artifacts (ok or warn), 2 if all failed/skipped.
    """
    started_at = datetime.utcnow().isoformat() + "Z"
    snapshot = resolve_snapshot(snapshot_arg, pid)
    sources = discover_sources(pid, auto_zip=auto_zip)
    qc_rows: list[dict] = []
    manifest: dict = {"pid": pid, "snapshot": snapshot, "home_tz": None, "started_at": started_at, "ended_at": None, "sources": [], "warnings": []}

    if dry_run:
        print(f"DRY RUN: discovered {len(sources)} sources for {pid} -> snapshot {snapshot}")
        for s in sources:
            print(f" - {s['vendor']}/{s['variant']}: {s['zip_path']}")
        return 0 if sources else 2

    any_ok = False
    for s in sources:
        vendor = s.get("vendor")
        variant = s.get("variant")
        zip_path: Path = s.get("zip_path")
        t0 = time.time()
        parsed_xml = 0
        parsed_cda = 0
        rows_out = 0
        status = "skip"
        found_zip = 1 if zip_path and zip_path.exists() else 0

        out_dir = _ensure_outdir(pid, snapshot, vendor, variant)
        try:
            if not found_zip:
                status = "skip"
                elapsed = 0.0
            else:
                # extract into variant out_dir (idempotent: overwrite files)
                # Use pyzipper for Zepp archives to support AES-encrypted ZIPs; fallback to zipfile for others
                if vendor == "zepp":
                    pwd = os.environ.get("ZEPP_ZIP_PASSWORD")
                    if not pwd:
                        print(f"WARNING: ZEPP_ZIP_PASSWORD not set; skipping Zepp {zip_path}")
                        manifest["warnings"].append(f"Zepp skipped (no password): {zip_path}")
                        status = "skip"
                        elapsed = time.time() - t0
                        qc_rows.append({"vendor": vendor, "variant": variant, "found_zip": found_zip, "parsed_xml": 0, "parsed_cda": 0, "rows_out": 0, "seconds": round(elapsed, 2), "status": status})
                        continue
                    try:
                        import pyzipper

                        with pyzipper.AESZipFile(str(zip_path)) as zf:
                            if pwd:
                                zf.pwd = pwd.encode("utf-8")
                            members = zf.namelist()
                            with progress_bar(total=len(members), desc=f"Zepp extract {vendor}/{variant}", unit="files") as bar:
                                for member in members:
                                    target = out_dir / Path(member)
                                    if member.endswith('/'):
                                        target.mkdir(parents=True, exist_ok=True)
                                        bar.update(1)
                                        continue
                                    target.parent.mkdir(parents=True, exist_ok=True)
                                    try:
                                        data = zf.read(member)
                                    except RuntimeError:
                                        # bad password or other runtime issue
                                        print(f"WARNING: Zepp zip appears encrypted or bad password: {zip_path}; skipping")
                                        manifest["warnings"].append(f"Zepp bad password: {zip_path}")
                                        status = "skip"
                                        elapsed = time.time() - t0
                                        qc_rows.append({"vendor": vendor, "variant": variant, "found_zip": found_zip, "parsed_xml": 0, "parsed_cda": 0, "rows_out": 0, "seconds": round(elapsed, 2), "status": status})
                                        raise
                                    with open(str(target), "wb") as fh:
                                        fh.write(data)
                                    bar.update(1)
                    except RuntimeError:
                        # already handled above and manifested
                        continue
                    except Exception as e:
                        # any other extraction problem -> skip Zepp and warn
                        print(f"WARNING: failed to extract Zepp zip {zip_path}: {e}; skipping")
                        manifest["warnings"].append(f"Zepp extract failed: {zip_path}: {e}")
                        status = "skip"
                        elapsed = time.time() - t0
                        qc_rows.append({"vendor": vendor, "variant": variant, "found_zip": found_zip, "parsed_xml": 0, "parsed_cda": 0, "rows_out": 0, "seconds": round(elapsed, 2), "status": status})
                        continue
                else:
                    # regular zip extraction for Apple variants
                    with zipfile.ZipFile(zip_path, "r") as zf:
                        members = zf.namelist()
                        with progress_bar(total=len(members), desc=f"Apple extract {vendor}/{variant}", unit="files") as bar:
                            for member in members:
                                target = out_dir / Path(member)
                                if member.endswith('/'):
                                    target.mkdir(parents=True, exist_ok=True)
                                else:
                                    target.parent.mkdir(parents=True, exist_ok=True)
                                    with open(str(target), "wb") as fh:
                                        fh.write(zf.read(member))
                                bar.update(1)

                # attempt to find export.xml and export_cda.xml inside out_dir
                export_xml = None
                cda_xml = None
                cand1 = out_dir / EXPORT_XML
                cand2 = out_dir / "apple_health_export" / EXPORT_XML
                if cand1.exists():
                    export_xml = cand1
                elif cand2.exists():
                    export_xml = cand2
                else:
                    found_any = list(out_dir.rglob(EXPORT_XML))
                    if found_any:
                        export_xml = found_any[0]

                if export_xml:
                    parsed_xml = _parse_export_xml(export_xml)
                    print(f"extract: [{vendor}/{variant}] parse export.xml: {parsed_xml} records")
                else:
                    print(f"extract: [{vendor}/{variant}] no export.xml found in {out_dir}")

                # check cda (non-fatal). Use the new CDA probe which produces a small QC summary.
                cda_cand = out_dir / "apple_health_export" / "export_cda.xml"
                if cda_cand.exists():
                    # determine home_tz from participant profile if present
                    home_tz = None
                    try:
                        prof_path = Path("data") / "config" / f"{pid}_profile.json"
                        if prof_path.exists():
                            prof = json.loads(prof_path.read_text(encoding="utf-8"))
                            home_tz = prof.get("home_tz")
                    except Exception:
                        home_tz = None

                    import logging
                    logger = logging.getLogger("etl.extract")
                    try:
                        from domains.cda import parse_cda

                        summary = parse_cda.cda_probe(cda_cand, home_tz=home_tz)
                        parsed_cda = summary.get("n_observation", 0)
                        if parsed_cda > 0:
                            logger.info(
                                f"CDA parsed: {summary.get('n_section',0)} sections, {parsed_cda} observations, {len(summary.get('codes',{}))} unique codes"
                            )
                        else:
                            logger.warning(f"CDA parsed but empty: {cda_cand.name}")
                        snapshot_root = etl_snapshot_root(pid, snapshot)
                        parse_cda.write_cda_qc(snapshot_root, summary)
                    except Exception as e:
                        # attempt to write a minimal QC summary so users can inspect the error
                        try:
                            from domains.cda import parse_cda

                            snapshot_root = etl_snapshot_root(pid, snapshot)
                            summary = {"error": f"{type(e).__name__}: {e}", "n_section": 0, "n_observation": 0, "codes": {}}
                            try:
                                parse_cda.write_cda_qc(snapshot_root, summary)
                            except Exception:
                                pass
                        except Exception:
                            pass
                        except Exception:
                            pass
                        logger.warning(f"CDA parse failed: {cda_cand.name} ({type(e).__name__}: {e})")

                rows_out = parsed_xml + parsed_cda
                status = "ok"
                elapsed = time.time() - t0
                any_ok = True

            qc_rows.append({"vendor": vendor, "variant": variant, "found_zip": found_zip, "parsed_xml": parsed_xml, "parsed_cda": parsed_cda, "rows_out": rows_out, "seconds": round(elapsed, 2), "status": status})
            manifest["sources"].append({"vendor": vendor, "variant": variant, "zip_path": str(zip_path), "out_dir": str(out_dir), "seconds": round(time.time() - t0, 2), "status": status})
        except Exception as e:
            elapsed = time.time() - t0
            status = "fail"
            qc_rows.append({"vendor": vendor, "variant": variant, "found_zip": found_zip, "parsed_xml": parsed_xml, "parsed_cda": parsed_cda, "rows_out": 0, "seconds": round(elapsed, 2), "status": status})
            manifest.setdefault("warnings", []).append(f"extract failure {vendor}/{variant}: {e}")

    # write QC & manifest
    qc_dir = etl_snapshot_root(pid, canon_snap_id(snapshot)) / "qc"
    _write_qc_rows(qc_rows, qc_dir)
    manifest["ended_at"] = datetime.utcnow().isoformat() + "Z"
    _write_manifest(manifest, qc_dir)

    return 0 if any_ok else 2


def ensure_extracted(pid: str, snapshot: str, zip_path: Path) -> Path:
    """Ensure export.xml is present under data/etl/<pid>/snapshots/<snapshot>/extracted/apple.

    If not present, extract the provided zip to that location (idempotent).
    Returns the Path to the export.xml inside the extracted folder.
    """
    snap_iso = canon_snap_id(snapshot)
    out = etl_snapshot_root(pid, snap_iso) / "extracted" / "apple"
    # Common layouts: either export.xml at out/ or under out/apple_health_export/export.xml
    candidate1 = out / EXPORT_XML
    candidate2 = out / "apple_health_export" / EXPORT_XML
    # If already present in either location, prefer candidate1 then candidate2
    if candidate1.exists():
        print(f"[extract] already present -> {candidate1}")
        return candidate1
    if candidate2.exists():
        print(f"[extract] already present -> {candidate2}")
        return candidate2
    out.mkdir(parents=True, exist_ok=True)
    # safety: only extract archives from the participant raw folder
    _ensure_within_base(zip_path, RAW_ARCHIVE / pid)
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(str(out))
    print(f"[extract] EXTRACTED -> {out}")
    # Locate the actual export.xml inside extracted tree (could be nested)
    # Prefer out/export.xml, then out/apple_health_export/export.xml, then any export.xml under out
    if candidate1.exists():
        return candidate1
    if candidate2.exists():
        return candidate2
    found = list(out.rglob(EXPORT_XML))
    if found:
        return found[0]
    # if nothing found, raise
    raise FileNotFoundError(
        f"export.xml not found after extracting {zip_path} into {out}"
    )


def zip_contains(zip_path: Path, patterns: Iterable[str]) -> bool:
    """Return True if any filename inside zip matches any substring in patterns (case-insensitive)."""
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            names = [n.lower() for n in zf.namelist()]
            for pat in patterns:
                p = pat.lower()
                for n in names:
                    if p in n:
                        return True
        return False
    except zipfile.BadZipFile:
        return False


def sha256_path(path: Path) -> str:
    return _sha256_file(str(path))


def read_sha256(path: Path) -> Optional[str]:
    try:
        if not path.exists():
            return None
        return path.read_text(encoding="utf-8").strip()
    except Exception:
        return None


def write_sha256(path: Path, hash_str: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(hash_str + "\n", encoding="utf-8")


def _ensure_within_base(candidate: Path, base_dir: Path) -> None:
    real = candidate.resolve(strict=False)
    base = base_dir.resolve(strict=False)
    try:
        # Python 3.9+ has is_relative_to
        if hasattr(real, "is_relative_to"):
            if not real.is_relative_to(base):
                raise SystemExit(
                    f"ERROR: ZIP outside participant scope is not allowed: {candidate}"
                )
        else:
            if str(real)[: len(str(base))] != str(base):
                raise SystemExit(
                    f"ERROR: ZIP outside participant scope is not allowed: {candidate}"
                )
    except Exception as e:
        if isinstance(e, SystemExit):
            raise
        raise SystemExit(
            f"ERROR: ZIP outside participant scope is not allowed: {candidate}"
        )


# ----------------------------------------------------------------------
# Timezone (zoneinfo, fallback pytz)
# ----------------------------------------------------------------------
try:
    from zoneinfo import ZoneInfo  # py3.9+
except Exception:
    ZoneInfo = None
    try:
        import pytz  # type: ignore
    except Exception:
        pytz = None


def get_tz(tz_name: str):
    if ZoneInfo is not None:
        return ZoneInfo(tz_name)
    if "pytz" in sys.modules or "pytz" in globals():
        return pytz.timezone(tz_name)  # type: ignore
    raise RuntimeError("No timezone support. Use Python 3.9+ or install pytz.")


# ----------------------------------------------------------------------
# HealthKit parsing
# ----------------------------------------------------------------------
TYPE_MAP = {
    "HKQuantityTypeIdentifierHeartRate": ("hr", "count/min"),
    "HKQuantityTypeIdentifierHeartRateVariabilitySDNN": ("hrv_sdnn", "ms"),
    "HKCategoryTypeIdentifierSleepAnalysis": ("sleep", "category"),
    "HKQuantityTypeIdentifierBodyTemperature": ("temp", "degC"),
    "HKQuantityTypeIdentifierAppleSleepingWristTemperature": ("temp", "degC"),
    "HKQuantityTypeIdentifierBasalBodyTemperature": ("temp", "degC"),
}

ASLEEP_STR = {
    "HKCategoryValueSleepAnalysisAsleep",
    "HKCategoryValueSleepAnalysisAsleepUnspecified",
    "HKCategoryValueSleepAnalysisAsleepCore",
    "HKCategoryValueSleepAnalysisAsleepDeep",
    "HKCategoryValueSleepAnalysisAsleepREM",
}
ASLEEP_INT = {1, 3, 4, 5}

DEVICE_KEY_MAP = {
    "name": "name",
    "manufacturer": "manufacturer",
    "model": "watch_model",
    "hardware": "watch_fw",
    "software": "ios_version",
}


def fix_iso_tz(s: str) -> str:
    if s.endswith("Z"):
        return s[:-1] + "+00:00"
    m = re.search(r"([+-]\d{2})(\d{2})$", s)
    if m:
        s = s[: m.start()] + f"{m.group(1)}:{m.group(2)}"
    return s


def parse_dt(s: str) -> datetime:
    return datetime.fromisoformat(fix_iso_tz(s))


def parse_device_string(device_str: str) -> Dict[str, str]:
    info: Dict[str, str] = {}
    if not device_str:
        return info
    parts = [p.strip() for p in device_str.strip("<>").split(",")]
    for part in parts:
        if ":" in part:
            k, v = part.split(":", 1)
            info[k.strip().lower()] = v.strip()
    mapped: Dict[str, str] = {}
    for k, v in info.items():
        if k in DEVICE_KEY_MAP:
            mapped[DEVICE_KEY_MAP[k]] = v
    return mapped




def iter_health_records(
    xml_source: Union[Path, io.BufferedReader]
) -> Iterable[
    Tuple[
        str,
        Any,
        datetime,
        Optional[datetime],
        Optional[str],
        Dict[str, str],
        str,
        str,
        str,
    ]
]:
    """
    xml_source: Path OU file-like binário (ex.: ProgressFile)
    
    Note: All timestamps are returned in UTC. Daily binning is computed at UTC midnight.
    """
    src = xml_source if hasattr(xml_source, "read") else str(xml_source)
    context = ET.iterparse(src, events=("end",))
    for _, elem in context:
        if elem.tag != "Record":
            continue
        type_id = elem.attrib.get("type")
        if type_id not in TYPE_MAP:
            elem.clear()
            continue

        s = elem.attrib.get("startDate")
        e = elem.attrib.get("endDate")
        if not s:
            elem.clear()
            continue

        sdt = parse_dt(s)
        edt = parse_dt(e) if e else None

        # Convert to UTC (all timestamps are tz-aware after parse_dt)
        s_utc = sdt.astimezone(get_tz("UTC"))
        e_utc = edt.astimezone(get_tz("UTC")) if edt else None

        unit = elem.attrib.get("unit")
        val_raw = elem.attrib.get("value")

        if type_id == "HKCategoryTypeIdentifierSleepAnalysis":
            value = val_raw
        else:
            try:
                value = float(val_raw) if val_raw is not None else None
            except Exception:
                value = None

        device = parse_device_string(elem.attrib.get("device", ""))
        tz_name = "UTC"  # Always UTC now
        source_version = elem.attrib.get("sourceVersion", "")
        source_name = elem.attrib.get("sourceName", "")

        yield type_id, value, s_utc, e_utc, unit, device, tz_name, source_version, source_name
        elem.clear()


# ----------------------------------------------------------------------
# Segments
# ----------------------------------------------------------------------
def build_segments_from_versions(vdf: pd.DataFrame) -> pd.DataFrame:
    if vdf.empty:
        return pd.DataFrame(
            columns=[
                "segment_id",
                "start",
                "end",
                "ios_version",
                "watch_model",
                "watch_fw",
                "tz_name",
                "source_name",
            ]
        )
    df = vdf.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")
    keys = ["ios_version", "watch_model", "watch_fw", "tz_name", "source_name"]
    sig = df[keys].astype(str).agg(" | ".join, axis=1)
    change = sig.ne(sig.shift(fill_value="__START__"))
    segments = []
    seg_id = 0
    seg_start = df["date"].iloc[0]
    last_vals = df.loc[df.index[0], keys].to_dict()
    for i in range(1, len(df)):
        if change.iloc[i]:
            seg_id += 1
            segments.append(
                {
                    "segment_id": seg_id,
                    "start": seg_start.date(),
                    "end": df["date"].iloc[i - 1].date(),
                    **last_vals,
                }
            )
            seg_start = df["date"].iloc[i]
            last_vals = df.loc[df.index[i], keys].to_dict()
    seg_id += 1
    segments.append(
        {
            "segment_id": seg_id,
            "start": seg_start.date(),
            "end": df["date"].iloc[-1].date(),
            **last_vals,
        }
    )
    return pd.DataFrame(segments)


# ----------------------------------------------------------------------
# Apple → per-metric (extract)
# ----------------------------------------------------------------------
def extract_apple_per_metric(
    xml_file: Path, outdir: Path
) -> Dict[str, str]:
    """Extract Apple health records to per-metric CSVs in UTC.
    
    All timestamps are converted to UTC; daily binning is at UTC midnight.
    """
    out_pm = outdir / "per-metric"
    out_pm.mkdir(parents=True, exist_ok=True)

    hr_rows, hrv_rows, sleep_rows, version_rows = [], [], [], []
    HR_ID = "HKQuantityTypeIdentifierHeartRate"
    HRV_ID = "HKQuantityTypeIdentifierHeartRateVariabilitySDNN"
    SLEEP_ID = "HKCategoryTypeIdentifierSleepAnalysis"

    def _estimate_record_count(p: Path) -> int:
        # fast binary scan counting occurrences of '<Record' as an estimate
        try:
            cnt = 0
            with p.open("rb") as fh:
                for chunk in iter(lambda: fh.read(1 << 20), b""):
                    cnt += chunk.count(b"<Record")
            return cnt or 0
        except Exception:
            return 0

    with Timer("extract: parse export.xml"):
        total_records = _estimate_record_count(xml_file)
        # stream-parse the XML and update the bar per Record event
    
    with progress_bar(
        total=total_records or None, desc="Parsing Records", unit="items"
    ) as _bar:
        src = str(xml_file)
        context = ET.iterparse(src, events=("end",))
        for _, elem in context:
            if elem.tag != "Record":
                elem.clear()
                continue
            type_id = elem.attrib.get("type")
            if type_id not in TYPE_MAP:
                elem.clear()
                try:
                    _bar.update(1)
                except Exception:
                    pass
                continue

            s = elem.attrib.get("startDate")
            e = elem.attrib.get("endDate")
            if not s:
                try:
                    _bar.update(1)
                except Exception:
                    pass
                elem.clear()
                continue

            try:
                sdt = parse_dt(s)
            except Exception:
                try:
                    _bar.update(1)
                except Exception:
                    pass
                elem.clear()
                continue

            edt = parse_dt(e) if e else None
            # Convert to UTC
            s_utc = sdt.astimezone(get_tz("UTC"))
            e_utc = edt.astimezone(get_tz("UTC")) if edt else None

            unit = elem.attrib.get("unit")
            val_raw = elem.attrib.get("value")

            if type_id == "HKCategoryTypeIdentifierSleepAnalysis":
                value = val_raw
            else:
                try:
                    value = float(val_raw) if val_raw is not None else None
                except Exception:
                    value = None

            device = parse_device_string(elem.attrib.get("device", ""))
            tz_name = "UTC"
            src_ver = elem.attrib.get("sourceVersion", "")
            src_name = elem.attrib.get("sourceName", "")

            # now map into outputs (UTC day = UTC midnight bucket)
            day = s_utc.date()
            ios_ver = device.get("ios_version", "") or src_ver
            if ios_ver or device:
                version_rows.append(
                    {
                        "date": day,
                        "ios_version": ios_ver,
                        "watch_model": device.get("watch_model", ""),
                        "watch_fw": device.get("watch_fw", ""),
                        "tz_name": tz_name,
                        "source_name": src_name,
                    }
                )
            if type_id == HR_ID and value is not None:
                hr_rows.append({"timestamp": s_utc.isoformat(), "bpm": float(value)})
            elif type_id == HRV_ID and value is not None:
                hrv_rows.append(
                    {"timestamp": s_utc.isoformat(), "sdnn_ms": float(value)}
                )
            elif type_id == SLEEP_ID and e_utc is not None:
                sleep_rows.append(
                    {
                        "start": s_utc.isoformat(),
                        "end": e_utc.isoformat(),
                        "raw_value": value,
                    }
                )
            
            # Progress update and memory cleanup
            try:
                _bar.update(1)
            except Exception:
                pass
            elem.clear()

    written: Dict[str, str] = {}

    if hr_rows:
        p = out_pm / "apple_heart_rate.csv"
        _write_atomic_csv(pd.DataFrame(hr_rows).sort_values("timestamp"), p)
        written["apple_heart_rate"] = str(p)
    if hrv_rows:
        p = out_pm / "apple_hrv_sdnn.csv"
        _write_atomic_csv(pd.DataFrame(hrv_rows).sort_values("timestamp"), p)
        written["apple_hrv_sdnn"] = str(p)
    if sleep_rows:
        p = out_pm / "apple_sleep_intervals.csv"
        _write_atomic_csv(pd.DataFrame(sleep_rows).sort_values("start"), p)
        written["apple_sleep_intervals"] = str(p)

    # version logs
    vdf = (
        pd.DataFrame(version_rows).drop_duplicates().sort_values("date")
        if version_rows
        else pd.DataFrame()
    )
    if not vdf.empty:
        _write_atomic_csv(vdf, outdir / VERSION_RAW)
        seg_df = build_segments_from_versions(vdf)
        _write_atomic_csv(seg_df, outdir / VERSION_LOG)
        written["version_raw"] = str(outdir / VERSION_RAW)
        written["version_log_enriched"] = str(outdir / VERSION_LOG)
    else:
        _write_atomic_csv(
            pd.DataFrame(
                columns=[
                    "date",
                    "ios_version",
                    "watch_model",
                    "watch_fw",
                    "tz_name",
                    "source_name",
                ]
            ),
            outdir / VERSION_RAW,
        )
        _write_atomic_csv(
            pd.DataFrame(
                columns=[
                    "segment_id",
                    "start",
                    "end",
                    "ios_version",
                    "watch_model",
                    "watch_fw",
                    "tz_name",
                    "source_name",
                ]
            ),
            outdir / VERSION_LOG,
        )

    # manifest
    manifest = {
        "type": "extract",
        "export_xml": str(xml_file),
        "params": {
            "timezone": "UTC"
        },
        "outputs": {},
    }
    try:
        if os.path.exists(xml_file):
            manifest["export_sha256"] = _sha256_file(xml_file)
    except Exception:
        manifest["export_sha256"] = None
    for k, v in written.items():
        try:
            manifest["outputs"][k] = {"path": v, "sha256": _sha256_file(v)}
        except Exception:
            manifest["outputs"][k] = {"path": v, "sha256": None}
    _write_atomic_json(manifest, outdir / "extract_manifest.json")

    # ---- manifest com hashes do export.xml e saídas ----
    manifest = {
        "type": "extract",
        "export_xml": str(xml_file),
        "params": {
            "timezone": "UTC"
        },
        "outputs": {},
    }
    try:
        if os.path.exists(xml_file):
            manifest["export_sha256"] = _sha256_file(str(xml_file))
    except Exception:
        pass
    for k, v in written.items():
        try:
            manifest["outputs"][k] = {"path": v, "sha256": _sha256_file(v)}
        except Exception:
            manifest["outputs"][k] = {"path": v, "sha256": None}
    _write_atomic_json(manifest, str(outdir / "extract_manifest.json"))
    return written


# ----------------------------------------------------------------------
# Apple CDA (State of Mind) parser
# ----------------------------------------------------------------------
def _parse_apple_cda_som(cda_xml_path: Path) -> pd.DataFrame:
    """Parse export_cda.xml (CCDA-like) to extract Mood / State of Mind observations.

    Returns DataFrame with columns: timestamp_utc (ISO UTC), date (YYYY-MM-DD UTC),
    source (apple_som), value_raw, value_norm, notes

    Heuristic mapping for value_norm (documented):
      positive (+1): tokens like 'very pleasant','pleasant','happy','excited','joyful'
      neutral  (0) : tokens like 'neutral','calm','ok','okay'
      negative (-1): tokens like 'very unpleasant','unpleasant','sad','angry','stressed','anxious'

    The parser is intentionally tolerant: if the file is missing or no mood-like
    observations are found, returns an empty DataFrame.
    
    Note: All timestamps are returned in UTC. Daily binning is at UTC midnight.
    """
    rows = []
    if not cda_xml_path.exists():
        return pd.DataFrame(
            columns=[
                "timestamp_utc",
                "date",
                "source",
                "value_raw",
                "value_norm",
                "notes",
            ]
        )

    try:
        tree = ET.parse(str(cda_xml_path))
        root = tree.getroot()
    except Exception:
        print(f"WARNING: failed to parse CDA XML: {cda_xml_path}")
        return pd.DataFrame(
            columns=[
                "timestamp_utc",
                "date",
                "source",
                "value_raw",
                "value_norm",
                "notes",
            ]
        )

    # candidate tags/paths that may contain mood/state obs
    candidates = []
    for elem in root.iter():
        tag = elem.tag.lower()
        text = (elem.text or "").strip()
        # look for nodes mentioning 'mood' or 'state of mind' or named Observation
        if "mood" in tag or "stateofmind" in tag or "state" in tag and "mind" in tag:
            candidates.append(elem)
        # also look for displayName/code elements that include 'Mood'
        if "mood" in text.lower() or "state of mind" in text.lower():
            candidates.append(elem)

    # fallback: scan for Observation entries that reference mood-like codes
    for entry in root.iter():
        try:
            tag = entry.tag.lower()
            if "observation" in tag or "entry" in tag:
                txt = "".join([(c.text or "") for c in entry.iter() if c.text])
                if (
                    "mood" in txt.lower()
                    or "state of mind" in txt.lower()
                    or "mood" in (entry.attrib.get("displayName") or "").lower()
                ):
                    candidates.append(entry)
        except Exception:
            continue

    seen = set()
    for elem in candidates:
        try:
            txt = " ".join([(c.text or "").strip() for c in elem.iter() if c.text])
            if not txt or txt.strip() == "":
                continue
            key = (elem.tag, txt)
            if key in seen:
                continue
            seen.add(key)

            # Try to find a timestamp inside the entry (effectiveTime, time, value/@value)
            ts = None
            for c in elem.iter():
                if (
                    c.tag.lower().endswith("effective")
                    or c.tag.lower().endswith("time")
                    or "time" in c.tag.lower()
                ):
                    ttxt = (c.attrib.get("value") or c.text or "").strip()
                    if ttxt:
                        try:
                            ts = pd.to_datetime(ttxt)
                            break
                        except Exception:
                            pass
                # value elements
                if c.tag.lower().endswith("value") or c.tag.lower().endswith(
                    "valuecode"
                ):
                    if "value" in c.attrib:
                        vraw = c.attrib.get("value")
                    else:
                        vraw = (c.text or "").strip()
                    if vraw:
                        # keep as candidate
                        pass

            # fallback: use root document date
            if ts is None:
                ts = pd.Timestamp.now(tz=pd.Timestamp.utcnow().tz)

            # normalize raw value (take code display or text)
            value_raw = txt.strip()

            vr = value_raw.lower()
            value_norm = None
            if any(
                tok in vr
                for tok in (
                    "very pleasant",
                    "very_pleasant",
                    "verypleasant",
                    "very pleasant",
                    "joy",
                    "happy",
                    "excited",
                    "pleasant",
                )
            ):
                value_norm = 1
            elif any(tok in vr for tok in ("neutral", "calm", "ok", "okay", "fine")):
                value_norm = 0
            elif any(
                tok in vr
                for tok in (
                    "very unpleasant",
                    "very_unpleasant",
                    "sad",
                    "angry",
                    "stressed",
                    "anxious",
                    "unpleasant",
                    "upset",
                )
            ):
                value_norm = -1
            else:
                # unknown tokens -> try simple heuristics
                if "pleasant" in vr or "happy" in vr:
                    value_norm = 1
                elif "neutral" in vr or "calm" in vr:
                    value_norm = 0
                elif "sad" in vr or "angry" in vr or "stress" in vr:
                    value_norm = -1
                else:
                    value_norm = None

            # convert timestamp to UTC ISO
            try:
                ts_parsed = pd.to_datetime(ts)
                # assume naive -> UTC
                if ts_parsed.tzinfo is None:
                    ts_parsed = ts_parsed.tz_localize("UTC")
                ts_utc = ts_parsed.astimezone(pd.Timestamp.utcnow().tz)
            except Exception:
                ts_utc = pd.Timestamp.utcnow()

            # date in UTC
            date_utc = ts_utc.date()

            rows.append(
                {
                    "timestamp_utc": ts_utc.isoformat(),
                    "date": date_utc.isoformat(),
                    "source": "apple_som",
                    "value_raw": value_raw,
                    "value_norm": int(value_norm) if value_norm is not None else pd.NA,
                    "notes": f"parsed_from:{cda_xml_path.name}",
                }
            )
        except Exception:
            continue

    if not rows:
        return pd.DataFrame(
            columns=[
                "timestamp_utc",
                "date",
                "source",
                "value_raw",
                "value_norm",
                "notes",
            ]
        )
    return pd.DataFrame(rows)


def _parse_apple_cda_som_lxml(
    cda_xml_path: Path, parser=None
) -> pd.DataFrame:
    """Attempt parsing using lxml.iterparse with recover=True for robustness.

    Falls back to standard heuristics similar to _parse_apple_cda_som.
    
    Note: All timestamps are returned in UTC.
    """
    try:
        import lxml.etree as LET
    except Exception:
        raise

    rows = []
    if not cda_xml_path.exists():
        return pd.DataFrame(
            columns=[
                "timestamp_utc",
                "date",
                "source",
                "value_raw",
                "value_norm",
                "notes",
            ]
        )

    parser = parser or LET.XMLParser(recover=True, huge_tree=True, encoding="utf-8")

    # helper to strip namespace
    def lname(tag):
        return tag.split("}", 1)[-1] if "}" in tag else tag

    try:
        it = LET.iterparse(str(cda_xml_path), events=("end",), recover=True)
    except Exception:
        raise

    seen = set()
    elem_count = 0
    TOK_POS = ("very pleasant", "pleasant", "happy", "excited", "joy")
    TOK_NEU = ("neutral", "calm", "ok", "okay", "fine")
    TOK_NEG = (
        "very unpleasant",
        "unpleasant",
        "sad",
        "angry",
        "stressed",
        "anxious",
        "negative",
    )

    for event, elem in it:
        elem_count += 1
        tag = lname(elem.tag).lower() if isinstance(elem.tag, str) else ""
        if tag in ("observation", "entry"):
            try:
                txt = " ".join(
                    [
                        (c.text or "").strip()
                        for c in elem.iter()
                        if isinstance(c.tag, str) and (c.text or "").strip()
                    ]
                )
                if not txt:
                    elem.clear()
                    continue
                key = (lname(elem.tag), txt[:256])
                if key in seen:
                    elem.clear()
                    continue
                seen.add(key)

                # get timestamp from children if present
                ts = None
                for c in elem.iter():
                    try:
                        cn = lname(c.tag).lower() if isinstance(c.tag, str) else ""
                        if cn in ("effectivetime", "effective", "time") or "time" in cn:
                            ttxt = (c.get("value") or (c.text or "")).strip()
                            if ttxt:
                                try:
                                    ts = pd.to_datetime(ttxt)
                                    break
                                except Exception:
                                    pass
                    except Exception:
                        continue

                value_raw = txt.strip()
                vr = value_raw.lower()
                value_norm = pd.NA
                if any(tok in vr for tok in TOK_POS):
                    value_norm = 1
                elif any(tok in vr for tok in TOK_NEU):
                    value_norm = 0
                elif any(tok in vr for tok in TOK_NEG):
                    value_norm = -1

                try:
                    ts_parsed = (
                        pd.to_datetime(ts) if ts is not None else pd.Timestamp.utcnow()
                    )
                    if ts_parsed.tzinfo is None:
                        ts_parsed = ts_parsed.tz_localize("UTC")
                    ts_utc = ts_parsed.astimezone(pd.Timestamp.utcnow().tz)
                except Exception:
                    ts_utc = pd.Timestamp.utcnow()

                # date in UTC
                date_utc = ts_utc.date()

                rows.append(
                    {
                        "timestamp_utc": ts_utc.isoformat(),
                        "date": date_utc.isoformat(),
                        "source": "apple_som",
                        "value_raw": value_raw,
                        "value_norm": (
                            int(value_norm) if value_norm is not pd.NA else pd.NA
                        ),
                        "notes": f"parsed_from:{cda_xml_path.name}:lxml",
                    }
                )
            except Exception:
                pass
            finally:
                # free memory
                try:
                    elem.clear()
                    while elem.getprevious() is not None:
                        del elem.getparent()[0]
                except Exception:
                    pass

        # progress hint
        if elem_count and elem_count % 50000 == 0:
            mb = elem_count / 1000.0
            print(f"INFO: CDA stream parsed ~{int(mb)} K elements...")

    if not rows:
        return pd.DataFrame(
            columns=[
                "timestamp_utc",
                "date",
                "source",
                "value_raw",
                "value_norm",
                "notes",
            ]
        )
    # dedupe
    df = pd.DataFrame(rows)
    df = (
        df.drop_duplicates(
            subset=[
                "timestamp_utc",
                (
                    df.columns.get_loc("value_raw")
                    if "value_raw" in df.columns
                    else "value_raw"
                ),
            ],
            keep="first",
        )
        if not df.empty
        else df
    )
    return df


def _parse_cda_textscan(cda_xml_path: Path) -> pd.DataFrame:
    """Fallback text-scan: stream file, find tokens and grab snippets and possible timestamps.
    
    Note: All timestamps are returned in UTC. Daily binning is at UTC midnight.
    """
    TOKENS = [
        "mood",
        "pleasant",
        "unpleasant",
        "neutral",
        "anxious",
        "stressed",
        "happy",
        "sad",
        "angry",
        "valence",
        "affect",
    ]
    token_pat = re.compile(
        r"(" + "|".join(re.escape(t) for t in TOKENS) + r")", re.IGNORECASE
    )
    iso_ts = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:Z|[+-]\d{2}:?\d{2})?")
    ymdts = re.compile(r"\b\d{8,14}\b")
    rows = []
    if not cda_xml_path.exists():
        return pd.DataFrame(
            columns=[
                "timestamp_utc",
                "date",
                "source",
                "value_raw",
                "value_norm",
                "notes",
            ]
        )

    CHUNK = 16 * 1024 * 1024
    buf = ""
    scanned = 0
    with open(cda_xml_path, "rb") as f:
        while True:
            chunk = f.read(CHUNK)
            if not chunk:
                break
            try:
                s = chunk.decode("utf-8", errors="ignore")
            except Exception:
                s = chunk.decode("latin1", errors="ignore")
            scanned += len(chunk)
            buf = (buf + s)[-1000 * 4 :]  # keep tail for timestamp context
            for m in token_pat.finditer(s):
                start = max(0, m.start() - 200)
                end = min(len(s), m.end() + 200)
                snippet = s[start:end].replace("\n", "\\n")
                # try find timestamp near snippet
                ts = None
                iso = iso_ts.search(snippet)
                if iso:
                    try:
                        ts = pd.to_datetime(iso.group(0))
                    except Exception:
                        ts = None
                if ts is None:
                    y = ymdts.search(snippet)
                    if y:
                        try:
                            ts = pd.to_datetime(y.group(0))
                        except Exception:
                            ts = None
                value_raw = snippet
                vr = value_raw.lower()
                value_norm = pd.NA
                if any(
                    tok in vr
                    for tok in ("very pleasant", "pleasant", "happy", "excited", "joy")
                ):
                    value_norm = 1
                elif any(
                    tok in vr for tok in ("neutral", "calm", "ok", "okay", "fine")
                ):
                    value_norm = 0
                elif any(
                    tok in vr
                    for tok in (
                        "very unpleasant",
                        "unpleasant",
                        "sad",
                        "angry",
                        "stressed",
                        "anxious",
                    )
                ):
                    value_norm = -1

                ts_iso = ts.isoformat() if ts is not None else pd.NA
                date_utc = ""
                try:
                    if ts is not None:
                        # Ensure UTC
                        if ts.tzinfo is None:
                            ts_utc = ts.tz_localize("UTC")
                        else:
                            ts_utc = ts.astimezone(get_tz("UTC"))
                        date_utc = ts_utc.date().isoformat()
                except Exception:
                    date_utc = ""

                rows.append(
                    {
                        "timestamp_utc": ts_iso if ts is not None else pd.NA,
                        "date": date_utc,
                        "source": "apple_som",
                        "value_raw": value_raw,
                        "value_norm": (
                            int(value_norm) if value_norm is not pd.NA else pd.NA
                        ),
                        "notes": "textscan",
                    }
                )

            if scanned and scanned % (50 * 1024 * 1024) < CHUNK:
                print(f"INFO: scanned {scanned//(1024*1024)} MB...")

    if not rows:
        return pd.DataFrame(
            columns=[
                "timestamp_utc",
                "date",
                "source",
                "value_raw",
                "value_norm",
                "notes",
            ]
        )
    df = pd.DataFrame(rows)
    # dedupe by timestamp+prefix
    df = df.drop_duplicates(subset=["timestamp_utc", "value_raw"], keep="first")
    return df


def _filter_som_candidates(df_som: pd.DataFrame) -> pd.DataFrame:
    """Filter DataFrame rows to likely State-of-Mind entries.

    Keeps rows where value_norm is present OR value_raw contains mood-like tokens.
    Removes obvious noise (vitals/LOINC/SNOMED hints), dedupes and sorts by timestamp.
    """
    if df_som is None or df_som.empty:
        return pd.DataFrame(
            columns=[
                "timestamp_utc",
                "date",
                "source",
                "value_raw",
                "value_norm",
                "notes",
            ]
        )

    TOKENS = (
        "stateofmind",
        "mood",
        "mindful",
        "mindfulness",
        "pleasant",
        "unpleasant",
        "neutral",
        "anxious",
        "stressed",
        "happy",
        "sad",
        "angry",
        "valence",
        "affect",
    )

    def _has_token(s: str) -> bool:
        s = (s or "").lower()
        return any(t in s for t in TOKENS)

    df_som = df_som.copy()
    df_som["value_raw_str"] = df_som["value_raw"].astype(str)
    df_som_f = df_som[
        (df_som["value_norm"].notna()) | (df_som["value_raw_str"].map(_has_token))
    ]

    # Remove obvious noise (vitals/LOINC/SNOMED etc.)
    BAD_HINTS = (
        "vital signs",
        "39156-5",
        "result",
        "loinc",
        "snomed",
        "pq value=",
        "unit=",
    )
    bad_re = re.compile("|".join(re.escape(b) for b in BAD_HINTS), re.IGNORECASE)
    df_som_f = df_som_f[~df_som_f["value_raw_str"].str.contains(bad_re, na=False)]

    df_som_f = df_som_f.drop(columns=["value_raw_str"], errors="ignore")
    if not df_som_f.empty:
        df_som_f = df_som_f.drop_duplicates(
            subset=["timestamp_utc", "value_raw"]
        ).sort_values("timestamp_utc")
    else:
        # keep columns consistent
        df_som_f = pd.DataFrame(
            columns=[
                "timestamp_utc",
                "date",
                "source",
                "value_raw",
                "value_norm",
                "notes",
            ]
        )
    return df_som_f


# ----------------------------------------------------------------------
# Zepp HEALTH_DATA emotion/stress parser
# ----------------------------------------------------------------------
def _parse_zepp_health_emotion(
    health_csv_paths: Iterable[Union[str, Path]]
) -> pd.DataFrame:
    """Parse Zepp HEALTH_DATA CSVs to extract emotion and stress information.

    Returns DataFrame with columns: timestamp_utc, date, source, emotion_raw, emotion_norm, stress_raw, stress_norm, notes
    Mapping heuristics: common emotion labels mapped to {-1,0,1} where possible; stress_norm scaled to [0..1] per-file.
    
    Note: All timestamps are returned in UTC.
    """
    rows = []
    paths = [Path(p) for p in health_csv_paths] if health_csv_paths else []
    if not paths:
        return pd.DataFrame(
            columns=[
                "timestamp_utc",
                "date",
                "source",
                "emotion_raw",
                "emotion_norm",
                "stress_raw",
                "stress_norm",
                "notes",
            ]
        )

    for p in paths:
        if not p.exists():
            continue
        try:
            df = pd.read_csv(p)
        except Exception:
            print(f"WARNING: failed to read Zepp health CSV: {p}")
            continue

        # detect timestamp/emotion/stress columns
        cols = [c.lower() for c in df.columns]
        ts_col = None
        emotion_col = None
        stress_col = None
        for c in df.columns:
            lc = c.lower()
            if lc in ("timestamp", "time", "datetime", "ts") and ts_col is None:
                ts_col = c
            if "emotion" in lc and emotion_col is None:
                emotion_col = c
            if "stress" in lc and stress_col is None:
                stress_col = c

        # if no timestamp column, try first column
        if ts_col is None and df.shape[1] > 0:
            ts_col = df.columns[0]

        # parse stress numeric normalization per file
        stress_vals = None
        if stress_col:
            try:
                stress_vals = pd.to_numeric(df[stress_col], errors="coerce")
                s_min = (
                    float(stress_vals.min(skipna=True))
                    if not stress_vals.dropna().empty
                    else 0.0
                )
                s_max = (
                    float(stress_vals.max(skipna=True))
                    if not stress_vals.dropna().empty
                    else 1.0
                )
            except Exception:
                s_min, s_max = 0.0, 1.0
        else:
            s_min, s_max = 0.0, 1.0

        for _, r in df.iterrows():
            try:
                tval = r[ts_col] if ts_col in r else None
                ts = pd.to_datetime(tval, errors="coerce")
                if pd.isna(ts):
                    continue
                # localize naive -> assume UTC
                if ts.tzinfo is None:
                    ts = ts.tz_localize("UTC")
                ts_utc = ts.astimezone(pd.Timestamp.utcnow().tz)

                # date in UTC
                date_utc = ts_utc.date()

                emotion_raw = ""
                emotion_norm = pd.NA
                if emotion_col and emotion_col in r and pd.notna(r[emotion_col]):
                    emotion_raw = str(r[emotion_col])
                    er = emotion_raw.lower()
                    if any(
                        tok in er for tok in ("happy", "excited", "joy", "pleasant")
                    ):
                        emotion_norm = 1
                    elif any(tok in er for tok in ("neutral", "calm", "ok")):
                        emotion_norm = 0
                    elif any(
                        tok in er for tok in ("sad", "angry", "stressed", "anxious")
                    ):
                        emotion_norm = -1
                    else:
                        emotion_norm = pd.NA

                stress_raw = ""
                stress_norm = pd.NA
                if stress_col and stress_col in r and pd.notna(r[stress_col]):
                    try:
                        sval = float(r[stress_col])
                        stress_raw = sval
                        # scale to [0..1]
                        if s_max > s_min:
                            stress_norm = (sval - s_min) / (s_max - s_min)
                        else:
                            stress_norm = 0.0
                    except Exception:
                        stress_raw = str(r[stress_col])
                        stress_norm = pd.NA

                rows.append(
                    {
                        "timestamp_utc": ts_utc.isoformat(),
                        "date": date_utc.isoformat(),
                        "source": "zepp",
                        "emotion_raw": emotion_raw,
                        "emotion_norm": (
                            int(emotion_norm) if pd.notna(emotion_norm) else pd.NA
                        ),
                        "stress_raw": stress_raw,
                        "stress_norm": (
                            float(stress_norm) if pd.notna(stress_norm) else pd.NA
                        ),
                        "notes": f"parsed_from:{p.name}",
                    }
                )
            except Exception:
                continue

    if not rows:
        return pd.DataFrame(
            columns=[
                "timestamp_utc",
                "date",
                "source",
                "emotion_raw",
                "emotion_norm",
                "stress_raw",
                "stress_norm",
                "notes",
            ]
        )
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# Cardiovascular stage
# ----------------------------------------------------------------------
def run_stage_cardio(snapshot_dir: Path) -> Dict[str, str]:
    from src.etl.config import CardioCfg
    from src.etl.cardiovascular.cardio_etl import run_stage_cardio as _run

    cfg = CardioCfg()
    res = _run(str(snapshot_dir), str(snapshot_dir), cfg)
    return {k: str(v) for k, v in res.items()}


# ----------------------------------------------------------------------
# Labels stage (synthetic)
# ----------------------------------------------------------------------
def _resolve_snapshot_dir(participant: str, snapshot: str) -> Path:
    """Resolve snapshot dir without creating it (uses canonical snapshot id)."""
    snap_iso = canon_snap_id(snapshot)
    return AI_ROOT / participant / "snapshots" / snap_iso


def run_stage_labels_synthetic(
    snapshot_dir: Path, synthetic_path: Path | None = None
) -> Dict[str, str]:
    """
    Merge synthetic labels (state_of_mind_synthetic.csv) into features_daily_updated.csv

    Returns dict with output path and manifest path.
    """
    # Prefer canonical ETL new joined filename
    # Resolve canonical joined features CSV (migrates legacy if present)
    features_path = ensure_joined_snapshot(snapshot_dir)

    if not features_path.exists():
        # if data_ai has the snapshot content, attempt safe migration into data/etl
        pid = snapshot_dir.parts[-3] if len(snapshot_dir.parts) >= 3 else None
        snap = snapshot_dir.parts[-1] if len(snapshot_dir.parts) >= 1 else None
        if pid and snap:
            moved, msg = migrate_from_data_ai_if_present(pid, snap)
            if moved:
                print(f"[migrate] moved data_ai snapshot into data/etl: {msg}")
        # re-evaluate
        if not features_path.exists():
            raise FileNotFoundError(
                f"joined_features_daily.csv not found in snapshot (joined): {features_path}"
            )

    synth_path = synthetic_path or (snapshot_dir / "state_of_mind_synthetic.csv")
    if not synth_path.exists():
        raise FileNotFoundError(
            f"Synthetic labels not found: {synth_path}.\nRun: python etl_tools/generate_heuristic_labels.py --participant <PID> --snapshot <SNAP>"
        )

    # load data
    fdf = pd.read_csv(features_path, dtype={"date": "string"})
    sdf = pd.read_csv(synth_path, dtype={"date": "string"})

    if fdf.empty:
        raise ValueError(f"features_daily_updated.csv is empty: {features_path}")

    if sdf.empty:
        raise ValueError(f"Synthetic labels file is empty: {synth_path}")

    # normalize date -> datetime.date
    fdf["date"] = pd.to_datetime(fdf["date"], errors="coerce").dt.date

    # normalize synthetic columns: prefer columns named like raw_label, label, score
    cols_l = {c.lower(): c for c in sdf.columns}
    if "date" not in cols_l:
        raise ValueError(f"Synthetic labels must contain a 'date' column: {synth_path}")

    # find label/raw_label/score candidate
    def _find(cols_map, keywords):
        for kw in keywords:
            for k, orig in cols_map.items():
                if kw == k or kw in k:
                    return orig
        return None

    label_col = (
        _find(cols_l, ["raw_label"])
        or _find(cols_l, ["label"])
        or _find(cols_l, ["state"])
    )
    # if label_col is raw_label, try to find a nicer 'label' column
    if label_col and label_col.lower().startswith("raw_"):
        raw_label_col = label_col
        label_col = _find(cols_l, ["label"]) or raw_label_col
    else:
        raw_label_col = _find(cols_l, ["raw_label"]) or None

    if not label_col:
        # try any column that contains 'label'
        label_col = _find(cols_l, ["label"]) if _find(cols_l, ["label"]) else None

    score_col = _find(cols_l, ["score", "value", "prob"]) or None

    # standardize synthetic df
    sdf = sdf.rename(columns={cols_l.get("date"): "date"} if "date" in cols_l else {})
    sdf["date"] = pd.to_datetime(sdf["date"], errors="coerce").dt.date

    # ensure label column exists
    if label_col:
        sdf = sdf.rename(columns={label_col: "label"})
    else:
        raise ValueError(
            f"No label-like column found in synthetic labels: {synth_path}"
        )

    if raw_label_col:
        sdf = sdf.rename(columns={raw_label_col: "raw_label"})
    else:
        # create raw_label same as label
        sdf["raw_label"] = sdf["label"]

    if score_col:
        sdf = sdf.rename(columns={score_col: "score"})
    else:
        if "score" not in sdf.columns:
            sdf["score"] = pd.NA

    # aggregate synthetic by date in case of duplicates: mean score, mode label
    def _mode_series(s):
        try:
            m = s.mode(dropna=True)
            return m.iloc[0] if not m.empty else s.iloc[0]
        except Exception:
            return s.iloc[0]

    aggs = {}
    if "score" in sdf.columns:
        aggs["score"] = "mean"
    aggs["label"] = lambda x: _mode_series(x)
    aggs["raw_label"] = lambda x: _mode_series(x)

    sdf_grouped = sdf.groupby("date", dropna=False).agg(aggs).reset_index()

    # merge
    out = fdf.merge(sdf_grouped, on="date", how="left")

    # write outputs
    out_csv = snapshot_dir / "joined" / "features_daily_labeled.csv"
    _write_atomic_csv(out, out_csv)

    # manifest
    counts = {}
    try:
        counts = dict(pd.Series(out["label"].dropna()).value_counts().to_dict())
    except Exception:
        counts = {}

    manifest = {
        "type": "labels_synthetic",
        "snapshot_dir": str(snapshot_dir),
        "inputs": {
            "features_daily_updated.csv": str(features_path),
            "state_of_mind_synthetic.csv": str(synth_path),
        },
        "outputs": {
            "features_daily_labeled.csv": _sha256_file(str(out_csv)),
            "label_counts": counts,
            "rows_labeled": int(out["label"].notna().sum()),
            "rows_total": int(len(out)),
        },
    }

    manifest_path = snapshot_dir / "labels_manifest.json"
    _write_atomic_json(manifest, manifest_path)

    return {
        "features_daily_labeled": str(out_csv),
        "labels_manifest": str(manifest_path),
    }


# ----------------------------------------------------------------------
# Labels (full): Apple SoM + Zepp Emotion merging
# ----------------------------------------------------------------------
# label mapping dictionaries
LABEL_MAP_APPLE_TERNARY = {
    "very_pleasant": "positive",
    "pleasant": "positive",
    "neutral": "neutral",
    "unpleasant": "negative",
    "very_unpleasant": "negative",
}

LABEL_MAP_ZEPP_TERNARY = {
    "happy": "positive",
    "excited": "positive",
    "calm": "neutral",
    "neutral": "neutral",
    "sad": "negative",
    "angry": "negative",
    "stressed": "negative",
    "anxious": "negative",
}

LABEL_MAP_APPLE_BINARY = {
    k: ("positive" if v == "positive" else "negative")
    for k, v in LABEL_MAP_APPLE_TERNARY.items()
}
LABEL_MAP_ZEPP_BINARY = {
    k: ("positive" if v == "positive" else "negative")
    for k, v in LABEL_MAP_ZEPP_TERNARY.items()
}

# five-class (Apple native, Zepp best-effort mapping)
LABEL_MAP_APPLE_FIVE = {
    "very_unpleasant": "very_unpleasant",
    "unpleasant": "unpleasant",
    "neutral": "neutral",
    "pleasant": "pleasant",
    "very_pleasant": "very_pleasant",
}

LABEL_MAP_ZEPP_FIVE = {
    "angry": "very_unpleasant",
    "stressed": "unpleasant",
    "anxious": "unpleasant",
    "sad": "unpleasant",
    "neutral": "neutral",
    "calm": "neutral",
    "happy": "pleasant",
    "excited": "pleasant",
}

SEVERITY_TERNARY = {"negative": 0, "neutral": 1, "positive": 2}
SEVERITY_FIVE = {
    "very_unpleasant": 0,
    "unpleasant": 1,
    "neutral": 2,
    "pleasant": 3,
    "very_pleasant": 4,
}


def _write_if_changed_df(df: "pd.DataFrame", out_path: Path) -> bool:
    """Write DataFrame to out_path atomically only if content changed.
    Returns True if written (changed), False if skipped because identical.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(
        prefix=".tmp_labels_", dir=str(out_path.parent), suffix=".csv"
    )
    os.close(fd)
    try:
        df.to_csv(tmp, index=False)
        if out_path.exists():
            try:
                old = _sha256_file(str(out_path))
                new = _sha256_file(tmp)
                if old == new:
                    os.remove(tmp)
                    return False
            except Exception:
                pass
        os.replace(tmp, str(out_path))
        return True
    finally:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except Exception:
                pass


def _parse_apple_autoexport_som(file_path: Path) -> pd.DataFrame:
    """Parse a Health Auto Export StateOfMind CSV and normalize it.

    Expected columns: Start, End, Kind, Labels, Associations,
                      Valence, Valence Classification
    Returns columns: timestamp_utc, date, source, value_raw, value_norm, notes
    
    Note: All timestamps are returned in UTC. Daily binning is at UTC midnight.
    """
    # If the file does not exist, return empty frame with expected columns
    if not file_path or not Path(file_path).exists():
        return pd.DataFrame(
            columns=[
                "timestamp_utc",
                "date",
                "source",
                "value_raw",
                "value_norm",
                "notes",
            ]
        )

    # Read whole file as text and recover records by finding timestamp anchors.
    try:
        txt = Path(file_path).read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return pd.DataFrame(
            columns=[
                "timestamp_utc",
                "date",
                "source",
                "value_raw",
                "value_norm",
                "notes",
            ]
        )

    # regex to find start-of-record timestamps like '2025-08-03 01:44:43 +0100'
    ts_re = re.compile(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} [+-]\d{4}")
    matches = list(ts_re.finditer(txt))
    if not matches:
        return pd.DataFrame(
            columns=[
                "timestamp_utc",
                "date",
                "source",
                "value_raw",
                "value_norm",
                "notes",
            ]
        )

    POS_SET = {
        "very pleasant",
        "pleasant",
        "happy",
        "joy",
        "excited",
        "great",
        "good",
        "positive",
        "excellent",
        "content",
        "calm",
        "relaxed",
        "ok",
        "okay",
        "fine",
    }
    NEU_SET = {"neutral", "average", "normal", "so-so", "soso", "ok"}
    NEG_SET = {
        "very unpleasant",
        "unpleasant",
        "sad",
        "angry",
        "stressed",
        "anxious",
        "bad",
        "negative",
        "depressed",
        "down",
        "awful",
        "terrible",
    }

    def _map_valence_text(s: str):
        if not s:
            return pd.NA
        s = s.strip().lower()
        if s in POS_SET:
            return 1
        if s in NEU_SET:
            return 0
        if s in NEG_SET:
            return -1
        if re.search(r"(very\s+pleasant|pleasant|happy|joy|excited|positive)", s):
            return 1
        if re.search(r"(neutral|average|normal|so-so|soso)", s):
            return 0
        if re.search(
            r"(very\s+unpleasant|unpleasant|sad|angry|stressed|anxious|negative|depress|awful|terrible|down|bad)",
            s,
        ):
            return -1
        return pd.NA

    def _parse_numeric_bucket_raw(v):
        try:
            f = float(str(v))
        except Exception:
            return pd.NA
        # reuse previous heuristics
        if 1.0 <= f <= 5.0:
            if f <= 2.0:
                return -1
            if abs(f - 3.0) < 0.001:
                return 0
            return 1
        if 0.0 <= f <= 1.0:
            if f < 0.33:
                return -1
            if f <= 0.66:
                return 0
            return 1
        if -1.0 <= f <= 1.0:
            if f < -0.33:
                return -1
            if f <= 0.33:
                return 0
            return 1
        if 0.0 <= f <= 100.0:
            if f < 33.0:
                return -1
            if f <= 66.0:
                return 0
            return 1
        # fallback
        try:
            nf = (f - 0.0) / (abs(f) + 1.0)
            if nf < -0.33:
                return -1
            if nf <= 0.33:
                return 0
            return 1
        except Exception:
            return pd.NA

    rows = []
    for i, m in enumerate(matches):
        start_idx = m.start()
        end_idx = matches[i + 1].start() if i + 1 < len(matches) else len(txt)
        chunk = txt[start_idx:end_idx]

        # The first token is the Start timestamp
        m2 = ts_re.match(chunk)
        if not m2:
            continue
        ts_raw = m2.group(0)

        # Try to extract valence and classification as the last two comma-separated fields
        parts = chunk.rsplit(",", 2)
        if len(parts) >= 3:
            head, valence_raw, class_raw = parts[0], parts[1], parts[2]
        elif len(parts) == 2:
            head, valence_raw = parts[0], parts[1]
            class_raw = ""
        else:
            # can't parse
            continue

        valence_raw = str(valence_raw).strip()
        class_raw = str(class_raw).strip()

        # parse timestamp (respect offset) and convert to UTC
        try:
            ts = pd.to_datetime(ts_raw, utc=True, errors="coerce")
        except Exception:
            ts = pd.to_datetime(ts_raw, errors="coerce")
            if not pd.isna(ts) and ts.tzinfo is None:
                try:
                    ts = ts.tz_localize("UTC")
                except Exception:
                    ts = pd.NaT
        if pd.isna(ts):
            continue
        try:
            ts_utc = (
                ts.tz_convert("UTC") if ts.tzinfo is not None else ts.tz_localize("UTC")
            )
        except Exception:
            try:
                ts_utc = pd.to_datetime(ts).tz_localize("UTC")
            except Exception:
                continue

        ts_iso = ts_utc.isoformat()
        # date in UTC
        date_utc = ts_utc.date().isoformat()

        # compute normalized value: prefer numeric valence, else classification text
        value_norm = pd.NA
        notes = "autoexport"
        # numeric try
        vnum = _parse_numeric_bucket_raw(valence_raw)
        if not (vnum is pd.NA or pd.isna(vnum)):
            value_norm = int(vnum)
            notes = "autoexport:valence"
        else:
            # maybe numeric is in the class_raw (some files swap cols)
            vnum2 = _parse_numeric_bucket_raw(class_raw)
            if not (vnum2 is pd.NA or pd.isna(vnum2)):
                value_norm = int(vnum2)
                notes = "autoexport:valence"
        # classification fallback
        if value_norm is pd.NA:
            vtxt = class_raw if class_raw else valence_raw
            mapped = _map_valence_text(vtxt)
            if not (mapped is pd.NA or pd.isna(mapped)):
                value_norm = int(mapped)
                notes = "autoexport:classification"

        if value_norm is pd.NA or pd.isna(value_norm):
            # skip rows we cannot map
            continue

        value_raw = f"{valence_raw}|{class_raw}".strip("|")
        rows.append(
            {
                "timestamp_utc": ts_iso,
                "date": date_utc,
                "source": "apple_autoexport",
                "value_raw": value_raw,
                "value_norm": int(value_norm),
                "notes": notes,
            }
        )

    if not rows:
        return pd.DataFrame(
            columns=[
                "timestamp_utc",
                "date",
                "source",
                "value_raw",
                "value_norm",
                "notes",
            ]
        )
    out = pd.DataFrame(rows)
    out = out.sort_values("timestamp_utc").drop_duplicates(
        subset=["timestamp_utc", "value_raw", "value_norm"], keep="first"
    )
    return out


def run_stage_labels(
    snapshot_dir: Path,
    label_schema: str = "ternary",
    prefer: str = "mixed",
    dry_run: bool = False,
) -> Dict[str, str]:
    """Merge Apple SoM and Zepp Emotion into features_daily_labeled.csv.

    snapshot_dir: Path to data/etl/<PID>/snapshots/<SNAP>/
    Returns a manifest dict like the synthetic runner.
    """
    # locate features base (prefer canonical ETL joined filename)
    pid = snapshot_dir.parts[-3] if len(snapshot_dir.parts) >= 3 else None
    snap_iso = snapshot_dir.parts[-1] if len(snapshot_dir.parts) >= 1 else None
    joined_dir = snapshot_dir / "joined"
    etl_joined = joined_dir / "joined_features_daily.csv"
    old_joined = joined_dir / "features_daily.csv"
    ai_fallback = AI_ROOT / pid / "snapshots" / snap_iso / "features_daily_updated.csv"

    # migrate old canonical name if present
    if not etl_joined.exists() and old_joined.exists():
        warnings.warn(
            f"Found legacy joined/features_daily.csv at {old_joined}; migrating to {etl_joined}",
            stacklevel=2,
        )
        try:
            joined_dir.mkdir(parents=True, exist_ok=True)
            old_joined.replace(etl_joined)
        except Exception:
            try:
                shutil.copy2(old_joined, etl_joined)
            except Exception:
                pass

    if etl_joined.exists():
        features_path = etl_joined
        used_backcompat = False
    else:
        # attempt safe migration from data_ai if present
        pid = snapshot_dir.parts[-3] if len(snapshot_dir.parts) >= 3 else None
        snap = snapshot_dir.parts[-1] if len(snapshot_dir.parts) >= 1 else None
        if pid and snap:
            moved, msg = migrate_from_data_ai_if_present(pid, snap)
            if moved:
                print(f"[migrate] moved data_ai snapshot into data/etl: {msg}")
        if etl_joined.exists():
            features_path = etl_joined
            used_backcompat = False
        elif ai_fallback.exists():
            # final fallback (read-only) from legacy data_ai
            features_path = ai_fallback
            used_backcompat = True
        else:
            raise FileNotFoundError(
                f"features_daily not found in either {etl_joined} or {ai_fallback}"
            )

    fdf = pd.read_csv(features_path, dtype={"date": "string"})
    if fdf.empty:
        raise ValueError(f"features_daily is empty: {features_path}")
    fdf["date"] = pd.to_datetime(fdf["date"], errors="coerce").dt.date

    # label sources under snapshot_dir (ETL canonical layout)
    apple_path = snapshot_dir / "normalized" / "apple_state_of_mind.csv"
    apple_auto_path = snapshot_dir / "normalized" / "apple_state_of_mind_autoexport.csv"
    zepp_path = snapshot_dir / "normalized" / "zepp" / "zepp_emotion.csv"

    apple_df = pd.DataFrame()
    zepp_df = pd.DataFrame()

    # helper to detect ts and label cols
    def _detect_ts_label(df: pd.DataFrame):
        cols = [c.lower() for c in df.columns]
        ts_candidates = [
            c
            for c in df.columns
            if c.lower() in ("timestamp", "time", "ts", "datetime", "date")
        ]
        lbl_candidates = [
            c
            for c in df.columns
            if any(k in c.lower() for k in ("label", "state", "emotion", "mood"))
        ]
        ts_col = (
            ts_candidates[0]
            if ts_candidates
            else (df.columns[0] if df.shape[1] >= 1 else None)
        )
        lbl_col = (
            lbl_candidates[0]
            if lbl_candidates
            else (df.columns[1] if df.shape[1] >= 2 else None)
        )
        return ts_col, lbl_col

    # infer tz_name from version_log_enriched if possible
    tz_name = None
    vpath = snapshot_dir / VERSION_LOG
    if not vpath.exists():
        # try ai fallback
        vpath = AI_ROOT / pid / "snapshots" / snap_iso / VERSION_LOG
    try:
        if vpath.exists():
            vdf = pd.read_csv(vpath)
            if "tz_name" in vdf.columns and not vdf["tz_name"].dropna().empty:
                tz_name = vdf["tz_name"].dropna().iloc[0]
    except Exception:
        tz_name = None

    target_tz = get_tz(tz_name) if tz_name else None

    # load apple (State-of-Mind) — robust text-column detection + series-safe regex mapping
    apple_days = {}
    apple_labels_daily = pd.DataFrame()
    if apple_path.exists():
        try:
            adf = pd.read_csv(apple_path)
            # pick first available text column
            text_candidates = ["value_raw", "text", "value", "narrative", "notes"]
            text_col = next((c for c in text_candidates if c in adf.columns), None)
            if text_col is None:
                print(
                    f"WARNING: apple_state_of_mind.csv exists but no text column found among {text_candidates}; skipping Apple SoM."
                )
                apple_df = pd.DataFrame()
            else:
                # ensure date exists or derive from timestamp_utc if present
                if "date" not in adf.columns:
                    if "timestamp_utc" in adf.columns:
                        try:
                            adf["date"] = pd.to_datetime(
                                adf["timestamp_utc"], errors="coerce"
                            ).dt.date
                        except Exception:
                            adf["date"] = pd.NaT
                    else:
                        adf["date"] = pd.NaT

                # prepare text series (Series, lowercased)
                som_text = adf[text_col].astype("string").str.lower()

                # non-capturing regexes to avoid pandas UserWarning
                POS_RE = re.compile(
                    r"(?:very pleasant|pleasant|happy|excited|joy|positive)",
                    re.IGNORECASE,
                )
                NEU_RE = re.compile(r"(?:neutral|calm|ok|okay|fine)", re.IGNORECASE)
                NEG_RE = re.compile(
                    r"(?:very unpleasant|unpleasant|sad|angry|stressed|anxious|negative)",
                    re.IGNORECASE,
                )

                # prefer numeric norm if present
                adf["_label"] = pd.NA
                if "value_norm" in adf.columns and pd.api.types.is_numeric_dtype(
                    adf["value_norm"]
                ):
                    try:
                        adf.loc[adf["value_norm"].notna(), "_label"] = adf.loc[
                            adf["value_norm"].notna(), "value_norm"
                        ].astype("Int8")
                    except Exception:
                        # fallback to text mapping below
                        adf["_label"] = pd.NA

                # regex fallback (only on the prepared Series)
                if adf["_label"].isna().all():
                    # POS, then NEU, then NEG (order matters)
                    try:
                        adf.loc[som_text.str.contains(POS_RE, na=False), "_label"] = 1
                        adf.loc[som_text.str.contains(NEU_RE, na=False), "_label"] = 0
                        adf.loc[som_text.str.contains(NEG_RE, na=False), "_label"] = -1
                    except Exception:
                        # in case som_text is malformed, ensure we don't crash
                        adf["_label"] = pd.NA

                # keep only rows with a label
                labdf = adf.dropna(subset=["_label"]).copy()
                if labdf.empty:
                    print(
                        "WARNING: Apple SoM present but no tokens matched; skipping Apple SoM."
                    )
                    apple_df = pd.DataFrame()
                else:
                    # normalize label to int in {-1,0,1}
                    labdf["_label"] = labdf["_label"].astype("Int8")
                    # ensure date column is date type
                    if labdf["date"].dtype == "O":
                        try:
                            labdf["date"] = pd.to_datetime(
                                labdf["date"], errors="coerce"
                            ).dt.date
                        except Exception:
                            pass

                    # aggregate per date using median (POS>NEU>NEG mapping via numeric)
                    apple_daily = labdf.groupby("date", as_index=False)[
                        "_label"
                    ].median()
                    apple_daily = apple_daily.rename(columns={"_label": "label"})
                    apple_daily["label"] = apple_daily["label"].astype("Int8")
                    apple_daily["label_source"] = "apple"
                    apple_daily["label_notes"] = "som:regex"
                    apple_labels_daily = apple_daily[
                        ["date", "label", "label_source", "label_notes"]
                    ]
                    # also expose apple_df in legacy shape for downstream compatibility
                    apple_df = labdf[["date", "_label"]].rename(
                        columns={"_label": "raw_label"}
                    )
        except Exception:
            apple_df = pd.DataFrame()

    # load zepp
    if zepp_path.exists():
        try:
            zdf = pd.read_csv(zepp_path)
            ts_col, lbl_col = _detect_ts_label(zdf)
            if ts_col and lbl_col:
                zdf["__ts"] = pd.to_datetime(zdf[ts_col], errors="coerce")
                if target_tz is not None:
                    zdf["__ts"] = (
                        zdf["__ts"]
                        .dt.tz_localize("UTC", ambiguous="NaT", nonexistent="NaT")
                        .dt.tz_convert(target_tz)
                    )
                else:
                    zdf["__ts"] = zdf["__ts"].dt.tz_localize(
                        "UTC", ambiguous="NaT", nonexistent="NaT"
                    )
                zdf["date"] = zdf["__ts"].dt.date
                zepp_df = zdf[["date", lbl_col]].rename(columns={lbl_col: "raw_label"})
        except Exception:
            zepp_df = pd.DataFrame()

    # Decide if we have any label sources (apple_labels_daily, apple_auto, or zepp daily will be used)
    zepp_labels_daily = pd.DataFrame()
    # treat apple_autoexport presence as a valid source as well (highest precedence)
    has_sources = (
        (apple_labels_daily is not None and not apple_labels_daily.empty)
        or (zepp_labels_daily is not None and not zepp_labels_daily.empty)
        or (apple_auto_path.exists())
    )
    if not has_sources:
        print("WARNING: no label sources found — generating synthetic neutral labels.")
        synth = pd.DataFrame(
            {
                "date": fdf["date"],
                "label": [
                    "neutral" if label_schema != "binary" else "positive"
                    for _ in range(len(fdf))
                ],
            }
        )
        out = fdf.merge(synth, on="date", how="left")
        out["label_source"] = "synthetic"
        out["label_notes"] = ""
        # write outputs
        out_path = snapshot_dir / "joined" / "features_daily_labeled.csv"
        if not dry_run:
            _write_if_changed_df(out, out_path)
            if used_backcompat:
                back_path = (
                    AI_ROOT
                    / pid
                    / "snapshots"
                    / snap_iso
                    / "features_daily_labeled.csv"
                )
                _write_if_changed_df(out, back_path)
        manifest = {
            "type": "labels_synthetic_fallback",
            "snapshot_dir": str(snapshot_dir),
            "inputs": {},
            "outputs": {"features_daily_labeled.csv": str(out_path)},
        }
        manifest_path = snapshot_dir / "labels_manifest.json"
        if not dry_run:
            _write_atomic_json(manifest, manifest_path)
        return {
            "features_daily_labeled": str(out_path),
            "labels_manifest": str(manifest_path),
        }

    # New labels handling (series-safe, numeric-first then regex fallback)
    # Regexes for mapping text -> {-1,0,1}
    POS_RE = re.compile(
        r"\b(very\s+pleasant|pleasant|happy|excited|joy|positive)\b", re.I
    )
    NEU_RE = re.compile(r"\b(neutral|calm|ok(ay)?|fine)\b", re.I)
    NEG_RE = re.compile(
        r"\b(very\s+unpleasant|unpleasant|sad|angry|stressed|anxious|negative)\b", re.I
    )

    # Helper to ensure a date column exists (as datetime.date)
    def _ensure_date_col(df, tz=None):
        if "date" in df.columns and not df["date"].isnull().all():
            # try to coerce to date
            try:
                df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
            except Exception:
                pass
            return df
        if "timestamp_utc" in df.columns:
            try:
                df["__ts"] = pd.to_datetime(df["timestamp_utc"], errors="coerce")
                if tz is not None:
                    df["__ts"] = (
                        df["__ts"]
                        .dt.tz_localize("UTC", ambiguous="NaT", nonexistent="NaT")
                        .dt.tz_convert(tz)
                    )
                else:
                    df["__ts"] = df["__ts"].dt.tz_localize(
                        "UTC", ambiguous="NaT", nonexistent="NaT"
                    )
                df["date"] = df["__ts"].dt.date
            except Exception:
                df["date"] = pd.NaT
        return df

    # Prepare apple source: prefer numeric value_norm, else derive from value_raw text
    if not apple_df.empty:
        adf = apple_df.copy()
        adf = _ensure_date_col(adf, target_tz)
        # compute _label
        if "value_norm" in adf.columns and pd.api.types.is_numeric_dtype(
            adf["value_norm"]
        ):
            adf["_label"] = adf["value_norm"].astype("Int8")
        else:
            som_text = adf.get("value_raw", pd.Series([""] * len(adf))).astype("string")
            adf["_label"] = pd.NA
            try:
                adf.loc[som_text.str.contains(POS_RE, na=False), "_label"] = 1
                adf.loc[som_text.str.contains(NEU_RE, na=False), "_label"] = 0
                adf.loc[som_text.str.contains(NEG_RE, na=False), "_label"] = -1
            except Exception:
                # defensive: ensure som_text is a Series
                som_text = pd.Series(som_text).astype("string")
                adf.loc[som_text.str.contains(POS_RE, na=False), "_label"] = 1
                adf.loc[som_text.str.contains(NEU_RE, na=False), "_label"] = 0
                adf.loc[som_text.str.contains(NEG_RE, na=False), "_label"] = -1
            adf["_label"] = adf["_label"].astype("Int8")
        # keep only date and _label
        apple_daily = pd.DataFrame()
        if "_label" in adf.columns and "date" in adf.columns:
            tmp = adf.dropna(subset=["_label"]).copy()
            if not tmp.empty:
                # median per day and round to nearest integer
                apple_daily = tmp.groupby("date", as_index=False)["_label"].median()
                apple_daily["_label"] = apple_daily["_label"].round().astype("Int8")
                apple_daily["label_source"] = "apple"
        else:
            apple_daily = pd.DataFrame()
    else:
        apple_daily = pd.DataFrame()

    # Prepare zepp source similarly (emotion_raw / emotion_norm)
    if not zepp_df.empty:
        zdf = zepp_df.copy()
        zdf = _ensure_date_col(zdf, target_tz)
        if "emotion_norm" in zdf.columns and pd.api.types.is_numeric_dtype(
            zdf["emotion_norm"]
        ):
            zdf["_label"] = zdf["emotion_norm"].astype("Int8")
        else:
            zepp_text = zdf.get("emotion_raw", pd.Series([""] * len(zdf))).astype(
                "string"
            )
            zdf["_label"] = pd.NA
            try:
                zdf.loc[zepp_text.str.contains(POS_RE, na=False), "_label"] = 1
                zdf.loc[zepp_text.str.contains(NEU_RE, na=False), "_label"] = 0
                zdf.loc[zepp_text.str.contains(NEG_RE, na=False), "_label"] = -1
            except Exception:
                zepp_text = pd.Series(zepp_text).astype("string")
                zdf.loc[zepp_text.str.contains(POS_RE, na=False), "_label"] = 1
                zdf.loc[zepp_text.str.contains(NEU_RE, na=False), "_label"] = 0
                zdf.loc[zepp_text.str.contains(NEG_RE, na=False), "_label"] = -1
            zdf["_label"] = zdf["_label"].astype("Int8")
        zepp_daily = pd.DataFrame()
        if "_label" in zdf.columns and "date" in zdf.columns:
            tmpz = zdf.dropna(subset=["_label"]).copy()
            if not tmpz.empty:
                zepp_daily = tmpz.groupby("date", as_index=False)["_label"].median()
                zepp_daily["_label"] = zepp_daily["_label"].round().astype("Int8")
                zepp_daily["label_source"] = "zepp"
        else:
            zepp_daily = pd.DataFrame()
    else:
        zepp_daily = pd.DataFrame()

    # Compose labels_daily with precedence: apple > zepp
    # load apple_autoexport labels if present (highest precedence)
    apple_auto_daily = pd.DataFrame()
    apple_auto_path = snapshot_dir / "normalized" / "apple_state_of_mind_autoexport.csv"
    if apple_auto_path.exists():
        try:
            aaf = pd.read_csv(apple_auto_path)
            if "date" not in aaf.columns:
                if "timestamp_utc" in aaf.columns:
                    aaf["date"] = pd.to_datetime(
                        aaf["timestamp_utc"], errors="coerce"
                    ).dt.date
                else:
                    aaf["date"] = pd.NaT
            # prefer numeric value_norm
            if "value_norm" in aaf.columns and pd.api.types.is_numeric_dtype(
                aaf["value_norm"]
            ):
                aaf["label"] = aaf["value_norm"].astype("Int8")
            else:
                # try to map classification-like text
                txt_col = next(
                    (
                        c
                        for c in aaf.columns
                        if "class" in c.lower() or "valence" in c.lower()
                    ),
                    None,
                )
                if txt_col:
                    t = aaf[txt_col].astype("string").str.lower()
                    aaf["label"] = pd.NA
                    aaf.loc[
                        t.str.contains(
                            r"(?:very pleasant|pleasant|happy|joy|joyful|positive)",
                            na=False,
                        ),
                        "label",
                    ] = 1
                    aaf.loc[
                        t.str.contains(
                            r"(?:neutral|uncertain|ok|okay|fine|calm)", na=False
                        ),
                        "label",
                    ] = 0
                    aaf.loc[
                        t.str.contains(
                            r"(?:very unpleasant|unpleasant|sad|angry|stressed|anxious|negative)",
                            na=False,
                        ),
                        "label",
                    ] = -1
                    try:
                        aaf["label"] = aaf["label"].astype("Int8")
                    except Exception:
                        pass
                else:
                    aaf["label"] = pd.NA
            tmpa = aaf.dropna(subset=["label"]).copy()
            if not tmpa.empty and "date" in tmpa.columns:
                apple_auto_daily = tmpa.groupby("date", as_index=False)[
                    "label"
                ].median()
                apple_auto_daily["label"] = (
                    apple_auto_daily["label"].round().astype("Int8")
                )
                apple_auto_daily["label_source"] = "apple_autoexport"
                apple_auto_daily["label_notes"] = "som:autoexport"
        except Exception:
            apple_auto_daily = pd.DataFrame()

    # Build a complete daily index from the features frame
    dates_df = pd.DataFrame({"date": fdf["date"]}).drop_duplicates().copy()

    # Prepare apple_auto_daily, apple_cda_daily, zepp_daily into consistent daily frames
    # apple_auto_daily already has 'date' and 'label' (numeric) if present
    if "apple_auto_daily" not in locals():
        apple_auto_daily = pd.DataFrame()
    if (
        "apple_labels_daily" in locals()
        and (apple_labels_daily is not None)
        and (not apple_labels_daily.empty)
    ):
        # legacy apple SoM produced earlier (label numeric)
        apple_cda_daily = apple_labels_daily[["date", "label"]].copy()
        apple_cda_daily["label_source"] = "apple_cda"
        apple_cda_daily["label_notes"] = "som:cda"
    elif (
        "apple_daily" in locals()
        and (apple_daily is not None)
        and (not apple_daily.empty)
    ):
        # apple_daily may have been produced from a different path; normalize
        tmp = apple_daily.rename(columns={"_label": "label"})[["date", "label"]].copy()
        tmp["label_source"] = "apple_cda"
        tmp["label_notes"] = "som:cda"
        apple_cda_daily = tmp
    else:
        apple_cda_daily = pd.DataFrame()

    if "zepp_daily" not in locals():
        zepp_daily = pd.DataFrame()

    # ensure apple_auto_daily labels have correct column names/notes
    if not apple_auto_daily.empty:
        aauto = apple_auto_daily[["date", "label"]].copy()
        try:
            aauto["date"] = pd.to_datetime(aauto["date"], errors="coerce").dt.date
        except Exception:
            pass
        aauto["label_source"] = "apple_autoexport"
        aauto["label_notes"] = "som:autoexport"
        apple_auto_daily = aauto

    # Merge daily labels onto date index
    merged = dates_df.copy()
    if not apple_auto_daily.empty:
        merged = merged.merge(
            apple_auto_daily[["date", "label", "label_source", "label_notes"]].rename(
                columns={
                    "label": "label_auto",
                    "label_source": "src_auto",
                    "label_notes": "notes_auto",
                }
            ),
            on="date",
            how="left",
        )
    else:
        merged["label_auto"] = pd.NA
        merged["src_auto"] = pd.NA
        merged["notes_auto"] = pd.NA

    if not apple_cda_daily.empty:
        try:
            apple_cda_daily["date"] = pd.to_datetime(
                apple_cda_daily["date"], errors="coerce"
            ).dt.date
        except Exception:
            pass
        merged = merged.merge(
            apple_cda_daily[["date", "label", "label_source", "label_notes"]].rename(
                columns={
                    "label": "label_cda",
                    "label_source": "src_cda",
                    "label_notes": "notes_cda",
                }
            ),
            on="date",
            how="left",
        )
    else:
        merged["label_cda"] = pd.NA
        merged["src_cda"] = pd.NA
        merged["notes_cda"] = pd.NA

    if not zepp_daily.empty:
        # zepp_daily may have '_label' or 'label'
        zdf = zepp_daily.copy()
        if "_label" in zdf.columns:
            zdf = zdf.rename(columns={"_label": "label"})
        try:
            zdf["date"] = pd.to_datetime(zdf["date"], errors="coerce").dt.date
        except Exception:
            pass
        merged = merged.merge(
            zdf[["date", "label"]].rename(columns={"label": "label_zepp"}),
            on="date",
            how="left",
        )
        # zepp notes/source set below when chosen
    else:
        merged["label_zepp"] = pd.NA

    # Pick label by precedence: apple_auto -> apple_cda -> zepp -> synthetic
    def _choose_label(row):
        for col, src, notes in (
            ("label_auto", "apple_autoexport", "som:autoexport"),
            ("label_cda", "apple_cda", "som:cda"),
            ("label_zepp", "zepp", "zepp:emotion"),
        ):
            v = row.get(col)
            if pd.notna(v):
                try:
                    iv = int(round(float(v)))
                except Exception:
                    continue
                if iv in (-1, 0, 1):
                    return iv, src, notes
        return pd.NA, "synthetic", ""

    picks = merged.apply(_choose_label, axis=1)
    merged[["label_num", "label_source", "label_notes"]] = pd.DataFrame(
        picks.tolist(), index=merged.index
    )

    # Map numeric label to string if requested
    def _num_to_str(v):
        return {1: "positive", 0: "neutral", -1: "negative"}.get(v, pd.NA)

    if label_schema == "ternary":
        merged["label"] = merged["label_num"].apply(
            lambda x: _num_to_str(int(x)) if pd.notna(x) else pd.NA
        )
    else:
        # keep numeric for non-ternary schemas
        merged["label"] = merged["label_num"]

    # For synthetic rows (label_source == 'synthetic'), set label to 'neutral' (string) per spec
    merged.loc[merged["label_source"] == "synthetic", "label"] = (
        "neutral" if label_schema == "ternary" else "neutral"
    )

    # finalize label_source and label_notes columns
    merged["label_source"] = merged["label_source"].astype("string")
    merged["label_notes"] = merged["label_notes"].astype("string")

    # Merge into feature frame (preserve fdf columns order)
    try:
        merged["date"] = pd.to_datetime(merged["date"], errors="coerce").dt.date
    except Exception:
        pass
    out = fdf.merge(
        merged[["date", "label", "label_source", "label_notes"]], on="date", how="left"
    )

    # write outputs and manifest
    out_path = snapshot_dir / "joined" / "features_daily_labeled.csv"
    if not dry_run:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        written = _write_if_changed_df(out, out_path)
        # Do NOT write new outputs to data_ai/. If old files exist there, they were migrated above.

        # manifest with counts
        counts_by_source = {
            "apple_autoexport": int((out["label_source"] == "apple_autoexport").sum()),
            "apple_cda": int((out["label_source"] == "apple_cda").sum()),
            "zepp": int((out["label_source"] == "zepp").sum()),
            "synthetic": int((out["label_source"] == "synthetic").sum()),
        }

        # ensure label_notes for apple_autoexport rows are preserved
        try:
            mask = (out["label_source"] == "apple_autoexport") & (
                out["label_notes"].isna() | (out["label_notes"] == "")
            )
            out.loc[mask, "label_notes"] = "som:autoexport"
        except Exception:
            # defensive: if label_notes missing, create it
            if "label_notes" not in out.columns:
                out["label_notes"] = ""
            mask = (out["label_source"] == "apple_autoexport") & (
                out["label_notes"].isna() | (out["label_notes"] == "")
            )
            out.loc[mask, "label_notes"] = "som:autoexport"
        # compute real_labels (non-synthetic)
        real_labels = int(
            counts_by_source.get("apple_autoexport", 0)
            + counts_by_source.get("apple_cda", 0)
            + counts_by_source.get("zepp", 0)
        )

        manifest = {
            "snapshot": snap_iso,
            "participant": pid,
            "dates_total": int(len(out)),
            "dates_with_label": int(out["label"].notna().sum()),
            "real_labels": real_labels,
            "counts_by_source": counts_by_source,
            "precedence": ["apple_autoexport", "apple_cda", "zepp", "synthetic"],
            "files_used": {
                "apple_autoexport": (
                    str(apple_auto_path) if apple_auto_path.exists() else ""
                ),
                "apple_cda": str(apple_path) if apple_path.exists() else "",
                "zepp_emotion": str(zepp_path) if zepp_path.exists() else "",
            },
        }
        manifest_path = snapshot_dir / "labels_manifest.json"
        _write_atomic_json(manifest, manifest_path)

        # ASCII-only log summary
        summary = f"[OK] labels: total={manifest['dates_total']}, real={real_labels}, apple_autoexport={counts_by_source['apple_autoexport']}, apple_cda={counts_by_source['apple_cda']}, zepp={counts_by_source['zepp']}, synthetic={counts_by_source['synthetic']} -> {out_path}"
        try:
            print(summary.encode("ascii", "replace").decode("ascii"))
        except Exception:
            print(summary)
    else:
        manifest = {"type": "labels_dry_run", "snapshot_dir": str(snapshot_dir)}

    return {
        "features_daily_labeled": str(out_path),
        "labels_manifest": str(manifest_path) if not dry_run else "",
    }


def _generate_join_qc(merged: "pd.DataFrame", snap: Path, used_prejoin: dict[str, bool]) -> dict:
    """Generate QC report for joined features.
    
    Returns dict with:
    - n_rows: number of rows in joined
    - date_min, date_max: period coverage
    - coverage_activity: % non-null act_steps (if exists)
    - coverage_cardio: % non-null hr_mean (if exists)
    - coverage_sleep: % non-null sleep_total_h (if exists)
    - used_prejoin_activity, used_prejoin_cardio: 1/0 flags
    """
    qc: dict[str, Any] = {
        "n_rows": len(merged),
        "date_min": None,
        "date_max": None,
        "coverage_activity": None,
        "coverage_cardio": None,
        "coverage_sleep": None,
        "used_prejoin_activity": used_prejoin.get("activity", 0),
        "used_prejoin_cardio": used_prejoin.get("cardio", 0),
    }
    
    # Date coverage
    if "date" in merged.columns:
        try:
            dates = pd.to_datetime(merged["date"], errors="coerce")
            qc["date_min"] = str(dates.min().date()) if not pd.isna(dates.min()) else None
            qc["date_max"] = str(dates.max().date()) if not pd.isna(dates.max()) else None
        except Exception:
            pass
    
    # Activity coverage: % non-null act_steps (or coalesced apple_steps/zepp_act_steps)
    act_col = None
    if "act_steps" in merged.columns:
        act_col = "act_steps"
    elif "apple_steps" in merged.columns:
        act_col = "apple_steps"
    elif "zepp_act_steps" in merged.columns:
        act_col = "zepp_act_steps"
    
    if act_col and len(merged) > 0:
        qc["coverage_activity"] = round(100.0 * merged[act_col].notna().sum() / len(merged), 2)
    
    # Cardio coverage: % non-null hr_mean
    hr_col = None
    if "hr_mean" in merged.columns:
        hr_col = "hr_mean"
    elif "apple_hr_mean" in merged.columns:
        hr_col = "apple_hr_mean"
    elif "zepp_hr_mean" in merged.columns:
        hr_col = "zepp_hr_mean"
    
    if hr_col and len(merged) > 0:
        qc["coverage_cardio"] = round(100.0 * merged[hr_col].notna().sum() / len(merged), 2)
    
    # Sleep coverage: % non-null sleep_total_h
    sleep_col = None
    if "sleep_total_h" in merged.columns:
        sleep_col = "sleep_total_h"
    elif "apple_slp_total_h" in merged.columns:
        sleep_col = "apple_slp_total_h"
    elif "zepp_slp_total_h" in merged.columns:
        sleep_col = "zepp_slp_total_h"
    
    if sleep_col and len(merged) > 0:
        qc["coverage_sleep"] = round(100.0 * merged[sleep_col].notna().sum() / len(merged), 2)
    
    return qc


def join_run(snapshot_dir: Path | str, *, dry_run: bool = False, months: int = 30) -> int:
    """Orchestrate a minimal join of per-domain features into canonical joined CSV.

    Current behaviour (P8+):
    - Look for per-domain enriched files under <snapshot_dir>/enriched/prejoin/:
      - enriched/prejoin/<domain>/<vendor>/<variant>/enriched_<domain>.csv
    - If prejudoin enrichment files are not found, fallback to:
      - features/<domain>/<vendor>/<variant>/features_daily.csv
    - If those are not found, fallback to legacy interim joined/ per-domain files:
      - features_cardiovascular.csv, features_activity.csv
    - Each domain may have multiple vendor/variant combinations (e.g., apple/inapp, zepp/cloud)
    - All vendor/variant files for a domain are read and concatenated before joining
    - Aggregate to one row per (date, domain) to ensure daily uniqueness
    - If both cardio and activity are present, concat them and merge
      into the canonical joined features CSV.
    - Use `ensure_joined_snapshot` and `write_joined_features` from
      `lib.io_guards` to resolve and write the canonical joined path.
    - Filter to last `months` months for data quality (default 30 months).

    Args:
        snapshot_dir: Path to snapshot directory
        dry_run: If True, only report what would be done
        months: Number of months to keep (default 30). Set to 0 to disable filtering.

    Returns 0 on success, 2 if no domain features found to merge.
    """
    snap = Path(snapshot_dir)
    print(f"INFO: join_run start snapshot_dir={snap} dry_run={dry_run} months_filter={months}")


    jdir = snap / "joined"

    # Collect ALL vendor/variant combinations for each domain (not just one)
    domains_data: dict[str, list[tuple[Path, str, str]]] = {
        "cardio": [],
        "activity": [],
        "sleep": [],
    }
    
    for d in ["cardio", "activity", "sleep"]:
        # Priority 1: enriched/prejoin
        enriched_dir = snap / "enriched" / "prejoin" / d
        if enriched_dir.exists():
            # Find all enriched_<domain>.csv files recursively
            for enriched_file in enriched_dir.rglob(f"enriched_{d}.csv"):
                domains_data[d].append((enriched_file, "enriched_prejoin"))
        
        # Priority 2: features/<domain>
        if not domains_data[d]:  # Only if no enriched found
            features_dir = snap / "features" / d
            if features_dir.exists():
                for features_file in features_dir.rglob("features_daily.csv"):
                    domains_data[d].append((features_file, "canonical_features"))
        
        # Priority 3: legacy fallback joined/<domain>
        if not domains_data[d]:  # Only if no enriched or features found
            if d == "cardio":
                legacy_file = jdir / "features_cardiovascular.csv"
            else:
                legacy_file = jdir / "features_activity.csv"
            
            if legacy_file.exists():
                domains_data[d].append((legacy_file, "legacy_joined"))
    
    # Flatten: list of (domain, path, source_kind)
    found: list[tuple[str, Path, str]] = []
    for d, files in domains_data.items():
        for fpath, source_kind in files:
            found.append((d, fpath, source_kind))
    
    if not found:
        # No per-domain files discovered
        if dry_run:
            print("[dry-run] nothing to join")
            print("INFO: no per-domain feature files found (cardio/activity) -> nothing to join")
            return 0
        print("INFO: no per-domain feature files found (cardio/activity) -> nothing to join")
        return 2

    print("INFO: discovered domain feature files:")
    for name, p, kind in found:
        try:
            df = pd.read_csv(p, parse_dates=["date"]) if p.exists() else pd.DataFrame()
            print(f"  - {name}: {p.relative_to(snap)} rows={len(df)} (source={kind})")
        except Exception:
            print(f"  - {name}: {p.relative_to(snap)} (could not read) (source={kind})")

    if dry_run:
        out_path = ensure_joined_snapshot(snap)
        print(f"DRY RUN: would write joined output -> {out_path}")
        print("INFO: join_run end (dry-run)")
        return 0

    # read dataframes and group by domain
    domain_dfs: dict[str, list[pd.DataFrame]] = {"cardio": [], "activity": [], "sleep": []}
    for name, p, kind in found:
        try:
            df = pd.read_csv(p, parse_dates=["date"]) if p.exists() else pd.DataFrame()
            if not df.empty:
                domain_dfs[name].append(df)
        except Exception:
            pass

    # For each domain, concat all vendor/variant files and apply coalescence
    parts = []
    used_prejoin: dict[str, bool] = {}  # Track if prejoin was used per domain
    
    for domain_name in ["cardio", "activity", "sleep"]:
        if not domain_dfs[domain_name]:
            continue
        
        # Concat all vendor/variant files for this domain
        domain_df = pd.concat(domain_dfs[domain_name], ignore_index=True, sort=False)
        domain_df = domain_df.copy()
        
        # Track if we used prejoin enrichment for this domain
        # (infer from presence of _7d or _zscore columns)
        used_prejoin[domain_name] = any(c.endswith(("_7d", "_zscore")) for c in domain_df.columns)
        
        # Apply coalescence for duplicate vendor/variant combinations
        if domain_name == "cardio":
            # hr_mean: coalesce(apple_hr_mean, zepp_hr_mean)
            if "apple_hr_mean" in domain_df.columns and "zepp_hr_mean" in domain_df.columns:
                domain_df["hr_mean"] = domain_df["apple_hr_mean"].fillna(domain_df["zepp_hr_mean"])
            
            # hr_std: coalesce(apple_hr_std, zepp_hr_std)
            if "apple_hr_std" in domain_df.columns and "zepp_hr_std" in domain_df.columns:
                domain_df["hr_std"] = domain_df["apple_hr_std"].fillna(domain_df["zepp_hr_std"])
            
            # n_hr: coalesce(apple_n_hr, zepp_n_hr)
            if "apple_n_hr" in domain_df.columns and "zepp_n_hr" in domain_df.columns:
                domain_df["n_hr"] = domain_df["apple_n_hr"].fillna(domain_df["zepp_n_hr"])
        
        elif domain_name == "activity":
            # act_steps: coalesce(apple_steps, zepp_steps)
            if "apple_steps" in domain_df.columns and "zepp_steps" in domain_df.columns:
                domain_df["act_steps"] = domain_df["apple_steps"].fillna(domain_df["zepp_steps"])
            
            # act_active_min: coalesce(apple_exercise_min, zepp_exercise_min)
            if "apple_exercise_min" in domain_df.columns and "zepp_exercise_min" in domain_df.columns:
                domain_df["act_active_min"] = domain_df["apple_exercise_min"].fillna(domain_df["zepp_exercise_min"])

        
        # Add provenance columns
        domain_df["source_domain"] = domain_name
        parts.append(domain_df)

    if not parts:
        print("INFO: no readable domain dataframes to merge")
        return 2

    # concat per-domain rows (provenance preserved)
    concat_df = pd.concat(parts, ignore_index=True, sort=False)
    
    # CRITICAL: Aggregate to one row per (date, domain) to ensure daily uniqueness
    # Multiple intra-day records per domain must be aggregated before merge
    if "date" in concat_df.columns and "source_domain" in concat_df.columns:
        print("INFO: aggregating intra-day records by (date, domain)...")
        concat_df["date"] = pd.to_datetime(concat_df["date"], errors="coerce")
        
        # Identify numeric columns to aggregate by mean
        numeric_cols = concat_df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Remove source_domain from numeric aggregation (keep first instead)
        if "source_domain" in numeric_cols:
            numeric_cols.remove("source_domain")
        
        # Define aggregation spec: mean for numeric, 'first' for non-numeric
        agg_spec = {col: "mean" for col in numeric_cols}
        agg_spec["source_domain"] = "first"  # Preserve domain indicator
        
        # Aggregate by (date, source_domain)
        concat_df = concat_df.groupby(["date", "source_domain"], as_index=False).agg(agg_spec)
        print(f"INFO: after aggregation: {len(concat_df)} records")
    
    # Keep date as datetime64 for now (used in filtering after merge)
    if "date" in concat_df.columns:
        try:
            concat_df["date"] = pd.to_datetime(concat_df["date"])
        except Exception:
            pass

    # Merge into canonical joined features by outer merge on date
    base_path = ensure_joined_snapshot(snap)
    if base_path.exists():
        try:
            base = pd.read_csv(base_path, parse_dates=["date"]) if base_path.exists() else pd.DataFrame()
        except Exception:
            base = pd.DataFrame()
    else:
        base = pd.DataFrame()

    # Ensure date column is datetime64 in both frames before merge
    if "date" in base.columns and not base.empty:
        base["date"] = pd.to_datetime(base["date"], errors="coerce")
    if "date" in concat_df.columns and not concat_df.empty:
        concat_df["date"] = pd.to_datetime(concat_df["date"], errors="coerce")

    # Final aggregation: merge domains to ONE ROW per date by averaging numeric columns
    # This collapses (date, source_domain) grouping into just (date) grouping
    if "date" in concat_df.columns and len(concat_df) > 0:
        numeric_cols = concat_df.select_dtypes(include=['number']).columns.tolist()
        agg_dict = {col: 'mean' for col in numeric_cols}
        concat_df = concat_df.groupby('date', as_index=False).agg(agg_dict)
        print(f"INFO: aggregated by date: {len(concat_df)} unique dates")

    # Merge base and concat_df on date only (concat_df now has one row per date)
    merged = base.merge(concat_df, on="date", how="outer", suffixes=("_x", "_y")) if not base.empty else concat_df.copy()

    # Resolve suffix conflicts: prefer _y (enriched/new) over _x (base/old)
    if not base.empty:
        for col in merged.columns:
            if col.endswith("_x"):
                col_y = col[:-2] + "_y"
                if col_y in merged.columns:
                    merged[col[:-2]] = merged[col_y].fillna(merged[col])
                    merged = merged.drop(columns=[col, col_y], errors="ignore")

    # Apply date window filtering for data quality (last N months) - AFTER merge
    # This filters BOTH base (old data) and concat_df (new data) to the recent window
    if months and months > 0 and "date" in merged.columns:
        merged["date"] = pd.to_datetime(merged["date"], errors="coerce")
        cutoff_date = pd.Timestamp.now() - pd.DateOffset(months=months)
        original_len = len(merged)
        merged = merged[merged["date"] >= cutoff_date]
        print(f"INFO: filtered to last {months} months (since {cutoff_date.date()}): {original_len} -> {len(merged)} records")
        print(f"INFO: unique dates after filtering: {merged['date'].nunique()}")
    
    # Remove any date column duplicates (shouldn't happen, but be safe)
    if len(merged) > 0 and "date" in merged.columns:
        # Keep only one row per date (prefer rows with more non-null values from merged)
        merged = merged.sort_values('date').reset_index(drop=True)
        merged = merged.drop_duplicates(subset=['date'], keep='last')

    # finalize: sort and reset index, keep datetime64 until write
    if "date" in merged.columns:
        try:
            merged = merged.sort_values("date").reset_index(drop=True)
        except Exception:
            pass

    # Generate QC report
    qc_record = _generate_join_qc(merged, snap, used_prejoin)

    # write using canonical helper (this creates backup named joined_features_daily_prev.csv)
    try:
        # Convert datetime64 to date string before write
        if "date" in merged.columns:
            try:
                merged["date"] = pd.to_datetime(merged["date"]).dt.strftime("%Y-%m-%d")
            except Exception:
                pass
        
        write_joined_features(merged, snap, dry_run=False)
        print(f"INFO: wrote joined features -> {ensure_joined_snapshot(snap)}")
    except Exception as e:
        print(f"ERROR: failed to write joined features: {e}")
        return 1

    # Write QC report
    try:
        qc_dir = snap / "qc"
        qc_dir.mkdir(parents=True, exist_ok=True)
        qc_path = qc_dir / "join_qc.csv"
        
        qc_df = pd.DataFrame([qc_record])
        qc_df.to_csv(qc_path, index=False)
        print(f"INFO: wrote QC report -> {qc_path}")
    except Exception as e:
        print(f"WARNING: failed to write QC report: {e}")

    print("INFO: join_run end")
    return 0


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="ETL Apple/Zepp — Nova arquitetura (definitivo)"
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    # extract
    p_ext = sub.add_parser(
        "extract", help="Apple → per-metric (a partir de export.xml)"
    )
    p_ext.add_argument("--participant", required=True)
    p_ext.add_argument(
        "--snapshot", required=False, default="auto", help=HELP_SNAPSHOT_ID
    )
    p_ext.add_argument(
        "--auto-zip",
        action="store_true",
        help="Auto-discover latest Apple/Zepp ZIPs under data/raw/<participant>/",
    )
    p_ext.add_argument(
        "--apple-zip",
        help="Optional override path to Apple ZIP (must be under data/raw/<participant>/)",
    )
    p_ext.add_argument(
        "--zepp-zip",
        help="Optional override path to Zepp ZIP (must be under data/raw/<participant>/)",
    )
    p_ext.add_argument(
        "--debug-paths",
        action="store_true",
        help="Print resolved base_dir and exact glob patterns and files found for apple/zepp (debug)",
    )
    p_ext.add_argument(
        "--zepp-password",
        help="Password for encrypted Zepp ZIP (or set ZEPP_ZIP_PASSWORD env var)",
    )
    p_ext.add_argument(
        "--dry-run",
        action="store_true",
        help="Discovery/validation only; do not extract",
    )

    # cardio
    p_car = sub.add_parser("cardio", help="Executa estágio Cardiovascular no snapshot")
    p_car.add_argument(
        "--zepp_dir", help="Diretório fixo com CSVs processados do Zepp (evita _latest)"
    )
    g = p_car.add_mutually_exclusive_group(required=True)
    g.add_argument("--snapshot_dir", help=HELP_SNAPSHOT_DIR)
    g.add_argument("--participant", help=HELP_PARTICIPANT)
    p_car.add_argument("--snapshot", help=HELP_SNAPSHOT_ID)

    # full
    p_full = sub.add_parser(
        "full", help="extract → cardio (pipeline novo ponta-a-ponta)"
    )
    p_full.add_argument("--participant", required=True)
    p_full.add_argument(
        "--snapshot", required=False, default="auto", help=HELP_SNAPSHOT_ID
    )
    p_full.add_argument(
        "--zepp_dir", help="Diretório fixo com CSVs processados do Zepp (evita _latest)"
    )

    # labels
    p_lab = sub.add_parser("labels", help="Join labels into features_daily_labeled.csv")
    g_lab = p_lab.add_mutually_exclusive_group(required=True)
    g_lab.add_argument("--snapshot_dir", help=HELP_SNAPSHOT_DIR)
    g_lab.add_argument("--participant", help=HELP_PARTICIPANT)
    p_lab.add_argument("--snapshot", help=HELP_SNAPSHOT_ID)
    p_lab.add_argument(
        "--synthetic",
        action="store_true",
        help="Use synthetic labels (state_of_mind_synthetic.csv)",
    )
    p_lab.add_argument(
        "--labels_path", help="Caminho alternativo para labels (opcional)"
    )
    p_lab.add_argument(
        "--label-schema",
        choices=["binary", "ternary", "five"],
        default="ternary",
        help="Label schema to produce",
    )
    p_lab.add_argument(
        "--prefer",
        choices=["apple", "zepp", "mixed"],
        default="mixed",
        help="Preference when both sources present",
    )
    p_lab.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not write outputs; just report summary",
    )

    # join
    p_join = sub.add_parser("join", help="Join per-domain features into canonical joined CSV")
    g_join = p_join.add_mutually_exclusive_group(required=True)
    g_join.add_argument("--snapshot_dir", help=HELP_SNAPSHOT_DIR)
    g_join.add_argument("--participant", help=HELP_PARTICIPANT)
    p_join.add_argument("--snapshot", help=HELP_SNAPSHOT_ID)
    p_join.add_argument(
        "--dry-run",
        action="store_true",
        help="Dry-run: print discovered domain features and output paths; do not write",
    )

    # enrich-prejoin (pre-join enrichment)
    p_prejoin = sub.add_parser("enrich-prejoin", help="Apply per-domain enrichment before join")
    g_prejoin = p_prejoin.add_mutually_exclusive_group(required=True)
    g_prejoin.add_argument("--snapshot_dir", help=HELP_SNAPSHOT_DIR)
    g_prejoin.add_argument("--participant", help=HELP_PARTICIPANT)
    p_prejoin.add_argument("--snapshot", help=HELP_SNAPSHOT_ID)
    p_prejoin.add_argument(
        "--max-records",
        type=int,
        default=None,
        help="Limit records per vendor/variant (for testing)",
    )
    p_prejoin.add_argument(
        "--dry-run",
        action="store_true",
        help="Dry-run: print what would be enriched; do not write",
    )

    # aggregate
    p_agg = sub.add_parser(
        "aggregate",
        help="Aggregate features to one row per day (optional synthetic labels)",
    )
    gagg = p_agg.add_mutually_exclusive_group(required=True)
    gagg.add_argument("--snapshot_dir", help=HELP_SNAPSHOT_DIR)
    gagg.add_argument("--participant", help=HELP_PARTICIPANT)
    p_agg.add_argument("--snapshot", help=HELP_SNAPSHOT_ID)
    p_agg.add_argument(
        "--labels",
        choices=["none", "synthetic"],
        default="none",
        help="Join labels (synthetic) into aggregated output",
    )

    # som-scan (quick debug scan of export_cda.xml for State-of-Mind / mood entries)
    p_som = sub.add_parser(
        "som-scan", help="Quick scan export_cda.xml for State-of-Mind entries (debug)"
    )
    p_som.add_argument("--participant", required=True)
    p_som.add_argument("--snapshot", required=True)
    p_som.add_argument(
        "--trace", action="store_true", help="Print traceback on parse errors"
    )
    p_som.add_argument(
        "--write-normalized",
        dest="write_normalized",
        action="store_true",
        help="Write normalized/apple_state_of_mind.csv when entries found",
    )

    args = ap.parse_args()

    # Prefer the participant string as-provided if a matching raw folder exists.
    # If not, try the canonical PID (P + zero-padded 6 digits) and use it only
    # when that folder exists. This avoids silently changing the participant
    # to an unexpected value while still supporting common shorthand inputs.
    if hasattr(args, "participant") and args.participant:
        prov = args.participant
        try:
            if (RAW_ARCHIVE / prov).exists():
                # keep as-provided
                pass
            else:
                canon = canon_pid(prov)
                if (RAW_ARCHIVE / canon).exists():
                    args.participant = canon
        except Exception:
            # on any error, leave participant as-provided
            pass

    # som-scan: simple utility to run the streaming CDA SoM parser and print results
    if args.cmd == "som-scan":
        # Resolve snapshot directory and CDA path
        part = args.participant
        snap = args.snapshot
        snap_dir = etl_snapshot_root(part, snap)
        cda_path = (
            snap_dir / "extracted" / "apple" / "apple_health_export" / "export_cda.xml"
        )
        if not cda_path.exists():
            print(f"ERROR: CDA file not found at {cda_path}")
            return 1

        print(f"INFO: scanning CDA for SoM → {cda_path}")
        df = pd.DataFrame()
        parse_exc = None

        # 1) Try lxml if available (parsing in UTC)
        try:
            try:
                import lxml.etree as LET  # presence check

                df = _parse_apple_cda_som_lxml(cda_path)
            except ImportError:
                df = pd.DataFrame()
        except Exception as e:
            parse_exc = e
            if args.trace:
                print("ERROR: lxml parse failed:\n", file=sys.stderr)
                traceback.print_exc()

        # 2) Stdlib fallback
        if df.empty:
            try:
                df = _parse_apple_cda_som(cda_path)
            except Exception as e:
                parse_exc = e
                if args.trace:
                    print("ERROR: stdlib parse failed:\n", file=sys.stderr)
                    traceback.print_exc()

        # 3) Text-scan fallback
        if df.empty:
            try:
                df = _parse_cda_textscan(cda_path)
            except Exception as e:
                parse_exc = e
                if args.trace:
                    print("ERROR: text-scan failed:\n", file=sys.stderr)
                    traceback.print_exc()

        if df.empty:
            # if we had a parse exception, return special code 3
            if parse_exc is not None:
                print("WARNING: CDA parse failed and text-scan found nothing.")
                return 3
            else:
                print("WARNING: no SoM entries found.")
                return 0

        # If we reached here, we have rows
        print(f"Found {len(df)} entries. Head:")
        try:
            print(df.head(10).to_string(index=False))
        except Exception:
            print(df.head(10))

        # Apply SoM content filter before writing
        df_som_f = _filter_som_candidates(df)

        if getattr(args, "write_normalized", False):
            snap_dir = snap_dir if "snap_dir" in locals() else (cda_path.parents[4])
            out_som = snap_dir / "normalized" / "apple_state_of_mind.csv"
            out_som.parent.mkdir(parents=True, exist_ok=True)
            if df_som_f.empty:
                print(
                    "INFO: SoM candidates found but none passed filters; skipping write."
                )
            else:
                try:
                    written = _write_if_changed_df(df_som_f, out_som)
                    if written:
                        print(f"INFO: wrote apple SoM -> {out_som}")
                    else:
                        print(f"[SKIP] apple SoM unchanged -> {out_som}")
                except Exception as e:
                    print(f"WARNING: failed to write normalized apple SoM: {e}")

        return 0

    if args.cmd == "extract":
        pid, snap = args.participant, args.snapshot
        # TEMP DEBUG: print incoming args and environment to help troubleshoot discovery
        try:
            print("DEBUG-TEMP: extract called with participant=", pid, "snapshot=", snap)
            print("DEBUG-TEMP: flags: auto_zip=", getattr(args, 'auto_zip', False), "apple_zip=", getattr(args, 'apple_zip', None), "zepp_zip=", getattr(args, 'zepp_zip', None), "dry_run=", getattr(args, 'dry_run', False))
            print("DEBUG-TEMP: PWD=", os.getcwd())
            print("DEBUG-TEMP: RAW_ARCHIVE=", RAW_ARCHIVE)
            print("DEBUG-TEMP: ETL_ROOT=", ETL_ROOT)
        except Exception:
            pass
        # If auto-zip discovery enabled, search under data/raw/<participant> only
        if args.auto_zip or args.apple_zip or args.zepp_zip:
            base_dir = RAW_ARCHIVE / pid
            if not base_dir.exists():
                print(
                    f"ERROR: No raw folder found for participant {pid} under data/raw/"
                )
                return 1

            apple_candidate = None
            zepp_candidate = None

            # Resolve overrides first (validate scope)
            if args.apple_zip:
                ac = Path(args.apple_zip)
                if not ac.exists():
                    print(f"ERROR: Provided --apple-zip does not exist: {ac}")
                    return 1
                _ensure_within_base(ac, base_dir)
                apple_candidate = ac

            if args.zepp_zip:
                zc = Path(args.zepp_zip)
                if not zc.exists():
                    print(f"ERROR: Provided --zepp-zip does not exist: {zc}")
                    return 1
                _ensure_within_base(zc, base_dir)
                zepp_candidate = zc

            # Log which participant folder is being used (helps debugging accidental PID normalization)
            try:
                print(
                    f"INFO: using participant '{pid}' -> raw path: {RAW_ARCHIVE / pid}"
                )
            except Exception:
                pass

            # Debug: print resolved base_dir, the exact glob patterns and files found
            if getattr(args, "debug_paths", False):
                try:
                    resolved = (base_dir).resolve(strict=False)
                    print(f"DEBUG: base_dir.resolve() = {resolved}")
                    apple_pattern = base_dir / "apple" / "**" / "*.zip"
                    zepp_pattern = base_dir / "zepp" / "**" / "*.zip"
                    print(f"DEBUG: apple glob pattern: {apple_pattern}")
                    print(f"DEBUG: zepp glob pattern: {zepp_pattern}")

                    apple_files = (
                        list((base_dir / "apple").rglob("*.zip"))
                        if (base_dir / "apple").exists()
                        else []
                    )
                    zepp_files = (
                        list((base_dir / "zepp").rglob("*.zip"))
                        if (base_dir / "zepp").exists()
                        else []
                    )

                    print(f"DEBUG: apple files found: {len(apple_files)}")
                    for p in sorted(apple_files):
                        try:
                            print(" -", p.resolve())
                        except Exception:
                            print(" -", p)

                    print(f"DEBUG: zepp files found: {len(zepp_files)}")
                    for p in sorted(zepp_files):
                        try:
                            print(" -", p.resolve())
                        except Exception:
                            print(" -", p)

                    if not apple_files and not zepp_files:
                        print(
                            f"ERROR: No zip files under {resolved}. Check your working directory (PWD={os.getcwd()}) or PARTICIPANT."
                        )
                        return 1
                except Exception as e:
                    print(f"DEBUG: failed to enumerate paths: {e}")

            # Auto-discover if not overridden
            if not apple_candidate and args.auto_zip:
                apple_search = base_dir / "apple"
                apple_candidate = find_latest_zip(apple_search)

            if not zepp_candidate and args.auto_zip:
                zepp_search = base_dir / "zepp"
                zepp_candidate = find_latest_zip(zepp_search)

            if not apple_candidate and not zepp_candidate:
                print(f"ERROR: No Apple or Zepp ZIPs found under {base_dir}")
                return 1

            # Validate contents
            if apple_candidate:
                print(
                    f"INFO: Apple candidate: {apple_candidate} size={apple_candidate.stat().st_size} mtime={apple_candidate.stat().st_mtime}"
                )
                _ensure_within_base(apple_candidate, base_dir)
                if not zip_contains(
                    apple_candidate, ["export.xml", "apple_health_export"]
                ):
                    print(
                        f"ERROR: export.xml not found inside {apple_candidate} (Apple Health export expected)."
                    )
                    return 1

            if zepp_candidate:
                print(
                    f"INFO: Zepp candidate: {zepp_candidate} size={zepp_candidate.stat().st_size} mtime={zepp_candidate.stat().st_mtime}"
                )
                _ensure_within_base(zepp_candidate, base_dir)
                if not zip_contains(
                    zepp_candidate, ["hr", "hrv", "sleep", "emotion", "temperature"]
                ):
                    print(
                        f"ERROR: No known Zepp patterns found inside {zepp_candidate} (expected hr/hrv/sleep/emotion/temperature)."
                    )
                    return 1

            if args.dry_run:
                print("DRY RUN: planned actions:")
                if apple_candidate:
                    print(
                        f" - apple: would extract {apple_candidate} -> {etl_snapshot_root(pid, canon_snap_id(snap)) / 'extracted' / 'apple'}"
                    )
                if zepp_candidate:
                    print(
                        f" - zepp: would extract {zepp_candidate} -> {etl_snapshot_root(pid, canon_snap_id(snap)) / 'extracted' / 'zepp'}"
                    )
                return 0

            # Proceed to extract with sha checks and atomic replace
            results = {}
            snap_iso = canon_snap_id(snap)
            for device, cand in (("apple", apple_candidate), ("zepp", zepp_candidate)):
                if not cand:
                    continue
                target_dir = etl_snapshot_root(pid, snap_iso) / "extracted" / device
                sha_file = target_dir / f"{device}_zip.sha256"
                cur_hash = sha256_path(cand)
                prev_hash = read_sha256(sha_file)
                if prev_hash and prev_hash == cur_hash:
                    print(f"[SKIP] skipped (hash match): {device} -> {target_dir}")
                    results[device] = str(target_dir)
                    continue

                print(f"INFO: extracting {device} -> {target_dir}")
                tmp_parent = target_dir.parent
                tmpdir = tmp_parent / (
                    ".tmp_extract_"
                    + device
                    + "_"
                    + datetime.now().strftime("%Y%m%d%H%M%S")
                )
                try:
                    if tmpdir.exists():
                        shutil.rmtree(tmpdir)
                    tmpdir.mkdir(parents=True, exist_ok=True)
                    # detect encrypted members
                    encrypted = False
                    try:
                        with zipfile.ZipFile(cand, "r") as zf:
                            infolist = zf.infolist()
                            encrypted = any(info.flag_bits & 0x1 for info in infolist)
                    except Exception:
                        # Could not open with stdlib zip; attempt pyzipper later
                        encrypted = True

                    # If not encrypted, use stdlib extractall for speed
                    if not encrypted:
                        with zipfile.ZipFile(cand, "r") as zf:
                            zf.extractall(tmpdir)
                    else:
                        # Prefer pyzipper for AES-encrypted archives
                        pwd = args.zepp_password or os.environ.get("ZEPP_ZIP_PASSWORD")
                        if not pwd:
                            print(
                                f"ERROR: Archive {cand} contains encrypted entries; provide --zepp-password or set ZEPP_ZIP_PASSWORD to extract."
                            )
                            return 1
                        # try stdlib first (may work for legacy encryption)
                        tried = False
                        try:
                            with zipfile.ZipFile(cand, "r") as zf:
                                zf.extractall(tmpdir, pwd=pwd.encode("utf-8"))
                                tried = True
                        except RuntimeError:
                            tried = False
                        except Exception:
                            tried = False

                        if not tried:
                            try:
                                import pyzipper

                                with pyzipper.AESZipFile(cand, "r") as az:
                                    az.pwd = pwd.encode("utf-8")
                                    az.extractall(path=str(tmpdir))
                            except ModuleNotFoundError:
                                print(
                                    "ERROR: AES-encrypted archive requires pyzipper (pip install pyzipper) or use a non-encrypted zip.",
                                    file=sys.stderr,
                                )
                                return 1
                            except RuntimeError as re:
                                print(
                                    f"ERROR: provided password seems invalid for {cand}: {re}"
                                )
                                return 1

                    # atomic replace
                    if target_dir.exists():
                        shutil.rmtree(target_dir)
                    shutil.move(str(tmpdir), str(target_dir))
                    write_sha256(sha_file, cur_hash)
                    print(f"[OK] extracted: {device} -> {target_dir}")
                    results[device] = str(target_dir)
                except Exception as e:
                    # cleanup
                    if tmpdir.exists():
                        try:
                            shutil.rmtree(tmpdir)
                        except Exception:
                            pass
                    print(f"ERROR extracting {cand}: {e}")
                    return 1

            # If apple was extracted and contains export.xml, run existing XML parser to produce per-metric outputs
            if "apple" in results:
                extracted_apple_dir = Path(results["apple"])
                xmls = list(extracted_apple_dir.rglob("export.xml"))
                if xmls:
                    xml_path = xmls[0]
                    outdir = etl_snapshot_root(pid, snap_iso)
                    # ensure outdir exists
                    outdir.mkdir(parents=True, exist_ok=True)
                    with Timer(f"extract (parse export.xml) [{pid}/{snap}]"):
                        try:
                            written = extract_apple_per_metric(
                                xml_file=xml_path,
                                outdir=outdir,
                            )
                        except Exception as e:
                            print(f"ERROR parsing export.xml: {e}")
                            return 1
                    print("[OK] extract (apple parse) concluído")
                    for k, v in written.items():
                        print(f" - {k}: {v}")
                    # Attempt to parse Zepp HEALTH_DATA for emotion/stress (non-fatal)
                    try:
                        if "zepp" in results:
                            zepp_dir = Path(results["zepp"])
                            health_files = list(
                                zepp_dir.rglob("HEALTH_DATA_*.csv")
                            ) + list(zepp_dir.rglob("HEALTH_DATA/**/*.csv"))
                            # dedupe
                            health_files = list(
                                {str(p): p for p in health_files}.values()
                            )
                            if health_files:
                                # Parse Zepp health data in UTC
                                zepp_df = _parse_zepp_health_emotion(
                                    health_files
                                )
                                norm_dir = outdir / "normalized" / "zepp"
                                norm_dir.mkdir(parents=True, exist_ok=True)
                                out_path = norm_dir / "zepp_emotion.csv"
                                if not zepp_df.empty:
                                    written_flag = _write_if_changed_df(
                                        zepp_df, out_path
                                    )
                                    if written_flag:
                                        print(f"INFO: wrote zepp emotion -> {out_path}")
                                    else:
                                        print(
                                            f"[SKIP] zepp emotion unchanged -> {out_path}"
                                        )
                                else:
                                    print(
                                        "WARNING: no zepp HEALTH_DATA emotion/stress observations found; skipping zepp_emotion.csv"
                                    )
                    except Exception as _e:
                        print(f"WARNING: zepp emotion parsing failed (non-fatal): {_e}")
                    # Attempt to parse Apple AutoExport StateOfMind CSVs (non-fatal)
                    try:
                        # prefer raw autoexport under data/raw/<pid>/apple/autoexport
                        raw_base = RAW_ARCHIVE / pid
                        auto_dir = raw_base / "apple" / "autoexport"
                        if auto_dir.exists():
                            auto_files = sorted(
                                list(auto_dir.glob("StateOfMind*.csv")),
                                key=lambda f: f.stat().st_mtime,
                                reverse=True,
                            )
                        else:
                            auto_files = []
                        if auto_files:
                            latest = auto_files[0]
                            print(f"INFO: found Apple AutoExport SoM -> {latest.name}")
                            y, m, d = map(int, args.cutover.split("-"))
                            tz_sel = make_tz_selector(
                                date(y, m, d), args.tz_before, args.tz_after
                            )
                            auto_df = _parse_apple_autoexport_som(latest, tz_sel)
                            norm_dir_auto = outdir / "normalized"
                            norm_dir_auto.mkdir(parents=True, exist_ok=True)
                            out_path_auto = (
                                norm_dir_auto / "apple_state_of_mind_autoexport.csv"
                            )
                            if not auto_df.empty:
                                written_flag = _write_if_changed_df(
                                    auto_df, out_path_auto
                                )
                                if written_flag:
                                    print(
                                        f"[OK] wrote apple autoexport SoM -> {out_path_auto}"
                                    )
                                else:
                                    print(
                                        f"[SKIP] apple autoexport unchanged -> {out_path_auto}"
                                    )
                            else:
                                print(
                                    "WARNING: Apple AutoExport SoM file empty or unparseable."
                                )
                        else:
                            print("INFO: no Apple AutoExport SoM file found.")
                    except Exception as _e:
                        print(
                            f"WARNING: apple autoexport parsing failed (non-fatal): {_e}"
                        )
                    # Attempt to parse export_cda.xml for State of Mind (non-fatal)
                    try:
                        cda_paths = list(extracted_apple_dir.rglob("export_cda.xml"))
                        if cda_paths:
                            cda = cda_paths[0]
                            # determine home_tz from participant profile if present
                            home_tz = None
                            try:
                                prof_path = Path("data") / "config" / f"{pid}_profile.json"
                                if prof_path.exists():
                                    prof = json.loads(prof_path.read_text(encoding="utf-8"))
                                    home_tz = prof.get("home_tz")
                            except Exception:
                                home_tz = None

                            # call the CDA probe in a non-fatal way and write QC
                            import logging

                            logger = logging.getLogger("etl.extract")
                            try:
                                from domains.cda import parse_cda

                                summary = parse_cda.cda_probe(cda, home_tz=home_tz)
                                n_obs = int(summary.get("n_observation", 0) or 0)
                                snapshot_root = etl_snapshot_root(pid, snap_iso)
                                # enrich summary with metadata
                                enriched = dict(summary or {})
                                enriched["snapshot_dir"] = str(snapshot_root)
                                enriched["source_path"] = str(cda.resolve())
                                enriched["home_tz"] = home_tz or "unknown"
                                enriched["n_observation"] = n_obs
                                enriched["n_section"] = int(enriched.get("n_section", 0) or 0)
                                enriched["codes"] = enriched.get("codes", {}) or {}
                                if n_obs > 0:
                                    enriched["status"] = "ok"
                                    logger.info(
                                        "CDA parsed: %s (obs=%d) [snapshot=%s]",
                                        str(cda.resolve()),
                                        n_obs,
                                        str(snapshot_root),
                                    )
                                else:
                                    enriched["status"] = "empty"
                                    logger.warning(
                                        "CDA parsed but empty: %s [snapshot=%s]",
                                        str(cda.resolve()),
                                        str(snapshot_root),
                                    )
                                parse_cda.write_cda_qc(snapshot_root, enriched)
                            except Exception as e:
                                # write minimal QC summary on failure
                                try:
                                    from domains.cda import parse_cda

                                    snapshot_root = etl_snapshot_root(pid, snap_iso)
                                    summary = {"error": f"{type(e).__name__}: {e}", "n_section": 0, "n_observation": 0, "codes": {}}
                                    try:
                                        parse_cda.write_cda_qc(snapshot_root, summary)
                                    except Exception:
                                        pass
                                except Exception:
                                    pass
                                logger.warning(f"CDA parse failed: {cda.name} ({type(e).__name__}: {e})")
                    except Exception as _e:
                        import logging

                        logging.getLogger("etl.extract").warning(f"apple SoM parsing failed (non-fatal): {_e}")

            return 0
    if args.cmd == "cardio":
        if args.snapshot_dir:
            snapdir = Path(args.snapshot_dir)
        else:
            if not args.participant or not args.snapshot:
                print(USAGE_SNAPSHOT_OR_PARTICIPANT)
                return 2
            snapdir = ensure_ai_outdir(args.participant, args.snapshot)
        if not snapdir.exists():
            print(SNAPSHOT_NOT_EXISTS, snapdir)
            return 1

        if args.zepp_dir:
            os.environ["ZEPPOVERRIDE_DIR"] = args.zepp_dir

        with Timer(f"cardio [{snapdir}]"):
            res = run_stage_cardio(snapdir)

        # manifest do cardio (inputs e outputs com hashes)
        inputs = {}
        for pm in [
            "apple_heart_rate.csv",
            "apple_hrv_sdnn.csv",
            "apple_sleep_intervals.csv",
        ]:
            pth = snapdir / "per-metric" / pm
            if pth.exists():
                try:
                    inputs[pm] = {"path": str(pth), "sha256": _sha256_file(pth)}
                except Exception:
                    inputs[pm] = {"path": str(pth), "sha256": None}
        for name in ["version_log_enriched.csv", "version_raw.csv"]:
            pth = snapdir / name
            if pth.exists():
                try:
                    inputs[name] = {"path": str(pth), "sha256": _sha256_file(pth)}
                except Exception:
                    inputs[name] = {"path": str(pth), "sha256": None}
        zdir = os.environ.get("ZEPPOVERRIDE_DIR", "")
        zepp = {}
        for z in (glob.glob(os.path.join(zdir, "*.csv")) if zdir else []):
            try:
                zepp[z] = {"path": z, "sha256": _sha256_file(z)}
            except Exception:
                zepp[z] = {"path": z, "sha256": None}

        outputs = {}
        for k, v in res.items():
            try:
                outputs[k] = {"path": v, "sha256": _sha256_file(v)}
            except Exception:
                outputs[k] = {"path": v, "sha256": None}

        manifest = {
            "type": "cardio",
            "snapshot_dir": str(snapdir),
            "zepp_dir": zdir or None,
            "inputs": {"per_metric": inputs, "zepp": zepp},
            "outputs": outputs,
        }
        _write_atomic_json(manifest, snapdir / "cardio_manifest.json")

        print("[OK] cardio concluído")
        print(res)
        return 0

    if args.cmd == "labels":
        # resolve snapshot dir
        if args.snapshot_dir:
            snapdir = Path(args.snapshot_dir)
        else:
            if not args.participant or not args.snapshot:
                print(USAGE_SNAPSHOT_OR_PARTICIPANT)
                return 2
            snapdir = _resolve_snapshot_dir(args.participant, args.snapshot)
        if not snapdir.exists():
            print(SNAPSHOT_NOT_EXISTS, snapdir)
            return 1

        labels_path = Path(args.labels_path) if args.labels_path else None
        # If user explicitly requested synthetic labels, run existing synthetic flow
        if args.synthetic:
            try:
                res = run_stage_labels_synthetic(snapdir, labels_path)
            except FileNotFoundError as e:
                print("WARNING:", e)
                return 1
            except Exception as e:
                print("ERROR: failed to run labels:", e)
                return 1

            print("[OK] labels (synthetic) concluído:", res)
            return 0

        # Otherwise run the full labels merger (Apple SoM + Zepp Emotion)
        # Prefer the ETL snapshot path (data/etl/...) as the canonical SNAP; pass it to the labels runner.
        etl_snapdir = (
            etl_snapshot_root(args.participant, canon_snap_id(args.snapshot))
            if args.participant and args.snapshot
            else snapdir
        )
        try:
            res = run_stage_labels(
                etl_snapdir,
                label_schema=args.label_schema,
                prefer=args.prefer,
                dry_run=args.dry_run,
            )
        except FileNotFoundError as e:
            print("ERROR:", e)
            return 1
        except Exception as e:
            print("ERROR: failed to run labels:", e)
            return 1

        print("[OK] labels concluído:", res)
        return 0

    if args.cmd == "aggregate":
        # resolve snapshot dir
        if args.snapshot_dir:
            snapdir = Path(args.snapshot_dir)
        else:
            if not args.participant or not args.snapshot:
                print(USAGE_SNAPSHOT_OR_PARTICIPANT)
                return 2
            snapdir = AI_ROOT / args.participant / "snapshots" / args.snapshot
        if not snapdir.exists():
            print(SNAPSHOT_NOT_EXISTS, snapdir)
            return 1
        # Call the aggregator via its public API. Import error or runtime will be shown to the user.
        try:
            from etl_tools.aggregate_features_daily import run as run_aggregate

            run_aggregate(snapdir, labels=args.labels)
        except Exception as e:
            print("ERROR: failed to run aggregate via module:", e)
            return 1

        print("[OK] aggregate concluído")
        return 0

    if args.cmd == "full":
        pid = args.participant
        snap = args.snapshot
        resolved_snap = snap
        xml = None
        # If snapshot is auto (or not provided), discover latest Apple ZIP and derive snapshot
        if not snap or snap == "auto":
            try:
                zip_p = find_latest_health_zip(pid)
            except FileNotFoundError as e:
                print(
                    f"ERROR: could not find latest Apple ZIP for participant {pid}: {e}"
                )
                return 1
            resolved_snap = snapshot_from_zip_file_date(zip_p)
            # Ensure extracted (idempotent)
            try:
                xml = ensure_extracted(pid, resolved_snap, zip_p)
            except Exception as e:
                print("ERROR: failed to extract zip:", e)
                return 1
        else:
            # explicit snapshot provided: look for export.xml under raw or fallback into data/etl extracted
            xml = find_export_xml(pid, snap)
            # Fallback: some historic runs placed the extracted Apple export under
            # data/etl/<pid>/snapshots/<snap>/extracted/apple. Prefer the canonical
            # etl_snapshot_root (data/etl/<pid>/<snap>/...) but still tolerate the
            # old layout via migrate_from_data_ai_if_present when appropriate.
            if xml is None:
                alt_path = etl_snapshot_root(pid, canon_snap_id(snap)) / "extracted" / "apple"
                xml_candidates = list(alt_path.rglob("export.xml")) if alt_path.exists() else []
                xml = xml_candidates[0] if xml_candidates else None
            if xml is None:
                print("WARNING: export.xml não encontrado para:", pid, snap)
                return 1
        # Use canonical data/etl snapshot root for outputs (avoid data_ai writes)
        outdir = etl_snapshot_root(pid, canon_snap_id(resolved_snap))
        with Timer(f"extract [{pid}/{resolved_snap}]"):
            written = extract_apple_per_metric(
                xml_file=xml,
                outdir=outdir,
            )
        print("[OK] extract concluído")
        for k, v in written.items():
            print(f" - {k}: {v}")
        if args.zepp_dir:
            os.environ["ZEPPOVERRIDE_DIR"] = args.zepp_dir
        with Timer(f"cardio [{outdir}]"):
            res = run_stage_cardio(outdir)
        print("[OK] cardio concluído")
        print(res)
        return 0

    if args.cmd == "join":
        # determine snapshot_dir
        if getattr(args, "snapshot_dir", None):
            snap_dir = Path(args.snapshot_dir)
        else:
            pid = args.participant
            snap = args.snapshot or "auto"
            snap = canon_snap_id(snap) if snap != "auto" else snap
            if snap == "auto":
                snap = resolve_snapshot(snap, pid)
            snap_dir = etl_snapshot_root(pid, snap)

        # Use last 30 months for data quality (can override via --months arg if added)
        months_filter = getattr(args, "months", 30)
        rc = join_run(snap_dir, dry_run=bool(getattr(args, "dry_run", False)), months=months_filter)
        return int(rc)

    if args.cmd == "enrich-prejoin":
        # determine snapshot_dir
        if getattr(args, "snapshot_dir", None):
            snap_dir = Path(args.snapshot_dir)
        else:
            pid = args.participant
            snap = args.snapshot or "auto"
            snap = canon_snap_id(snap) if snap != "auto" else snap
            if snap == "auto":
                snap = resolve_snapshot(snap, pid)
            snap_dir = etl_snapshot_root(pid, snap)

        max_records = getattr(args, "max_records", None)
        dry_run = bool(getattr(args, "dry_run", False))

        rc = enrich_prejoin_run(snap_dir, dry_run=dry_run, max_records=max_records)
        return int(rc)

    return 0


if __name__ == "__main__":
    sys.exit(main())

