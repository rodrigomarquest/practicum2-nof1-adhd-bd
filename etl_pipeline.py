# -*- coding: utf-8 -*-
from __future__ import annotations

# stdlib
import argparse
import os
import sys
import re
import json
import glob
import io
import tempfile
import hashlib
from datetime import datetime, date
from pathlib import Path
from typing import Optional, Iterable, Tuple, Any, Dict, Union
import xml.etree.ElementTree as ET

# third-party
import numpy as np
import pandas as pd
import subprocess

# local
from etl_modules.common.progress import Timer, progress_open

# ----------------------------------------------------------------------
# Atomic write helpers
# ----------------------------------------------------------------------
def _write_atomic_csv(df: 'pd.DataFrame', out_path: str | os.PathLike[str]):
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
AI_ROOT  = Path("data_ai")

# common filenames used in paths/manifests
EXPORT_XML = "export.xml"
VERSION_RAW = "version_raw.csv"
VERSION_LOG = "version_log_enriched.csv"

# common help strings
HELP_SNAPSHOT_DIR = "Caminho data_ai/<PID>/snapshots/<YYYY-MM-DD>"
HELP_PARTICIPANT = "PID (usar com --snapshot)"
HELP_SNAPSHOT_ID = "Snapshot id (YYYY-MM-DD ou YYYYMMDD)"
USAGE_SNAPSHOT_OR_PARTICIPANT = "Use --snapshot_dir OU (--participant e --snapshot)."
SNAPSHOT_NOT_EXISTS = "⚠️ snapshot_dir não existe:"

_SNAP_ISO  = re.compile(r"^20\d{2}-[01]\d-[0-3]\d$")
_SNAP_8DIG = re.compile(r"^20\d{6}$")  # YYYYMMDD

def canon_snap_id(name: str) -> str:
    name = name.strip()
    if _SNAP_ISO.match(name):
        return name
    if _SNAP_8DIG.match(name):
        return f"{name[:4]}-{name[4:6]}-{name[6:8]}"
    raise ValueError(f"Invalid snapshot '{name}'. Expected YYYY-MM-DD or YYYYMMDD.")

def ensure_ai_outdir(pid: str, snap: str) -> Path:
    snap_iso = canon_snap_id(snap)
    outdir = AI_ROOT / pid / "snapshots" / snap_iso
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
    if 'pytz' in sys.modules or 'pytz' in globals():
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
        s = s[:m.start()] + f"{m.group(1)}:{m.group(2)}"
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

def make_tz_selector(cutover: date, tz_before: str, tz_after: str):
    tz_b = get_tz(tz_before)
    tz_a = get_tz(tz_after)
    def selector(dt_aware: datetime):
        day_utc = dt_aware.astimezone(get_tz("UTC")).date()
        return tz_a if day_utc >= cutover else tz_b
    return selector

def iter_health_records(
    xml_source: Union[Path, io.BufferedReader], tz_selector
) -> Iterable[
    Tuple[str, Any, datetime, Optional[datetime], Optional[str], Dict[str, str], str, str, str]
]:
    """
    xml_source: Path OU file-like binário (ex.: ProgressFile)
    """
    src = xml_source if hasattr(xml_source, "read") else str(xml_source)
    context = ET.iterparse(src, events=("end",))
    for _, elem in context:
        if elem.tag != "Record":
            continue
        type_id = elem.attrib.get("type")
        if type_id not in TYPE_MAP:
            elem.clear(); continue

        s = elem.attrib.get("startDate")
        e = elem.attrib.get("endDate")
        if not s:
            elem.clear(); continue

        sdt = parse_dt(s)
        edt = parse_dt(e) if e else None

        tz = tz_selector(sdt)
        s_local = sdt.astimezone(tz)
        e_local = edt.astimezone(tz) if edt else None

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
        tz_name = getattr(tz, 'key', getattr(tz, 'zone', str(tz)))
        source_version = elem.attrib.get("sourceVersion", "")
        source_name    = elem.attrib.get("sourceName", "")

        yield type_id, value, s_local, e_local, unit, device, tz_name, source_version, source_name
        elem.clear()

# ----------------------------------------------------------------------
# Segments
# ----------------------------------------------------------------------
def build_segments_from_versions(vdf: pd.DataFrame) -> pd.DataFrame:
    if vdf.empty:
        return pd.DataFrame(columns=["segment_id","start","end","ios_version","watch_model","watch_fw","tz_name","source_name"])
    df = vdf.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")
    keys = ["ios_version","watch_model","watch_fw","tz_name","source_name"]
    sig = df[keys].astype(str).agg(" | ".join, axis=1)
    change = sig.ne(sig.shift(fill_value="__START__"))
    segments = []
    seg_id = 0
    seg_start = df["date"].iloc[0]
    last_vals = df.loc[df.index[0], keys].to_dict()
    for i in range(1, len(df)):
        if change.iloc[i]:
            seg_id += 1
            segments.append({
                "segment_id": seg_id,
                "start": seg_start.date(),
                "end": df["date"].iloc[i-1].date(),
                **last_vals
            })
            seg_start = df["date"].iloc[i]
            last_vals = df.loc[df.index[i], keys].to_dict()
    seg_id += 1
    segments.append({
        "segment_id": seg_id,
        "start": seg_start.date(),
        "end": df["date"].iloc[-1].date(),
        **last_vals
    })
    return pd.DataFrame(segments)

# ----------------------------------------------------------------------
# Apple → per-metric (extract)
# ----------------------------------------------------------------------
def extract_apple_per_metric(xml_file: Path, outdir: Path, cutover_str: str, tz_before: str, tz_after: str) -> Dict[str,str]:
    out_pm = outdir / "per-metric"
    out_pm.mkdir(parents=True, exist_ok=True)

    y, m, d = map(int, cutover_str.split("-"))
    tz_selector = make_tz_selector(date(y, m, d), tz_before, tz_after)

    hr_rows, hrv_rows, sleep_rows, version_rows = [], [], [], []
    HR_ID   = "HKQuantityTypeIdentifierHeartRate"
    HRV_ID  = "HKQuantityTypeIdentifierHeartRateVariabilitySDNN"
    SLEEP_ID= "HKCategoryTypeIdentifierSleepAnalysis"

    with Timer("extract: parse export.xml"):
        with progress_open(xml_file, desc="Parsing export.xml") as pf:
            records = iter_health_records(pf, tz_selector)
            for (type_id, value, s_local, e_local, unit, device, tz_name, src_ver, src_name) in records:
                day = s_local.date()
                ios_ver = device.get("ios_version", "") or src_ver
                if ios_ver or device:
                    version_rows.append({
                        "date": day,
                        "ios_version": ios_ver,
                        "watch_model": device.get("watch_model", ""),
                        "watch_fw": device.get("watch_fw", ""),
                        "tz_name": tz_name,
                        "source_name": src_name
                    })
                if type_id == HR_ID and value is not None:
                    hr_rows.append({"timestamp": s_local.isoformat(), "bpm": float(value)})
                elif type_id == HRV_ID and value is not None:
                    hrv_rows.append({"timestamp": s_local.isoformat(), "sdnn_ms": float(value)})
                elif type_id == SLEEP_ID and e_local is not None:
                    sleep_rows.append({"start": s_local.isoformat(), "end": e_local.isoformat(), "raw_value": value})

    written: Dict[str,str] = {}

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
    vdf = pd.DataFrame(version_rows).drop_duplicates().sort_values("date") if version_rows else pd.DataFrame()
    if not vdf.empty:
        _write_atomic_csv(vdf, outdir / VERSION_RAW)
        seg_df = build_segments_from_versions(vdf)
        _write_atomic_csv(seg_df, outdir / VERSION_LOG)
        written["version_raw"] = str(outdir / VERSION_RAW)
        written["version_log_enriched"] = str(outdir / VERSION_LOG)
    else:
        _write_atomic_csv(pd.DataFrame(columns=["date","ios_version","watch_model","watch_fw","tz_name","source_name"]), outdir / VERSION_RAW)
        _write_atomic_csv(pd.DataFrame(columns=["segment_id","start","end","ios_version","watch_model","watch_fw","tz_name","source_name"]), outdir / VERSION_LOG)

    # manifest
    manifest = {
        "type": "extract",
        "export_xml": str(xml_file),
        "params": {"cutover": cutover_str, "tz_before": tz_before, "tz_after": tz_after},
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
        "params": {"cutover": cutover_str, "tz_before": tz_before, "tz_after": tz_after},
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
# Cardiovascular stage
# ----------------------------------------------------------------------
def run_stage_cardio(snapshot_dir: Path) -> Dict[str,str]:
    from etl_modules.config import CardioCfg
    from etl_modules.cardiovascular.cardio_etl import run_stage_cardio as _run
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


def run_stage_labels_synthetic(snapshot_dir: Path, synthetic_path: Path | None = None) -> Dict[str, str]:
    """
    Merge synthetic labels (state_of_mind_synthetic.csv) into features_daily_updated.csv

    Returns dict with output path and manifest path.
    """
    features_path = snapshot_dir / "features_daily_updated.csv"
    if not features_path.exists():
        raise FileNotFoundError(f"features_daily_updated.csv not found in snapshot: {features_path}")

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

    label_col = _find(cols_l, ["raw_label"]) or _find(cols_l, ["label"]) or _find(cols_l, ["state"])
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
    sdf = sdf.rename(columns={cols_l.get('date'): 'date'} if 'date' in cols_l else {})
    sdf['date'] = pd.to_datetime(sdf['date'], errors='coerce').dt.date

    # ensure label column exists
    if label_col:
        sdf = sdf.rename(columns={label_col: 'label'})
    else:
        raise ValueError(f"No label-like column found in synthetic labels: {synth_path}")

    if raw_label_col:
        sdf = sdf.rename(columns={raw_label_col: 'raw_label'})
    else:
        # create raw_label same as label
        sdf['raw_label'] = sdf['label']

    if score_col:
        sdf = sdf.rename(columns={score_col: 'score'})
    else:
        if 'score' not in sdf.columns:
            sdf['score'] = pd.NA

    # aggregate synthetic by date in case of duplicates: mean score, mode label
    def _mode_series(s):
        try:
            m = s.mode(dropna=True)
            return m.iloc[0] if not m.empty else s.iloc[0]
        except Exception:
            return s.iloc[0]

    aggs = {}
    if 'score' in sdf.columns:
        aggs['score'] = 'mean'
    aggs['label'] = lambda x: _mode_series(x)
    aggs['raw_label'] = lambda x: _mode_series(x)

    sdf_grouped = sdf.groupby('date', dropna=False).agg(aggs).reset_index()

    # merge
    out = fdf.merge(sdf_grouped, on='date', how='left')

    # write outputs
    out_csv = snapshot_dir / 'features_daily_labeled.csv'
    _write_atomic_csv(out, out_csv)

    # manifest
    counts = {}
    try:
        counts = dict(pd.Series(out['label'].dropna()).value_counts().to_dict())
    except Exception:
        counts = {}

    manifest = {
        "type": "labels_synthetic",
        "snapshot_dir": str(snapshot_dir),
        "inputs": {
            "features_daily_updated.csv": str(features_path),
            "state_of_mind_synthetic.csv": str(synth_path)
        },
        "outputs": {
            "features_daily_labeled.csv": _sha256_file(str(out_csv)),
            "label_counts": counts,
            "rows_labeled": int(out['label'].notna().sum()),
            "rows_total": int(len(out))
        }
    }

    manifest_path = snapshot_dir / 'labels_manifest.json'
    _write_atomic_json(manifest, manifest_path)

    return {"features_daily_labeled": str(out_csv), "labels_manifest": str(manifest_path)}

# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="ETL Apple/Zepp — Nova arquitetura (definitivo)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    # extract
    p_ext = sub.add_parser("extract", help="Apple → per-metric (a partir de export.xml)")
    p_ext.add_argument("--participant", required=True)
    p_ext.add_argument("--snapshot", required=True)
    p_ext.add_argument("--cutover", required=True)
    p_ext.add_argument("--tz_before", required=True)
    p_ext.add_argument("--tz_after", required=True)

    # cardio
    p_car = sub.add_parser("cardio", help="Executa estágio Cardiovascular no snapshot")
    p_car.add_argument("--zepp_dir", help="Diretório fixo com CSVs processados do Zepp (evita _latest)")
    g = p_car.add_mutually_exclusive_group(required=True)
    g.add_argument("--snapshot_dir", help=HELP_SNAPSHOT_DIR)
    g.add_argument("--participant", help=HELP_PARTICIPANT)
    p_car.add_argument("--snapshot", help=HELP_SNAPSHOT_ID)

    # full
    p_full = sub.add_parser("full", help="extract → cardio (pipeline novo ponta-a-ponta)")
    p_full.add_argument("--participant", required=True)
    p_full.add_argument("--snapshot", required=True)
    p_full.add_argument("--cutover", required=True)
    p_full.add_argument("--tz_before", required=True)
    p_full.add_argument("--tz_after", required=True)
    p_full.add_argument("--zepp_dir", help="Diretório fixo com CSVs processados do Zepp (evita _latest)")

    # labels
    p_lab = sub.add_parser("labels", help="Join labels into features_daily_labeled.csv")
    g_lab = p_lab.add_mutually_exclusive_group(required=True)
    g_lab.add_argument("--snapshot_dir", help=HELP_SNAPSHOT_DIR)
    g_lab.add_argument("--participant", help=HELP_PARTICIPANT)
    p_lab.add_argument("--snapshot", help=HELP_SNAPSHOT_ID)
    p_lab.add_argument("--synthetic", action="store_true", help="Use synthetic labels (state_of_mind_synthetic.csv)")
    p_lab.add_argument("--labels_path", help="Caminho alternativo para labels (opcional)")

    # aggregate
    p_agg = sub.add_parser("aggregate", help="Aggregate features to one row per day (optional synthetic labels)")
    gagg = p_agg.add_mutually_exclusive_group(required=True)
    gagg.add_argument("--snapshot_dir", help=HELP_SNAPSHOT_DIR)
    gagg.add_argument("--participant", help=HELP_PARTICIPANT)
    p_agg.add_argument("--snapshot", help=HELP_SNAPSHOT_ID)
    p_agg.add_argument("--labels", choices=["none", "synthetic"], default="none", help="Join labels (synthetic) into aggregated output")

    args = ap.parse_args()

    if args.cmd == "extract":
        pid, snap = args.participant, args.snapshot
        xml = find_export_xml(pid, snap)
        if xml is None:
            print("⚠️ export.xml não encontrado para:", pid, snap)
            return 1
        outdir = ensure_ai_outdir(pid, snap)
        with Timer(f"extract [{pid}/{snap}]"):
            written = extract_apple_per_metric(xml_file=xml, outdir=outdir,
                                               cutover_str=args.cutover,
                                               tz_before=args.tz_before, tz_after=args.tz_after)
        print("✅ extract concluído")
        for k,v in written.items():
            print(f" - {k}: {v}")
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
        for pm in ["apple_heart_rate.csv","apple_hrv_sdnn.csv","apple_sleep_intervals.csv"]:
            pth = snapdir / "per-metric" / pm
            if pth.exists():
                try:
                    inputs[pm] = {"path": str(pth), "sha256": _sha256_file(pth)}
                except Exception:
                    inputs[pm] = {"path": str(pth), "sha256": None}
        for name in ["version_log_enriched.csv","version_raw.csv"]:
            pth = snapdir / name
            if pth.exists():
                try:
                    inputs[name] = {"path": str(pth), "sha256": _sha256_file(pth)}
                except Exception:
                    inputs[name] = {"path": str(pth), "sha256": None}
        zdir = os.environ.get("ZEPPOVERRIDE_DIR","")
        zepp = {}
        for z in (glob.glob(os.path.join(zdir, "*.csv")) if zdir else []):
            try:
                zepp[z] = {"path": z, "sha256": _sha256_file(z)}
            except Exception:
                zepp[z] = {"path": z, "sha256": None}

        outputs = {}
        for k,v in res.items():
            try:
                outputs[k] = {"path": v, "sha256": _sha256_file(v)}
            except Exception:
                outputs[k] = {"path": v, "sha256": None}

        manifest = {
            "type": "cardio",
            "snapshot_dir": str(snapdir),
            "zepp_dir": zdir or None,
            "inputs": {"per_metric": inputs, "zepp": zepp},
            "outputs": outputs
        }
        _write_atomic_json(manifest, snapdir / "cardio_manifest.json")

        print("✅ cardio concluído")
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

        if not args.synthetic:
            print("Somente --synthetic implementado nesta fase.")
            return 2

        labels_path = Path(args.labels_path) if args.labels_path else None
        try:
            res = run_stage_labels_synthetic(snapdir, labels_path)
        except FileNotFoundError as e:
            print("⚠️", e)
            return 1
        except Exception as e:
            print("❌ Erro ao rodar labels:", e)
            return 1

        print("✅ labels (synthetic) concluído:", res)
        return 0

    if args.cmd == "aggregate":
        # resolve snapshot dir
        if args.snapshot_dir:
            snapdir = Path(args.snapshot_dir)
        else:
            if not args.participant or not args.snapshot:
                print(USAGE_SNAPSHOT_OR_PARTICIPANT)
                return 2
            snapdir = AI_ROOT / args.participant / 'snapshots' / args.snapshot
        if not snapdir.exists():
            print(SNAPSHOT_NOT_EXISTS, snapdir)
            return 1
        # Call the aggregator via its public API. Import error or runtime will be shown to the user.
        try:
            from etl_tools.aggregate_features_daily import run as run_aggregate
            run_aggregate(snapdir, labels=args.labels)
        except Exception as e:
            print("❌ Erro ao rodar aggregate via module:", e)
            return 1

        print("✅ aggregate concluído")
        return 0

    if args.cmd == "full":
        pid, snap = args.participant, args.snapshot
        xml = find_export_xml(pid, snap)
        if xml is None:
            print("⚠️ export.xml não encontrado para:", pid, snap)
            return 1
        outdir = ensure_ai_outdir(pid, snap)
        with Timer(f"extract [{pid}/{snap}]"):
            written = extract_apple_per_metric(xml_file=xml, outdir=outdir,
                                               cutover_str=args.cutover,
                                               tz_before=args.tz_before, tz_after=args.tz_after)
        print("✅ extract concluído")
        for k,v in written.items():
            print(f" - {k}: {v}")
        if args.zepp_dir:
            os.environ["ZEPPOVERRIDE_DIR"] = args.zepp_dir
        with Timer(f"cardio [{outdir}]"):
            res = run_stage_cardio(outdir)
        print("✅ cardio concluído")
        print(res)
        return 0

    return 0

if __name__ == "__main__":
    sys.exit(main())
