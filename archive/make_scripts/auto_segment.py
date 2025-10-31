#!/usr/bin/env python3
"""Auto-segment injection and verification tool.

Implements:
- inject_new_segment(snapshot_path: str, days: int = 3) -> str
- verify_auto_segment(snapshot_path: str, outdir: str = "reports/px8_validate", tz: str = "Europe/Dublin") -> dict

This is intentionally lightweight and uses only the repository's existing runners.
"""
from pathlib import Path
import csv
import json
import re
import subprocess
from datetime import timedelta
from typing import Dict

import pandas as pd


def _parse_numeric_seg(s: str) -> int:
    """Return the integer portion of a segment id like 'S1977' or '1977'."""
    if pd.isna(s):
        return -1
    s = str(s)
    m = re.findall(r"(\d+)", s)
    if not m:
        return -1
    return int(m[-1])


def inject_new_segment(snapshot_path: str, days: int = 3) -> str:
    """Inject a synthetic segment into <snapshot>/joined/version_log_enriched.csv.

    Returns the new segment id as 'S{n}'. The CSV is written back sorted by start.
    """
    snap = Path(snapshot_path)
    vpath = snap / "version_log_enriched.csv"
    # also accept joined/version_log_enriched.csv
    if not vpath.exists():
        vpath = snap / "joined" / "version_log_enriched.csv"
    if not vpath.exists():
        raise FileNotFoundError(f"version_log_enriched.csv not found under {snapshot_path}")

    # Read with pandas for robust date parsing
    df = pd.read_csv(vpath, parse_dates=["start", "end"], dayfirst=False)

    # determine max existing id
    if "segment_id" in df.columns:
        max_id = max(_parse_numeric_seg(x) for x in df["segment_id"].astype(str).tolist())
    else:
        max_id = max(_parse_numeric_seg(x) for x in df.iloc[:, 0].astype(str).tolist())

    new_numeric = max_id + 1
    new_seg = f"S{new_numeric}"

    # last end date
    last_end = df["end"].max()
    if pd.isna(last_end):
        last_end = pd.Timestamp.now()

    start_dt = (last_end + timedelta(days=1)).date()
    end_dt = (pd.Timestamp(start_dt) + timedelta(days=days)).date()

    # prepare a new row mapping to existing columns where possible
    cols = list(df.columns)
    new_row = {c: "" for c in cols}
    # common mappings
    if "segment_id" in cols:
        new_row["segment_id"] = new_seg
    else:
        # put it first column if no header
        new_row[cols[0]] = new_seg

    # set start/end using ISO dates
    if "start" in cols:
        new_row["start"] = start_dt.isoformat()
    if "end" in cols:
        new_row["end"] = end_dt.isoformat()

    # reasonable defaults for other known columns
    for k in ("ios_version", "watch_model", "watch_fw", "tz_name", "source_name"):
        if k in cols:
            if k == "tz_name":
                new_row[k] = "Europe/Dublin"
            else:
                new_row[k] = "auto"

    # If file uses different column names, try to map common names
    fallback_map = {
        "src": "source_name",
        "firmware_version": "watch_fw",
        "app_version": "ios_version",
        "notes": "source_name",
    }
    for k, v in fallback_map.items():
        if k in cols and v in new_row:
            new_row[k] = new_row[v]

    # Append and sort by start
    df_append = pd.DataFrame([new_row])
    try:
        df_out = pd.concat([df, df_append], ignore_index=True)
    except Exception:
        # fallback: write row as plain CSV append
        with open(vpath, "a", newline="", encoding="utf8") as f:
            w = csv.writer(f)
            rowvals = [new_row.get(c, "") for c in cols]
            w.writerow(rowvals)
        return new_seg

    # normalize start/end to strings YYYY-MM-DD
    if "start" in df_out.columns:
        df_out["start"] = pd.to_datetime(df_out["start"]).dt.strftime("%Y-%m-%d")
    if "end" in df_out.columns:
        df_out["end"] = pd.to_datetime(df_out["end"]).dt.strftime("%Y-%m-%d")

    # Sort by start date if possible
    try:
        if "start" in df_out.columns:
            df_out = df_out.sort_values(by=["start"]) 
    except Exception:
        pass

    # write back, preserving header
    df_out.to_csv(vpath, index=False)

    return new_seg


def verify_auto_segment(snapshot_path: str, outdir: str = "reports/px8_validate", tz: str = "Europe/Dublin") -> Dict:
    """Run PX8-Lite runner against the given snapshot and return summary dict.

    The function calls the repo runner and parses stdout for rows and READY.
    """
    snap = Path(snapshot_path)
    vpath = snap / "joined" / "version_log_enriched.csv"
    prov = snap / "etl_qc_summary_manifest.json"

    cmd = [
        "python",
        "make_scripts/etl_px8_lite.py",
        "--snapshot",
        str(snap),
        "--outdir",
        outdir,
        "--version-log",
        str(vpath),
        "--provenance",
        str(prov),
        "--tz",
        tz,
    ]

    try:
        proc = subprocess.run(cmd, check=True, capture_output=True, text=True)
        out = proc.stdout + proc.stderr
    except subprocess.CalledProcessError as e:
        out = (e.stdout or "") + (e.stderr or "")

    # parse rows and READY
    rows = None
    ready = False
    m_rows = re.search(r"rows\s*=\s*(\d+)", out)
    if m_rows:
        try:
            rows = int(m_rows.group(1))
        except Exception:
            rows = None
    m_ready = re.search(r"READY\s*=\s*(true|false)", out, re.IGNORECASE)
    if m_ready:
        ready = m_ready.group(1).lower() == "true"

    # count segments in version_log_enriched.csv
    seg_count = 0
    if vpath.exists():
        try:
            vdf = pd.read_csv(vpath)
            if "segment_id" in vdf.columns:
                seg_count = len(vdf["segment_id"].dropna().unique())
            else:
                seg_count = len(vdf)
        except Exception:
            seg_count = 0

    return {"segments_total": seg_count, "px8_rows": rows or 0, "READY": bool(ready)}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Inject and verify a new auto segment")
    parser.add_argument("--snapshot", required=True, help="Path to snapshot folder")
    parser.add_argument("--days", type=int, default=3, help="Duration of new segment (days)")
    parser.add_argument("--tz", default="Europe/Dublin")
    args = parser.parse_args()

    seg_id = inject_new_segment(args.snapshot, args.days)
    result = verify_auto_segment(args.snapshot, tz=args.tz)
    out = {"AUTO-SEGMENT": {"new_segment_id": seg_id, **result}}
    print(json.dumps(out, indent=2))
