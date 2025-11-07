"""Minimal Zepp sleep daily aggregator.

Reads SLEEP/*.csv and aggregates total/deep/light/rem hours per day when
available. Columns in result are prefixed with `zepp_slp_`.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List
import logging
import pandas as pd
import argparse
import os
import json
import time
from ..parse_zepp_export import discover_zepp_tables
from lib.io_guards import write_csv
import math
import traceback

try:
    import src.etl_pipeline as etl
except Exception:
    try:
        import etl_pipeline as etl  # type: ignore
    except Exception:
        etl = None
from ..common.io import etl_snapshot_root

logger = logging.getLogger("etl.sleep")


def discover_zepp_sleep(snap_dir: Path) -> list[tuple[str, str]]:
    """Discover Zepp vendor/variant combination for sleep.
    
    Returns list of tuples: (vendor, variant)
    For Zepp, vendor='zepp' and variant='cloud' (standard location: extracted/zepp/cloud/)
    """
    zepp_root = snap_dir / "extracted" / "zepp" / "cloud"
    if zepp_root.exists():
        return [("zepp", "cloud")]
    return []


def discover_apple_sleep(root: Path) -> list[tuple[str, str]]:
    """Discover Apple vendor/variant combination for sleep.
    
    Returns list of tuples: (vendor, variant)
    Searches for apple_sleep_intervals.csv under extracted/apple/<variant>/apple_health_export/
    """
    out: list[tuple[str, str]] = []
    if not root.exists():
        return out
    
    for variant_dir in root.iterdir():
        if not variant_dir.is_dir():
            continue
        variant_name = variant_dir.name
        ah = variant_dir / "apple_health_export"
        if not ah.exists():
            ah = variant_dir
        
        # Look for apple_sleep_intervals.csv
        sleep_csvs = list(ah.rglob("*sleep*interval*.csv")) + list(ah.rglob("*sleep*.csv"))
        if sleep_csvs:
            out.append(("apple", variant_name))
    
    return out


def load_apple_sleep_daily(root: Path, home_tz: str = "UTC", max_records: int | None = None) -> pd.DataFrame:
    """Load Apple sleep intervals and aggregate per day.
    
    Searches for apple_sleep_intervals.csv under extracted/apple/<variant>/apple_health_export/
    and aggregates sleep durations per day.
    """
    csvs: List[Path] = []
    
    # Search for sleep interval CSVs under each variant
    for variant_dir in root.iterdir():
        if not variant_dir.is_dir():
            continue
        ah = variant_dir / "apple_health_export"
        if not ah.exists():
            ah = variant_dir
        csvs.extend(list(ah.rglob("*sleep*interval*.csv")))
        csvs.extend(list(ah.rglob("*sleep*.csv")))
    
    if not csvs:
        logger.info("apple sleep rows=0 (no sleep CSVs)")
        return pd.DataFrame()
    
    # Read and aggregate sleep CSVs
    parts = []
    rows_read = 0
    for p in csvs:
        try:
            df = pd.read_csv(p)
            
            # Limit rows if max_records is set
            if max_records is not None:
                rows_available = max_records - rows_read
                if rows_available <= 0:
                    break
                df = df.iloc[:rows_available]
                rows_read += len(df)
            
            parts.append(df)
        except Exception as e:
            logger.info("apple sleep: failed to read %s: %s", p, str(e))
    
    if not parts:
        logger.info("apple sleep rows=0 (failed to read CSVs)")
        return pd.DataFrame()
    
    # Combine all parts
    combined = pd.concat(parts, ignore_index=True, sort=False)
    
    # Find timestamp columns (case-insensitive)
    cols_lower = {c.lower(): c for c in combined.columns}
    start_col = None
    end_col = None
    for cand in ("start", "start_time", "startdate", "timestamp"):
        if cand in cols_lower:
            start_col = cols_lower[cand]
            break
    for cand in ("end", "end_time", "enddate"):
        if cand in cols_lower:
            end_col = cols_lower[cand]
            break
    
    if start_col is None or end_col is None:
        logger.info("apple sleep: missing start or end columns")
        return pd.DataFrame()
    
    # Convert timestamps and calculate durations
    try:
        start_dt = pd.to_datetime(combined[start_col], errors="coerce")
        end_dt = pd.to_datetime(combined[end_col], errors="coerce")
        
        if start_dt.dt.tz is None:
            start_dt = start_dt.dt.tz_localize("UTC")
        if end_dt.dt.tz is None:
            end_dt = end_dt.dt.tz_localize("UTC")
        
        start_dt = start_dt.dt.tz_convert(home_tz)
        end_dt = end_dt.dt.tz_convert(home_tz)
        
        combined["date"] = start_dt.dt.date.astype(str)
        combined["duration_hours"] = (end_dt - start_dt).dt.total_seconds() / 3600.0
    except Exception as e:
        logger.info("apple sleep: failed to process timestamps: %s", str(e))
        return pd.DataFrame()
    
    # Aggregate by date: sum total sleep hours
    grouped = combined.groupby("date")["duration_hours"].sum().reset_index()
    grouped = grouped.rename(columns={"duration_hours": "apple_slp_total_h"})
    grouped["apple_slp_total_h"] = grouped["apple_slp_total_h"].astype("float32")
    
    logger.info("apple sleep rows=%d", len(grouped))
    return grouped


def _read(paths: List[Path], max_records: int | None = None) -> pd.DataFrame:
    parts = []
    rows_read = 0
    for p in paths:
        try:
            # Try to read with various parameters to handle encoding and CSV quirks
            try:
                df = pd.read_csv(p, encoding='utf-8-sig', on_bad_lines='skip', engine='python')
            except Exception:
                try:
                    df = pd.read_csv(p, encoding='utf-8', on_bad_lines='skip', engine='python')
                except Exception:
                    df = pd.read_csv(p, on_bad_lines='skip', engine='python')
            
            # Limit rows if max_records is set
            if max_records is not None:
                rows_available = max_records - rows_read
                if rows_available <= 0:
                    break
                df = df.iloc[:rows_available]
                rows_read += len(df)
            
            parts.append(df)
        except Exception as e:
            logger.info("zepp: failed to read %s; error: %s", p, str(e))
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True, sort=False)


def _discover_zepp_sleep_files_cloud(snap_dir: Path) -> List[Path]:
    """Discover CSV files under extracted/zepp/cloud/SLEEP (case-insensitive).

    Returns list of CSV Paths (may be empty).
    """
    base = Path(snap_dir) / "extracted" / "zepp" / "cloud"
    if not base.exists():
        return []
    candidates: List[Path] = []
    # look for any child directory named sleep (case-insensitive)
    for p in base.iterdir():
        if not p.is_dir():
            continue
        if p.name.lower() == "sleep":
            candidates.extend([f for f in p.rglob("*.csv") if f.is_file()])
    # also accept top-level SLEEP dir
    for p in base.rglob("*/sleep"):
        if p.is_dir():
            candidates.extend([f for f in p.rglob("*.csv") if f.is_file()])
    # deduplicate and return
    seen = set()
    out = []
    for f in candidates:
        fp = str(f)
        if fp not in seen:
            seen.add(fp)
            out.append(f)
    return out


def _agg_daily(df: pd.DataFrame, home_tz: str = "UTC") -> pd.DataFrame:
    """Handle per-day summary CSVs. Normalize to datetime64 and compute hours."""
    if df.empty:
        return pd.DataFrame()
    # possible column names (Zepp uses Time suffix; others use Duration)
    possible_total = ["sleepDuration", "total_minutes", "total_mins", "duration", "total_hours", "duration_h", "sleepTime"]
    possible_deep = ["deepSleepDuration", "deep_minutes", "deep_mins", "deep_hours", "deep_h", "deepSleepTime"]
    possible_light = ["lightSleepDuration", "light_minutes", "light_mins", "light_hours", "light_h", "shallowSleepTime", "lightSleepTime"]
    possible_rem = ["remSleepDuration", "rem_minutes", "rem_mins", "rem_hours", "rem_h", "REMTime"]

    colmap = {c.lower(): c for c in df.columns}

    def pick(cols):
        for c in cols:
            if c in colmap:
                return colmap[c]
        return None

    total_col = pick([c.lower() for c in possible_total])
    deep_col = pick([c.lower() for c in possible_deep])
    light_col = pick([c.lower() for c in possible_light])
    rem_col = pick([c.lower() for c in possible_rem])

    out = pd.DataFrame()
    # determine date column
    date_col = None
    for cand in ("date", "day", "start_date", "start_time", "timestamp"):
        if cand in colmap:
            date_col = colmap[cand]
            break
    if date_col is None:
        # try first column
        date_col = df.columns[0]

    # UTC → home_tz conversion: localize to UTC, convert to home_tz, then extract date
    ser = pd.to_datetime(df[date_col], errors="coerce", utc=True)
    ser = ser.dt.tz_convert(home_tz).dt.normalize()
    out["date"] = ser

    def to_hours(col):
        if col is None:
            return pd.Series([0.0] * len(df))
        s = pd.to_numeric(df[col], errors="coerce").fillna(0)
        # if values look like minutes (max > 24), convert to hours
        if (s.max(skipna=True) or 0) > 24:
            s = s / 60.0
        return s.astype("float32")

    deep_h = to_hours(deep_col)
    light_h = to_hours(light_col)
    rem_h = to_hours(rem_col)
    
    out["zepp_slp_deep_h"] = deep_h
    out["zepp_slp_light_h"] = light_h
    out["zepp_slp_rem_h"] = rem_h
    
    # If total_col not found, compute as sum of components
    if total_col is not None:
        out["zepp_slp_total_h"] = to_hours(total_col)
    else:
        out["zepp_slp_total_h"] = (deep_h + light_h + rem_h).astype("float32")

    agg = {k: "sum" for k in out.columns if k != "date"}
    res = out.groupby("date", as_index=False).agg(agg)
    # drop rows where total==0
    if "zepp_slp_total_h" in res.columns:
        res = res[res["zepp_slp_total_h"] > 0]
    logger.info("zepp sleep rows=%d", len(res))
    return res


def _agg_intervals(df: pd.DataFrame, home_tz: str = "UTC") -> pd.DataFrame:
    """Handle interval CSVs with start/stop times and optional sleep stage columns.
    
    If dedicated sleep stage columns exist (deepSleepTime, shallowSleepTime, REMTime),
    use those. Otherwise, compute duration from start/stop times and aggregate by date.
    """
    if df.empty:
        return pd.DataFrame()
    colmap = {c.lower(): c for c in df.columns}
    
    # Find date column
    date_col = None
    for cand in ("date", "day", "start_date", "timestamp"):
        if cand in colmap:
            date_col = colmap[cand]
            break
    if date_col is None:
        date_col = df.columns[0]
    
    # Extract date from date column
    dates_raw = pd.to_datetime(df[date_col], errors="coerce", utc=True)
    dates = dates_raw.dt.tz_convert(home_tz).dt.normalize()
    
    # Find start/stop columns
    start_col = None
    end_col = None
    for cand in ("start_time", "start", "begin", "startdate", "start_date"):
        if cand in colmap:
            start_col = colmap[cand]
            break
    for cand in ("end_time", "end", "stop", "finish", "enddate", "end_date"):
        if cand in colmap:
            end_col = colmap[cand]
            break
    
    # Look for dedicated sleep stage columns (time in minutes)
    deep_col = None
    light_col = None
    rem_col = None
    for cand in ("deepsleeptime", "deep_sleep_time", "deep_sleep", "deep"):
        if cand in colmap:
            deep_col = colmap[cand]
            break
    for cand in ("shallowsleeptime", "light_sleep_time", "light_sleep", "light", "lightsleeptime"):
        if cand in colmap:
            light_col = colmap[cand]
            break
    for cand in ("remtime", "rem_time", "rem"):
        if cand in colmap:
            rem_col = colmap[cand]
            break
    
    # Build result with dedicated columns if available
    if deep_col or light_col or rem_col:
        deep_h = (df[deep_col].fillna(0) / 60.0) if deep_col else 0.0
        light_h = (df[light_col].fillna(0) / 60.0) if light_col else 0.0
        rem_h = (df[rem_col].fillna(0) / 60.0) if rem_col else 0.0
        
        df2 = pd.DataFrame({
            "date": dates,
            "zepp_slp_deep_h": deep_h,
            "zepp_slp_light_h": light_h,
            "zepp_slp_rem_h": rem_h,
        })
        df2["zepp_slp_total_h"] = df2["zepp_slp_deep_h"] + df2["zepp_slp_light_h"] + df2["zepp_slp_rem_h"]
        
        # Aggregate by date
        res = df2.groupby("date", as_index=False).agg({
            "zepp_slp_total_h": "sum",
            "zepp_slp_deep_h": "sum",
            "zepp_slp_light_h": "sum",
            "zepp_slp_rem_h": "sum",
        })
        # Drop zero rows
        res = res[res["zepp_slp_total_h"] > 0]
        logger.info("zepp sleep (intervals with stages) rows=%d", len(res))
        return res
    
    # Fallback: compute from start/stop
    if start_col is None or end_col is None:
        logger.debug("zepp sleep: no start/stop columns found; skipping intervals")
        return pd.DataFrame()

    start = pd.to_datetime(df[start_col], errors="coerce", utc=True)
    end = pd.to_datetime(df[end_col], errors="coerce", utc=True)
    dur_h = (end - start).dt.total_seconds().div(3600).fillna(0)
    
    df2 = pd.DataFrame({"date": dates, "dur_h": dur_h})
    
    stage_col = None
    for cand in ("stage", "sleep_stage", "state"):
        if cand in colmap:
            stage_col = colmap[cand]
            break
    
    if stage_col is None:
        # only total
        res = df2.groupby("date", as_index=False).sum()
        res = res.rename(columns={"dur_h": "zepp_slp_total_h"})
        res["zepp_slp_deep_h"] = 0.0
        res["zepp_slp_light_h"] = 0.0
        res["zepp_slp_rem_h"] = 0.0
        # drop zero totals
        res = res[res["zepp_slp_total_h"] > 0]
        logger.info("zepp sleep (intervals computed) rows=%d", len(res))
        return res

    stages = df[stage_col].astype(str).str.lower()
    df2["stage"] = stages

    def map_stage(s):
        if "deep" in s:
            return "deep"
        if "rem" in s:
            return "rem"
        if "light" in s:
            return "light"
        return "other"

    df2["stage_norm"] = df2["stage"].fillna("other").apply(map_stage)
    agg = df2.groupby(["date", "stage_norm"], as_index=False)["dur_h"].sum()
    # pivot
    piv = agg.pivot(index="date", columns="stage_norm", values="dur_h").fillna(0)
    piv = piv.rename(columns={
        "deep": "zepp_slp_deep_h",
        "light": "zepp_slp_light_h",
        "rem": "zepp_slp_rem_h",
    })
    piv["zepp_slp_total_h"] = piv.sum(axis=1)
    piv = piv.reset_index()
    # ensure columns exist
    for c in ["zepp_slp_total_h", "zepp_slp_deep_h", "zepp_slp_light_h", "zepp_slp_rem_h"]:
        if c not in piv.columns:
            piv[c] = 0.0
    # drop zero totals
    piv = piv[piv["zepp_slp_total_h"] > 0]
    logger.info("zepp sleep rows=%d", len(piv))
    return piv


def _parse_naps_column(naps_str, home_tz: str = "UTC"):
    """Parse naps JSON column into list of (start, end) tuples in hours.
    
    Format: [{"start":"2022-01-08 00:33:10+0000", "end":"2022-01-08 00:33:28+0000"}, ...]
    Returns list of duration in hours for each nap.
    """
    if pd.isna(naps_str) or naps_str == "" or naps_str == "[]":
        return []
    
    try:
        # Clean up common formatting issues
        naps_str = str(naps_str).strip()
        # Parse as JSON array
        naps_list = json.loads(naps_str)
        if not isinstance(naps_list, list):
            return []
        
        durations = []
        for nap in naps_list:
            if isinstance(nap, dict) and "start" in nap and "end" in nap:
                try:
                    start = pd.to_datetime(nap["start"], utc=True)
                    end = pd.to_datetime(nap["end"], utc=True)
                    duration_h = (end - start).total_seconds() / 3600
                    if duration_h > 0:  # only include positive durations
                        durations.append(duration_h)
                except Exception:
                    continue
        return durations
    except Exception as e:
        logger.debug("failed to parse naps: %s", str(e))
        return []


def _agg_naps(df: pd.DataFrame, home_tz: str = "UTC") -> pd.DataFrame:
    """Handle naps data from Zepp SLEEP CSV where naps are stored as JSON in columns.
    
    Format: columns starting from 'naps' contain JSON arrays of {start, end} timestamps.
    Each pair of columns (start_nap, end_nap) represents one or more naps.
    """
    if df.empty:
        return pd.DataFrame()
    
    colmap = {c.lower(): c for c in df.columns}
    
    # Find date column
    date_col = None
    for cand in ("date", "day", "start_date", "start_time", "timestamp"):
        if cand in colmap:
            date_col = colmap[cand]
            break
    if date_col is None:
        date_col = df.columns[0]
    
    # Extract date and normalize to home timezone
    ser = pd.to_datetime(df[date_col], errors="coerce", utc=True)
    dates = ser.dt.tz_convert(home_tz).dt.normalize()
    
    # Find naps columns (typically after a "naps" column)
    naps_total = []
    for idx, row in df.iterrows():
        total_nap_h = 0.0
        # Look for columns that start with "naps" or contain JSON with start/end
        for col in df.columns:
            val = row[col]
            if pd.isna(val) or val == "" or val == "[]":
                continue
            # Try to parse as naps JSON
            try:
                val_str = str(val).strip()
                if val_str.startswith("["):
                    durations = _parse_naps_column(val_str, home_tz)
                    total_nap_h += sum(durations)
            except Exception:
                continue
        naps_total.append(total_nap_h)
    
    out = pd.DataFrame({
        "date": dates,
        "zepp_slp_total_h": pd.Series(naps_total, dtype="float32"),
        "zepp_slp_deep_h": 0.0,
        "zepp_slp_light_h": 0.0,
        "zepp_slp_rem_h": 0.0,
    })
    
    # Group by date to aggregate multiple entries per day
    agg = {k: "sum" for k in out.columns if k != "date"}
    res = out.groupby("date", as_index=False).agg(agg)
    # Drop rows where total==0
    if "zepp_slp_total_h" in res.columns:
        res = res[res["zepp_slp_total_h"] > 0]
    logger.info("zepp sleep (naps) rows=%d", len(res))
    return res


def load_zepp_sleep_daily_from_cloud(snap_dir: Path, home_tz: str = "UTC", max_records: int | None = None) -> pd.DataFrame:
    """Discover and load Zepp cloud sleep CSVs, handling daily, interval, and naps formats.
    
    Args:
        snap_dir: Snapshot directory path
        home_tz: User's home timezone
        max_records: Limit parsing to max_records (for testing)
    """
    files = _discover_zepp_sleep_files_cloud(snap_dir)
    if not files:
        logger.info("zepp sleep rows=0 (no files)")
        return pd.DataFrame()

    parts_daily = []
    parts_intervals = []
    parts_naps = []
    rows_read = 0
    for p in files:
        try:
            # Use robust CSV reading with BOM and bad line handling
            df = pd.read_csv(p, encoding='utf-8-sig', on_bad_lines='skip', engine='python')
        except Exception:
            try:
                df = pd.read_csv(p, on_bad_lines='skip', engine='python')
            except Exception as e:
                logger.info("zepp: failed to read %s; error: %s", p, str(e))
                continue
        
        # Limit rows if max_records is set
        if max_records is not None:
            rows_available = max_records - rows_read
            if rows_available <= 0:
                break
            df = df.iloc[:rows_available]
            rows_read += len(df)
        
        cols = {c.lower(): c for c in df.columns}
        
        # Detect format by presence of key columns
        has_start_end = any(c in cols for c in ("start_time", "start", "end_time", "end"))
        has_naps = any("nap" in c.lower() for c in df.columns)
        
        if has_start_end:
            parts_intervals.append(df)
        elif has_naps:
            parts_naps.append(df)
        else:
            parts_daily.append(df)

    res_parts = []
    if parts_daily:
        df_all = pd.concat(parts_daily, ignore_index=True, sort=False)
        res_parts.append(_agg_daily(df_all, home_tz=home_tz))
    if parts_intervals:
        df_all = pd.concat(parts_intervals, ignore_index=True, sort=False)
        res_parts.append(_agg_intervals(df_all, home_tz=home_tz))
    if parts_naps:
        df_all = pd.concat(parts_naps, ignore_index=True, sort=False)
        res_parts.append(_agg_naps(df_all, home_tz=home_tz))

    if not res_parts:
        logger.info("zepp sleep rows=0 (no parsable files)")
        return pd.DataFrame()

    combined = pd.concat(res_parts, ignore_index=True, sort=False)
    # group again by date to merge daily+intervals+naps
    agg = {c: "sum" for c in combined.columns if c != "date"}
    out = combined.groupby("date", as_index=False).agg(agg)
    # ensure date is datetime64 normalized
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.normalize()
    # keep column order
    cols = ["date", "zepp_slp_total_h", "zepp_slp_deep_h", "zepp_slp_light_h", "zepp_slp_rem_h"]
    for c in cols:
        if c not in out.columns:
            out[c] = 0.0
    out = out[cols]
    logger.info("zepp sleep rows=%d", len(out))
    return out


def _write_qc(snap_dir: Path, domain: str, df: pd.DataFrame, input_files: List[Path] = None, dry_run: bool = False) -> None:
    qc_dir = snap_dir / "qc"
    qc_dir.mkdir(parents=True, exist_ok=True)
    qc_path = qc_dir / f"{domain}_seed_qc.csv"
    if df is None or getattr(df, "empty", True):
        meta = {
            "date_min": "", 
            "date_max": "", 
            "n_days": 0, 
            "n_rows": 0,
            "coverage_pct": 0.0,
            "source_files": "none",
            "processed_at": pd.Timestamp.now(tz="UTC").isoformat(),
        }
    else:
        dates = pd.to_datetime(df["date"], errors="coerce").dropna()
        date_min = dates.min().date().isoformat() if not dates.empty else ""
        date_max = dates.max().date().isoformat() if not dates.empty else ""
        n_days = df["date"].nunique() if "date" in df.columns else 0
        expected_days = (pd.to_datetime(date_max) - pd.to_datetime(date_min)).days + 1 if date_min and date_max else n_days
        coverage_pct = round((n_days / expected_days * 100), 2) if expected_days > 0 else 0.0
        source_files_str = ";".join([p.name for p in (input_files or [])]) if input_files else "none"
        meta = {
            "date_min": date_min, 
            "date_max": date_max, 
            "n_days": int(n_days), 
            "n_rows": int(len(df)),
            "coverage_pct": float(coverage_pct),
            "source_files": source_files_str,
            "processed_at": pd.Timestamp.now(tz="UTC").isoformat(),
        }
    meta_df = pd.DataFrame([meta])
    write_csv(meta_df, qc_path, dry_run=dry_run)


def write_seed(
    df: pd.DataFrame,
    snap_dir: Path,
    dry_run: bool = False,
    vendor: str | None = None,
    variant: str | None = None,
) -> None:
    """Write sleep features to both new vendor/variant structure and legacy path.
    
    If vendor and variant are provided, writes to:
      - features/sleep/<vendor>/<variant>/features_daily.csv (NEW)
      - features/sleep/features_daily.csv (LEGACY for backward compatibility)
    Otherwise writes only to legacy path.
    """
    if df.empty:
        return
    
    snap_dir = Path(snap_dir)
    
    # Write to new vendor/variant structure if provided
    if vendor is not None and variant is not None:
        new_path = snap_dir / "features" / "sleep" / vendor / variant / "features_daily.csv"
        new_path.parent.mkdir(parents=True, exist_ok=True)
        if not dry_run:
            write_csv(df, new_path, dry_run=False)
            logger.info("sleep: %s/%s rows=%d", vendor, variant, len(df))
    
    # Always write to legacy path for backward compatibility
    legacy_path = snap_dir / "features" / "sleep" / "features_daily.csv"
    legacy_path.parent.mkdir(parents=True, exist_ok=True)
    if not dry_run:
        write_csv(df, legacy_path, dry_run=False)


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        import sys

        argv = sys.argv[1:]

    ap = argparse.ArgumentParser(prog="sleep_from_extracted")
    ap.add_argument("--pid", required=True)
    ap.add_argument("--snapshot", required=True)
    ap.add_argument("--dry-run", dest="dry_run", type=int, default=1)
    ap.add_argument("--max-records", type=int, default=None, help="Limit parsing to max_records (for testing)")
    ap.add_argument("--allow-empty", dest="allow_empty", type=int, default=0,
                    help="Exit 0 even if resulting features are empty when set to 1")
    args = ap.parse_args(argv)

    pid = args.pid
    snap = args.snapshot
    dry_run = bool(args.dry_run)
    max_records = getattr(args, "max_records", None)

    if snap == "auto":
        if etl is None:
            print("[error] cannot resolve 'auto' snapshot; etl resolver not available")
            return 3
        snap = etl.resolve_snapshot(snap, pid)

    snap_dir = etl_snapshot_root(pid, snap)

    # Determine home_tz from participant profile if available
    home_tz = "UTC"
    try:
        profile_path = Path("data") / "config" / f"{pid}_profile.json"
        if profile_path.exists():
            with open(profile_path, "r", encoding="utf-8") as fh:
                prof = json.load(fh)
                home_tz = prof.get("home_tz", home_tz)
    except Exception:
        logger.info("could not read participant profile for home_tz; defaulting to UTC")

    # Discover input files for QC
    input_files = _discover_zepp_sleep_files_cloud(snap_dir)

    # Load Apple sleep if available
    apple_base = snap_dir / "extracted" / "apple"
    apple_df = pd.DataFrame()
    if apple_base.exists():
        try:
            apple_df = load_apple_sleep_daily(apple_base, home_tz, max_records=max_records)
        except Exception:
            logger.info("apple sleep processing failed: %s", traceback.format_exc())
    
    # Load Zepp sleep from extracted/zepp/cloud/SLEEP
    zepp_df = pd.DataFrame()
    try:
        zepp_df = load_zepp_sleep_daily_from_cloud(snap_dir, home_tz=home_tz, max_records=max_records)
    except Exception:
        logger.info("zepp sleep failed: %s", traceback.format_exc())
    
    # Merge Apple and Zepp on date (outer join)
    df = pd.DataFrame()
    if not apple_df.empty and not zepp_df.empty:
        df = pd.merge(apple_df, zepp_df, on="date", how="outer")
    elif not apple_df.empty:
        df = apple_df.copy()
    elif not zepp_df.empty:
        df = zepp_df.copy()

    if dry_run:
        out = snap_dir / "features" / "sleep" / "features_daily.csv"
        print(f"[dry-run] sleep seed — OUT={out} rows={0 if df.empty else len(df)}")
        _write_qc(snap_dir, "sleep", df, input_files=input_files, dry_run=True)
        return 0

    # Real run: ensure output dir exists and write features even if empty
    # Guarantee 'date' is datetime64 (day-normalized) until write
    if not df.empty and "date" in df.columns:
        try:
            df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
        except Exception:
            pass

    # Write individual vendor/variant files for Apple
    if not apple_df.empty:
        write_seed(apple_df, snap_dir, dry_run=False, vendor="apple", variant="inapp")
        print(f"INFO: apple/inapp: {len(apple_df)} rows")
    
    # Write individual vendor/variant files for Zepp
    zepp_vendors = discover_zepp_sleep(snap_dir)
    for vendor, variant in zepp_vendors:
        if not zepp_df.empty:
            write_seed(zepp_df, snap_dir, dry_run=False, vendor=vendor, variant=variant)
            print(f"INFO: zepp/cloud: {len(zepp_df)} rows")

    # Write merged to legacy path for backward compatibility
    write_seed(df, snap_dir, dry_run=False, vendor=None, variant=None)
    _write_qc(snap_dir, "sleep", df, input_files=input_files, dry_run=False)

    n_rows = 0 if df.empty else len(df)
    if n_rows == 0 and not bool(args.allow_empty):
        print("[info] sleep seed: no rows")
        return 2
    out = snap_dir / "features" / "sleep" / "features_daily.csv"
    print(f"[ok] wrote sleep features -> {out} rows={n_rows}")
    return 0



if __name__ == "__main__":
    import sys

    raise SystemExit(main())
