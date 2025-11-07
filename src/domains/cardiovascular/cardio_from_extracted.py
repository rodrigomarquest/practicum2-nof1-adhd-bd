"""Simple Zepp cardio (heart-rate) per-day aggregator.

This is intentionally minimal: reads HEARTRATE / HEARTRATE_AUTO CSVs and
aggregates mean/max/count per day, returning Zepp-prefixed columns.
Also supports Apple heartrate from export.xml and export_cda.xml.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List
import logging
import pandas as pd
import argparse
import os
import json
import traceback
import xml.etree.ElementTree as ET
import re
from datetime import datetime
import pytz

from tqdm import tqdm

from ..parse_zepp_export import discover_zepp_tables
from lib.io_guards import write_csv

# Use lxml for fast streaming (10-50x faster than ElementTree)
try:
    from lxml import etree as LET
    HAS_LXML = True
except ImportError:
    LET = None  # type: ignore
    HAS_LXML = False

try:
    import src.etl_pipeline as etl
except Exception:
    try:
        import etl_pipeline as etl  # type: ignore
    except Exception:
        etl = None
from ..common.io import etl_snapshot_root

logger = logging.getLogger("etl.cardio")


def _parse_apple_timestamp_fast(ts_str: str) -> str | None:
    """Parse Apple timestamp string directly without pandas overhead.
    
    Apple uses format like: "2021-05-14 03:20:15 +0100"
    Returns: date string like "2021-05-14" in UTC, or None if parse fails
    """
    try:
        # Parse using strptime (fast, no pandas overhead)
        dt = datetime.strptime(ts_str[:19], "%Y-%m-%d %H:%M:%S")
        
        # Extract timezone offset if present
        offset_str = ts_str[19:].strip()
        if offset_str and offset_str[0] in '+-':
            try:
                # Parse +0100 or -0500 format
                sign = 1 if offset_str[0] == '+' else -1
                hours = int(offset_str[1:3])
                minutes = int(offset_str[3:5]) if len(offset_str) >= 5 else 0
                offset_mins = sign * (hours * 60 + minutes)
                
                # Convert to UTC
                from datetime import timedelta
                dt_utc = dt - timedelta(minutes=offset_mins)
                return str(dt_utc.date())
            except (ValueError, IndexError):
                # If offset parsing fails, assume already UTC
                return str(dt.date())
        else:
            # No offset, assume UTC
            return str(dt.date())
    except (ValueError, AttributeError):
        return None


def load_apple_cardio_from_xml(xml_path: Path, home_tz: str = "UTC", max_records: int | None = None) -> pd.DataFrame:
    """Extract Apple heartrate from export.xml using FAST binary regex streaming.
    
    Instead of full XML parsing, uses regex to extract HR records directly from
    binary file stream. This is 100-500x faster than iterparse for huge files.
    
    For 3.9GB files, this achieves ~500MB/sec vs ~10MB/sec with iterparse.
    
    Args:
        xml_path: Path to export.xml or export_cda.xml
        home_tz: Timezone for date conversion
        max_records: Limit parsing to max_records HR records (for testing)
    
    Returns:
        DataFrame with columns: date, apple_hr_mean, apple_hr_max, apple_n_hr
    """
    if not xml_path.exists():
        logger.info("apple cardio: %s not found", xml_path.name)
        return pd.DataFrame()
    
    # Get file size for progress tracking
    file_size_mb = xml_path.stat().st_size / (1024 * 1024)
    logger.info("apple cardio: fast-parsing %s (%.1f MB) with binary regex", xml_path.name, file_size_mb)
    
    # Regex to match HR records: Extract all attributes from the Record tag
    # Pattern 1: Find all Record tags with type="HKQuantityTypeIdentifierHeartRate"
    # Then extract value="..." and date attributes (creationDate, startDate, endDate)
    record_pattern = rb'<Record[^>]*?type="HKQuantityTypeIdentifierHeartRate"[^>]*?>'
    value_pattern = rb'value="([^"]+)"'
    date_patterns = [rb'(?:creationDate|startDate|endDate)="([^"]+)"']
    
    # Track statistics per day
    stats: dict[str, tuple[float, float, int]] = {}
    hr_count = 0
    bytes_read = 0
    chunk_size = 1024 * 1024 * 10  # 10MB chunks
    overlap = 200  # Keep 200 bytes overlap to catch records split across chunks
    
    try:
        pbar = tqdm(
            total=int(file_size_mb),
            unit="MB",
            desc=f"  {xml_path.name}",
            leave=False,
            disable=os.environ.get('ETL_TQDM') != '1'
        )
        
        with open(xml_path, 'rb') as fh:
            chunk_num = 0
            
            while True:
                # Read next chunk
                chunk = fh.read(chunk_size)
                if not chunk:
                    break
                
                chunk_num += 1
                bytes_read += len(chunk)
                pbar.update(len(chunk) / (1024 * 1024))
                
                # Find all HR Record tags in this chunk
                for record_match in re.finditer(record_pattern, chunk):
                    record_tag = record_match.group(0)
                    
                    # Extract value from the record
                    val_match = re.search(value_pattern, record_tag)
                    if not val_match:
                        continue
                    val_str = val_match.group(1)
                    
                    # Extract any date (prefer startDate > creationDate > endDate)
                    ts_str = None
                    for date_pat in date_patterns:
                        date_match = re.search(date_pat, record_tag)
                        if date_match:
                            ts_str = date_match.group(1)
                            break
                    
                    if not ts_str:
                        continue
                    
                    try:
                        hr = float(val_str)
                        ts = ts_str.decode('utf-8', errors='ignore')
                        
                        # Parse timestamp using FAST method (no pandas overhead)
                        d = _parse_apple_timestamp_fast(ts)
                        if d:
                            # Update stats for the day
                            cur = stats.get(d, (0.0, 0.0, 0))
                            ssum, smax, scnt = cur
                            ssum += hr
                            smax = max(smax, hr)
                            scnt += 1
                            stats[d] = (ssum, smax, scnt)
                            
                            hr_count += 1
                            if max_records is not None and hr_count >= max_records:
                                pbar.close()
                                break
                    except (ValueError, TypeError, AttributeError):
                        pass
                
                if max_records is not None and hr_count >= max_records:
                    break
        
        pbar.close()
        logger.info("apple cardio: %s parsed %d HR records into %d days", xml_path.name, hr_count, len(stats))
    
    except Exception as e:
        logger.warning("apple cardio: binary regex parse failed on %s: %s", xml_path.name, str(e))
        # Fall back to iterparse
        logger.info("apple cardio: falling back to ElementTree iterparse")
        stats = {}
        hr_count = 0
        try:
            it = ET.iterparse(str(xml_path), events=("end",))
            pbar = tqdm(unit=" records", desc=f"  {xml_path.name} [ET]", leave=False, disable=os.environ.get('ETL_TQDM') != '1')
            
            for event, elem in it:
                tag = elem.tag
                if tag.endswith("Record"):
                    typ = elem.get("type") or ""
                    if typ == "HKQuantityTypeIdentifierHeartRate":
                        val = elem.get("value")
                        ts = elem.get("startDate") or elem.get("creationDate") or elem.get("endDate")
                        
                        if ts is not None and val is not None:
                            try:
                                hr = float(val)
                                d = _parse_apple_timestamp_fast(ts)
                                if d:
                                    cur = stats.get(d, (0.0, 0.0, 0))
                                    ssum, smax, scnt = cur
                                    ssum += hr
                                    smax = max(smax, hr)
                                    scnt += 1
                                    stats[d] = (ssum, smax, scnt)
                                    
                                    hr_count += 1
                                    if hr_count % 5000 == 0:
                                        pbar.update(5000)
                                    
                                    if max_records is not None and hr_count >= max_records:
                                        pbar.close()
                                        break
                            except (ValueError, TypeError):
                                pass
                
                elem.clear()
            
            pbar.close()
            logger.info("apple cardio: %s parsed %d HR records into %d days (ElementTree fallback)", xml_path.name, hr_count, len(stats))
        except Exception as e2:
            logger.warning("apple cardio: ElementTree fallback also failed: %s", str(e2))
    
    if not stats:
        logger.info("apple cardio: no HR records found in %s", xml_path.name)
        return pd.DataFrame()
    
    # Aggregate into daily summaries
    rows = []
    for d, (ssum, smax, scnt) in sorted(stats.items()):
        rows.append({
            "date": d,
            "apple_hr_mean": float(ssum / scnt),
            "apple_hr_max": float(smax),
            "apple_n_hr": int(scnt)
        })
    
    df = pd.DataFrame(rows)
    df["apple_hr_mean"] = df["apple_hr_mean"].astype("float32")
    df["apple_hr_max"] = df["apple_hr_max"].astype("float32")
    df["apple_n_hr"] = df["apple_n_hr"].astype("int32")
    logger.info("apple cardio: %d records -> %d days from %s", hr_count, len(df), xml_path.name)
    return df


def load_apple_cardio_daily(root: Path, home_tz: str = "UTC", max_records: int | None = None) -> pd.DataFrame:
    """Load Apple heartrate data and aggregate per day.
    
    Searches for:
    1. heartrate*.csv under extracted/apple/<variant>/apple_health_export/
    2. export.xml and export_cda.xml
    
    Aggregates mean/max/count per day. XML takes precedence over CSVs.
    """
    all_dfs = []
    rows_read = 0
    
    # First try XML files (export.xml and export_cda.xml)
    xml_files_found = False
    for variant_dir in root.iterdir():
        if not variant_dir.is_dir():
            continue
        ah = variant_dir / "apple_health_export"
        if not ah.exists():
            ah = variant_dir
        
        # Try export.xml
        export_xml = ah / "export.xml"
        if export_xml.exists():
            df_xml = load_apple_cardio_from_xml(export_xml, home_tz, max_records)
            if not df_xml.empty:
                all_dfs = [df_xml]  # XML data takes precedence
                xml_files_found = True
                break
        
        # Try export_cda.xml (only if no export.xml found)
        if not xml_files_found:
            export_cda = ah / "export_cda.xml"
            if export_cda.exists():
                df_cda = load_apple_cardio_from_xml(export_cda, home_tz, max_records)
                if not df_cda.empty:
                    all_dfs = [df_cda]  # XML data takes precedence
                    xml_files_found = True
                    break
    
    # If no XML found, try CSV files
    if not xml_files_found:
        csvs: List[Path] = []
        
        # Search for heartrate CSVs under each variant
        for variant_dir in root.iterdir():
            if not variant_dir.is_dir():
                continue
            ah = variant_dir / "apple_health_export"
            if not ah.exists():
                ah = variant_dir
            csvs.extend(list(ah.rglob("heartrate*.csv")))
            csvs.extend(list(ah.rglob("heart_rate*.csv")))
        
        # Read CSV files
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
                
                # Find timestamp and heart rate columns (case-insensitive)
                cols_lower = {c.lower(): c for c in df.columns}
                ts_col = None
                for cand in ("timestamp", "time", "ts", "date", "start_time"):
                    if cand in cols_lower:
                        ts_col = cols_lower[cand]
                        break
                
                hr_col = None
                for cand in ("bpm", "heartrate", "heart_rate", "hr", "value"):
                    if cand in cols_lower:
                        hr_col = cols_lower[cand]
                        break
                
                if ts_col is None or hr_col is None:
                    logger.info("apple cardio: missing timestamp or HR column in %s", p)
                    continue
                
                # Convert timestamp and heart rate
                try:
                    ser = pd.to_datetime(df[ts_col], errors="coerce")
                    if ser.dt.tz is None:
                        ser = ser.dt.tz_localize("UTC")
                    ser = ser.dt.tz_convert(home_tz)
                    df["date"] = ser.dt.date.astype(str)
                except Exception:
                    df["date"] = pd.to_datetime(df[ts_col], errors="coerce").dt.date.astype(str)
                
                df[hr_col] = pd.to_numeric(df[hr_col], errors="coerce")
                all_dfs.append(df[["date", hr_col]].rename(columns={hr_col: "hr"}))
            except Exception as e:
                logger.info("apple cardio: failed to read CSV %s: %s", p, str(e))
    
    if not all_dfs:
        logger.info("apple cardio rows=0 (no heartrate data)")
        return pd.DataFrame()
    
    # Combine all data and aggregate
    combined = pd.concat(all_dfs, ignore_index=True, sort=False)
    
    # Handle aggregated (from XML) vs raw (from CSV) data
    if "apple_hr_mean" in combined.columns:
        # Already aggregated from XML
        grouped = combined.groupby("date")[["apple_hr_mean", "apple_hr_max", "apple_n_hr"]].agg({
            "apple_hr_mean": "mean",
            "apple_hr_max": "max",
            "apple_n_hr": "sum",
        }).reset_index()
    else:
        # Raw data from CSV - aggregate by date
        grouped = combined.groupby("date")["hr"].agg(["mean", "max", "count"]).reset_index()
        grouped = grouped.rename(columns={
            "mean": "apple_hr_mean",
            "max": "apple_hr_max",
            "count": "apple_n_hr",
        })
    
    logger.info("apple cardio rows=%d", len(grouped))
    return grouped


def _read(paths: List[Path], max_records: int | None = None) -> pd.DataFrame:
    parts = []
    rows_read = 0
    for p in paths:
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
        except Exception:
            logger.info("zepp: failed to read %s; skipping", p)
    if not parts:
        return pd.DataFrame()
    df = pd.concat(parts, ignore_index=True, sort=False)
    return df


def load_zepp_cardio_daily(tables: Dict[str, List[Path]], home_tz: str, max_records: int | None = None) -> pd.DataFrame:
    # read both HEARTRATE and HEARTRATE_AUTO
    paths = []
    if "HEARTRATE" in tables:
        paths.extend(tables["HEARTRATE"])
    if "HEARTRATE_AUTO" in tables:
        paths.extend(tables["HEARTRATE_AUTO"])
    if not paths:
        logger.info("zepp cardio rows=0")
        return pd.DataFrame()

    df = _read(paths, max_records=max_records)
    if df.empty:
        logger.info("zepp cardio rows=0")
        return pd.DataFrame()

    # try to find timestamp and hr value
    cols = {c.lower(): c for c in df.columns}
    ts_col = None
    for cand in ("timestamp", "time", "ts", "date"):
        if cand in cols:
            ts_col = cols[cand]
            break
    hr_col = None
    for cand in ("value", "heartrate", "hr", "bpm"):
        if cand in cols:
            hr_col = cols[cand]
            break

    if ts_col is None or hr_col is None:
        logger.info("zepp cardio insufficient columns: rows=0")
        return pd.DataFrame()

    try:
        ser = pd.to_datetime(df[ts_col], errors="coerce")
        if ser.dt.tz is None:
            ser = ser.dt.tz_localize("UTC")
        ser = ser.dt.tz_convert(home_tz)
        df["date"] = ser.dt.date.astype(str)
    except Exception:
        df["date"] = pd.to_datetime(df[ts_col], errors="coerce").dt.date.astype(str)

    df[hr_col] = pd.to_numeric(df[hr_col], errors="coerce")
    grouped = df.groupby("date")[hr_col].agg(["mean", "max", "count"]).reset_index()
    grouped = grouped.rename(columns={
        "mean": "zepp_hr_mean",
        "max": "zepp_hr_max",
        "count": "zepp_n_hr",
    })
    logger.info("zepp cardio rows=%d", len(grouped))
    return grouped


def _write_qc(snap_dir: Path, domain: str, df: pd.DataFrame, dry_run: bool = False) -> None:
    qc_dir = snap_dir / "qc"
    qc_dir.mkdir(parents=True, exist_ok=True)
    qc_path = qc_dir / f"{domain}_seed_qc.csv"
    if df is None or getattr(df, "empty", True):
        meta = {"date_min": "", "date_max": "", "n_days": 0, "n_rows": 0}
    else:
        dates = pd.to_datetime(df["date"], errors="coerce").dropna()
        date_min = dates.min().date().isoformat() if not dates.empty else ""
        date_max = dates.max().date().isoformat() if not dates.empty else ""
        n_days = df["date"].nunique() if "date" in df.columns else 0
        meta = {"date_min": date_min, "date_max": date_max, "n_days": int(n_days), "n_rows": int(len(df))}
    meta_df = pd.DataFrame([meta])
    write_csv(meta_df, qc_path, dry_run=dry_run)


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        import sys

        argv = sys.argv[1:]

    ap = argparse.ArgumentParser(prog="cardio_from_extracted")
    ap.add_argument("--pid", required=True)
    ap.add_argument("--snapshot", required=True)
    ap.add_argument("--dry-run", dest="dry_run", type=int, default=1)
    ap.add_argument("--max-records", type=int, default=None, help="Limit parsing to max_records (for testing)")
    args = ap.parse_args(argv)

    pid = args.pid
    snap = args.snapshot
    dry_run = bool(args.dry_run)
    max_records = args.max_records

    # resolve snapshot if needed
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
    
    # Load Apple cardio if available
    apple_base = snap_dir / "extracted" / "apple"
    apple_df = pd.DataFrame()
    if apple_base.exists():
        try:
            apple_df = load_apple_cardio_daily(apple_base, home_tz, max_records=max_records)
        except Exception:
            logger.info("apple cardio processing failed: %s", traceback.format_exc())
    
    # Load Zepp cardio if available
    zepp_root = snap_dir / "extracted" / "zepp" / "cloud"
    tables = discover_zepp_tables(zepp_root) if zepp_root.exists() else {}
    zepp_df = pd.DataFrame()
    if tables:
        try:
            zepp_df = load_zepp_cardio_daily(tables, home_tz, max_records=max_records)
        except Exception:
            logger.info("zepp cardio processing failed: %s", traceback.format_exc())
    
    # Merge Apple and Zepp on date (outer join)
    df = pd.DataFrame()
    if not apple_df.empty and not zepp_df.empty:
        df = pd.merge(apple_df, zepp_df, on="date", how="outer")
    elif not apple_df.empty:
        df = apple_df.copy()
    elif not zepp_df.empty:
        df = zepp_df.copy()

    # New vendor/variant paths
    out_apple = snap_dir / "features" / "cardio" / "apple" / "inapp" / "features_daily.csv"
    out_zepp = snap_dir / "features" / "cardio" / "zepp" / "cloud" / "features_daily.csv"
    # Legacy path for backward compatibility
    out_legacy = snap_dir / "features" / "cardio" / "features_daily.csv"

    # Dry-run: report counts but do not write
    if dry_run:
        print(f"[dry-run] cardio seed — OUT={out_legacy} rows={0 if df is None else len(df)}")
        _write_qc(snap_dir, "cardio", df, dry_run=True)
        return 0

    # Real run
    if df.empty:
        print("[info] cardio seed: no rows")
        _write_qc(snap_dir, "cardio", pd.DataFrame(), dry_run=False)
        return 2

    # Write Apple features if available
    if not apple_df.empty:
        out_apple.parent.mkdir(parents=True, exist_ok=True)
        write_csv(apple_df, out_apple, dry_run=False)
        print(f"INFO: apple/inapp: {len(apple_df)} rows")
    
    # Write Zepp features if available
    if not zepp_df.empty:
        out_zepp.parent.mkdir(parents=True, exist_ok=True)
        write_csv(zepp_df, out_zepp, dry_run=False)
        print(f"INFO: zepp/cloud: {len(zepp_df)} rows")
    
    # Write merged to legacy path for backward compatibility
    out_legacy.parent.mkdir(parents=True, exist_ok=True)
    write_csv(df, out_legacy, dry_run=False)
    
    _write_qc(snap_dir, "cardio", df, dry_run=False)
    print(f"[ok] wrote cardio features -> {out_legacy} rows={len(df)}")
    return 0


if __name__ == "__main__":
    import sys

    raise SystemExit(main())
