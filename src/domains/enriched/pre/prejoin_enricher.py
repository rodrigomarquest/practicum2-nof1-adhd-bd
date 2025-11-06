# -*- coding: utf-8 -*-
"""Pre-join enrichment orchestrator.

This module implements per-domain enrichment that:
1. Reads features/<domain>/<vendor>/<variant>/features_daily.csv
2. Applies domain-specific enrichments (rolling averages, standardization, etc.)
3. Writes to enriched/prejoin/<domain>/<vendor>/<variant>/enriched_<domain>.csv

The enrichment is idempotent and respects MAX_RECORDS for testing.

Public API
----------
enrich_prejoin_run(snapshot_dir, *, dry_run=False, max_records=None) -> int

Exit codes:
- 0 : success (or dry-run with valid input)
- 2 : no features found or missing required columns
- 1 : IO / unexpected error
"""
from __future__ import annotations

from pathlib import Path
import tempfile
import traceback
from typing import Optional
import os

import pandas as pd
import numpy as np

from lib.io_guards import write_csv


def _ensure_dir(p: Path) -> Path:
    """Ensure directory exists."""
    p.mkdir(parents=True, exist_ok=True)
    return p


def _write_atomic_csv(df: pd.DataFrame, out_path: Path | str):
    """Atomic CSV write with temp file."""
    out_path = Path(out_path)
    _ensure_dir(out_path.parent)
    
    d = str(out_path.parent) or "."
    fd, tmp = tempfile.mkstemp(prefix=".tmp_", dir=d, suffix=".csv")
    os.close(fd)
    try:
        df.to_csv(tmp, index=False)
        os.replace(tmp, str(out_path))
    finally:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except Exception:
                pass


def _zscore(series: pd.Series) -> pd.Series:
    """Compute z-score, handling NaN gracefully."""
    mask = series.notna()
    result = pd.Series([np.nan] * len(series), index=series.index)
    if mask.sum() > 0:
        mean = series[mask].mean()
        std = series[mask].std()
        if std > 0:
            result[mask] = (series[mask] - mean) / std
    return result


def _rolling_mean_7d(df: pd.DataFrame, col: str, new_col: str) -> pd.DataFrame:
    """Add 7-day rolling average column, respecting date ordering."""
    df = df.copy()
    if col not in df.columns:
        return df
    
    if "date" not in df.columns:
        return df
    
    # Ensure date is datetime
    try:
        df["date"] = pd.to_datetime(df["date"])
    except Exception:
        return df
    
    # Sort by date to ensure proper rolling
    df = df.sort_values("date").reset_index(drop=True)
    
    # Rolling mean with 7-day window (min_periods=1 to include partial windows)
    df[new_col] = df[col].rolling(window=7, min_periods=1).mean()
    
    return df


def enrich_activity(df: pd.DataFrame, max_records: int | None = None) -> pd.DataFrame:
    """Enrich activity features with rolling averages and standardized metrics.
    
    Input columns expected: date, zepp_steps, zepp_active_kcal, apple_active_min, etc.
    Output: same + zepp_steps_7d, zepp_active_kcal_7d, zepp_steps_zscore, etc.
    """
    df = df.copy()
    
    # Respect MAX_RECORDS
    if max_records is not None and len(df) > max_records:
        df = df.iloc[:max_records].copy()
    
    # Ensure date column exists
    if "date" not in df.columns:
        return df
    
    # Find all numeric columns that could be enriched (vendor prefixed or unprefixed)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # For each numeric column, add rolling average and zscore
    for col in numeric_cols:
        if col == "date":
            continue
        
        # 7-day rolling average
        df = _rolling_mean_7d(df, col, f"{col}_7d")
        
        # Z-score
        df[f"{col}_zscore"] = _zscore(df[col])
    
    return df


def enrich_cardio(df: pd.DataFrame, max_records: int | None = None) -> pd.DataFrame:
    """Enrich cardiovascular features with rolling averages and standardized metrics.
    
    Input columns expected: date, zepp_hr_mean, zepp_hr_max, apple_heartrate, etc.
    Output: same + zepp_hr_mean_7d, zepp_hr_max_7d, zepp_hr_mean_zscore, etc.
    """
    df = df.copy()
    
    # Respect MAX_RECORDS
    if max_records is not None and len(df) > max_records:
        df = df.iloc[:max_records].copy()
    
    # Ensure date column exists
    if "date" not in df.columns:
        return df
    
    # Find all numeric columns that could be enriched
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # For each numeric column, add rolling average and zscore
    for col in numeric_cols:
        if col == "date":
            continue
        
        # 7-day rolling average
        df = _rolling_mean_7d(df, col, f"{col}_7d")
        
        # Z-score
        df[f"{col}_zscore"] = _zscore(df[col])
    
    return df


def enrich_sleep(df: pd.DataFrame, max_records: int | None = None) -> pd.DataFrame:
    """Enrich sleep features with rolling averages and standardized metrics.
    
    Input columns expected: date, zepp_slp_total_h, zepp_slp_deep_h, apple_sleep_hours, etc.
    Output: same + zepp_slp_total_h_7d, zepp_slp_deep_h_7d, zepp_slp_total_h_zscore, etc.
    """
    df = df.copy()
    
    # Respect MAX_RECORDS
    if max_records is not None and len(df) > max_records:
        df = df.iloc[:max_records].copy()
    
    # Ensure date column exists
    if "date" not in df.columns:
        return df
    
    # Find all numeric columns that could be enriched
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # For each numeric column, add rolling average and zscore
    for col in numeric_cols:
        if col == "date":
            continue
        
        # 7-day rolling average
        df = _rolling_mean_7d(df, col, f"{col}_7d")
        
        # Z-score
        df[f"{col}_zscore"] = _zscore(df[col])
    
    return df


def enrich_prejoin_run(snapshot_dir: Path | str, *, dry_run: bool = False, max_records: int | None = None) -> int:
    """Orchestrate pre-join enrichment across all domains and vendors.
    
    Args:
        snapshot_dir: Path to snapshot directory (e.g., data/etl/P000001/2025-11-06)
        dry_run: If True, skip writes and just report what would be done
        max_records: Limit records per vendor/variant (for testing)
    
    Returns:
        0 on success, 2 if no features found, 1 on error
    """
    snap = Path(snapshot_dir)
    print(f"INFO: enrich_prejoin_run start snapshot_dir={snap} dry_run={dry_run} max_records={max_records}")
    
    features_root = snap / "features"
    enriched_root = snap / "enriched" / "prejoin"
    
    if not features_root.exists():
        print(f"INFO: features_root not found at {features_root}")
        return 2
    
    # Collect all domain/vendor/variant combinations
    domain_items = []  # List of (domain, vendor, variant, features_csv_path)
    
    for domain_dir in features_root.iterdir():
        if not domain_dir.is_dir():
            continue
        
        domain = domain_dir.name
        
        # Find all vendor/variant combinations for this domain
        for features_csv in domain_dir.rglob("features_daily.csv"):
            try:
                # Extract vendor/variant from path: features/<domain>/<vendor>/<variant>/features_daily.csv
                rel_path = features_csv.relative_to(domain_dir)
                parts = rel_path.parts[:-1]  # Everything except 'features_daily.csv'
                
                if len(parts) >= 2:
                    vendor, variant = parts[0], parts[1]
                else:
                    # Legacy path or unexpected structure
                    continue
                
                domain_items.append((domain, vendor, variant, features_csv))
            except Exception as e:
                print(f"WARNING: could not parse path {features_csv}: {e}")
                continue
    
    if not domain_items:
        print("INFO: no features found to enrich")
        return 2
    
    print(f"INFO: discovered {len(domain_items)} domain/vendor/variant combinations to enrich")
    
    # Process each combination
    success_count = 0
    error_count = 0
    
    for domain, vendor, variant, features_csv in domain_items:
        try:
            # Read features
            df = pd.read_csv(features_csv)
            rows_before = len(df)
            
            # Apply domain-specific enrichment
            if domain == "activity":
                df_enriched = enrich_activity(df, max_records=max_records)
            elif domain == "cardio":
                df_enriched = enrich_cardio(df, max_records=max_records)
            elif domain == "sleep":
                df_enriched = enrich_sleep(df, max_records=max_records)
            else:
                print(f"  [{domain}/{vendor}/{variant}] unknown domain, skipping")
                continue
            
            rows_after = len(df_enriched)
            cols_added = len(df_enriched.columns) - len(df.columns)
            
            if dry_run:
                print(f"  [{domain}/{vendor}/{variant}] DRY RUN: {rows_after} rows, +{cols_added} columns")
                success_count += 1
                continue
            
            # Write to enriched/prejoin/<domain>/<vendor>/<variant>/enriched_<domain>.csv
            out_dir = enriched_root / domain / vendor / variant
            out_path = out_dir / f"enriched_{domain}.csv"
            
            _write_atomic_csv(df_enriched, out_path)
            print(f"  [{domain}/{vendor}/{variant}] wrote {rows_after} rows (+{cols_added} columns) to {out_path.relative_to(snap)}")
            success_count += 1
            
        except Exception as e:
            print(f"  [{domain}/{vendor}/{variant}] ERROR: {e}")
            traceback.print_exc()
            error_count += 1
    
    if dry_run:
        print(f"INFO: enrich_prejoin_run end (dry-run, would process {success_count} combinations)")
        return 0
    
    print(f"INFO: enrich_prejoin_run end (success={success_count}, errors={error_count})")
    return 0 if error_count == 0 else 1


if __name__ == "__main__":
    import argparse
    import sys
    
    parser = argparse.ArgumentParser(prog="prejoin_enricher", description="Pre-join per-domain enrichment")
    
    # Support both --pid and legacy --participant (like activity_from_extracted.py)
    parser.add_argument("--pid", required=False, help="Participant id (Pxxxxxx)")
    parser.add_argument("--participant", dest="pid", required=False, help="(legacy) participant id")
    parser.add_argument("--snapshot", required=True, help="Snapshot id YYYY-MM-DD")
    parser.add_argument("--dry-run", type=int, default=0, help="If 1 do a dry-run (no writes)")
    parser.add_argument("--max-records", type=int, default=None, help="Limit records per vendor/variant for testing")
    
    args = parser.parse_args()
    
    if not args.pid:
        parser.error("--pid or --participant required")
    
    # Resolve snapshot directory
    try:
        import src.etl_pipeline as etl
    except Exception:
        try:
            import etl_pipeline as etl  # type: ignore
        except Exception:
            etl = None
    
    if etl is None:
        print("ERROR: could not import etl_pipeline")
        sys.exit(1)
    
    snap_dir = etl.etl_snapshot_root(args.pid, args.snapshot)
    
    if not snap_dir.exists():
        print(f"ERROR: snapshot_dir does not exist: {snap_dir}")
        sys.exit(1)
    
    rc = enrich_prejoin_run(snap_dir, dry_run=bool(args.dry_run), max_records=args.max_records)
    sys.exit(rc)
