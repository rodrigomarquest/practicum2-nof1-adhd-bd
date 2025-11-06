# -*- coding: utf-8 -*-
"""Post-join enrichment: cross-domain enrichments applied to joined features.

Fase 3 (Enriched/Global):
- Read canonical joined_features_daily.csv
- Apply cross-domain enrichments (correlations, ratios, aggregations, NAs handling)
- Write enriched/<domain>/enriched_<domain>.csv split per domain
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from typing import Optional, Any


def _rolling_corr_7d(df: pd.DataFrame, col1: str, col2: str, new_col: str) -> pd.DataFrame:
    """Compute 7-day rolling correlation between two columns.
    
    Handles NaN gracefully, preserves date order.
    """
    if col1 not in df.columns or col2 not in df.columns:
        return df
    
    # Ensure date is sorted
    if "date" in df.columns:
        try:
            df = df.sort_values("date")
        except Exception:
            pass
    
    # Compute rolling correlation
    df[new_col] = df[col1].rolling(window=7, min_periods=1).corr(df[col2])
    
    return df


def _ratio(df: pd.DataFrame, numerator: str, denominator: str, new_col: str) -> pd.DataFrame:
    """Compute ratio: numerator / denominator, handling division by zero.
    
    Returns NaN where denominator is 0 or NaN.
    """
    if numerator not in df.columns or denominator not in df.columns:
        return df
    
    df[new_col] = df[numerator] / df[denominator]
    # Replace inf with NaN
    df[new_col] = df[new_col].replace([np.inf, -np.inf], np.nan)
    
    return df


def _handle_missing_domains(df: pd.DataFrame, max_records: int | None = None) -> pd.DataFrame:
    """Fill missing domain data with linear interpolation + forward fill.
    
    Strategy: Try to infer missing dates from date column, interpolate, then ffill residual gaps.
    This is a light touch - only for daily metrics when we have sparse coverage.
    """
    df = df.copy()
    
    if max_records is not None and len(df) > max_records:
        df = df.iloc[:max_records].copy()
    
    if "date" not in df.columns:
        return df
    
    # Ensure sorted by date
    try:
        df = df.sort_values("date").reset_index(drop=True)
    except Exception:
        pass
    
    # For numeric columns (except derived metrics), try interpolation
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for col in numeric_cols:
        # Skip already derived columns
        if col.endswith(("_7d", "_zscore", "_7d_corr", "_ratio")):
            continue
        
        # Linear interpolation if we have at least 2 non-NaN values
        if df[col].notna().sum() >= 2:
            df[col] = df[col].interpolate(method="linear", limit_direction="both")
        
        # Forward fill any remaining NaNs
        df[col] = df[col].ffill()
    
    return df


def enrich_activity_postjoin(df: pd.DataFrame, max_records: int | None = None) -> pd.DataFrame:
    """Apply cross-domain enrichments to activity rows (post-join).
    
    Enhancements:
    - Correlations with cardio (act_steps vs hr_mean)
    - Handle missing dates (interpolate)
    """
    df = df.copy()
    
    if max_records is not None and len(df) > max_records:
        df = df.iloc[:max_records].copy()
    
    # 7-day rolling correlation: act_steps vs hr_mean
    if "act_steps" in df.columns and "hr_mean" in df.columns:
        df = _rolling_corr_7d(df, "act_steps", "hr_mean", "act_steps_vs_hr_7d_corr")
    
    # Handle missing by interpolation
    df = _handle_missing_domains(df, max_records=None)
    
    return df


def enrich_cardio_postjoin(df: pd.DataFrame, max_records: int | None = None) -> pd.DataFrame:
    """Apply cross-domain enrichments to cardio rows (post-join).
    
    Enhancements:
    - Correlations with activity (hr_mean vs act_steps)
    - HR variability ratio (hr_std / hr_mean) when available
    - Handle missing dates
    """
    df = df.copy()
    
    if max_records is not None and len(df) > max_records:
        df = df.iloc[:max_records].copy()
    
    # 7-day rolling correlation: hr_mean vs act_steps
    if "hr_mean" in df.columns and "act_steps" in df.columns:
        df = _rolling_corr_7d(df, "hr_mean", "act_steps", "hr_mean_vs_act_7d_corr")
    
    # HR variability ratio: hr_std / hr_mean (when both available)
    if "hr_std" in df.columns and "hr_mean" in df.columns:
        df = _ratio(df, "hr_std", "hr_mean", "hr_variability_ratio")
    
    # Handle missing by interpolation
    df = _handle_missing_domains(df, max_records=None)
    
    return df


def enrich_sleep_postjoin(df: pd.DataFrame, max_records: int | None = None) -> pd.DataFrame:
    """Apply cross-domain enrichments to sleep rows (post-join).
    
    Enhancements:
    - Ratio: sleep_total_h / activity (act_active_min converted to hours)
    - Handle missing dates
    """
    df = df.copy()
    
    if max_records is not None and len(df) > max_records:
        df = df.iloc[:max_records].copy()
    
    # Sleep efficiency: sleep_total_h / (exercise_minutes / 60) when both available
    if "sleep_total_h" in df.columns and "act_active_min" in df.columns:
        # Convert exercise_min to hours, then ratio
        df = _ratio(df, "sleep_total_h", "act_active_min", "sleep_activity_ratio")
    
    # Handle missing by interpolation
    df = _handle_missing_domains(df, max_records=None)
    
    return df


def enrich_postjoin_run(snapshot_dir: Path | str, *, dry_run: bool = False, max_records: int | None = None) -> int:
    """Orchestrate post-join enrichment: read joined, apply cross-domain enhancements, write back per domain.
    
    Input:
    - <snapshot_dir>/joined/joined_features_daily.csv (with coalesced columns)
    
    Output:
    - <snapshot_dir>/enriched/postjoin/<domain>/enriched_<domain>.csv (one per domain present in joined)
    
    Flow:
    1. Read joined CSV
    2. Group by source_domain (activity, cardio, sleep)
    3. Apply domain-specific postjoin enrichments
    4. Write enriched per domain
    
    Returns:
    - 0: success (1+ domains enriched)
    - 2: no joined CSV found
    - 1: error
    """
    snap = Path(snapshot_dir)
    joined_path = snap / "joined" / "joined_features_daily.csv"
    
    print(f"INFO: enrich_postjoin_run start snapshot_dir={snap} dry_run={dry_run} max_records={max_records}")
    
    if not joined_path.exists():
        print(f"WARNING: joined CSV not found: {joined_path}")
        return 2
    
    try:
        joined_df = pd.read_csv(joined_path, parse_dates=["date"])
    except Exception as e:
        print(f"ERROR: failed to read joined CSV: {e}")
        return 1
    
    if joined_df.empty:
        print("WARNING: joined CSV is empty")
        return 2
    
    # Group by source_domain (if present), otherwise infer from columns
    if "source_domain" in joined_df.columns:
        domains_in_joined = list(joined_df["source_domain"].unique())
    else:
        # Infer from presence of domain-specific columns
        domains_in_joined = []
        if any(c.startswith(("act_", "apple_steps", "zepp_act_")) for c in joined_df.columns):
            domains_in_joined.append("activity")
        if any(c.startswith(("hr_", "apple_hr_", "zepp_hr_")) for c in joined_df.columns):
            domains_in_joined.append("cardio")
        if any(c.startswith(("sleep_", "apple_slp_", "zepp_slp_")) for c in joined_df.columns):
            domains_in_joined.append("sleep")
    
    if len(domains_in_joined) == 0:
        print("WARNING: could not detect domains in joined CSV")
        return 2
    
    success_count = 0
    error_count = 0
    
    print(f"INFO: discovered domains: {domains_in_joined}")
    
    for domain_name in domains_in_joined:
        try:
            # Filter rows for this domain
            if "source_domain" in joined_df.columns:
                domain_df = joined_df[joined_df["source_domain"] == domain_name].copy()
            else:
                # All rows contain all columns; just apply domain-specific enrichments
                domain_df = joined_df.copy()
            
            if domain_df.empty:
                print(f"  [{domain_name}] no rows found")
                continue
            
            # Apply domain-specific enrichments
            if domain_name == "activity":
                domain_df = enrich_activity_postjoin(domain_df, max_records=max_records)
            elif domain_name == "cardio":
                domain_df = enrich_cardio_postjoin(domain_df, max_records=max_records)
            elif domain_name == "sleep":
                domain_df = enrich_sleep_postjoin(domain_df, max_records=max_records)
            
            if dry_run:
                print(f"  [{domain_name}] (dry-run) would write {len(domain_df)} rows")
                success_count += 1
                continue
            
            # Write enriched per domain
            out_dir = snap / "enriched" / "postjoin" / domain_name
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"enriched_{domain_name}.csv"
            
            # Convert datetime64 to string before write
            if "date" in domain_df.columns:
                try:
                    domain_df["date"] = pd.to_datetime(domain_df["date"]).dt.strftime("%Y-%m-%d")
                except Exception:
                    pass
            
            domain_df.to_csv(out_path, index=False)
            
            # Count new columns (those ending with _corr, _ratio, or previously from prejoin)
            new_cols = [c for c in domain_df.columns if any(c.endswith(s) for s in ["_corr", "_ratio"])]
            print(f"  [{domain_name}] wrote {len(domain_df)} rows (+{len(new_cols)} enriched cols)")
            
            success_count += 1
        
        except Exception as e:
            print(f"  [{domain_name}] ERROR: {e}")
            error_count += 1
    
    if success_count == 0:
        print("INFO: enrich_postjoin_run end (no domains processed)")
        return 2
    
    print(f"INFO: enrich_postjoin_run end (success={success_count}, errors={error_count})")
    return 0 if error_count == 0 else 1


if __name__ == "__main__":
    import argparse
    import sys
    
    parser = argparse.ArgumentParser(
        prog="postjoin_enricher",
        description="Post-join cross-domain enrichment"
    )
    
    parser.add_argument("--pid", required=False, help="Participant id (Pxxxxxx)")
    parser.add_argument("--participant", dest="pid", required=False, help="(legacy) participant id")
    parser.add_argument("--snapshot", required=True, help="Snapshot id YYYY-MM-DD")
    parser.add_argument("--dry-run", type=int, default=0, help="If 1 do a dry-run (no writes)")
    parser.add_argument("--max-records", type=int, default=None, help="Limit records per domain for testing")
    
    args = parser.parse_args()
    
    if not args.pid:
        parser.error("--pid or --participant required")
    
    try:
        import src.etl_pipeline as etl
    except Exception:
        try:
            import etl_pipeline as etl
        except Exception:
            etl = None
    
    if etl is None:
        print("ERROR: could not import etl_pipeline")
        sys.exit(1)
    
    snap_dir = etl.etl_snapshot_root(args.pid, args.snapshot)
    
    if not snap_dir.exists():
        print(f"ERROR: snapshot_dir does not exist: {snap_dir}")
        sys.exit(1)
    
    rc = enrich_postjoin_run(snap_dir, dry_run=bool(args.dry_run), max_records=args.max_records)
    sys.exit(rc)
