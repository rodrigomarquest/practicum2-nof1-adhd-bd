#!/usr/bin/env python3
"""
HR Daily Aggregation Consistency Check (QC Module)

Purpose:
--------
Verifies that the daily HR aggregates used by the pipeline are consistent
with the underlying event-level (record-by-record) heart rate data.

This is a STRUCTURAL QC CHECK - it re-aggregates event-level Parquet data
and compares it against the official daily CSV produced by stage_csv_aggregation.

Workflow:
---------
1. Load event-level HR Parquet cache: export_apple_hr_events.parquet
2. Re-compute daily aggregates (reference "ground truth")
3. Load official daily HR CSV: daily_cardio.csv
4. Compare on per-day basis
5. Output:
   - CSV of differences: hr_daily_aggregation_diff.csv
   - Markdown report: hr_daily_aggregation_consistency_report.md

Usage:
------
    python -m src.etl.hr_daily_aggregation_consistency_check P000001 2025-11-07 \\
        --start-date 2024-01-01 --end-date 2024-03-01

Author: PhD-level Data Engineer + Test Architect
Date: 2025-11-19
"""

import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Tuple
import time

import pandas as pd
import numpy as np

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)


# =============================================================================
# Path Resolution
# =============================================================================

def resolve_paths(participant_id: str, snapshot: str) -> dict:
    """
    Resolve canonical paths for HR data sources.
    
    Returns:
        dict with keys:
            - parquet_events: Path to event-level Parquet cache
            - parquet_daily: Path to daily aggregated Parquet cache
            - csv_daily: Path to canonical daily HR CSV used by pipeline
            - qc_dir: Path to QC output directory
    """
    base_dir = Path("data/etl") / participant_id / snapshot
    
    # Event-level Parquet cache (source of truth for re-aggregation)
    parquet_events = base_dir / "extracted" / "apple" / "apple_health_export" / ".cache" / "export_apple_hr_events.parquet"
    
    # Daily aggregated Parquet cache (fast loading, but we won't use for QC)
    parquet_daily = base_dir / "extracted" / "apple" / "apple_health_export" / ".cache" / "export_apple_hr_daily.parquet"
    
    # CANONICAL daily HR CSV used by downstream pipeline stages
    # This is the "fast" aggregation output from stage_csv_aggregation.py
    # Columns: date, hr_mean, hr_min, hr_max, hr_std, hr_samples
    csv_daily = base_dir / "extracted" / "apple" / "daily_cardio.csv"
    
    # QC output directory
    qc_dir = Path("data/ai") / participant_id / snapshot / "qc"
    qc_dir.mkdir(parents=True, exist_ok=True)
    
    return {
        "parquet_events": parquet_events,
        "parquet_daily": parquet_daily,
        "csv_daily": csv_daily,
        "qc_dir": qc_dir
    }


# =============================================================================
# Data Loading
# =============================================================================

def load_event_level_hr(parquet_path: Path, 
                        start_date: Optional[str] = None,
                        end_date: Optional[str] = None) -> pd.DataFrame:
    """
    Load event-level HR data from Parquet cache.
    
    Args:
        parquet_path: Path to export_apple_hr_events.parquet
        start_date: Optional start date (inclusive) YYYY-MM-DD
        end_date: Optional end date (inclusive) YYYY-MM-DD
    
    Returns:
        DataFrame with columns: timestamp, date, hr_value
        
    Raises:
        FileNotFoundError: If Parquet file doesn't exist
    """
    if not parquet_path.exists():
        raise FileNotFoundError(
            f"Event-level Parquet cache not found: {parquet_path}\n"
            f"Run 'make aggregate' first to generate the cache."
        )
    
    logger.info(f"Loading event-level HR from: {parquet_path}")
    start_load = time.time()
    
    df = pd.read_parquet(parquet_path)
    elapsed_load = time.time() - start_load
    
    logger.info(f"  ✓ Loaded {len(df):,} HR records in {elapsed_load:.2f}s")
    logger.info(f"  Columns: {df.columns.tolist()}")
    
    # Ensure date column is string (YYYY-MM-DD)
    if "date" not in df.columns:
        raise ValueError(f"Expected 'date' column in Parquet, got: {df.columns.tolist()}")
    
    # Filter by date range if provided
    if start_date or end_date:
        df_orig_len = len(df)
        
        if start_date:
            df = df[df["date"] >= start_date]
            logger.info(f"  Filtered by start_date >= {start_date}: {len(df):,} records")
        
        if end_date:
            df = df[df["date"] <= end_date]
            logger.info(f"  Filtered by end_date <= {end_date}: {len(df):,} records")
        
        logger.info(f"  Date filter: {df_orig_len:,} → {len(df):,} records ({len(df)/df_orig_len*100:.1f}%)")
    
    return df


def load_daily_hr_csv(csv_path: Path,
                      start_date: Optional[str] = None,
                      end_date: Optional[str] = None) -> pd.DataFrame:
    """
    Load canonical daily HR CSV produced by the pipeline.
    
    This is the "FAST" aggregation output from stage_csv_aggregation.py
    that is used by all downstream pipeline stages.
    
    Args:
        csv_path: Path to daily_cardio.csv
        start_date: Optional start date (inclusive) YYYY-MM-DD
        end_date: Optional end date (inclusive) YYYY-MM-DD
    
    Returns:
        DataFrame with columns: date, hr_mean, hr_min, hr_max, hr_std, hr_samples
        
    Raises:
        FileNotFoundError: If CSV doesn't exist
    """
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Daily HR CSV not found: {csv_path}\n"
            f"Run 'make aggregate' first to generate daily aggregates."
        )
    
    logger.info(f"Loading daily HR CSV from: {csv_path}")
    start_load = time.time()
    
    df = pd.read_csv(csv_path)
    elapsed_load = time.time() - start_load
    
    logger.info(f"  ✓ Loaded {len(df)} days in {elapsed_load:.2f}s")
    logger.info(f"  Columns: {df.columns.tolist()}")
    
    # Validate expected columns
    required_cols = ["date", "hr_mean", "hr_samples"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in daily CSV: {missing_cols}")
    
    # Ensure date is string for consistent merging
    df["date"] = df["date"].astype(str)
    
    # Filter by date range if provided
    if start_date or end_date:
        df_orig_len = len(df)
        
        if start_date:
            df = df[df["date"] >= start_date]
            logger.info(f"  Filtered by start_date >= {start_date}: {len(df)} days")
        
        if end_date:
            df = df[df["date"] <= end_date]
            logger.info(f"  Filtered by end_date <= {end_date}: {len(df)} days")
        
        logger.info(f"  Date filter: {df_orig_len} → {len(df)} days ({len(df)/df_orig_len*100:.1f}%)")
    
    return df


# =============================================================================
# Reference Aggregation (Ground Truth)
# =============================================================================

def compute_reference_daily_aggregates(df_events: pd.DataFrame) -> pd.DataFrame:
    """
    Re-compute daily HR aggregates from event-level data.
    
    This is the "ground truth" reference aggregation that we compare
    against the fast pipeline aggregation.
    
    Args:
        df_events: Event-level HR data with columns: timestamp, date, hr_value
    
    Returns:
        DataFrame with columns: date, ref_hr_n_records, ref_hr_mean, ref_hr_min, ref_hr_max, ref_hr_std
    """
    logger.info("Re-aggregating event-level HR to daily (reference ground truth)...")
    start_agg = time.time()
    
    # Group by date and compute statistics
    agg_results = df_events.groupby("date")["hr_value"].agg([
        ("ref_hr_n_records", "count"),
        ("ref_hr_mean", "mean"),
        ("ref_hr_min", "min"),
        ("ref_hr_max", "max"),
        ("ref_hr_std", "std")
    ]).reset_index()
    
    elapsed_agg = time.time() - start_agg
    
    logger.info(f"  ✓ Aggregated {len(agg_results)} days in {elapsed_agg:.2f}s")
    logger.info(f"  Date range: {agg_results['date'].min()} to {agg_results['date'].max()}")
    logger.info(f"  Total records: {agg_results['ref_hr_n_records'].sum():,}")
    logger.info(f"  Mean records/day: {agg_results['ref_hr_n_records'].mean():.1f}")
    
    return agg_results


# =============================================================================
# Comparison & Diff Calculation
# =============================================================================

def compute_diff_metrics(df_merged: pd.DataFrame) -> pd.DataFrame:
    """
    Compute difference metrics between fast and reference aggregations.
    
    Args:
        df_merged: Merged DataFrame with both fast_ and ref_ columns
    
    Returns:
        DataFrame with added diff columns and flags
    """
    logger.info("Computing difference metrics...")
    
    # Compute absolute differences
    df_merged["diff_hr_n_records"] = df_merged["fast_hr_samples"] - df_merged["ref_hr_n_records"]
    df_merged["diff_hr_mean"] = df_merged["fast_hr_mean"] - df_merged["ref_hr_mean"]
    df_merged["diff_hr_min"] = df_merged["fast_hr_min"] - df_merged["ref_hr_min"]
    df_merged["diff_hr_max"] = df_merged["fast_hr_max"] - df_merged["ref_hr_max"]
    df_merged["diff_hr_std"] = df_merged["fast_hr_std"] - df_merged["ref_hr_std"]
    
    # Compute relative differences (handle division by zero)
    df_merged["rel_diff_hr_mean"] = np.where(
        df_merged["ref_hr_mean"].abs() > 1e-6,
        df_merged["diff_hr_mean"].abs() / df_merged["ref_hr_mean"].abs(),
        0.0
    )
    
    # Define thresholds for "OK" vs "MISMATCH"
    # These thresholds are based on expected rounding/precision differences
    THRESHOLD_N_RECORDS = 5  # ±5 records difference is acceptable (edge cases at midnight)
    THRESHOLD_HR_MEAN_ABS = 1.0  # ±1 bpm absolute difference
    THRESHOLD_HR_MEAN_REL = 0.05  # 5% relative difference
    
    # Flag mismatches
    df_merged["flag_hr_n_records"] = np.where(
        df_merged["diff_hr_n_records"].abs() <= THRESHOLD_N_RECORDS,
        "OK",
        "MISMATCH"
    )
    
    df_merged["flag_hr_mean"] = np.where(
        (df_merged["diff_hr_mean"].abs() <= THRESHOLD_HR_MEAN_ABS) |
        (df_merged["rel_diff_hr_mean"] <= THRESHOLD_HR_MEAN_REL),
        "OK",
        "MISMATCH"
    )
    
    # Count mismatches
    n_mismatch_records = (df_merged["flag_hr_n_records"] == "MISMATCH").sum()
    n_mismatch_mean = (df_merged["flag_hr_mean"] == "MISMATCH").sum()
    
    logger.info(f"  Mismatches in hr_n_records: {n_mismatch_records}/{len(df_merged)} days ({n_mismatch_records/len(df_merged)*100:.2f}%)")
    logger.info(f"  Mismatches in hr_mean: {n_mismatch_mean}/{len(df_merged)} days ({n_mismatch_mean/len(df_merged)*100:.2f}%)")
    
    return df_merged


# =============================================================================
# Output Generation
# =============================================================================

def save_diff_csv(df_diff: pd.DataFrame, output_path: Path):
    """Save difference table to CSV."""
    df_diff.to_csv(output_path, index=False)
    logger.info(f"✓ Saved diff CSV: {output_path} ({len(df_diff)} days)")


def generate_markdown_report(
    df_diff: pd.DataFrame,
    paths: dict,
    participant_id: str,
    snapshot: str,
    start_date: Optional[str],
    end_date: Optional[str],
    output_path: Path
):
    """Generate comprehensive Markdown QC report."""
    
    report_lines = []
    
    # Header
    report_lines.append("# HR Daily Aggregation Consistency Report")
    report_lines.append("")
    report_lines.append(f"**Participant**: {participant_id}")
    report_lines.append(f"**Snapshot**: {snapshot}")
    report_lines.append(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("")
    
    # Date range
    actual_start = df_diff["date"].min()
    actual_end = df_diff["date"].max()
    report_lines.append(f"**Date Range**: {actual_start} to {actual_end}")
    if start_date:
        report_lines.append(f"  - Requested start: {start_date}")
    if end_date:
        report_lines.append(f"  - Requested end: {end_date}")
    report_lines.append("")
    
    # Data sources
    report_lines.append("## Data Sources")
    report_lines.append("")
    report_lines.append(f"- **Event-level Parquet**: `{paths['parquet_events']}`")
    report_lines.append(f"- **Daily CSV (canonical)**: `{paths['csv_daily']}`")
    report_lines.append("")
    
    # Summary statistics
    report_lines.append("## Summary Statistics")
    report_lines.append("")
    report_lines.append(f"- **Total days analyzed**: {len(df_diff)}")
    report_lines.append("")
    
    # HR N Records
    n_mismatch_records = (df_diff["flag_hr_n_records"] == "MISMATCH").sum()
    pct_ok_records = (1 - n_mismatch_records / len(df_diff)) * 100
    report_lines.append("### HR Record Count (`hr_n_records`)")
    report_lines.append("")
    report_lines.append(f"- **OK**: {len(df_diff) - n_mismatch_records}/{len(df_diff)} days ({pct_ok_records:.2f}%)")
    report_lines.append(f"- **MISMATCH**: {n_mismatch_records}/{len(df_diff)} days ({n_mismatch_records/len(df_diff)*100:.2f}%)")
    report_lines.append(f"- **Max absolute diff**: {df_diff['diff_hr_n_records'].abs().max():.0f} records")
    report_lines.append(f"- **Mean absolute diff**: {df_diff['diff_hr_n_records'].abs().mean():.2f} records")
    report_lines.append(f"- **Median absolute diff**: {df_diff['diff_hr_n_records'].abs().median():.2f} records")
    report_lines.append("")
    
    # HR Mean
    n_mismatch_mean = (df_diff["flag_hr_mean"] == "MISMATCH").sum()
    pct_ok_mean = (1 - n_mismatch_mean / len(df_diff)) * 100
    report_lines.append("### HR Mean (`hr_mean`)")
    report_lines.append("")
    report_lines.append(f"- **OK**: {len(df_diff) - n_mismatch_mean}/{len(df_diff)} days ({pct_ok_mean:.2f}%)")
    report_lines.append(f"- **MISMATCH**: {n_mismatch_mean}/{len(df_diff)} days ({n_mismatch_mean/len(df_diff)*100:.2f}%)")
    report_lines.append(f"- **Max absolute diff**: {df_diff['diff_hr_mean'].abs().max():.2f} bpm")
    report_lines.append(f"- **Mean absolute diff**: {df_diff['diff_hr_mean'].abs().mean():.4f} bpm")
    report_lines.append(f"- **Median absolute diff**: {df_diff['diff_hr_mean'].abs().median():.4f} bpm")
    report_lines.append(f"- **Max relative diff**: {df_diff['rel_diff_hr_mean'].max() * 100:.2f}%")
    report_lines.append("")
    
    # Top discrepancies
    report_lines.append("## Top 10 Days with Largest Discrepancies")
    report_lines.append("")
    
    # Top by hr_n_records
    top_records = df_diff.nlargest(10, "diff_hr_n_records", keep="first")[
        ["date", "fast_hr_samples", "ref_hr_n_records", "diff_hr_n_records", "flag_hr_n_records"]
    ]
    report_lines.append("### By HR Record Count")
    report_lines.append("")
    report_lines.append("| Date | Fast | Ref | Diff | Flag |")
    report_lines.append("|------|------|-----|------|------|")
    for _, row in top_records.iterrows():
        report_lines.append(f"| {row['date']} | {row['fast_hr_samples']:.0f} | {row['ref_hr_n_records']:.0f} | {row['diff_hr_n_records']:.0f} | {row['flag_hr_n_records']} |")
    report_lines.append("")
    
    # Top by hr_mean (absolute value)
    df_diff["abs_diff_hr_mean"] = df_diff["diff_hr_mean"].abs()
    top_mean = df_diff.nlargest(10, "abs_diff_hr_mean", keep="first")[
        ["date", "fast_hr_mean", "ref_hr_mean", "diff_hr_mean", "flag_hr_mean"]
    ]
    report_lines.append("### By HR Mean")
    report_lines.append("")
    report_lines.append("| Date | Fast (bpm) | Ref (bpm) | Diff (bpm) | Flag |")
    report_lines.append("|------|------------|-----------|------------|------|")
    for _, row in top_mean.iterrows():
        report_lines.append(f"| {row['date']} | {row['fast_hr_mean']:.2f} | {row['ref_hr_mean']:.2f} | {row['diff_hr_mean']:.2f} | {row['flag_hr_mean']} |")
    report_lines.append("")
    
    # Interpretation
    report_lines.append("## Interpretation")
    report_lines.append("")
    
    if pct_ok_records >= 98.0 and pct_ok_mean >= 95.0:
        report_lines.append(f"✅ **High Consistency**: For `hr_n_records`, {pct_ok_records:.1f}% of days are within ±5 records difference. ")
        report_lines.append(f"For `hr_mean`, {pct_ok_mean:.1f}% of days are within ±1 bpm or 5% relative difference.")
        report_lines.append("")
        report_lines.append("Remaining discrepancies are likely due to:")
        report_lines.append("- Timestamp edge cases at midnight boundaries")
        report_lines.append("- Timezone conversion rounding")
        report_lines.append("- Floating-point precision in aggregation")
    else:
        report_lines.append(f"⚠️ **Moderate Consistency**: `hr_n_records` OK rate: {pct_ok_records:.1f}%, `hr_mean` OK rate: {pct_ok_mean:.1f}%")
        report_lines.append("")
        report_lines.append("Some days show larger discrepancies. Possible causes:")
        report_lines.append("- Days with very few HR samples (higher sensitivity to rounding)")
        report_lines.append("- Timezone handling differences between fast and reference aggregation")
        report_lines.append("- Data quality issues on specific days")
    
    report_lines.append("")
    
    # Conclusion
    report_lines.append("## Conclusion")
    report_lines.append("")
    
    if pct_ok_records >= 98.0 and pct_ok_mean >= 97.0:
        report_lines.append("✅ **Daily aggregation from event-level Parquet appears highly consistent with the fast ETL pipeline within defined tolerances.**")
        report_lines.append("")
        report_lines.append("The fast binary regex aggregation in `stage_csv_aggregation.py` is trustworthy for downstream analysis.")
    elif pct_ok_records >= 95.0 and pct_ok_mean >= 90.0:
        report_lines.append("⚠️ **Daily aggregation shows acceptable consistency, but some discrepancies warrant investigation.**")
        report_lines.append("")
        report_lines.append("Review the top discrepancy days to understand edge cases. Consider tightening tolerance thresholds if needed.")
    else:
        report_lines.append("❌ **Significant discrepancies detected. Further investigation of the fast ETL's HR aggregation is recommended.**")
        report_lines.append("")
        report_lines.append("Examine the following:")
        report_lines.append("- Timezone handling in `stage_csv_aggregation.py` binary regex parsing")
        report_lines.append("- Date truncation logic (midnight boundaries)")
        report_lines.append("- Data type conversions and rounding")
    
    report_lines.append("")
    report_lines.append("---")
    report_lines.append("")
    report_lines.append(f"*Generated by `hr_daily_aggregation_consistency_check.py` on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")
    
    # Write report
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    
    logger.info(f"✓ Saved Markdown report: {output_path}")


# =============================================================================
# Main Workflow
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="HR Daily Aggregation Consistency Check (QC Module)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m src.etl.hr_daily_aggregation_consistency_check P000001 2025-11-07
  python -m src.etl.hr_daily_aggregation_consistency_check P000001 2025-11-07 --start-date 2024-01-01 --end-date 2024-03-01
        """
    )
    
    parser.add_argument("participant_id", help="Participant ID (e.g., P000001)")
    parser.add_argument("snapshot", help="Snapshot date (YYYY-MM-DD)")
    parser.add_argument("--start-date", help="Start date filter (YYYY-MM-DD, inclusive)", default=None)
    parser.add_argument("--end-date", help="End date filter (YYYY-MM-DD, inclusive)", default=None)
    
    args = parser.parse_args()
    
    logger.info("="*80)
    logger.info("HR DAILY AGGREGATION CONSISTENCY CHECK (QC MODULE)")
    logger.info("="*80)
    logger.info(f"Participant: {args.participant_id}")
    logger.info(f"Snapshot: {args.snapshot}")
    if args.start_date:
        logger.info(f"Start date: {args.start_date}")
    if args.end_date:
        logger.info(f"End date: {args.end_date}")
    logger.info("")
    
    # Resolve paths
    paths = resolve_paths(args.participant_id, args.snapshot)
    
    logger.info("Data sources:")
    logger.info(f"  Event-level Parquet: {paths['parquet_events']}")
    logger.info(f"  Daily CSV (canonical): {paths['csv_daily']}")
    logger.info(f"  QC output dir: {paths['qc_dir']}")
    logger.info("")
    
    # Load data
    try:
        # 1. Load event-level HR
        df_events = load_event_level_hr(paths["parquet_events"], args.start_date, args.end_date)
        
        # 2. Re-aggregate to daily (reference ground truth)
        df_ref = compute_reference_daily_aggregates(df_events)
        
        # 3. Load daily CSV (fast aggregation)
        df_fast = load_daily_hr_csv(paths["csv_daily"], args.start_date, args.end_date)
        
    except FileNotFoundError as e:
        logger.error(f"✗ {e}")
        logger.error("Make sure to run 'make aggregate' first to generate data.")
        return 1
    except Exception as e:
        logger.error(f"✗ Failed to load data: {e}", exc_info=True)
        return 1
    
    # Merge and compute diffs
    logger.info("")
    logger.info("Merging fast and reference aggregations...")
    
    # Rename columns for clarity
    df_fast = df_fast.rename(columns={
        "hr_mean": "fast_hr_mean",
        "hr_min": "fast_hr_min",
        "hr_max": "fast_hr_max",
        "hr_std": "fast_hr_std",
        "hr_samples": "fast_hr_samples"
    })
    
    # Merge on date (outer join to catch any missing days)
    df_merged = df_fast.merge(df_ref, on="date", how="outer", suffixes=("_fast", "_ref"))
    
    logger.info(f"  Fast days: {len(df_fast)}")
    logger.info(f"  Reference days: {len(df_ref)}")
    logger.info(f"  Merged days: {len(df_merged)}")
    logger.info("")
    
    # Compute difference metrics
    df_diff = compute_diff_metrics(df_merged)
    
    # Save outputs
    logger.info("")
    logger.info("Saving QC outputs...")
    
    diff_csv_path = paths["qc_dir"] / "hr_daily_aggregation_diff.csv"
    save_diff_csv(df_diff, diff_csv_path)
    
    report_md_path = paths["qc_dir"] / "hr_daily_aggregation_consistency_report.md"
    generate_markdown_report(
        df_diff,
        paths,
        args.participant_id,
        args.snapshot,
        args.start_date,
        args.end_date,
        report_md_path
    )
    
    logger.info("")
    logger.info("="*80)
    logger.info("QC CHECK COMPLETE")
    logger.info("="*80)
    logger.info(f"Diff CSV: {diff_csv_path}")
    logger.info(f"Report: {report_md_path}")
    logger.info("")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
