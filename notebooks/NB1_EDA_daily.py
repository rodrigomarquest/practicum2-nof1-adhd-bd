#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
NB1_EDA_daily.py
================

Non-interactive Python version of NB1_EDA_daily.ipynb.

Purpose: Validate daily ETL outputs from `joined_features_daily.csv` and surface
actionable insights for researcher/participant.

Usage:
    python NB1_EDA_daily.py [--pid P000001] [--snapshot auto]

Inputs:
    - data/etl/<PID>/<SNAPSHOT>/joined/joined_features_daily.csv (primary source of truth)
    - enriched/prejoin/**/enriched_*.csv (enriched features)

Outputs:
    - reports/nb1_eda_summary.md — Human-readable summary
    - reports/nb1_feature_stats.csv — Per-column descriptive statistics
    - reports/plots/*.png — Inline visualizations
    - reports/nb1_manifest.json — Metadata & artifact manifest
    - latest/ — Mirror with symlinks for quick access

Version: 1.0 (exported from NB1_EDA_daily.ipynb)
Date: 2025-11-07
Environment: Offline, no internet calls, fully reproducible
"""

from __future__ import annotations

import sys
import argparse
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path
import json
from datetime import datetime
import textwrap
from collections import defaultdict
from scipy.stats import spearmanr
import warnings
import shutil
import logging

warnings.filterwarnings('ignore')

# ============================================================================
# LOGGING SETUP
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='[NB1] %(asctime)s — %(levelname)s — %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Pandas display options
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.float_format', '{:.2f}'.format)


# ============================================================================
# SECTION 0: ARGUMENT PARSING & INITIALIZATION
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description='NB1_EDA_daily: Non-interactive EDA for joined features',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python NB1_EDA_daily.py
  python NB1_EDA_daily.py --pid P000001 --snapshot 2025-10-15
  python NB1_EDA_daily.py --pid P000002 --snapshot auto
        """
    )
    parser.add_argument('--pid', default='P000001', help='Participant ID (default: P000001)')
    parser.add_argument('--snapshot', default='auto', help='Snapshot date or "auto" (default: auto)')
    parser.add_argument('--repo-root', default=None, help='Repository root (auto-detect if not provided)')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    return parser.parse_args()


def initialize_paths(pid: str, snapshot: str, repo_root: Path | None = None) -> tuple[Path, Path, Path, Path, str]:
    """Initialize and validate all paths."""
    # Resolve repo root
    if repo_root is None:
        # Try to auto-detect: go up from notebooks/ to project root
        notebook_dir = Path(__file__).parent
        repo_root = notebook_dir.parent
    else:
        repo_root = Path(repo_root)

    logger.info(f"Notebook dir: {notebook_dir}")
    logger.info(f"Repo root: {repo_root}")

    # Base paths
    base = repo_root / "data" / "etl" / pid
    out = repo_root / "reports"
    plots = out / "plots"
    latest = repo_root / "latest"

    # Resolve snapshot
    if snapshot == "auto":
        if not base.exists():
            raise ValueError(f"BASE path {base} does not exist")
        snapshots = sorted([d.name for d in base.iterdir() if d.is_dir()])
        if not snapshots:
            raise ValueError(f"No snapshots found under {base}")
        snapshot = snapshots[-1]
        logger.info(f"Resolved SNAPSHOT='auto' → '{snapshot}'")
    else:
        logger.info(f"Using SNAPSHOT='{snapshot}'")

    # Define JOINED path
    joined = base / snapshot / "joined" / "joined_features_daily.csv"

    # Create output directories
    out.mkdir(parents=True, exist_ok=True)
    plots.mkdir(parents=True, exist_ok=True)
    latest.mkdir(parents=True, exist_ok=True)

    # Verify JOINED exists
    if not joined.exists():
        raise FileNotFoundError(f"JOINED file not found: {joined}")

    logger.info(f"✓ Config ready | PID={pid} | SNAPSHOT={snapshot} | OUT={out}")

    return base, out, plots, latest, snapshot


# ============================================================================
# SECTION 0B: EXTRACTION VERIFICATION
# ============================================================================

def check_extraction(base: Path, pid: str, snapshot: str) -> dict:
    """
    Verify ETL extraction status for Apple INAPP and Zepp CLOUD.
    
    Returns status dict with structure for each component.
    """
    logger.info("=" * 80)
    logger.info(f"EXTRACTION VERIFICATION: {pid} / {snapshot}")
    logger.info("=" * 80)
    
    status = {
        'apple_inapp': {},
        'zepp_cloud': {},
        'joined_features': {}
    }
    
    extracted = base / snapshot / "extracted"
    
    # --- APPLE INAPP ---
    apple_dir = extracted / "apple" / "inapp"
    logger.info("\n📱 APPLE INAPP:")
    
    if apple_dir.exists():
        status['apple_inapp']['exists'] = True
        logger.info(f"  ✓ Directory exists: {apple_dir}")
        
        # Check for export.xml
        export_xml = apple_dir / "apple_health_export" / "export.xml"
        if export_xml.exists():
            size_mb = export_xml.stat().st_size / (1024 * 1024)
            status['apple_inapp']['export_xml'] = True
            logger.info(f"  ✓ export.xml found: {size_mb:.1f} MB")
        else:
            status['apple_inapp']['export_xml'] = False
            logger.warning(f"  ✗ export.xml NOT found")
        
        # Check for CSV files
        csv_files = list(apple_dir.glob("HKQuantityType*.csv"))
        status['apple_inapp']['csv_count'] = len(csv_files)
        logger.info(f"  ✓ CSV files extracted: {len(csv_files)}")
        if csv_files:
            sample = [f.name for f in csv_files[:3]]
            logger.info(f"    Sample: {sample}")
    else:
        status['apple_inapp']['exists'] = False
        logger.warning(f"  ✗ Directory NOT found: {apple_dir}")
    
    # --- ZEPP CLOUD ---
    zepp_dir = extracted / "zepp" / "cloud"
    logger.info("\n⌚ ZEPP CLOUD:")
    
    if zepp_dir.exists():
        status['zepp_cloud']['exists'] = True
        logger.info(f"  ✓ Directory exists: {zepp_dir}")
        
        # Check each subdirectory
        domains = ['HEARTRATE_AUTO', 'SLEEP', 'ACTIVITY_STAGE', 'ACTIVITY_MINUTE']
        for domain in domains:
            domain_dir = zepp_dir / domain
            if domain_dir.exists():
                csv_count = len(list(domain_dir.glob("*.csv")))
                status['zepp_cloud'][domain] = csv_count
                logger.info(f"  ✓ {domain}: {csv_count} CSV file(s)")
            else:
                status['zepp_cloud'][domain] = 0
                logger.warning(f"  ✗ {domain}: NOT found")
    else:
        status['zepp_cloud']['exists'] = False
        logger.warning(f"  ✗ Directory NOT found: {zepp_dir}")
    
    # --- JOINED FEATURES ---
    joined = base / snapshot / "joined" / "joined_features_daily.csv"
    logger.info("\n📊 JOINED FEATURES:")
    
    if joined.exists():
        status['joined_features']['daily'] = True
        size_mb = joined.stat().st_size / (1024 * 1024)
        logger.info(f"  ✓ joined_features_daily.csv: {size_mb:.1f} MB")
        
        try:
            df = pd.read_csv(joined, nrows=1)
            n_cols = len(df.columns)
            status['joined_features']['daily_columns'] = n_cols
            logger.info(f"    Columns: {n_cols}")
            first_cols = list(df.columns[:5])
            logger.info(f"    First columns: {first_cols}")
        except Exception as e:
            logger.warning(f"    Could not read: {e}")
    else:
        status['joined_features']['daily'] = False
        logger.warning(f"  ✗ joined_features_daily.csv NOT found")
    
    # --- BIOMARKERS ---
    biomarkers = base / snapshot / "joined" / "joined_features_daily_biomarkers.csv"
    logger.info("\n🧬 BIOMARKERS:")
    
    if biomarkers.exists():
        status['joined_features']['biomarkers'] = True
        size_mb = biomarkers.stat().st_size / (1024 * 1024)
        logger.info(f"  ✓ joined_features_daily_biomarkers.csv: {size_mb:.1f} MB")
        
        try:
            df = pd.read_csv(biomarkers, nrows=1)
            biomarker_cols = [c for c in df.columns if any(x in c.lower() for x in ['hrv', 'sleep', 'activity', 'circadian'])]
            logger.info(f"    Biomarker columns: {len(biomarker_cols)}")
            logger.info(f"    Sample: {biomarker_cols[:5]}")
        except Exception as e:
            logger.warning(f"    Could not read: {e}")
    else:
        status['joined_features']['biomarkers'] = False
        logger.info(f"  ⏳ Biomarkers not yet extracted (run: make biomarkers)")
    
    logger.info("\n" + "=" * 80)
    return status


# ============================================================================
# SECTION 1: DATA LOADING & VALIDATION
# ============================================================================

def load_and_validate_data(joined_path: Path) -> tuple[pd.DataFrame, int, int, pd.Timestamp, pd.Timestamp, int]:
    """Load, validate, and prepare data."""
    logger.info(f"Loading data from {joined_path}")

    # Load with date parsing
    df = pd.read_csv(joined_path, parse_dates=['date'])

    # Normalize date to midnight
    df['date'] = pd.to_datetime(df['date']).dt.normalize()

    # Sort by date
    df = df.sort_values('date').reset_index(drop=True)

    # Drop duplicate rows (keep latest by index)
    n_before = len(df)
    df = df.drop_duplicates(subset=['date'], keep='last').reset_index(drop=True)
    n_after = len(df)
    if n_before > n_after:
        logger.warning(f"Removed {n_before - n_after} duplicate date rows")

    # Basic info
    n_rows, n_cols = df.shape
    date_min = df['date'].min()
    date_max = df['date'].max()
    date_range = (date_max - date_min).days + 1

    logger.info(f"Loaded {n_rows} rows × {n_cols} cols | Date range: {date_min.date()} to {date_max.date()} ({date_range} days)")
    logger.info(f"First rows:\n{df.head()}")
    logger.info(f"Last rows:\n{df.tail()}")

    # Verify monotonic dates
    if not df['date'].is_monotonic_increasing:
        logger.warning("Dates are not monotonic! Sorting...")
        df = df.sort_values('date').reset_index(drop=True)
    else:
        logger.info("✓ Dates are monotonic")

    return df, n_rows, n_cols, date_min, date_max, date_range


def coerce_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce numeric columns."""
    logger.info("Coercing numeric columns...")
    numeric_cols = df.select_dtypes(include=['object', 'float64', 'int64']).columns
    for col in numeric_cols:
        if col not in ['date', 'label', 'label_source', 'segment_id', 'source_domain']:
            try:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            except Exception:
                pass
    logger.info("✓ Numeric coercion complete")
    return df


# ============================================================================
# SECTION 2: COLUMN INVENTORY
# ============================================================================

def inventory_columns(df: pd.DataFrame) -> dict:
    """Inventory all columns by domain."""
    logger.info("=== COLUMN INVENTORY ===")

    cols_activity_apple = ['apple_steps', 'apple_distance_m', 'apple_active_kcal', 'apple_exercise_min',
                           'apple_stand_hours', 'apple_move_goal_kcal', 'apple_exercise_goal_min',
                           'apple_stand_goal_hours', 'apple_rings_close_move', 'apple_rings_close_exercise',
                           'apple_rings_close_stand']
    cols_activity_zepp = ['zepp_act_steps', 'zepp_act_distance_km', 'zepp_act_cal_active', 'zepp_act_cal_total',
                          'zepp_act_sedentary_min', 'zepp_act_stand_hours', 'zepp_act_sport_sessions',
                          'zepp_act_exercise_min', 'zepp_act_score_daily']
    cols_activity_coalesced = ['act_steps', 'act_active_min']
    cols_cardio = ['apple_hr_mean', 'apple_hr_std', 'apple_n_hr', 'zepp_hr_mean', 'zepp_hr_std',
                   'zepp_n_hr', 'hr_mean', 'hr_std', 'n_hr']
    cols_sleep = ['zepp_slp_total_h', 'zepp_slp_deep_h', 'zepp_slp_light_h', 'zepp_slp_rem_h', 'sleep_total_h']
    cols_label = ['label', 'label_source']
    cols_segment = ['segment_id']

    domains = {
        'Activity (Apple)': cols_activity_apple,
        'Activity (Zepp)': cols_activity_zepp,
        'Activity (Coalesced)': cols_activity_coalesced,
        'Cardio': cols_cardio,
        'Sleep': cols_sleep,
    }

    for domain_name, cols in domains.items():
        present = [c for c in cols if c in df.columns]
        missing = [c for c in cols if c not in df.columns]
        logger.info(f"{domain_name}:")
        logger.info(f"  Present ({len(present)}): {present[:5]}{'...' if len(present) > 5 else ''}")
        if missing:
            logger.info(f"  Missing ({len(missing)}): {missing[:5]}{'...' if len(missing) > 5 else ''}")

    # Enriched columns
    enriched_7d = [c for c in df.columns if '_7d' in c]
    enriched_zscore = [c for c in df.columns if '_zscore' in c]
    logger.info(f"Enriched (*_7d): {len(enriched_7d)} columns {enriched_7d[:3]}{'...' if len(enriched_7d) > 3 else ''}")
    logger.info(f"Enriched (*_zscore): {len(enriched_zscore)} columns {enriched_zscore[:3]}{'...' if len(enriched_zscore) > 3 else ''}")

    logger.info("✓ Column inventory complete")

    return {
        'cols_activity_apple': cols_activity_apple,
        'cols_activity_zepp': cols_activity_zepp,
        'cols_activity_coalesced': cols_activity_coalesced,
        'cols_cardio': cols_cardio,
        'cols_sleep': cols_sleep,
        'cols_label': cols_label,
        'cols_segment': cols_segment,
        'enriched_7d': enriched_7d,
        'enriched_zscore': enriched_zscore,
    }


# ============================================================================
# SECTION 3: DATA HEALTH CHECKS
# ============================================================================

def health_checks(df: pd.DataFrame, plots_dir: Path) -> dict:
    """Perform data health checks and return results."""
    logger.info("=== DATA HEALTH CHECKS ===")

    results = {}

    # Coverage
    coverage = df.notna().mean() * 100
    coverage_sorted = coverage.sort_values()
    logger.info(f"Coverage (% non-null) - Worst 15 columns:\n{coverage_sorted.head(15)}")

    # Plot coverage
    fig, ax = plt.subplots(figsize=(10, 5))
    coverage_worst_15 = coverage_sorted.head(15)
    ax.barh(range(len(coverage_worst_15)), coverage_worst_15.values)
    ax.set_yticks(range(len(coverage_worst_15)))
    ax.set_yticklabels(coverage_worst_15.index)
    ax.set_xlabel('Coverage (%)')
    ax.set_title('Column Coverage: 15 Worst')
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(plots_dir / "00_coverage_worst15.png", dpi=144)
    plt.close()
    logger.info(f"✓ Saved 00_coverage_worst15.png")

    # Date continuity
    date_min = df['date'].min()
    date_max = df['date'].max()
    expected_dates = pd.date_range(date_min, date_max, freq='D')
    actual_dates = set(df['date'].values)
    missing_dates = [d for d in expected_dates if d not in actual_dates]

    if missing_dates:
        logger.warning(f"⚠ {len(missing_dates)} missing date(s)")
    else:
        logger.info("✓ No missing dates (continuous)")

    results['missing_dates'] = missing_dates
    results['date_min'] = date_min
    results['date_max'] = date_max

    # Plot date continuity
    core_trio = ['act_steps', 'hr_mean', 'sleep_total_h']
    core_present = [c for c in core_trio if c in df.columns]

    if core_present:
        daily_counts = df[[c for c in core_present]].notna().sum(axis=1)
        fig, ax = plt.subplots(figsize=(12, 3))
        ax.plot(df['date'], daily_counts, marker='o', markersize=2, linewidth=1)
        ax.set_xlabel('Date')
        ax.set_ylabel('Non-null count')
        ax.set_title(f'Daily Data Availability: {core_present}')
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(plots_dir / "00_date_continuity.png", dpi=144)
        plt.close()
        logger.info(f"✓ Saved 00_date_continuity.png")

    # Duplicates
    n_unique_dates = df['date'].nunique()
    if n_unique_dates < len(df):
        logger.warning(f"⚠ {len(df) - n_unique_dates} duplicate date row(s) found")
    else:
        logger.info("✓ All dates unique")

    # Value ranges
    ranges_checks = []
    if 'sleep_total_h' in df.columns:
        bad_sleep = df[(df['sleep_total_h'] < 0) | (df['sleep_total_h'] > 16)]
        if len(bad_sleep) > 0:
            ranges_checks.append(f"sleep_total_h out of [0, 16]: {len(bad_sleep)} rows")

    if ranges_checks:
        logger.warning(f"⚠ Value range issues found: {ranges_checks}")
    else:
        logger.info("✓ Value ranges OK")

    logger.info("✓ Data health checks complete")
    return results


# ============================================================================
# SECTION 4: DESCRIPTIVE STATISTICS
# ============================================================================

def compute_stats(df: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    """Compute and save descriptive statistics."""
    logger.info("=== DESCRIPTIVE STATISTICS ===")

    stats_rows = []
    for col in df.columns:
        if col in ['date', 'label', 'label_source', 'segment_id', 'source_domain']:
            continue
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue

        non_null = df[col].notna().sum()
        if non_null == 0:
            continue

        stats_rows.append({
            'column': col,
            'non_null': non_null,
            'mean': df[col].mean(),
            'std': df[col].std(),
            'min': df[col].min(),
            'p25': df[col].quantile(0.25),
            'p50': df[col].quantile(0.50),
            'p75': df[col].quantile(0.75),
            'max': df[col].max(),
            'missing_pct': (1 - non_null / len(df)) * 100,
        })

    stats_df = pd.DataFrame(stats_rows).sort_values('missing_pct', ascending=False)

    # Timestamped output to keep historic runs
    ts = datetime.now().strftime('%Y%m%dT%H%M%S')
    stats_path_ts = out_dir / f"nb1_feature_stats_{ts}.csv"
    stats_df.to_csv(stats_path_ts, index=False)
    # Also write a stable name for quick access
    stats_path = out_dir / "nb1_feature_stats.csv"
    stats_df.to_csv(stats_path, index=False)

    logger.info(f"✓ Saved nb1_feature_stats.csv and {stats_path_ts.name} ({len(stats_df)} columns)")

    return stats_df


# ============================================================================
# SECTION 5: VISUALIZATIONS – DAILY SIGNALS
# ============================================================================

def plot_daily_signals(df: pd.DataFrame, plots_dir: Path) -> None:
    """Create daily signal visualizations."""
    logger.info("=== PLOTTING DAILY SIGNALS ===")

    # Activity: steps
    if 'act_steps' in df.columns:
        fig, ax = plt.subplots(figsize=(14, 4))
        ax.plot(df['date'], df['act_steps'], marker='o', markersize=3, linewidth=1, label='act_steps')

        # Annotate top 5
        top5_high = df.nlargest(5, 'act_steps')
        for idx, row in top5_high.iterrows():
            if pd.notna(row['act_steps']):
                ax.annotate(f"{row['act_steps']:.0f}", xy=(row['date'], row['act_steps']),
                           xytext=(0, 5), textcoords='offset points', fontsize=7, ha='center')

        ax.set_xlabel('Date')
        ax.set_ylabel('Steps')
        ax.set_title('Activity: Steps Over Time (Top 5 highs annotated)')
        ax.grid(alpha=0.3)
        ax.legend()
        plt.tight_layout()
        plt.savefig(plots_dir / "01_activity_steps.png", dpi=144)
        plt.close()
        logger.info("✓ Saved 01_activity_steps.png")

    # Exercise minutes
    if 'apple_exercise_min' in df.columns:
        fig, ax = plt.subplots(figsize=(14, 4))
        ax.plot(df['date'], df['apple_exercise_min'], marker='s', markersize=3, linewidth=1,
                label='apple_exercise_min', color='orange')
        ax.set_xlabel('Date')
        ax.set_ylabel('Exercise Minutes')
        ax.set_title('Activity: Apple Exercise Minutes')
        ax.grid(alpha=0.3)
        ax.legend()
        plt.tight_layout()
        plt.savefig(plots_dir / "01_activity_exercise_min.png", dpi=144)
        plt.close()
        logger.info("✓ Saved 01_activity_exercise_min.png")

    # Cardio: HR mean & std
    if 'hr_mean' in df.columns:
        fig, ax = plt.subplots(figsize=(14, 4))
        ax.plot(df['date'], df['hr_mean'], marker='o', markersize=3, linewidth=1, label='hr_mean', color='red')

        if 'hr_std' in df.columns:
            ax2 = ax.twinx()
            ax2.plot(df['date'], df['hr_std'], marker='s', markersize=3, linewidth=1, label='hr_std', color='blue')
            ax2.set_ylabel('HR Std Dev', color='blue')
            ax2.tick_params(axis='y', labelcolor='blue')

        ax.set_xlabel('Date')
        ax.set_ylabel('HR Mean (bpm)', color='red')
        ax.tick_params(axis='y', labelcolor='red')
        ax.set_title('Cardio: HR Mean & Std Over Time')
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(plots_dir / "02_cardio_hr_mean_std.png", dpi=144)
        plt.close()
        logger.info("✓ Saved 02_cardio_hr_mean_std.png")

    # Sleep
    if 'sleep_total_h' in df.columns:
        fig, ax = plt.subplots(figsize=(14, 4))
        ax.plot(df['date'], df['sleep_total_h'], marker='o', markersize=3, linewidth=1.5,
                label='sleep_total_h', color='purple')

        if 'zepp_slp_deep_h' in df.columns:
            ax.plot(df['date'], df['zepp_slp_deep_h'], marker='s', markersize=2, linewidth=1,
                    label='zepp_slp_deep_h', alpha=0.7)

        ax.set_xlabel('Date')
        ax.set_ylabel('Sleep (hours)')
        ax.set_title('Sleep: Total Hours & Components')
        ax.grid(alpha=0.3)
        ax.legend()
        plt.tight_layout()
        plt.savefig(plots_dir / "03_sleep_total_components.png", dpi=144)
        plt.close()
        logger.info("✓ Saved 03_sleep_total_components.png")

    logger.info("✓ Daily signals plotted")


# ============================================================================
# SECTION 6: CORRELATIONS
# ============================================================================

def plot_correlations(df: pd.DataFrame, plots_dir: Path) -> None:
    """Plot key correlations."""
    logger.info("=== CORRELATIONS & CROSS-DOMAIN HINTS ===")

    corr_cols = []
    for col in ['act_steps', 'act_active_min', 'hr_mean', 'hr_std', 'sleep_total_h']:
        if col in df.columns:
            corr_cols.append(col)

    enriched_7d = [c for c in df.columns if '_7d' in c and pd.api.types.is_numeric_dtype(df[c])]
    corr_cols.extend(enriched_7d[:5])
    corr_cols = list(set(corr_cols))

    if len(corr_cols) > 1:
        corr_matrix = df[corr_cols].corr(method='spearman')
        logger.info(f"Computed Spearman correlations for {len(corr_cols)} columns")

        # Key scatter plots
        scatter_pairs = [
            ('act_steps', 'sleep_total_h'),
            ('hr_mean', 'sleep_total_h'),
            ('act_active_min', 'hr_mean'),
        ]

        for var1, var2 in scatter_pairs:
            if var1 in df.columns and var2 in df.columns:
                data = df[[var1, var2]].dropna()

                if len(data) > 300:
                    data = data.sample(300, random_state=42)

                fig, ax = plt.subplots(figsize=(8, 5))
                ax.scatter(data[var1], data[var2], alpha=0.5, s=30)
                ax.set_xlabel(var1)
                ax.set_ylabel(var2)
                ax.set_title(f'Scatter: {var1} vs {var2} (n={len(data)})')
                ax.grid(alpha=0.3)

                # Add correlation info
                rho, pval = spearmanr(data[var1], data[var2])
                ax.text(0.05, 0.95, f"ρ={rho:.2f} (p={pval:.2e})", transform=ax.transAxes,
                       verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

                plt.tight_layout()
                plt.savefig(plots_dir / f"06_scatter_{var1}_vs_{var2}.png", dpi=144)
                plt.close()
                logger.info(f"✓ Saved 06_scatter_{var1}_vs_{var2}.png")

    logger.info("✓ Correlations plotted")


# ============================================================================
# SECTION 7: LABEL COVERAGE
# ============================================================================

def check_labels(df: pd.DataFrame, plots_dir: Path) -> dict:
    """Check label coverage."""
    logger.info("=== LABELS COVERAGE ===")

    results = {}

    if 'label' in df.columns:
        label_coverage = df['label'].notna().sum() / len(df) * 100
        logger.info(f"Label coverage: {label_coverage:.1f}%")

        label_counts = df['label'].value_counts()
        logger.info(f"Label Distribution:\n{label_counts}")

        # Plot
        fig, ax = plt.subplots(figsize=(8, 4))
        label_counts.plot(kind='bar', ax=ax, color='steelblue')
        ax.set_xlabel('Label')
        ax.set_ylabel('Count')
        ax.set_title('Label Distribution')
        ax.grid(axis='y', alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(plots_dir / "05_label_distribution.png", dpi=144)
        plt.close()
        logger.info("✓ Saved 05_label_distribution.png")

        results['label_coverage'] = label_coverage
        results['label_counts'] = label_counts.to_dict()

    logger.info("✓ Labels checked")
    return results


# ============================================================================
# SECTION 8: MANIFEST & SUMMARY GENERATION
# ============================================================================

def generate_manifest_and_summary(
    pid: str,
    snapshot: str,
    joined_path: Path,
    n_rows: int,
    n_cols: int,
    date_range: int,
    df: pd.DataFrame,
    date_min: pd.Timestamp,
    date_max: pd.Timestamp,
    col_inventory: dict,
    out_dir: Path,
    plots_dir: Path,
) -> tuple[dict, str]:
    """Generate manifest and summary markdown."""
    logger.info("=== GENERATING MANIFEST & SUMMARY ===")

    # timestamp for this run (used to version artifacts)
    run_ts = datetime.now().strftime('%Y%m%dT%H%M%S')

    # Build manifest
    manifest = {
        'pid': pid,
        'snapshot': snapshot,
        'input_path': str(joined_path),
        'row_count': int(n_rows),
        'col_count': int(n_cols),
        'date_range': {
            'min': str(date_min.date()),
            'max': str(date_max.date()),
            'days': int(date_range),
        },
        'domain_coverage': {
            'activity_apple': sum(1 for c in col_inventory['cols_activity_apple'] if c in df.columns),
            'activity_zepp': sum(1 for c in col_inventory['cols_activity_zepp'] if c in df.columns),
            'activity_coalesced': sum(1 for c in col_inventory['cols_activity_coalesced'] if c in df.columns),
            'cardio': sum(1 for c in col_inventory['cols_cardio'] if c in df.columns),
            'sleep': sum(1 for c in col_inventory['cols_sleep'] if c in df.columns),
            'enriched_7d': len(col_inventory['enriched_7d']),
            'enriched_zscore': len(col_inventory['enriched_zscore']),
        },
        'plots_saved': sorted([p.name for p in plots_dir.glob('*.png')]),
        'generation_timestamp': datetime.now().isoformat(),
        'notebook_version': '1.0',
    }

    # Save manifest (timestamped + stable copy)
    manifest_path_ts = out_dir / f"nb1_manifest_{run_ts}.json"
    with open(manifest_path_ts, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2)
    manifest_path = out_dir / "nb1_manifest.json"
    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2)
    logger.info(f"✓ Saved manifest: {manifest_path.name} and {manifest_path_ts.name}")

    # Generate summary markdown
    activity_msg = f"   - {df['act_steps'].mean():.0f} mean steps/day (sigma={df['act_steps'].std():.0f})" if 'act_steps' in df.columns else "   - No activity data"
    cardio_msg = f"   - {df['hr_mean'].mean():.1f} mean HR bpm (sigma={df['hr_mean'].std():.1f})" if 'hr_mean' in df.columns else "   - No HR data"
    sleep_msg = f"   - {df['sleep_total_h'].mean():.1f} mean sleep hours (sigma={df['sleep_total_h'].std():.1f})" if 'sleep_total_h' in df.columns else "   - No sleep data"

    summary_lines = [
        "# NB1 EDA Daily Summary",
        f"\n**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Participant:** {pid} | **Snapshot:** {snapshot}",
        f"\n## Coverage Overview",
        f"\n- **Rows:** {n_rows} | **Columns:** {n_cols} | **Date Range:** {date_range} days",
        f"- **Activity (Apple):** {manifest['domain_coverage']['activity_apple']}/{len(col_inventory['cols_activity_apple'])} columns",
        f"- **Activity (Zepp):** {manifest['domain_coverage']['activity_zepp']}/{len(col_inventory['cols_activity_zepp'])} columns",
        f"- **Cardio:** {manifest['domain_coverage']['cardio']}/{len(col_inventory['cols_cardio'])} columns",
        f"- **Sleep:** {manifest['domain_coverage']['sleep']}/{len(col_inventory['cols_sleep'])} columns",
        f"- **Enriched (*_7d):** {len(col_inventory['enriched_7d'])} columns",
        f"\n## Data Quality",
        f"\n**Date Range:** {date_min.date()} to {date_max.date()} ({date_range} days)",
        f"**Activity:** {activity_msg}",
        f"**Cardio:** {cardio_msg}",
        f"**Sleep:** {sleep_msg}",
        f"\n## Artifacts Generated",
        f"\n- `nb1_feature_stats.csv` — Descriptive statistics",
        f"- `plots/` — {len(manifest['plots_saved'])} PNG visualizations",
        f"- `nb1_manifest.json` — Metadata & manifest",
        f"\n✨ All data local, no internet calls. Ready for offline analysis.",
    ]

    summary_md = "\n".join(summary_lines)

    # Save summary (timestamped + stable copy)
    summary_path_ts = out_dir / f"nb1_eda_summary_{run_ts}.md"
    with open(summary_path_ts, 'w', encoding='utf-8') as f:
        f.write(summary_md)
    summary_path = out_dir / "nb1_eda_summary.md"
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(summary_md)
    logger.info(f"✓ Saved summary: {summary_path.name} and {summary_path_ts.name}")

    return manifest, summary_md


# ============================================================================
# SECTION 9: SETUP LATEST/ MIRROR
# ============================================================================

def setup_latest_mirror(out_dir: Path, plots_dir: Path, latest_dir: Path) -> None:
    """Mirror key artifacts to latest/."""
    logger.info("=== SETTING UP LATEST/ MIRROR ===")

    # Copy any NB1 artifacts (both stable and timestamped) to latest/
    for p in sorted(out_dir.glob('nb1_*')):
        if p.is_file():
            try:
                dst = latest_dir / p.name
                shutil.copy2(p, dst)
                logger.info(f"  Copied {p.name} → latest/")
            except Exception as e:
                logger.warning(f"  Could not copy {p.name} → latest/: {e}")

    # Copy key plots
    for p in sorted(plots_dir.glob('*.png')):
        if p.is_file():
            dst = latest_dir / p.name
            shutil.copy2(p, dst)
            logger.info(f"  Copied {p.name} → latest/")

    logger.info("✓ Latest/ mirror ready")


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main entry point."""
    args = parse_args()

    try:
        # Initialize paths
        base, out, plots, latest, snapshot = initialize_paths(
            args.pid, args.snapshot, args.repo_root
        )

        # Verify extraction status
        check_extraction(base, args.pid, snapshot)

        # Load and validate data
        joined = base / snapshot / "joined" / "joined_features_daily.csv"
        df, n_rows, n_cols, date_min, date_max, date_range = load_and_validate_data(joined)

        # Coerce numeric
        df = coerce_numeric(df)

        # Inventory columns
        col_inventory = inventory_columns(df)

        # Health checks
        health_checks(df, plots)

        # Compute stats
        stats_df = compute_stats(df, out)

        # Plot daily signals
        plot_daily_signals(df, plots)

        # Plot correlations
        plot_correlations(df, plots)

        # Check labels
        label_results = check_labels(df, plots)

        # Generate manifest and summary
        manifest, summary_md = generate_manifest_and_summary(
            args.pid, snapshot, joined,
            n_rows, n_cols, date_range,
            df, date_min, date_max,
            col_inventory, out, plots
        )

        # Setup latest mirror
        setup_latest_mirror(out, plots, latest)

        # Final summary
        logger.info("=" * 70)
        logger.info("✅ NB1 EDA COMPLETE")
        logger.info("=" * 70)
        logger.info(f"📁 Artifacts Location: {out}")
        logger.info(f"📊 Key Files:")
        logger.info(f"   - {out / 'nb1_eda_summary.md'}")
        logger.info(f"   - {out / 'nb1_feature_stats.csv'}")
        logger.info(f"   - {out / 'nb1_manifest.json'}")
        logger.info(f"   - {plots} (contains {len(list(plots.glob('*.png')))} PNG files)")
        logger.info(f"🔗 Quick Access: {latest}")
        logger.info(f"✨ All data local, no internet calls. Ready for offline analysis.")
        logger.info("=" * 70)

        return 0

    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    sys.exit(main())
