"""
ETL Audit Module for practicum2-nof1-adhd-bd

PhD-level domain-specific regression test suite for ETL pipeline integrity.
Supports Cardio, Activity, Sleep, Meds, and SoM feature audits with PASS/FAIL reporting.

Usage:
    python -m src.etl.etl_audit --participant P000001 --snapshot 2025-11-07 --domain cardio
    python -m src.etl.etl_audit --participant P000001 --snapshot 2025-11-07 --domain activity
    python -m src.etl.etl_audit --participant P000001 --snapshot 2025-11-07 --domain sleep
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Tuple, Optional, Any
import sys
import argparse
import json

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


def numpy_json_serializer(obj: Any) -> Any:
    """Convert numpy types to Python native types for JSON serialization."""
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


class ETLAuditor:
    """Audits ETL pipeline integrity for a single participant/snapshot with domain-specific checks."""
    
    def __init__(self, participant: str, snapshot: str, domain: str = "cardio"):
        """
        Initialize auditor for specific participant/snapshot/domain.
        
        Args:
            participant: e.g., "P000001"
            snapshot: e.g., "2025-11-07"
            domain: "cardio", "activity", "sleep", "meds", or "som"
        """
        self.participant = participant
        self.snapshot = snapshot
        self.domain = domain.lower()
        
        # Paths
        self.raw_dir = Path("data/raw") / participant
        self.etl_dir = Path("data/etl") / participant / snapshot
        self.extracted_dir = self.etl_dir / "extracted"
        self.joined_dir = self.etl_dir / "joined"
        self.qc_dir = self.etl_dir / "qc"
        
        # Create QC directory
        self.qc_dir.mkdir(parents=True, exist_ok=True)
        
        # Results storage
        self.audit_results = {
            "participant": participant,
            "snapshot": snapshot,
            "domain": self.domain,
            "timestamp": datetime.now().isoformat(),
            "pass": True,  # Will be set to False if critical issues found
            "raw_files": {},
            "extracted_stats": {},
            "unified_stats": {},
            "issues": [],
            "warnings": []
        }
        
        # Per-day results for CSV export
        self.per_day_results = []
    
    def banner(self, text: str):
        """Print a banner for section headers."""
        print()
        print("=" * 80)
        print(text)
        print("=" * 80)
        print()
    
    def add_issue(self, severity: str, category: str, description: str, details: Dict = None):
        """Record an issue found during audit."""
        issue = {
            "severity": severity,  # "CRITICAL", "WARNING", "INFO"
            "category": category,  # "RAW_COVERAGE", "EXTRACTION", "AGGREGATION", "JOIN", "DATA_QUALITY"
            "description": description,
            "details": details or {}
        }
        
        if severity == "CRITICAL":
            self.audit_results["issues"].append(issue)
            self.audit_results["pass"] = False  # Mark as failed
        else:
            self.audit_results["warnings"].append(issue)
        
        prefix = "🔴" if severity == "CRITICAL" else "⚠️" if severity == "WARNING" else "ℹ️"
        logger.info(f"{prefix} [{category}] {description}")
        if details:
            for key, value in details.items():
                logger.info(f"    {key}: {value}")
    
    # ========================================================================
    # AUDIT SECTION 1: RAW FILE COVERAGE
    # ========================================================================
    
    def audit_raw_files(self):
        """Audit which raw files exist and are being used."""
        self.banner("AUDIT SECTION 1: RAW FILE COVERAGE")
        
        # Check Apple raw files
        apple_export_dir = self.raw_dir / "apple" / "export"
        if apple_export_dir.exists():
            apple_zips = list(apple_export_dir.glob("*.zip"))
            self.audit_results["raw_files"]["apple_zips"] = [z.name for z in apple_zips]
            logger.info(f"[RAW] Found {len(apple_zips)} Apple ZIP(s):")
            for z in apple_zips:
                logger.info(f"    - {z.name} ({z.stat().st_size / (1024**2):.1f} MB)")
        else:
            self.add_issue("WARNING", "RAW_COVERAGE", 
                          f"Apple export directory not found: {apple_export_dir}")
        
        # Check Zepp raw files
        zepp_dir = self.raw_dir / "zepp"
        if zepp_dir.exists():
            zepp_zips = list(zepp_dir.glob("*.zip"))
            self.audit_results["raw_files"]["zepp_zips"] = [z.name for z in zepp_zips]
            logger.info(f"[RAW] Found {len(zepp_zips)} Zepp ZIP(s):")
            for z in zepp_zips:
                logger.info(f"    - {z.name} ({z.stat().st_size / (1024**2):.1f} MB)")
        else:
            self.add_issue("WARNING", "RAW_COVERAGE",
                          f"Zepp directory not found: {zepp_dir}")
    
    # ========================================================================
    # AUDIT SECTION 2: EXTRACTED LAYER
    # ========================================================================
    
    def audit_extracted_layer(self):
        """Audit extracted daily CSVs from Apple and Zepp."""
        self.banner("AUDIT SECTION 2: EXTRACTED LAYER")
        
        if not self.extracted_dir.exists():
            self.add_issue("CRITICAL", "EXTRACTION",
                          f"Extracted directory does not exist: {self.extracted_dir}")
            return
        
        # Audit Apple extracted files
        logger.info("\n[EXTRACTED] Apple Data:")
        apple_dir = self.extracted_dir / "apple"
        apple_metrics = ["daily_sleep", "daily_cardio", "daily_activity"]
        
        for metric in apple_metrics:
            csv_path = apple_dir / f"{metric}.csv"
            if csv_path.exists():
                df = pd.read_csv(csv_path)
                df["date"] = pd.to_datetime(df["date"])
                
                stats = {
                    "rows": len(df),
                    "date_min": df["date"].min().strftime("%Y-%m-%d"),
                    "date_max": df["date"].max().strftime("%Y-%m-%d"),
                    "date_span_days": (df["date"].max() - df["date"].min()).days + 1,
                    "unique_dates": df["date"].nunique(),
                    "duplicate_dates": len(df) - df["date"].nunique()
                }
                
                self.audit_results["extracted_stats"][f"apple_{metric}"] = stats
                
                logger.info(f"  {metric}.csv:")
                logger.info(f"    Rows: {stats['rows']}")
                logger.info(f"    Date range: {stats['date_min']} to {stats['date_max']} ({stats['date_span_days']} days)")
                logger.info(f"    Unique dates: {stats['unique_dates']}")
                
                if stats["duplicate_dates"] > 0:
                    self.add_issue("WARNING", "EXTRACTION",
                                  f"Apple {metric} has duplicate dates: {stats['duplicate_dates']} duplicates",
                                  {"file": str(csv_path)})
                
                # Check for missing days
                expected_days = stats["date_span_days"]
                actual_days = stats["unique_dates"]
                missing_days = expected_days - actual_days
                if missing_days > 0:
                    self.add_issue("INFO", "EXTRACTION",
                                  f"Apple {metric} has {missing_days} missing days in date range",
                                  {"expected": expected_days, "actual": actual_days})
            else:
                self.add_issue("WARNING", "EXTRACTION",
                              f"Apple {metric}.csv not found",
                              {"expected_path": str(csv_path)})
        
        # Audit Zepp extracted files
        logger.info("\n[EXTRACTED] Zepp Data:")
        zepp_dir = self.extracted_dir / "zepp"
        zepp_metrics = ["daily_sleep", "daily_cardio", "daily_activity"]
        
        for metric in zepp_metrics:
            csv_path = zepp_dir / f"{metric}.csv"
            if csv_path.exists():
                df = pd.read_csv(csv_path)
                df["date"] = pd.to_datetime(df["date"])
                
                stats = {
                    "rows": len(df),
                    "date_min": df["date"].min().strftime("%Y-%m-%d"),
                    "date_max": df["date"].max().strftime("%Y-%m-%d"),
                    "date_span_days": (df["date"].max() - df["date"].min()).days + 1,
                    "unique_dates": df["date"].nunique(),
                    "duplicate_dates": len(df) - df["date"].nunique()
                }
                
                self.audit_results["extracted_stats"][f"zepp_{metric}"] = stats
                
                logger.info(f"  {metric}.csv:")
                logger.info(f"    Rows: {stats['rows']}")
                logger.info(f"    Date range: {stats['date_min']} to {stats['date_max']} ({stats['date_span_days']} days)")
                logger.info(f"    Unique dates: {stats['unique_dates']}")
                
                if stats["duplicate_dates"] > 0:
                    self.add_issue("WARNING", "EXTRACTION",
                                  f"Zepp {metric} has duplicate dates: {stats['duplicate_dates']} duplicates",
                                  {"file": str(csv_path)})
                
                # Check for missing days
                expected_days = stats["date_span_days"]
                actual_days = stats["unique_dates"]
                missing_days = expected_days - actual_days
                if missing_days > 0:
                    self.add_issue("INFO", "EXTRACTION",
                                  f"Zepp {metric} has {missing_days} missing days in date range",
                                  {"expected": expected_days, "actual": actual_days})
            else:
                self.add_issue("INFO", "EXTRACTION",
                              f"Zepp {metric}.csv not found (may be normal if no Zepp data)",
                              {"expected_path": str(csv_path)})
    
    # ========================================================================
    # AUDIT SECTION 3: UNIFIED JOIN
    # ========================================================================
    
    def audit_unified_join(self):
        """Audit features_daily_unified.csv for join integrity."""
        self.banner("AUDIT SECTION 3: UNIFIED JOIN")
        
        unified_path = self.joined_dir / "features_daily_unified.csv"
        
        if not unified_path.exists():
            self.add_issue("CRITICAL", "JOIN",
                          f"Unified CSV does not exist: {unified_path}")
            return
        
        df = pd.read_csv(unified_path)
        df["date"] = pd.to_datetime(df["date"])
        
        # Basic stats
        stats = {
            "rows": len(df),
            "date_min": df["date"].min().strftime("%Y-%m-%d"),
            "date_max": df["date"].max().strftime("%Y-%m-%d"),
            "date_span_days": (df["date"].max() - df["date"].min()).days + 1,
            "unique_dates": df["date"].nunique(),
            "duplicate_dates": len(df) - df["date"].nunique()
        }
        
        logger.info(f"[UNIFIED] features_daily_unified.csv:")
        logger.info(f"  Rows: {stats['rows']}")
        logger.info(f"  Date range: {stats['date_min']} to {stats['date_max']}")
        logger.info(f"  Date span: {stats['date_span_days']} days")
        logger.info(f"  Unique dates: {stats['unique_dates']}")
        
        self.audit_results["unified_stats"]["basic"] = stats
        
        # Check for duplicates
        if stats["duplicate_dates"] > 0:
            self.add_issue("CRITICAL", "JOIN",
                          f"Unified CSV has DUPLICATE dates: {stats['duplicate_dates']} duplicates")
            duplicates = df[df.duplicated(subset=["date"], keep=False)].sort_values("date")
            logger.info(f"  Duplicate dates: {duplicates['date'].tolist()}")
        
        # Check for missing days in range
        date_range = pd.date_range(df["date"].min(), df["date"].max(), freq="D")
        missing_dates = set(date_range) - set(df["date"])
        if missing_dates:
            self.add_issue("WARNING", "JOIN",
                          f"Unified CSV has {len(missing_dates)} MISSING days in date range",
                          {"missing_count": len(missing_dates)})
            logger.info(f"  First 10 missing dates: {sorted(missing_dates)[:10]}")
        
        # Check for non-monotonic dates
        if not df["date"].is_monotonic_increasing:
            self.add_issue("CRITICAL", "JOIN",
                          "Unified CSV has NON-MONOTONIC date ordering")
        
        # Domain coverage analysis
        logger.info(f"\n[UNIFIED] Domain Coverage:")
        
        # Sleep domain
        sleep_cols = [c for c in df.columns if "sleep" in c.lower()]
        sleep_coverage = df[sleep_cols].notna().any(axis=1).sum()
        logger.info(f"  Sleep: {sleep_coverage}/{len(df)} days ({100*sleep_coverage/len(df):.1f}%)")
        
        # Cardio domain
        cardio_cols = [c for c in df.columns if "hr_" in c.lower() or "heart" in c.lower()]
        cardio_coverage = df[cardio_cols].notna().any(axis=1).sum()
        logger.info(f"  Cardio: {cardio_coverage}/{len(df)} days ({100*cardio_coverage/len(df):.1f}%)")
        
        # Activity domain
        activity_cols = [c for c in df.columns if "step" in c.lower() or "distance" in c.lower() or "energy" in c.lower()]
        activity_coverage = df[activity_cols].notna().any(axis=1).sum()
        logger.info(f"  Activity: {activity_coverage}/{len(df)} days ({100*activity_coverage/len(df):.1f}%)")
        
        # Days with all three domains
        all_three = df[sleep_cols].notna().any(axis=1) & df[cardio_cols].notna().any(axis=1) & df[activity_cols].notna().any(axis=1)
        all_three_count = all_three.sum()
        logger.info(f"  All 3 domains: {all_three_count}/{len(df)} days ({100*all_three_count/len(df):.1f}%)")
        
        # Days with no data
        no_data = ~(df[sleep_cols].notna().any(axis=1) | df[cardio_cols].notna().any(axis=1) | df[activity_cols].notna().any(axis=1))
        no_data_count = no_data.sum()
        if no_data_count > 0:
            self.add_issue("WARNING", "DATA_QUALITY",
                          f"Unified CSV has {no_data_count} days with NO data in any domain",
                          {"count": no_data_count})
        
        self.audit_results["unified_stats"]["coverage"] = {
            "sleep": int(sleep_coverage),
            "cardio": int(cardio_coverage),
            "activity": int(activity_coverage),
            "all_three": int(all_three_count),
            "no_data": int(no_data_count)
        }
        
        # Histogram of domain coverage per day
        logger.info(f"\n[UNIFIED] Domain Coverage Histogram:")
        domain_count_per_day = (
            df[sleep_cols].notna().any(axis=1).astype(int) +
            df[cardio_cols].notna().any(axis=1).astype(int) +
            df[activity_cols].notna().any(axis=1).astype(int)
        )
        hist = domain_count_per_day.value_counts().sort_index()
        for n_domains, count in hist.items():
            logger.info(f"  {n_domains} domains: {count} days ({100*count/len(df):.1f}%)")
    
    # ========================================================================
    # AUDIT SECTION 4: CROSS-CHECK SAMPLE DAYS
    # ========================================================================
    
    def audit_sample_days(self, sample_dates: List[str] = None):
        """
        Cross-check specific days between raw/extracted/unified.
        
        Args:
            sample_dates: List of dates to check (e.g., ["2019-01-15", "2020-06-20"])
                         If None, picks 3 random days from unified CSV
        """
        self.banner("AUDIT SECTION 4: SAMPLE DAY CROSS-CHECK")
        
        unified_path = self.joined_dir / "features_daily_unified.csv"
        if not unified_path.exists():
            logger.warning("[SAMPLE] Cannot run sample check - unified CSV missing")
            return
        
        df_unified = pd.read_csv(unified_path)
        df_unified["date"] = pd.to_datetime(df_unified["date"]).dt.strftime("%Y-%m-%d")
        
        if sample_dates is None:
            # Pick 3 random dates
            if len(df_unified) < 3:
                sample_dates = df_unified["date"].tolist()
            else:
                sample_dates = df_unified.sample(3)["date"].tolist()
        
        logger.info(f"[SAMPLE] Checking {len(sample_dates)} sample days:")
        
        for date in sample_dates:
            logger.info(f"\n  Date: {date}")
            
            # Check if date exists in unified
            if date in df_unified["date"].values:
                row = df_unified[df_unified["date"] == date].iloc[0]
                logger.info(f"    ✓ Exists in unified CSV")
                logger.info(f"      Sleep: hours={row.get('sleep_hours', 'N/A')}, quality={row.get('sleep_quality_score', 'N/A')}")
                logger.info(f"      Cardio: hr_mean={row.get('hr_mean', 'N/A')}, hr_max={row.get('hr_max', 'N/A')}")
                logger.info(f"      Activity: steps={row.get('total_steps', 'N/A')}, distance={row.get('total_distance', 'N/A')}")
            else:
                logger.info(f"    ✗ NOT FOUND in unified CSV")
                self.add_issue("WARNING", "DATA_QUALITY",
                              f"Sample date {date} not found in unified CSV")
            
            # TODO: Check extracted Apple/Zepp CSVs for this date
            # TODO: Check raw XML/CSV for this date (requires parsing)
    
    # ========================================================================
    # DOMAIN-SPECIFIC AUDITS
    # ========================================================================
    
    def run_cardio_audit(self) -> bool:
        """
        Run Cardio (HR) feature integrity audit.
        
        Checks:
        - Event-level Parquet exists and is valid
        - Daily cache has all 5 metrics (mean, min, max, std, n)
        - Fabrication rates < 10% (hr_min == hr_mean, hr_std == 0)
        - hr_std > 0 for multi-sample days
        - Consistency across cache → daily_cardio.csv → unified → labeled
        - No duplicate columns (hr_mean vs apple_hr_mean)
        
        Returns:
            True if all checks pass, False otherwise
        """
        self.banner(f"CARDIO (HR) FEATURE INTEGRITY AUDIT: {self.participant}/{self.snapshot}")
        
        # Check 1: Event-level Parquet
        apple_dir = self.extracted_dir / "apple"
        cache_dir = apple_dir / "apple_health_export" / ".cache"
        events_file = cache_dir / "export_apple_hr_events.parquet"
        
        if not events_file.exists():
            self.add_issue("CRITICAL", "HR_EXTRACTION",
                          f"Event-level Parquet not found: {events_file}")
            return False
        
        df_events = pd.read_parquet(events_file)
        logger.info(f"✓ Event-level Parquet: {len(df_events):,} HR records")
        
        # Check 2: Daily cache Parquet with all 5 metrics
        daily_cache_file = cache_dir / "export_apple_hr_daily.parquet"
        
        if not daily_cache_file.exists():
            self.add_issue("CRITICAL", "HR_EXTRACTION",
                          f"Daily cache Parquet not found: {daily_cache_file}")
            return False
        
        df_cache = pd.read_parquet(daily_cache_file)
        logger.info(f"✓ Daily cache Parquet: {len(df_cache)} days")
        
        required_metrics = ['apple_hr_mean', 'apple_hr_min', 'apple_hr_max', 'apple_hr_std', 'apple_n_hr']
        missing_metrics = [m for m in required_metrics if m not in df_cache.columns]
        
        if missing_metrics:
            self.add_issue("CRITICAL", "HR_EXTRACTION",
                          f"Missing metrics in cache: {missing_metrics}")
            return False
        
        logger.info(f"✓ Cache schema: All 5 metrics present")
        
        # Check 3: Fabrication rates
        df_cache_valid = df_cache[df_cache['apple_hr_mean'].notna()].copy()
        
        fabricated_min = (df_cache_valid['apple_hr_min'] == df_cache_valid['apple_hr_mean']).sum()
        fabricated_std = (df_cache_valid['apple_hr_std'] == 0.0).sum()
        total_days = len(df_cache_valid)
        
        fab_rate_min = fabricated_min / total_days if total_days > 0 else 0
        fab_rate_std = fabricated_std / total_days if total_days > 0 else 0
        
        logger.info(f"\nFabrication Check:")
        logger.info(f"  hr_min == hr_mean: {fabricated_min}/{total_days} ({fab_rate_min*100:.1f}%)")
        logger.info(f"  hr_std == 0.0: {fabricated_std}/{total_days} ({fab_rate_std*100:.1f}%)")
        
        if fab_rate_min > 0.10:
            self.add_issue("CRITICAL", "HR_DATA_QUALITY",
                          f"Excessive fabrication: {fab_rate_min*100:.1f}% of days have hr_min == hr_mean",
                          {"threshold": "10%", "actual": f"{fab_rate_min*100:.1f}%"})
        
        if fab_rate_std > 0.10:
            self.add_issue("CRITICAL", "HR_DATA_QUALITY",
                          f"Excessive zero std: {fab_rate_std*100:.1f}% of days have hr_std == 0",
                          {"threshold": "10%", "actual": f"{fab_rate_std*100:.1f}%"})
        
        # Check 4: daily_cardio.csv consistency
        daily_cardio_file = apple_dir / "daily_cardio.csv"
        
        if not daily_cardio_file.exists():
            self.add_issue("CRITICAL", "HR_AGGREGATION",
                          f"daily_cardio.csv not found: {daily_cardio_file}")
            return False
        
        df_cardio = pd.read_csv(daily_cardio_file)
        logger.info(f"\n✓ daily_cardio.csv: {len(df_cardio)} days")
        
        if len(df_cardio) != len(df_cache):
            self.add_issue("WARNING", "HR_CONSISTENCY",
                          f"Cache has {len(df_cache)} days but daily_cardio.csv has {len(df_cardio)} days")
        
        # Check 5: Unified CSV
        unified_file = self.joined_dir / "features_daily_unified.csv"
        
        if not unified_file.exists():
            self.add_issue("WARNING", "HR_JOIN",
                          f"Unified CSV not found: {unified_file}")
        else:
            df_unified = pd.read_csv(unified_file)
            hr_cols = [c for c in df_unified.columns if 'hr' in c.lower() and any(x in c for x in ['mean', 'min', 'max', 'std', 'sample'])]
            logger.info(f"✓ features_daily_unified.csv: {len(df_unified)} days, HR columns: {hr_cols}")
            
            hr_days = df_unified['hr_mean'].notna().sum()
            logger.info(f"  Days with HR data: {hr_days}/{len(df_unified)} ({hr_days/len(df_unified)*100:.1f}%)")
        
        # Check 6: Labeled CSV
        labeled_file = self.joined_dir / "features_daily_labeled.csv"
        
        if labeled_file.exists():
            df_labeled = pd.read_csv(labeled_file)
            hr_cols_labeled = [c for c in df_labeled.columns if 'hr' in c.lower() and any(x in c for x in ['mean', 'min', 'max', 'std', 'sample'])]
            logger.info(f"✓ features_daily_labeled.csv: {len(df_labeled)} days, HR columns: {hr_cols_labeled}")
        
        # Compute per-day statistics for CSV export
        logger.info(f"\nComputing per-day statistics...")
        
        for _, row in df_cache_valid.iterrows():
            self.per_day_results.append({
                'date': row['date'],
                'hr_mean': row['apple_hr_mean'],
                'hr_min': row['apple_hr_min'],
                'hr_max': row['apple_hr_max'],
                'hr_std': row['apple_hr_std'],
                'hr_samples': row['apple_n_hr'],
                'is_fabricated_min': row['apple_hr_min'] == row['apple_hr_mean'],
                'is_fabricated_std': row['apple_hr_std'] == 0.0,
                'is_single_sample': row['apple_n_hr'] == 1
            })
        
        # Summary statistics
        logger.info(f"\nHR Summary Statistics:")
        logger.info(f"  Mean HR (daily avg): {df_cache_valid['apple_hr_mean'].mean():.2f} ± {df_cache_valid['apple_hr_mean'].std():.2f} bpm")
        logger.info(f"  Min HR range: {df_cache_valid['apple_hr_min'].min():.1f} to {df_cache_valid['apple_hr_min'].max():.1f} bpm")
        logger.info(f"  Max HR range: {df_cache_valid['apple_hr_max'].min():.1f} to {df_cache_valid['apple_hr_max'].max():.1f} bpm")
        logger.info(f"  Std HR (daily): {df_cache_valid['apple_hr_std'].mean():.2f} ± {df_cache_valid['apple_hr_std'].std():.2f} bpm")
        logger.info(f"  Samples/day: {df_cache_valid['apple_n_hr'].median():.0f} (median), range {df_cache_valid['apple_n_hr'].min():.0f}-{df_cache_valid['apple_n_hr'].max():.0f}")
        
        return self.audit_results["pass"]
    
    def run_activity_audit(self) -> bool:
        """
        Run Activity (Steps) feature integrity audit.
        
        Checks:
        - daily_activity.csv exists with steps column
        - Steps present in unified CSV
        - No all-NaN columns
        - Vendor prioritization (Apple vs Zepp)
        - Consistency across extraction → unification → labeling
        
        Returns:
            True if all checks pass, False otherwise
        """
        self.banner(f"ACTIVITY (STEPS) FEATURE INTEGRITY AUDIT: {self.participant}/{self.snapshot}")
        
        # Check 1: daily_activity.csv
        apple_dir = self.extracted_dir / "apple"
        activity_file = apple_dir / "daily_activity.csv"
        
        if not activity_file.exists():
            self.add_issue("CRITICAL", "STEPS_EXTRACTION",
                          f"daily_activity.csv not found: {activity_file}")
            return False
        
        df_activity = pd.read_csv(activity_file)
        logger.info(f"✓ daily_activity.csv: {len(df_activity)} days")
        
        # Find steps column
        steps_cols = [c for c in df_activity.columns if 'step' in c.lower()]
        logger.info(f"  Steps columns: {steps_cols}")
        
        if not steps_cols:
            self.add_issue("CRITICAL", "STEPS_EXTRACTION",
                          "No steps column found in daily_activity.csv")
            return False
        
        steps_col = steps_cols[0]
        steps_data = df_activity[steps_col]
        days_with_steps = steps_data.notna().sum()
        
        logger.info(f"  Days with steps: {days_with_steps}/{len(df_activity)} ({days_with_steps/len(df_activity)*100:.1f}%)")
        
        # Check 2: Unified CSV
        unified_file = self.joined_dir / "features_daily_unified.csv"
        
        if not unified_file.exists():
            self.add_issue("WARNING", "STEPS_JOIN",
                          f"Unified CSV not found: {unified_file}")
        else:
            df_unified = pd.read_csv(unified_file)
            steps_unified_cols = [c for c in df_unified.columns if 'step' in c.lower()]
            logger.info(f"✓ features_daily_unified.csv: {len(df_unified)} days, Steps columns: {steps_unified_cols}")
            
            # Check for all-NaN columns
            for col in steps_unified_cols:
                if df_unified[col].isna().all():
                    self.add_issue("CRITICAL", "STEPS_DATA_QUALITY",
                                  f"Column '{col}' is all NaN in unified CSV")
        
        # Check 3: Labeled CSV
        labeled_file = self.joined_dir / "features_daily_labeled.csv"
        
        if labeled_file.exists():
            df_labeled = pd.read_csv(labeled_file)
            steps_labeled_cols = [c for c in df_labeled.columns if 'step' in c.lower()]
            logger.info(f"✓ features_daily_labeled.csv: {len(df_labeled)} days, Steps columns: {steps_labeled_cols}")
        
        # Compute per-day statistics
        logger.info(f"\nComputing per-day statistics...")
        
        for _, row in df_activity.iterrows():
            if 'date' in row:
                steps_value = row[steps_col] if pd.notna(row[steps_col]) else 0
                self.per_day_results.append({
                    'date': row['date'],
                    'steps': steps_value,
                    'has_data': pd.notna(row[steps_col])
                })
        
        # Summary statistics
        valid_steps = steps_data[steps_data.notna()]
        if len(valid_steps) > 0:
            logger.info(f"\nSteps Summary Statistics:")
            logger.info(f"  Mean steps/day: {valid_steps.mean():.0f} ± {valid_steps.std():.0f}")
            logger.info(f"  Median steps/day: {valid_steps.median():.0f}")
            logger.info(f"  Range: {valid_steps.min():.0f} to {valid_steps.max():.0f}")
            logger.info(f"  Days with data: {len(valid_steps)}/{len(df_activity)} ({len(valid_steps)/len(df_activity)*100:.1f}%)")
        
        return self.audit_results["pass"]
    
    def run_sleep_audit(self) -> bool:
        """
        Run Sleep feature integrity audit.
        
        Checks:
        - daily_sleep.csv exists with sleep_hours
        - Sleep hours in plausible range (0-24)
        - Sleep efficiency in range [0, 1]
        - Consistency across extraction → unification → labeling
        
        Returns:
            True if all checks pass, False otherwise
        """
        self.banner(f"SLEEP FEATURE INTEGRITY AUDIT: {self.participant}/{self.snapshot}")
        
        # Check 1: daily_sleep.csv
        apple_dir = self.extracted_dir / "apple"
        sleep_file = apple_dir / "daily_sleep.csv"
        
        if not sleep_file.exists():
            self.add_issue("CRITICAL", "SLEEP_EXTRACTION",
                          f"daily_sleep.csv not found: {sleep_file}")
            return False
        
        df_sleep = pd.read_csv(sleep_file)
        logger.info(f"✓ daily_sleep.csv: {len(df_sleep)} days")
        
        # Find sleep columns
        sleep_cols = [c for c in df_sleep.columns if 'sleep' in c.lower()]
        logger.info(f"  Sleep columns: {sleep_cols}")
        
        # Check for sleep_hours or similar
        sleep_hours_col = None
        for col in sleep_cols:
            if 'hour' in col.lower() or 'duration' in col.lower():
                sleep_hours_col = col
                break
        
        if not sleep_hours_col:
            self.add_issue("WARNING", "SLEEP_EXTRACTION",
                          "No sleep hours column found in daily_sleep.csv")
        else:
            sleep_hours = df_sleep[sleep_hours_col]
            days_with_sleep = sleep_hours[sleep_hours > 0].count()
            
            logger.info(f"  Days with sleep: {days_with_sleep}/{len(df_sleep)} ({days_with_sleep/len(df_sleep)*100:.1f}%)")
            
            # Check range
            invalid_sleep = sleep_hours[(sleep_hours < 0) | (sleep_hours > 24)]
            if len(invalid_sleep) > 0:
                self.add_issue("CRITICAL", "SLEEP_DATA_QUALITY",
                              f"Found {len(invalid_sleep)} days with invalid sleep hours (< 0 or > 24)")
        
        # Check sleep efficiency if present
        efficiency_col = None
        for col in df_sleep.columns:
            if 'efficiency' in col.lower():
                efficiency_col = col
                break
        
        if efficiency_col:
            efficiency = df_sleep[efficiency_col]
            invalid_efficiency = efficiency[(efficiency < 0) | (efficiency > 1)]
            if len(invalid_efficiency) > 0:
                self.add_issue("WARNING", "SLEEP_DATA_QUALITY",
                              f"Found {len(invalid_efficiency)} days with invalid sleep efficiency (< 0 or > 1)")
        
        # Check 2: Unified CSV
        unified_file = self.joined_dir / "features_daily_unified.csv"
        
        if not unified_file.exists():
            self.add_issue("WARNING", "SLEEP_JOIN",
                          f"Unified CSV not found: {unified_file}")
        else:
            df_unified = pd.read_csv(unified_file)
            sleep_unified_cols = [c for c in df_unified.columns if 'sleep' in c.lower()]
            logger.info(f"✓ features_daily_unified.csv: {len(df_unified)} days, Sleep columns: {sleep_unified_cols}")
        
        # Check 3: Labeled CSV
        labeled_file = self.joined_dir / "features_daily_labeled.csv"
        
        if labeled_file.exists():
            df_labeled = pd.read_csv(labeled_file)
            sleep_labeled_cols = [c for c in df_labeled.columns if 'sleep' in c.lower()]
            logger.info(f"✓ features_daily_labeled.csv: {len(df_labeled)} days, Sleep columns: {sleep_labeled_cols}")
        
        # Compute per-day statistics
        if sleep_hours_col:
            logger.info(f"\nComputing per-day statistics...")
            
            for _, row in df_sleep.iterrows():
                if 'date' in row:
                    sleep_value = row[sleep_hours_col] if pd.notna(row[sleep_hours_col]) else 0
                    efficiency_value = row[efficiency_col] if efficiency_col and pd.notna(row.get(efficiency_col)) else None
                    
                    result = {
                        'date': row['date'],
                        'sleep_hours': sleep_value,
                        'has_sleep': sleep_value > 0,
                        'is_valid_range': 0 <= sleep_value <= 24
                    }
                    
                    if efficiency_value is not None:
                        result['sleep_efficiency'] = efficiency_value
                    
                    self.per_day_results.append(result)
            
            # Summary statistics
            valid_sleep = df_sleep[sleep_hours_col]
            valid_sleep = valid_sleep[(valid_sleep > 0) & (valid_sleep <= 24)]
            
            if len(valid_sleep) > 0:
                logger.info(f"\nSleep Summary Statistics:")
                logger.info(f"  Mean sleep hours: {valid_sleep.mean():.2f} ± {valid_sleep.std():.2f}")
                logger.info(f"  Median sleep hours: {valid_sleep.median():.2f}")
                logger.info(f"  Range: {valid_sleep.min():.2f} to {valid_sleep.max():.2f}")
                logger.info(f"  Days with sleep: {len(valid_sleep)}/{len(df_sleep)} ({len(valid_sleep)/len(df_sleep)*100:.1f}%)")
                
                if efficiency_col:
                    valid_efficiency = df_sleep[efficiency_col]
                    valid_efficiency = valid_efficiency[(valid_efficiency >= 0) & (valid_efficiency <= 1)]
                    if len(valid_efficiency) > 0:
                        logger.info(f"  Mean sleep efficiency: {valid_efficiency.mean():.2f} ± {valid_efficiency.std():.2f}")
        
        return self.audit_results["pass"]
    
    # ========================================================================
    # MEDS DOMAIN AUDIT (Phase 1)
    # ========================================================================
    
    def run_meds_audit(self) -> bool:
        """
        Run Medications feature integrity audit.
        
        Checks:
        - At least one meds source file exists (apple or autoexport)
        - Schema validation for available meds CSVs
        - Unified layer has expected meds columns
        - Data quality: med_any ∈ {0,1}, med_event_count >= 0, etc.
        
        Returns:
            True if all checks pass, False otherwise
        """
        self.banner(f"MEDS FEATURE INTEGRITY AUDIT: {self.participant}/{self.snapshot}")
        
        apple_dir = self.extracted_dir / "apple"
        meds_apple_file = apple_dir / "daily_meds_apple.csv"
        meds_autoexport_file = apple_dir / "daily_meds_autoexport.csv"
        unified_file = self.joined_dir / "features_daily_unified.csv"
        
        # Track found sources
        meds_sources_found = []
        df_meds_combined = None
        
        # ---- Check 1: Existence ----
        logger.info("[MEDS] Checking source file existence...")
        
        if meds_apple_file.exists():
            meds_sources_found.append("apple_export")
            logger.info(f"  ✓ Found: {meds_apple_file.name}")
        else:
            logger.info(f"  ✗ Not found: {meds_apple_file.name}")
        
        if meds_autoexport_file.exists():
            meds_sources_found.append("apple_autoexport")
            logger.info(f"  ✓ Found: {meds_autoexport_file.name}")
        else:
            logger.info(f"  ✗ Not found: {meds_autoexport_file.name}")
        
        if not meds_sources_found:
            self.add_issue("CRITICAL", "MEDS_EXISTENCE",
                          "No meds source files found (neither daily_meds_apple.csv nor daily_meds_autoexport.csv)")
            return False
        
        logger.info(f"  Sources found: {meds_sources_found}")
        
        # ---- Check 2: Schema validation for each source ----
        logger.info("\n[MEDS] Schema validation...")
        
        required_meds_cols = ["date", "med_any", "med_event_count", "med_names", "med_sources"]
        optional_meds_cols = ["med_dose_total"]
        
        for source in meds_sources_found:
            if source == "apple_export":
                csv_path = meds_apple_file
            else:
                csv_path = meds_autoexport_file
            
            df = pd.read_csv(csv_path)
            logger.info(f"  {csv_path.name}: {len(df)} rows, columns: {list(df.columns)}")
            
            missing_cols = [c for c in required_meds_cols if c not in df.columns]
            if missing_cols:
                self.add_issue("CRITICAL", "MEDS_SCHEMA",
                              f"Missing required columns in {csv_path.name}: {missing_cols}")
            else:
                logger.info(f"    ✓ All required columns present")
            
            # Combine for later analysis
            df["_source"] = source
            if df_meds_combined is None:
                df_meds_combined = df
            else:
                df_meds_combined = pd.concat([df_meds_combined, df], ignore_index=True)
        
        # ---- Check 3: Unified layer validation ----
        logger.info("\n[MEDS] Unified layer validation...")
        
        if not unified_file.exists():
            self.add_issue("CRITICAL", "MEDS_UNIFIED",
                          f"Unified CSV not found: {unified_file}")
            return False
        
        df_unified = pd.read_csv(unified_file)
        
        unified_meds_required = ["med_any", "med_event_count", "med_names", "med_sources", "med_vendor"]
        unified_meds_optional = ["med_dose_total"]
        
        missing_unified = [c for c in unified_meds_required if c not in df_unified.columns]
        if missing_unified:
            self.add_issue("CRITICAL", "MEDS_UNIFIED_SCHEMA",
                          f"Missing meds columns in unified CSV: {missing_unified}")
        else:
            logger.info(f"  ✓ All required meds columns present in unified CSV")
        
        # Check optional columns
        has_dose_total = "med_dose_total" in df_unified.columns
        logger.info(f"  med_dose_total present: {has_dose_total}")
        
        # ---- Check 4: Data quality checks on unified ----
        logger.info("\n[MEDS] Data quality checks...")
        
        # Filter to rows with any meds data
        df_meds_unified = df_unified[df_unified["med_any"].notna()].copy()
        n_meds_days = len(df_meds_unified)
        n_total_days = len(df_unified)
        
        logger.info(f"  Days with meds data: {n_meds_days}/{n_total_days}")
        
        if n_meds_days == 0:
            self.add_issue("WARNING", "MEDS_COVERAGE",
                          "No days with meds data in unified CSV")
        else:
            # Check med_any ∈ {0, 1}
            invalid_med_any = df_meds_unified[~df_meds_unified["med_any"].isin([0, 1])]
            if len(invalid_med_any) > 0:
                pct = 100 * len(invalid_med_any) / n_meds_days
                msg = f"med_any has invalid values (not 0 or 1): {len(invalid_med_any)} rows ({pct:.1f}%)"
                if pct > 1:
                    self.add_issue("CRITICAL", "MEDS_DATA_QUALITY", msg)
                else:
                    self.add_issue("WARNING", "MEDS_DATA_QUALITY", msg)
            else:
                logger.info("  ✓ med_any values all in {0, 1}")
            
            # Check med_event_count >= 0
            invalid_count = df_meds_unified[
                df_meds_unified["med_event_count"].notna() & 
                (df_meds_unified["med_event_count"] < 0)
            ]
            if len(invalid_count) > 0:
                pct = 100 * len(invalid_count) / n_meds_days
                msg = f"med_event_count has negative values: {len(invalid_count)} rows ({pct:.1f}%)"
                if pct > 1:
                    self.add_issue("CRITICAL", "MEDS_DATA_QUALITY", msg)
                else:
                    self.add_issue("WARNING", "MEDS_DATA_QUALITY", msg)
            else:
                logger.info("  ✓ med_event_count all >= 0")
            
            # Check med_dose_total >= 0 if present
            if has_dose_total:
                invalid_dose = df_meds_unified[
                    df_meds_unified["med_dose_total"].notna() & 
                    (df_meds_unified["med_dose_total"] < 0)
                ]
                if len(invalid_dose) > 0:
                    pct = 100 * len(invalid_dose) / n_meds_days
                    msg = f"med_dose_total has negative values: {len(invalid_dose)} rows ({pct:.1f}%)"
                    if pct > 1:
                        self.add_issue("CRITICAL", "MEDS_DATA_QUALITY", msg)
                    else:
                        self.add_issue("WARNING", "MEDS_DATA_QUALITY", msg)
                else:
                    logger.info("  ✓ med_dose_total all >= 0")
            
            # When med_any == 1: med_event_count > 0 and med_names not empty
            med_any_1 = df_meds_unified[df_meds_unified["med_any"] == 1]
            if len(med_any_1) > 0:
                # Check event count > 0
                invalid_count_when_med = med_any_1[
                    med_any_1["med_event_count"].notna() & 
                    (med_any_1["med_event_count"] <= 0)
                ]
                if len(invalid_count_when_med) > 0:
                    pct = 100 * len(invalid_count_when_med) / len(med_any_1)
                    msg = f"med_any=1 but med_event_count<=0: {len(invalid_count_when_med)} rows ({pct:.1f}%)"
                    if pct > 1:
                        self.add_issue("CRITICAL", "MEDS_DATA_QUALITY", msg)
                    else:
                        self.add_issue("WARNING", "MEDS_DATA_QUALITY", msg)
                
                # Check med_names not empty
                empty_names = med_any_1[
                    med_any_1["med_names"].isna() | 
                    (med_any_1["med_names"].astype(str).str.strip() == "")
                ]
                if len(empty_names) > 0:
                    pct = 100 * len(empty_names) / len(med_any_1)
                    msg = f"med_any=1 but med_names empty: {len(empty_names)} rows ({pct:.1f}%)"
                    if pct > 1:
                        self.add_issue("CRITICAL", "MEDS_DATA_QUALITY", msg)
                    else:
                        self.add_issue("WARNING", "MEDS_DATA_QUALITY", msg)
            
            # Check med_vendor values
            valid_vendors = {"apple_export", "apple_autoexport", "fallback"}
            vendor_values = df_meds_unified["med_vendor"].dropna().unique()
            invalid_vendors = [v for v in vendor_values if v not in valid_vendors]
            if invalid_vendors:
                self.add_issue("CRITICAL", "MEDS_DATA_QUALITY",
                              f"Invalid med_vendor values: {invalid_vendors}")
            else:
                logger.info(f"  ✓ med_vendor values valid: {list(vendor_values)}")
            
            # Vendor distribution
            vendor_dist = df_meds_unified["med_vendor"].value_counts()
            logger.info("\n[MEDS] Vendor distribution:")
            for vendor, count in vendor_dist.items():
                logger.info(f"    {vendor}: {count} days ({100*count/n_meds_days:.1f}%)")
        
        # ---- Compute per-day results ----
        logger.info("\n[MEDS] Computing per-day statistics...")
        
        for _, row in df_meds_unified.iterrows():
            has_meds = row["med_any"] == 1 if pd.notna(row["med_any"]) else False
            is_valid = True
            
            # Validation flags
            if pd.notna(row["med_any"]) and row["med_any"] not in [0, 1]:
                is_valid = False
            if pd.notna(row["med_event_count"]) and row["med_event_count"] < 0:
                is_valid = False
            if has_dose_total and pd.notna(row["med_dose_total"]) and row["med_dose_total"] < 0:
                is_valid = False
            
            result = {
                'date': row['date'],
                'med_any': row['med_any'],
                'med_event_count': row['med_event_count'],
                'med_names': row['med_names'],
                'med_vendor': row['med_vendor'],
                'has_meds': has_meds,
                'is_valid': is_valid
            }
            if has_dose_total:
                result['med_dose_total'] = row['med_dose_total']
            
            self.per_day_results.append(result)
        
        # ---- Summary statistics ----
        n_days_with_meds = len(df_meds_unified[df_meds_unified["med_any"] == 1])
        
        logger.info(f"\n[MEDS] Summary Statistics:")
        logger.info(f"  Total days in unified: {n_total_days}")
        logger.info(f"  Days with meds data: {n_meds_days}")
        logger.info(f"  Days with med_any=1: {n_days_with_meds} ({100*n_days_with_meds/n_total_days:.1f}%)")
        
        if n_meds_days > 0:
            date_col = pd.to_datetime(df_meds_unified["date"])
            logger.info(f"  Date range: {date_col.min().strftime('%Y-%m-%d')} to {date_col.max().strftime('%Y-%m-%d')}")
        
        # Store summary stats
        self.audit_results["meds_stats"] = {
            "n_days_total": n_total_days,
            "n_days_with_meds": n_days_with_meds,
            "n_meds_rows": n_meds_days,
            "sources_found": meds_sources_found,
            "vendor_distribution": vendor_dist.to_dict() if n_meds_days > 0 else {}
        }
        
        return self.audit_results["pass"]
    
    # ========================================================================
    # SOM DOMAIN AUDIT (Phase 1)
    # ========================================================================
    
    def run_som_audit(self) -> bool:
        """
        Run State of Mind (SoM) feature integrity audit.
        
        Checks:
        - SoM source file exists OR SoM columns in unified
        - Schema validation for daily_som_autoexport.csv
        - Unified layer has expected SoM columns
        - Value ranges: som_category_3class ∈ {-1, 0, 1}, scores in [-1, 1]
        - Temporal coverage analysis
        
        Returns:
            True if all checks pass, False otherwise
        """
        self.banner(f"SOM FEATURE INTEGRITY AUDIT: {self.participant}/{self.snapshot}")
        
        apple_dir = self.extracted_dir / "apple"
        som_autoexport_file = apple_dir / "daily_som_autoexport.csv"
        unified_file = self.joined_dir / "features_daily_unified.csv"
        
        # ---- Check 1: Existence ----
        logger.info("[SOM] Checking source file existence...")
        
        som_source_exists = som_autoexport_file.exists()
        if som_source_exists:
            logger.info(f"  ✓ Found: {som_autoexport_file.name}")
        else:
            logger.info(f"  ✗ Not found: {som_autoexport_file.name}")
        
        # Load unified to check if SoM columns exist there
        if not unified_file.exists():
            self.add_issue("CRITICAL", "SOM_UNIFIED",
                          f"Unified CSV not found: {unified_file}")
            return False
        
        df_unified = pd.read_csv(unified_file)
        
        unified_som_required = [
            "som_mean_score", "som_last_score", "som_n_entries",
            "som_category_3class", "som_kind_dominant", "som_labels",
            "som_associations", "som_vendor"
        ]
        
        som_cols_in_unified = [c for c in unified_som_required if c in df_unified.columns]
        
        if not som_source_exists and len(som_cols_in_unified) == 0:
            self.add_issue("CRITICAL", "SOM_EXISTENCE",
                          "No SoM source file found AND no SoM columns in unified CSV")
            return False
        
        # ---- Check 2: Schema validation for source file ----
        if som_source_exists:
            logger.info("\n[SOM] Schema validation for source file...")
            
            df_som_source = pd.read_csv(som_autoexport_file)
            logger.info(f"  {som_autoexport_file.name}: {len(df_som_source)} rows")
            logger.info(f"  Columns: {list(df_som_source.columns)}")
            
            required_som_cols = [
                "date", "som_mean_score", "som_last_score", "som_n_entries",
                "som_category_3class", "som_kind_dominant", "som_labels", "som_associations"
            ]
            
            missing_cols = [c for c in required_som_cols if c not in df_som_source.columns]
            if missing_cols:
                self.add_issue("CRITICAL", "SOM_SCHEMA",
                              f"Missing required columns in {som_autoexport_file.name}: {missing_cols}")
            else:
                logger.info("  ✓ All required columns present in source file")
        
        # ---- Check 3: Unified layer validation ----
        logger.info("\n[SOM] Unified layer validation...")
        
        missing_unified = [c for c in unified_som_required if c not in df_unified.columns]
        if missing_unified:
            self.add_issue("CRITICAL", "SOM_UNIFIED_SCHEMA",
                          f"Missing SoM columns in unified CSV: {missing_unified}")
        else:
            logger.info("  ✓ All required SoM columns present in unified CSV")
        
        # ---- Check 4: Value range checks ----
        logger.info("\n[SOM] Data quality checks...")
        
        # Filter to rows with SoM data
        df_som_unified = df_unified[df_unified["som_n_entries"].notna()].copy()
        n_som_days = len(df_som_unified)
        n_total_days = len(df_unified)
        
        logger.info(f"  Days with SoM data: {n_som_days}/{n_total_days}")
        
        if n_som_days == 0:
            self.add_issue("WARNING", "SOM_COVERAGE",
                          "No days with SoM data in unified CSV")
        else:
            # Check som_category_3class ∈ {-1, 0, 1}
            valid_categories = {-1, 0, 1}
            cat_values = df_som_unified["som_category_3class"].dropna()
            invalid_cat = cat_values[~cat_values.isin(valid_categories)]
            if len(invalid_cat) > 0:
                pct = 100 * len(invalid_cat) / len(cat_values)
                msg = f"som_category_3class has invalid values: {invalid_cat.unique().tolist()} ({len(invalid_cat)} rows, {pct:.1f}%)"
                if pct > 1:
                    self.add_issue("CRITICAL", "SOM_DATA_QUALITY", msg)
                else:
                    self.add_issue("WARNING", "SOM_DATA_QUALITY", msg)
            else:
                logger.info(f"  ✓ som_category_3class values valid: {sorted(cat_values.unique().tolist())}")
            
            # Check som_n_entries >= 1
            invalid_entries = df_som_unified[
                df_som_unified["som_n_entries"].notna() & 
                (df_som_unified["som_n_entries"] < 1)
            ]
            if len(invalid_entries) > 0:
                pct = 100 * len(invalid_entries) / n_som_days
                msg = f"som_n_entries < 1: {len(invalid_entries)} rows ({pct:.1f}%)"
                if pct > 1:
                    self.add_issue("CRITICAL", "SOM_DATA_QUALITY", msg)
                else:
                    self.add_issue("WARNING", "SOM_DATA_QUALITY", msg)
            else:
                logger.info("  ✓ som_n_entries all >= 1")
            
            # Check som_vendor values
            valid_vendors = {"apple_autoexport", "fallback"}
            vendor_values = df_som_unified["som_vendor"].dropna().unique()
            invalid_vendors = [v for v in vendor_values if v not in valid_vendors]
            if invalid_vendors:
                self.add_issue("CRITICAL", "SOM_DATA_QUALITY",
                              f"Invalid som_vendor values: {invalid_vendors}")
            else:
                logger.info(f"  ✓ som_vendor values valid: {list(vendor_values)}")
            
            # Check score ranges (if scores are present)
            for score_col in ["som_mean_score", "som_last_score"]:
                if score_col in df_som_unified.columns:
                    scores = df_som_unified[score_col].dropna()
                    if len(scores) > 0:
                        out_of_range = scores[(scores < -1) | (scores > 1)]
                        if len(out_of_range) > 0:
                            pct = 100 * len(out_of_range) / len(scores)
                            msg = f"{score_col} outside [-1, 1]: {len(out_of_range)} values ({pct:.1f}%)"
                            self.add_issue("WARNING", "SOM_DATA_QUALITY", msg)
                        else:
                            logger.info(f"  ✓ {score_col} all in [-1, 1] range")
            
            # Vendor distribution
            vendor_dist = df_som_unified["som_vendor"].value_counts()
            logger.info("\n[SOM] Vendor distribution:")
            for vendor, count in vendor_dist.items():
                logger.info(f"    {vendor}: {count} days ({100*count/n_som_days:.1f}%)")
        
        # ---- Check 5: Temporal coverage ----
        logger.info("\n[SOM] Temporal coverage analysis...")
        
        if n_som_days > 0:
            df_som_unified["date"] = pd.to_datetime(df_som_unified["date"])
            df_som_unified = df_som_unified.sort_values("date")
            
            first_date = df_som_unified["date"].min()
            last_date = df_som_unified["date"].max()
            
            logger.info(f"  First SoM date: {first_date.strftime('%Y-%m-%d')}")
            logger.info(f"  Last SoM date: {last_date.strftime('%Y-%m-%d')}")
            logger.info(f"  Total days with SoM: {n_som_days}")
            
            # Compute max gap
            if n_som_days > 1:
                date_diffs = df_som_unified["date"].diff().dropna()
                max_gap = date_diffs.max().days
                logger.info(f"  Max gap between SoM entries: {max_gap} days")
                
                if max_gap > 90:
                    self.add_issue("WARNING", "SOM_TEMPORAL",
                                  f"Large gap in SoM data: {max_gap} days (> 90 days)")
            else:
                max_gap = 0
        else:
            first_date = None
            last_date = None
            max_gap = None
        
        # ---- Compute per-day results ----
        logger.info("\n[SOM] Computing per-day statistics...")
        
        df_unified["date"] = pd.to_datetime(df_unified["date"])
        
        for _, row in df_unified.iterrows():
            has_som = pd.notna(row.get("som_n_entries")) and row["som_n_entries"] >= 1
            is_valid = True
            
            # Validation flags
            if has_som:
                if pd.notna(row.get("som_category_3class")) and row["som_category_3class"] not in [-1, 0, 1]:
                    is_valid = False
                if pd.notna(row.get("som_n_entries")) and row["som_n_entries"] < 1:
                    is_valid = False
            
            self.per_day_results.append({
                'date': row['date'].strftime('%Y-%m-%d') if hasattr(row['date'], 'strftime') else row['date'],
                'som_mean_score': row.get('som_mean_score'),
                'som_last_score': row.get('som_last_score'),
                'som_n_entries': row.get('som_n_entries'),
                'som_category_3class': row.get('som_category_3class'),
                'som_vendor': row.get('som_vendor'),
                'has_som': has_som,
                'is_valid': is_valid
            })
        
        # ---- Summary statistics ----
        logger.info(f"\n[SOM] Summary Statistics:")
        logger.info(f"  Total days in unified: {n_total_days}")
        logger.info(f"  Days with SoM data: {n_som_days} ({100*n_som_days/n_total_days:.1f}%)")
        
        # Category distribution
        if n_som_days > 0:
            cat_dist = df_som_unified["som_category_3class"].value_counts()
            logger.info("  Category distribution:")
            for cat, count in sorted(cat_dist.items()):
                cat_label = {-1: "negative", 0: "neutral", 1: "positive"}.get(cat, str(cat))
                logger.info(f"    {cat} ({cat_label}): {count} days ({100*count/n_som_days:.1f}%)")
        
        # Store summary stats
        self.audit_results["som_stats"] = {
            "n_days_total": n_total_days,
            "n_days_with_som": n_som_days,
            "first_date": first_date.strftime('%Y-%m-%d') if first_date else None,
            "last_date": last_date.strftime('%Y-%m-%d') if last_date else None,
            "max_gap_days": max_gap,
            "vendor_distribution": vendor_dist.to_dict() if n_som_days > 0 else {},
            "category_distribution": cat_dist.to_dict() if n_som_days > 0 else {}
        }
        
        return self.audit_results["pass"]
    
    # ========================================================================
    # UNIFIED EXTENSION AUDIT (Phase 1)
    # ========================================================================
    
    def run_unified_extension_audit(self) -> bool:
        """
        Run unified dataset extension QC.
        
        Validates that the extended unified dataset is structurally healthy:
        - File exists
        - No duplicate dates
        - Monotonic date ordering
        - Vendor columns have valid values
        - No all-NaN columns among key meds/SoM features
        
        Returns:
            True if all checks pass, False otherwise
        """
        self.banner(f"UNIFIED EXTENSION AUDIT: {self.participant}/{self.snapshot}")
        
        unified_file = self.joined_dir / "features_daily_unified.csv"
        
        # ---- Check 1: Existence ----
        logger.info("[UNIFIED-EXT] Checking file existence...")
        
        if not unified_file.exists():
            self.add_issue("CRITICAL", "UNIFIED_EXISTENCE",
                          f"Unified CSV not found: {unified_file}")
            return False
        
        df = pd.read_csv(unified_file)
        logger.info(f"  ✓ Loaded: {len(df)} rows, {len(df.columns)} columns")
        
        # ---- Check 2: No duplicate dates ----
        logger.info("\n[UNIFIED-EXT] Checking for duplicate dates...")
        
        df["date"] = pd.to_datetime(df["date"])
        duplicates = df[df.duplicated(subset=["date"], keep=False)]
        if len(duplicates) > 0:
            self.add_issue("CRITICAL", "UNIFIED_DUPLICATES",
                          f"Duplicate dates found: {len(duplicates)} rows",
                          {"dates": duplicates["date"].dt.strftime('%Y-%m-%d').unique().tolist()[:10]})
        else:
            logger.info("  ✓ No duplicate dates")
        
        # ---- Check 3: Monotonic ordering ----
        logger.info("\n[UNIFIED-EXT] Checking date ordering...")
        
        if not df["date"].is_monotonic_increasing:
            self.add_issue("CRITICAL", "UNIFIED_ORDERING",
                          "Date column is not monotonically increasing")
        else:
            logger.info("  ✓ Dates are monotonically increasing")
        
        # ---- Check 4: Vendor columns validation ----
        logger.info("\n[UNIFIED-EXT] Checking vendor columns...")
        
        vendor_checks = {
            "med_vendor": {"apple_export", "apple_autoexport", "fallback"},
            "som_vendor": {"apple_autoexport", "fallback"}
        }
        
        for col, valid_values in vendor_checks.items():
            if col in df.columns:
                values = df[col].dropna().unique()
                invalid = [v for v in values if v not in valid_values]
                if invalid:
                    self.add_issue("CRITICAL", "UNIFIED_VENDOR",
                                  f"Invalid {col} values: {invalid}")
                else:
                    logger.info(f"  ✓ {col} values valid: {list(values)}")
            else:
                logger.info(f"  ⚠️ {col} column not present")
        
        # ---- Check 5: No all-NaN key columns ----
        logger.info("\n[UNIFIED-EXT] Checking for all-NaN columns...")
        
        key_cols = [
            "med_any", "med_event_count", "med_names",
            "som_category_3class", "som_mean_score", "som_last_score"
        ]
        
        all_nan_cols = []
        for col in key_cols:
            if col in df.columns:
                if df[col].isna().all():
                    all_nan_cols.append(col)
                    logger.info(f"  ⚠️ {col} is all NaN")
                else:
                    non_null = df[col].notna().sum()
                    logger.info(f"  ✓ {col}: {non_null} non-null values")
            else:
                logger.info(f"  - {col}: column not present")
        
        if all_nan_cols:
            self.add_issue("WARNING", "UNIFIED_ALL_NAN",
                          f"Columns with all NaN values: {all_nan_cols}")
        
        # ---- Phase 2: Vendor coverage stats ----
        logger.info("\n[UNIFIED-EXT] Vendor coverage analysis (Phase 2)...")
        
        n_total = len(df)
        
        # Meds vendor stats
        if "med_vendor" in df.columns:
            med_vendor_dist = df["med_vendor"].value_counts(dropna=True)
            logger.info("  med_vendor distribution:")
            for vendor, count in med_vendor_dist.items():
                logger.info(f"    {vendor}: {count} days ({100*count/n_total:.1f}%)")
        
        # SoM vendor stats
        if "som_vendor" in df.columns:
            som_vendor_dist = df["som_vendor"].value_counts(dropna=True)
            logger.info("  som_vendor distribution:")
            for vendor, count in som_vendor_dist.items():
                logger.info(f"    {vendor}: {count} days ({100*count/n_total:.1f}%)")
        
        # Overlap analysis
        has_meds = df["med_any"].notna() if "med_any" in df.columns else pd.Series([False] * n_total)
        has_som = df["som_n_entries"].notna() if "som_n_entries" in df.columns else pd.Series([False] * n_total)
        
        days_meds_only = (has_meds & ~has_som).sum()
        days_som_only = (~has_meds & has_som).sum()
        days_both = (has_meds & has_som).sum()
        days_neither = (~has_meds & ~has_som).sum()
        
        logger.info("\n  Domain overlap:")
        logger.info(f"    Days with meds only: {days_meds_only}")
        logger.info(f"    Days with SoM only: {days_som_only}")
        logger.info(f"    Days with both: {days_both}")
        logger.info(f"    Days with neither: {days_neither}")
        
        # Store stats
        self.audit_results["unified_extension_stats"] = {
            "n_total_days": n_total,
            "n_columns": len(df.columns),
            "med_vendor_distribution": med_vendor_dist.to_dict() if "med_vendor" in df.columns else {},
            "som_vendor_distribution": som_vendor_dist.to_dict() if "som_vendor" in df.columns else {},
            "overlap": {
                "meds_only": int(days_meds_only),
                "som_only": int(days_som_only),
                "both": int(days_both),
                "neither": int(days_neither)
            }
        }
        
        return self.audit_results["pass"]
    
    # ========================================================================
    # LABEL LAYER QC (Phase 2)
    # ========================================================================
    
    def run_labels_audit(self) -> bool:
        """
        Run Label layer QC (Phase 2).
        
        Descriptive QC for features_daily_labeled.csv:
        - Check if file exists
        - Verify label columns exist (pbsi_score, label_3cls, segment_id)
        - Distribution of label_3cls
        - Ratio of labeled vs unlabeled days
        
        NOTE: This is descriptive QC only, not judgment of label design.
        
        Returns:
            True if file exists and is valid, False otherwise
        """
        self.banner(f"LABELS LAYER AUDIT: {self.participant}/{self.snapshot}")
        
        labeled_file = self.joined_dir / "features_daily_labeled.csv"
        
        # ---- Check 1: Existence ----
        logger.info("[LABELS] Checking file existence...")
        
        if not labeled_file.exists():
            self.add_issue("WARNING", "LABELS_EXISTENCE",
                          f"Labeled CSV not found: {labeled_file}")
            logger.info("  ⚠️ No labeled file - labels QC skipped")
            return self.audit_results["pass"]
        
        df = pd.read_csv(labeled_file)
        logger.info(f"  ✓ Loaded: {len(df)} rows, {len(df.columns)} columns")
        
        # ---- Check 2: Label columns ----
        logger.info("\n[LABELS] Checking label columns...")
        
        expected_label_cols = ["pbsi_score", "label_3cls", "segment_id"]
        found_cols = []
        missing_cols = []
        
        for col in expected_label_cols:
            if col in df.columns:
                found_cols.append(col)
                logger.info(f"  ✓ {col} present")
            else:
                missing_cols.append(col)
                logger.info(f"  ⚠️ {col} not found")
        
        if missing_cols:
            self.add_issue("WARNING", "LABELS_SCHEMA",
                          f"Missing expected label columns: {missing_cols}")
        
        # ---- Check 3: label_3cls distribution ----
        if "label_3cls" in df.columns:
            logger.info("\n[LABELS] label_3cls distribution:")
            
            label_counts = df["label_3cls"].value_counts(dropna=False)
            n_total = len(df)
            n_labeled = df["label_3cls"].notna().sum()
            n_unlabeled = df["label_3cls"].isna().sum()
            
            logger.info(f"  Total days: {n_total}")
            logger.info(f"  Labeled: {n_labeled} ({100*n_labeled/n_total:.1f}%)")
            logger.info(f"  Unlabeled: {n_unlabeled} ({100*n_unlabeled/n_total:.1f}%)")
            
            logger.info("\n  Class distribution (labeled only):")
            label_dist = df["label_3cls"].dropna().value_counts().sort_index()
            for label, count in label_dist.items():
                pct = 100 * count / n_labeled if n_labeled > 0 else 0
                label_name = {-1: "Low", 0: "Mid", 1: "High"}.get(int(label), str(label))
                logger.info(f"    {int(label)} ({label_name}): {count} days ({pct:.1f}%)")
            
            # Store stats
            self.audit_results["labels_stats"] = {
                "n_total_days": n_total,
                "n_labeled": n_labeled,
                "n_unlabeled": n_unlabeled,
                "label_ratio": n_labeled / n_total if n_total > 0 else 0,
                "class_distribution": {str(int(k)): v for k, v in label_dist.items()}
            }
        
        # ---- Check 4: pbsi_score stats ----
        if "pbsi_score" in df.columns:
            logger.info("\n[LABELS] pbsi_score statistics:")
            
            pbsi = df["pbsi_score"].dropna()
            if len(pbsi) > 0:
                logger.info(f"  Mean: {pbsi.mean():.3f}")
                logger.info(f"  Std: {pbsi.std():.3f}")
                logger.info(f"  Min: {pbsi.min():.3f}")
                logger.info(f"  Max: {pbsi.max():.3f}")
                logger.info(f"  Days with score: {len(pbsi)}")
        
        # ---- Check 5: segment_id stats ----
        if "segment_id" in df.columns:
            logger.info("\n[LABELS] segment_id statistics:")
            
            seg = df["segment_id"].dropna()
            if len(seg) > 0:
                n_segments = seg.nunique()
                logger.info(f"  Unique segments: {n_segments}")
                logger.info(f"  Days with segment: {len(seg)}")
        
        return self.audit_results["pass"]
    
    def save_results(self):
        """Save audit results to CSV and JSON."""
        # Save per-day results to CSV
        if self.per_day_results:
            df_per_day = pd.DataFrame(self.per_day_results)
            csv_path = self.qc_dir / f"{self.domain}_feature_audit.csv"
            df_per_day.to_csv(csv_path, index=False)
            logger.info(f"\n✓ Saved per-day results: {csv_path}")
        
        # Save audit summary to JSON
        json_path = self.qc_dir / f"{self.domain}_audit_summary.json"
        with open(json_path, 'w') as f:
            json.dump(self.audit_results, f, indent=2, default=numpy_json_serializer)
        logger.info(f"✓ Saved audit summary: {json_path}")
        
        # Generate Markdown report (Phase 1)
        self._generate_markdown_report()
    
    def _generate_markdown_report(self):
        """Generate human-readable Markdown QC report under docs/reports/qc/."""
        # Create reports directory
        reports_dir = Path("docs/reports/qc")
        reports_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate timestamp for filename
        timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%SZ")
        report_filename = f"QC_{self.participant}_{self.snapshot}_{self.domain}_{timestamp}.md"
        report_path = reports_dir / report_filename
        
        # Build report content
        lines = []
        lines.append(f"# QC Report: {self.domain.upper()}")
        lines.append("")
        lines.append(f"**Participant:** {self.participant}")
        lines.append(f"**Snapshot:** {self.snapshot}")
        lines.append(f"**Domain:** {self.domain}")
        lines.append(f"**Generated:** {datetime.now().isoformat()}")
        lines.append("")
        
        # Status
        status = "✅ PASS" if self.audit_results["pass"] else "❌ FAIL"
        lines.append(f"## Status: {status}")
        lines.append("")
        
        # Issues
        n_critical = len(self.audit_results["issues"])
        n_warnings = len(self.audit_results["warnings"])
        
        if n_critical > 0:
            lines.append(f"### 🔴 Critical Issues ({n_critical})")
            lines.append("")
            for issue in self.audit_results["issues"]:
                lines.append(f"- **[{issue['category']}]** {issue['description']}")
            lines.append("")
        
        if n_warnings > 0:
            lines.append(f"### ⚠️ Warnings ({n_warnings})")
            lines.append("")
            for warning in self.audit_results["warnings"]:
                lines.append(f"- **[{warning['category']}]** {warning['description']}")
            lines.append("")
        
        # Domain-specific stats
        if self.domain == "meds" and "meds_stats" in self.audit_results:
            stats = self.audit_results["meds_stats"]
            lines.append("## Meds Statistics")
            lines.append("")
            lines.append(f"- **Total days:** {stats.get('n_days_total', 'N/A')}")
            lines.append(f"- **Days with meds:** {stats.get('n_days_with_meds', 'N/A')}")
            lines.append(f"- **Sources found:** {', '.join(stats.get('sources_found', []))}")
            lines.append("")
            
            if stats.get("vendor_distribution"):
                lines.append("### Vendor Distribution")
                lines.append("")
                lines.append("| Vendor | Days | % |")
                lines.append("|--------|------|---|")
                total = stats.get('n_meds_rows', 1)
                for vendor, count in stats["vendor_distribution"].items():
                    pct = 100 * count / total if total > 0 else 0
                    lines.append(f"| {vendor} | {count} | {pct:.1f}% |")
                lines.append("")
        
        elif self.domain == "som" and "som_stats" in self.audit_results:
            stats = self.audit_results["som_stats"]
            lines.append("## SoM Statistics")
            lines.append("")
            lines.append(f"- **Total days:** {stats.get('n_days_total', 'N/A')}")
            lines.append(f"- **Days with SoM:** {stats.get('n_days_with_som', 'N/A')}")
            lines.append(f"- **First SoM date:** {stats.get('first_date', 'N/A')}")
            lines.append(f"- **Last SoM date:** {stats.get('last_date', 'N/A')}")
            lines.append(f"- **Max gap (days):** {stats.get('max_gap_days', 'N/A')}")
            lines.append("")
            
            if stats.get("vendor_distribution"):
                lines.append("### Vendor Distribution")
                lines.append("")
                lines.append("| Vendor | Days | % |")
                lines.append("|--------|------|---|")
                total = stats.get('n_days_with_som', 1)
                for vendor, count in stats["vendor_distribution"].items():
                    pct = 100 * count / total if total > 0 else 0
                    lines.append(f"| {vendor} | {count} | {pct:.1f}% |")
                lines.append("")
            
            if stats.get("category_distribution"):
                lines.append("### Category Distribution")
                lines.append("")
                lines.append("| Category | Label | Days | % |")
                lines.append("|----------|-------|------|---|")
                total = stats.get('n_days_with_som', 1)
                cat_labels = {-1: "negative", 0: "neutral", 1: "positive"}
                for cat, count in sorted(stats["category_distribution"].items(), key=lambda x: int(x[0])):
                    label = cat_labels.get(int(cat), str(cat))
                    pct = 100 * count / total if total > 0 else 0
                    lines.append(f"| {cat} | {label} | {count} | {pct:.1f}% |")
                lines.append("")
        
        elif self.domain == "unified_ext" and "unified_extension_stats" in self.audit_results:
            stats = self.audit_results["unified_extension_stats"]
            lines.append("## Unified Extension Statistics")
            lines.append("")
            lines.append(f"- **Total days:** {stats.get('n_total_days', 'N/A')}")
            lines.append(f"- **Total columns:** {stats.get('n_columns', 'N/A')}")
            lines.append("")
            
            if stats.get("overlap"):
                overlap = stats["overlap"]
                lines.append("### Domain Overlap")
                lines.append("")
                lines.append("| Category | Days |")
                lines.append("|----------|------|")
                lines.append(f"| Meds only | {overlap.get('meds_only', 0)} |")
                lines.append(f"| SoM only | {overlap.get('som_only', 0)} |")
                lines.append(f"| Both | {overlap.get('both', 0)} |")
                lines.append(f"| Neither | {overlap.get('neither', 0)} |")
                lines.append("")
        
        elif self.domain == "labels" and "labels_stats" in self.audit_results:
            stats = self.audit_results["labels_stats"]
            lines.append("## Labels Statistics")
            lines.append("")
            lines.append(f"- **Total days:** {stats.get('n_total_days', 'N/A')}")
            lines.append(f"- **Labeled days:** {stats.get('n_labeled', 'N/A')}")
            lines.append(f"- **Unlabeled days:** {stats.get('n_unlabeled', 'N/A')}")
            lines.append(f"- **Label ratio:** {stats.get('label_ratio', 0)*100:.1f}%")
            lines.append("")
            
            if stats.get("class_distribution"):
                lines.append("### Class Distribution (label_3cls)")
                lines.append("")
                lines.append("| Class | Label | Days | % |")
                lines.append("|-------|-------|------|---|")
                total = stats.get('n_labeled', 1)
                class_labels = {"-1": "Low", "0": "Mid", "1": "High"}
                for cls, count in sorted(stats["class_distribution"].items(), key=lambda x: int(x[0])):
                    label = class_labels.get(cls, cls)
                    pct = 100 * count / total if total > 0 else 0
                    lines.append(f"| {cls} | {label} | {count} | {pct:.1f}% |")
                lines.append("")
        
        # Footer
        lines.append("---")
        lines.append("")
        lines.append(f"*Report generated by `etl_audit.py` at {datetime.now().isoformat()}*")
        
        # Write report
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
        
        logger.info(f"✓ Saved Markdown report: {report_path}")
        
        # Update QC_latest.md (Phase 2)
        self._update_qc_latest(report_path)
    
    def _update_qc_latest(self, report_path: Path):
        """Update QC_latest.md with pointer to most recent QC report."""
        reports_dir = Path("docs/reports/qc")
        latest_path = reports_dir / "QC_latest.md"
        
        # Read existing latest file or create new
        existing_reports = {}
        if latest_path.exists():
            try:
                with open(latest_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    # Parse existing reports (simple key-value extraction)
                    import re
                    for match in re.finditer(r'\*\*(\w+)\*\*:\s*\[([^\]]+)\]\(([^)]+)\)', content):
                        domain = match.group(1).lower()
                        existing_reports[domain] = {
                            'name': match.group(2),
                            'path': match.group(3)
                        }
            except Exception:
                pass
        
        # Update with current report
        existing_reports[self.domain] = {
            'name': report_path.name,
            'path': report_path.name
        }
        
        # Build new latest content
        lines = []
        lines.append("# QC Latest Reports")
        lines.append("")
        lines.append(f"**Last Updated:** {datetime.now().isoformat()}")
        lines.append(f"**Participant:** {self.participant}")
        lines.append(f"**Snapshot:** {self.snapshot}")
        lines.append("")
        lines.append("## Latest Reports by Domain")
        lines.append("")
        
        for domain, info in sorted(existing_reports.items()):
            status_icon = "✅" if self.audit_results["pass"] and domain == self.domain else "📄"
            lines.append(f"- **{domain}**: [{info['name']}]({info['path']}) {status_icon}")
        
        lines.append("")
        lines.append("---")
        lines.append("")
        lines.append("*This file is auto-updated by `etl_audit.py`*")
        
        with open(latest_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
        
        logger.info(f"✓ Updated QC_latest.md")
    
    def print_final_summary(self):
        """Print final PASS/FAIL summary."""
        logger.info("\n" + "="*80)
        logger.info("AUDIT SUMMARY")
        logger.info("="*80 + "\n")
        
        n_critical = len(self.audit_results["issues"])
        n_warnings = len(self.audit_results["warnings"])
        
        if self.audit_results["pass"]:
            logger.info("✅ AUDIT STATUS: PASS")
            logger.info(f"   Domain: {self.domain.upper()}")
            logger.info(f"   Participant: {self.participant}")
            logger.info(f"   Snapshot: {self.snapshot}")
            
            if n_warnings > 0:
                logger.info(f"\n⚠️  {n_warnings} warnings found (non-blocking):")
                for warning in self.audit_results["warnings"]:
                    logger.info(f"   - [{warning['category']}] {warning['description']}")
        else:
            logger.info("❌ AUDIT STATUS: FAIL")
            logger.info(f"   Domain: {self.domain.upper()}")
            logger.info(f"   Participant: {self.participant}")
            logger.info(f"   Snapshot: {self.snapshot}")
            logger.info(f"\n🔴 {n_critical} CRITICAL issues found:")
            for issue in self.audit_results["issues"]:
                logger.info(f"   - [{issue['category']}] {issue['description']}")
            
            if n_warnings > 0:
                logger.info(f"\n⚠️  {n_warnings} warnings found:")
                for warning in self.audit_results["warnings"]:
                    logger.info(f"   - [{warning['category']}] {warning['description']}")
        
        logger.info("\n" + "="*80)
    
    # ========================================================================
    # MAIN AUDIT RUNNER
    # ========================================================================
    
    def run_full_audit(self):
        """Run complete ETL audit."""
        print("\n" + "="*80)
        print(f"ETL AUDIT: {self.participant} / {self.snapshot}")
        print("="*80)
        
        self.audit_raw_files()
        self.audit_extracted_layer()
        self.audit_unified_join()
        self.audit_sample_days()
        
        # Summary
        self.banner("AUDIT SUMMARY")
        
        n_critical = len(self.audit_results["issues"])
        n_warnings = len(self.audit_results["warnings"])
        
        if n_critical == 0 and n_warnings == 0:
            logger.info("✅ NO ISSUES FOUND - ETL pipeline appears healthy")
        else:
            if n_critical > 0:
                logger.info(f"🔴 {n_critical} CRITICAL issues found")
                for issue in self.audit_results["issues"]:
                    logger.info(f"   - [{issue['category']}] {issue['description']}")
            
            if n_warnings > 0:
                logger.info(f"⚠️  {n_warnings} warnings found")
                for warning in self.audit_results["warnings"]:
                    logger.info(f"   - [{warning['category']}] {warning['description']}")
        
        return self.audit_results


def main():
    """CLI entry point with domain-specific auditing."""
    parser = argparse.ArgumentParser(
        description="ETL Feature Integrity Audit - Domain-specific regression testing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m src.etl.etl_audit --participant P000001 --snapshot 2025-11-07 --domain cardio
  python -m src.etl.etl_audit --participant P000001 --snapshot 2025-11-07 --domain activity
  python -m src.etl.etl_audit --participant P000001 --snapshot 2025-11-07 --domain sleep
  python -m src.etl.etl_audit --participant P000001 --snapshot 2025-11-07 --domain meds
  python -m src.etl.etl_audit --participant P000001 --snapshot 2025-11-07 --domain som
  python -m src.etl.etl_audit --participant P000001 --snapshot 2025-11-07 --domain unified_ext
  python -m src.etl.etl_audit --participant P000001 --snapshot 2025-11-07 --domain labels
        """
    )
    
    parser.add_argument("--participant", "--pid", required=True,
                       help="Participant ID (e.g., P000001)")
    parser.add_argument("--snapshot", required=True,
                       help="Snapshot date (e.g., 2025-11-07)")
    parser.add_argument("--domain", required=True,
                       choices=["cardio", "activity", "sleep", "meds", "som", "unified_ext", "labels"],
                       help="Domain to audit: cardio, activity, sleep, meds, som, unified_ext, or labels")
    
    args = parser.parse_args()
    
    # Create auditor
    auditor = ETLAuditor(args.participant, args.snapshot, args.domain)
    
    # Route to domain-specific audit
    if args.domain == "cardio":
        passed = auditor.run_cardio_audit()
    elif args.domain == "activity":
        passed = auditor.run_activity_audit()
    elif args.domain == "sleep":
        passed = auditor.run_sleep_audit()
    elif args.domain == "meds":
        passed = auditor.run_meds_audit()
    elif args.domain == "som":
        passed = auditor.run_som_audit()
    elif args.domain == "unified_ext":
        passed = auditor.run_unified_extension_audit()
    elif args.domain == "labels":
        passed = auditor.run_labels_audit()
    else:
        logger.error(f"Unknown domain: {args.domain}")
        sys.exit(1)
    
    # Save results
    auditor.save_results()
    
    # Print final summary
    auditor.print_final_summary()
    
    # Exit with appropriate code
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
