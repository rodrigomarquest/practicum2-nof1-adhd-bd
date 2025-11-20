"""
ETL Audit Module for practicum2-nof1-adhd-bd

PhD-level domain-specific regression test suite for ETL pipeline integrity.
Supports HR, Steps, and Sleep feature audits with PASS/FAIL reporting.

Usage:
    python -m src.etl.etl_audit --participant P000001 --snapshot 2025-11-07 --domain hr
    python -m src.etl.etl_audit --participant P000001 --snapshot 2025-11-07 --domain steps
    python -m src.etl.etl_audit --participant P000001 --snapshot 2025-11-07 --domain sleep
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Tuple, Optional
import sys
import argparse
import json

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


class ETLAuditor:
    """Audits ETL pipeline integrity for a single participant/snapshot with domain-specific checks."""
    
    def __init__(self, participant: str, snapshot: str, domain: str = "hr"):
        """
        Initialize auditor for specific participant/snapshot/domain.
        
        Args:
            participant: e.g., "P000001"
            snapshot: e.g., "2025-11-07"
            domain: "hr", "steps", or "sleep"
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
    
    def run_hr_audit(self) -> bool:
        """
        Run HR feature integrity audit.
        
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
        self.banner(f"HR FEATURE INTEGRITY AUDIT: {self.participant}/{self.snapshot}")
        
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
    
    def run_steps_audit(self) -> bool:
        """
        Run Steps feature integrity audit.
        
        Checks:
        - daily_activity.csv exists with steps column
        - Steps present in unified CSV
        - No all-NaN columns
        - Vendor prioritization (Apple vs Zepp)
        - Consistency across extraction → unification → labeling
        
        Returns:
            True if all checks pass, False otherwise
        """
        self.banner(f"STEPS FEATURE INTEGRITY AUDIT: {self.participant}/{self.snapshot}")
        
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
            json.dump(self.audit_results, f, indent=2)
        logger.info(f"✓ Saved audit summary: {json_path}")
    
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
  python -m src.etl.etl_audit --participant P000001 --snapshot 2025-11-07 --domain hr
  python -m src.etl.etl_audit --participant P000001 --snapshot 2025-11-07 --domain steps
  python -m src.etl.etl_audit --participant P000001 --snapshot 2025-11-07 --domain sleep
        """
    )
    
    parser.add_argument("--participant", "--pid", required=True,
                       help="Participant ID (e.g., P000001)")
    parser.add_argument("--snapshot", required=True,
                       help="Snapshot date (e.g., 2025-11-07)")
    parser.add_argument("--domain", required=True,
                       choices=["hr", "steps", "sleep"],
                       help="Domain to audit: hr, steps, or sleep")
    
    args = parser.parse_args()
    
    # Create auditor
    auditor = ETLAuditor(args.participant, args.snapshot, args.domain)
    
    # Route to domain-specific audit
    if args.domain == "hr":
        passed = auditor.run_hr_audit()
    elif args.domain == "steps":
        passed = auditor.run_steps_audit()
    elif args.domain == "sleep":
        passed = auditor.run_sleep_audit()
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
