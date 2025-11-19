"""
ETL Audit Module for practicum2-nof1-adhd-bd

PhD-level surgical audit of ETL pipeline (Stages 0-2) for a single participant/snapshot.
Focus: Data extraction, daily aggregation, and unified join integrity.

Usage:
    python -m src.etl.etl_audit P000001 2025-11-07
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Tuple, Optional
import sys

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


class ETLAuditor:
    """Audits ETL pipeline integrity for a single participant/snapshot."""
    
    def __init__(self, participant: str, snapshot: str):
        """
        Initialize auditor for specific participant/snapshot.
        
        Args:
            participant: e.g., "P000001"
            snapshot: e.g., "2025-11-07"
        """
        self.participant = participant
        self.snapshot = snapshot
        
        # Paths
        self.raw_dir = Path("data/raw") / participant
        self.etl_dir = Path("data/etl") / participant / snapshot
        self.extracted_dir = self.etl_dir / "extracted"
        self.joined_dir = self.etl_dir / "joined"
        
        # Results storage
        self.audit_results = {
            "participant": participant,
            "snapshot": snapshot,
            "timestamp": datetime.now().isoformat(),
            "raw_files": {},
            "extracted_stats": {},
            "unified_stats": {},
            "issues": [],
            "warnings": []
        }
    
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
    """CLI entry point."""
    if len(sys.argv) != 3:
        print("Usage: python -m src.etl.etl_audit <PARTICIPANT> <SNAPSHOT>")
        print("Example: python -m src.etl.etl_audit P000001 2025-11-07")
        sys.exit(1)
    
    participant = sys.argv[1]
    snapshot = sys.argv[2]
    
    auditor = ETLAuditor(participant, snapshot)
    results = auditor.run_full_audit()
    
    # Save results to JSON
    import json
    output_path = Path("ETL_AUDIT_RESULTS.json")
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"\n📋 Audit results saved to: {output_path}")


if __name__ == "__main__":
    main()
