#!/usr/bin/env python
"""
Extract biomarkers from raw data and generate daily features matrix.

Usage:
    python -m src.biomarkers.extract --participant P000001 --snapshot 2025-11-07 \
        --data-dir data/etl/P000001/2025-11-07/extracted

This script:
1. Loads raw Zepp and Apple data from extracted/ folder
2. Computes all Tier 1+2+X biomarkers
3. Outputs joined_features_daily_biomarkers.csv
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def find_data_files(data_dir: Path, participant: str = "P000001") -> dict:
    """Discover raw data files in directory structure.
    
    Looks in:
    1. {data_dir} (extracted/ subfolder)
    2. data/raw/{participant}/zepp (default Zepp path)
    """
    files = {}
    _find_zepp_files(data_dir, files)
    _find_apple_files(data_dir, files)
    
    # Fallback: check default Zepp raw path
    zepp_raw_path = Path("data/raw") / participant / "zepp"
    if not files.get("zepp_hr_auto") and zepp_raw_path.exists():
        logger.info(f"Checking default Zepp path: {zepp_raw_path}")
        _find_zepp_files(zepp_raw_path, files)
    
    return files


def _find_zepp_files(data_dir: Path, files: dict) -> None:
    """Find Zepp extracted files."""
    zepp_cloud = data_dir / "zepp" / "cloud"
    if not zepp_cloud.exists():
        return

    # HEARTRATE_AUTO
    hr_auto_files = list(zepp_cloud.glob("HEARTRATE_AUTO/*.csv"))
    if hr_auto_files:
        files["zepp_hr_auto"] = str(hr_auto_files[0])
        logger.info(f"Found Zepp HR_AUTO: {hr_auto_files[0].name}")

    # SLEEP
    sleep_files = list(zepp_cloud.glob("SLEEP/*.csv"))
    if sleep_files:
        files["zepp_sleep"] = str(sleep_files[0])
        logger.info(f"Found Zepp SLEEP: {sleep_files[0].name}")

    # ACTIVITY_STAGE
    activity_stage_files = list(zepp_cloud.glob("ACTIVITY_STAGE/*.csv"))
    if activity_stage_files:
        files["zepp_activity_stage"] = str(activity_stage_files[0])
        logger.info(f"Found Zepp ACTIVITY_STAGE: {activity_stage_files[0].name}")

    # ACTIVITY_MINUTE
    activity_minute_files = list(zepp_cloud.glob("ACTIVITY_MINUTE/*.csv"))
    if activity_minute_files:
        files["zepp_activity_minute"] = str(activity_minute_files[0])
        logger.info(f"Found Zepp ACTIVITY_MINUTE: {activity_minute_files[0].name}")


def _find_apple_files(data_dir: Path, files: dict) -> None:
    """Find Apple extracted files."""
    apple_inapp = data_dir / "apple" / "inapp"
    if not apple_inapp.exists():
        return

    apple_hr_file = apple_inapp / "HKQuantityTypeIdentifierHeartRate.csv"
    if apple_hr_file.exists():
        files["apple_hr"] = str(apple_hr_file)
        logger.info("Found Apple HR")

    apple_hrv_file = apple_inapp / "HKQuantityTypeIdentifierHeartRateVariabilitySDNN.csv"
    if apple_hrv_file.exists():
        files["apple_hrv"] = str(apple_hrv_file)
        logger.info("Found Apple HRV SDNN")

    apple_activity_file = apple_inapp / "HKQuantityTypeIdentifierActiveEnergyBurned.csv"
    if apple_activity_file.exists():
        files["apple_activity"] = str(apple_activity_file)
        logger.info("Found Apple Activity")


def main():
    parser = argparse.ArgumentParser(description="Extract clinical biomarkers from wearable data")
    parser.add_argument("--participant", required=True, help="Participant ID (e.g. P000001)")
    parser.add_argument("--snapshot", required=True, help="Snapshot date (e.g. 2025-11-07)")
    parser.add_argument("--data-dir", required=True, help="Path to extracted/ data directory")
    parser.add_argument("--output-dir", default="data/etl", help="Output directory for features")
    parser.add_argument("--cutoff-months", type=int, default=30, help="Months of recent data to use")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")

    args = parser.parse_args()

    # Setup logging
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Validate paths
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        logger.error(f"Data directory not found: {data_dir}")
        return 1

    # Find data files
    logger.info(f"Discovering data files in {data_dir}...")
    files = find_data_files(data_dir, args.participant)

    if not files:
        logger.error("No data files found!")
        return 1

    # Compute cutoff date
    cutoff_date = pd.Timestamp.now() - pd.DateOffset(months=args.cutoff_months)
    logger.info(f"Using cutoff date: {cutoff_date.date()}")

    # Extract biomarkers (import here to avoid circular dependencies)
    try:
        from . import aggregate

        df_biomarkers = aggregate.aggregate_daily_biomarkers(
            zepp_hr_auto_path=files.get("zepp_hr_auto"),
            zepp_sleep_path=files.get("zepp_sleep"),
            zepp_activity_stage_path=files.get("zepp_activity_stage"),
            zepp_activity_minute_path=files.get("zepp_activity_minute"),
            apple_hr_path=files.get("apple_hr"),
            apple_hrv_path=files.get("apple_hrv"),
            apple_activity_path=files.get("apple_activity"),
            cutoff_date=cutoff_date,
            participant=args.participant,
            snapshot=args.snapshot,
        )

        if df_biomarkers.empty:
            logger.error("No biomarkers extracted!")
            return 1

        # Save output
        output_dir = Path(args.output_dir) / args.participant / args.snapshot / "joined"
        output_dir.mkdir(parents=True, exist_ok=True)

        output_file = output_dir / "joined_features_daily_biomarkers.csv"
        df_biomarkers.to_csv(output_file, index=False)
        logger.info(f"Biomarkers saved: {output_file}")
        logger.info(f"Dimensions: {df_biomarkers.shape}")
        logger.info(f"Columns: {list(df_biomarkers.columns)}")

        # Print summary
        print("\n=== Biomarkers Extraction Summary ===")
        print(f"Participant: {args.participant}")
        print(f"Snapshot: {args.snapshot}")
        print(f"Daily records: {len(df_biomarkers)}")
        print(f"Date range: {df_biomarkers['date'].min().date()} to {df_biomarkers['date'].max().date()}")
        print(f"Total features: {len(df_biomarkers.columns)}")
        print("Feature groups:")
        print(f"  - HRV: {sum(1 for c in df_biomarkers.columns if 'hrv' in c.lower() or 'hr_' in c)}")
        print(f"  - Sleep: {sum(1 for c in df_biomarkers.columns if 'sleep' in c.lower())}")
        print(f"  - Activity: {sum(1 for c in df_biomarkers.columns if 'activity' in c.lower())}")
        print(f"  - Circadian: {sum(1 for c in df_biomarkers.columns if 'nocturnal' in c.lower() or 'peak_hour' in c)}")

        # Data quality summary
        if "data_quality_score" in df_biomarkers.columns:
            mean_quality = df_biomarkers["data_quality_score"].mean()
            print(f"\nData Quality: {mean_quality:.1f}% (mean)")

        return 0

    except Exception as e:
        logger.exception(f"Error during biomarkers extraction: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
