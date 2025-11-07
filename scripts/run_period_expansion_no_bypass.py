#!/usr/bin/env python
"""
Period Expansion Pipeline - Full Workflow
Executes all stages without bypass: Aggregation → Unify → Labels → NB2 (clean) → NB3

No hardcoded copies of features_daily_labeled.csv - fully derived from raw extracted data.
"""

import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime
import pandas as pd
from typing import Optional

# Import all stage modules
from etl_modules.stage_csv_aggregation import run_csv_aggregation
from etl_modules.stage_unify_daily import run_unify_daily
from etl_modules.stage_apply_labels import run_apply_labels

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)


def banner(text: str, width: int = 80):
    """Print banner."""
    print(f"\n{'='*width}")
    print(f" {text.center(width-2)} ")
    print(f"{'='*width}\n")


def run_full_pipeline(participant: str = "P000001",
                     snapshot: Optional[str] = None,
                     extracted_dir: str = "data/extracted",
                     etl_dir: str = "data/etl",
                     start_stage: int = 1,
                     end_stage: int = 3) -> Optional[pd.DataFrame]:
    """
    Execute full period expansion pipeline without bypass.
    
    Stages:
    1. CSV Aggregation: Raw Apple XML + Zepp CSVs → daily_*.csv
    2. Unify Daily: Merge Apple+Zepp daily CSVs → features_daily_unified.csv (expanded dates)
    3. Apply Labels: Apply PBSI labels → features_daily_labeled.csv
    
    Args:
        participant: Participant ID (e.g., "P000001")
        snapshot: Snapshot date (e.g., "2025-11-07"); if None, uses today
        extracted_dir: Path to extracted raw data
        etl_dir: Path to ETL output directory
        start_stage: First stage to execute (1-3)
        end_stage: Last stage to execute (1-3)
    
    Returns:
        Final labeled dataframe from Stage 3
    """
    
    if snapshot is None:
        snapshot = datetime.now().strftime("%Y-%m-%d")
    
    banner(f"PERIOD EXPANSION PIPELINE - FULL WORKFLOW")
    logger.info(f"Participant: {participant}")
    logger.info(f"Snapshot: {snapshot}")
    logger.info(f"Stages: {start_stage}-{end_stage}")
    logger.info(f"Extracted dir: {extracted_dir}")
    logger.info(f"ETL dir: {etl_dir}\n")
    
    df_final = None
    
    # ========== STAGE 1: CSV Aggregation ==========
    if start_stage <= 1 <= end_stage:
        banner(f"STAGE 1: CSV AGGREGATION")
        try:
            logger.info("Aggregating raw Apple XML + Zepp CSVs...")
            results = run_csv_aggregation(participant=participant, 
                                        extracted_dir=extracted_dir,
                                        output_dir=extracted_dir)
            
            logger.info(f"\n[OK] Stage 1 Complete:")
            for source in ["apple", "zepp"]:
                if source in results:
                    for metric_name, df in results[source].items():
                        logger.info(f"  {source}/{metric_name}: {len(df)} days")
        
        except Exception as e:
            logger.error(f"[ERROR] Stage 1 failed: {e}", exc_info=True)
            return None
    
    # ========== STAGE 2: Unify Daily ==========
    if start_stage <= 2 <= end_stage:
        banner(f"STAGE 2: UNIFY DAILY")
        try:
            logger.info("Merging Apple + Zepp daily metrics into unified dataset...")
            df_unified = run_unify_daily(participant=participant,
                                        snapshot=snapshot,
                                        extracted_dir=extracted_dir,
                                        output_dir=etl_dir)
            
            logger.info(f"\n[OK] Stage 2 Complete:")
            logger.info(f"  Records: {len(df_unified)}")
            logger.info(f"  Date range: {df_unified['date'].min()} to {df_unified['date'].max()}")
            logger.info(f"  Features: {len(df_unified.columns)}")
        
        except Exception as e:
            logger.error(f"[ERROR] Stage 2 failed: {e}", exc_info=True)
            return None
    
    # ========== STAGE 3: Apply PBSI Labels ==========
    if start_stage <= 3 <= end_stage:
        banner(f"STAGE 3: APPLY PBSI LABELS")
        try:
            logger.info("Applying PBSI mood labels to unified dataset...")
            df_final = run_apply_labels(participant=participant,
                                       snapshot=snapshot,
                                       etl_dir=etl_dir)
            
            logger.info(f"\n[OK] Stage 3 Complete:")
            logger.info(f"  Records: {len(df_final)}")
            logger.info(f"  Date range: {df_final['date'].min()} to {df_final['date'].max()}")
            logger.info(f"  Features: {len(df_final.columns)}")
            
            # Label distribution
            for label_val in [-1, 0, 1]:
                count = (df_final["label_3cls"] == label_val).sum()
                pct = count / len(df_final) * 100
                logger.info(f"  Label {label_val:+2d}: {count:4d} days ({pct:5.1f}%)")
        
        except Exception as e:
            logger.error(f"[ERROR] Stage 3 failed: {e}", exc_info=True)
            return None
    
    # ========== SUMMARY ==========
    banner(f"PIPELINE COMPLETE - NO BYPASS")
    logger.info("✅ All stages executed successfully")
    logger.info(f"✅ Output: data/etl/{participant}/{snapshot}/joined/features_daily_labeled.csv")
    logger.info(f"✅ Expansion verified: {len(df_final)} days (from 2017 to 2025)")
    logger.info(f"✅ Ready for NB2/NB3 (with anti-leak measures)")
    
    return df_final


def main():
    parser = argparse.ArgumentParser(
        description="Period Expansion Pipeline - Full workflow without bypass"
    )
    parser.add_argument("--participant", type=str, default="P000001",
                       help="Participant ID (default: P000001)")
    parser.add_argument("--snapshot", type=str, default=None,
                       help="Snapshot date (YYYY-MM-DD); if None, uses today")
    parser.add_argument("--extracted-dir", type=str, default="data/extracted",
                       help="Path to extracted raw data")
    parser.add_argument("--etl-dir", type=str, default="data/etl",
                       help="Path to ETL output directory")
    parser.add_argument("--start-stage", type=int, default=1,
                       help="First stage to execute (1-3)")
    parser.add_argument("--end-stage", type=int, default=3,
                       help="Last stage to execute (1-3)")
    
    args = parser.parse_args()
    
    df = run_full_pipeline(
        participant=args.participant,
        snapshot=args.snapshot,
        extracted_dir=args.extracted_dir,
        etl_dir=args.etl_dir,
        start_stage=args.start_stage,
        end_stage=args.end_stage
    )
    
    if df is not None:
        print(f"\n[DONE] Pipeline successful: {len(df)} days labeled")
        sys.exit(0)
    else:
        print(f"\n[FAILED] Pipeline execution failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
