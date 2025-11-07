#!/usr/bin/env python3
"""
NB2 Pipeline: Main orchestrator.

Usage:
    python scripts/run_nb2_pipeline.py --stage all --participant P000001 --snapshot 2025-09-29
    python scripts/run_nb2_pipeline.py --stage unify --participant P000001
    python scripts/run_nb2_pipeline.py --stage labels --participant P000001
    python scripts/run_nb2_pipeline.py --stage baselines --participant P000001
"""

import argparse
import logging
from pathlib import Path
from datetime import datetime
import sys

import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from features.unify_daily import unify_daily
from labels.build_pbsi import build_pbsi_labels
from models.run_nb2 import run_temporal_cv

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="NB2 Pipeline: Unify data, build labels, run CV"
    )
    parser.add_argument(
        "--stage",
        choices=["all", "unify", "labels", "baselines"],
        default="all",
        help="Pipeline stage to run",
    )
    parser.add_argument(
        "--participant",
        type=str,
        default="P000001",
        help="Participant ID",
    )
    parser.add_argument(
        "--snapshot",
        type=str,
        default=None,
        help="Snapshot date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--apple-dir",
        type=Path,
        default=Path("data/raw/apple"),
        help="Apple data directory",
    )
    parser.add_argument(
        "--zepp-dir",
        type=Path,
        default=Path("data/raw/zepp_processed"),
        help="Zepp data directory",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/etl"),
        help="Output directory for unified/labeled data",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("nb2"),
        help="Results directory for CV outputs",
    )
    
    args = parser.parse_args()
    
    logger.info("="*80)
    logger.info("NB2 PIPELINE")
    logger.info("="*80)
    logger.info(f"Stage: {args.stage}")
    logger.info(f"Participant: {args.participant}")
    logger.info(f"Snapshot: {args.snapshot}")
    
    # Paths
    unified_path = args.output_dir / "features_daily_unified.csv"
    labeled_path = args.output_dir / "features_daily_labeled.csv"
    version_log_path = Path("data") / "version_log_enriched.csv"
    
    # ========== STAGE 1: UNIFY ==========
    if args.stage in ["all", "unify"]:
        logger.info("\n" + "="*80)
        logger.info("STAGE 1: UNIFY APPLE + ZEPP")
        logger.info("="*80)
        
        unify_daily(
            apple_dir=args.apple_dir,
            zepp_dir=args.zepp_dir,
            output_path=unified_path,
        )
    
    # ========== STAGE 2: BUILD LABELS ==========
    if args.stage in ["all", "labels"]:
        logger.info("\n" + "="*80)
        logger.info("STAGE 2: BUILD PBSI LABELS")
        logger.info("="*80)
        
        if not unified_path.exists():
            logger.error(f"Unified data not found: {unified_path}")
            logger.error("Run unify stage first: --stage unify")
            sys.exit(1)
        
        df_unified = pd.read_csv(unified_path, parse_dates=['date'])
        
        build_pbsi_labels(
            df_unified,
            version_log_path=version_log_path,
            output_path=labeled_path,
        )
    
    # ========== STAGE 3: BASELINES ==========
    if args.stage in ["all", "baselines"]:
        logger.info("\n" + "="*80)
        logger.info("STAGE 3: RUN TEMPORAL CV + BASELINES")
        logger.info("="*80)
        
        if not labeled_path.exists():
            logger.error(f"Labeled data not found: {labeled_path}")
            logger.error("Run labels stage first: --stage labels")
            sys.exit(1)
        
        df_labeled = pd.read_csv(labeled_path, parse_dates=['date'])
        
        # Feature columns (exclude administrative columns)
        exclude_cols = {
            'date', 'missing_sleep', 'missing_cardio', 'missing_activity',
            'source_sleep', 'source_cardio', 'source_activity',
            'label_3cls', 'label_2cls', 'label_clinical',
            'sleep_sub', 'cardio_sub', 'activity_sub', 'pbsi_score',
            'pbsi_quality'
        }
        feature_cols = [c for c in df_labeled.columns if c not in exclude_cols]
        
        logger.info(f"Features: {feature_cols}")
        
        # Create results directory
        args.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Run 3-class CV
        logger.info("\n--- 3-CLASS TASK ---")
        run_temporal_cv(
            df_labeled,
            feature_cols=feature_cols,
            label_col='label_3cls',
            n_folds=6,
            output_dir=args.results_dir,
        )
        
        # Run 2-class CV
        logger.info("\n--- 2-CLASS TASK ---")
        run_temporal_cv(
            df_labeled,
            feature_cols=feature_cols,
            label_col='label_2cls',
            n_folds=6,
            output_dir=args.results_dir,
        )
    
    logger.info("\n" + "="*80)
    logger.info("PIPELINE COMPLETE")
    logger.info("="*80)


if __name__ == "__main__":
    main()
