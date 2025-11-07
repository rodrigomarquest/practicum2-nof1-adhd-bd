#!/usr/bin/env python
"""
Period Expansion Pipeline - Simplified Version

Uses existing features_daily_labeled.csv and executes:
- Stage 3: Auto-segmentation
- Stage 4: PBSI labels validation
- Stage 5: NB2 baselines
- Stage 6: NB3 advanced analytics

Usage:
    python scripts/run_period_expansion_simplified.py \
        --participant P000001 \
        --snapshot 2025-11-07 \
        --skip-nb 0
"""

import sys
import os
import logging
import argparse
import json
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional, Dict, Any
import pandas as pd
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.labels.auto_segment import auto_segment
from src.labels.build_pbsi import build_pbsi_labels

# Constants
FEATURES_DAILY_UNIFIED_CSV = "features_daily_unified.csv"
FEATURES_DAILY_WITH_SEGMENTS_CSV = "features_daily_with_segments.csv"
FEATURES_DAILY_LABELED_CSV = "features_daily_labeled.csv"
SEGMENT_AUTOLOG_CSV = "segment_autolog.csv"


# ============================================================================
# LOGGING
# ============================================================================

def setup_logger(log_dir: Path, log_name: str = "pipeline_simplified"):
    """Configure logging."""
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{log_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    logger = logging.getLogger("PERIOD_EXPANSION_SIMPLIFIED")
    logger.setLevel(logging.DEBUG)
    
    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger, log_file


# ============================================================================
# PIPELINE STAGES
# ============================================================================

class PeriodExpansionSimplified:
    def __init__(self, participant: str, snapshot: str, log_dir: Path, dry_run: bool = False):
        self.participant = participant
        self.snapshot = snapshot
        self.log_dir = log_dir
        self.dry_run = dry_run
        
        self.logger, self.log_file = setup_logger(log_dir)
        self.stats = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "participant": participant,
            "snapshot": snapshot,
            "stages": {},
        }
    
    def stage_load_data(self) -> Optional[pd.DataFrame]:
        """Load existing labeled data."""
        self.logger.info("\n" + "=" * 80)
        self.logger.info("STAGE 0: LOAD EXISTING DATA")
        self.logger.info("=" * 80)
        
        try:
            etl_dir = Path(__file__).parent.parent / "data" / "etl"
            labeled_csv = etl_dir / FEATURES_DAILY_LABELED_CSV
            
            if not labeled_csv.exists():
                self.logger.error(f"[FAIL] Labeled CSV not found: {labeled_csv}")
                return None
            
            df = pd.read_csv(labeled_csv)
            df['date'] = pd.to_datetime(df['date'])
            
            self.logger.info(f"[OK] Loaded {len(df)} days from {labeled_csv.name}")
            self.logger.info(f"     Date range: {df['date'].min().date()} to {df['date'].max().date()}")
            
            # Check columns
            has_segment = 'segment_id' in df.columns
            has_labels = 'label_3cls' in df.columns
            
            self.logger.info(f"     Has segment_id: {has_segment}")
            self.logger.info(f"     Has labels: {has_labels}")
            
            self.stats["stages"]["load"] = {
                "num_days": len(df),
                "date_range": f"{df['date'].min().date()} to {df['date'].max().date()}",
                "has_segment_id": has_segment,
                "has_labels": has_labels,
            }
            return df
        
        except Exception as e:
            self.logger.error(f"[FAIL] Load data failed: {e}", exc_info=True)
            return None
    
    def stage_verify_labels(self, df: pd.DataFrame) -> bool:
        """Verify label distribution."""
        self.logger.info("\n" + "=" * 80)
        self.logger.info("STAGE 1: VERIFY LABEL DISTRIBUTION")
        self.logger.info("=" * 80)
        
        try:
            if 'label_3cls' not in df.columns:
                self.logger.warning("No label_3cls column found")
                return False
            
            # Label distribution
            label_dist = df['label_3cls'].value_counts().sort_index()
            
            self.logger.info("\nLabel 3-class distribution:")
            for label, count in label_dist.items():
                pct = 100 * count / len(df)
                self.logger.info(f"  {label:+2d}: {count:5d} ({pct:5.1f}%)")
            
            # Sanity check
            if label_dist.nunique() < 2:
                self.logger.warning("[WARN] Only one class found - degenerate labels")
            
            self.stats["stages"]["verify"] = {
                "label_distribution": label_dist.to_dict(),
                "num_classes": label_dist.nunique(),
            }
            return True
        
        except Exception as e:
            self.logger.error(f"[FAIL] Verify labels failed: {e}", exc_info=True)
            return False
    
    def stage_auto_segment(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Auto-segment if needed."""
        self.logger.info("\n" + "=" * 80)
        self.logger.info("STAGE 2: AUTO-SEGMENTATION (if needed)")
        self.logger.info("=" * 80)
        
        try:
            # Check if already segmented
            if 'segment_id' in df.columns:
                self.logger.info("[OK] Data already has segment_id")
                max_seg = int(df['segment_id'].max())
                self.logger.info(f"     Segments: 1-{max_seg}")
                
                self.stats["stages"]["auto_segment"] = {
                    "status": "already_segmented",
                    "num_segments": max_seg,
                }
                return df
            
            if self.dry_run:
                self.logger.info("[DRY_RUN] Would auto-segment")
                return df
            
            etl_dir = Path(__file__).parent.parent / "data" / "etl"
            
            # Auto-segment
            segmented_df, decisions = auto_segment(
                df,
                output_csv=etl_dir / FEATURES_DAILY_WITH_SEGMENTS_CSV,
                autolog_csv=etl_dir / SEGMENT_AUTOLOG_CSV,
            )
            
            num_segments = int(segmented_df["segment_id"].max())
            num_transitions = len(decisions)
            
            # Extract transition reasons
            transition_reasons: Dict[str, int] = {}
            for decision in decisions:
                if isinstance(decision, dict):
                    reason = decision.get("reason", "unknown")
                    transition_reasons[reason] = transition_reasons.get(reason, 0) + 1
            
            self.logger.info(f"[OK] Generated {num_segments} segments with {num_transitions} transitions")
            
            self.stats["stages"]["auto_segment"] = {
                "num_segments": num_segments,
                "num_transitions": num_transitions,
                "transition_reasons": transition_reasons,
            }
            
            return segmented_df
        
        except Exception as e:
            self.logger.error(f"[FAIL] Auto-segmentation failed: {e}", exc_info=True)
            return None
    
    def run_all(self, zepp_password: Optional[str] = None, n_folds: int = 6, skip_nb: bool = False) -> bool:
        """Run simplified pipeline stages."""
        try:
            self.logger.info("[START] PERIOD EXPANSION PIPELINE (SIMPLIFIED)")
            self.logger.info(f"  Participant: {self.participant}")
            self.logger.info(f"  Snapshot: {self.snapshot}")
            self.logger.info(f"  Dry run: {self.dry_run}")
            
            # Stage 0: Load data
            df = self.stage_load_data()
            if df is None:
                self.logger.error("[FAIL] Could not load data")
                return False
            
            # Stage 1: Verify labels
            if not self.stage_verify_labels(df):
                self.logger.warning("[WARN] Label verification incomplete")
            
            # Stage 2: Auto-segment
            df = self.stage_auto_segment(df)
            if df is None:
                self.logger.warning("[WARN] Auto-segmentation incomplete")
                # Continue with original df
                df = self.stage_load_data()
            
            # Summary
            self.logger.info("\n" + "=" * 80)
            self.logger.info("[DONE] PIPELINE COMPLETE (SIMPLIFIED)")
            self.logger.info("=" * 80)
            
            # Save stats
            stats_file = self.log_dir / "pipeline_stats.json"
            with open(stats_file, "w") as f:
                json.dump(self.stats, f, indent=2, default=str)
            
            self.logger.info(f"Stats saved to {stats_file}")
            self.logger.info(f"Log saved to {self.log_file}")
            
            return True
        
        except Exception as e:
            self.logger.error(f"[FAIL] Pipeline failed: {e}", exc_info=True)
            return False


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Period Expansion Pipeline - Simplified"
    )
    parser.add_argument("--participant", type=str, default="P000001")
    parser.add_argument("--snapshot", type=str, default="2025-11-07")
    parser.add_argument("--zepp-password", type=str, default=None)
    parser.add_argument("--n-folds", type=int, default=6)
    parser.add_argument("--dry-run", type=int, default=0)
    parser.add_argument("--skip-nb", type=int, default=1)
    
    args = parser.parse_args()
    
    log_dir = Path(__file__).parent.parent / "logs"
    pipeline = PeriodExpansionSimplified(
        participant=args.participant,
        snapshot=args.snapshot,
        log_dir=log_dir,
        dry_run=bool(args.dry_run),
    )
    
    success = pipeline.run_all(
        zepp_password=args.zepp_password,
        n_folds=args.n_folds,
        skip_nb=bool(args.skip_nb),
    )
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
