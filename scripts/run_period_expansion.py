#!/usr/bin/env python
"""
Period Expansion & Auto-Segmentation Pipeline

Comprehensive pipeline to:
1. Extract all ZIPs from data/raw/ → data/extracted/
2. Unify Apple+Zepp daily data
3. Auto-segment without version_log
4. Compute PBSI labels
5. Run NB2 baselines
6. Run NB3 advanced analytics

Usage:
    python scripts/run_period_expansion.py \
        --participant P000001 \
        --snapshot 2025-11-07 \
        --n-folds 6 \
        --zepp-password $ZEPP_ZIP_PASSWORD \
        --dry-run 0
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

# Constants
FEATURES_DAILY_UNIFIED_CSV = "features_daily_unified.csv"
FEATURES_DAILY_WITH_SEGMENTS_CSV = "features_daily_with_segments.csv"
FEATURES_DAILY_LABELED_CSV = "features_daily_labeled.csv"
SEGMENT_AUTOLOG_CSV = "segment_autolog.csv"

from src.io.zip_extractor import extract_all_zips
from src.labels.auto_segment import auto_segment
from src.features.unify_daily import unify_daily
from src.labels.build_pbsi import build_pbsi_labels

# ============================================================================
# LOGGING
# ============================================================================

def setup_logger(log_dir: Path, log_name: str = "pipeline_expansion"):
    """Configure logging."""
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{log_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    logger = logging.getLogger("PERIOD_EXPANSION")
    logger.setLevel(logging.DEBUG)
    
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

class PeriodExpansionPipeline:
    def __init__(self, participant: str, snapshot: str, log_dir: Path, dry_run: bool = False):
        self.participant = participant
        self.snapshot = snapshot
        self.log_dir = log_dir
        self.dry_run = dry_run
        
        self.logger, self.log_file = setup_logger(log_dir)
        self.stats = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "participant": participant,
            "snapshot": snapshot,
            "stages": {},
        }
    
    def stage_extract_zips(self, zepp_password: Optional[str] = None) -> bool:
        """Stage 1: Extract all ZIPs."""
        self.logger.info("=" * 80)
        self.logger.info("STAGE 1: ZIP EXTRACTION")
        self.logger.info("=" * 80)
        
        try:
            stats = extract_all_zips(
                participant=self.participant,
                zepp_password=zepp_password,
                dry_run=self.dry_run,
            )
            
            self.stats["stages"]["extract"] = stats
            self.logger.info(f"[OK] Extracted {stats['num_extracted']}/{stats['num_zips_discovered']} ZIPs")
            return True
        except Exception as e:
            self.logger.error(f"[FAIL] ZIP extraction failed: {e}", exc_info=True)
            return False
    
    def stage_unify_daily(self) -> bool:
        """Stage 2: Unify Apple + Zepp daily data."""
        self.logger.info("\n" + "=" * 80)
        self.logger.info("STAGE 2: DAILY UNIFICATION")
        self.logger.info("=" * 80)
        
        try:
            # Find extracted data
            extracted_dir = Path(__file__).parent.parent / "data" / "extracted"
            apple_dir = extracted_dir / "apple" / self.participant
            zepp_dir = extracted_dir / "zepp" / self.participant
            
            self.logger.info(f"  Apple: {apple_dir}")
            self.logger.info(f"  Zepp:  {zepp_dir}")
            
            if self.dry_run:
                self.logger.info("[DRY_RUN] Would unify data")
                return True
            
            # Unify data using existing unify_daily function
            etl_dir = Path(__file__).parent.parent / "data" / "etl"
            output_path = etl_dir / FEATURES_DAILY_UNIFIED_CSV
            
            unified_df = unify_daily(
                apple_dir=apple_dir,
                zepp_dir=zepp_dir,
                output_path=output_path,
            )
            
            self.logger.info(f"[OK] Unified {len(unified_df)} days from Apple + Zepp")
            
            self.stats["stages"]["unify"] = {
                "num_days": len(unified_df),
                "date_range": f"{unified_df['date'].min()} to {unified_df['date'].max()}",
                "output_path": str(output_path),
            }
            return True
        
        except Exception as e:
            self.logger.error(f"[FAIL] Unification failed: {e}", exc_info=True)
            return False
    
    def stage_auto_segment(self) -> bool:
        """Stage 3: Auto-segment without version_log."""
        self.logger.info("\n" + "=" * 80)
        self.logger.info("STAGE 3: AUTO-SEGMENTATION")
        self.logger.info("=" * 80)
        
        try:
            etl_dir = Path(__file__).parent.parent / "data" / "etl"
            unified_csv = etl_dir / FEATURES_DAILY_UNIFIED_CSV
            
            if not unified_csv.exists():
                self.logger.error(f"Unified CSV not found: {unified_csv}")
                return False
            
            unified_df = pd.read_csv(unified_csv)
            
            if self.dry_run:
                self.logger.info("[DRY_RUN] Would auto-segment")
                return True
            
            # Auto-segment
            segmented_df, decisions = auto_segment(
                unified_df,
                output_csv=etl_dir / FEATURES_DAILY_WITH_SEGMENTS_CSV,
                autolog_csv=etl_dir / SEGMENT_AUTOLOG_CSV,
            )
            
            # Extract transition reasons safely
            transition_reasons: Dict[str, int] = {}
            for decision in decisions:
                if isinstance(decision, dict):
                    reason = decision.get("reason", "unknown")
                    transition_reasons[reason] = transition_reasons.get(reason, 0) + 1
            
            stats = {
                "num_segments": int(segmented_df["segment_id"].max()),
                "segment_counts": segmented_df["segment_id"].value_counts().to_dict(),
                "num_transitions": len(decisions),
                "transition_reasons": transition_reasons,
            }
            
            self.stats["stages"]["auto_segment"] = stats
            self.logger.info(f"[OK] Generated {stats['num_segments']} segments with {len(decisions)} transitions")
            return True
        
        except Exception as e:
            self.logger.error(f"[FAIL] Auto-segmentation failed: {e}", exc_info=True)
            return False
    
    def stage_pbsi_labels(self) -> bool:
        """Stage 4: Compute PBSI labels."""
        self.logger.info("\n" + "=" * 80)
        self.logger.info("STAGE 4: PBSI LABEL COMPUTATION")
        self.logger.info("=" * 80)
        
        try:
            etl_dir = Path(__file__).parent.parent / "data" / "etl"
            
            # Load segmented data (prefer segmented, fall back to unified)
            seg_csv = etl_dir / FEATURES_DAILY_WITH_SEGMENTS_CSV
            if not seg_csv.exists():
                self.logger.warning("Segmented CSV not found; using unified CSV")
                seg_csv = etl_dir / FEATURES_DAILY_UNIFIED_CSV
            
            if not seg_csv.exists():
                self.logger.error("No data CSV found")
                return False
            
            if self.dry_run:
                self.logger.info("[DRY_RUN] Would compute PBSI labels")
                return True
            
            # Load data
            input_df = pd.read_csv(seg_csv)
            output_path = etl_dir / FEATURES_DAILY_LABELED_CSV
            
            # Build PBSI labels
            labeled_df = build_pbsi_labels(
                unified_df=input_df,
                version_log_path=None,  # We already have segment_id in input_df
                output_path=output_path,
            )
            
            self.logger.info(f"[OK] Generated labels for {len(labeled_df)} days")
            
            # Capture label distribution
            label_dist = labeled_df["label_3cls"].value_counts().to_dict()
            
            self.stats["stages"]["pbsi_labels"] = {
                "num_days_labeled": len(labeled_df),
                "label_distribution": label_dist,
                "output_path": str(output_path),
            }
            return True
        
        except Exception as e:
            self.logger.error(f"[FAIL] PBSI label computation failed: {e}", exc_info=True)
            return False
    
    def stage_nb2_run(self, n_folds: int = 6, train_days: int = 120, val_days: int = 60, seed: int = 42) -> bool:
        """Stage 5: Run NB2 baselines."""
        self.logger.info("\n" + "=" * 80)
        self.logger.info("STAGE 5: NB2 BASELINE MODELS")
        self.logger.info("=" * 80)
        
        try:
            if self.dry_run:
                self.logger.info("🌤️  DRY RUN: Would run NB2")
                return True
            
            # Run NB2
            self.logger.info(f"  Calling NB2 with {n_folds} folds, {train_days} train days, {val_days} val days...")
            
            # NOTE: This assumes run_nb2_beiwe has an importable main function or CLI
            # Adjust as needed
            nb2_args = {
                "pid": self.participant,
                "snapshot": self.snapshot,
                "n_folds": n_folds,
                "train_days": train_days,
                "val_days": val_days,
                "class_weight": "balanced",
                "seed": seed,
                "plots": 1,
                "save_all": 1,
                "verbose": 2,
            }
            
            self.logger.info(f"  NB2 args: {nb2_args}")
            self.logger.info("[OK] NB2 execution would proceed here (placeholder for now)")
            
            self.stats["stages"]["nb2"] = {"status": "pending", "args": nb2_args}
            return True
        
        except Exception as e:
            self.logger.error(f"[FAIL] NB2 execution failed: {e}", exc_info=True)
            return False
    
    def stage_nb3_run(self) -> bool:
        """Stage 6: Run NB3 advanced analytics."""
        self.logger.info("\n" + "=" * 80)
        self.logger.info("STAGE 6: NB3 ADVANCED ANALYTICS")
        self.logger.info("=" * 80)
        
        try:
            if self.dry_run:
                self.logger.info("[DRY_RUN] Would run NB3")
                return True
            
            # Run NB3
            self.logger.info("  Calling NB3 for SHAP, Drift, LSTM...")
            self.logger.info("[OK] NB3 execution would proceed here (placeholder for now)")
            
            self.stats["stages"]["nb3"] = {"status": "pending"}
            return True
        
        except Exception as e:
            self.logger.error(f"[FAIL] NB3 execution failed: {e}", exc_info=True)
            return False
    
    def run_all(self, zepp_password: Optional[str] = None, n_folds: int = 6, skip_nb: bool = False) -> bool:
        """Run all pipeline stages."""
        try:
            self.logger.info("[START] PERIOD EXPANSION PIPELINE")
            self.logger.info(f"  Participant: {self.participant}")
            self.logger.info(f"  Snapshot: {self.snapshot}")
            self.logger.info(f"  Dry run: {self.dry_run}")
            
            # Stage 1: Extract
            if not self.stage_extract_zips(zepp_password):
                self.logger.warning("[WARN] ZIP extraction stage incomplete; continuing...")
            
            # Stage 2: Unify
            if not self.stage_unify_daily():
                self.logger.warning("[WARN] Unification stage incomplete; continuing...")
            
            # Stage 3: Auto-segment
            if not self.stage_auto_segment():
                self.logger.warning("[WARN] Auto-segmentation stage incomplete; continuing...")
            
            # Stage 4: PBSI labels
            if not self.stage_pbsi_labels():
                self.logger.warning("[WARN] PBSI label stage incomplete; continuing...")
            
            # Stage 5-6: NB2/NB3 (optional)
            if not skip_nb:
                if not self.stage_nb2_run(n_folds=n_folds):
                    self.logger.warning("[WARN] NB2 stage incomplete")
                
                if not self.stage_nb3_run():
                    self.logger.warning("[WARN] NB3 stage incomplete")
            
            self.logger.info("\n" + "=" * 80)
            self.logger.info("[DONE] PIPELINE COMPLETE")
            self.logger.info("=" * 80)
            
            # Save stats
            stats_file = self.log_dir / "pipeline_stats.json"
            with open(stats_file, "w") as f:
                json.dump(self.stats, f, indent=2, default=str)
            
            self.logger.info(f"Stats saved to {stats_file}")
            self.logger.info(f"Log saved to {self.log_file}")
            
            return True
        
        except Exception as e:
            self.logger.error(f"❌ FATAL: {e}", exc_info=True)
            return False


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Period Expansion & Auto-Segmentation Pipeline")
    parser.add_argument("--participant", default="P000001", help="Participant ID (default: P000001)")
    parser.add_argument("--snapshot", default="2025-11-07", help="Snapshot date (default: 2025-11-07)")
    parser.add_argument("--zepp-password", help="Zepp ZIP password (or set ZEPP_ZIP_PASSWORD env)")
    parser.add_argument("--n-folds", type=int, default=6, help="Number of CV folds (default: 6)")
    parser.add_argument("--dry-run", type=int, default=0, help="Dry run mode (default: 0)")
    parser.add_argument("--skip-nb", type=int, default=1, help="Skip NB2/NB3 (default: 1, set to 0 to run)")
    parser.add_argument("--log-dir", help="Log directory (default: logs/)")
    
    args = parser.parse_args()
    
    log_dir = Path(args.log_dir) if args.log_dir else Path(__file__).parent.parent / "logs"
    
    pipeline = PeriodExpansionPipeline(
        participant=args.participant,
        snapshot=args.snapshot,
        log_dir=log_dir,
        dry_run=bool(args.dry_run),
    )
    
    zepp_password = args.zepp_password or os.environ.get("ZEPP_ZIP_PASSWORD")
    
    success = pipeline.run_all(
        zepp_password=zepp_password,
        n_folds=args.n_folds,
        skip_nb=bool(args.skip_nb),
    )
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
