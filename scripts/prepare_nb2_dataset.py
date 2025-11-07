"""
Data Preparation for NB2 - Anti-Leak Safeguards
Remove label-related features and prepare clean dataset for training.

Removes:
- pbsi_score, pbsi_quality
- All label_* columns
- Optional: segment_id (for fair temporal split)

Only keeps: date, original health metrics (sleep, HR, activity), label_3cls
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Tuple

logger = logging.getLogger(__name__)


def prepare_nb2_dataset(csv_path: str, 
                       remove_segment_id: bool = False,
                       output_path: str = None) -> Tuple[pd.DataFrame, dict]:
    """
    Prepare clean dataset for NB2 training with anti-leak safeguards.
    
    Removes:
    - pbsi_score, pbsi_quality (label metadata)
    - label_3cls (will be added as target)
    - All other label_* columns
    - Optional: segment_id (if remove_segment_id=True, for fair evaluation)
    
    Keeps:
    - date (for temporal split)
    - Health metrics: sleep_hours, sleep_quality_score, hr_mean, hr_min, hr_max, hr_std, 
                     hr_samples, total_steps, total_distance, total_active_energy
    - label_3cls (as target 'y')
    
    Args:
        csv_path: Path to features_daily_labeled.csv
        remove_segment_id: If True, remove segment_id for fair temporal split
        output_path: Optional path to save cleaned dataset
    
    Returns:
        (df_clean, stats_dict)
    """
    
    logger.info(f"\n{'='*80}")
    logger.info(f"DATA PREPARATION: Anti-Leak Safeguards for NB2")
    logger.info(f"{'='*80}\n")
    
    # Load data
    logger.info(f"[Prep] Loading: {csv_path}")
    df = pd.read_csv(csv_path)
    logger.info(f"[Prep] Original shape: {df.shape}")
    logger.info(f"[Prep] Original columns: {list(df.columns)}")
    
    stats = {
        "original_rows": len(df),
        "original_cols": len(df.columns),
        "removed_cols": []
    }
    
    # Identify columns to keep
    keep_cols = ["date"]
    
    # Health metrics (features for X)
    health_metrics = [
        "sleep_hours", "sleep_quality_score",
        "hr_mean", "hr_min", "hr_max", "hr_std", "hr_samples",
        "total_steps", "total_distance", "total_active_energy"
    ]
    
    for col in health_metrics:
        if col in df.columns:
            keep_cols.append(col)
    
    # Target variable
    if "label_3cls" in df.columns:
        keep_cols.append("label_3cls")
    else:
        logger.error("[Prep] ERROR: label_3cls not found!")
        raise ValueError("label_3cls column not found in dataset")
    
    # Optional: keep segment_id if not removing
    if not remove_segment_id and "segment_id" in df.columns:
        keep_cols.append("segment_id")
    
    # Identify columns to remove (for logging)
    for col in df.columns:
        if col not in keep_cols:
            stats["removed_cols"].append(col)
    
    # Create cleaned dataframe
    df_clean = df[keep_cols].copy()
    
    logger.info(f"\n[Prep] === COLUMN CLEANUP ===")
    logger.info(f"[Prep] Removed {len(stats['removed_cols'])} columns:")
    for col in sorted(stats["removed_cols"]):
        logger.info(f"  - {col}")
    
    logger.info(f"\n[Prep] Kept {len(keep_cols)} columns (features + target):")
    for col in keep_cols:
        if col != "date":
            logger.info(f"  + {col}")
    
    # Check for data quality
    missing_data = df_clean.isnull().sum().sum()
    logger.info(f"\n[Prep] === DATA QUALITY ===")
    logger.info(f"[Prep] Missing values: {missing_data}")
    
    # Label distribution
    logger.info(f"[Prep] === LABEL DISTRIBUTION ===")
    label_dist = df_clean["label_3cls"].value_counts().sort_index()
    for label_val, count in label_dist.items():
        pct = count / len(df_clean) * 100
        logger.info(f"[Prep] Label {label_val:+2d}: {count:4d} days ({pct:5.1f}%)")
    
    # Date range
    logger.info(f"\n[Prep] === TEMPORAL RANGE ===")
    logger.info(f"[Prep] Date range: {df_clean['date'].min()} to {df_clean['date'].max()}")
    logger.info(f"[Prep] Total days: {len(df_clean)}")
    
    stats.update({
        "clean_rows": len(df_clean),
        "clean_cols": len(df_clean.columns),
        "features_count": len([c for c in df_clean.columns if c not in ["date", "label_3cls", "segment_id"]]),
        "date_min": df_clean["date"].min(),
        "date_max": df_clean["date"].max(),
        "missing_values": missing_data,
        "label_distribution": label_dist.to_dict()
    })
    
    # Optional: save cleaned data
    if output_path:
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        df_clean.to_csv(output_path, index=False)
        logger.info(f"\n[OK] Saved cleaned data: {output_path}")
    
    logger.info(f"\n[OK] Data preparation complete")
    logger.info(f"[OK] Cleaned dataset: {df_clean.shape[0]} rows × {df_clean.shape[1]} columns")
    logger.info(f"[OK] Anti-leak verified: pbsi_score removed, label_* removed, segment_id {('removed' if remove_segment_id else 'kept')}")
    
    return df_clean, stats


if __name__ == "__main__":
    import sys
    import json
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "data/etl/P000001/2025-11-07/joined/features_daily_labeled.csv"
    remove_segment = "--remove-segment-id" in sys.argv
    output_path = None
    
    # Find output path in argv
    for i, arg in enumerate(sys.argv):
        if arg == "--output":
            output_path = sys.argv[i+1]
            break
    
    df_clean, stats = prepare_nb2_dataset(csv_path, remove_segment_id=remove_segment, output_path=output_path)
    
    print(f"\n[DONE] Preparation complete: {df_clean.shape}")
