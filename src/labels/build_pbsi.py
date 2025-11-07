"""
Build PBSI (Physio-Behavioral Stability Index) labels.

Uses only: sleep + cardio + activity (no temperature, screen time, or EMA/SoM).

Labels:
    label_3cls: +1 (stable), 0 (neutral), -1 (unstable)
    label_2cls: 1 (stable/high-pbsi), 0 (low-pbsi)
    label_clinical: 1 if pbsi_score >= 0.75 else 0
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict
import logging

logger = logging.getLogger(__name__)


def compute_z_scores_by_segment(
    df: pd.DataFrame,
    version_log_path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Compute z-scores within each segment (S1..S6).
    
    If version_log not available, compute global z-scores.
    """
    df = df.copy()
    
    # Join segment info if available
    if version_log_path and version_log_path.exists():
        try:
            vlog = pd.read_csv(version_log_path)
            vlog['date'] = pd.to_datetime(vlog['date'])
            df = df.merge(vlog[['date', 'segment_id']], on='date', how='left')
            logger.info("✓ Joined segment info from version_log_enriched")
        except Exception as e:
            logger.warning(f"Could not load version_log: {e}. Using global z-scores.")
            df['segment_id'] = 'global'
    else:
        df['segment_id'] = 'global'
    
    # Compute z-scores per segment
    features_for_zscores = [
        'sleep_total_h', 'sleep_efficiency',
        'apple_hr_mean', 'apple_hrv_rmssd', 'apple_hr_max',
        'steps', 'exercise_min',
    ]
    
    for feat in features_for_zscores:
        if feat not in df.columns:
            continue
        
        z_col = f'z_{feat}'
        df[z_col] = np.nan
        
        for segment in df['segment_id'].unique():
            mask = df['segment_id'] == segment
            seg_data = df[mask][feat]
            
            mean = seg_data.mean()
            std = seg_data.std()
            
            if pd.notna(mean) and std > 0:
                df.loc[mask, z_col] = (seg_data - mean) / std
            else:
                df.loc[mask, z_col] = 0.0
    
    logger.info("✓ Computed z-scores by segment")
    return df


def _get_z_safe(row: pd.Series, col: str, default: float = 0.0) -> float:
    """Safely get z-score with NaN handling."""
    val = row.get(col, default)
    return val if pd.notna(val) else default


def compute_pbsi_score(row: pd.Series) -> Dict:
    """Compute PBSI subscores and composite score for a single row."""
    
    result = {}
    
    # Sleep subscore
    z_sleep_dur = _get_z_safe(row, 'z_sleep_total_h')
    z_sleep_eff = _get_z_safe(row, 'z_sleep_efficiency')
    
    sleep_sub = -0.6 * z_sleep_dur + 0.4 * z_sleep_eff
    sleep_sub = np.clip(sleep_sub, -3, 3)
    result['sleep_sub'] = sleep_sub
    
    # Cardio subscore
    z_hr_mean = _get_z_safe(row, 'z_apple_hr_mean')
    z_hrv = _get_z_safe(row, 'z_apple_hrv_rmssd')
    z_hr_max = _get_z_safe(row, 'z_apple_hr_max')
    
    cardio_sub = 0.5 * z_hr_mean - 0.6 * z_hrv + 0.2 * z_hr_max
    cardio_sub = np.clip(cardio_sub, -3, 3)
    result['cardio_sub'] = cardio_sub
    
    # Activity subscore
    z_steps = _get_z_safe(row, 'z_steps')
    z_exercise = _get_z_safe(row, 'z_exercise_min')
    if pd.isna(row.get('z_exercise_min')):
        z_exercise = _get_z_safe(row, 'z_move_kcal')
    
    activity_sub = -0.7 * z_steps - 0.3 * z_exercise
    activity_sub = np.clip(activity_sub, -3, 3)
    result['activity_sub'] = activity_sub
    
    # Composite
    pbsi_score = 0.40 * sleep_sub + 0.35 * cardio_sub + 0.25 * activity_sub
    result['pbsi_score'] = pbsi_score
    
    # Labels
    result['label_3cls'] = 1 if pbsi_score <= -0.5 else (-1 if pbsi_score >= 0.5 else 0)
    result['label_2cls'] = 1 if result['label_3cls'] == 1 else 0
    result['label_clinical'] = 1 if pbsi_score >= 0.75 else 0
    
    # Quality score
    quality = 1.0
    if row.get('missing_sleep', 0):
        quality *= 0.8
    if row.get('missing_cardio', 0):
        quality *= 0.8
    if row.get('missing_activity', 0):
        quality *= 0.8
    result['pbsi_quality'] = max(quality, 0.5)
    
    return result


def build_pbsi_labels(
    unified_df: pd.DataFrame,
    version_log_path: Optional[Path] = None,
    output_path: Optional[Path] = None,
) -> pd.DataFrame:
    """Main function: compute z-scores, PBSI, and save."""
    
    logger.info("Building PBSI labels")
    
    df = unified_df.copy()
    
    # Compute z-scores
    df = compute_z_scores_by_segment(df, version_log_path)
    
    # Compute PBSI for each row
    pbsi_cols = ['sleep_sub', 'cardio_sub', 'activity_sub', 'pbsi_score',
                 'label_3cls', 'label_2cls', 'label_clinical', 'pbsi_quality']
    
    pbsi_data = df.apply(compute_pbsi_score, axis=1, result_type='expand')
    for col in pbsi_cols:
        df[col] = pbsi_data[col]
    
    logger.info("✓ Computed PBSI scores")
    
    # Report
    logger.info("\n" + "="*80)
    logger.info("LABEL DISTRIBUTION")
    logger.info("="*80)
    
    for label_col in ['label_3cls', 'label_2cls', 'label_clinical']:
        logger.info(f"\n{label_col}:")
        vc = df[label_col].value_counts().sort_index()
        for val, count in vc.items():
            pct = 100 * count / len(df)
            logger.info(f"  {val}: {count} ({pct:.1f}%)")
    
    # Sanity check
    assert df['label_3cls'].nunique() > 1, "Degenerate: only one class in label_3cls!"
    assert df['label_2cls'].nunique() > 1, "Degenerate: only one class in label_2cls!"
    
    logger.info("No degenerate labels")
    
    # Example rows
    logger.info("\nExample stable (+1) rows:")
    stable = df[df['label_3cls'] == 1][['date', 'pbsi_score', 'sleep_sub', 'cardio_sub', 'activity_sub', 'pbsi_quality']].head(3)
    logger.info(stable.to_string())
    
    logger.info("\nExample unstable (-1) rows:")
    unstable = df[df['label_3cls'] == -1][['date', 'pbsi_score', 'sleep_sub', 'cardio_sub', 'activity_sub', 'pbsi_quality']].head(3)
    logger.info(unstable.to_string())
    
    logger.info("="*80 + "\n")
    
    # Save
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        logger.info(f"✓ Saved labeled data to {output_path}")
    
    return df
