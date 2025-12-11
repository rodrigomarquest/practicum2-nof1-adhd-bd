"""
Build PBSI (Physio-Behavioral Stability Index) labels.

CANONICAL IMPLEMENTATION FOR CA2 PAPER (v4.1.6):
This module implements the segment-wise z-scored PBSI described in the research paper.
It is integrated into the main ETL pipeline via stage_apply_labels.py.

Key features:
- Segment-wise z-score normalization (anti-leak safeguard)
- Sleep/cardio/activity subscores with documented weights
- Composite PBSI: 0.40*sleep + 0.35*cardio + 0.25*activity
- Percentile-based thresholds (P25/P75) for balanced class distribution

Sign convention (v4.1.7 - INTUITIVE):
✓ HIGHER PBSI score = BETTER physiological regulation
  - More sleep, lower HR, higher HRV → positive subscores → HIGHER pbsi → +1 (regulated)
  - Less sleep, higher HR, lower HRV → negative subscores → LOWER pbsi → -1 (dysregulated)

Labels (v4.1.7):
    label_3cls: +1 (high_pbsi/regulated), 0 (mid_pbsi/typical), -1 (low_pbsi/dysregulated)
    label_2cls: 1 (regulated), 0 (typical/dysregulated)
    label_clinical: deprecated (replaced by label_3cls)

⚠️ IMPORTANT - Clinical Validity:
These labels are COMPOSITE PHYSIOLOGICAL INDICES and have NOT been validated against
psychiatric ground truth (mood diaries, clinician ratings, or diagnostic interviews).
They capture variance in sleep/cardio/activity patterns but should NOT be interpreted
as direct proxies for psychiatric states (mania, depression, ADHD severity).

For clinical interpretation, cross-reference with:
- Mood diaries / self-reports
- Medication changes / life events
- Clinical assessments (YMRS, MADRS, ASRS)

Future work (v5.x): Validate against ecological momentary assessments (EMA) and
DSM-5 diagnostic criteria to map physiological patterns to psychiatric states.
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
            # Handle both 'date' and 'day_boundary' column names
            date_col = 'date' if 'date' in vlog.columns else 'day_boundary'
            vlog[date_col] = pd.to_datetime(vlog[date_col])
            df['date'] = pd.to_datetime(df['date'])
            df = df.merge(vlog[[date_col, 'segment_id']], left_on='date', right_on=date_col, how='left')
            if date_col != 'date':
                df = df.drop(columns=[date_col])
            logger.info("✓ Joined segment info from version_log")
        except Exception as e:
            logger.warning(f"Could not load version_log: {e}. Using existing segment_id or global.")
            if 'segment_id' not in df.columns:
                df['segment_id'] = 'global'
    else:
        # Preserve existing segment_id if present (e.g., from stage_apply_labels temporal segmentation)
        if 'segment_id' not in df.columns:
            df['segment_id'] = 'global'
            logger.info("ℹ No segment_id in data, using global z-scores")
        else:
            n_segments = df['segment_id'].nunique()
            logger.info(f"✓ Using existing segment_id ({n_segments} segments) for z-scores")
    
    # Compute z-scores per segment
    features_for_zscores = [
        'sleep_total_h', 'sleep_efficiency',
        'hr_mean', 'hrv_rmssd', 'hr_max',
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


def compute_pbsi_score(
    row: pd.Series,
    threshold_low: float = -0.5,
    threshold_high: float = 0.5
) -> Dict:
    """
    Compute PBSI subscores and composite score for a single row.
    
    Args:
        row: DataFrame row with z-scored features
        threshold_low: Lower threshold for label_3cls (default: -0.5, v4.1.6: P25)
        threshold_high: Upper threshold for label_3cls (default: 0.5, v4.1.6: P75)
    
    Returns:
        Dictionary with subscores, pbsi_score, labels, and quality metrics
    """
    result = {}
    
    # Sleep subscore (v4.1.7: INVERTED - higher = better)
    z_sleep_dur = _get_z_safe(row, 'z_sleep_total_h')
    z_sleep_eff = _get_z_safe(row, 'z_sleep_efficiency')
    
    sleep_sub = 0.6 * z_sleep_dur + 0.4 * z_sleep_eff  # More sleep + better efficiency = HIGHER
    sleep_sub = np.clip(sleep_sub, -3, 3)
    result['sleep_sub'] = sleep_sub
    
    # Cardio subscore (v4.1.7: INVERTED - higher = better)
    z_hr_mean = _get_z_safe(row, 'z_hr_mean')
    z_hrv = _get_z_safe(row, 'z_hrv_rmssd')
    z_hr_max = _get_z_safe(row, 'z_hr_max')
    
    cardio_sub = -0.5 * z_hr_mean + 0.6 * z_hrv - 0.2 * z_hr_max  # Lower HR + higher HRV = HIGHER
    cardio_sub = np.clip(cardio_sub, -3, 3)
    result['cardio_sub'] = cardio_sub
    
    # Activity subscore (v4.1.7: INVERTED - higher = better)
    z_steps = _get_z_safe(row, 'z_steps')
    z_exercise = _get_z_safe(row, 'z_exercise_min')
    if pd.isna(row.get('z_exercise_min')):
        z_exercise = _get_z_safe(row, 'z_move_kcal')
    
    activity_sub = 0.7 * z_steps + 0.3 * z_exercise  # More steps + more exercise = HIGHER
    activity_sub = np.clip(activity_sub, -3, 3)
    result['activity_sub'] = activity_sub
    
    # Composite (v4.1.7: HIGHER PBSI = BETTER regulation)
    pbsi_score = 0.40 * sleep_sub + 0.35 * cardio_sub + 0.25 * activity_sub
    result['pbsi_score'] = pbsi_score
    
    # Labels (v4.1.7: INVERTED thresholds - higher = better)
    # +1 (high_pbsi): physiologically regulated (HIGH score = GOOD)
    # 0 (mid_pbsi): typical patterns
    # -1 (low_pbsi): physiologically dysregulated (LOW score = BAD)
    result['label_3cls'] = 1 if pbsi_score >= threshold_high else (
        -1 if pbsi_score <= threshold_low else 0
    )
    result['label_2cls'] = 1 if result['label_3cls'] == 1 else 0
    
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
    use_percentile_thresholds: bool = True,
    threshold_low_percentile: float = 0.25,
    threshold_high_percentile: float = 0.75,
    threshold_low_fixed: float = -0.5,
    threshold_high_fixed: float = 0.5,
) -> pd.DataFrame:
    """
    Main function: compute z-scores, PBSI, and save.
    
    Args:
        unified_df: Input dataframe with raw features
        version_log_path: Path to segment version log
        output_path: Path to save labeled dataset
        use_percentile_thresholds: If True, use percentile-based thresholds (v4.1.6)
        threshold_low_percentile: Percentile for lower threshold (default: 0.25 = P25)
        threshold_high_percentile: Percentile for upper threshold (default: 0.75 = P75)
        threshold_low_fixed: Fixed lower threshold (fallback, default: -0.5)
        threshold_high_fixed: Fixed upper threshold (fallback, default: 0.5)
    
    Returns:
        DataFrame with PBSI scores and labels
    """
    logger.info("Building PBSI labels (v4.1.7 - INTUITIVE sign convention)")
    
    df = unified_df.copy()
    
    # Compute z-scores
    df = compute_z_scores_by_segment(df, version_log_path)
    
    # First pass: compute PBSI scores with default thresholds
    pbsi_cols = ['sleep_sub', 'cardio_sub', 'activity_sub', 'pbsi_score',
                 'label_3cls', 'label_2cls', 'pbsi_quality']
    
    pbsi_data = df.apply(lambda row: compute_pbsi_score(row), axis=1, result_type='expand')
    for col in pbsi_cols:
        df[col] = pbsi_data[col]
    
    # Determine thresholds
    if use_percentile_thresholds:
        pbsi_scores = df['pbsi_score'].dropna()
        threshold_low = pbsi_scores.quantile(threshold_low_percentile)
        threshold_high = pbsi_scores.quantile(threshold_high_percentile)
        logger.info(f"✓ Using percentile-based thresholds (P{int(threshold_low_percentile*100)}/P{int(threshold_high_percentile*100)})")
        logger.info(f"  Threshold low (P{int(threshold_low_percentile*100)}):  {threshold_low:.3f} → -1 (low_pbsi/dysregulated)")
        logger.info(f"  Threshold high (P{int(threshold_high_percentile*100)}): {threshold_high:.3f} → +1 (high_pbsi/regulated)")
        logger.info(f"  v4.1.7: HIGHER PBSI = BETTER regulation (intuitive!)")
    else:
        threshold_low = threshold_low_fixed
        threshold_high = threshold_high_fixed
        logger.info(f"✓ Using fixed thresholds: {threshold_low:.3f} / {threshold_high:.3f}")
    
    # Second pass: re-compute labels with determined thresholds
    pbsi_data = df.apply(
        lambda row: compute_pbsi_score(row, threshold_low, threshold_high),
        axis=1,
        result_type='expand'
    )
    for col in pbsi_cols:
        df[col] = pbsi_data[col]
    
    logger.info("✓ Computed PBSI scores and labels")
    
    # Report
    logger.info("\n" + "="*80)
    logger.info("LABEL DISTRIBUTION (v4.1.6)")
    logger.info("="*80)
    
    for label_col in ['label_3cls', 'label_2cls']:
        if label_col in df.columns:
            logger.info(f"\n{label_col}:")
            vc = df[label_col].value_counts().sort_index()
            for val, count in vc.items():
                pct = 100 * count / len(df)
                # Label interpretation
                if label_col == 'label_3cls':
                    label_name = {1: 'low_pbsi (regulated)', 0: 'mid_pbsi (typical)', -1: 'high_pbsi (dysregulated)'}
                    logger.info(f"  {val} [{label_name.get(val, 'unknown')}]: {count} ({pct:.1f}%)")
                else:
                    logger.info(f"  {val}: {count} ({pct:.1f}%)")
    
    # Sanity check
    assert df['label_3cls'].nunique() > 1, "Degenerate: only one class in label_3cls!"
    assert df['label_2cls'].nunique() > 1, "Degenerate: only one class in label_2cls!"
    
    logger.info("\n✓ No degenerate labels")
    
    # Example rows
    logger.info("\nExample LOW_PBSI (+1, regulated) rows:")
    stable = df[df['label_3cls'] == 1][['date', 'pbsi_score', 'sleep_sub', 'cardio_sub', 'activity_sub', 'pbsi_quality']].head(3)
    logger.info(stable.to_string())
    
    logger.info("\nExample HIGH_PBSI (-1, dysregulated) rows:")
    unstable = df[df['label_3cls'] == -1][['date', 'pbsi_score', 'sleep_sub', 'cardio_sub', 'activity_sub', 'pbsi_quality']].head(3)
    logger.info(unstable.to_string())
    
    logger.info("="*80 + "\n")
    
    # Save
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        logger.info(f"✓ Saved labeled data to {output_path}")
    
    return df
