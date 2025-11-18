"""
Stage 3: Apply PBSI Labels
Apply PBSI mood labels to expanded daily dataset.

CANONICAL IMPLEMENTATION (CA2 Paper):
This stage now uses the segment-wise z-scored PBSI from src/labels/build_pbsi.py.
This is the implementation described in the research paper and documentation.

Input: features_daily_unified.csv (from Stage 2)
Output: features_daily_labeled.csv with:
        - segment_id (temporal segments for z-score normalization)
        - pbsi_score (z-scored composite: 0.40*sleep + 0.35*cardio + 0.25*activity)
        - label_3cls: +1 (stable, pbsi≤-0.5), 0 (neutral), -1 (unstable, pbsi≥0.5)
        - label_2cls, label_clinical, pbsi_quality

Sign convention: Lower PBSI score = more stable (counterintuitive but by design)
  - More sleep, lower HR, higher HRV → lower subscores → lower pbsi_score → +1 (stable)
  - Less sleep, higher HR, lower HRV → higher subscores → higher pbsi_score → -1 (unstable)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import logging
import yaml
from typing import Dict, Optional

# Import the canonical PBSI implementation
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from labels.build_pbsi import build_pbsi_labels

logger = logging.getLogger(__name__)


def _create_temporal_segments(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create temporal segment_id for the dataframe based on gaps and time boundaries.
    
    This provides the segment boundaries needed for segment-wise z-score normalization
    in the canonical PBSI implementation.
    
    Segmentation rules:
    - New segment on gap > 1 day
    - New segment on month/year boundary
    
    Returns:
        DataFrame with added 'segment_id' column (1-indexed integer)
    """
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    
    segment_id = np.ones(len(df), dtype=int)
    current_segment = 1
    
    for i in range(1, len(df)):
        prev_date = df.iloc[i-1]['date']
        curr_date = df.iloc[i]['date']
        delta = (curr_date - prev_date).days
        
        # Gap detection
        if delta > 1:
            current_segment += 1
            logger.debug(f"  Segment break at {curr_date} (gap={delta} days)")
        # Time boundary (month/year change)
        elif prev_date.month != curr_date.month or prev_date.year != curr_date.year:
            current_segment += 1
            logger.debug(f"  Segment break at {curr_date} (month/year boundary)")
        
        segment_id[i] = current_segment
    
    df['segment_id'] = segment_id
    
    n_segments = df['segment_id'].nunique()
    segment_sizes = df.groupby('segment_id').size()
    logger.info(f"[Segments] Created {n_segments} temporal segments")
    logger.info(f"[Segments] Size range: {segment_sizes.min()}-{segment_sizes.max()} days")
    
    return df


def _normalize_column_names_for_pbsi(df: pd.DataFrame) -> pd.DataFrame:
    """
    Map unified daily column names to the names expected by build_pbsi.py.
    
    Stage 2 (unify_daily) produces:
        sleep_hours, sleep_quality_score, hr_mean, hr_max, hr_std, total_steps, ...
    
    build_pbsi.py expects:
        sleep_total_h, sleep_efficiency, apple_hr_mean, apple_hr_max, 
        apple_hrv_rmssd, steps, exercise_min
        missing_sleep, missing_cardio, missing_activity (for quality scores)
    
    This function creates the necessary columns (with best approximations where needed).
    """
    df = df.copy()
    
    # Sleep mapping
    if 'sleep_hours' in df.columns:
        df['sleep_total_h'] = df['sleep_hours']
        df['missing_sleep'] = df['sleep_hours'].isna() | (df['sleep_hours'] == 0)
    else:
        df['sleep_total_h'] = np.nan
        df['missing_sleep'] = True
    
    if 'sleep_quality_score' in df.columns:
        # sleep_quality_score is 0-100, sleep_efficiency is typically 0-1 or 0-100
        # Assuming sleep_quality_score is already a percentage (0-100)
        df['sleep_efficiency'] = df['sleep_quality_score'] / 100.0
    else:
        df['sleep_efficiency'] = np.nan
    
    # Cardio mapping
    if 'hr_mean' in df.columns:
        df['apple_hr_mean'] = df['hr_mean']
        df['missing_cardio'] = df['hr_mean'].isna()
    else:
        df['apple_hr_mean'] = np.nan
        df['missing_cardio'] = True
    
    if 'hr_max' in df.columns:
        df['apple_hr_max'] = df['hr_max']
    else:
        df['apple_hr_max'] = np.nan
    
    # HRV: if not present, we need to handle this
    # build_pbsi will use z_hrv, so missing HRV will just use default z-score of 0
    if 'hrv' in df.columns:
        df['apple_hrv_rmssd'] = df['hrv']
    elif 'hr_std' in df.columns:
        # APPROXIMATION: Use HR std as proxy for HRV variability
        # This is not ideal but better than nothing
        # Inverse relationship: higher std = more variability (could indicate higher HRV)
        logger.warning("[Mapping] No HRV data, using hr_std as rough proxy")
        df['apple_hrv_rmssd'] = df['hr_std'] * 2.0  # Scale factor
    else:
        logger.warning("[Mapping] No HRV or hr_std available")
        df['apple_hrv_rmssd'] = np.nan
    
    # Activity mapping
    if 'total_steps' in df.columns:
        df['steps'] = df['total_steps']
        df['missing_activity'] = df['total_steps'].isna() | (df['total_steps'] == 0)
    else:
        df['steps'] = np.nan
        df['missing_activity'] = True
    
    # Exercise: if not present in unified data, try to infer
    if 'exercise_min' not in df.columns:
        if 'total_active_energy' in df.columns:
            # APPROXIMATION: Estimate exercise minutes from active energy
            # Rough estimate: 1 MET-min ≈ 1 kcal, moderate exercise ≈ 5 METs
            # So 50 kcal ≈ 10 min moderate exercise
            logger.warning("[Mapping] No exercise_min, estimating from active_energy")
            df['exercise_min'] = df['total_active_energy'] / 5.0
        else:
            logger.warning("[Mapping] No exercise data available")
            df['exercise_min'] = 0.0
    
    # Ensure missing flags are boolean
    df['missing_sleep'] = df['missing_sleep'].fillna(False).astype(int)
    df['missing_cardio'] = df['missing_cardio'].fillna(False).astype(int)
    df['missing_activity'] = df['missing_activity'].fillna(False).astype(int)
    
    logger.info("[Mapping] Normalized column names for canonical PBSI")
    
    return df


class PBSILabeler:
    """
    Apply PBSI labels using the CANONICAL segment-wise z-scored implementation.
    
    This delegates to build_pbsi.py which implements:
    - Segment-wise z-score normalization (anti-leak safeguard)
    - Sleep/cardio/activity subscores
    - Composite PBSI: 0.40*sleep + 0.35*cardio + 0.25*activity
    - Thresholds: pbsi≤-0.5 → +1 (stable), pbsi≥0.5 → -1 (unstable)
    """
    
    def __init__(self, label_rules_path: str = "config/label_rules.yaml"):
        """Initialize (label_rules currently unused but kept for API compatibility)."""
        self.label_rules_path = Path(label_rules_path)
        logger.info(f"[Labels] Using canonical PBSI (segment-wise z-scored)")
    
    def apply_labels(self, df: pd.DataFrame, version_log_path: Optional[Path] = None) -> pd.DataFrame:
        """
        Apply canonical PBSI labels to dataframe.
        
        This method:
        1. Normalizes column names (unify_daily → build_pbsi naming)
        2. Ensures segment_id exists (creates it if needed)
        3. Calls build_pbsi_labels() for segment-wise z-scored PBSI
        4. Returns dataframe with pbsi_score, label_3cls, etc.
        
        Args:
            df: Input dataframe with daily features
            version_log_path: Optional path to version_log_enriched.csv for segments
                             (if None, will use segment_id from df or create temporal segments)
        
        Returns:
            DataFrame with added columns:
            - segment_id: temporal segment identifier
            - z_{feature}: z-scores per segment
            - sleep_sub, cardio_sub, activity_sub: PBSI subscores
            - pbsi_score: composite score (z-scaled, lower=more stable)
            - label_3cls: +1 (stable), 0 (neutral), -1 (unstable)
            - label_2cls: binary version
            - label_clinical: high-threshold flag
            - pbsi_quality: data completeness score
        """
        logger.info(f"[Labels] Applying canonical PBSI to {len(df)} days...")
        
        # Step 1: Normalize column names
        df = _normalize_column_names_for_pbsi(df)
        
        # Step 2: Ensure segment_id exists
        if 'segment_id' not in df.columns:
            logger.info("[Labels] segment_id not found, creating temporal segments...")
            df = _create_temporal_segments(df)
        else:
            logger.info(f"[Labels] Using existing segment_id ({df['segment_id'].nunique()} segments)")
        
        # Step 3: Call canonical PBSI implementation
        df = build_pbsi_labels(
            unified_df=df,
            version_log_path=version_log_path,
            output_path=None  # Don't save here, we'll save in run_apply_labels
        )
        
        logger.info(f"[Labels] Canonical PBSI applied")
        logger.info(f"[Labels] Score range: {df['pbsi_score'].min():.3f} to {df['pbsi_score'].max():.3f}")
        logger.info(f"[Labels] Distribution:")
        for label in [1, 0, -1]:  # Order: stable, neutral, unstable
            count = (df["label_3cls"] == label).sum()
            pct = count / len(df) * 100
            label_name = {1: "stable", 0: "neutral", -1: "unstable"}[label]
            logger.info(f"  Label {label:+2d} ({label_name:8s}): {count:4d} days ({pct:5.1f}%)")
        
        return df


def run_apply_labels(participant: str = "P000001",
                     snapshot: Optional[str] = None,
                     etl_dir: str = "data/etl",
                     label_rules: str = "config/label_rules.yaml") -> pd.DataFrame:
    """
    Execute Stage 3: Apply PBSI Labels
    
    Takes unified daily data and applies PBSI mood labels.
    
    Args:
        participant: Participant ID
        snapshot: Snapshot date; if None, uses today
        etl_dir: ETL data directory
        label_rules: Path to label rules YAML
    
    Returns:
        Labeled dataframe with columns: ..., pbsi_score, pbsi_quality, label_3cls
    """
    
    if snapshot is None:
        snapshot = datetime.now().strftime("%Y-%m-%d")
    
    logger.info(f"\n{'='*80}")
    logger.info(f"STAGE 3: APPLY PBSI LABELS")
    logger.info(f"Participant: {participant}, Snapshot: {snapshot}")
    logger.info(f"{'='*80}\n")
    
    # Load unified data
    input_path = Path(etl_dir) / participant / snapshot / "joined" / "features_daily_unified.csv"
    if not input_path.exists():
        raise FileNotFoundError(f"Unified data not found: {input_path}")
    
    logger.info(f"[Labels] Loading: {input_path}")
    df = pd.read_csv(input_path)
    logger.info(f"[Labels] Loaded {len(df)} days")
    
    # Apply labels
    labeler = PBSILabeler(label_rules_path=label_rules)
    df = labeler.apply_labels(df)
    
    # Save labeled data
    output_dir = Path(etl_dir) / participant / snapshot / "joined"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / "features_daily_labeled.csv"
    df.to_csv(output_path, index=False)
    
    logger.info(f"\n[OK] Saved labeled data: {output_path}")
    logger.info(f"[OK] Records: {len(df)}, Columns: {len(df.columns)}")
    
    return df


if __name__ == "__main__":
    import sys
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    participant = sys.argv[1] if len(sys.argv) > 1 else "P000001"
    snapshot = sys.argv[2] if len(sys.argv) > 2 else None
    
    df = run_apply_labels(participant=participant, snapshot=snapshot)
    
    print(f"\n[DONE] Apply Labels complete: {len(df)} days with labels")
