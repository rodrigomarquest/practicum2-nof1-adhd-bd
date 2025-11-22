"""
Temporal Instability Score Computation for Feature Regularization.

This module computes variance-based instability scores for features across
behavioral segments. High instability indicates features that change dramatically
between segments, which may not generalize well for temporal prediction.

Usage:
    from src.utils.temporal_instability import compute_instability_scores
    
    scores = compute_instability_scores(
        features_df=df,
        segments_csv='data/etl/P000001/2025-11-07/segment_autolog.csv',
        feature_cols=['hr_mean', 'sleep_hours', ...],
        date_col='date'
    )
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


def load_segments(segments_path: str) -> pd.DataFrame:
    """Load behavioral segments from segment_autolog.csv."""
    df = pd.read_csv(segments_path)
    df['date_start'] = pd.to_datetime(df['date_start'])
    df['date_end'] = pd.to_datetime(df['date_end'])
    return df


def assign_segment_ids(
    df: pd.DataFrame, 
    segments: pd.DataFrame, 
    date_col: str = 'date'
) -> pd.DataFrame:
    """
    Assign segment_id to each row based on date falling within segment boundaries.
    
    Args:
        df: DataFrame with features and dates
        segments: DataFrame from segment_autolog.csv
        date_col: Name of date column
    
    Returns:
        df with 'segment_id' column added
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df['segment_id'] = -1  # Default for unmatched
    
    for idx, seg in segments.iterrows():
        mask = (df[date_col] >= seg['date_start']) & (df[date_col] <= seg['date_end'])
        df.loc[mask, 'segment_id'] = idx
    
    n_unassigned = (df['segment_id'] == -1).sum()
    if n_unassigned > 0:
        logger.warning(f"{n_unassigned} rows not assigned to any segment")
    
    return df


def compute_instability_scores(
    features_df: pd.DataFrame,
    segments_csv: str,
    feature_cols: List[str],
    date_col: str = 'date',
    normalize: bool = True
) -> Dict[str, float]:
    """
    Compute temporal instability scores for each feature.
    
    Algorithm:
        1. Assign each data point to a behavioral segment
        2. For each feature f:
            - Compute mean(f) per segment
            - Compute variance across segment means
            - instability[f] = Var(segment_means[f])
        3. Normalize to [0, 1]: instability_norm[f] = instability[f] / max(instability)
    
    Args:
        features_df: DataFrame with features and dates
        segments_csv: Path to segment_autolog.csv
        feature_cols: List of feature column names
        date_col: Name of date column
        normalize: Whether to normalize scores to [0, 1]
    
    Returns:
        Dictionary mapping feature_name -> instability_score
    """
    logger.info(f"Computing instability scores for {len(feature_cols)} features")
    
    # Load segments
    segments = load_segments(segments_csv)
    logger.info(f"Loaded {len(segments)} behavioral segments")
    
    # Assign segment IDs
    df = assign_segment_ids(features_df, segments, date_col)
    
    # Remove unassigned rows
    df = df[df['segment_id'] != -1].copy()
    
    # Compute instability per feature
    instability = {}
    
    for feat in feature_cols:
        # Skip if column doesn't exist or is all NaN
        if feat not in df.columns or df[feat].isna().all():
            instability[feat] = 0.0
            continue
        
        # Compute mean per segment
        segment_means = df.groupby('segment_id')[feat].mean()
        
        # Variance across segments (instability score)
        var = segment_means.var()
        
        # Handle NaN (can happen if only one segment or all values identical)
        if pd.isna(var):
            var = 0.0
        
        instability[feat] = float(var)
    
    logger.info(f"Computed raw instability scores (mean={np.mean(list(instability.values())):.4f})")
    
    # Normalize to [0, 1]
    if normalize:
        max_instability = max(instability.values()) if instability else 1.0
        if max_instability > 0:
            instability = {k: v / max_instability for k, v in instability.items()}
        logger.info("Normalized instability scores to [0, 1]")
    
    # Log top unstable features
    sorted_features = sorted(instability.items(), key=lambda x: x[1], reverse=True)
    logger.info("Top 5 most unstable features:")
    for feat, score in sorted_features[:5]:
        logger.info(f"  {feat}: {score:.4f}")
    
    return instability


def save_instability_scores(scores: Dict[str, float], output_path: str):
    """Save instability scores to CSV."""
    df = pd.DataFrame([
        {'feature': k, 'instability_score': v}
        for k, v in sorted(scores.items(), key=lambda x: x[1], reverse=True)
    ])
    df.to_csv(output_path, index=False)
    logger.info(f"Saved instability scores to {output_path}")


if __name__ == '__main__':
    # Example usage
    import sys
    logging.basicConfig(level=logging.INFO)
    
    if len(sys.argv) < 2:
        print("Usage: python temporal_instability.py <features_csv> <segments_csv> <output_csv>")
        sys.exit(1)
    
    features_path = sys.argv[1]
    segments_path = sys.argv[2]
    output_path = sys.argv[3] if len(sys.argv) > 3 else 'instability_scores.csv'
    
    # Load features
    df = pd.read_csv(features_path)
    
    # Detect feature columns (exclude date and labels)
    exclude_cols = ['date', 'label_3cls', 'label_2cls', 'segment_id']
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    
    # Compute scores
    scores = compute_instability_scores(
        features_df=df,
        segments_csv=segments_path,
        feature_cols=feature_cols
    )
    
    # Save
    save_instability_scores(scores, output_path)
    print(f"✅ Instability scores saved to {output_path}")
