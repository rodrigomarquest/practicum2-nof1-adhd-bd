"""
Aggregate all biomarkers into daily feature matrix.

This module orchestrates extraction of all Tier 1, 2, and X features from raw data
and combines them into a single daily DataFrame suitable for ML/clinical analysis.

Architecture:
1. Load raw data (Apple, Zepp, Ring)
2. Apply segmentation (S1-S6)
3. Extract Tier 1 features (HRV, Sleep, Activity, Circadian)
4. Extract Tier 2 features (Sleep timing, HR trends, Body composition)
5. Extract Tier X features (Cross-device validation)
6. Generate data quality flags
7. Merge all into daily matrix

Output: joined_features_daily_biomarkers.csv
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple
import logging

from . import segmentation, hrv, sleep, activity, circadian, validators

logger = logging.getLogger(__name__)


def aggregate_daily_biomarkers(
    zepp_hr_auto_path: Optional[str] = None,
    zepp_sleep_path: Optional[str] = None,
    zepp_activity_stage_path: Optional[str] = None,
    zepp_activity_minute_path: Optional[str] = None,
    apple_hr_path: Optional[str] = None,
    apple_hrv_path: Optional[str] = None,
    apple_activity_path: Optional[str] = None,
    cutoff_date: Optional[pd.Timestamp] = None,
    participant: str = "P000001",
    snapshot: str = "2025-11-07",
) -> pd.DataFrame:
    """
    Main aggregation pipeline for biomarkers.

    Parameters
    ----------
    zepp_hr_auto_path : str, optional
        Path to Zepp HEARTRATE_AUTO CSV
    zepp_sleep_path : str, optional
        Path to Zepp SLEEP CSV
    zepp_activity_stage_path : str, optional
        Path to Zepp ACTIVITY_STAGE CSV
    zepp_activity_minute_path : str, optional
        Path to Zepp ACTIVITY_MINUTE CSV
    apple_hr_path : str, optional
        Path to Apple HR CSV (per-metric aggregated)
    apple_hrv_path : str, optional
        Path to Apple HRV CSV
    apple_activity_path : str, optional
        Path to Apple Activity CSV
    cutoff_date : pd.Timestamp, optional
        Filter data to <= cutoff_date
    participant : str
        Participant ID
    snapshot : str
        Snapshot date (for file naming)

    Returns
    -------
    pd.DataFrame
        Daily biomarkers matrix with all Tier 1+2+X features
    """
    logger.info(f"Starting biomarkers aggregation for {participant}/{snapshot}")

    # Step 1: Load raw data
    zepp_hr_auto_df = _load_csv(zepp_hr_auto_path)
    zepp_sleep_df = _load_csv(zepp_sleep_path)
    zepp_activity_stage_df = _load_csv(zepp_activity_stage_path)
    zepp_activity_minute_df = _load_csv(zepp_activity_minute_path)
    apple_hr_df = _load_csv(apple_hr_path)
    apple_hrv_df = _load_csv(apple_hrv_path)
    apple_activity_df = _load_csv(apple_activity_path)

    # Step 2: Apply cutoff date
    if cutoff_date:
        zepp_hr_auto_df = _apply_date_filter(zepp_hr_auto_df, cutoff_date)
        zepp_sleep_df = _apply_date_filter(zepp_sleep_df, cutoff_date)
        zepp_activity_stage_df = _apply_date_filter(zepp_activity_stage_df, cutoff_date)
        zepp_activity_minute_df = _apply_date_filter(zepp_activity_minute_df, cutoff_date)
        apple_hr_df = _apply_date_filter(apple_hr_df, cutoff_date)
        apple_hrv_df = _apply_date_filter(apple_hrv_df, cutoff_date)
        apple_activity_df = _apply_date_filter(apple_activity_df, cutoff_date)

    # Step 3: Extract Tier 1 features
    logger.info("Extracting Tier 1 features...")
    df_hrv = hrv.compute_hrv_daily(zepp_hr_auto_df, apple_hrv_df)
    df_sleep = sleep.compute_sleep_metrics(zepp_sleep_df)
    df_activity = activity.compute_activity_metrics(apple_activity_df) if not apple_activity_df.empty else pd.DataFrame()
    df_activity_stage_var = activity.compute_activity_stage_variance(zepp_activity_stage_df)

    # Step 4: Extract Tier 2 features
    logger.info("Extracting Tier 2 features...")
    df_circadian = circadian.compute_circadian_metrics(zepp_activity_minute_df, zepp_sleep_df)
    df_sleep_var = circadian.compute_sleep_timing_variability(zepp_sleep_df)

    # Step 5: Extract Tier X features (cross-device validation)
    logger.info("Extracting Tier X features...")
    df_cross_device = validators.validate_cross_device(apple_hr_df, zepp_hr_auto_df)

    # Step 6: Apply segmentation to all
    logger.info("Applying segmentation...")
    all_dfs = [df_hrv, df_sleep, df_activity, df_activity_stage_var, df_circadian, df_sleep_var, df_cross_device]
    segmented_dfs = []
    for df in all_dfs:
        if not df.empty:
            df = segmentation.assign_segmentation(df, date_col="date", inplace=False)
            segmented_dfs.append(df)

    # Step 7: Merge all features
    logger.info("Merging all features...")
    if not segmented_dfs:
        logger.warning("No biomarker data available")
        return pd.DataFrame()

    df_merged = segmented_dfs[0]
    for df_next in segmented_dfs[1:]:
        df_merged = df_merged.merge(df_next, on=["date", "segment"], how="outer", validate="1:1")

    # Remove duplicates
    df_merged = df_merged.drop_duplicates(subset=["date"]).sort_values("date").reset_index(drop=True)

    # Step 8: Generate data quality flags
    logger.info("Generating data quality flags...")
    df_quality = validators.get_data_quality_flags(df_merged)
    df_merged = df_merged.merge(df_quality, on="date", how="left")

    logger.info(f"Aggregation complete: {len(df_merged)} daily records with {len(df_merged.columns)} features")

    return df_merged


def _load_csv(path: Optional[str]) -> pd.DataFrame:
    """Load CSV file safely."""
    if not path:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except FileNotFoundError:
        logger.warning(f"File not found: {path}")
        return pd.DataFrame()
    except Exception as e:
        logger.error(f"Error loading {path}: {e}")
        return pd.DataFrame()


def _apply_date_filter(df: pd.DataFrame, cutoff_date: pd.Timestamp, date_col: str = "date") -> pd.DataFrame:
    """Filter DataFrame by date."""
    if df.empty or date_col not in df.columns:
        return df

    if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
        df[date_col] = pd.to_datetime(df[date_col])

    return df[df[date_col] <= cutoff_date].copy()
