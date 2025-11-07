"""
Cross-device validation and data quality assessment.

Tier X exploratory features:
- HR correlation: Apple HR vs Zepp HR_AUTO in overlapping dates
- Device reliability per segment (S1-S6)
- Data completeness flags
- Systematic biases (e.g., Zepp reads higher HR)

Purpose:
- Validate device data replication
- Identify systematic measurement differences
- Flag periods of low data quality
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


def validate_cross_device(
    apple_hr_df: pd.DataFrame,
    zepp_hr_df: pd.DataFrame,
    date_col: str = "date",
    apple_hr_col: str = "heart_rate",
    zepp_hr_col: str = "heart_rate",
) -> pd.DataFrame:
    """
    Compare Apple HR vs Zepp HR to validate device replication.

    Parameters
    ----------
    apple_hr_df : pd.DataFrame
        Apple HR data (daily aggregated or per-record)
    zepp_hr_df : pd.DataFrame
        Zepp HR_AUTO data (per-record)
    date_col : str
        Name of date column
    apple_hr_col : str
        Name of Apple HR column
    zepp_hr_col : str
        Name of Zepp HR column

    Returns
    -------
    pd.DataFrame
        Daily cross-device validation:
        - date
        - apple_hr_mean: Mean Apple HR
        - zepp_hr_mean: Mean Zepp HR
        - hr_correlation: Pearson correlation (0-1)
        - hr_bias: Systematic difference (Zepp - Apple)
        - agreement_score: Cross-device agreement (0-100%)
    """
    if apple_hr_df.empty or zepp_hr_df.empty:
        logger.warning("One or both HR DataFrames are empty")
        return pd.DataFrame()

    # Prepare DataFrames
    apple_hr_df = apple_hr_df.copy()
    zepp_hr_df = zepp_hr_df.copy()

    if not pd.api.types.is_datetime64_any_dtype(apple_hr_df[date_col]):
        apple_hr_df[date_col] = pd.to_datetime(apple_hr_df[date_col])
    if not pd.api.types.is_datetime64_any_dtype(zepp_hr_df[date_col]):
        zepp_hr_df[date_col] = pd.to_datetime(zepp_hr_df[date_col])

    # Group and compute daily means
    apple_daily = apple_hr_df.groupby(date_col)[apple_hr_col].agg(["mean", "std", "count"])
    apple_daily.columns = ["apple_hr_mean", "apple_hr_std", "apple_hr_samples"]

    zepp_daily = zepp_hr_df.groupby(date_col)[zepp_hr_col].agg(["mean", "std", "count"])
    zepp_daily.columns = ["zepp_hr_mean", "zepp_hr_std", "zepp_hr_samples"]

    # Merge on dates
    df_validation = apple_daily.merge(zepp_daily, left_index=True, right_index=True, how="inner")
    df_validation.index.name = date_col
    df_validation = df_validation.reset_index()

    # Compute correlation per-day (using per-record comparison)
    correlations = []
    for date in df_validation[date_col]:
        apple_records = apple_hr_df[apple_hr_df[date_col] == date][apple_hr_col].values
        zepp_records = zepp_hr_df[zepp_hr_df[date_col] == date][zepp_hr_col].values

        if len(apple_records) > 1 and len(zepp_records) > 1:
            # Sample both to match sizes (use first n records for deterministic comparison)
            min_len = min(len(apple_records), len(zepp_records))
            apple_sample = apple_records[:min_len]
            zepp_sample = zepp_records[:min_len]
            corr = float(np.corrcoef(apple_sample, zepp_sample)[0, 1])
        else:
            corr = np.nan

        correlations.append(corr)

    df_validation["hr_correlation"] = correlations

    # Compute bias and agreement
    df_validation["hr_bias"] = df_validation["zepp_hr_mean"] - df_validation["apple_hr_mean"]
    df_validation["hr_diff_pct"] = (
        100 * np.abs(df_validation["hr_bias"]) / df_validation["apple_hr_mean"]
    ).fillna(0)

    # Agreement score: higher is better (0-100%)
    # Agreement = 100 - |Zepp - Apple| / Apple * 100
    df_validation["agreement_score"] = (100 - df_validation["hr_diff_pct"]).clip(0, 100)

    logger.info(f"Validated {len(df_validation)} days of cross-device HR data")
    return df_validation[[date_col, "apple_hr_mean", "zepp_hr_mean", "hr_correlation", "hr_bias", "agreement_score"]]


def get_data_quality_flags(
    df_daily: pd.DataFrame,
    date_col: str = "date",
) -> pd.DataFrame:
    """
    Generate data quality flags for each day.

    Parameters
    ----------
    df_daily : pd.DataFrame
        Daily biomarkers DataFrame (any metrics)
    date_col : str
        Name of date column

    Returns
    -------
    pd.DataFrame
        Data quality assessment:
        - date
        - missing_hrv: HRV data missing
        - missing_sleep: Sleep data missing
        - missing_activity: Activity data missing
        - data_quality_score: Overall quality (0-100%)
    """
    if df_daily.empty:
        return pd.DataFrame()

    df_daily = df_daily.copy()

    # Detect missing metrics
    df_daily["missing_hrv"] = (
        df_daily[[col for col in ["zepp_hrv_sdnn_ms", "zepp_hr_cv_pct"] if col in df_daily.columns]].isna().all(axis=1)
        if any(col in df_daily.columns for col in ["zepp_hrv_sdnn_ms", "zepp_hr_cv_pct"])
        else False
    )

    df_daily["missing_sleep"] = (
        df_daily[[col for col in ["sleep_duration_h", "sleep_rem_pct"] if col in df_daily.columns]].isna().all(axis=1)
        if any(col in df_daily.columns for col in ["sleep_duration_h", "sleep_rem_pct"])
        else False
    )

    df_daily["missing_activity"] = (
        df_daily[[col for col in ["daily_steps", "activity_variance_std"] if col in df_daily.columns]].isna().all(axis=1)
        if any(col in df_daily.columns for col in ["daily_steps", "activity_variance_std"])
        else False
    )

    # Calculate quality score
    missing_flags = [
        df_daily["missing_hrv"].astype(int),
        df_daily["missing_sleep"].astype(int),
        df_daily["missing_activity"].astype(int),
    ]
    missing_count = sum(missing_flags)
    df_daily["data_quality_score"] = 100 * (1 - missing_count / 3)
    df_daily["data_quality_score"] = df_daily["data_quality_score"].clip(0, 100)

    return df_daily[
        [
            date_col,
            "missing_hrv",
            "missing_sleep",
            "missing_activity",
            "data_quality_score",
        ]
    ]


def get_device_reliability_per_segment(
    df_daily: pd.DataFrame,
    segmentation_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute device reliability metrics per segment (S1-S6).

    Parameters
    ----------
    df_daily : pd.DataFrame
        Daily biomarkers with agreement_score column
    segmentation_df : pd.DataFrame
        Segmentation mapping with [date, segment]

    Returns
    -------
    pd.DataFrame
        Per-segment reliability:
        - segment
        - mean_agreement_score
        - data_completeness_pct
        - n_days
    """
    if df_daily.empty or segmentation_df.empty:
        return pd.DataFrame()

    # Merge with segmentation
    df_merged = df_daily.merge(segmentation_df, on="date", how="left")

    # Group by segment
    segment_stats = []
    for segment in sorted(df_merged["segment"].dropna().unique()):
        subset = df_merged[df_merged["segment"] == segment]

        if not subset.empty:
            mean_agreement = subset["agreement_score"].mean() if "agreement_score" in subset.columns else 0
            completeness = 100 * (~subset["date"].duplicated()).sum() / len(subset)

            segment_stats.append(
                {
                    "segment": segment,
                    "mean_agreement_score": float(mean_agreement),
                    "data_completeness_pct": float(completeness),
                    "n_days": int(len(subset)),
                }
            )

    return pd.DataFrame(segment_stats)
