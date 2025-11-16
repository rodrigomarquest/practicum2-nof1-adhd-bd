"""
Circadian rhythm metrics - Tier 2 biomarker.

Metrics:
- sleep_timing_deviation_h: How far bedtime deviates from typical (23h)
- nocturnal_activity_pct: Activity during 22h-06h as % of daily
- activity_peak_hour: Hour of maximum activity (12-23h preferred)
- sleep_duration_variability: Std of sleep duration across week
- activity_rhythm_consistency: Correlation of activity pattern across days

Sources:
- Zepp ACTIVITY_MINUTE (for hourly patterns)
- Zepp SLEEP (for sleep timing)

Clinical significance:
- Nocturnal activity > 20%: Mania marker (BD)
- Sleep timing irregularity: ADHD marker
- Activity peak in early morning: Depression marker (early awakening)
- High sleep variability: Sleep disorder or ADHD
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


def compute_circadian_metrics(
    activity_minute_df: pd.DataFrame,
    sleep_df: pd.DataFrame,
    date_col: str = "date",
    timestamp_col: str = "timestamp",
    intensity_col: str = "steps",
    sleep_duration_col: str = "sleep_duration_h",
) -> pd.DataFrame:
    """
    Compute circadian rhythm metrics from activity and sleep data.

    Parameters
    ----------
    activity_minute_df : pd.DataFrame
        Minute-level activity data with [date, timestamp, intensity]
    sleep_df : pd.DataFrame
        Daily sleep data with [date, sleep_duration_h]
    date_col : str
        Name of date column
    timestamp_col : str
        Name of timestamp column
    intensity_col : str
        Name of intensity/steps column
    sleep_duration_col : str
        Name of sleep duration column

    Returns
    -------
    pd.DataFrame
        Daily circadian metrics:
        - date
        - nocturnal_activity_pct: 22h-06h activity as % of total
        - activity_peak_hour: Hour of maximum activity (0-23)
        - early_morning_activity_pct: 04h-08h activity (depression marker)
    """
    if activity_minute_df.empty:
        logger.warning("Activity minute DataFrame is empty")
        return pd.DataFrame()

    required_cols = {date_col, timestamp_col, intensity_col}
    missing = required_cols - set(activity_minute_df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    activity_minute_df = activity_minute_df.copy()
    if not pd.api.types.is_datetime64_any_dtype(activity_minute_df[timestamp_col]):
        activity_minute_df[timestamp_col] = pd.to_datetime(activity_minute_df[timestamp_col])

    records = []
    for date, group in activity_minute_df.groupby(date_col):
        record = _compute_daily_circadian_record(str(date), group, timestamp_col, intensity_col)
        if record:
            records.append(record)

    df_daily = pd.DataFrame(records)
    if not df_daily.empty:
        df_daily = df_daily.sort_values(date_col).reset_index(drop=True)
        logger.info(f"Computed {len(df_daily)} daily circadian records")

    return df_daily


def _compute_daily_circadian_record(
    date_str: str,
    group: pd.DataFrame,
    timestamp_col: str,
    intensity_col: str,
) -> Optional[Dict]:
    """Compute circadian record for single day."""
    if len(group) < 60:  # Need at least 1 hour of data
        return None

    # Extract hour from timestamp
    group_local = group.copy()
    group_local["hour"] = group_local[timestamp_col].dt.hour

    # Calculate hourly activity
    hourly_activity = group_local.groupby("hour")[intensity_col].sum()

    if hourly_activity.empty:
        return None

    # Nocturnal activity: 22h-06h (covers cross-midnight sleep period)
    nocturnal_hours = list(range(22, 24)) + list(range(0, 6))
    nocturnal_activity = sum(hourly_activity.get(h, 0) for h in nocturnal_hours)
    total_activity = hourly_activity.sum()
    nocturnal_pct = 100 * nocturnal_activity / total_activity if total_activity > 0 else 0

    # Activity peak hour
    peak_hour = int(hourly_activity.idxmax()) if len(hourly_activity) > 0 else 12

    # Early morning activity (depression marker): 04h-08h
    early_morning_hours = list(range(4, 8))
    early_activity = sum(hourly_activity.get(h, 0) for h in early_morning_hours)
    early_pct = 100 * early_activity / total_activity if total_activity > 0 else 0

    return {
        "date": pd.to_datetime(date_str),
        "nocturnal_activity_pct": float(nocturnal_pct),
        "activity_peak_hour": int(peak_hour),
        "early_morning_activity_pct": float(early_pct),
        "daytime_activity_pct": 100 - float(nocturnal_pct),
        "total_activity_count": int(total_activity),
    }


def compute_sleep_timing_variability(
    sleep_df: pd.DataFrame,
    date_col: str = "date",
    duration_col: str = "sleep_duration_h",
    window_days: int = 7,
) -> pd.DataFrame:
    """
    Compute sleep timing variability over rolling window.

    Parameters
    ----------
    sleep_df : pd.DataFrame
        Daily sleep data with [date, sleep_duration_h]
    date_col : str
        Name of date column
    duration_col : str
        Name of sleep duration column
    window_days : int
        Window for computing variability (default 7 days = 1 week)

    Returns
    -------
    pd.DataFrame
        Rolling sleep variability:
        - date
        - sleep_duration_var: Variance of sleep across window
        - sleep_duration_cv: Coefficient of variation (std/mean)
    """
    if sleep_df.empty:
        return pd.DataFrame()

    if date_col not in sleep_df.columns or duration_col not in sleep_df.columns:
        raise ValueError("Missing required columns")

    sleep_df = sleep_df.copy()
    if not pd.api.types.is_datetime64_any_dtype(sleep_df[date_col]):
        sleep_df[date_col] = pd.to_datetime(sleep_df[date_col])

    sleep_df = sleep_df.sort_values(date_col).reset_index(drop=True)

    # Rolling statistics
    sleep_df["sleep_var"] = sleep_df[duration_col].rolling(window=window_days, min_periods=3).var()
    sleep_df["sleep_mean"] = sleep_df[duration_col].rolling(window=window_days, min_periods=3).mean()
    sleep_df["sleep_std"] = sleep_df[duration_col].rolling(window=window_days, min_periods=3).std()

    # Coefficient of variation
    sleep_df["sleep_cv"] = (sleep_df["sleep_std"] / sleep_df["sleep_mean"] * 100).fillna(0)

    result = sleep_df[
        [
            date_col,
            "sleep_var",
            "sleep_cv",
        ]
    ].copy()
    result.columns = [date_col, "sleep_duration_var", "sleep_duration_cv"]

    logger.info(f"Computed {len(result)} sleep timing variability records")
    return result


def get_circadian_clinical_assessment(
    nocturnal_activity_pct: Optional[float],
    early_morning_activity_pct: Optional[float],
    sleep_duration_cv: Optional[float],
) -> Dict[str, bool]:
    """
    Get clinical assessment of circadian rhythm based on thresholds.

    Parameters
    ----------
    nocturnal_activity_pct : float or None
        Percentage of activity during nocturnal hours
    early_morning_activity_pct : float or None
        Percentage of activity during 04h-08h (depression marker)
    sleep_duration_cv : float or None
        Coefficient of variation of sleep duration

    Returns
    -------
    dict
        Clinical flags:
        - nocturnal_hyperactivity: nocturnal > 20% (BD mania marker)
        - early_morning_awakening: early_morning > 20% (depression marker)
        - irregular_sleep_timing: cv > 40% (ADHD/sleep disorder marker)
    """
    flags = {
        "nocturnal_hyperactivity": False,
        "early_morning_awakening": False,
        "irregular_sleep_timing": False,
    }

    if nocturnal_activity_pct is not None and nocturnal_activity_pct > 20:
        flags["nocturnal_hyperactivity"] = True

    if early_morning_activity_pct is not None and early_morning_activity_pct > 20:
        flags["early_morning_awakening"] = True

    if sleep_duration_cv is not None and sleep_duration_cv > 40:
        flags["irregular_sleep_timing"] = True

    return flags
