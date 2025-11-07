"""
Activity metrics extraction - Tier 1 biomarker.

Metrics:
- daily_steps: Total steps per day
- daily_distance_km: Total distance in kilometers
- daily_calories: Active energy expenditure
- activity_variance: Standard deviation of activity intensity
- sedentary_blocks_count: Number of blocks > 120min without movement
- sedentary_time_pct: Percentage of time sedentary
- activity_peaks_count: Number of activity peaks (intensity > mean + std)
- activity_fragmentation: Ratio of inactive to active periods

Sources:
- Zepp ACTIVITY_STAGE (4366 intra-day events) - primary for variance/fragmentation
- Zepp ACTIVITY_MINUTE (86K minute-level records) - for detailed timing
- Apple Activity (2721 daily records) - supplementary

Clinical significance:
- High activity variance: ADHD marker (restlessness, variable engagement)
- High sedentariness: Depression marker
- Activity fragmentation: ADHD (difficulty sustaining focus)
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, List, Tuple
import logging

logger = logging.getLogger(__name__)


def compute_activity_metrics(
    activity_df: pd.DataFrame,
    date_col: str = "date",
    steps_col: str = "steps",
    distance_col: str = "distance",
    calories_col: str = "calories",
) -> pd.DataFrame:
    """
    Compute daily activity metrics from aggregated activity data.

    Parameters
    ----------
    activity_df : pd.DataFrame
        Activity data with columns: [date, steps, distance, calories]
    date_col : str
        Name of date column
    steps_col, distance_col, calories_col : str
        Names of activity columns

    Returns
    -------
    pd.DataFrame
        Daily activity metrics:
        - date
        - daily_steps
        - daily_distance_km
        - daily_calories
    """
    if activity_df.empty:
        logger.warning("Activity DataFrame is empty")
        return pd.DataFrame()

    required_cols = {date_col, steps_col, distance_col, calories_col}
    missing = required_cols - set(activity_df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    activity_df = activity_df.copy()
    if not pd.api.types.is_datetime64_any_dtype(activity_df[date_col]):
        activity_df[date_col] = pd.to_datetime(activity_df[date_col])

    records = []
    for _, row in activity_df.iterrows():
        record = {
            "date": pd.Timestamp(row[date_col]),
            "daily_steps": float(row[steps_col]) if pd.notna(row[steps_col]) else 0,
            "daily_distance_km": float(row[distance_col]) if pd.notna(row[distance_col]) else 0,
            "daily_calories": float(row[calories_col]) if pd.notna(row[calories_col]) else 0,
        }
        records.append(record)

    df_daily = pd.DataFrame(records)
    if not df_daily.empty:
        df_daily = df_daily.sort_values(date_col).reset_index(drop=True)
        logger.info(f"Computed {len(df_daily)} daily activity records")

    return df_daily


def compute_activity_stage_variance(
    activity_stage_df: pd.DataFrame,
    date_col: str = "date",
    intensity_col: str = "intensity",
    timestamp_col: str = "timestamp",
) -> pd.DataFrame:
    """
    Compute activity variance from intra-day stage changes.

    Parameters
    ----------
    activity_stage_df : pd.DataFrame
        Zepp ACTIVITY_STAGE table with columns: [date, timestamp, intensity, ...]
    date_col : str
        Name of date column
    intensity_col : str
        Name of intensity/steps column per event
    timestamp_col : str
        Name of timestamp column

    Returns
    -------
    pd.DataFrame
        Daily variance metrics:
        - date
        - activity_variance_std: Standard deviation of event intensities
        - activity_peaks_count: Number of peaks (> mean + std)
        - activity_fragmentation_ratio: Inactive/active period ratio
    """
    if activity_stage_df.empty:
        logger.warning("Activity stage DataFrame is empty")
        return pd.DataFrame()

    if date_col not in activity_stage_df.columns:
        raise ValueError(f"Column '{date_col}' not found")
    if intensity_col not in activity_stage_df.columns:
        raise ValueError(f"Column '{intensity_col}' not found")

    activity_stage_df = activity_stage_df.copy()
    if not pd.api.types.is_datetime64_any_dtype(activity_stage_df[date_col]):
        activity_stage_df[date_col] = pd.to_datetime(activity_stage_df[date_col])

    records = []
    for date, group in activity_stage_df.groupby(date_col):
        intensities = np.asarray(group[intensity_col].values, dtype=np.float64)

        if len(intensities) < 5:  # Need sufficient events for variance
            logger.debug(f"Insufficient activity events for date (n={len(intensities)})")
            continue

        record = _compute_activity_variance_record(str(date), intensities)
        records.append(record)

    df_daily = pd.DataFrame(records)
    if not df_daily.empty:
        df_daily = df_daily.sort_values(date_col).reset_index(drop=True)
        logger.info(f"Computed {len(df_daily)} daily activity variance records")

    return df_daily


def _compute_activity_variance_record(date_str: str, intensities: np.ndarray) -> Dict:
    """Compute activity variance record for single date."""
    mean_intensity = np.mean(intensities)
    std_intensity = np.std(intensities, ddof=1)

    # Activity peaks: events > mean + std
    peaks_count = np.sum(intensities > (mean_intensity + std_intensity))

    # Fragmentation: ratio of zero-intensity to non-zero events
    zero_events = np.sum(intensities == 0)
    active_events = np.sum(intensities > 0)
    fragmentation_ratio = zero_events / active_events if active_events > 0 else 0

    return {
        "date": pd.to_datetime(date_str),
        "activity_variance_std": float(std_intensity),
        "activity_intensity_mean": float(mean_intensity),
        "activity_peaks_count": int(peaks_count),
        "activity_fragmentation_ratio": float(fragmentation_ratio),
        "activity_events_count": int(len(intensities)),
    }


def compute_sedentariness_metrics(
    activity_minute_df: pd.DataFrame,
    date_col: str = "date",
    intensity_col: str = "steps",
    sedentary_threshold: int = 0,
    block_duration_min: int = 120,
) -> pd.DataFrame:
    """
    Compute sedentariness metrics from minute-level activity data.

    Parameters
    ----------
    activity_minute_df : pd.DataFrame
        Zepp ACTIVITY_MINUTE table with columns: [date, timestamp, steps, ...]
    date_col : str
        Name of date column
    intensity_col : str
        Name of intensity/steps column
    sedentary_threshold : int
        Steps threshold for sedentary minute (default 0)
    block_duration_min : int
        Duration threshold for sedentary blocks (default 120 min = 2h)

    Returns
    -------
    pd.DataFrame
        Daily sedentariness metrics:
        - date
        - sedentary_time_pct: Percentage of sedentary minutes
        - sedentary_blocks_count: Number of blocks > threshold duration
        - max_sedentary_block_min: Maximum sedentary block duration
    """
    if activity_minute_df.empty:
        logger.warning("Activity minute DataFrame is empty")
        return pd.DataFrame()

    if date_col not in activity_minute_df.columns:
        raise ValueError(f"Column '{date_col}' not found")
    if intensity_col not in activity_minute_df.columns:
        raise ValueError(f"Column '{intensity_col}' not found")

    activity_minute_df = activity_minute_df.copy()
    if not pd.api.types.is_datetime64_any_dtype(activity_minute_df[date_col]):
        activity_minute_df[date_col] = pd.to_datetime(activity_minute_df[date_col])

    records = []
    for date, group in activity_minute_df.groupby(date_col):
        intensities = np.asarray(group[intensity_col].values, dtype=np.float64)

        if len(intensities) < 60:  # Need at least 1 hour of data
            logger.debug(f"Insufficient activity minutes for date (n={len(intensities)})")
            continue

        sedentary_pct, blocks_count, max_block_min = _compute_sedentary_blocks(
            intensities, sedentary_threshold, block_duration_min
        )

        record = {
            "date": pd.to_datetime(str(date)),
            "sedentary_time_pct": float(sedentary_pct),
            "sedentary_blocks_count": int(blocks_count),
            "max_sedentary_block_min": float(max_block_min),
        }
        records.append(record)

    df_daily = pd.DataFrame(records)
    if not df_daily.empty:
        df_daily = df_daily.sort_values(date_col).reset_index(drop=True)
        logger.info(f"Computed {len(df_daily)} daily sedentariness records")

    return df_daily


def _compute_sedentary_blocks(
    intensities: np.ndarray,
    threshold: int,
    block_duration_min: int,
) -> Tuple[float, int, float]:
    """
    Compute sedentary time percentage and block statistics.

    Parameters
    ----------
    intensities : np.ndarray
        Per-minute intensity values
    threshold : int
        Sedentary threshold
    block_duration_min : int
        Duration threshold for counting as block

    Returns
    -------
    tuple
        (sedentary_pct, blocks_count, max_block_duration_min)
    """
    sedentary_mask = intensities <= threshold
    sedentary_pct = 100 * np.sum(sedentary_mask) / len(intensities)

    # Find blocks of consecutive sedentary minutes
    blocks = _find_consecutive_blocks(sedentary_mask)
    blocks_count = sum(1 for block_len in blocks if block_len >= block_duration_min)
    max_block_min = max(blocks) if blocks else 0

    return sedentary_pct, blocks_count, max_block_min


def _find_consecutive_blocks(mask: np.ndarray) -> List[int]:
    """Find lengths of consecutive True values in boolean array."""
    # Convert to int to find differences
    changes = np.diff(np.concatenate(([0], mask.astype(int), [0])))
    starts = np.nonzero(changes == 1)[0]
    ends = np.nonzero(changes == -1)[0]
    return [int(end - start) for start, end in zip(starts, ends)]


def get_activity_clinical_assessment(
    activity_variance_std: Optional[float],
    sedentary_time_pct: Optional[float],
    activity_fragmentation_ratio: Optional[float],
) -> Dict[str, bool]:
    """
    Get clinical assessment of activity based on thresholds.

    Parameters
    ----------
    activity_variance_std : float or None
        Standard deviation of activity intensity
    sedentary_time_pct : float or None
        Percentage of time sedentary
    activity_fragmentation_ratio : float or None
        Ratio of inactive to active periods

    Returns
    -------
    dict
        Clinical flags:
        - high_activity_variance: std > 80th percentile (ADHD marker)
        - high_sedentariness: pct > 80% (depression marker)
        - highly_fragmented: fragmentation > 1.5 (ADHD marker)
    """
    flags = {
        "high_activity_variance": False,
        "high_sedentariness": False,
        "highly_fragmented": False,
    }

    # These thresholds would be calibrated from population data
    if activity_variance_std is not None and activity_variance_std > 50:
        flags["high_activity_variance"] = True

    if sedentary_time_pct is not None and sedentary_time_pct > 80:
        flags["high_sedentariness"] = True

    if activity_fragmentation_ratio is not None and activity_fragmentation_ratio > 1.5:
        flags["highly_fragmented"] = True

    return flags
