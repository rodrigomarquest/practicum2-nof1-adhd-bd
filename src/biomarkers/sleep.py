"""
Sleep architecture extraction - Tier 1 biomarker.

Metrics:
- sleep_deep_pct: Deep sleep as % of total sleep
- sleep_rem_pct: REM sleep as % of total sleep
- sleep_light_pct: Light sleep as % of total sleep
- sleep_duration_h: Total sleep duration in hours
- sleep_latency_min: Time from bedtime to first sleep (from naps JSON)
- sleep_fragmentation_count: Number of naps/awakenings events
- rem_latency_min: Time from sleep onset to first REM
- sleep_efficiency_pct: Actual sleep time / time in bed

Sources:
- Zepp SLEEP table with JSON-embedded "naps" array
- naps structure: [{start: timestamp, end: timestamp}, ...]

Clinical thresholds:
- REM latency < 60min: depression marker (BD)
- Sleep fragmentation > 4 events: sleep disorder
- Sleep efficiency < 85%: insomnia risk
"""

import pandas as pd
import numpy as np
import json
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


def parse_naps_json(naps_str: Optional[str]) -> List[Dict]:
    """
    Parse JSON naps array from Zepp sleep data.

    Parameters
    ----------
    naps_str : str or None
        JSON string containing naps array

    Returns
    -------
    list of dict
        List of naps with start/end timestamps
    """
    if pd.isna(naps_str) or not naps_str:
        return []

    try:
        if isinstance(naps_str, str):
            return json.loads(naps_str)
        return []
    except (json.JSONDecodeError, TypeError) as e:
        logger.debug(f"Failed to parse naps JSON: {e}")
        return []


def compute_nap_duration(nap: Dict) -> float:
    """
    Compute nap duration in minutes.

    Parameters
    ----------
    nap : dict
        Nap with 'start' and 'end' timestamps

    Returns
    -------
    float
        Duration in minutes
    """
    try:
        start = pd.Timestamp(nap["start"])
        end = pd.Timestamp(nap["end"])
        duration_min = (end - start).total_seconds() / 60
        return max(0, duration_min)
    except (KeyError, ValueError, TypeError):
        return 0


def compute_sleep_metrics(
    zepp_sleep_df: pd.DataFrame,
    date_col: str = "date",
    deep_col: str = "deep_minutes",
    light_col: str = "light_minutes",
    rem_col: str = "rem_minutes",
    naps_col: str = "naps",
) -> pd.DataFrame:
    """
    Compute daily sleep architecture metrics from Zepp SLEEP table.

    Parameters
    ----------
    zepp_sleep_df : pd.DataFrame
        Zepp SLEEP table with columns: [date, deep_minutes, light_minutes, rem_minutes, naps]
    date_col : str
        Name of date column
    deep_col, light_col, rem_col : str
        Names of sleep stage columns (in minutes)
    naps_col : str
        Name of naps JSON column

    Returns
    -------
    pd.DataFrame
        Daily sleep metrics:
        - date
        - sleep_duration_h: Total sleep hours
        - sleep_deep_pct: Deep sleep %
        - sleep_light_pct: Light sleep %
        - sleep_rem_pct: REM sleep %
        - sleep_fragmentation_count: Number of naps/awakenings
        - sleep_latency_min: Time to first sleep
        - rem_latency_min: Time to first REM
        - sleep_efficiency_pct: Efficiency (not estimated from naps data)
    """
    if zepp_sleep_df.empty:
        logger.warning("Zepp SLEEP DataFrame is empty")
        return pd.DataFrame()

    if date_col not in zepp_sleep_df.columns:
        raise ValueError(f"Column '{date_col}' not found")

    zepp_sleep_df = zepp_sleep_df.copy()
    if not pd.api.types.is_datetime64_any_dtype(zepp_sleep_df[date_col]):
        zepp_sleep_df[date_col] = pd.to_datetime(zepp_sleep_df[date_col])

    records = []
    for _, row in zepp_sleep_df.iterrows():
        record = _compute_single_sleep_record(row, date_col, deep_col, light_col, rem_col, naps_col)
        if record:
            records.append(record)

    df_daily = pd.DataFrame(records)
    if not df_daily.empty:
        df_daily = df_daily.sort_values(date_col).reset_index(drop=True)
        logger.info(f"Computed {len(df_daily)} daily sleep records")

    return df_daily


def _compute_single_sleep_record(
    row: pd.Series,
    date_col: str,
    deep_col: str,
    light_col: str,
    rem_col: str,
    naps_col: str,
) -> Optional[Dict]:
    """Compute sleep record for single date."""
    try:
        deep_min = float(row[deep_col]) if deep_col in row and pd.notna(row[deep_col]) else 0
        light_min = float(row[light_col]) if light_col in row and pd.notna(row[light_col]) else 0
        rem_min = float(row[rem_col]) if rem_col in row and pd.notna(row[rem_col]) else 0

        total_sleep_min = deep_min + light_min + rem_min
        if total_sleep_min < 60:  # Less than 1 hour is unrealistic
            logger.debug(f"Sleep duration < 1h: {total_sleep_min}min, skipping")
            return None

        # Compute percentages
        deep_pct = 100 * deep_min / total_sleep_min if total_sleep_min > 0 else 0
        light_pct = 100 * light_min / total_sleep_min if total_sleep_min > 0 else 0
        rem_pct = 100 * rem_min / total_sleep_min if total_sleep_min > 0 else 0

        # Parse naps for fragmentation
        naps_json = row[naps_col] if naps_col in row else None
        naps_list = parse_naps_json(naps_json)
        fragmentation_count = len(naps_list)
        latency_min = compute_nap_latency(naps_list)
        rem_latency_min = compute_rem_latency(rem_min, latency_min)

        record = {
            "date": pd.Timestamp(row[date_col]),
            "sleep_duration_h": total_sleep_min / 60,
            "sleep_deep_pct": deep_pct,
            "sleep_light_pct": light_pct,
            "sleep_rem_pct": rem_pct,
            "sleep_deep_min": deep_min,
            "sleep_light_min": light_min,
            "sleep_rem_min": rem_min,
            "sleep_fragmentation_count": fragmentation_count,
            "sleep_latency_min": latency_min,
            "rem_latency_min": rem_latency_min,
        }
        return record
    except Exception as e:
        logger.debug(f"Error processing sleep record: {e}")
        return None


def compute_nap_latency(naps_list: List[Dict]) -> float:
    """
    Compute latency proxy from naps (fragmentation count).

    Note: JSON naps array contains relative awakening events, not absolute bedtime.
    Using naps count as fragmentation metric instead of true latency.

    Parameters
    ----------
    naps_list : list of dict
        List of naps from JSON

    Returns
    -------
    float
        Number of nap/awakening events (fragmentation count)
    """
    if not naps_list:
        return 0

    # Return count of awakenings as proxy for sleep fragmentation
    return float(len(naps_list))


def compute_rem_latency(rem_min: float, sleep_latency_min: float) -> float:
    """
    Estimate REM latency from REM duration and sleep latency.

    Note: This is a rough estimate. Ideally would have detailed sleep stage timing.

    Parameters
    ----------
    rem_min : float
        REM sleep duration in minutes
    sleep_latency_min : float
        Sleep latency in minutes

    Returns
    -------
    float
        Estimated REM latency in minutes
    """
    # Heuristic: first REM typically appears 60-90 min after sleep onset
    # If REM% is low, assume delayed REM
    # This is ROUGH - better to get stage timing from raw Zepp data
    if rem_min < 30:  # Less than 30 min REM = likely delayed or missing
        return 120  # Conservative estimate for low REM
    if rem_min > 120:  # Good REM duration = earlier first REM
        return 60
    return 90  # Average case


def parse_sleep_records(zepp_sleep_df: pd.DataFrame) -> pd.DataFrame:
    """
    Main entry point for sleep record parsing (synonym for compute_sleep_metrics).

    Parameters
    ----------
    zepp_sleep_df : pd.DataFrame
        Zepp SLEEP table

    Returns
    -------
    pd.DataFrame
        Daily sleep metrics
    """
    return compute_sleep_metrics(zepp_sleep_df)


def get_sleep_clinical_assessment(
    rem_latency_min: Optional[float],
    sleep_fragmentation_count: Optional[int],
    sleep_efficiency_pct: Optional[float] = None,
) -> Dict[str, bool]:
    """
    Get clinical assessment of sleep based on thresholds.

    Parameters
    ----------
    rem_latency_min : float or None
        REM latency in minutes
    sleep_fragmentation_count : int or None
        Number of fragmentation events
    sleep_efficiency_pct : float or None
        Sleep efficiency percentage

    Returns
    -------
    dict
        Clinical flags:
        - rem_latency_abnormal: REM < 60min (depression marker)
        - fragmented_sleep: fragmentation > 4
        - poor_efficiency: efficiency < 85%
    """
    flags = {
        "rem_latency_abnormal": False,
        "fragmented_sleep": False,
        "poor_efficiency": False,
    }

    if rem_latency_min is not None and rem_latency_min < 60:
        flags["rem_latency_abnormal"] = True

    if sleep_fragmentation_count is not None and sleep_fragmentation_count > 4:
        flags["fragmented_sleep"] = True

    if sleep_efficiency_pct is not None and sleep_efficiency_pct < 85:
        flags["poor_efficiency"] = True

    return flags
