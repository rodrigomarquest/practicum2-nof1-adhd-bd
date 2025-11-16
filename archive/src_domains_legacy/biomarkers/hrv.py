"""
HRV (Heart Rate Variability) extraction - Tier 1 biomarker.

Metrics:
- SDNN: Standard deviation of NN intervals (ms)
- RMSSD: Root mean square of successive NN interval differences
- pNN50: Percentage of NN intervals > 50ms different from previous
- CV: Coefficient of variation (std/mean)

Sources:
- Zepp HR_AUTO (430K intra-diurnal records) - primary for SDNN calculation
- Apple HRV (19 SDNN records) - supplementary validation

Clinical thresholds:
- SDNN < 50ms: abnormal HRV (risk factor)
- SDNN > 100ms: healthy HRV
- Low SDNN associated with ADHD, anxiety
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


def _compute_daily_hrv_record(date, hrs: np.ndarray) -> Dict:
    """Compute daily HRV record from HR array."""
    return {
        "date": pd.Timestamp(date),
        "zepp_hrv_sdnn_ms": compute_sdnn(hrs),
        "zepp_hrv_rmssd_ms": compute_rmssd(hrs),
        "zepp_hrv_pnn50_pct": compute_pnn50(hrs),
        "zepp_hr_cv_pct": compute_hr_cv(hrs),
        "zepp_hr_mean": float(np.mean(hrs)),
        "zepp_hr_std": float(np.std(hrs, ddof=1)),
        "zepp_hr_min": float(np.min(hrs)),
        "zepp_hr_max": float(np.max(hrs)),
        "zepp_hr_samples": int(len(hrs)),
    }


def compute_nn_intervals(heart_rates: np.ndarray) -> np.ndarray:
    """
    Convert heart rate samples to NN intervals (RR intervals in ms).

    Parameters
    ----------
    heart_rates : np.ndarray
        Array of heart rate values (bpm)

    Returns
    -------
    np.ndarray
        NN intervals in milliseconds
    """
    if len(heart_rates) < 2:
        return np.array([])

    # Assume uniform sampling (e.g., every second or from equidistant timestamps)
    # NN interval (ms) = 60000 / HR (bpm)
    nn_intervals = (60000.0 / heart_rates).astype(np.float64)
    return nn_intervals


def compute_sdnn(heart_rates: np.ndarray) -> Optional[float]:
    """
    Compute SDNN (standard deviation of NN intervals).

    Parameters
    ----------
    heart_rates : np.ndarray
        Array of HR samples (bpm)

    Returns
    -------
    float or None
        SDNN value in ms, or None if insufficient data
    """
    if len(heart_rates) < 10:
        return None

    nn_intervals = compute_nn_intervals(heart_rates)
    if len(nn_intervals) < 10:
        return None

    sdnn = np.std(nn_intervals, ddof=1)
    return float(sdnn)


def compute_rmssd(heart_rates: np.ndarray) -> Optional[float]:
    """
    Compute RMSSD (root mean square of successive NN interval differences).

    Parameters
    ----------
    heart_rates : np.ndarray
        Array of HR samples (bpm)

    Returns
    -------
    float or None
        RMSSD value in ms, or None if insufficient data
    """
    if len(heart_rates) < 10:
        return None

    nn_intervals = compute_nn_intervals(heart_rates)
    if len(nn_intervals) < 2:
        return None

    # Successive differences
    successive_diffs = np.diff(nn_intervals)

    # Root mean square
    rmssd = np.sqrt(np.mean(successive_diffs**2))
    return float(rmssd)


def compute_pnn50(heart_rates: np.ndarray) -> Optional[float]:
    """
    Compute pNN50 (percentage of successive NN intervals differing > 50ms).

    Parameters
    ----------
    heart_rates : np.ndarray
        Array of HR samples (bpm)

    Returns
    -------
    float or None
        pNN50 percentage (0-100), or None if insufficient data
    """
    if len(heart_rates) < 10:
        return None

    nn_intervals = compute_nn_intervals(heart_rates)
    if len(nn_intervals) < 2:
        return None

    successive_diffs = np.abs(np.diff(nn_intervals))
    pnn50 = 100.0 * np.sum(successive_diffs > 50) / len(successive_diffs)
    return float(pnn50)


def compute_hr_cv(heart_rates: np.ndarray) -> Optional[float]:
    """
    Compute coefficient of variation of heart rate (proxy for HRV).

    Parameters
    ----------
    heart_rates : np.ndarray
        Array of HR samples (bpm)

    Returns
    -------
    float or None
        CV as percentage, or None if insufficient data
    """
    if len(heart_rates) < 5:
        return None

    mean_hr = np.mean(heart_rates)
    if mean_hr == 0:
        return None

    std_hr = np.std(heart_rates, ddof=1)
    cv = 100.0 * std_hr / mean_hr
    return float(cv)


def compute_hrv_daily(
    zepp_hr_df: pd.DataFrame,
    apple_hrv_df: Optional[pd.DataFrame] = None,
    date_col: str = "date",
    hr_col: str = "heart_rate",
    timestamp_col: str = "timestamp",
) -> pd.DataFrame:
    """
    Compute daily HRV metrics from Zepp HR_AUTO and Apple HRV data.

    Parameters
    ----------
    zepp_hr_df : pd.DataFrame
        Zepp HR_AUTO table with columns: [date, timestamp, heart_rate]
    apple_hrv_df : pd.DataFrame, optional
        Apple HRV table with columns: [date, sdnn_value]
    date_col : str
        Name of date column
    hr_col : str
        Name of heart rate column in Zepp data
    timestamp_col : str
        Name of timestamp column

    Returns
    -------
    pd.DataFrame
        Daily HRV metrics with zepp and apple columns
    """
    df_daily = _compute_zepp_hrv_daily(zepp_hr_df, date_col, hr_col)
    df_daily = _merge_apple_hrv(df_daily, apple_hrv_df)
    return df_daily


def _compute_zepp_hrv_daily(
    zepp_hr_df: pd.DataFrame,
    date_col: str,
    hr_col: str,
) -> pd.DataFrame:
    """Compute daily HRV from Zepp data."""
    if zepp_hr_df.empty:
        logger.warning("Zepp HR DataFrame is empty")
        return pd.DataFrame()

    if date_col not in zepp_hr_df.columns or hr_col not in zepp_hr_df.columns:
        raise ValueError(f"Missing required columns: {date_col}, {hr_col}")

    zepp_hr_df = zepp_hr_df.copy()
    if not pd.api.types.is_datetime64_any_dtype(zepp_hr_df[date_col]):
        zepp_hr_df[date_col] = pd.to_datetime(zepp_hr_df[date_col])

    daily_hrv = []
    for date, group in zepp_hr_df.groupby(date_col):
        hrs = np.asarray(group[hr_col].values, dtype=np.float64)
        valid_mask = (hrs >= 30) & (hrs <= 220)
        hrs_filtered = hrs[valid_mask]

        if len(hrs_filtered) >= 10:
            record = _compute_daily_hrv_record(date, hrs_filtered)
            daily_hrv.append(record)
        else:
            logger.debug(f"Insufficient HR samples for {date} (n={len(hrs_filtered)})")

    return pd.DataFrame(daily_hrv) if daily_hrv else pd.DataFrame()


def _merge_apple_hrv(
    df_daily: pd.DataFrame,
    apple_hrv_df: Optional[pd.DataFrame],
) -> pd.DataFrame:
    """Merge Apple HRV data if available."""
    if apple_hrv_df is None or apple_hrv_df.empty:
        df_daily["hrv_data_quality"] = "zepp_only"
        return df_daily

    if "date" not in apple_hrv_df.columns or "sdnn_value" not in apple_hrv_df.columns:
        df_daily["hrv_data_quality"] = "zepp_only"
        return df_daily

    apple_hrv_df = apple_hrv_df.copy()
    if not pd.api.types.is_datetime64_any_dtype(apple_hrv_df["date"]):
        apple_hrv_df["date"] = pd.to_datetime(apple_hrv_df["date"])

    apple_hrv_df = apple_hrv_df[["date", "sdnn_value"]].rename(
        columns={"sdnn_value": "apple_hrv_sdnn_ms"}
    )
    df_daily = df_daily.merge(apple_hrv_df, on="date", how="left", validate="1:1")
    df_daily["hrv_data_quality"] = df_daily["apple_hrv_sdnn_ms"].isna().apply(
        lambda x: "zepp_only" if x else "both_available"
    )

    logger.info(f"Computed {len(df_daily)} daily HRV records from {len(df_daily)} unique dates")
    return df_daily.sort_values("date").reset_index(drop=True)


def get_hrv_clinical_assessment(
    sdnn_ms: Optional[float],
    rmssd_ms: Optional[float],
) -> Dict[str, bool]:
    """
    Get clinical assessment of HRV based on thresholds.

    Parameters
    ----------
    sdnn_ms : float or None
        SDNN in milliseconds
    rmssd_ms : float or None
        RMSSD in milliseconds

    Returns
    -------
    dict
        Clinical flags:
        - is_abnormal: SDNN < 50ms or RMSSD < 20ms
        - is_healthy: SDNN > 100ms and RMSSD > 50ms
    """
    flags = {
        "is_abnormal": False,
        "is_healthy": False,
    }

    if sdnn_ms is not None and sdnn_ms < 50:
        flags["is_abnormal"] = True
    if rmssd_ms is not None and rmssd_ms < 20:
        flags["is_abnormal"] = True

    if sdnn_ms is not None and sdnn_ms > 100 and rmssd_ms is not None and rmssd_ms > 50:
        flags["is_healthy"] = True

    return flags
