"""
Auto-Segmentation without version_log

Purpose:
    Generate segment_id automatically by detecting real context changes.
    No version_log_enriched.csv required.

Rules (applied in order; first match creates new segment):
    1. Source change: dominant source_cardio switches (apple↔zepp) for ≥5 consecutive days
    2. Signal change: ≥7-day sustained change in HR_mean (Δ≥8 bpm), HRV (Δ≥10 ms), or sleep_eff (Δ≥0.08)
    3. Gap detection: ≥3-day simultaneous missing (missing_cardio==1 & missing_sleep==1) → new segment on recovery
    4. Temporal fallback: force new segment every ~60 days to maintain fold compatibility

Output:
    - segment_id (int, 1-indexed): incremented when rule triggers
    - segment_autolog.csv: decision log with (date, reason, metric, old_seg, new_seg)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, List, Dict, Optional
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


def _get_dominant_source(source_window) -> Optional[str]:
    """Get dominant source from window, or None."""
    try:
        modes = pd.Series(source_window).mode()
        val = modes.iloc[0] if len(modes) > 0 else None
        return str(val) if val is not None else None
    except Exception:
        return None


def detect_source_change(df: pd.DataFrame, window: int = 5) -> np.ndarray:
    """
    Detect when source_cardio changes and persists for ≥window days.
    
    Returns array of booleans: True if this day starts a new segment due to source change.
    """
    triggers = np.zeros(len(df), dtype=bool)
    
    if "source_cardio" not in df.columns:
        return triggers
    
    source = df["source_cardio"].fillna("none").values
    
    for i in range(1, len(df) - window):
        prev_window = source[max(0, i - window):i]
        curr_window = source[i:min(len(source), i + window)]
        
        prev_dominant = _get_dominant_source(prev_window)
        curr_dominant = _get_dominant_source(curr_window)
        
        if prev_dominant and curr_dominant and prev_dominant != "none" and prev_dominant != curr_dominant:
            triggers[i] = True
    
    return triggers


def detect_signal_change(df: pd.DataFrame, window: int = 7) -> Tuple[np.ndarray, List[str]]:
    """
    Detect sustained, abrupt changes in biomarkers.
    
    Thresholds:
        - hr_mean: Δ≥8 bpm
        - hrv_rmssd: Δ≥10 ms
        - sleep_efficiency: Δ≥0.08
    
    Returns:
        (trigger_array, reason_strings)
    """
    triggers = np.zeros(len(df), dtype=bool)
    reasons = [""] * len(df)
    
    # Minimum data density required in window to compute trigger (e.g., 70% non-NaN)
    min_data_density = 0.7
    
    # HR mean change
    # NOTE (v4.1.5): Forward-fill removed for scientific integrity.
    # Triggers are only computed on windows with sufficient real data (≥70% non-NaN).
    if "hr_mean" in df.columns:
        hr = df["hr_mean"]  # Keep NaN, do not forward-fill
        for i in range(window, len(df)):
            prev_window = hr.iloc[max(0, i - window):i]
            curr_window = hr.iloc[i:min(len(hr), i + window)]
            
            # Only compute trigger if both windows have sufficient data
            prev_density = prev_window.notna().sum() / len(prev_window)
            curr_density = curr_window.notna().sum() / len(curr_window)
            
            if prev_density >= min_data_density and curr_density >= min_data_density:
                prev_mean = prev_window.mean()
                curr_mean = curr_window.mean()
                
                if not pd.isna(prev_mean) and not pd.isna(curr_mean) and abs(curr_mean - prev_mean) >= 8.0:
                    triggers[i] = True
                    reasons[i] = f"HR_mean_change(Δ={abs(curr_mean - prev_mean):.1f}bpm)"
                    break
    
    # HRV change
    if "hrv_rmssd" in df.columns and not triggers.any():
        hrv = df["hrv_rmssd"]  # Keep NaN, do not forward-fill
        for i in range(window, len(df)):
            prev_window = hrv.iloc[max(0, i - window):i]
            curr_window = hrv.iloc[i:min(len(hrv), i + window)]
            
            # Only compute trigger if both windows have sufficient data
            prev_density = prev_window.notna().sum() / len(prev_window)
            curr_density = curr_window.notna().sum() / len(curr_window)
            
            if prev_density >= min_data_density and curr_density >= min_data_density:
                prev_mean = prev_window.mean()
                curr_mean = curr_window.mean()
                
                if not pd.isna(prev_mean) and not pd.isna(curr_mean) and abs(curr_mean - prev_mean) >= 10.0:
                    triggers[i] = True
                    reasons[i] = f"HRV_change(Δ={abs(curr_mean - prev_mean):.1f}ms)"
                    break
    
    # Sleep efficiency change
    if "sleep_efficiency" in df.columns and not triggers.any():
        sleep = df["sleep_efficiency"]  # Keep NaN, do not forward-fill
        for i in range(window, len(df)):
            prev_window = sleep.iloc[max(0, i - window):i]
            curr_window = sleep.iloc[i:min(len(sleep), i + window)]
            
            # Only compute trigger if both windows have sufficient data
            prev_density = prev_window.notna().sum() / len(prev_window)
            curr_density = curr_window.notna().sum() / len(curr_window)
            
            if prev_density >= min_data_density and curr_density >= min_data_density:
                prev_mean = prev_window.mean()
                curr_mean = curr_window.mean()
                
                if not pd.isna(prev_mean) and not pd.isna(curr_mean) and abs(curr_mean - prev_mean) >= 0.08:
                    triggers[i] = True
                    reasons[i] = f"SleepEff_change(Δ={abs(curr_mean - prev_mean):.2f})"
                    break
    
    return triggers, reasons


def detect_gap_recovery(df: pd.DataFrame, min_gap: int = 3) -> np.ndarray:
    """
    Detect when signal recovers after ≥min_gap day gap (both missing_cardio & missing_sleep == 1).
    
    Returns array of booleans: True if this day starts new segment (recovery after gap).
    """
    triggers = np.zeros(len(df), dtype=bool)
    
    if "missing_cardio" not in df.columns or "missing_sleep" not in df.columns:
        return triggers
    
    missing = ((df["missing_cardio"] == 1) & (df["missing_sleep"] == 1)).values
    
    gap_start = None
    for i in range(len(missing)):
        if missing[i]:
            if gap_start is None:
                gap_start = i
        else:
            # Signal recovered
            if gap_start is not None:
                gap_length = i - gap_start
                if gap_length >= min_gap:
                    triggers[i] = True
                gap_start = None
    
    return triggers


def detect_temporal_fallback(df: pd.DataFrame, period_days: int = 60) -> np.ndarray:
    """
    Force new segment every ~period_days if no other rule triggered.
    
    Ensures fold compatibility with 4m/2m calendar CV.
    """
    triggers = np.zeros(len(df), dtype=bool)
    
    if "date" not in df.columns:
        return triggers
    
    dates = pd.to_datetime(df["date"])
    start_date = dates.min()
    
    for i, d in enumerate(dates):
        days_since_start = (d - start_date).days
        if days_since_start > 0 and days_since_start % period_days == 0:
            triggers[i] = True
    
    return triggers


def _check_temporal_trigger(date_i, last_seg_date, temporal_period: int) -> bool:
    """Check if temporal fallback should trigger."""
    if last_seg_date is None or temporal_period <= 0:
        return False
    try:
        diff_days = (pd.to_datetime(date_i) - pd.to_datetime(last_seg_date)).days
        return diff_days >= temporal_period
    except Exception:
        return False


def generate_segments(
    df: pd.DataFrame,
    source_window: int = 5,
    signal_window: int = 7,
    gap_min: int = 3,
    temporal_period: int = 60,
) -> Tuple[pd.DataFrame, List[Dict]]:
    """
    Apply segmentation rules and generate segment_id (simplified).
    
    Returns:
        (df_with_segments, decision_log)
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    
    segment_id = np.ones(len(df), dtype=int)
    decisions = []
    current_segment = 1
    last_seg_date = None
    
    for i in range(len(df)):
        date_i = df.loc[i, "date"]
        triggered = False
        
        # Rule 4: Temporal fallback (check first, lowest priority)
        if _check_temporal_trigger(date_i, last_seg_date, temporal_period):
            current_segment += 1
            decisions.append({
                "date": date_i,
                "reason": "temporal_fallback",
                "metric": f"≥{temporal_period}d",
                "old_seg": current_segment - 1,
                "new_seg": current_segment,
            })
            triggered = True
            last_seg_date = date_i
        
        # Rule 3: Gap recovery
        if not triggered and i >= gap_min:
            gap_triggers = detect_gap_recovery(df.iloc[max(0, i - gap_min):i + 1], min_gap=gap_min)
            if gap_triggers.any():
                current_segment += 1
                decisions.append({
                    "date": date_i,
                    "reason": "gap_recovery",
                    "metric": f"gap≥{gap_min}d",
                    "old_seg": current_segment - 1,
                    "new_seg": current_segment,
                })
                triggered = True
                last_seg_date = date_i
        
        # Rule 2: Signal change
        if not triggered and i >= signal_window:
            signal_triggers, reasons = detect_signal_change(df.iloc[max(0, i - signal_window):i + signal_window])
            if signal_triggers.any():
                current_segment += 1
                metric_str = reasons[signal_window] if signal_window < len(reasons) else "unknown"
                decisions.append({
                    "date": date_i,
                    "reason": "signal_change",
                    "metric": metric_str,
                    "old_seg": current_segment - 1,
                    "new_seg": current_segment,
                })
                triggered = True
                last_seg_date = date_i
        
        # Rule 1: Source change
        if not triggered and i > 0 and "source_cardio" in df.columns:
            source_prev = df.loc[max(0, i - source_window):i, "source_cardio"].mode()
            source_curr = df.loc[i:min(len(df) - 1, i + source_window), "source_cardio"].mode()
            
            if len(source_prev) > 0 and len(source_curr) > 0:
                prev_val = source_prev[0]
                curr_val = source_curr[0]
                if prev_val != curr_val and str(prev_val) != "none":
                    current_segment += 1
                    decisions.append({
                        "date": date_i,
                        "reason": "source_change",
                        "metric": f"{prev_val}→{curr_val}",
                        "old_seg": current_segment - 1,
                        "new_seg": current_segment,
                    })
                    last_seg_date = date_i
        
        segment_id[i] = current_segment
    
    df["segment_id"] = segment_id
    
    logger.info(f"✓ Generated {current_segment} segments with {len(decisions)} transitions")
    
    return df, decisions


def auto_segment(
    unified_df: pd.DataFrame,
    output_csv: Optional[Path] = None,
    autolog_csv: Optional[Path] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    End-to-end auto-segmentation pipeline.
    
    Args:
        unified_df: Input features_daily_unified.csv (with missing_*, source_* columns)
        output_csv: Where to save features_daily_with_segments.csv
        autolog_csv: Where to save segment_autolog.csv
    
    Returns:
        (df_with_segments, decision_log_df)
    """
    logger.info("=" * 80)
    logger.info("AUTO-SEGMENTATION")
    logger.info("=" * 80)
    
    df, decisions = generate_segments(unified_df)
    
    # Save outputs
    if output_csv:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_csv, index=False)
        logger.info(f"✓ Saved {len(df)} rows to {output_csv}")
    
    if autolog_csv:
        autolog_csv.parent.mkdir(parents=True, exist_ok=True)
        decision_df = pd.DataFrame(decisions)
        decision_df.to_csv(autolog_csv, index=False)
        logger.info(f"✓ Saved {len(decision_df)} decisions to {autolog_csv}")
    
    decision_df = pd.DataFrame(decisions) if decisions else pd.DataFrame()
    return df, decision_df


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    )
    
    # Test with mock data
    rng = np.random.default_rng(42)
    test_data = {
        "date": pd.date_range("2025-01-01", periods=180, freq="D"),
        "sleep_total_h": rng.normal(7, 1, 180),
        "sleep_efficiency": rng.uniform(0.7, 0.95, 180),
        "hr_mean": np.concatenate([rng.normal(70, 5, 90), rng.normal(80, 5, 90)]),
        "hrv_rmssd": rng.uniform(20, 60, 180),
        "source_cardio": np.where(np.arange(180) < 90, "apple", "zepp"),
        "missing_cardio": np.where(rng.uniform(0, 1, 180) < 0.1, 1, 0),
        "missing_sleep": np.where(rng.uniform(0, 1, 180) < 0.05, 1, 0),
    }
    test_df = pd.DataFrame(test_data)
    
    result_df, log_df = auto_segment(test_df)
    print("\nSegmentation Results:")
    print(result_df[["date", "segment_id", "source_cardio", "missing_cardio", "missing_sleep"]].head(20))
    print(f"\nDecisions: {len(log_df)}")
    print(log_df)
