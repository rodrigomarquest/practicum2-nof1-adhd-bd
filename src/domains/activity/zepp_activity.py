"""Load Zepp activity tables and produce per-day summarized dataframe.

This module implements `load_zepp_activity_daily(tables, home_tz)` which
accepts the mapping produced by `discover_zepp_tables()` and returns a
pandas.DataFrame with columns:
  ['date','zepp_steps','zepp_distance_m','zepp_active_kcal','zepp_exercise_min','zepp_stand_hours']

Only Zepp-origin columns are produced (prefixed with zepp_). Caller is
responsible for merging with Apple-derived features.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List
import logging
import pandas as pd
from zoneinfo import ZoneInfo
import numpy as np

logger = logging.getLogger("etl.activity")


ALIASES = {
    "steps": ["steps", "total_steps"],
    "distance_m": ["distance_m", "distance", "distance_km"],
    "active_kcal": ["active_kcal", "active_energy", "calories", "total_calories"],
    "ex_min": ["exercise_minutes", "workout_minutes", "active_minutes", "ex_min"],
    "stand_h": ["stand_hours", "stand_hour", "stand"],
    "date": ["date", "day", "timestamp", "time", "local_date"],
}


def _normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    # normalize column names
    df = df.rename(columns={c: c.strip().lower() for c in df.columns})
    return df


def _coerce_date_col(df: pd.DataFrame, home_tz: str) -> pd.Series:
    # find a candidate date-like column
    for c in ALIASES["date"]:
        if c in df.columns:
            try:
                ser = pd.to_datetime(df[c], errors="coerce")
                # localize naive to UTC then convert to home_tz
                # If tzinfo missing, assume UTC
                if ser.dt.tz is None:
                    ser = ser.dt.tz_localize("UTC")
                ser = ser.dt.tz_convert(home_tz)
                return ser.dt.date.astype(str)
            except Exception:
                continue
    # fallback: try index or return NA
    return pd.Series([pd.NA] * len(df))


def _find_alias(df: pd.DataFrame, keys: List[str]):
    for k in keys:
        if k in df.columns:
            return k
    return None


def _read_and_normalize(paths: List[Path]) -> pd.DataFrame:
    parts = []
    for p in paths:
        try:
            parts.append(pd.read_csv(p))
        except Exception:
            logger.info("zepp: failed to read %s; skipping", p)
    if not parts:
        return pd.DataFrame()
    df = pd.concat(parts, ignore_index=True, sort=False)
    return _normalize_cols(df)


def load_zepp_activity_daily(tables: Dict[str, List[Path]], home_tz: str, max_records: int | None = None) -> pd.DataFrame:
    """
    Uses priority:
      1) ACTIVITY/*.csv (daily aggregates)
      2) HEALTH_DATA/*.csv
      3) SPORT + ACTIVITY_MINUTE to derive minutes/kcal (best-effort)

    Args:
        tables: Dict mapping domain names to list of CSV file paths
        home_tz: User's home timezone
        max_records: Limit parsing to max_records (for testing)

    Returns DataFrame with date and zepp_* columns (may be empty).
    """
    # 1) ACTIVITY
    df = pd.DataFrame()
    if "ACTIVITY" in tables:
        df = _read_and_normalize(tables["ACTIVITY"])
        if not df.empty:
            # try to coerce date
            if "date" not in df.columns:
                df["date"] = _coerce_date_col(df, home_tz)
            out = pd.DataFrame()
            out["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date.astype(str)
            # map aliases
            steps_col = _find_alias(df, ALIASES["steps"]) or "steps"
            if steps_col in df.columns:
                out["zepp_steps"] = df[steps_col].fillna(0).astype(float)
            else:
                out["zepp_steps"] = 0.0
            dist_col = _find_alias(df, ALIASES["distance_m"]) or None
            if dist_col and dist_col in df.columns:
                if dist_col == "distance_km":
                    out["zepp_distance_m"] = df[dist_col].astype(float) * 1000.0
                else:
                    out["zepp_distance_m"] = df[dist_col].astype(float)
            else:
                out["zepp_distance_m"] = 0.0
            kcal_col = _find_alias(df, ALIASES["active_kcal"]) or None
            out["zepp_active_kcal"] = df[kcal_col].fillna(0).astype(float) if kcal_col and kcal_col in df.columns else 0.0
            ex_col = _find_alias(df, ALIASES["ex_min"]) or None
            out["zepp_exercise_min"] = df[ex_col].fillna(0).astype(float) if ex_col and ex_col in df.columns else 0.0
            stand_col = _find_alias(df, ALIASES["stand_h"]) or None
            out["zepp_stand_hours"] = df[stand_col].fillna(0).astype(float) if stand_col and stand_col in df.columns else 0.0
            
            # NEW: Additional activity metrics (graceful fallbacks if missing)
            total_cal_col = _find_alias(df, ["total_calories", "calories_total", "cal_total", "total_cal"]) or None
            out["zepp_act_cal_total"] = df[total_cal_col].fillna(0).astype(float) if total_cal_col and total_cal_col in df.columns else 0.0
            
            sedentary_col = _find_alias(df, ["sedentary_minutes", "sedentary_time", "inactive_min", "sedentary_min"]) or None
            out["zepp_act_sedentary_min"] = df[sedentary_col].fillna(0).astype(float) if sedentary_col and sedentary_col in df.columns else 0.0
            
            sport_col = _find_alias(df, ["sport_count", "workout_count", "sessions", "sport_sessions"]) or None
            out["zepp_act_sport_sessions"] = df[sport_col].fillna(0).astype(int) if sport_col and sport_col in df.columns else 0
            
            score_col = _find_alias(df, ["activity_score", "daily_score", "score", "zepp_score"]) or None
            out["zepp_act_score_daily"] = df[score_col].fillna(0).astype(float) if score_col and score_col in df.columns else 0.0
            
            # aggregate by date
            agg = {
                "zepp_steps": "sum",
                "zepp_distance_m": "sum",
                "zepp_active_kcal": "sum",
                "zepp_act_cal_total": "sum",
                "zepp_exercise_min": "sum",
                "zepp_act_sedentary_min": "sum",
                "zepp_stand_hours": "sum",
                "zepp_act_sport_sessions": "sum",
                "zepp_act_score_daily": "mean",
            }
            out = out.groupby("date").agg(agg).reset_index()
            
            # Limit to max_records if specified
            if max_records is not None and len(out) > max_records:
                out = out.iloc[:max_records]
            
            logger.info("zepp activity rows=%d", len(out))
            return out

    # 2) HEALTH_DATA
    if "HEALTH_DATA" in tables:
        df = _read_and_normalize(tables["HEALTH_DATA"])  # many vendors include steps/kcal here
        if not df.empty:
            if "date" not in df.columns:
                df["date"] = _coerce_date_col(df, home_tz)
            # attempt same mapping as above
            out = pd.DataFrame()
            out["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date.astype(str)
            steps_col = _find_alias(df, ALIASES["steps"]) or None
            out["zepp_steps"] = df[steps_col].fillna(0).astype(float) if steps_col and steps_col in df.columns else 0.0
            dist_col = _find_alias(df, ALIASES["distance_m"]) or None
            if dist_col and dist_col in df.columns:
                out["zepp_distance_m"] = df[dist_col].astype(float)
            else:
                out["zepp_distance_m"] = 0.0
            kcal_col = _find_alias(df, ALIASES["active_kcal"]) or None
            out["zepp_active_kcal"] = df[kcal_col].fillna(0).astype(float) if kcal_col and kcal_col in df.columns else 0.0
            ex_col = _find_alias(df, ALIASES["ex_min"]) or None
            out["zepp_exercise_min"] = df[ex_col].fillna(0).astype(float) if ex_col and ex_col in df.columns else 0.0
            out["zepp_stand_hours"] = 0.0
            
            # NEW: Additional activity metrics (graceful fallbacks if missing)
            total_cal_col = _find_alias(df, ["total_calories", "calories_total", "cal_total", "total_cal"]) or None
            out["zepp_act_cal_total"] = df[total_cal_col].fillna(0).astype(float) if total_cal_col and total_cal_col in df.columns else 0.0
            
            sedentary_col = _find_alias(df, ["sedentary_minutes", "sedentary_time", "inactive_min", "sedentary_min"]) or None
            out["zepp_act_sedentary_min"] = df[sedentary_col].fillna(0).astype(float) if sedentary_col and sedentary_col in df.columns else 0.0
            
            sport_col = _find_alias(df, ["sport_count", "workout_count", "sessions", "sport_sessions"]) or None
            out["zepp_act_sport_sessions"] = df[sport_col].fillna(0).astype(int) if sport_col and sport_col in df.columns else 0
            
            score_col = _find_alias(df, ["activity_score", "daily_score", "score", "zepp_score"]) or None
            out["zepp_act_score_daily"] = df[score_col].fillna(0).astype(float) if score_col and score_col in df.columns else 0.0
            
            agg = {k: "sum" if "score" not in k else "mean" for k in out.columns if k != "date"}
            out = out.groupby("date").agg(agg).reset_index()
            logger.info("zepp activity rows=%d", len(out))
            return out

    # 3) SPORT + ACTIVITY_MINUTE: best effort (not implemented in depth)
    if "SPORT" in tables or "ACTIVITY_MINUTE" in tables:
        df_sport = _read_and_normalize(tables.get("SPORT", []))
        df_min = _read_and_normalize(tables.get("ACTIVITY_MINUTE", []))
        parts = []
        if not df_sport.empty:
            parts.append(df_sport)
        if not df_min.empty:
            parts.append(df_min)
        if parts:
            df = pd.concat(parts, ignore_index=True, sort=False)
            if "date" not in df.columns:
                df["date"] = _coerce_date_col(df, home_tz)
            out = pd.DataFrame()
            out["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date.astype(str)
            out["zepp_steps"] = 0.0
            out["zepp_distance_m"] = 0.0
            out["zepp_active_kcal"] = df.get("calories", 0).fillna(0).astype(float)
            out["zepp_exercise_min"] = df.get("duration_min", 0).fillna(0).astype(float)
            out["zepp_stand_hours"] = 0.0
            agg = {k: "sum" for k in out.columns if k != "date"}
            out = out.groupby("date").agg(agg).reset_index()
            logger.info("zepp activity rows=%d", len(out))
            return out

    # nothing found
    logger.info("zepp activity rows=0")
    return pd.DataFrame()
