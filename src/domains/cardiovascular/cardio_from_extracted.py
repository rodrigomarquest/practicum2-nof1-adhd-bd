"""Simple Zepp cardio (heart-rate) per-day aggregator.

This is intentionally minimal: reads HEARTRATE / HEARTRATE_AUTO CSVs and
aggregates mean/max/count per day, returning Zepp-prefixed columns.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List
import logging
import pandas as pd

logger = logging.getLogger("etl.cardio")


def _read(paths: List[Path]) -> pd.DataFrame:
    parts = []
    for p in paths:
        try:
            parts.append(pd.read_csv(p))
        except Exception:
            logger.info("zepp: failed to read %s; skipping", p)
    if not parts:
        return pd.DataFrame()
    df = pd.concat(parts, ignore_index=True, sort=False)
    return df


def load_zepp_cardio_daily(tables: Dict[str, List[Path]], home_tz: str) -> pd.DataFrame:
    # read both HEARTRATE and HEARTRATE_AUTO
    paths = []
    if "HEARTRATE" in tables:
        paths.extend(tables["HEARTRATE"])
    if "HEARTRATE_AUTO" in tables:
        paths.extend(tables["HEARTRATE_AUTO"])
    if not paths:
        logger.info("zepp cardio rows=0")
        return pd.DataFrame()

    df = _read(paths)
    if df.empty:
        logger.info("zepp cardio rows=0")
        return pd.DataFrame()

    # try to find timestamp and hr value
    cols = {c.lower(): c for c in df.columns}
    ts_col = None
    for cand in ("timestamp", "time", "ts", "date"):
        if cand in cols:
            ts_col = cols[cand]
            break
    hr_col = None
    for cand in ("value", "heartrate", "hr", "bpm"):
        if cand in cols:
            hr_col = cols[cand]
            break

    if ts_col is None or hr_col is None:
        logger.info("zepp cardio insufficient columns: rows=0")
        return pd.DataFrame()

    try:
        ser = pd.to_datetime(df[ts_col], errors="coerce")
        if ser.dt.tz is None:
            ser = ser.dt.tz_localize("UTC")
        ser = ser.dt.tz_convert(home_tz)
        df["date"] = ser.dt.date.astype(str)
    except Exception:
        df["date"] = pd.to_datetime(df[ts_col], errors="coerce").dt.date.astype(str)

    df[hr_col] = pd.to_numeric(df[hr_col], errors="coerce")
    grouped = df.groupby("date")[hr_col].agg(["mean", "max", "count"]).reset_index()
    grouped = grouped.rename(columns={
        "mean": "zepp_hr_mean",
        "max": "zepp_hr_max",
        "count": "zepp_n_hr",
    })
    logger.info("zepp cardio rows=%d", len(grouped))
    return grouped
