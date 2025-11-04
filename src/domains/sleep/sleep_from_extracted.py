"""Minimal Zepp sleep daily aggregator.

Reads SLEEP/*.csv and aggregates total/deep/light/rem hours per day when
available. Columns in result are prefixed with `zepp_slp_`.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List
import logging
import pandas as pd

logger = logging.getLogger("etl.sleep")


def _read(paths: List[Path]) -> pd.DataFrame:
    parts = []
    for p in paths:
        try:
            parts.append(pd.read_csv(p))
        except Exception:
            logger.info("zepp: failed to read %s; skipping", p)
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True, sort=False)


def load_zepp_sleep_daily(tables: Dict[str, List[Path]], home_tz: str) -> pd.DataFrame:
    if "SLEEP" not in tables:
        logger.info("zepp sleep rows=0")
        return pd.DataFrame()
    df = _read(tables["SLEEP"])
    if df.empty:
        logger.info("zepp sleep rows=0")
        return pd.DataFrame()

    # normalize date column
    if "date" not in df.columns:
        for c in ("start_time", "start_date", "day", "timestamp"):
            if c in df.columns:
                try:
                    df["date"] = pd.to_datetime(df[c], errors="coerce").dt.date.astype(str)
                    break
                except Exception:
                    continue
    # common columns: total_hours, deep_hours, light_hours, rem_hours
    out = pd.DataFrame()
    out["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date.astype(str)
    out["zepp_slp_total_h"] = pd.to_numeric(df.get("total_hours", df.get("duration_h", df.get("duration", 0))), errors="coerce").fillna(0)
    out["zepp_slp_deep_h"] = pd.to_numeric(df.get("deep_hours", 0), errors="coerce").fillna(0)
    out["zepp_slp_light_h"] = pd.to_numeric(df.get("light_hours", 0), errors="coerce").fillna(0)
    out["zepp_slp_rem_h"] = pd.to_numeric(df.get("rem_hours", 0), errors="coerce").fillna(0)

    agg = {k: "sum" for k in out.columns if k != "date"}
    out = out.groupby("date").agg(agg).reset_index()
    logger.info("zepp sleep rows=%d", len(out))
    return out
