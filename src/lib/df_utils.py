"""Dataframe helper utilities: z-score, rolling CV, and safe merges.

These are small, well-tested helpers intended for reuse across domains.
"""
from __future__ import annotations

import pandas as pd
import numpy as np
from typing import List


def zscore(s: pd.Series, eps: float = 1e-9, ddof: int = 0) -> pd.Series:
    if s is None:
        return pd.Series(dtype=float)
    s = s.astype(float)
    m = s.mean(skipna=True)
    sd = s.std(ddof=ddof, skipna=True)
    if pd.isna(sd) or sd == 0:
        return pd.Series([np.nan] * len(s), index=s.index, dtype=float)
    return (s - m) / (sd + eps)


def rolling_cv(s: pd.Series, window: int = 7, eps: float = 1e-9) -> pd.Series:
    if s is None:
        return pd.Series(dtype=float)
    s = s.astype(float)
    roll_std = s.rolling(window=window, min_periods=1).std()
    roll_mean = s.rolling(window=window, min_periods=1).mean()
    return 100.0 * roll_std / (roll_mean + eps)


def ensure_columns(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c not in out.columns:
            out[c] = pd.NA
    return out


def safe_merge_on_date(left: pd.DataFrame, right: pd.DataFrame, how: str = "left") -> pd.DataFrame:
    if left is None or right is None:
        return left.copy() if left is not None else pd.DataFrame()
    if "date" not in left.columns or "date" not in right.columns:
        raise ValueError("Both left and right must contain a 'date' column to merge on")
    L = left.copy()
    R = right.copy()
    L["date"] = L["date"].astype(str)
    R["date"] = R["date"].astype(str)
    out = pd.merge(L.sort_values("date"), R.sort_values("date"), on="date", how=how)
    return out
