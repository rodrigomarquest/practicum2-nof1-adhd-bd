import pandas as pd
import numpy as np


def apply_rolling(df: pd.DataFrame, windows=[7, 14, 28]):
    """Add rolling mean/std and delta features for numeric columns (exclude label/date).
    Rolling is computed on shifted values (past only).
    """
    df2 = df.copy()
    exclude = {"label", "label_source", "label_notes", "date"}
    num_cols = [c for c in df2.select_dtypes(include=[np.number]).columns if c not in exclude]
    for w in windows:
        for c in num_cols:
            mname = f"{c}_r{w}_mean"
            sname = f"{c}_r{w}_std"
            dname = f"{c}_r{w}_delta"
            try:
                rolled_mean = df2[c].shift(1).rolling(window=w, min_periods=1).mean().fillna(0)
                rolled_std = df2[c].shift(1).rolling(window=w, min_periods=1).std().fillna(0)
                df2[mname] = rolled_mean
                df2[sname] = rolled_std
                df2[dname] = (rolled_mean - rolled_mean.shift(1)).fillna(0)
            except Exception:
                df2[mname] = 0
                df2[sname] = 0
                df2[dname] = 0
    return df2


def add_deltas(df: pd.DataFrame, windows=[7, 14, 28]):
    # for compatibility: deltas already added in apply_rolling; provide noop wrapper
    return df


def zscore_by_segment(df: pd.DataFrame, segment_col="segment_id"):
    # group-wise zscore for numeric columns; noop if segment missing
    if segment_col not in df.columns:
        return df
    df2 = df.copy()
    num_cols = df2.select_dtypes(include=[np.number]).columns.tolist()
    for name, g in df2.groupby(segment_col):
        means = g[num_cols].mean()
        stds = g[num_cols].std().replace(0, 1)
        df2.loc[g.index, num_cols] = (g[num_cols] - means) / stds
    return df2
