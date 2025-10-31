"""
Apple cardio loader (STRICT).
Aceita apenas per-metric no snapshot:
  - per-metric/apple_heart_rate.csv   (timestamp,bpm)
  - per-metric/apple_hrv_sdnn.csv     (timestamp,sdnn_ms)
"""

from typing import Optional
import os
import numpy as np
import pandas as pd
from etl_modules.common.io import read_csv_if_exists, to_local_dt
from etl_modules.common.adapters import ProviderContext, HRProvider, register_provider


def _date_col(series) -> pd.Series:
    s = pd.to_datetime(series)
    return s.dt.date if hasattr(s, "dt") else pd.to_datetime(series).dt.date


class AppleCardio(HRProvider):
    def load_hr(self, ctx: ProviderContext) -> Optional[pd.DataFrame]:
        path = os.path.join(ctx.snapshot_dir, "per-metric", "apple_heart_rate.csv")
        df = read_csv_if_exists(path)
        if df is None or df.empty:
            raise FileNotFoundError(f"Missing per-metric HR: {path}")
        cols = {c.lower(): c for c in df.columns}
        tcol = cols.get("timestamp") or cols.get("time") or list(df.columns)[0]
        hcol = cols.get("bpm") or cols.get("hr") or list(df.columns)[1]
        df = df[[tcol, hcol]].rename(columns={tcol: "timestamp", hcol: "bpm"})
        df["timestamp"] = to_local_dt(df["timestamp"], ctx.tz)
        df = df.dropna(subset=["timestamp"]).sort_values("timestamp")
        df.loc[(df["bpm"] < 35) | (df["bpm"] > 220), "bpm"] = np.nan
        df["date"] = _date_col(df["timestamp"])
        df["source"] = "apple"
        return df

    def load_hrv(self, ctx: ProviderContext) -> Optional[pd.DataFrame]:
        path = os.path.join(ctx.snapshot_dir, "per-metric", "apple_hrv_sdnn.csv")
        df = read_csv_if_exists(path)
        if df is None or df.empty:
            raise FileNotFoundError(f"Missing per-metric HRV: {path}")
        cols = {c.lower(): c for c in df.columns}
        tcol = cols.get("timestamp") or list(df.columns)[0]
        vcol = cols.get("sdnn_ms") or cols.get("value") or list(df.columns)[1]
        df = df[[tcol, vcol]].rename(columns={tcol: "timestamp", vcol: "val"})
        df["timestamp"] = to_local_dt(df["timestamp"], ctx.tz)
        df = df.dropna(subset=["timestamp"]).sort_values("timestamp")
        df.loc[(df["val"] <= 0) | (df["val"] > 350), "val"] = np.nan
        df["date"] = _date_col(df["timestamp"])
        df["metric"] = "sdnn_ms"
        return df


register_provider("cardio", "apple", AppleCardio())
