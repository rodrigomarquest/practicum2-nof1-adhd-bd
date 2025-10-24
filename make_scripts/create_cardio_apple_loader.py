#!/usr/bin/env python3
"""Compatibility shim: delegate to make_scripts.apple.create_cardio_apple_loader"""
from importlib import import_module
_m = import_module('make_scripts.apple.create_cardio_apple_loader')
for _k in [k for k in dir(_m) if not k.startswith('_')]:
    globals()[_k] = getattr(_m, _k)
#!/usr/bin/env python3
"""Write etl_modules/cardiovascular/apple/loader.py with canonical content."""
from pathlib import Path

CONTENT = r'''"""
Apple cardio loader (functional minimal).
Reads per-metric Apple CSVs and maps to a common schema:
  HR  -> ["timestamp","bpm","date","source"]
  HRV -> ["timestamp","val","date","metric"] (metric="sdnn_ms")
"""
from typing import Optional
import os
import numpy as np
import pandas as pd
from etl_modules.common.io import read_csv_if_exists, to_local_dt
from etl_modules.common.adapters import ProviderContext, HRProvider, register_provider

def _date_col(series) -> pd.Series:
    s = pd.to_datetime(series)
    if hasattr(s, "dt"):
        return s.dt.date
    return s.astype("datetime64[ns]").astype("datetime64[D]")

class AppleCardio(HRProvider):
    def load_hr(self, ctx: ProviderContext) -> Optional[pd.DataFrame]:
        path = os.path.join(ctx.snapshot_dir, "per-metric", "apple_heart_rate.csv")
        df = read_csv_if_exists(path)
        if df is None or df.empty:
            return None

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
            return None
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
'''

def main():
    target = Path('etl_modules/cardiovascular/apple/loader.py')
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.read_text(encoding='utf-8') == CONTENT:
        print('SKIP:', target)
        return 0
    target.write_text(CONTENT, encoding='utf-8')
    print('WROTE:', target)
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
