#!/usr/bin/env bash
set -euo pipefail

# Regrava os 4 arquivos do domínio cardiovascular com conteúdo correto.
# Faz backup .bak.TIMESTAMP se existir.

ts="$(date +%Y%m%d%H%M%S)"
backup() { [ -f "$1" ] && cp -f "$1" "$1.bak.$ts" && echo "BACKUP: $1 -> $1.bak.$ts" || true; }

# 1) cardio/_join/join.py
mkdir -p etl_modules/cardiovascular/_join
backup etl_modules/cardiovascular/_join/join.py
cat > etl_modules/cardiovascular/_join/join.py <<'PY'
"""
Cross-platform join rules for cardio domain (functional minimal).

HR:
  - If Apple exists -> average Apple + others by timestamp (keeps alignment)
  - Else -> average across providers
HRV:
  - Prefer Apple SDNN when present
  - Else first available metric (e.g., Zepp RMSSD/SDNN)
"""
from typing import List, Optional
import pandas as pd

def _valid(dfs: List[Optional[pd.DataFrame]]) -> List[pd.DataFrame]:
    return [d for d in dfs if d is not None and not d.empty]

def join_hr(hr_sources: List[Optional[pd.DataFrame]]) -> Optional[pd.DataFrame]:
    valid = _valid(hr_sources)
    if not valid:
        return None
    if len(valid) == 1:
        df = valid[0].copy()
        return df.sort_values("timestamp")

    def is_apple(df: pd.DataFrame) -> bool:
        return "source" in df.columns and (df["source"] == "apple").any()

    has_apple = any(is_apple(df) for df in valid)

    merged = pd.concat([df[["timestamp","bpm"]] for df in valid], ignore_index=True)
    agg = merged.groupby("timestamp", as_index=False).agg({"bpm": "mean"})
    agg = agg.sort_values("timestamp")
    agg["date"] = pd.to_datetime(agg["timestamp"]).dt.date
    agg["source"] = "apple+mix" if has_apple else "mix"
    return agg

def select_hrv(hrv_sources: List[Optional[pd.DataFrame]]) -> Optional[pd.DataFrame]:
    valid = _valid(hrv_sources)
    if not valid:
        return None

    # prefer Apple SDNN
    for df in valid:
        if "metric" in df.columns and (df["metric"] == "sdnn_ms").any():
            return df.sort_values("timestamp")

    # fallback: primeira disponível
    return valid[0].sort_values("timestamp")
PY
echo "WROTE:  etl_modules/cardiovascular/_join/join.py"

# 2) cardio/apple/loader.py
mkdir -p etl_modules/cardiovascular/apple
backup etl_modules/cardiovascular/apple/loader.py
cat > etl_modules/cardiovascular/apple/loader.py <<'PY'
"""
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
PY
echo "WROTE:  etl_modules/cardiovascular/apple/loader.py"

# 3) cardio/zepp/loader.py
mkdir -p etl_modules/cardiovascular/zepp
backup etl_modules/cardiovascular/zepp/loader.py
cat > etl_modules/cardiovascular/zepp/loader.py <<'PY'
"""
Zepp cardio loader (functional minimal).
Looks for:
  - data_etl/P000001/zepp_processed/_latest/zepp_hr_daily.csv (HR)
  - data_etl/P000001/zepp_processed/_latest/zepp_daily_features.csv (HRV)
Falls back to per-metric files if you copy them to the snapshot.
"""
from typing import Optional, List
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

def _first_existing(paths: List[str]) -> Optional[str]:
    for p in paths:
        if os.path.exists(p):
            return p
    return None

class ZeppCardio(HRProvider):
    def load_hr(self, ctx: ProviderContext) -> Optional[pd.DataFrame]:
        candidates = [
            os.path.join("data_etl", "P000001", "zepp_processed", "_latest", "zepp_hr_daily.csv"),
            os.path.join(ctx.snapshot_dir, "per-metric", "zepp_hr_daily.csv"),
        ]
        path = _first_existing(candidates)
        if not path:
            return None
        df = read_csv_if_exists(path)
        if df is None or df.empty:
            return None

        cols = {c.lower(): c for c in df.columns}
        tcol = cols.get("timestamp") or cols.get("time") or cols.get("date") or list(df.columns)[0]
        hcol = cols.get("bpm") or cols.get("hr") or list(df.columns)[1]

        if tcol == "date":
            df["timestamp"] = pd.to_datetime(df["date"])
        else:
            df["timestamp"] = to_local_dt(df[tcol], ctx.tz)

        df = df.rename(columns={hcol: "bpm"})
        df = df[["timestamp", "bpm"]].dropna(subset=["timestamp"]).sort_values("timestamp")
        df.loc[(df["bpm"] < 35) | (df["bpm"] > 220), "bpm"] = np.nan
        df["date"] = _date_col(df["timestamp"])
        df["source"] = "zepp"
        return df

    def load_hrv(self, ctx: ProviderContext) -> Optional[pd.DataFrame]:
        candidates = [
            os.path.join("data_etl", "P000001", "zepp_processed", "_latest", "zepp_daily_features.csv"),
            os.path.join(ctx.snapshot_dir, "per-metric", "zepp_hrv.csv"),
        ]
        path = _first_existing(candidates)
        if not path:
            return None
        df = read_csv_if_exists(path)
        if df is None or df.empty:
            return None

        metric = None
        if "rmssd_ms" in df.columns:
            metric = "rmssd_ms"
        elif "sdnn_ms" in df.columns:
            metric = "sdnn_ms"
        else:
            return None

        if "timestamp" in df.columns:
            ts = to_local_dt(df["timestamp"], ctx.tz)
        elif "date" in df.columns:
            ts = pd.to_datetime(df["date"])
        elif "day" in df.columns:
            ts = pd.to_datetime(df["day"])
        else:
            return None

        out = pd.DataFrame({"timestamp": ts, "val": df[metric].astype(float)})
        out.loc[(out["val"] <= 0) | (out["val"] > 350), "val"] = np.nan
        out["date"] = _date_col(out["timestamp"])
        out["metric"] = metric
        return out

register_provider("cardio", "zepp", ZeppCardio())
PY
echo "WROTE:  etl_modules/cardiovascular/zepp/loader.py"

# 4) cardio/cardio_features.py
mkdir -p etl_modules/cardiovascular
backup etl_modules/cardiovascular/cardio_features.py
cat > etl_modules/cardiovascular/cardio_features.py <<'PY'
"""
Cardio feature derivation (functional minimal).
"""
import math, os
import numpy as np
import pandas as pd
from etl_modules.config import CardioCfg
from etl_modules.common.io import read_csv_if_exists

def _ensure_date_col(df: pd.DataFrame) -> pd.DataFrame:
    if "date" not in df.columns:
        df = df.copy()
        df["date"] = pd.to_datetime(df["timestamp"]).dt.date
    return df

def _split_awake_night(idx: pd.DatetimeIndex, cfg: CardioCfg) -> pd.Series:
    hours = idx.tz_localize(None).hour
    is_night = (hours >= 22) | (hours < 6)
    return pd.Series(np.where(is_night, "night", "awake"), index=idx)

def derive_hr_features(hr: pd.DataFrame | None, sleep, segments, cfg: CardioCfg) -> pd.DataFrame:
    if hr is None or hr.empty:
        return pd.DataFrame(columns=["date","segment_id"])
    df = hr.copy()
    df = _ensure_date_col(df)
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").set_index("timestamp")

    cov = df["bpm"].notna().groupby(df["date"]).mean().mul(100.0).rename("cv_hr_coverage_pct_day")
    agg_day = df.groupby("date")["bpm"].agg(
        cv_hr_mean_day="mean",
        cv_hr_median_day="median",
        cv_hr_std_day="std",
        cv_hr_min_day="min",
        cv_hr_max_day="max",
        cv_hr_iqr_day=lambda s: (np.nanpercentile(s, 75) - np.nanpercentile(s, 25))
    )
    periods = _split_awake_night(df.index, cfg)
    df["period"] = periods.values
    agg_period = df.groupby(["date","period"])["bpm"].mean().unstack("period")
    if "awake" not in agg_period.columns: agg_period["awake"] = np.nan
    if "night" not in agg_period.columns: agg_period["night"] = np.nan
    agg_period = agg_period.rename(columns={"awake":"cv_hr_mean_awake","night":"cv_hr_mean_night"})
    agg_period["cv_hr_circ_amp"] = agg_period["cv_hr_mean_awake"] - agg_period["cv_hr_mean_night"]

    out = pd.concat([agg_day, cov, agg_period], axis=1).reset_index()
    out["cv_flag_low_coverage"] = (out["cv_hr_coverage_pct_day"] < cfg.low_coverage_pct).astype(int)

    if segments is not None and not segments.empty:
        seg = segments.copy()
        seg["date"] = pd.to_datetime(seg["date"]).dt.date
        out = out.merge(seg[["date","segment_id"]], on="date", how="left")
    else:
        out["segment_id"] = pd.NA
    return out

def derive_hrv_features(hrv: pd.DataFrame | None, hr, sleep, segments, cfg: CardioCfg) -> pd.DataFrame:
    if hrv is None or hrv.empty:
        return pd.DataFrame(columns=["date","segment_id"])
    df = hrv.copy()
    df = _ensure_date_col(df)
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").set_index("timestamp")

    day_mean = df.groupby("date")["val"].mean().rename("cv_hrv_day")
    day_std  = df.groupby("date")["val"].std().rename("cv_hrv_std_day")

    periods = _split_awake_night(df.index, cfg)
    df["period"] = periods.values
    agg_period = df.groupby(["date","period"])["val"].mean().unstack("period")
    if "awake" not in agg_period.columns: agg_period["awake"] = np.nan
    if "night" not in agg_period.columns: agg_period["night"] = np.nan
    agg_period = agg_period.rename(columns={"awake":"cv_hrv_awake","night":"cv_hrv_night"})
    agg_period["cv_hrv_delta"] = agg_period["cv_hrv_night"] - agg_period["cv_hrv_awake"]

    out = pd.concat([day_mean, day_std, agg_period], axis=1).reset_index()

    metric_name = "hrv_ms"
    if "metric" in hrv.columns:
        met = hrv["metric"].dropna().astype(str)
        metric_name = met.iloc[0] if not met.empty else "hrv_ms"

    out = out.rename(columns={
        "cv_hrv_day": f"cv_{metric_name}_day",
        "cv_hrv_std_day": f"cv_{metric_name}_std_day",
        "cv_hrv_awake": f"cv_{metric_name}_awake",
        "cv_hrv_night": f"cv_{metric_name}_night",
        "cv_hrv_delta": f"cv_{metric_name}_delta",
    })

    if segments is not None and not segments.empty:
        seg = segments.copy()
        seg["date"] = pd.to_datetime(seg["date"]).dt.date
        out = out.merge(seg[["date","segment_id"]], on="date", how="left")
    else:
        out["segment_id"] = pd.NA
    return out

def derive_circadian_features(hr: pd.DataFrame | None, segments, cfg: CardioCfg) -> pd.DataFrame:
    if hr is None or hr.empty or "timestamp" not in hr.columns:
        return pd.DataFrame(columns=["date","segment_id","cv_hr_cosinor_mesor","cv_hr_cosinor_amp","cv_hr_cosinor_acrophase"])
    df = hr.copy()
    df["date"] = pd.to_datetime(df["timestamp"]).dt.date
    df = df.dropna(subset=["timestamp","bpm"]).sort_values("timestamp").set_index("timestamp")

    rows = []
    for d, g in df.groupby("date"):
        cov = g["bpm"].notna().mean() * 100.0
        if cov < 20:
            rows.append({"date": d, "cv_hr_cosinor_mesor": np.nan, "cv_hr_cosinor_amp": np.nan, "cv_hr_cosinor_acrophase": np.nan})
            continue
        idx = g.index.tz_localize(None)
        minutes = (idx.hour * 60 + idx.minute).astype(float).values
        omega = 2.0 * math.pi / (24.0 * 60.0)
        X = np.column_stack([np.ones_like(minutes), np.cos(omega * minutes), np.sin(omega * minutes)])
        y = g["bpm"].astype(float).values
        try:
            beta, *_ = np.linalg.lstsq(X, y, rcond=None)
            mesor = float(beta[0])
            amp = float((beta[1]**2 + beta[2]**2) ** 0.5)
            acroph = float(math.atan2(-beta[2], beta[1]))
        except Exception:
            mesor = amp = acroph = np.nan
        rows.append({"date": d, "cv_hr_cosinor_mesor": mesor, "cv_hr_cosinor_amp": amp, "cv_hr_cosinor_acrophase": acroph})
    out = pd.DataFrame(rows)

    if segments is not None and not segments.empty:
        seg = segments.copy()
        seg["date"] = pd.to_datetime(seg["date"]).dt.date
        out = out.merge(seg[["date","segment_id"]], on="date", how="left")
    else:
        out["segment_id"] = pd.NA
    return out

def merge_cardio_outputs(hr_feats: pd.DataFrame, hrv_feats: pd.DataFrame, circ_feats: pd.DataFrame) -> pd.DataFrame:
    def _safe(df):
        if df is None or df.empty:
            return pd.DataFrame(columns=["date","segment_id"])
        return df
    out = _safe(hr_feats)
    for df in (_safe(hrv_feats), _safe(circ_feats)):
        out = out.merge(df, on=["date","segment_id"], how="outer")
    out = out.sort_values("date")
    return out

def update_features_daily(features_cardio: pd.DataFrame, features_daily_path: str) -> str:
    out_dir = os.path.dirname(features_daily_path)
    out_path = os.path.join(out_dir, "features_daily_updated.csv")
    base = read_csv_if_exists(features_daily_path)
    if base is None or base.empty:
        features_cardio.to_csv(out_path, index=False)
        return out_path
    base = base.copy()
    if "date" in base.columns:
        base["date"] = pd.to_datetime(base["date"]).dt.date
    keys = ["date"]
    if "segment_id" in base.columns and "segment_id" in features_cardio.columns:
        keys.append("segment_id")
    merged = base.merge(features_cardio, on=keys, how="left")
    merged.to_csv(out_path, index=False)
    return out_path
PY
echo "WROTE:  etl_modules/cardiovascular/cardio_features.py"

echo "✅ Cardio files fixed."
