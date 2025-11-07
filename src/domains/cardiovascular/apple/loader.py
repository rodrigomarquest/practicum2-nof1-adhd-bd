# etl_modules/cardiovascular/apple/loader.py
# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path
from typing import Optional
import pandas as pd

from src.etl.common.adapters import CardioProvider, ProviderContext


class AppleCardioProvider(CardioProvider):
    name = "apple"

    def _pm(self, ctx: ProviderContext) -> Path:
        return Path(ctx.snapshot_dir) / "per-metric"

    def load_hr(self, ctx: ProviderContext) -> Optional[pd.DataFrame]:
        """Lê per-metric/apple_heart_rate.csv → ['timestamp','bpm']"""
        p = self._pm(ctx) / "apple_heart_rate.csv"
        if not p.exists():
            return None
        df = pd.read_csv(p, parse_dates=["timestamp"])
        # normalizar variações de nome
        if "bpm" not in df.columns:
            for cand in ["value", "heart_rate", "hr", "BPM", "HeartRate"]:
                if cand in df.columns:
                    df = df.rename(columns={cand: "bpm"})
                    break
        if "timestamp" not in df.columns:
            for cand in ["time", "datetime", "date"]:
                if cand in df.columns:
                    df = df.rename(columns={cand: "timestamp"})
                    break
        if not {"timestamp", "bpm"}.issubset(df.columns):
            return None
        df = df[["timestamp", "bpm"]].dropna(subset=["timestamp"])
        df = df.sort_values("timestamp").reset_index(drop=True)
        return df

    def load_hrv(self, ctx: ProviderContext) -> Optional[pd.DataFrame]:
        """Lê per-metric/apple_hrv_sdnn.csv → ['timestamp','val','metric'] (val=ms)"""
        p = self._pm(ctx) / "apple_hrv_sdnn.csv"
        if not p.exists():
            return None
        df = pd.read_csv(p, parse_dates=["timestamp"])
        if "sdnn_ms" in df.columns and "val" not in df.columns:
            df = df.rename(columns={"sdnn_ms": "val"})
        if "val" not in df.columns:
            for cand in ["sdnn", "value", "hrv"]:
                if cand in df.columns:
                    df = df.rename(columns={cand: "val"})
                    break
        if "timestamp" not in df.columns:
            for cand in ["time", "datetime", "date"]:
                if cand in df.columns:
                    df = df.rename(columns={cand: "timestamp"})
                    break
        if not {"timestamp", "val"}.issubset(df.columns):
            return None
        df["metric"] = "hrv_ms"
        df = df[["timestamp", "val", "metric"]].dropna(subset=["timestamp"])
        df = df.sort_values("timestamp").reset_index(drop=True)
        return df
