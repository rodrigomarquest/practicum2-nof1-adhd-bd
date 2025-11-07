# etl_modules/cardiovascular/zepp/loader.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import glob
from pathlib import Path
from typing import Optional
import pandas as pd

from src.etl.common.adapters import CardioProvider, ProviderContext


class ZeppCardioProvider(CardioProvider):
    name = "zepp"

    def _data_dir(self) -> Optional[Path]:
        z = os.environ.get("ZEPPOVERRIDE_DIR", "").strip()
        return Path(z) if z else None

    def load_hr(self, ctx: ProviderContext) -> Optional[pd.DataFrame]:
        """
        Procura por CSVs de HR no diretório ZEPPOVERRIDE_DIR.
        Formatos aceitos:
          - zepp_daily_features.csv (com colunas 'timestamp','bpm' ou similares)
          - *heart_rate*.csv (com 'timestamp'/'bpm' ou 'hr')
        """
        zdir = self._data_dir()
        if not zdir or not zdir.exists():
            return None

        # 1) arquivo consolidado
        cands = [zdir / "zepp_daily_features.csv"]
        # 2) arquivos por padrão
        cands += [Path(p) for p in glob.glob(str(zdir / "*heart*rate*.csv"))]

        for p in cands:
            if not p.exists():
                continue
            try:
                df = pd.read_csv(p, parse_dates=["timestamp"])
            except Exception:
                try:
                    df = pd.read_csv(p)
                    # tentar parse posterior
                    for cand in ["timestamp", "time", "datetime", "date"]:
                        if cand in df.columns:
                            df[cand] = pd.to_datetime(df[cand], errors="coerce")
                            df = df.rename(columns={cand: "timestamp"})
                            break
                except Exception:
                    continue

            # normalizar bpm
            if "bpm" not in df.columns:
                for cand in ["value", "heart_rate", "hr", "BPM"]:
                    if cand in df.columns:
                        df = df.rename(columns={cand: "bpm"})
                        break
            if not {"timestamp", "bpm"}.issubset(df.columns):
                continue

            df = df[["timestamp", "bpm"]].dropna(subset=["timestamp"])
            df = df.sort_values("timestamp").reset_index(drop=True)
            return df

        return None

    def load_hrv(self, ctx: ProviderContext) -> Optional[pd.DataFrame]:
        # Muitos datasets Zepp não trazem SDNN diário no mesmo layout;
        # tentamos detectar 'sdnn'/'hrv' se existir.
        zdir = self._data_dir()
        if not zdir or not zdir.exists():
            return None

        cands = [zdir / "zepp_daily_features.csv"]
        cands += [Path(p) for p in glob.glob(str(zdir / "*hrv*.csv"))]

        for p in cands:
            if not p.exists():
                continue
            try:
                df = pd.read_csv(p, parse_dates=["timestamp"])
            except Exception:
                try:
                    df = pd.read_csv(p)
                    for cand in ["timestamp", "time", "datetime", "date"]:
                        if cand in df.columns:
                            df[cand] = pd.to_datetime(df[cand], errors="coerce")
                            df = df.rename(columns={cand: "timestamp"})
                            break
                except Exception:
                    continue

            if "val" not in df.columns:
                for cand in ["sdnn_ms", "sdnn", "hrv", "value"]:
                    if cand in df.columns:
                        df = df.rename(columns={cand: "val"})
                        break

            if not {"timestamp", "val"}.issubset(df.columns):
                continue

            df["metric"] = "hrv_ms"
            df = df[["timestamp", "val", "metric"]].dropna(subset=["timestamp"])
            df = df.sort_values("timestamp").reset_index(drop=True)
            return df

        return None
