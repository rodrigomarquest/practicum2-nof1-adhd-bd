# etl_modules/cardiovascular/_join/join.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import pandas as pd
from typing import List, Optional


def _concat_clean(dfs: List[pd.DataFrame]) -> Optional[pd.DataFrame]:
    dfs = [d for d in dfs if d is not None and not d.empty]
    if not dfs:
        return None
    df = pd.concat(dfs, ignore_index=True)
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    return df


def join_hr(*dfs: pd.DataFrame) -> Optional[pd.DataFrame]:
    """
    HR join simples:
      - concatena todas as fontes disponíveis
      - agrupa por timestamp (exato) e faz média do bpm
    """
    df = _concat_clean(list(dfs))
    if df is None:
        return None
    if "bpm" not in df.columns:
        return None
    # média por timestamp
    out = (
        df.groupby("timestamp", as_index=False)["bpm"]
        .mean()
        .sort_values("timestamp")
        .reset_index(drop=True)
    )
    return out


def select_hrv(*dfs: pd.DataFrame) -> Optional[pd.DataFrame]:
    """
    HRV seleção preferencial:
      - se houver Apple, usa Apple
      - senão, usa primeira fonte não vazia
      - normaliza para ['timestamp','val','metric'] (metric='hrv_ms')
    """
    candidates = [d for d in dfs if d is not None and not d.empty]
    if not candidates:
        return None
    # preferir Apple se houver pista na coluna/índice (não sempre disponível)
    # fallback: primeira não-vazia
    df = candidates[0]
    keep = [c for c in ["timestamp", "val", "metric"] if c in df.columns]
    if "val" not in keep:
        return None
    if "metric" not in keep:
        df = df.copy()
        df["metric"] = "hrv_ms"
        keep = ["timestamp", "val", "metric"]
    out = (
        df[keep]
        .dropna(subset=["timestamp"])
        .sort_values("timestamp")
        .reset_index(drop=True)
    )
    return out
