# etl_modules/cardiovascular/cardio_etl.py
# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import pandas as pd

# Timer com fallback (para não quebrar se o módulo não existir)
try:
    from etl_modules.common.progress import Timer
except Exception:
    import time
    from contextlib import contextmanager

    @contextmanager
    def Timer(label: str):
        t0 = time.time()
        print(f"▶ {label} ...")
        try:
            yield
        finally:
            dt = time.time() - t0
            print(f"⏱ {label}: {dt:.2f}s [OK]")

from etl_modules.common.adapters import get_providers, ProviderContext
from etl_modules.cardiovascular._join.join import join_hr, select_hrv
from etl_modules.cardiovascular.cardio_features import (
    _daily_from_hr, _daily_from_hrv,
    build_cardio_features, write_cardio_outputs
)

# Config placeholder (mantém compat com seu config atual)
class _Cfg:
    tz: str | None = None


def _shape(df: Optional[pd.DataFrame]) -> str:
    if df is None:
        return "None"
    return f"{df.shape}"


def run_stage_cardio(input_dir: str | Path,
                     output_dir: str | Path = None,
                     cfg: _Cfg | None = None) -> Dict[str, str]:
    """
    Pipeline Cardiovascular:
      1) Carrega HR/HRV de Apple (per-metric) e, se disponível, Zepp
      2) Join simples de HR e seleção de HRV
      3) Agregações diárias e escrita atômica:
         - features_cardiovascular.csv
         - features_daily_updated.csv  (merge com features_daily.csv, se existir)
    """
    snapdir = Path(input_dir)
    if output_dir is None:
        output_dir = snapdir
    tz = getattr(cfg, "tz", None)

    with Timer(f"cardio [{snapdir}]"):
        # 1) Providers
        with Timer("load: providers & context"):
            providers = get_providers("cardio")
            ctx = ProviderContext(snapshot_dir=str(snapdir), tz=tz)

        # 2) Carregamentos
        with Timer("load: apple HR"):
            hr_apple = providers.get("apple").load_hr(ctx) if providers.get("apple") else None
            print(f"  apple HR shape: {_shape(hr_apple)}")

        with Timer("load: apple HRV"):
            hrv_apple = providers.get("apple").load_hrv(ctx) if providers.get("apple") else None
            print(f"  apple HRV shape: {_shape(hrv_apple)}")

        with Timer("load: zepp HR (opcional)"):
            hr_zepp = providers.get("zepp").load_hr(ctx) if providers.get("zepp") else None
            print(f"  zepp HR shape:  {_shape(hr_zepp)}")

        with Timer("load: zepp HRV (opcional)"):
            hrv_zepp = providers.get("zepp").load_hrv(ctx) if providers.get("zepp") else None
            print(f"  zepp HRV shape: {_shape(hrv_zepp)}")

        # 3) Join / seleção
        with Timer("join: HR average-by-timestamp"):
            hr_all = join_hr(hr_apple, hr_zepp)
            print(f"  HR joined shape: {_shape(hr_all)}")

        with Timer("select: HRV prefer Apple"):
            hrv_sel = select_hrv(hrv_apple, hrv_zepp)
            print(f"  HRV selected shape: {_shape(hrv_sel)}")

        # 4) Agregações diárias (robustas a timezone)
        with Timer("aggregate: daily HR"):
            hr_daily = _daily_from_hr(hr_all) if hr_all is not None else None
            print(f"  HR daily shape:  {_shape(hr_daily)}")

        with Timer("aggregate: daily HRV"):
            hrv_daily = _daily_from_hrv(hrv_sel) if hrv_sel is not None else None
            print(f"  HRV daily shape: {_shape(hrv_daily)}")

        with Timer("build: cardio features (outer-merge by date)"):
            cardio_feat = build_cardio_features(hr_daily, hrv_daily)
            print(f"  cardio features shape: {_shape(cardio_feat)}")

        # 5) Escritas
        with Timer("write: features_cardiovascular.csv + features_daily_updated.csv"):
            outputs = write_cardio_outputs(snapdir, cardio_feat)

    return outputs
