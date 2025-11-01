# etl_modules/cardiovascular/cardio_features.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Dict, Optional, Iterable

import numpy as np
import pandas as pd

from src.domains.common.io import joined_dir


# ------------------------ optional progress ------------------------
def _progress(iterable: Iterable, desc: str = "") -> Iterable:
    """
    Wraps an iterable with tqdm if available; otherwise returns as-is.
    """
    try:
        from tqdm import tqdm  # type: ignore

        # total may not be known; tqdm handles None
        return tqdm(iterable, desc=desc)
    except Exception:
        return iterable


# ------------------------ helpers: atomic write ------------------------
def _write_atomic_csv(df: pd.DataFrame, out_path: str | os.PathLike[str]) -> None:
    d = os.path.dirname(out_path) or "."
    fd, tmp = tempfile.mkstemp(prefix=".tmp_", dir=d, suffix=".csv")
    os.close(fd)
    try:
        df.to_csv(tmp, index=False)
        os.replace(tmp, out_path)
        print(
            f"[atomic] wrote CSV -> {out_path} ({Path(out_path).stat().st_size} bytes)"
        )
    finally:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except Exception:
                pass


# ------------------------ datetime normalização ------------------------
def _to_utc_naive(ts: pd.Series) -> pd.Series:
    """
    Converte uma série de timestamps (strings, datetime tz-aware/naive)
    para datetime64[ns] **naive** em UTC (offset removido).

    Estratégia:
      1) parse com utc=True -> tz-aware UTC
      2) remover tz -> naive
    """
    s = pd.to_datetime(ts, errors="coerce", utc=True)
    # s é tz-aware (UTC). Removemos a info de tz:
    s = s.dt.tz_convert("UTC").dt.tz_localize(None)
    return s


# ------------------------ features engineering ------------------------
def _daily_from_hr(hr: pd.DataFrame) -> Optional[pd.DataFrame]:
    """
    Agrega HR por dia com barra de progresso opcional (tqdm).
    Mantém robustez a tz-aware e tipos.
    """
    if hr is None or hr.empty:
        return None
    df = hr.copy()
    if "timestamp" not in df.columns or "bpm" not in df.columns:
        return None

    # normalizar timestamp para UTC naive
    df["timestamp"] = _to_utc_naive(df["timestamp"])
    df = df.dropna(subset=["timestamp"])

    # garantir tipo numérico de bpm
    df["bpm"] = pd.to_numeric(df["bpm"], errors="coerce")
    df = df.dropna(subset=["bpm"])

    df["date"] = df["timestamp"].dt.date
    g = df.groupby("date", sort=True)

    rows = []
    for date_key, sub in _progress(g, desc="aggregate: HR by day"):
        # sub["bpm"] é uma Series já filtrada
        bpm = sub["bpm"].to_numpy(dtype=float, copy=False)
        n = bpm.size
        # usar numpy para velocidade
        mean = float(np.mean(bpm)) if n else np.nan
        # ddof=1; se n <= 1, std=0.0
        std = float(np.std(bpm, ddof=1)) if n > 1 else 0.0
        rows.append(
            {
                "date": date_key,
                "hr_mean": mean,
                "hr_std": std,
                "hr_min": float(np.min(bpm)) if n else np.nan,
                "hr_max": float(np.max(bpm)) if n else np.nan,
                "n_hr": int(n),
            }
        )

    out = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    return out


def _daily_from_hrv(hrv: pd.DataFrame) -> Optional[pd.DataFrame]:
    """
    Agrega HRV por dia com barra de progresso opcional (tqdm).
    """
    if hrv is None or hrv.empty:
        return None
    df = hrv.copy()
    if "timestamp" not in df.columns or "val" not in df.columns:
        return None

    # normalizar timestamp para UTC naive
    df["timestamp"] = _to_utc_naive(df["timestamp"])
    df = df.dropna(subset=["timestamp"])

    # garantir tipo numérico da métrica
    df["val"] = pd.to_numeric(df["val"], errors="coerce")
    df = df.dropna(subset=["val"])

    df["date"] = df["timestamp"].dt.date
    g = df.groupby("date", sort=True)

    rows = []
    for date_key, sub in _progress(g, desc="aggregate: HRV by day"):
        v = sub["val"].to_numpy(dtype=float, copy=False)
        n = v.size
        mean = float(np.mean(v)) if n else np.nan
        std = float(np.std(v, ddof=1)) if n > 1 else 0.0
        rows.append(
            {
                "date": date_key,
                "hrv_ms_mean": mean,
                "hrv_ms_std": std,
                "n_hrv": int(n),
            }
        )

    out = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    return out


def build_cardio_features(
    hr_daily: pd.DataFrame | None, hrv_daily: pd.DataFrame | None
) -> pd.DataFrame:
    parts = []
    if hr_daily is not None and not hr_daily.empty:
        parts.append(hr_daily)
    if hrv_daily is not None and not hrv_daily.empty:
        parts.append(hrv_daily)
    if not parts:
        return pd.DataFrame(columns=["date"])
    # merge outer por 'date'
    out = parts[0]
    for p in parts[1:]:
        out = out.merge(p, on="date", how="outer")
    out = out.sort_values("date").reset_index(drop=True)
    return out


def write_cardio_outputs(
    snapshot_dir: str | os.PathLike[str], cardio_feat: pd.DataFrame
) -> Dict[str, str]:
    snapdir = Path(snapshot_dir)
    # Expect snapdir like data/etl/<PID>/<SNAPSHOT>
    # parts: ("data","etl","<PID>","<SNAPSHOT>") -> pid at -2, snap at -1
    pid_part = snapdir.parts[-2] if len(snapdir.parts) >= 2 else ""
    snap_part = snapdir.parts[-1] if len(snapdir.parts) >= 1 else ""
    jdir = joined_dir(pid_part, snap_part)

    # write cardio features into joined/
    f_cardio = jdir / "features_cardiovascular.csv"
    _write_atomic_csv(cardio_feat, f_cardio)

    # Merge into canonical features_daily.csv (create/merge in joined/)
    f_daily = jdir / "features_daily.csv"
    if f_daily.exists():
        base = (
            pd.read_csv(f_daily, parse_dates=["date"])
            if f_daily.exists()
            else pd.DataFrame(columns=["date"])
        )
        # normalize date column
        if "date" in base.columns:
            try:
                base["date"] = pd.to_datetime(base["date"], errors="coerce", utc=True)
                base["date"] = base["date"].dt.tz_convert("UTC").dt.tz_localize(None)
                base["date"] = base["date"].dt.date
            except Exception:
                pass
    else:
        base = pd.DataFrame(columns=["date"])

    upd = cardio_feat.copy()
    if "date" in upd.columns:
        try:
            upd["date"] = pd.to_datetime(upd["date"], errors="coerce").dt.date
        except Exception:
            pass

    merged = (
        base.merge(upd, on="date", how="outer")
        .sort_values("date")
        .reset_index(drop=True)
    )
    _write_atomic_csv(merged, f_daily)

    # Do NOT produce legacy alias `features_daily_updated.csv` anymore.
    # Consumers should read `joined/features_daily.csv`.
    return {"features_cardio": str(f_cardio), "features_daily": str(f_daily)}
