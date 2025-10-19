"""Lightweight I/O utilities (stubs)."""
import os, warnings, pandas as pd

def read_csv_if_exists(path: str, **kwargs) -> pd.DataFrame | None:
    try:
        return pd.read_csv(path, **kwargs) if os.path.exists(path) else None
    except Exception as e:
        warnings.warn(f'read_csv_if_exists falhou: {e}')
        return None

def ensure_dir(path: str): os.makedirs(path, exist_ok=True)

def to_local_dt(series, tz: str):
    s = pd.to_datetime(series, errors='coerce', utc=True)
    try:
        return s.dt.tz_convert(tz)
    except Exception:
        return s

def date_col(series): return pd.to_datetime(series).dt.date

