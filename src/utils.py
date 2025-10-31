"""Small utilities for ETL/EDA tasks.

Contains simple IO helpers and z-score utilities used by v4 refactor skeletons.
"""

from pathlib import Path
import json
import pandas as pd
from datetime import datetime


def load_csv(path):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"CSV not found: {p}")
    return pd.read_csv(p, parse_dates=True)


def write_csv(df: pd.DataFrame, path):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, index=False)
    return p


def zscore_by_segment(df: pd.DataFrame, cols, segment_col="segment_id"):
    if segment_col not in df.columns:
        return df
    out = df.copy()
    for seg, g in out.groupby(segment_col):
        means = g[cols].mean()
        stds = g[cols].std().replace(0, 1)
        out.loc[g.index, cols] = (g[cols] - means) / stds
    return out


def write_manifest(outdir: str, meta_dict: dict):
    p = Path(outdir)
    p.mkdir(parents=True, exist_ok=True)
    fname = p / f"run_manifest_{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}.json"
    fname.write_text(json.dumps(meta_dict, indent=2), encoding="utf-8")
    return fname
