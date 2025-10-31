"""Segment helpers (S1–S6)."""

import os
import pandas as pd
from .io import read_csv_if_exists


def load_segments(snapshot_dir: str):
    path = os.path.join(snapshot_dir, "version_log_enriched.csv")
    df = read_csv_if_exists(path)
    if df is None or df.empty:
        return None
    df["date"] = pd.to_datetime(df["date"]).dt.date
    return df


def attach_segment(df, segments):
    if df is None or df.empty or segments is None or segments.empty:
        if df is None:
            return df
        df = df.copy()
        df["segment_id"] = pd.NA
        return df
    return df.merge(segments[["date", "segment_id"]], on="date", how="left")
