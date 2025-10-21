#!/usr/bin/env python3
from __future__ import annotations

"""
Generate heuristic state-of-mind labels from daily features.

Saves:
 - state_of_mind_synthetic.csv
 - labels_manifest.json

Usage:
 python etl_tools/generate_heuristic_labels.py --participant P000001 --snapshot 2025-09-29

"""
import argparse
import json
import hashlib
import os
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# ---------------- atomic writers ----------------

def _write_atomic_csv(df: pd.DataFrame, out_path: str | os.PathLike[str]):
    d = os.path.dirname(str(out_path)) or "."
    fd, tmp = tempfile.mkstemp(prefix=".tmp_", dir=d, suffix=".csv")
    os.close(fd)
    try:
        df.to_csv(tmp, index=False)
        os.replace(tmp, out_path)
    finally:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except Exception:
                pass


def _write_atomic_json(obj: dict, out_path: str | os.PathLike[str]):
    d = os.path.dirname(str(out_path)) or "."
    fd, tmp = tempfile.mkstemp(prefix=".tmp_", dir=d, suffix=".json")
    os.close(fd)
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
        os.replace(tmp, out_path)
    finally:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except Exception:
                pass


def _sha256_file(path: str | os.PathLike[str]) -> Optional[str]:
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return None


# ---------------- fuzzy column finder ----------------

def _find_col(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    cols = {c.lower(): c for c in df.columns}
    for cand in candidates:
        cand_l = cand.lower()
        # 1) exact-ish
        for k, orig in cols.items():
            if k == cand_l:
                return orig
    # 2) substring match with priority order
    for cand in candidates:
        cand_l = cand.lower()
        for k, orig in cols.items():
            if cand_l in k:
                return orig
    return None


def _safe_z(series: pd.Series) -> pd.Series:
    # return z-score series, filling NaN with 0 (neutral contribution)
    try:
        s = pd.to_numeric(series, errors="coerce")
    except Exception:
        s = series.astype(float, errors="ignore")
    if s.isna().all():
        return pd.Series(0.0, index=series.index)
    mean = s.mean(skipna=True)
    std = s.std(skipna=True)
    if std == 0 or np.isnan(std):
        return pd.Series(0.0, index=series.index)
    z = (s - mean) / std
    z = z.fillna(0.0)
    return z


def generate_labels(snapshot_dir: Path) -> dict:
    in_path = snapshot_dir / "features_daily_updated.csv"
    if not in_path.exists():
        raise FileNotFoundError(f"Input not found: {in_path}")

    df = pd.read_csv(in_path, dtype={"date": "string"})
    if df.empty:
        raise ValueError(f"Input file is empty: {in_path}")

    # find candidate columns (expanded to include zepp_ prefixes and common variants)
    hrv_candidates = [
        "rmssd", "sdnn", "hrv", "hrv_sdnn", "hrv_rmssd", "hrv_ms_mean", "hrv_ms"
    ]
    resthr_candidates = [
        "resting_hr", "rest_hr", "resthr", "restingheart", "resting heart", "hr_rest", "hr_mean", "rest", "hr_mean_all"
    ]
    sleep_candidates = [
        "sleep_efficiency", "sleep_eff", "sleep_minutes", "sleep", "zepp_sleep_minutes", "sleep_quality", "sleep_total_minutes"
    ]
    screen_candidates = [
        "screen_time", "screen_time_min", "screen_min", "screen", "usage_minutes", "screenminutes", "screen_time_minutes"
    ]
    steps_candidates = ["steps", "step_count", "zepp_steps", "daily_steps"]

    hrv_col = _find_col(df, hrv_candidates)
    resthr_col = _find_col(df, resthr_candidates)
    sleep_col = _find_col(df, sleep_candidates)
    screen_col = _find_col(df, screen_candidates)
    steps_col = _find_col(df, steps_candidates)

    # prepare series (default to zeros when missing)
    if hrv_col:
        z_hrv = _safe_z(df[hrv_col])
    else:
        z_hrv = pd.Series(0.0, index=df.index)

    if sleep_col:
        z_sleep = _safe_z(df[sleep_col])
    else:
        z_sleep = pd.Series(0.0, index=df.index)

    if screen_col:
        z_screen = _safe_z(df[screen_col])
    else:
        z_screen = pd.Series(0.0, index=df.index)

    if resthr_col:
        z_resthr = _safe_z(df[resthr_col])
    else:
        z_resthr = pd.Series(0.0, index=df.index)

    # compute raw heuristic score
    # score = z(HRV) + z(Sleep) - z(ScreenTime) - z(RestHR)
    score_raw = z_hrv + z_sleep - z_screen - z_resthr

    # normalize via tanh for bounded range
    score_norm = np.tanh(score_raw)

    # assign labels
    labels = []
    for v in score_norm:
        if v > 0.5:
            labels.append("positive")
        elif v < -0.5:
            labels.append("negative")
        else:
            labels.append("neutral")

    out_df = pd.DataFrame({"date": df.get("date", pd.Series(df.index.astype(str)).astype(str)),
                           "score": score_norm,
                           "label": labels})

    # AGGREGATE: if multiple rows per date exist (segments), aggregate by date
    # compute mean score per date and take majority label
    try:
        out_df["date"] = out_df["date"].astype(str)
        grouped = out_df.groupby("date")
        agg_score = grouped["score"].mean().reset_index()
        # majority label per day
        maj_label = grouped["label"].agg(lambda s: s.value_counts().idxmax()).reset_index()
        agg = agg_score.merge(maj_label, on="date")
        agg = agg.rename(columns={"label": "label_majority", "score": "score"})
        # keep columns: date, score, label (majority)
        out_df = pd.DataFrame({"date": agg["date"], "score": agg["score"], "label": agg["label_majority"]})
    except Exception:
        # fallback: keep original out_df
        pass

    # write outputs
    out_csv = snapshot_dir / "state_of_mind_synthetic.csv"
    _write_atomic_csv(out_df, out_csv)

    manifest = {
        "type": "labels_heuristic",
        "input": str(in_path),
        "output": str(out_csv),
        "rows": int(len(out_df)),
        "date_min": str(out_df['date'].min()) if 'date' in out_df else None,
        "date_max": str(out_df['date'].max()) if 'date' in out_df else None,
        "counts": dict(pd.Series(labels).value_counts(dropna=False).to_dict()),
        "columns_used": {
            "hrv": hrv_col,
            "resting_hr": resthr_col,
            "sleep": sleep_col,
            "screen_time": screen_col,
            "steps": steps_col,
        },
        "sha256": _sha256_file(out_csv),
    }

    manifest_path = snapshot_dir / "labels_manifest.json"
    _write_atomic_json(manifest, manifest_path)

    return {"csv": str(out_csv), "manifest": str(manifest_path), "manifest_obj": manifest}


def parse_args():
    ap = argparse.ArgumentParser(description="Generate heuristic state-of-mind labels from features_daily_updated.csv")
    ap.add_argument("--participant", required=True)
    ap.add_argument("--snapshot", required=True)
    return ap.parse_args()


def main():
    args = parse_args()
    pid = args.participant
    snap = args.snapshot
    snap_dir = Path("data_ai") / pid / "snapshots" / snap
    if not snap_dir.exists():
        print(f"\u26a0\ufe0f Snapshot dir not found: {snap_dir}")
        return 2

    try:
        res = generate_labels(snap_dir)
    except Exception as e:
        print(f"\u274c Error: {e}")
        return 1

    print(f"\u2705 state_of_mind_synthetic.csv generated for {pid}/{snap}: {res['csv']}")
    print(f"\u2705 labels_manifest.json written: {res['manifest']}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
