#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import hashlib
import os
import tempfile
from pathlib import Path
from collections import Counter
from typing import Dict, Callable, Any

import numpy as np
import pandas as pd


def _write_atomic_csv(df: pd.DataFrame, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + '.tmp')
    df.to_csv(tmp, index=False)
    os.replace(str(tmp), str(out_path))


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def wmean(s: pd.Series, w: pd.Series) -> float:
    import numpy as _np
    s = pd.to_numeric(s, errors='coerce')
    w = pd.to_numeric(w, errors='coerce')
    valid = (~s.isna()) & (~w.isna()) & (w > 0)
    if valid.sum() == 0:
        return float('nan')
    return float((_np.average(s[valid], weights=w[valid])))


def agg_map_for(df: pd.DataFrame) -> Dict[str, Any]:
    cols = df.columns.tolist()
    m: Dict[str, Callable] = {}
    for c in cols:
        if c == 'date':
            continue
        low = c.lower()
        if low == 'segment_id':
            # handled separately
            continue
        if any(k in low for k in ['steps', 'active_energy', 'calorie', 'screen_time', 'usage_min', 'duration_min', 'minutes']):
            m[c] = 'sum'
        elif any(k in low for k in ['resting_hr', 'rhr']):
            m[c] = 'mean'
        elif any(k in low for k in ['hrv', 'rmssd', 'sdnn']):
            m[c] = 'mean'
        elif any(k in low for k in ['sleep_efficiency', 'efficiency']):
            # weighted by sleep_duration_min if present later
            m[c] = 'wmean_sleep'
        elif low.endswith('__z'):
            m[c] = 'mean'
        else:
            # default numeric -> mean; non-numeric will be excluded later
            m[c] = 'mean'
    return m


def aggregate_snapshot(snapshot_dir: Path, labels: str = 'none') -> Dict[str, str | None]:
    # common filenames
    FEATURES_UPDATED = 'features_daily_updated.csv'
    STATE_SYNTHETIC = 'state_of_mind_synthetic.csv'

    features_path = snapshot_dir / FEATURES_UPDATED
    if not features_path.exists():
        raise FileNotFoundError(f"{FEATURES_UPDATED} not found: {features_path}")

    fdf = pd.read_csv(features_path, dtype={'date': 'string'})
    if fdf.empty:
        raise ValueError('features_daily_updated.csv is empty')

    # ensure date column -> date
    fdf['date'] = pd.to_datetime(fdf['date'], errors='coerce').dt.date

    # drop exact duplicates
    fdf = fdf.drop_duplicates()

    # determine aggregation map
    amap = agg_map_for(fdf)

    # prepare weighted mean if needed
    sleep_dur_col = None
    for c in fdf.columns:
        if 'sleep_duration' in c.lower() or 'sleep_minutes' in c.lower() or 'sleep_min' in c.lower():
            sleep_dur_col = c
            break

    # numeric cols only
    # numeric_cols computed below when needed

    agg_dict = {}
    for c, rule in amap.items():
        if c not in fdf.columns:
            continue
        if not pd.api.types.is_numeric_dtype(fdf[c]):
            continue
        if rule == 'sum':
            agg_dict[c] = 'sum'
        elif rule == 'mean':
            agg_dict[c] = 'mean'
        elif rule == 'wmean_sleep':
            if sleep_dur_col:
                # placeholder - will handle separately
                agg_dict[c] = ('wmean_sleep', sleep_dur_col)
            else:
                agg_dict[c] = 'mean'
        else:
            agg_dict[c] = 'mean'

    # perform groupby aggregation manually to support weighted means and dtype handling
    grouped = []
    for d, g in fdf.groupby('date', sort=True):
        row = {'date': d}
        # segment_id mode
        if 'segment_id' in fdf.columns:
            try:
                sc = g['segment_id'].dropna()
                if not sc.empty:
                    vc = sc.value_counts()
                    mode_val = vc.idxmax()
                    # if tie, idxmax picks the first; ensure numeric coercion when possible
                    try:
                        mode_val = int(mode_val)
                    except Exception:
                        pass
                    row['segment_id'] = mode_val
                else:
                    row['segment_id'] = pd.NA
            except Exception:
                row['segment_id'] = pd.NA

        for c, rule in agg_dict.items():
            if isinstance(rule, tuple) and rule[0] == 'wmean_sleep':
                wcol = rule[1]
                try:
                    val = wmean(g[c], g[wcol])
                except Exception:
                    val = float('nan')
                row[c] = val
            else:
                try:
                    if rule == 'sum':
                        row[c] = float(g[c].sum(skipna=True))
                    else:
                        row[c] = float(g[c].mean(skipna=True))
                except Exception:
                    row[c] = float('nan')

        grouped.append(row)

    outdf = pd.DataFrame(grouped)
    # ensure deterministic order
    cols = ['date']
    if 'segment_id' in outdf.columns:
        cols.append('segment_id')
    other_cols = [c for c in sorted(outdf.columns) if c not in cols]
    cols.extend(other_cols)
    outdf = outdf[cols]

    # format date as YYYY-MM-DD strings
    outdf['date'] = outdf['date'].astype('datetime64[ns]').dt.date.astype('string')

    out_csv = snapshot_dir / 'features_daily_agg.csv'
    _write_atomic_csv(outdf.sort_values('date'), out_csv)

    labeled_csv_path = None
    label_counts = None
    if labels == 'synthetic':
        synth_path = snapshot_dir / STATE_SYNTHETIC
        if synth_path.exists():
            sdf = pd.read_csv(synth_path, dtype={'date': 'string'})
            sdf['date'] = pd.to_datetime(sdf['date'], errors='coerce').dt.date
            # normalize label columns
            cols_l = {c.lower(): c for c in sdf.columns}
            label_col = cols_l.get('label') or cols_l.get('raw_label') or cols_l.get('state')
            if not label_col:
                raise ValueError('Synthetic labels file has no label-like column')
            sdf = sdf.rename(columns={label_col: 'label'})
            if 'raw_label' not in sdf.columns:
                sdf['raw_label'] = sdf['label']
            if 'score' not in sdf.columns:
                sdf['score'] = pd.NA

            # aggregate synthetic by date (mean score, mode label)
            def _mode(x):
                try:
                    m = x.mode(dropna=True)
                    return m.iloc[0] if not m.empty else x.iloc[0]
                except Exception:
                    return x.iloc[0]

            sdfg = sdf.groupby('date', dropna=False).agg({'score': 'mean', 'label': lambda x: _mode(x), 'raw_label': lambda x: _mode(x)}).reset_index()
            sdfg['date'] = sdfg['date'].astype('datetime64[ns]').dt.date.astype('string')

            outl = outdf.merge(sdfg, on='date', how='left')
            labeled_csv_path = snapshot_dir / 'features_daily_labeled_agg.csv'
            _write_atomic_csv(outl.sort_values('date'), labeled_csv_path)

            # label counts
            try:
                label_counts = dict(outl['label'].dropna().value_counts().to_dict())
            except Exception:
                label_counts = None
        else:
            # labels requested but no synthetic file
            raise FileNotFoundError(f"Requested labels='synthetic' but not found: {snapshot_dir / STATE_SYNTHETIC}")

    # manifest
    manifest = {
        'type': 'aggregate_daily',
        'snapshot_dir': str(snapshot_dir),
        'inputs': {
            FEATURES_UPDATED: _sha256_file(features_path),
            STATE_SYNTHETIC: _sha256_file(snapshot_dir / STATE_SYNTHETIC) if (snapshot_dir / STATE_SYNTHETIC).exists() else None
        },
        'outputs': {
            'features_daily_agg.csv': _sha256_file(out_csv),
            'features_daily_labeled_agg.csv': _sha256_file(labeled_csv_path) if labeled_csv_path else None,
            'rows_total': int(len(outdf)),
            'cols_total': int(len(outdf.columns)),
            'label_counts': label_counts
        }
    }

    manifest_path = snapshot_dir / 'agg_manifest.json'
    with open(manifest_path.with_suffix('.tmp'), 'w', encoding='utf-8') as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    os.replace(str(manifest_path.with_suffix('.tmp')), str(manifest_path))

    return {'features_daily_agg': str(out_csv), 'features_daily_labeled_agg': str(labeled_csv_path) if labeled_csv_path else None, 'manifest': str(manifest_path)}


def run(snapshot_dir: Path, labels: str = "none") -> dict:
    """Aggregates to one-row-per-day; optionally merges synthetic labels; returns paths and basic stats.

    This is the public API expected by `etl_pipeline.py`.
    """
    # Call the core implementation
    res = aggregate_snapshot(snapshot_dir, labels=labels)

    # Re-open outputs to compute and print summary/logs (minimal, once per run)
    features_path = snapshot_dir / 'features_daily_updated.csv'
    fdf = pd.read_csv(features_path, dtype={'date': 'string'})
    # numeric cols count
    numeric_cols = [c for c in fdf.columns if pd.api.types.is_numeric_dtype(fdf[c])]
    # rules summary: compute mapping from agg_map_for
    amap = agg_map_for(fdf)
    # compute agg_dict as in aggregate_snapshot to know which rules applied
    sleep_dur_col = None
    for c in fdf.columns:
        if 'sleep_duration' in c.lower() or 'sleep_minutes' in c.lower() or 'sleep_min' in c.lower():
            sleep_dur_col = c
            break
    agg_dict = {}
    for c, rule in amap.items():
        if c not in fdf.columns:
            continue
        if not pd.api.types.is_numeric_dtype(fdf[c]):
            continue
        if rule == 'sum':
            agg_dict[c] = 'sum'
        elif rule == 'mean':
            agg_dict[c] = 'mean'
        elif rule == 'wmean_sleep':
            if sleep_dur_col:
                agg_dict[c] = ('wmean_sleep', sleep_dur_col)
            else:
                agg_dict[c] = 'mean'
        else:
            agg_dict[c] = 'mean'

    # dropped non-numeric from amap
    dropped = [c for c in amap.keys() if c in fdf.columns and not pd.api.types.is_numeric_dtype(fdf[c]) and c != 'segment_id']

    # summary by rule
    summary_by_rule = {}
    for v in agg_dict.values():
        key = v[0] if isinstance(v, tuple) else v
        summary_by_rule[key] = summary_by_rule.get(key, 0) + 1

    print(f"[aggregate] numeric_cols={len(numeric_cols)}; dropped_non_numeric={len(dropped)}")
    print(f"[aggregate] rules: {summary_by_rule}")

    return res


if __name__ == "__main__":
    import argparse as _arg
    ap = _arg.ArgumentParser()
    ap.add_argument("--participant", required=True)
    ap.add_argument("--snapshot", required=True)
    ap.add_argument("--labels", choices=["none", "synthetic"], default="none")
    a = ap.parse_args()
    from pathlib import Path as _P
    snap = _P(f"data_ai/{a.participant}/snapshots/{a.snapshot}")
    out = run(snap, labels=a.labels)
    print("✅ aggregate done:", out)
