#!/usr/bin/env python
"""PX8-Lite runner (per instructions)
"""
from pathlib import Path
import argparse
import sys
import json
import csv
import math

import pandas as pd
import numpy as np
from scipy.stats import ks_2samp


def find_daily_source(snapshot: Path):
    joined = snapshot / 'joined'
    processed = snapshot / 'processed'
    candidates = [
        (joined / 'features_daily.csv', str(joined / 'features_daily.csv')),
        (joined / 'features_daily_updated.csv', str(joined / 'features_daily_updated.csv')),
        (processed / 'features_daily.csv', str(processed / 'features_daily.csv')),
    ]
    for p, tag in candidates:
        if p.exists():
            return p, tag
    # try merging per-metric processed files
    # look for common per-metric names under processed/*
    proc = snapshot / 'processed'
    per_metric_files = []
    if proc.exists():
        for sub in proc.iterdir():
            if sub.is_dir():
                for f in sub.glob('*.csv'):
                    per_metric_files.append(f)
    # narrow to likely daily metrics
    names = ['hr','hrv','sleep','usage','activity']
    chosen = []
    for f in per_metric_files:
        n = f.name.lower()
        if any(x in n for x in names):
            chosen.append(f)
    if chosen:
        return ('merged_processed', 'merged_processed', chosen)
    return None, ''


def derive_merged_processed(snapshot: Path, chosen_files):
    # read each chosen file, ensure date column, and merge on date
    dfs = []
    for f in chosen_files:
        try:
            df = pd.read_csv(f)
        except Exception:
            continue
        # find date-like column
        date_col = None
        for c in df.columns:
            if c.lower() == 'date' or 'date' in c.lower() or c.lower() in ('start','end'):
                date_col = c
                break
        if date_col is None:
            continue
        try:
            df['date'] = pd.to_datetime(df[date_col]).dt.date
        except Exception:
            continue
        # select numeric columns besides date
        num_cols = [c for c in df.columns if c != date_col and pd.api.types.is_numeric_dtype(df[c])]
        if not num_cols:
            continue
        df = df[['date'] + num_cols]
        # rename numeric cols to include file stem to avoid collisions
        rename_map = {c: f"{f.stem}_{c}" for c in num_cols}
        df = df.rename(columns=rename_map)
        dfs.append(df)
    if not dfs:
        return pd.DataFrame()
    out = dfs[0]
    for d in dfs[1:]:
        out = out.merge(d, on='date', how='outer')
    # possibly compute means for columns that represent same metric prefixes - skip for simplicity
    return out


def drop_sparse_or_constant(feat: pd.DataFrame):
    if feat.empty:
        return feat
    cols = [c for c in feat.columns if c != 'date']
    keep = []
    for c in cols:
        s = feat[c]
        if s.isna().mean() > 0.9:
            continue
        if pd.api.types.is_numeric_dtype(s) and s.nunique(dropna=True) <= 1:
            continue
        keep.append(c)
    return feat[['date'] + keep]


def compute_drift(feat: pd.DataFrame, seg_df: pd.DataFrame):
    rows_out = []
    from collections import Counter
    flag_counts = Counter()
    for _, seg in seg_df.iterrows():
        sid = int(seg['segment_id'])
        start = seg['start']; end = seg['end']
        for c in feat.columns:
            if c == 'date':
                continue
            ser = feat[['date', c]].dropna()
            seg_vals = ser.loc[(ser['date'] >= start) & (ser['date'] <= end), c].astype(float).values
            other_vals = ser.loc[~((ser['date'] >= start) & (ser['date'] <= end)), c].astype(float).values
            if len(seg_vals) < 2 or len(other_vals) < 2:
                continue
            mean_a = float(np.nanmean(seg_vals))
            mean_b = float(np.nanmean(other_vals))
            delta = mean_a - mean_b
            delta_pct = (delta / (abs(mean_b) + 1e-9)) * 100.0
            try:
                ks_stat, ks_p = ks_2samp(seg_vals, other_vals)
            except Exception:
                ks_p = float('nan')
            flag = (abs(delta_pct) > 20.0) or (not math.isnan(ks_p) and ks_p < 0.05)
            rows_out.append({'feature': c, 'segment_id': f'S{sid}', 'n_seg': len(seg_vals), 'n_other': len(other_vals), 'delta_mean': delta, 'delta_pct': delta_pct, 'ks_p': ks_p, 'flag': flag})
            if flag:
                flag_counts[f'S{sid}'] += 1
    return rows_out, flag_counts


def write_reports(outdir: Path, rows_out, flag_counts, daily_source_tag):
    outdir.mkdir(parents=True, exist_ok=True)
    out_csv = outdir / 'drift_hint_summary.csv'
    with out_csv.open('w', newline='', encoding='utf-8') as fh:
        fieldnames = ['feature','segment_id','n_seg','n_other','delta_mean','delta_pct','ks_p','flag']
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows_out:
            writer.writerow(r)

    out_md = outdir / 'drift_hint_summary.md'
    lines = ['# Drift hint summary (PX8-Lite)', '']
    if flag_counts:
        for sid, cnt in sorted(flag_counts.items()):
            lines.append(f'- {sid}: {cnt} features flagged')
    else:
        lines.append('- No drift hints flagged across segments')
    lines.append('')
    lines.append('Recommendation: review flagged features and check temporal coverage per-source (see etl_provenance_report.csv)')
    lines.append(f'daily_source: {daily_source_tag}')
    with out_md.open('w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--snapshot', required=True)
    ap.add_argument('--outdir', required=True)
    ap.add_argument('--version-log', required=False)
    ap.add_argument('--provenance', required=False)
    ap.add_argument('--tz', default='Europe/Dublin')
    return ap.parse_args()


def main():
    args = parse_args()
    snap = Path(args.snapshot)
    outdir = Path(args.outdir)

    # version log
    if args.version_log:
        vpath = Path(args.version_log)
    else:
        vpath = snap / 'joined' / 'version_log_enriched.csv'
        if not vpath.exists():
            vpath = snap / 'version_log_enriched.csv'
    if not vpath.exists():
        print('NEED: version_log_enriched.csv')
        sys.exit(2)
    seg_df = pd.read_csv(vpath)
    seg_df['start'] = pd.to_datetime(seg_df['start']).dt.date
    seg_df['end'] = pd.to_datetime(seg_df['end']).dt.date

    found = find_daily_source(snap)
    if not found[0]:
        print('NEED: daily features source (none found and cannot derive)')
        sys.exit(2)

    # handle merged_processed special return
    if isinstance(found[0], tuple) or (found[0] == 'merged_processed'):
        # our helper returned merged marker
        # earlier we attempted to return ('merged_processed','merged_processed', chosen)
        # but here simply derive from processed per-metric
        # re-run to get chosen files
        proc = snap / 'processed'
        per_metric_files = []
        for sub in proc.iterdir():
            if sub.is_dir():
                for f in sub.glob('*.csv'):
                    per_metric_files.append(f)
        names = ['hr','hrv','sleep','usage','activity']
        chosen = [f for f in per_metric_files if any(x in f.name.lower() for x in names)]
        feat = derive_merged_processed(snap, chosen)
        daily_source_tag = 'merged_processed'
    else:
        p, tag = found
        if p == 'merged_processed':
            feat = pd.DataFrame()
            daily_source_tag = 'merged_processed'
        else:
            feat = pd.read_csv(p)
            if 'date' in feat.columns:
                feat['date'] = pd.to_datetime(feat['date']).dt.date
            daily_source_tag = tag

    if feat.empty:
        print('NEED: daily features table (no candidates)')
        sys.exit(2)

    feat = drop_sparse_or_constant(feat)
    rows_out, flag_counts = compute_drift(feat, seg_df)

    write_reports(outdir, rows_out, flag_counts, daily_source_tag)

    # READY true if at least one valid IN vs OUT comparison occurred (rows_out > 0)
    ready = len(rows_out) > 0
    fps = {k: int(v) for k, v in flag_counts.items()} if flag_counts else {}
    # Print exactly one SUMMARY line
    print(f'SUMMARY: rows={len(rows_out)} flagged_per_segment={json.dumps(fps)} daily_source="{daily_source_tag}" READY={str(bool(ready)).lower()}')


if __name__ == '__main__':
    main()
