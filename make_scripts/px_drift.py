#!/usr/bin/env python3
"""PX: Produce dataset drift summary per metric.

Writes drift_summary.csv and drift_summary.md with mean ± sd delta, KS p-value and ADWIN flag.
"""
from __future__ import annotations
import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd

try:
    from scipy.stats import ks_2samp
except Exception:
    def ks_2samp(a, b):
        return float('nan'), float('nan')


def load_feature_series(root: Path, metric_name: str) -> Dict[str, float]:
    # search for metric in processed apple/zepp/fused files
    candidates = list(root.glob(f'**/*{metric_name}*.csv'))
    out = {}
    for c in candidates:
        try:
            df = pd.read_csv(c)
            date_col = 'date' if 'date' in df.columns else (df.columns[0] if df.columns.size>0 else None)
            val_col = None
            for col in df.columns:
                if metric_name in col:
                    val_col = col; break
            if val_col and date_col:
                for _, r in df.iterrows():
                    d = str(r[date_col])
                    v = r[val_col]
                    try:
                        out[d] = float(v) if not pd.isna(v) else math.nan
                    except Exception:
                        out[d] = math.nan
        except Exception:
            continue
    return out


def adwin_like_flag(a_vals, b_vals, delta=0.002):
    # simple check: relative mean shift > delta
    ma = np.nanmean(a_vals) if len(a_vals)>0 else float('nan')
    mb = np.nanmean(b_vals) if len(b_vals)>0 else float('nan')
    if math.isnan(ma) or math.isnan(mb):
        return False
    return abs(mb - ma) > delta * (abs(ma) + 1e-6)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--processed-root', required=True)
    p.add_argument('--joined-dir', required=True)
    p.add_argument('--metrics', nargs='*', default=['hr','hrv_sdnn','hrv_rmssd','sleep_minutes'])
    args = p.parse_args(argv)

    root = Path(args.processed_root)
    outdir = Path(args.joined_dir); outdir.mkdir(parents=True, exist_ok=True)

    rows = []
    for m in args.metrics:
        series = load_feature_series(root, m)
        # naive split: first half vs second half of dates
        dates = sorted(series.keys())
        n = len(dates)
        if n < 2:
            continue
        mid = n//2
        first = [series[d] for d in dates[:mid] if d in series]
        second = [series[d] for d in dates[mid:] if d in series]
        # compute mean delta and sd
        mean_a = float(np.nanmean(first)) if first else float('nan')
        mean_b = float(np.nanmean(second)) if second else float('nan')
        delta = mean_b - mean_a if not math.isnan(mean_a) and not math.isnan(mean_b) else float('nan')
        sd_delta = float(np.nanstd([x - y for x,y in zip(first[:len(second)], second[:len(first)])])) if first and second else float('nan')
        try:
            ks_stat, ks_p = ks_2samp([x for x in first if not math.isnan(x)], [x for x in second if not math.isnan(x)])
        except Exception:
            ks_stat, ks_p = float('nan'), float('nan')
        adwin_flag = adwin_like_flag(first, second)
        rows.append({'metric': m, 'n_days': n, 'mean_delta': delta, 'sd_delta': sd_delta, 'ks_p': ks_p, 'adwin_flag': adwin_flag})

    # write CSV
    outcsv = outdir / 'drift_summary.csv'
    with outcsv.open('w', newline='', encoding='utf-8') as fh:
        writer = csv.DictWriter(fh, fieldnames=['metric','n_days','mean_delta','sd_delta','ks_p','adwin_flag'])
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    # write MD summary
    md = ['# Drift Summary','']
    for r in rows:
        md.append(f"- {r['metric']}: mean_delta={r['mean_delta']}, sd_delta={r['sd_delta']}, ks_p={r['ks_p']}, adwin={r['adwin_flag']}")
    (outdir / 'drift_summary.md').write_text('\n'.join(md), encoding='utf-8')
    print('WROTE:', outcsv, outdir / 'drift_summary.md')


if __name__ == '__main__':
    main()
