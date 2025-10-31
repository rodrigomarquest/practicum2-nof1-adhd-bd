#!/usr/bin/env python3
"""NB4: SHAP explainability and drift analysis for best fused model.

Scaffold that computes SHAP (if available), ranks features, plots top-5 and summary, runs KS tests across folds and a simple ADWIN-like flag.
"""
from __future__ import annotations
import argparse
import csv
import math
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from scipy.stats import ks_2samp
except Exception:
    def ks_2samp(a, b):
        # scipy not available: return nan stats so downstream still runs
        return float('nan'), float('nan')

try:
    import shap
except Exception:
    shap = None


def atomic_write(path: Path, text: str):
    tmp = path.with_suffix(path.suffix + '.tmp')
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_text(text, encoding='utf-8')
    tmp.replace(path)


def compute_shap_importance(model, X: np.ndarray, feature_names: List[str]) -> Dict[str, float]:
    if shap is None:
        # fallback: use permutation-like importance approximated by feature std * coef (if linear)
        if hasattr(model, 'coef_'):
            coefs = np.abs(model.coef_).ravel()
            imp = {fn: float(coef) for fn, coef in zip(feature_names, coefs)}
            return imp
        else:
            # uniform importance
            return {fn: 1.0 for fn in feature_names}

    expl = shap.Explainer(model, X)
    vals = expl(X)
    # use mean absolute SHAP per feature
    arr = np.abs(vals.values).mean(axis=0)
    return {fn: float(v) for fn, v in zip(feature_names, arr)}


def plot_top5_shap(imp: Dict[str, float], outpng: Path, topn=5):
    items = sorted(imp.items(), key=lambda kv: -kv[1])[:topn]
    names, vals = zip(*items)
    plt.figure(figsize=(6,4))
    plt.barh(names[::-1], vals[::-1])
    plt.title('Top SHAP features')
    plt.tight_layout()
    outpng.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outpng)
    plt.close()


def ks_adwin_flag(series_a: np.ndarray, series_b: np.ndarray, adwin_delta=0.002) -> Dict[str, Any]:
    # KS test
    try:
        stat, p = ks_2samp(series_a, series_b)
    except Exception:
        stat, p = float('nan'), float('nan')
    # simple adwin-like: flag if mean change relative > delta
    m_a, m_b = float(np.nanmean(series_a)), float(np.nanmean(series_b))
    shift_pct = abs(m_b - m_a) / (abs(m_a) + 1e-6) * 100.0 if not math.isnan(m_a) else float('nan')
    adwin_flag = abs(m_b - m_a) > adwin_delta * (abs(m_a) + 1e-6)
    return {'ks_p': p, 'ks_stat': stat, 'adwin_flag': bool(adwin_flag), 'shift_pct': shift_pct}


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--ai-input', required=True)
    p.add_argument('--model-pkl', required=True)
    p.add_argument('--labels', required=True)
    p.add_argument('--out-dir', required=True)
    args = p.parse_args(argv)

    ai_path = Path(args.ai_input)
    model_pkl = Path(args.model_pkl)
    labels_path = Path(args.labels)
    outdir = Path(args.out_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(ai_path)
    # features selection
    feat_cols = [c for c in df.columns if c.startswith('fused_')]
    if not feat_cols:
        feat_cols = [c for c in df.columns if c.startswith('apple_') or c.startswith('zepp_')]

    X = df[feat_cols].fillna(0).values

    # load model
    import pickle
    with model_pkl.open('rb') as fh:
        model = pickle.load(fh)

    # compute SHAP importances
    imp = compute_shap_importance(model, X, feat_cols)
    # write top5 plot and summary
    plot_top5_shap(imp, outdir / 'shap_fused_top5.png', topn=5)
    # save shap summary plot placeholder
    plt.figure(figsize=(6,6))
    plt.text(0.1,0.5,'SHAP summary placeholder', fontsize=12)
    plt.axis('off')
    plt.savefig(outdir / 'shap_summary.png')
    plt.close()

    # temporal drift: split into folds via labels file (assume labels contains fold column)
    lbl = pd.read_csv(labels_path)
    df2 = df.merge(lbl[['date','fold']], on='date', how='left')
    folds = sorted([c for c in df2['fold'].unique() if not pd.isna(c)])

    drift_rows = []
    # compare fold0 vs foldN as example pairings
    if len(folds) >= 2:
        base_fold = folds[0]
        for f in folds[1:]:
            left = df2[df2['fold']==base_fold]
            right = df2[df2['fold']==f]
            for feat in feat_cols:
                stats = ks_adwin_flag(left[feat].fillna(0).values, right[feat].fillna(0).values)
                drift_rows.append({'fold_a': int(base_fold), 'fold_b': int(f), 'feature': feat, 'ks_p': stats['ks_p'], 'adwin_flag': stats['adwin_flag'], 'shap_drift_pct': stats['shift_pct']})

    drift_csv = outdir / 'drift_summary.csv'
    with drift_csv.open('w', newline='', encoding='utf-8') as fh:
        writer = csv.DictWriter(fh, fieldnames=['fold_a','fold_b','feature','ks_p','adwin_flag','shap_drift_pct'])
        writer.writeheader()
        for r in drift_rows:
            writer.writerow(r)

    # write markdown report
    md = ['# SHAP & Drift Report', f'Generated: {datetime.utcnow().isoformat()}Z', '## Top features by SHAP']
    top10 = sorted(imp.items(), key=lambda kv: -kv[1])[:10]
    for name, v in top10:
        md.append(f'- {name}: {v:.6f}')
    md.append('\n## Drift Summary')
    md.append('See drift_summary.csv')
    atomic_write(outdir / 'shap_drift_report.md', '\n'.join(md))

    print('WROTE:', outdir / 'shap_fused_top5.png', outdir / 'shap_summary.png', drift_csv, outdir / 'shap_drift_report.md')


if __name__ == '__main__':
    main()
