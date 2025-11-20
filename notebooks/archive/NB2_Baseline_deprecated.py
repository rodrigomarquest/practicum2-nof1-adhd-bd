"""NB2 Baselines only entrypoint.

Simple baselines (naive, ma7, rule-based, logistic regression). Rolling features optional.
"""
import argparse
from pathlib import Path
import sys
import os
import random
import numpy as np
import pandas as pd

# ensure project root is on sys.path so `src.nb_common` imports work when run from Make
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.nb_common.env import detect_env
from src.nb_common.io import resolve_slug_path, write_run_config
from src.nb_common.features import apply_rolling
from src.nb_common.folds import build_temporal_folds
from src.nb_common.metrics import eval_metrics, safe_class_report
from src.nb_common.reports import save_class_report_csv, save_confmat_png

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


def naive_persistence(y_series: pd.Series):
    out = y_series.shift(1)
    out.iloc[0] = y_series.mode().iloc[0] if not y_series.mode().empty else y_series.iloc[0]
    return out


def moving_avg_label(y_series: pd.Series, window=7):
    out = []
    for i in range(len(y_series)):
        lo = max(0, i - window)
        hist = y_series.iloc[lo:i]
        if len(hist) == 0:
            out.append(y_series.iloc[i])
        else:
            mode = hist.mode()
            out.append(mode.iloc[0] if not mode.empty else hist.iloc[-1])
    return pd.Series(out, index=y_series.index)


def run_baselines(df, out_root: Path, use_class_weight=True):
    df = df.sort_values('date').reset_index(drop=True)
    labels = df['label'].astype(str)
    classes = sorted(labels.unique())
    exclude = {'label', 'label_source', 'label_notes', 'date'}
    numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in exclude]
    Xnum = df[numeric_cols].ffill().fillna(0)
    scaler = StandardScaler(); Xnum_scaled = pd.DataFrame(scaler.fit_transform(Xnum), columns=Xnum.columns, index=Xnum.index)

    raw_folds = build_temporal_folds(df['date'], train_days=120, gap_days=10, val_days=60)
    folds = []
    for (tr_start, tr_end), (te_start, te_end) in raw_folds:
        tr_mask = df['date'].between(tr_start, tr_end)
        te_mask = df['date'].between(te_start, te_end)
        folds.append((tr_mask.values, te_mask.values, (tr_start, tr_end, te_start, te_end)))

    fig_dir = out_root / 'figures'; tab_dir = out_root / 'tables'; tab_dir.mkdir(parents=True, exist_ok=True); fig_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, (tr_mask, te_mask, windows) in enumerate(folds, start=1):
        tr_idx = np.where(tr_mask)[0]; te_idx = np.where(te_mask)[0]
        if len(tr_idx)==0 or len(te_idx)==0: continue
        y_tr = labels.iloc[tr_idx]; y_te = labels.iloc[te_idx]
        # naive
        y_pred_naive = naive_persistence(labels).iloc[te_idx]
        rows.append({'model':'naive','fold':i, **eval_metrics(y_te, y_pred_naive, classes=classes)})
        crep = safe_class_report(y_te, y_pred_naive)
        save_class_report_csv(crep, tab_dir / f'class_support_fold_{i}.csv')
        save_confmat_png(y_te, y_pred_naive, classes, fig_dir / f'confmat_fold_{i}.png')
        # ma7
        y_pred_ma7 = moving_avg_label(labels, window=7).iloc[te_idx]
        rows.append({'model':'ma7','fold':i, **eval_metrics(y_te, y_pred_ma7, classes=classes)})
        # rule-based (example)
        y_pred_rule = pd.Series('neutral', index=labels.index)
        rows.append({'model':'rule_based','fold':i, **eval_metrics(y_te, y_pred_rule.iloc[te_idx], classes=classes)})
        # logreg
        if numeric_cols:
            if labels.iloc[tr_idx].nunique() >= 2:
                try:
                    clf = LogisticRegression(max_iter=400, class_weight=('balanced' if use_class_weight else None), multi_class='multinomial', random_state=42)
                    clf.fit(Xnum_scaled.iloc[tr_idx], labels.iloc[tr_idx])
                    pred = clf.predict(Xnum_scaled.iloc[te_idx])
                    proba = clf.predict_proba(Xnum_scaled.iloc[te_idx])
                    rows.append({'model':'logreg','fold':i, **eval_metrics(labels.iloc[te_idx], pred, proba, classes=classes)})
                except Exception:
                    pass

    baseline_df = pd.DataFrame(rows)
    baseline_df.to_csv(tab_dir / 'nb2_baseline_metrics_per_fold.csv', index=False)
    return baseline_df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--slug', required=True)
    ap.add_argument('--local-root', required=False, help='(ignored) local datasets root')
    ap.add_argument('--dry-run', action='store_true', help='(ignored) dry run')
    ap.add_argument('--rolling-windows', nargs='+', type=int, default=[7,14])
    ap.add_argument('--rolling-features', action='store_true')
    ap.add_argument('--use-class-weight', dest='use_class_weight', action='store_true')
    ap.add_argument('--no-class-weight', dest='use_class_weight', action='store_false')
    args = ap.parse_args()
    env = detect_env()
    features_p = resolve_slug_path(args.slug, env['data_root'])
    if not features_p.exists():
        print('ERROR: features not found', features_p); return 2
    df = pd.read_csv(features_p, parse_dates=['date'])
    if args.rolling_features:
        df = apply_rolling(df, args.rolling_windows)
    outroot = Path('notebooks/outputs/NB2') / args.slug / Path().absolute().name
    outroot.mkdir(parents=True, exist_ok=True)
    baseline_df = run_baselines(df, outroot, use_class_weight=bool(args.use_class_weight))
    print('INFO: Baselines complete. Models:', baseline_df['model'].unique().tolist())
    # write run config
    write_run_config(outroot / 'run_config.json', {'slug': args.slug, 'rows': len(df)})
    return 0


if __name__ == '__main__':
    sys.exit(main())
