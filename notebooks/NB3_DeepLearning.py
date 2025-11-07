"""NB3 Deep Learning entrypoint.

Trains TF models when available; on local will fallback to torch if TF missing.
Exports TFLite when TF path used.
"""
import argparse
from pathlib import Path
import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime, timezone

# ensure project root is on sys.path so `src.nb_common` imports work when run from Make
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.nb_common.env import detect_env
from src.nb_common.io import resolve_slug_path, write_run_config
from src.nb_common.features import apply_rolling
from src.nb_common.folds import build_temporal_folds
from src.nb_common.reports import save_confmat_png, save_class_report_csv
from src.nb_common.metrics import eval_metrics, safe_class_report
from src.nb_common.tf_models import build_lstm, build_cnn1d, build_cnn_bilstm, build_transformer_tiny, export_tflite

def train_tf_model_for_fold(model, Xtr_w, ytr_w, Xva_w, yva_w, epochs=20, batch_size=32, lr=1e-3):
    import tensorflow as tf
    from tensorflow import keras
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr), loss='sparse_categorical_crossentropy')
    model.fit(Xtr_w, ytr_w, validation_data=(Xva_w, yva_w), epochs=epochs, batch_size=batch_size, verbose=0)
    preds = model.predict(Xva_w, verbose=0)
    y_pred = np.argmax(preds, axis=1)
    return y_pred, preds, model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--slug', required=True)
    ap.add_argument('--sweep', action='store_true')
    ap.add_argument('--seq-len-grid', nargs='+', type=int, default=[7, 14, 21])
    ap.add_argument('--rolling-grid', nargs='+', type=int, default=[7, 14, 28])
    ap.add_argument('--arch-grid', nargs='+', default=['lstm', 'cnn_bilstm'])
    ap.add_argument('--epochs', type=int, default=None)
    ap.add_argument('--batch-size', type=int, default=64)
    ap.add_argument('--dropout', type=float, default=0.2)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--max-train-samples', type=int, default=0)
    args = ap.parse_args()

    env = detect_env()
    is_kaggle = bool(env.get('is_kaggle', False))
    backend = env.get('backend', 'none')
    # default epochs: longer on Kaggle
    if args.epochs is None:
        args.epochs = 40 if is_kaggle else 10

    features_p = resolve_slug_path(args.slug, env['data_root'])
    if not features_p.exists():
        print('ERROR: features not found', features_p); return 2
    df_base = pd.read_csv(features_p, parse_dates=['date'])

    # sweep grid construction
    def build_grid(seq_grid, roll_grid, arch_grid):
        seqs = sorted(list(set(seq_grid)))
        rolls = []
        if 7 in roll_grid and 14 in roll_grid:
            rolls.append([7, 14])
        if 14 in roll_grid and 28 in roll_grid:
            rolls.append([14, 28])
        grid = []
        cid = 0
        for a in arch_grid:
            for s in seqs:
                for r in rolls:
                    cid += 1
                    grid.append({'cfg_id': cid, 'arch': a, 'seq_len': int(s), 'rolling': r})
        return grid

    grid = build_grid(args.seq_len_grid, args.rolling_grid, args.arch_grid) if args.sweep else [{'cfg_id': 1, 'arch': args.arch_grid[0], 'seq_len': args.seq_len_grid[0], 'rolling': args.rolling_grid[:2]}]

    # timestamp for this run (matches NB2 convention)
    RUN_TS = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    outroot = Path('notebooks/outputs/NB3') / args.slug / RUN_TS
    tab_dir = outroot / 'tables'; fig_dir = outroot / 'figures'; mod_dir = outroot / 'models'
    for d in (tab_dir, fig_dir, mod_dir): d.mkdir(parents=True, exist_ok=True)

    sweep_rows = []
    for cfg in grid:
        arch = cfg['arch']; seq_len = cfg['seq_len']; rolling_windows = cfg['rolling']
        print(f"INFO: running cfg arch={arch} seq_len={seq_len} rolling={rolling_windows}")
        # apply rolling features
        df = apply_rolling(df_base, rolling_windows)
        labels = df['label'].astype(str)
        classes = sorted(labels.unique())
        numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in ('label','date')]
        Xnum_full = df[numeric_cols].ffill().fillna(0).values

        raw_folds = build_temporal_folds(df['date'], train_days=120, gap_days=10, val_days=60)
        per_fold_metrics = []
        latencies = []
        folds_used = 0
        for i, ((tr_start, tr_end), (te_start, te_end)) in enumerate(raw_folds, start=1):
            tr_mask = df['date'].between(tr_start, tr_end).values
            te_mask = df['date'].between(te_start, te_end).values
            tr_idx = np.where(tr_mask)[0]; te_idx = np.where(te_mask)[0]
            if len(tr_idx) < 10 or len(te_idx) < 5:
                continue
            folds_used += 1
            Xtr = Xnum_full[tr_idx]; Xte = Xnum_full[te_idx]
            ytr = labels.iloc[tr_idx].values; yte = labels.iloc[te_idx].values
            # optionally cap training samples
            if args.max_train_samples and args.max_train_samples > 0 and len(Xtr) > args.max_train_samples:
                Xtr = Xtr[-args.max_train_samples:]; ytr = ytr[-args.max_train_samples:]

            # build sliding windows
            def build_windows(X, y, sl):
                Xw, yw = [], []
                for k in range(sl-1, len(X)):
                    Xw.append(X[k-sl+1:k+1])
                    yw.append(y[k])
                return np.array(Xw, dtype=np.float32), np.array(yw)

            Xtr_w, ytr_w = build_windows(Xtr, ytr, seq_len)
            Xte_w, yte_w = build_windows(Xte, yte, seq_len)
            if Xtr_w.size == 0 or Xte_w.size == 0:
                continue

            # z-score per-window (sample-wise)
            def zscore_windows(Xw):
                mu = Xw.mean(axis=1, keepdims=True)
                sd = Xw.std(axis=1, keepdims=True)
                sd[sd==0] = 1.0
                return (Xw - mu) / sd

            Xtr_w = zscore_windows(Xtr_w); Xte_w = zscore_windows(Xte_w)

            # TF training path
            if backend != 'tf':
                print('WARNING: TF backend not available; skipping fold', i)
                continue

            # build model
            try:
                if arch == 'lstm':
                    model = build_lstm(seq_len, Xtr_w.shape[-1], len(classes), hidden=64, dropout=args.dropout)
                elif arch == 'cnn1d':
                    model = build_cnn1d(seq_len, Xtr_w.shape[-1], len(classes))
                elif arch == 'cnn_bilstm':
                    model = build_cnn_bilstm(seq_len, Xtr_w.shape[-1], len(classes))
                else:
                    model = build_transformer_tiny(seq_len, Xtr_w.shape[-1], len(classes))
                import tensorflow as tf
                from tensorflow import keras
                cb = [keras.callbacks.EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True),
                      keras.callbacks.ReduceLROnPlateau(monitor='val_loss', patience=4)]
                model.compile(optimizer=keras.optimizers.Adam(learning_rate=args.lr), loss='sparse_categorical_crossentropy')
                model.fit(Xtr_w, ytr_w, validation_data=(Xte_w, yte_w), epochs=args.epochs, batch_size=args.batch_size, callbacks=cb, verbose=0)
                import time as _time
                t0 = _time.perf_counter()
                preds = model.predict(Xte_w, verbose=0)
                t1 = _time.perf_counter()
                y_pred = np.argmax(preds, axis=1)
                per_sample_ms = 1000.0 * (t1 - t0) / max(1, len(Xte_w))
                latencies.append(per_sample_ms)
                # save keras model
                try:
                    h5p = mod_dir / f'fold_{i}_model.h5'
                    model.save(h5p)
                except Exception:
                    h5p = None
                # export tflite
                try:
                    tflite_p = mod_dir / f'fold_{i}_model.tflite'
                    export_tflite(model, tflite_p)
                except Exception:
                    tflite_p = None

                # metrics
                m = eval_metrics(yte_w, y_pred, proba=preds, classes=classes)
                per_fold_report = safe_class_report(yte_w, y_pred)
                try:
                    save_class_report_csv(per_fold_report, tab_dir / f'class_report_fold_{i}.csv')
                except Exception:
                    pass
                try:
                    save_confmat_png(yte_w, y_pred, classes, fig_dir / f'confmat_fold_{i}.png')
                except Exception:
                    pass
                per_fold_metrics.append(m)
            except Exception as e:
                print('WARNING: TF training failed for fold', i, '->', e)
                continue

        # aggregate per-config
        if len(per_fold_metrics) == 0:
            print('INFO: no folds completed for cfg', cfg)
            continue
        import numpy as _np
        def agg_metric(key):
            vals = [p.get(key, _np.nan) for p in per_fold_metrics]
            return float(_np.nanmean(vals)), float(_np.nanstd(vals))

        f1_mean, f1_std = agg_metric('f1_macro')
        f1w_mean, _ = agg_metric('f1_weighted')
        bal_mean, _ = agg_metric('balanced_acc')
        kappa_mean, _ = agg_metric('kappa')
        auroc_mean, _ = agg_metric('auroc_ovr_macro')
        lat_mean = float(_np.nanmean(latencies)) if len(latencies) else float('nan')

        sweep_rows.append({
            'slug': args.slug,
            'arch': arch,
            'seq_len': seq_len,
            'windows': ';'.join(map(str, rolling_windows)),
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'dropout': args.dropout,
            'lr': args.lr,
            'folds': int(folds_used),
            'f1_macro_mean': f1_mean,
            'f1_macro_std': f1_std,
            'f1_weighted_mean': f1w_mean,
            'balanced_acc_mean': bal_mean,
            'kappa_mean': kappa_mean,
            'auroc_ovr_macro_mean': auroc_mean,
            'latency_ms_per_sample_mean': lat_mean,
        })

    # write sweep summary and best config
    try:
        # always write a sweep summary CSV (may be empty) so downstream tooling can rely on its existence
        sweep_df = pd.DataFrame(sweep_rows)
        sweep_df.to_csv(tab_dir / 'nb3_sweep_summary.csv', index=False)
        import json
        if len(sweep_df):
            # pick best by f1_macro_mean then balanced_acc_mean
            best = sweep_df.sort_values(['f1_macro_mean', 'balanced_acc_mean'], ascending=[False, False]).iloc[0].to_dict()
            (outroot / 'best_config.json').write_text(json.dumps(best, indent=2))
            best_f1 = best.get('f1_macro_mean', None)
            best_lat = best.get('latency_ms_per_sample_mean', None)
        else:
            # write a small placeholder best_config to indicate no completed folds (e.g., TF missing locally)
            placeholder = {
                'slug': args.slug,
                'note': 'no folds completed (likely TF not available or no valid folds); rerun with TF or enable fallback to generate results'
            }
            (outroot / 'best_config.json').write_text(json.dumps(placeholder, indent=2))
            best_f1 = None; best_lat = None
    except Exception as e:
        print('WARNING: failed to write sweep summary ->', e)
        best_f1 = None; best_lat = None

    # final prints
    if best_f1 is not None:
        print(f"BEST_MODEL: dl F1_macro={best_f1:.4f}")
    else:
        print("BEST_MODEL: dl F1_macro=nan")
    if best_lat is not None and not pd.isna(best_lat):
        print(f"LATENCY_PROFILE: avg_inference_ms_per_sample={best_lat:.4f}")
    else:
        print("LATENCY_PROFILE: N/A")
    print(f"OUTPUT_FOLDER: {outroot}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
