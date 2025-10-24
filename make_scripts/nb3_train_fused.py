#!/usr/bin/env python3
"""NB3: Train models on fused daily dataset and write metrics, model artifact and manifest.

This is a deterministic scaffold: trains LogisticRegression and RandomForest, evaluates via temporal CV,
selects best by F1-macro, writes CSV metrics and a manifest with SHA256 of input dataset.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, roc_auc_score, balanced_accuracy_score, cohen_kappa_score, confusion_matrix


def sha256_of_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open('rb') as fh:
        for chunk in iter(lambda: fh.read(8192), b''):
            h.update(chunk)
    return h.hexdigest()


def atomic_write(path: Path, text: str):
    tmp = path.with_suffix(path.suffix + '.tmp')
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_text(text, encoding='utf-8')
    tmp.replace(path)


def temporal_cv_splits(df: pd.DataFrame, date_col='date', n_splits=6):
    # Simple month-based splits: assume date in YYYY-MM-DD
    df = df.sort_values(date_col)
    dates = pd.to_datetime(df[date_col]).dt.to_period('M')
    uniq = dates.unique()
    # take last n_splits periods if many
    if len(uniq) < n_splits:
        raise RuntimeError('Not enough periods for CV')
    # windows of 6: 4 train / 2 val sliding
    splits = []
    for i in range(len(uniq) - n_splits + 1):
        train_periods = uniq[i:i+4]
        val_periods = uniq[i+4:i+6]
        train_idx = dates.isin(train_periods)
        val_idx = dates.isin(val_periods)
        splits.append((train_idx.values, val_idx.values))
    return splits


def evaluate_model(clf, X_train, y_train, X_val, y_val):
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_val)
    try:
        y_prob = clf.predict_proba(X_val)[:,1] if hasattr(clf, 'predict_proba') else None
    except Exception:
        y_prob = None
    metrics = {
        'f1_macro': float(f1_score(y_val, y_pred, average='macro')),
        'balanced_acc': float(balanced_accuracy_score(y_val, y_pred)),
        'cohen_kappa': float(cohen_kappa_score(y_val, y_pred)),
    }
    if y_prob is not None:
        try:
            metrics['auroc_ovr'] = float(roc_auc_score(y_val, y_prob))
        except Exception:
            metrics['auroc_ovr'] = None
    else:
        metrics['auroc_ovr'] = None
    # confusion matrix stored separately
    metrics['confusion'] = confusion_matrix(y_val, y_pred).tolist()
    return metrics


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--ai-input', required=True, help='features_daily.csv')
    p.add_argument('--labels', required=True, help='features_daily_labeled.csv')
    p.add_argument('--out-dir', required=True)
    args = p.parse_args(argv)

    ai_path = Path(args.ai_input)
    labels_path = Path(args.labels)
    outdir = Path(args.out_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    df_x = pd.read_csv(ai_path)
    df_y = pd.read_csv(labels_path)
    df = df_x.merge(df_y, on='date', how='inner')

    # features: use fused_* if present else apple_/zepp_*
    feat_cols = [c for c in df.columns if c.startswith('fused_')]
    if not feat_cols:
        feat_cols = [c for c in df.columns if c.startswith('apple_') or c.startswith('zepp_')]

    X = df[feat_cols].fillna(0).values
    y = df['label'].values

    splits = temporal_cv_splits(df, date_col='date', n_splits=6)

    results_rows = []
    models = {'logreg': LogisticRegression(max_iter=1000, solver='liblinear'), 'rf': RandomForestClassifier(n_estimators=100, random_state=0)}
    model_agg_scores = {k: [] for k in models}
    seed = 0
    np.random.seed(seed)
    import random
    random.seed(seed)

    for fold_idx, (train_mask, val_mask) in enumerate(splits):
        X_train, X_val = X[train_mask], X[val_mask]
        y_train, y_val = y[train_mask], y[val_mask]
        for name, clf in models.items():
            metrics = evaluate_model(clf, X_train, y_train, X_val, y_val)
            row = {'fold': fold_idx, 'model': name}
            row.update(metrics)
            results_rows.append(row)
            model_agg_scores[name].append(metrics['f1_macro'])

    # select best model by mean F1-macro across folds
    best_model_name = max(model_agg_scores.items(), key=lambda kv: (float(np.mean(kv[1])), kv[0]))[0]

    # retrain best model on full data
    best_clf = models[best_model_name]
    best_clf.fit(X, y)

    # serialize model (pickle) as artifact
    import pickle
    model_path = outdir / 'model_fused_best.pkl'
    with model_path.open('wb') as fh:
        pickle.dump(best_clf, fh)

    # Optional: try TensorFlow TFLite conversion if tensorflow present and model is compatible
    try:
        import tensorflow as tf
        # Attempt a minimal conversion for linear models by building a simple Keras wrapper
        if hasattr(best_clf, 'coef_'):
            # build small Keras model
            inp_dim = X.shape[1]
            model_tf = tf.keras.Sequential([tf.keras.layers.InputLayer(input_shape=(inp_dim,)), tf.keras.layers.Dense(1, activation='sigmoid')])
            # set weights from coef_ and intercept_
            w = best_clf.coef_.reshape(-1,1)
            b = np.array([best_clf.intercept_])
            model_tf.layers[0].set_weights([w, b])
            tf_model_path = outdir / 'model_fused_best_tf'
            model_tf.save(str(tf_model_path), include_optimizer=False)
            # convert to tflite
            converter = tf.lite.TFLiteConverter.from_saved_model(str(tf_model_path))
            tflite_model = converter.convert()
            tflite_path = outdir / 'model_fused_best.tflite'
            tflite_path.write_bytes(tflite_model)
        else:
            # skip
            tflite_path = None
    except Exception:
        tflite_path = None

    # write metrics CSV deterministically
    metrics_csv = outdir / 'results_fused_metrics.csv'
    with metrics_csv.open('w', newline='', encoding='utf-8') as fh:
        writer = csv.DictWriter(fh, fieldnames=['fold','model','f1_macro','balanced_acc','cohen_kappa','auroc_ovr','confusion'])
        writer.writeheader()
        for r in results_rows:
            writer.writerow(r)

    # write manifest
    manifest = {
        'generated_at': datetime.utcnow().isoformat() + 'Z',
        'input_features_sha256': sha256_of_file(ai_path),
        'input_labels_sha256': sha256_of_file(labels_path),
        'best_model': best_model_name,
        'models_evaluated': list(models.keys()),
        'n_folds': len(splits),
        'seed': seed,
        'hyperparams': {'logreg': {'solver':'liblinear','max_iter':1000}, 'rf': {'n_estimators':100}}
    }
    atomic_write(outdir / 'model_fused_best_manifest.json', json.dumps(manifest, sort_keys=True, indent=2))

    print('WROTE:', metrics_csv, model_path, outdir / 'model_fused_best_manifest.json')


if __name__ == '__main__':
    main()
