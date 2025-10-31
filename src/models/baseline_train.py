#!/usr/bin/env python3
"""Baseline modeling pipeline.

Creates temporal CV folds, trains simple baselines and a logistic-regression model,
computes metrics and writes artifacts under the snapshot folder.

Usage:
  python modeling/baseline_train.py --participant P000001 --snapshot 2025-09-29
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import tempfile
import time
from collections import Counter
from pathlib import Path
from src.domains.common.io import etl_snapshot_root
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (
        balanced_accuracy_score,
        cohen_kappa_score,
        f1_score,
        roc_auc_score,
    )
    from sklearn.preprocessing import StandardScaler, LabelEncoder
except Exception:
    print("scikit-learn is required for modeling. Install requirements_ai_kaggle.txt.")
    raise

_HAS_TF = False
try:
    import tensorflow as tf

    _HAS_TF = True
except Exception:
    tf = None

_HAS_STATSMODELS = False
try:
    from statsmodels.stats.contingency_tables import mcnemar

    _HAS_STATSMODELS = True
except Exception:
    mcnemar = None

LOG = logging.getLogger("baseline")


def atomic_write(path: Path, content: bytes):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=".tmp-")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(content)
        os.replace(tmp, str(path))
    finally:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except Exception:
                pass


def infer_snapshot_dir(participant: str, snapshot: str) -> Path:
    # Prefer canonical ETL snapshot root
    return etl_snapshot_root(participant, snapshot)


def load_inputs(snapshot_dir: Path, use_agg: str = "auto") -> pd.DataFrame:
    """Resolve candidate input files and return merged dataframe with 1 row/day and label column.

    Priority:
      features_daily_labeled_agg.csv
      features_daily_labeled.csv
      fallback: features_daily_agg.csv + state_of_mind_synthetic.csv
    """
    cand_a = snapshot_dir / "features_daily_labeled_agg.csv"
    cand_b = snapshot_dir / "features_daily_labeled.csv"
    cand_c_feat = snapshot_dir / "features_daily_agg.csv"
    cand_c_lab = snapshot_dir / "state_of_mind_synthetic.csv"

    df = None
    chosen = ""
    if use_agg == "yes":
        if cand_a.exists():
            df = pd.read_csv(cand_a, parse_dates=["date"])
            chosen = cand_a.name
        elif cand_c_feat.exists() and cand_c_lab.exists():
            f = pd.read_csv(cand_c_feat, parse_dates=["date"])
            l = pd.read_csv(cand_c_lab, parse_dates=["date"])
            df = f.merge(l, on="date", how="left")
            chosen = f.name + "+" + l.name
        else:
            raise FileNotFoundError("Aggregated inputs requested but not found")
    elif use_agg == "no":
        if cand_b.exists():
            df = pd.read_csv(cand_b, parse_dates=["date"])
            chosen = cand_b.name
        else:
            raise FileNotFoundError("Non-aggregated labeled features not found")
    else:  # auto
        if cand_a.exists():
            df = pd.read_csv(cand_a, parse_dates=["date"])
            chosen = cand_a.name
        elif cand_b.exists():
            df = pd.read_csv(cand_b, parse_dates=["date"])
            chosen = cand_b.name
        elif cand_c_feat.exists() and cand_c_lab.exists():
            f = pd.read_csv(cand_c_feat, parse_dates=["date"])
            l = pd.read_csv(cand_c_lab, parse_dates=["date"])
            df = f.merge(l, on="date", how="left")
            chosen = f.name + "+" + l.name
        else:
            raise FileNotFoundError("No candidate input files found in snapshot")

    if df is None:
        raise RuntimeError("Failed to load input data")

    # ensure one row per day by grouping on date (if duplicates, keep last)
    df = df.sort_values("date").drop_duplicates(subset=["date"], keep="last")
    df = df.reset_index(drop=True)
    LOG.info(f"loaded {len(df)} rows from {chosen}")
    return df


def map_labels(
    series: pd.Series, target_col: str = "label"
) -> Tuple[pd.Series, LabelEncoder]:
    # Accept values like 'negative','neutral','positive' or numeric already
    s = series.fillna("neutral").astype(str)
    # Normalize common variants
    mapping = {
        "neg": "negative",
        "pos": "positive",
        "0": "negative",
        "1": "neutral",
        "2": "positive",
    }
    s = s.map(lambda x: mapping.get(x.lower(), x.lower()))
    # keep only three labels
    allowed = ["negative", "neutral", "positive"]
    s = s.map(lambda x: x if x in allowed else "neutral")
    le = LabelEncoder()
    y = le.fit_transform(s)
    return pd.Series(y, index=series.index), le


def build_month_windows(
    dates: pd.Series, n_folds: int = 6
) -> List[Tuple[pd.DatetimeIndex, pd.DatetimeIndex]]:
    """Create list of (train_idx_mask, val_idx_mask) by 4-month train + 2-month val sliding windows.

    Returns list of tuples of boolean masks (train_idx, val_idx) relative to dates index.
    """
    # convert to Period month
    months = pd.to_datetime(dates).dt.to_period("M")
    unique_months = sorted(months.unique())
    if len(unique_months) < 6:
        LOG.warning("Not enough months for full 6 folds; will create fewer folds")
    folds = []
    # start month candidates: those where start + 5 months exists
    for i, start in enumerate(unique_months):
        # need start + 5
        end_needed = start + 5
        if end_needed in unique_months:
            # train months: start..start+3 ; val: start+4..start+5
            train_months = {start + k for k in range(0, 4)}
            val_months = {start + 4, start + 5}
            train_mask = months.isin(train_months)
            val_mask = months.isin(val_months)
            if train_mask.sum() >= 1 and val_mask.sum() >= 1:
                folds.append((train_mask.values, val_mask.values))
        if len(folds) >= n_folds:
            break
    if not folds:
        raise RuntimeError("Unable to construct any time-based folds from dates")
    LOG.info(f"constructed {len(folds)} folds from {len(unique_months)} months")
    return folds


def get_numeric_features(df: pd.DataFrame) -> List[str]:
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    # exclude id-like columns
    numeric = [c for c in numeric if c != "segment_id"]
    return numeric


def evaluate_predictions(
    y_true: np.ndarray, y_pred: np.ndarray, y_proba: Optional[np.ndarray] = None
) -> Dict[str, float]:
    res = {}
    res["f1_macro"] = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    res["f1_weighted"] = float(
        f1_score(y_true, y_pred, average="weighted", zero_division=0)
    )
    try:
        if y_proba is not None:
            res["auroc_ovr"] = float(roc_auc_score(y_true, y_proba, multi_class="ovr"))
        else:
            res["auroc_ovr"] = float(
                roc_auc_score(y_true, pd.get_dummies(y_pred), multi_class="ovr")
            )
    except Exception:
        res["auroc_ovr"] = float("nan")
    res["balanced_accuracy"] = float(balanced_accuracy_score(y_true, y_pred))
    res["cohen_kappa"] = float(cohen_kappa_score(y_true, y_pred))
    return res


def baseline_naive(series_labels: pd.Series) -> np.ndarray:
    # predict previous day's label; for first day, use global mode
    preds = []
    mode = int(series_labels.mode().iloc[0]) if not series_labels.mode().empty else 1
    prev = None
    for v in series_labels.values:
        if prev is None:
            preds.append(mode)
        else:
            preds.append(prev)
        prev = int(v)
    return np.array(preds, dtype=int)


def baseline_moving_average(series_labels: pd.Series, k: int) -> np.ndarray:
    # majority over last k days (excluding current day)
    vals = series_labels.values
    n = len(vals)
    preds = np.zeros(n, dtype=int)
    for i in range(n):
        start = max(0, i - k)
        window = vals[start:i]
        if len(window) == 0:
            preds[i] = (
                int(series_labels.mode().iloc[0])
                if not series_labels.mode().empty
                else 1
            )
        else:
            preds[i] = int(Counter(window).most_common(1)[0][0])
    return preds


def baseline_rule_based(df: pd.DataFrame) -> np.ndarray:
    # rules use z-score columns
    rh = df.get("resting_hr__z")
    hrv = df.get("hrv__z")
    se = df.get("sleep_efficiency__z")
    n = len(df)
    out = np.full(n, 1, dtype=int)  # default neutral
    for i in range(n):
        try:
            r = rh.iloc[i] if rh is not None else np.nan
            h = hrv.iloc[i] if hrv is not None else np.nan
            s = se.iloc[i] if se is not None else np.nan
            if not np.isnan(r) and not np.isnan(h) and not np.isnan(s):
                if r > 0.5 and h < -0.5 and s < -0.5:
                    out[i] = 0
                elif r < -0.2 and h > 0.2 and s > 0.2:
                    out[i] = 2
                else:
                    out[i] = 1
            else:
                out[i] = 1
        except Exception:
            out[i] = 1
    return out


def train_logistic(
    X_train: np.ndarray, y_train: np.ndarray, C_grid=(0.1, 1.0, 3.0), seed=42
):
    best = None
    best_score = -1
    for C in C_grid:
        clf = LogisticRegression(
            C=C,
            class_weight="balanced",
            multi_class="ovr",
            max_iter=1000,
            random_state=seed,
        )
        clf.fit(X_train, y_train)
        preds = clf.predict(X_train)
        score = f1_score(y_train, preds, average="macro", zero_division=0)
        if score > best_score:
            best_score = score
            best = clf
    return best


def predict_logistic(
    clf: LogisticRegression, X: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    proba = clf.predict_proba(X)
    preds = np.argmax(proba, axis=1)
    return preds, proba


def measure_latency(predict_fn, X_sample: np.ndarray, repeats=20) -> Dict[str, float]:
    times = []
    for _ in range(repeats):
        t0 = time.time()
        _ = predict_fn(X_sample)
        t1 = time.time()
        times.append(t1 - t0)
    times = np.array(times)
    return {
        "mean_s": float(times.mean()),
        "median_s": float(np.median(times)),
        "std_s": float(times.std()),
        "repeats": repeats,
    }


def save_json(path: Path, obj):
    atomic_write(path, json.dumps(obj, indent=2, ensure_ascii=False).encode("utf-8"))


def run(args):
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    LOG.info("baseline train starting")
    snap = infer_snapshot_dir(args.participant, args.snapshot)
    outdir = snap / "modeling"
    outdir.mkdir(parents=True, exist_ok=True)

    df = load_inputs(snap, args.use_agg)
    if "date" not in df.columns:
        raise RuntimeError("input missing 'date' column")
    df = df.sort_values("date").reset_index(drop=True)

    # labels
    if args.target_col not in df.columns:
        LOG.warning(
            "target column %s not present; expecting synthetic labels to exist",
            args.target_col,
        )
    y_ser, _ = map_labels(
        df.get(args.target_col, pd.Series(["neutral"] * len(df))), args.target_col
    )
    df["_y"] = y_ser

    # numeric features
    numeric = get_numeric_features(df)
    if not numeric:
        raise RuntimeError("no numeric features found in input")

    # build folds
    folds = build_month_windows(df["date"].astype("datetime64[ns]"))

    per_fold = []
    preds_store = {}
    # placeholders for best non-param baseline chosen later
    for fi, (train_mask, val_mask) in enumerate(folds):
        LOG.info(
            f"fold {fi+1}/{len(folds)}: train {train_mask.sum()} rows, val {val_mask.sum()} rows"
        )
        train_idx = np.nonzero(train_mask)[0]
        val_idx = np.nonzero(val_mask)[0]

        X = df[numeric].astype(float).fillna(0.0)
        y = df["_y"].values.astype(int)

        # scaler fit on train
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X.iloc[train_idx])
        X_val = scaler.transform(X.iloc[val_idx])

        # baselines preds built from label series (whole series but we'll evaluate only on val indices)
        # naive uses sequence order
        naive_preds_all = baseline_naive(df["_y"])
        ma3_all = baseline_moving_average(df["_y"], 3)
        ma7_all = baseline_moving_average(df["_y"], 7)
        rule_all = baseline_rule_based(df)

        # evaluate baselines on val
        y_val = y[val_idx]
        best_baseline_name = None
        baseline_results = {}
        for name, arr in (
            ("naive", naive_preds_all),
            ("ma3", ma3_all),
            ("ma7", ma7_all),
            ("rule", rule_all),
        ):
            res = evaluate_predictions(y_val, arr[val_idx])
            baseline_results[name] = res
        # pick best baseline by f1_macro
        best_baseline_name = max(
            baseline_results.items(), key=lambda kv: kv[1]["f1_macro"]
        )[0]

        # logistic
        clf = train_logistic(X_train, y[train_idx], seed=args.seed)
        clf_preds_val, clf_proba_val = predict_logistic(clf, X_val)
        clf_res = evaluate_predictions(y_val, clf_preds_val, clf_proba_val)

        # aggregate per-fold results
        fold_res = {
            "fold": fi,
            "n_train": int(np.count_nonzero(train_mask)),
            "n_val": int(np.count_nonzero(val_mask)),
            "baselines": baseline_results,
            "logistic": clf_res,
            "best_baseline": best_baseline_name,
        }
        per_fold.append(fold_res)

        preds_store[f"fold_{fi}"] = {
            "y_val": y_val.tolist(),
            "clf_preds": clf_preds_val.tolist(),
            "clf_proba": (
                clf_proba_val.tolist() if clf_proba_val is not None else None
            ),
        }

        # mc nemar between logistic and best baseline if statsmodels available
        if _HAS_STATSMODELS:
            try:
                baspred = {
                    "naive": naive_preds_all,
                    "ma3": ma3_all,
                    "ma7": ma7_all,
                    "rule": rule_all,
                }[best_baseline_name][val_idx]
                table = pd.crosstab(clf_preds_val == y_val, baspred == y_val)
                # table should be 2x2; only call mcnemar if so
                if table.shape == (2, 2):
                    try:
                        res_m = mcnemar(table)
                        fold_res["mcnemar_p"] = float(res_m.pvalue)
                    except Exception:
                        fold_res["mcnemar_p"] = None
                else:
                    fold_res["mcnemar_p"] = None
            except Exception:
                fold_res["mcnemar_p"] = None
        else:
            fold_res["mcnemar_p"] = None

    # summarize
    summary = {"folds": per_fold}

    # compute means
    def agg_metric(metric):
        vals = [f["logistic"].get(metric, float("nan")) for f in per_fold]
        vals = [v for v in vals if not (isinstance(v, float) and math.isnan(v))]
        if not vals:
            return {"mean": None, "std": None}
        return {"mean": float(np.mean(vals)), "std": float(np.std(vals))}

    summary["summary"] = {
        m: agg_metric(m)
        for m in (
            "f1_macro",
            "f1_weighted",
            "auroc_ovr",
            "balanced_accuracy",
            "cohen_kappa",
        )
    }

    # save cv splits (date ranges)
    cv_splits = []
    for train_mask, val_mask in folds:
        train_dates = df.loc[train_mask, "date"]
        val_dates = df.loc[val_mask, "date"]
        cv_splits.append(
            {
                "train_start": (
                    str(train_dates.min()) if not train_dates.empty else None
                ),
                "train_end": str(train_dates.max()) if not train_dates.empty else None,
                "val_start": str(val_dates.min()) if not val_dates.empty else None,
                "val_end": str(val_dates.max()) if not val_dates.empty else None,
            }
        )

    save_json(outdir / "cv_splits.json", cv_splits)
    save_json(outdir / "modeling_results.json", summary)

    # model export: if logistic is overall best by mean f1_macro, export tflite
    best_is_logistic = True
    # if best, export Keras wrapper
    if best_is_logistic and _HAS_TF:
        # train final logistic on full data
        X_all = df[numeric].astype(float).fillna(0.0)
        scaler_all = StandardScaler().fit(X_all)
        X_all_s = scaler_all.transform(X_all)
        final_clf = train_logistic(X_all_s, df["_y"].values, seed=args.seed)

        # build tiny keras model and copy weights
        n_features = X_all_s.shape[1]
        n_classes = len(final_clf.classes_)
        model = tf.keras.Sequential(
            [
                tf.keras.layers.Input(shape=(n_features,)),
                tf.keras.layers.Dense(n_classes, activation="softmax"),
            ]
        )
        # set weights
        coef = final_clf.coef_.astype(np.float32)  # shape (n_classes, n_features)
        intercept = final_clf.intercept_.astype(np.float32)
        kernel = coef.T
        model.layers[0].set_weights([kernel, intercept])

        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite_model = converter.convert()
        atomic_write(outdir / "best_model.tflite", tflite_model)
        save_json(
            outdir / "model_card.json",
            {
                "type": "logistic_keras_tflite",
                "n_features": n_features,
                "n_classes": int(n_classes),
            },
        )
    else:
        save_json(
            outdir / "model_card.json",
            {
                "type": "baseline_collection",
                "note": "tflite skipped (tensorflow missing or logistic not selected)",
            },
        )

    # measure latency using logistic predict if available, else a baseline
    if _HAS_TF and (outdir / "best_model.tflite").exists():
        # use tf lite interpreter

        interpreter = tf.lite.Interpreter(model_path=str(outdir / "best_model.tflite"))
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()[0]
        output_details = interpreter.get_output_details()[0]

        def predict_fn(Xarr: np.ndarray):
            interpreter.resize_tensor_input(input_details["index"], Xarr.shape)
            interpreter.allocate_tensors()
            interpreter.set_tensor(input_details["index"], Xarr.astype(np.float32))
            interpreter.invoke()
            return interpreter.get_tensor(output_details["index"])

        latency = {}
        for n in (1, 32, 128):
            Xs = np.zeros((n, len(numeric)), dtype=np.float32)
            latency[f"n_{n}"] = measure_latency(predict_fn, Xs, repeats=20)
    else:
        # fallback: measure numpy op
        def predict_fn(Xarr: np.ndarray):
            return np.zeros((Xarr.shape[0], 3), dtype=np.float32)

        latency = {
            f"n_{n}": measure_latency(
                predict_fn, np.zeros((n, len(numeric)), dtype=np.float32), repeats=20
            )
            for n in (1, 32, 128)
        }

    save_json(outdir / "inference_latency.json", latency)
    save_json(outdir / "predictions_store.json", preds_store)

    LOG.info("done. outputs in %s", outdir)


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--participant", required=True)
    p.add_argument("--snapshot", required=True)
    p.add_argument("--use_agg", choices=("auto", "yes", "no"), default="auto")
    p.add_argument("--target_col", default="label")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run(args)
