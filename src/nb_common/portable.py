"""Consolidated ML7 utility module.

This file unifies helpers from src/nb_common into a single utility file intended
for easy upload to Kaggle or use as a standalone script. It preserves public
function names and docstrings and is safe to run even if TF/torch are missing.
"""

from pathlib import Path
import os
import json
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# sklearn imports used by metrics/reports
from sklearn.metrics import (
    f1_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    roc_auc_score,
    brier_score_loss,
    classification_report,
    roc_curve,
    auc,
    confusion_matrix,
)
from sklearn.preprocessing import label_binarize


# ======================================================
# MODULE: env
# ======================================================
def detect_env():
    """Detect environment and available backends.

    Returns dict: {is_kaggle, data_root, out_root, backend, tf_available, torch_available}
    """
    is_kaggle = os.path.exists("/kaggle/input") or bool(
        os.environ.get("KAGGLE_KERNEL_RUN_TYPE")
    )
    data_root = "/kaggle/input" if is_kaggle else "./data/ai/local"
    out_root = "/kaggle/working/outputs/ML6" if is_kaggle else "notebooks/outputs/ML6"

    tf_available = False
    torch_available = False
    try:
        import tensorflow as _tf  # noqa

        tf_available = True
    except Exception:
        tf_available = False
    try:
        import torch  # noqa

        torch_available = True
    except Exception:
        torch_available = False

    # backend policy
    if is_kaggle:
        backend = "tf" if tf_available else ("none")
    else:
        if tf_available:
            backend = "tf"
        elif torch_available:
            backend = "torch"
        else:
            backend = "none"

    return {
        "is_kaggle": bool(is_kaggle),
        "data_root": data_root,
        "out_root": out_root,
        "backend": backend,
        "tf_available": tf_available,
        "torch_available": torch_available,
    }


# ======================================================
# MODULE: io
# ======================================================
def resolve_slug_path(slug: str, data_root: str):
    root = Path(data_root)
    p = root / slug / "features_daily_labeled.csv"
    return p


def write_run_config(path: Path, cfg: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf8") as fh:
        json.dump(cfg, fh, indent=2)


# ======================================================
# MODULE: folds
# ======================================================
def build_temporal_folds(
    dates, train_days=120, gap_days=10, val_days=60, max_train_days=240, min_classes=2
):
    ser_dates = pd.to_datetime(dates)
    df = pd.DataFrame({"date": ser_dates}).sort_values("date")
    start, end = df["date"].min(), df["date"].max()
    folds = []
    anchor = start
    while True:
        tr_start = anchor
        tr_end = tr_start + pd.Timedelta(days=train_days - 1)
        te_start = tr_end + pd.Timedelta(days=gap_days)
        te_end = te_start + pd.Timedelta(days=val_days - 1)
        if te_end > end:
            break
        dtr = df[(df["date"] >= tr_start) & (df["date"] <= tr_end)]
        dte = df[(df["date"] >= te_start) & (df["date"] <= te_end)]
        tr_span = train_days
        # if class info not provided here, caller should check class diversity
        while (dtr.empty or dte.empty) and tr_span < max_train_days:
            tr_span += 30
            tr_end = tr_start + pd.Timedelta(days=tr_span - 1)
            te_start = tr_end + pd.Timedelta(days=gap_days)
            te_end = te_start + pd.Timedelta(days=val_days - 1)
            if te_end > end:
                break
            dtr = df[(df["date"] >= tr_start) & (df["date"] <= tr_end)]
            dte = df[(df["date"] >= te_start) & (df["date"] <= te_end)]
        if dtr.empty or dte.empty:
            anchor = anchor + pd.Timedelta(days=30)
            continue
        folds.append(((tr_start, tr_end), (te_start, te_end)))
        anchor = te_end + pd.Timedelta(days=1)
    return folds


# ======================================================
# MODULE: features
# ======================================================
def apply_rolling(df: pd.DataFrame, windows=[7, 14, 28]):
    """Add rolling mean/std and delta features for numeric columns (exclude label/date).
    Rolling is computed on shifted values (past only).
    """
    df2 = df.copy()
    exclude = {"label", "label_source", "label_notes", "date"}
    num_cols = [
        c for c in df2.select_dtypes(include=[np.number]).columns if c not in exclude
    ]
    for w in windows:
        for c in num_cols:
            mname = f"{c}_r{w}_mean"
            sname = f"{c}_r{w}_std"
            dname = f"{c}_r{w}_delta"
            try:
                rolled_mean = (
                    df2[c].shift(1).rolling(window=w, min_periods=1).mean().fillna(0)
                )
                rolled_std = (
                    df2[c].shift(1).rolling(window=w, min_periods=1).std().fillna(0)
                )
                df2[mname] = rolled_mean
                df2[sname] = rolled_std
                df2[dname] = (rolled_mean - rolled_mean.shift(1)).fillna(0)
            except Exception:
                df2[mname] = 0
                df2[sname] = 0
                df2[dname] = 0
    return df2


def add_deltas(df: pd.DataFrame, windows=[7, 14, 28]):
    # for compatibility: deltas already added in apply_rolling; provide noop wrapper
    return df


def zscore_by_segment(df: pd.DataFrame, segment_col="segment_id"):
    # group-wise zscore for numeric columns; noop if segment missing
    if segment_col not in df.columns:
        return df
    df2 = df.copy()
    num_cols = df2.select_dtypes(include=[np.number]).columns.tolist()
    for name, g in df2.groupby(segment_col):
        means = g[num_cols].mean()
        stds = g[num_cols].std().replace(0, 1)
        df2.loc[g.index, num_cols] = (g[num_cols] - means) / stds
    return df2


# ======================================================
# MODULE: metrics
# ======================================================
def eval_metrics(y_true, y_pred, proba=None, classes=None):
    res = {
        "f1_macro": np.nan,
        "f1_weighted": np.nan,
        "balanced_acc": np.nan,
        "kappa": np.nan,
        "auroc_ovr_macro": np.nan,
        "brier_mean_ovr": np.nan,
    }
    label_list = classes if classes is not None else None
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="A single label was found in 'y_true' and .*"
        )
        try:
            if label_list is not None:
                res["f1_macro"] = float(
                    f1_score(
                        y_true,
                        y_pred,
                        average="macro",
                        labels=label_list,
                        zero_division=0,
                    )
                )
                res["f1_weighted"] = float(
                    f1_score(
                        y_true,
                        y_pred,
                        average="weighted",
                        labels=label_list,
                        zero_division=0,
                    )
                )
            else:
                res["f1_macro"] = float(
                    f1_score(y_true, y_pred, average="macro", zero_division=0)
                )
                res["f1_weighted"] = float(
                    f1_score(y_true, y_pred, average="weighted", zero_division=0)
                )
        except Exception:
            res["f1_macro"] = np.nan
            res["f1_weighted"] = np.nan
        try:
            res["balanced_acc"] = float(balanced_accuracy_score(y_true, y_pred))
        except Exception:
            res["balanced_acc"] = np.nan
        try:
            res["kappa"] = float(cohen_kappa_score(y_true, y_pred))
        except Exception:
            res["kappa"] = np.nan

    # AUROC guard
    if proba is not None and classes is not None and len(classes) > 1:
        try:
            y_true_bin = label_binarize(y_true, classes=classes)
            if proba.ndim == 1:
                proba = np.vstack([1 - proba, proba]).T
            pred_from_proba = np.argmax(proba, axis=1)
            true_classes = set(np.unique(y_true))
            pred_classes = set(pred_from_proba)
            if len(true_classes) >= 2 and true_classes.issubset(pred_classes):
                res["auroc_ovr_macro"] = float(
                    roc_auc_score(y_true_bin, proba, average="macro", multi_class="ovr")
                )
            else:
                res["auroc_ovr_macro"] = np.nan
            try:
                briers = [
                    brier_score_loss(y_true_bin[:, c], proba[:, c])
                    for c in range(len(classes))
                ]
                res["brier_mean_ovr"] = float(np.mean(briers))
            except Exception:
                res["brier_mean_ovr"] = np.nan
        except Exception:
            res["auroc_ovr_macro"] = np.nan
            res["brier_mean_ovr"] = np.nan
    return res


def safe_class_report(y_true, y_pred):
    try:
        labels = sorted(list(np.unique(y_true)))
        return classification_report(
            y_true, y_pred, labels=labels, zero_division=0, output_dict=True
        )
    except Exception:
        return None


# ======================================================
# MODULE: reports
# ======================================================
def save_class_report_csv(report_dict, path: Path):
    # report_dict from sklearn classification_report (output_dict=True)
    rows = []
    if report_dict is None:
        return None
    for k, v in report_dict.items():
        if k in ("accuracy", "macro avg", "weighted avg"):
            continue
        rows.append({"class": k, **v})
    df = pd.DataFrame(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def save_confmat_png(y_true, y_pred, labels, path: Path):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.set_title("Confusion matrix")
    fig.colorbar(im, ax=ax)
    tick_marks = np.arange(len(labels))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(labels, rotation=45)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(labels)
    for (x, y), val in np.ndenumerate(cm):
        ax.text(y, x, int(val), ha="center", va="center", color="black")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def save_roc_ovr_png(y_true_bin, proba, path: Path):
    # y_true_bin: binarized labels [N, C], proba: [N, C]
    n_classes = y_true_bin.shape[1]
    fig, ax = plt.subplots(figsize=(6, 4))
    for c in range(n_classes):
        fpr, tpr, _ = roc_curve(y_true_bin[:, c], proba[:, c])
        ax.plot(fpr, tpr, label=f"Class {c} (AUC={auc(fpr,tpr):.2f})")
    ax.plot([0, 1], [0, 1], "k--")
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.set_title("ROC OvR")
    ax.legend(loc="lower right")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# ======================================================
# MODULE: tf_models
# ======================================================
def build_lstm(seq_len, n_feats, n_classes, hidden=64, dropout=0.3):
    try:
        from tensorflow import keras
    except Exception:
        raise
    model = keras.Sequential(
        [
            keras.layers.Input(shape=(seq_len, n_feats)),
            keras.layers.LSTM(hidden),
            keras.layers.Dropout(dropout),
            keras.layers.Dense(max(32, hidden // 2), activation="relu"),
            keras.layers.Dropout(max(0.1, dropout / 2)),
            keras.layers.Dense(n_classes, activation="softmax"),
        ]
    )
    return model


def build_cnn1d(seq_len, n_feats, n_classes, filters=64, kernel_size=3, dropout=0.2):
    try:
        from tensorflow import keras
    except Exception:
        raise
    inp = keras.layers.Input(shape=(seq_len, n_feats))
    x = keras.layers.Conv1D(filters, kernel_size, activation="relu", padding="same")(
        inp
    )
    x = keras.layers.GlobalMaxPool1D()(x)
    x = keras.layers.Dense(64, activation="relu")(x)
    x = keras.layers.Dropout(dropout)(x)
    out = keras.layers.Dense(n_classes, activation="softmax")(x)
    model = keras.Model(inputs=inp, outputs=out)
    return model


def build_cnn_bilstm(
    seq_len, n_feats, n_classes, filters=64, kernel_size=3, hidden=64, dropout=0.3
):
    try:
        from tensorflow import keras
    except Exception:
        raise
    inp = keras.layers.Input(shape=(seq_len, n_feats))
    x = keras.layers.Conv1D(filters, kernel_size, activation="relu", padding="same")(
        inp
    )
    x = keras.layers.Bidirectional(keras.layers.LSTM(hidden, return_sequences=False))(x)
    x = keras.layers.Dense(64, activation="relu")(x)
    x = keras.layers.Dropout(dropout)(x)
    out = keras.layers.Dense(n_classes, activation="softmax")(x)
    return keras.Model(inputs=inp, outputs=out)


def build_transformer_tiny(
    seq_len,
    n_feats,
    n_classes,
    head_size=32,
    num_heads=2,
    ff_dim=64,
    num_transformer_blocks=1,
    dropout=0.1,
):
    try:
        from tensorflow import keras
    except Exception:
        raise
    inp = keras.layers.Input(shape=(seq_len, n_feats))
    x = inp
    for _ in range(num_transformer_blocks):
        attn = keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=head_size)(
            x, x
        )
        x = keras.layers.Add()([x, attn])
        x = keras.layers.LayerNormalization()(x)
        ff = keras.layers.Dense(ff_dim, activation="relu")(x)
        x = keras.layers.Add()([x, ff])
        x = keras.layers.LayerNormalization()(x)
    x = keras.layers.GlobalAveragePooling1D()(x)
    x = keras.layers.Dense(64, activation="relu")(x)
    out = keras.layers.Dense(n_classes, activation="softmax")(x)
    return keras.Model(inputs=inp, outputs=out)


def export_tflite(model, out_path):
    try:
        import tensorflow as tf
    except Exception:
        raise
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as fh:
        fh.write(tflite_model)
    return out_path


if __name__ == "__main__":
    env = detect_env()
    print("Environment:", env)
    print(
        "Modules available:",
        [
            f.__name__
            for f in [
                detect_env,
                resolve_slug_path,
                build_temporal_folds,
                apply_rolling,
                eval_metrics,
                build_lstm,
            ]
        ],
    )
