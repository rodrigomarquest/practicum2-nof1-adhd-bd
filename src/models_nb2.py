"""NB2 — Baseline models and LSTM scaffold

This script is runnable as a script. It computes simple baselines and writes
results and plots to timestamped output folders under `notebooks/outputs/NB2/<ts>/`.

Usage:
  python notebooks/NB2_Baseline_and_LSTM.py --features /path/to/features_daily_labeled.csv [--dry-run]

The script now strictly honors the --features path and performs schema checks.
"""

from pathlib import Path
from datetime import datetime, timezone
import argparse
import sys
import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import json
import fnmatch

from sklearn.metrics import (
    f1_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    roc_auc_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
)
from sklearn.preprocessing import label_binarize, StandardScaler
from sklearn.linear_model import LogisticRegression


def in_notebook():
    try:
        from IPython import get_ipython

        ip = get_ipython()
        return ip is not None and hasattr(ip, "kernel")
    except Exception:
        return False


RUN_TS = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
# FIG_DIR / TAB_DIR will be created by prepare_output_dirs(out_root) from main()
OUT_ROOT = None
FIG_DIR = None
TAB_DIR = None
MOD_DIR = None

# Deterministic seeds at module import (also set in main() to be sure)
random.seed(42)
np.random.seed(42)
os.environ.setdefault("PYTHONHASHSEED", "42")

# Control whether LSTM/backends are attempted. Set in main() based on kaggle-mode or runtime flags.
ENABLE_LSTM = False
# Environment / backend globals (populated by detect_env())
IS_KAGGLE = False
DEFAULT_LOCAL_ROOT = "./data/ai/local"
DEFAULT_OUT_ROOT = "notebooks/outputs/NB2"
BACKEND = "none"  # one of 'tf','torch','none'
TF_AVAILABLE = False
TORCH_AVAILABLE = False
LSTM_SEQ_LEN = 7
USE_CLASS_WEIGHT = True
CURRENT_SLUG = None


def prepare_output_dirs(out_root: Path):
    """Create and set global output directories for tables/figures/models."""
    global OUT_ROOT, FIG_DIR, TAB_DIR, MOD_DIR
    OUT_ROOT = Path(out_root)
    FIG_DIR = OUT_ROOT / "figures"
    TAB_DIR = OUT_ROOT / "tables"
    MOD_DIR = OUT_ROOT / "models"
    for d in (FIG_DIR, TAB_DIR, MOD_DIR):
        d.mkdir(parents=True, exist_ok=True)


def detect_env():
    """Detect running environment (Kaggle vs local) and available backends.

    Returns a dict with keys: is_kaggle, data_root, out_root, backend, enable_lstm
    """
    global IS_KAGGLE, DEFAULT_LOCAL_ROOT, DEFAULT_OUT_ROOT, BACKEND, TF_AVAILABLE, TORCH_AVAILABLE
    is_kaggle = os.path.exists("/kaggle/input") or bool(
        os.environ.get("KAGGLE_KERNEL_RUN_TYPE")
    )
    IS_KAGGLE = bool(is_kaggle)
    DEFAULT_LOCAL_ROOT = "/kaggle/input" if IS_KAGGLE else "./data/ai/local"
    DEFAULT_OUT_ROOT = (
        "/kaggle/working/outputs/NB2" if IS_KAGGLE else "notebooks/outputs/NB2"
    )

    # detect tf/torch availability
    try:
        import tensorflow as _tf  # noqa

        TF_AVAILABLE = True
    except Exception:
        TF_AVAILABLE = False
    try:
        import torch  # noqa

        TORCH_AVAILABLE = True
    except Exception:
        TORCH_AVAILABLE = False

    # backend policy
    if IS_KAGGLE:
        # prefer TF on Kaggle; if not present, fall back to none (do not use torch)
        if TF_AVAILABLE:
            BACKEND = "tf"
            enable_lstm = True
        else:
            BACKEND = "none"
            enable_lstm = False
    else:
        # local: prefer TF if present, else torch, else none
        if TF_AVAILABLE:
            BACKEND = "tf"
            enable_lstm = True
        elif TORCH_AVAILABLE:
            BACKEND = "torch"
            enable_lstm = True
        else:
            BACKEND = "none"
            enable_lstm = False

    print(
        "ENV: {} | DATA_ROOT={} | OUT_ROOT={} | ENABLE_LSTM={} | BACKEND={}".format(
            ("kaggle" if IS_KAGGLE else "local"),
            DEFAULT_LOCAL_ROOT,
            DEFAULT_OUT_ROOT,
            enable_lstm,
            BACKEND,
        )
    )
    return {
        "is_kaggle": IS_KAGGLE,
        "data_root": DEFAULT_LOCAL_ROOT,
        "out_root": DEFAULT_OUT_ROOT,
        "backend": BACKEND,
        "enable_lstm": enable_lstm,
        "tf_available": TF_AVAILABLE,
        "torch_available": TORCH_AVAILABLE,
    }


def discover_nb2_datasets(root: str):
    """Discover datasets under a root following slug pattern: p000001-s20250929-nb2v303-r1

    Returns list of dicts with keys: slug, participant, snapshot, nbver, run, features, version_log, root
    """
    items = []
    root_path = Path(root)
    if not root_path.exists():
        return items
    for d in sorted(root_path.iterdir()):
        if not d.is_dir():
            continue
        slug = d.name
        parts = slug.split("-")
        if len(parts) < 3:
            continue
        participant = parts[0]
        snapshot = parts[1]
        nbver = parts[2]
        run = parts[3] if len(parts) > 3 else "r1"
        features = d / "features_daily_labeled.csv"
        vlog = d / "version_log_enriched.csv"
        if features.exists():
            items.append(
                {
                    "slug": slug,
                    "participant": participant,
                    "snapshot": snapshot,
                    "nbver": nbver,
                    "run": run,
                    "features": features,
                    "version_log": vlog if vlog.exists() else None,
                    "root": d,
                }
            )
    return items


def select_datasets(items, f_part=None, f_snap=None, limit=None):
    def ok(pat, s):
        return (pat is None) or fnmatch.fnmatch(s, pat)

    sel = [
        it
        for it in items
        if ok(f_part, it["participant"]) and ok(f_snap, it["snapshot"])
    ]
    sel.sort(key=lambda x: x["snapshot"])  # sort by snapshot
    if limit:
        return sel[-limit:]
    return sel


def savefig(fname: str):
    p = FIG_DIR / fname
    plt.tight_layout()
    plt.savefig(p, dpi=150)
    print("INFO: wrote figure ->", p)
    return p


def get_temporal_folds(
    dates: pd.Series, train_days=120, test_days=60, gap_days=10, n_folds=6
):
    """Return list of (tr_mask, te_mask, (tr_start,tr_end,te_start,te_end)).

    Masks are boolean arrays aligned with the input dates Series.
    """
    dmin = dates.min()
    folds = []
    anchor = dmin
    for k in range(n_folds):
        tr_start = anchor
        tr_end = tr_start + pd.Timedelta(days=train_days - 1)
        te_start = tr_end + pd.Timedelta(days=gap_days + 1)
        te_end = te_start + pd.Timedelta(days=test_days - 1)
        tr_mask = dates.between(tr_start, tr_end)
        te_mask = dates.between(te_start, te_end)
        if tr_mask.sum() == 0 or te_mask.sum() == 0:
            break
        folds.append(
            (tr_mask.values, te_mask.values, (tr_start, tr_end, te_start, te_end))
        )
        # advance anchor by test period (non-overlapping)
        anchor = te_start
    return folds


# --- CV builder: garante >=2 classes em treino e validacao ---
def build_temporal_folds(
    dates,
    y,
    train_days=120,
    val_days=60,
    gap_days=10,
    max_train_days=240,
    min_classes=2,
):
    import pandas as pd

    ser_dates = pd.to_datetime(dates)
    df = pd.DataFrame({"date": ser_dates, "y": y}).sort_values("date")
    start, end = df["date"].min(), df["date"].max()
    folds, anchor = [], start
    while True:
        tr_start = anchor
        tr_end = tr_start + pd.Timedelta(days=train_days - 1)
        te_start = tr_end + pd.Timedelta(days=gap_days)
        te_end = te_start + pd.Timedelta(days=val_days - 1)
        if te_end > end:
            break
        # janelas iniciais
        dtr = df[(df["date"] >= tr_start) & (df["date"] <= tr_end)]
        dte = df[(df["date"] >= te_start) & (df["date"] <= te_end)]
        # expandir treino ate max_train_days se faltar diversidade
        tr_span = train_days
        while (
            dtr["y"].nunique() < min_classes or dte["y"].nunique() < min_classes
        ) and tr_span < max_train_days:
            tr_span += 30
            tr_end = tr_start + pd.Timedelta(days=tr_span - 1)
            te_start = tr_end + pd.Timedelta(days=gap_days)
            te_end = te_start + pd.Timedelta(days=val_days - 1)
            if te_end > end:
                break
            dtr = df[(df["date"] >= tr_start) & (df["date"] <= tr_end)]
            dte = df[(df["date"] >= te_start) & (df["date"] <= te_end)]
        # se ainda nao deu, avanca 1 mes e tenta de novo
        if dtr["y"].nunique() < min_classes or dte["y"].nunique() < min_classes:
            anchor = anchor + pd.Timedelta(days=30)
            continue
        print(
            f"CV fold: train {tr_start.date()}..{tr_end.date()} (classes={dtr['y'].nunique()}), "
            f"val {te_start.date()}..{te_end.date()} (classes={dte['y'].nunique()})"
        )
        folds.append(((tr_start, tr_end), (te_start, te_end)))
        anchor = te_end + pd.Timedelta(days=1)
    return folds


def eval_metrics(y_true, y_pred, proba=None, classes=None):
    """Compute a set of metrics in a NaN-safe way and avoid sklearn warnings

    - If only a single class is present in y_true (or classes has length 1),
      metrics that require multiple classes (AUROC / Brier) will be set to np.nan.
    - Suppress the sklearn/UserWarning emitted when confusion matrix has a
      single label by catching and filtering that specific warning.
    """
    res = {
        "f1_macro": np.nan,
        "f1_weighted": np.nan,
        "balanced_acc": np.nan,
        "kappa": np.nan,
        "auroc_ovr_macro": np.nan,
        "brier_mean_ovr": np.nan,
    }

    # Use the full set of known classes if provided (helps keep confusion shapes stable)
    label_list = classes if classes is not None else None

    # Compute discrete metrics while suppressing the sklearn single-label warning
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="A single label was found in 'y_true' and .*"
        )
        # suppress runtime warnings from sklearn internals when confusion matrices are degenerate
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            # use zero_division=0 to avoid exceptions when a label has zero support
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
            # capture classification report (including per-class support) for optional diagnostics
            try:
                labels_for_report = (
                    label_list if label_list is not None else sorted(np.unique(y_true))
                )
                report = classification_report(
                    y_true,
                    y_pred,
                    labels=labels_for_report,
                    zero_division=0,
                    output_dict=True,
                )
            except Exception:
                report = None
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

    # Probabilistic metrics: AUROC / Brier
    if proba is not None and classes is not None and len(classes) > 1:
        try:
            # Binarize true labels to compute OvR AUROC
            y_true_bin = label_binarize(y_true, classes=classes)
            # Ensure proba shape matches
            if proba.ndim == 1:
                proba = np.vstack([1 - proba, proba]).T
            # Guard: only compute AUROC if model predictions cover the true classes
            try:
                pred_from_proba = np.argmax(proba, axis=1)
                true_classes = set(np.unique(y_true))
                pred_classes = set(pred_from_proba)
                if len(true_classes) >= 2 and true_classes.issubset(pred_classes):
                    with warnings.catch_warnings():
                        # suppress any warnings coming from sklearn internals for degenerate folds
                        warnings.filterwarnings("ignore", message=".*")
                        res["auroc_ovr_macro"] = float(
                            roc_auc_score(
                                y_true_bin, proba, average="macro", multi_class="ovr"
                            )
                        )
                else:
                    res["auroc_ovr_macro"] = np.nan
            except Exception:
                res["auroc_ovr_macro"] = np.nan
            # Brier mean OvR
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


def naive_persistence(y_series: pd.Series):
    # previous label -> current; fill start with most frequent label
    out = y_series.shift(1)
    out.iloc[0] = (
        y_series.mode().iloc[0] if not y_series.mode().empty else y_series.iloc[0]
    )
    return out


def moving_avg_label(y_series: pd.Series, window=7):
    # majority vote over previous 'window' days (exclude current)
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


def run_baselines(df: pd.DataFrame, features_path: str = None):
    print("INFO: Preparing data")

    # Basic checks
    if "date" not in df.columns:
        print("ERROR: 'date' column not found in dataset")
        raise SystemExit(2)
    if "label" not in df.columns:
        print(
            "ERROR: 'label' column not found in the dataset; run 'make etl-labels' first."
        )
        raise SystemExit(2)

    # ensure date parsed
    if not np.issubdtype(df["date"].dtype, np.datetime64):
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    if df["date"].isna().any():
        print("ERROR: some 'date' values could not be parsed")
        raise SystemExit(2)

    df = df.sort_values("date").reset_index(drop=True)

    labels = df["label"].astype(str)
    classes = sorted(labels.unique())

    # Numeric feature selection (exclude label-like and date)
    exclude = {"label", "label_source", "label_notes", "date"}
    numeric_cols = [
        c for c in df.select_dtypes(include=[np.number]).columns if c not in exclude
    ]
    if len(numeric_cols) == 0:
        print("ERROR: no numeric features found after exclusions; cannot run baselines")
        raise SystemExit(2)

    # forward-fill missing numeric values (use .ffill() to avoid FutureWarning)
    Xnum = df[numeric_cols].ffill().fillna(0)
    scaler = StandardScaler()
    Xnum_scaled = pd.DataFrame(
        scaler.fit_transform(Xnum), columns=Xnum.columns, index=Xnum.index
    )

    # Build temporal folds with class-diversity safety (may expand train window)
    raw_folds = build_temporal_folds(
        df["date"],
        labels,
        train_days=120,
        val_days=60,
        gap_days=10,
        max_train_days=240,
        min_classes=2,
    )
    # convert to masks compatible with existing loop expectations
    folds = []
    for (tr_start, tr_end), (te_start, te_end) in raw_folds:
        tr_mask = df["date"].between(tr_start, tr_end)
        te_mask = df["date"].between(te_start, te_end)
        folds.append(
            (tr_mask.values, te_mask.values, (tr_start, tr_end, te_start, te_end))
        )
    print(
        f"INFO: folds -> built {len(folds)} folds (train initial=120d, test initial=60d, gap=10d)"
    )

    rows = []
    # prepare structures to collect per-fold class supports
    from collections import defaultdict

    class_support_totals = defaultdict(int)
    class_support_frames = []
    # counters
    n_train_total = 0
    n_val_total = 0
    for i, (tr_mask, te_mask, windows) in enumerate(folds, start=1):
        tr_idx = np.where(tr_mask)[0]
        te_idx = np.where(te_mask)[0]
        if len(tr_idx) == 0 or len(te_idx) == 0:
            continue

        print(
            f"INFO: Fold {i}: train {windows[0].date()}..{windows[1].date()} test {windows[2].date()}..{windows[3].date()}"
        )

        y_tr = labels.iloc[tr_idx]
        y_te = labels.iloc[te_idx]
        n_train_total += int(len(tr_idx))
        n_val_total += int(len(te_idx))

        # Naive persistence
        y_pred_naive = naive_persistence(labels).iloc[te_idx]
        # compute metrics and capture classification report for naive predictor
        metrics_naive = eval_metrics(y_te, y_pred_naive, classes=classes)
        rows.append({"model": "naive", "fold": i, **metrics_naive})

        # save per-fold class support from classification_report (using naive predictions as reference)
        try:
            crep = classification_report(
                y_te, y_pred_naive, labels=classes, output_dict=True, zero_division=0
            )
            # collect rows for classes only
            cs_rows = []
            for k, v in crep.items():
                if k in ("accuracy", "macro avg", "weighted avg"):
                    continue
                # v may contain precision, recall, f1-score, support
                prec = v.get("precision", "")
                rec = v.get("recall", "")
                f1 = v.get("f1-score", "")
                supp = int(v.get("support", 0))
                cs_rows.append(
                    {
                        "class": k,
                        "precision": prec,
                        "recall": rec,
                        "f1-score": f1,
                        "support": supp,
                    }
                )
                class_support_totals[k] += supp
            # write per-fold CSV
            try:
                cs_df = pd.DataFrame(cs_rows)
                cs_path = TAB_DIR / f"class_support_fold_{i}.csv"
                cs_df.to_csv(cs_path, index=False)
                print("INFO: wrote class support per-fold ->", cs_path)
            except Exception as e:
                print("WARNING: failed to write class support per-fold ->", e)
        except Exception:
            pass

        # Moving-average majority (window=7)
        y_pred_ma7 = moving_avg_label(labels, window=7).iloc[te_idx]
        rows.append(
            {
                "model": "ma7",
                "fold": i,
                **eval_metrics(y_te, y_pred_ma7, classes=classes),
            }
        )

        # Rule-based (if features exist)
        y_pred_rule = pd.Series("neutral", index=labels.index)
        if "apple_hr_mean" in df.columns and "sleep_total_min" in df.columns:
            hr_med = df["apple_hr_mean"].median()
            sleep_med = df["sleep_total_min"].median()
            hr_hi = df["apple_hr_mean"].fillna(hr_med) > hr_med
            low_sleep = df["sleep_total_min"].fillna(sleep_med) < sleep_med
            rule_idx = hr_hi & low_sleep
            y_pred_rule[rule_idx] = "negative"
        rows.append(
            {
                "model": "rule_based",
                "fold": i,
                **eval_metrics(y_te, y_pred_rule.iloc[te_idx], classes=classes),
            }
        )

        # Logistic Regression
        if Xnum_scaled.shape[1] > 0:
            # skip training if only a single class is present in the training split
            if labels.iloc[tr_idx].nunique() < 2:
                print(
                    f"WARNING: fold {i} has only a single class in training data; skipping logistic regression"
                )
            else:
                try:
                    # respect use_class_weight if provided via global variable (defaulting to True for backward compatibility)
                    cweight = globals().get("USE_CLASS_WEIGHT", True)
                    clf = LogisticRegression(
                        max_iter=400,
                        class_weight=("balanced" if cweight else None),
                        multi_class="multinomial",
                        random_state=42,
                    )
                    clf.fit(Xnum_scaled.iloc[tr_idx], labels.iloc[tr_idx])
                    pred = clf.predict(Xnum_scaled.iloc[te_idx])
                    proba = clf.predict_proba(Xnum_scaled.iloc[te_idx])
                    rows.append(
                        {
                            "model": "logreg",
                            "fold": i,
                            **eval_metrics(
                                labels.iloc[te_idx], pred, proba, classes=classes
                            ),
                        }
                    )
                except Exception as e:
                    print(
                        f"WARNING: logistic regression failed on fold {i}: {e}; skipping"
                    )
        else:
            print("WARNING: Skipping logistic regression for fold", i)

        # determine a prediction to use for confusion matrix plotting (prefer logreg pred if available)
        try:
            pred_for_conf = None
            if "pred" in locals():
                pred_for_conf = pred
            else:
                pred_for_conf = y_pred_naive
            # compute and save confusion matrix
            try:
                cm = confusion_matrix(y_te, pred_for_conf, labels=classes)
                fig = plt.figure(figsize=(4, 4))
                plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
                plt.title(f"Confusion matrix - fold {i}")
                plt.colorbar()
                tick_marks = np.arange(len(classes))
                plt.xticks(tick_marks, classes, rotation=45)
                plt.yticks(tick_marks, classes)
                plt.ylabel("True label")
                plt.xlabel("Predicted label")
                for (x, y), val in np.ndenumerate(cm):
                    plt.text(y, x, int(val), ha="center", va="center", color="black")
                fname = f"confmat_fold_{i}.png"
                fig_path = savefig(fname)
                plt.close(fig)
            except Exception:
                pass
        except Exception:
            pass

    baseline_df = pd.DataFrame(rows)

    # write per-fold table
    per_fold_path = TAB_DIR / "nb2_baseline_metrics_per_fold.csv"
    try:
        baseline_df.to_csv(per_fold_path, index=False)
        print("INFO: wrote per-fold metrics ->", per_fold_path)
    except Exception as e:
        print("WARNING: failed to write per-fold metrics ->", per_fold_path, e)

    # Aggregate metrics across folds: mean and std per model
    metrics = [
        "f1_macro",
        "f1_weighted",
        "balanced_acc",
        "kappa",
        "auroc_ovr_macro",
        "brier_mean_ovr",
    ]
    if baseline_df.empty:
        print(
            "WARNING: No baseline rows were produced (no folds?). Writing empty metrics table."
        )
        agg_df = pd.DataFrame(
            columns=["model"]
            + [f"{m}_mean" for m in metrics]
            + [f"{m}_std" for m in metrics]
            + ["fold_count"]
        )
    else:
        agg = baseline_df.groupby("model")[metrics].agg(["mean", "std"])
        agg.columns = [f"{col[0]}_{col[1]}" for col in agg.columns]
        counts = baseline_df.groupby("model").size().rename("fold_count")
        agg_df = agg.join(counts).reset_index()

    # write aggregated metrics
    baseline_path = TAB_DIR / "nb2_baseline_metrics.csv"
    try:
        agg_df.to_csv(baseline_path, index=False)
    except Exception as e:
        print("ERROR: failed to write metrics ->", baseline_path, e)
        raise
    print("INFO: wrote metrics ->", baseline_path)

    # write aggregated class support summary
    try:
        if class_support_totals:
            summary_df = pd.DataFrame(
                [
                    {"class": k, "total_support": v}
                    for k, v in class_support_totals.items()
                ]
            )
            summary_path = TAB_DIR / "class_support_summary.csv"
            summary_df.to_csv(summary_path, index=False)
            print("INFO: wrote class support summary ->", summary_path)
    except Exception as e:
        print("WARNING: failed to write class support summary ->", e)

    # --- LSTM phase (dual backend) ---------------------------------------------
    # If TensorFlow is available, attempt the Keras path; otherwise use PyTorch fallback

    def run_lstm_fallback_pytorch(
        X_train,
        y_train,
        X_val,
        y_val,
        out_dir,
        seed=42,
        seq_len=7,
        hidden=64,
        epochs=200,
        patience=10,
        batch_size=64,
    ):
        import os
        import time
        import numpy as np

        try:
            import torch
            from torch import nn
        except Exception:
            raise
        torch.manual_seed(seed)

        # Expect X_* as 2D arrays [N, F]. Build rolling windows of length seq_len.
        def build_windows(X, y, sl):
            Xw, yw = [], []
            for i in range(sl - 1, len(X)):
                Xw.append(X[i - sl + 1 : i + 1])
                yw.append(y[i])
            return np.array(Xw, dtype=np.float32), np.array(yw, dtype=np.int64)

        Xtr_w, ytr_w = build_windows(X_train, y_train, seq_len)
        Xva_w, yva_w = build_windows(X_val, y_val, seq_len)
        if len(Xtr_w) == 0 or len(Xva_w) == 0:
            return None, {"note": "not_enough_sequences"}

        n_classes = int(len(np.unique(y_train)))
        n_feats = int(Xtr_w.shape[-1])

        class LSTMClf(nn.Module):
            def __init__(self, f, h, c):
                super().__init__()
                self.lstm = nn.LSTM(input_size=f, hidden_size=h, batch_first=True)
                self.head = nn.Sequential(nn.Flatten(), nn.Linear(h * seq_len, c))

            def forward(self, x):
                out, _ = self.lstm(x)
                return self.head(out)

        model = LSTMClf(n_feats, hidden, n_classes)
        device = torch.device("cpu")
        model.to(device)

        train_ds = torch.utils.data.TensorDataset(
            torch.from_numpy(Xtr_w), torch.from_numpy(ytr_w)
        )
        val_ds = torch.utils.data.TensorDataset(
            torch.from_numpy(Xva_w), torch.from_numpy(yva_w)
        )
        train_ld = torch.utils.data.DataLoader(
            train_ds, batch_size=batch_size, shuffle=True
        )
        val_ld = torch.utils.data.DataLoader(
            val_ds, batch_size=batch_size, shuffle=False
        )

        crit = nn.CrossEntropyLoss()
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)

        best_f1, best_state, wait = -1.0, None, 0
        from sklearn.metrics import f1_score

        for ep in range(1, epochs + 1):
            model.train()
            for xb, yb in train_ld:
                xb, yb = xb.to(device), yb.to(device)
                opt.zero_grad()
                logits = model(xb)
                loss = crit(logits, yb)
                loss.backward()
                opt.step()
            # val
            model.eval()
            with torch.no_grad():
                preds, ys = [], []
                for xb, yb in val_ld:
                    logits = model(xb.to(device))
                    preds.append(torch.argmax(logits, dim=1).cpu().numpy())
                    ys.append(yb.numpy())
            y_pred = np.concatenate(preds)
            y_true = np.concatenate(ys)
            f1m = f1_score(y_true, y_pred, average="macro", zero_division=0)
            if f1m > best_f1:
                best_f1, wait = f1m, 0
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            else:
                wait += 1
            if wait >= patience:
                break

        if best_state is None:
            return None, {"note": "no_improvement"}
        model.load_state_dict(best_state)

        # save .pt
        os.makedirs(os.path.join(out_dir, "models"), exist_ok=True)
        pt_path = os.path.join(out_dir, "models", "lstm_fold_pytorch.pt")
        torch.save(model.state_dict(), pt_path)

        # simple latency on val
        t0 = time.perf_counter()
        with torch.no_grad():
            for xb, _ in val_ld:
                _ = model(xb.to(device))
        t1 = time.perf_counter()
        per_sample_ms = 1000.0 * (t1 - t0) / max(1, len(val_ds))

        # optional: export ONNX (best-effort)
        try:
            onnx_path = os.path.join(out_dir, "models", "lstm_fold_pytorch.onnx")
            dummy = torch.from_numpy(Xva_w[:1])
            torch.onnx.export(
                model,
                dummy,
                onnx_path,
                input_names=["x"],
                output_names=["logits"],
                opset_version=12,
            )
        except Exception:
            pass

        return {"pt": pt_path, "latency_ms_per_sample": per_sample_ms}, {
            "best_f1_macro": float(best_f1)
        }

    def try_lstm_any_backend(
        X_train, y_train, X_val, y_val, out_dir, seed=42, seq_len=7
    ):
        # Respect global BACKEND preference and availability
        global BACKEND
        if BACKEND == "none":
            return None, {"note": "lstm_disabled"}
        if BACKEND == "tf":
            # attempt TF only; do not fallback to torch on Kaggle
            try:
                import tensorflow as tf  # noqa
                from tensorflow import keras

                tf.get_logger().setLevel("ERROR")
                import numpy as _np

                def build_windows(X, y, sl):
                    Xw, yw = [], []
                    for i in range(sl - 1, len(X)):
                        Xw.append(X[i - sl + 1 : i + 1])
                        yw.append(y[i])
                    return _np.array(Xw, dtype=_np.float32), _np.array(
                        yw, dtype=_np.int64
                    )

                Xtr_w, ytr_w = build_windows(X_train, y_train, seq_len)
                Xva_w, yva_w = build_windows(X_val, y_val, seq_len)
                if len(Xtr_w) == 0 or len(Xva_w) == 0:
                    return None, {"note": "not_enough_sequences"}
                n_feats = Xtr_w.shape[-1]
                n_classes = int(len(_np.unique(y_train)))
                keras.backend.clear_session()
                model = keras.Sequential(
                    [
                        keras.layers.Input(shape=(seq_len, n_feats)),
                        keras.layers.LSTM(64),
                        keras.layers.Dropout(0.3),
                        keras.layers.Dense(32, activation="relu"),
                        keras.layers.Dropout(0.2),
                        keras.layers.Dense(n_classes, activation="softmax"),
                    ]
                )
                model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")
                cb = [
                    keras.callbacks.EarlyStopping(
                        monitor="val_loss",
                        patience=10,
                        restore_best_weights=True,
                        verbose=0,
                    )
                ]
                try:
                    model.fit(
                        Xtr_w,
                        ytr_w,
                        validation_data=(Xva_w, yva_w),
                        epochs=100,
                        batch_size=32,
                        callbacks=cb,
                        verbose=0,
                    )
                    preds = model.predict(Xva_w, verbose=0)
                    y_pred = _np.argmax(preds, axis=1)
                    from sklearn.metrics import f1_score as _f1

                    f1m = float(_f1(yva_w, y_pred, average="macro", zero_division=0))
                    # save Keras model
                    try:
                        mpath = MOD_DIR / "lstm_fold_keras.h5"
                        model.save(mpath)
                    except Exception:
                        mpath = None
                    # latency
                    import time as _time

                    t0 = _time.time()
                    _ = model.predict(Xva_w[: min(32, len(Xva_w))], verbose=0)
                    t1 = _time.time()
                    lat = ((t1 - t0) * 1000.0) / max(1, min(32, len(Xva_w)))
                    return (
                        {
                            "keras_model": str(mpath) if mpath is not None else None,
                            "latency_ms_per_sample": lat,
                        },
                        {"best_f1_macro": f1m},
                    )
                except Exception:
                    return None, {"note": "keras_failed"}
            except Exception:
                return None, {"note": "tf_not_available"}
        # BACKEND == 'torch' -> use PyTorch fallback only
        return run_lstm_fallback_pytorch(
            X_train, y_train, X_val, y_val, out_dir, seed=seed, seq_len=seq_len
        )

    # run LSTM per fold (on windows); include results in best-model selection
    lstm_fold_results = []
    for i, (tr_mask, te_mask, windows) in enumerate(folds, start=1):
        tr_idx = np.where(tr_mask)[0]
        te_idx = np.where(te_mask)[0]
        if len(tr_idx) == 0 or len(te_idx) == 0:
            continue
        Xtr = Xnum_scaled.iloc[tr_idx].values
        Xte = Xnum_scaled.iloc[te_idx].values
        ytr = labels.iloc[tr_idx].astype(str).values
        yte = labels.iloc[te_idx].astype(str).values
        # encode labels to integers
        from sklearn.preprocessing import LabelEncoder as _LE

        le_local = _LE()
        y_all = np.concatenate([ytr, yte])
        le_local.fit(y_all)
        ytr_enc = le_local.transform(ytr)
        yte_enc = le_local.transform(yte)

        fold_out = OUT_ROOT / f"fold_{i}"
        fold_out.mkdir(parents=True, exist_ok=True)
        try:
            art, stats = try_lstm_any_backend(
                Xtr, ytr_enc, Xte, yte_enc, str(fold_out), seed=42
            )
            if art is not None and isinstance(stats, dict) and "best_f1_macro" in stats:
                lstm_fold_results.append((i, art, stats))
                print(f"INFO: LSTM fold {i} best_f1={stats['best_f1_macro']}")
        except Exception as e:
            print(f"WARNING: LSTM fallback failed on fold {i}: {e}")

    # summarize LSTM results across folds
    if lstm_fold_results:
        lstm_mean_f1 = float(
            np.mean([s[2]["best_f1_macro"] for s in lstm_fold_results])
        )
        lstm_mean_latency = None
        # prefer latency from first art if present
        for _, art, _ in lstm_fold_results:
            if isinstance(art, dict) and "latency_ms_per_sample" in art:
                lstm_mean_latency = float(art["latency_ms_per_sample"])
                break
        print(f"INFO: LSTM mean F1_macro across folds: {lstm_mean_f1:.4f}")
    else:
        lstm_mean_f1 = np.nan
        lstm_mean_latency = None

    # prepare a small summary dict that callers (and sweep) can consume
    run_summary = {
        "lstm_mean_f1": (lstm_mean_f1 if not pd.isna(lstm_mean_f1) else None),
        "lstm_mean_latency_ms": (
            lstm_mean_latency if lstm_mean_latency is not None else None
        ),
        "n_train_total": int(n_train_total),
        "n_val_total": int(n_val_total),
    }

    # determine best model among baselines
    best_model = None
    best_f1 = -np.inf
    try:
        if not agg_df.empty and "f1_macro_mean" in agg_df.columns:
            for _, r in agg_df.iterrows():
                if r.get("f1_macro_mean", -np.inf) > best_f1:
                    best_f1 = r.get("f1_macro_mean")
                    best_model = r["model"]
    except Exception:
        pass
    # compare with LSTM
    if not np.isnan(lstm_mean_f1) and lstm_mean_f1 > best_f1:
        best_model = "lstm"
        best_f1 = lstm_mean_f1

    # print requested summary lines
    if best_model is not None:
        try:
            print(f"BEST_MODEL: {best_model} F1_macro={best_f1:.4f}")
        except Exception:
            print(f"BEST_MODEL: {best_model}")
    else:
        print("BEST_MODEL: unknown")

    if best_model == "lstm" and lstm_mean_latency is not None:
        print(f"LATENCY_PROFILE: avg_inference_ms_per_sample={lstm_mean_latency:.4f}")
    else:
        print("LATENCY_PROFILE: N/A")

    print(f"OUTPUT_FOLDER: {OUT_ROOT}")

    # run config (small provenance file)
    try:
        class_map = (
            sorted(list(pd.Series(df["label"].astype(str)).unique()))
            if "label" in df.columns
            else []
        )
        run_cfg = {
            "features": str(features_path) if features_path is not None else None,
            "run_ts": RUN_TS,
            "env": {
                "is_kaggle": bool(IS_KAGGLE),
                "data_root": DEFAULT_LOCAL_ROOT,
                "out_root": DEFAULT_OUT_ROOT,
            },
            "backend": BACKEND,
            "seq_len": LSTM_SEQ_LEN,
            "class_map": class_map,
            "latency_ms_per_sample": (
                lstm_mean_latency if lstm_mean_latency is not None else None
            ),
            "slug": (
                CURRENT_SLUG
                if "CURRENT_SLUG" in globals() and CURRENT_SLUG is not None
                else None
            ),
            "rows": len(df) if df is not None else 0,
            "features": (
                numeric_cols if "numeric_cols" in locals() else list(df.columns)
            ),
            "fold_params": {
                "train_days": 120,
                "test_days": 60,
                "gap_days": 10,
                "n_folds_built": len(folds),
            },
            "models_run": (
                sorted(baseline_df["model"].unique().tolist())
                if not baseline_df.empty
                else []
            ),
        }
        cfgp = OUT_ROOT / "run_config.json"
        with open(cfgp, "w", encoding="utf8") as fh:
            json.dump(run_cfg, fh, indent=2)
        print("INFO: wrote run config ->", cfgp)
    except Exception as e:
        print("WARNING: failed to write run_config.json ->", e)

    # Summary plot: mean F1_macro per model (agg from baseline_df)
    plt.figure(figsize=(7, 4))
    if not baseline_df.empty:
        mean_f1 = baseline_df.groupby("model")["f1_macro"].mean().sort_values()
        mean_f1.plot(kind="barh")
    else:
        plt.text(0.5, 0.5, "No data", ha="center")
    plt.title("Baselines - Mean F1_macro (temporal folds)")
    plt.xlabel("F1 macro")
    if in_notebook():
        plt.show()
    figp = savefig("baseline_f1_macro.png")
    print("INFO: wrote figure  ->", figp)
    plt.close()

    return agg_df, run_summary


def main():
    ap = argparse.ArgumentParser(
        description="Run NB2 baselines on labeled daily features CSV"
    )
    ap.add_argument(
        "--features", help="Path to features_daily_labeled.csv", required=False
    )
    ap.add_argument(
        "--sweep",
        action="store_true",
        help="Run small local parameter sweep (seq-len x rolling windows)",
    )
    ap.add_argument(
        "--seq-len",
        type=int,
        default=7,
        help="Sequence length for LSTM when not sweeping",
    )
    ap.add_argument(
        "--seq-len-grid",
        nargs="+",
        type=int,
        default=[7, 14],
        help="Grid of seq-len values for sweep",
    )
    ap.add_argument(
        "--rolling-windows",
        nargs="+",
        type=int,
        default=[7, 14],
        help="Rolling windows to use when not sweeping",
    )
    ap.add_argument(
        "--rolling-grid",
        nargs="+",
        type=int,
        default=[7, 14, 28],
        help="Candidate rolling windows for sweep",
    )
    ap.add_argument(
        "--use-class-weight",
        dest="use_class_weight",
        action="store_true",
        help="Use class_weight='balanced' for classifiers (default)",
    )
    ap.add_argument(
        "--no-class-weight",
        dest="use_class_weight",
        action="store_false",
        help="Do not use class weight for classifiers",
    )
    ap.add_argument(
        "--batch",
        action="store_true",
        help="Run NB2 in batch mode over discovered datasets",
    )
    ap.add_argument(
        "--filter-participant",
        help="Glob filter for participants (e.g. p0000*)",
        default=None,
    )
    ap.add_argument(
        "--filter-snapshot", help="Glob filter for snapshot (e.g. s2025*)", default=None
    )
    ap.add_argument(
        "--limit", type=int, help="Limit number of datasets to process", default=None
    )
    ap.add_argument(
        "--kaggle-root",
        help="Root to discover datasets on Kaggle",
        default="/kaggle/input",
    )
    ap.add_argument(
        "--local-root",
        help="Local root to discover datasets",
        default="./data/ai/local",
    )
    ap.add_argument(
        "--kaggle-mode",
        action="store_true",
        help="When set, discovery uses --kaggle-root and enables TF LSTM path",
    )
    ap.add_argument(
        "--outdir",
        help="Output directory root (optional). Default: notebooks/outputs/NB2/<ts>",
    )
    ap.add_argument(
        "--dry-run", action="store_true", help="Load and validate dataset, then exit"
    )
    ap.add_argument(
        "--slug", help="Dataset slug (e.g. p000001-s20250929-nb2v303-r1)", default=None
    )
    args = ap.parse_args()
    # deterministic seeds
    random.seed(42)
    np.random.seed(42)
    os.environ["PYTHONHASHSEED"] = "42"

    # environment detection and defaults
    env = detect_env()
    # prefer CLI overrides when provided; otherwise use detected defaults
    if not args.local_root:
        args.local_root = env["data_root"]
    if not args.outdir:
        args.outdir = env["out_root"]

    # set globals for downstream functions
    global ENABLE_LSTM, BACKEND, CURRENT_SLUG
    ENABLE_LSTM = bool(env.get("enable_lstm", False))
    BACKEND = env.get("backend", "none")
    # set use_class_weight global
    global USE_CLASS_WEIGHT, LSTM_SEQ_LEN
    USE_CLASS_WEIGHT = (
        bool(args.use_class_weight) if hasattr(args, "use_class_weight") else True
    )
    LSTM_SEQ_LEN = (
        int(args.seq_len)
        if hasattr(args, "seq_len") and args.seq_len is not None
        else LSTM_SEQ_LEN
    )

    # Resolve features path according to slug / kaggle-mode / discovery rules
    features_p = None
    slug = args.slug
    if slug:
        if args.kaggle_mode:
            features_p = Path(args.kaggle_root) / slug / "features_daily_labeled.csv"
            version_log_p = Path(args.kaggle_root) / slug / "version_log_enriched.csv"
        else:
            features_p = Path(args.local_root) / slug / "features_daily_labeled.csv"
            version_log_p = Path(args.local_root) / slug / "version_log_enriched.csv"
        print(f"INFO: resolving slug -> {slug} => {features_p}")
        if not features_p.exists():
            print(f"ERROR: features dataset not found at {features_p}")
            raise SystemExit(2)
    elif args.features:
        features_p = Path(args.features).expanduser().resolve()
        if not features_p.exists():
            print(f"ERROR: features dataset not found at {features_p}")
            raise SystemExit(2)
        # derive slug from parent folder when possible
        try:
            slug = Path(features_p).parent.name
        except Exception:
            slug = slug or "unknown-slug"
    else:
        # discovery mode: look under local or kaggle root and pick the latest snapshot by name
        root = Path(args.kaggle_root) if args.kaggle_mode else Path(args.local_root)
        items = discover_nb2_datasets(str(root))
        sel = select_datasets(
            items, f_part=args.filter_participant, f_snap=args.filter_snapshot, limit=1
        )
        if len(sel) == 0:
            print(f"ERROR: no datasets found under {root} matching filters")
            return 2
        it = sel[-1]
        slug = it["slug"]
        features_p = it["features"]
        version_log_p = it.get("version_log")
        print(f"INFO: discovered dataset -> {slug} -> {features_p}")

    # prepare per-slug output dir
    outroot = Path("notebooks/outputs/NB2") / slug / RUN_TS
    prepare_output_dirs(outroot)
    print("INFO: outputs ->", OUT_ROOT)

    # load CSV (attempt to parse date if header contains 'date')
    try:
        parse_dates = (
            ["date"] if "date" in pd.read_csv(features_p, nrows=0).columns else None
        )
        df = pd.read_csv(features_p, parse_dates=parse_dates)
    except Exception:
        df = pd.read_csv(features_p)

    print(
        f"INFO: loaded dataframe: rows={len(df)}, cols={len(df.columns)}, has_columns={list(df.columns)}"
    )

    # If dry-run, validate and exit 0
    if args.dry_run:
        print("INFO: dry-run requested; exiting without running models")
        return 0

    # helper: derive rolling features (mean/std + delta mean) for given rolling windows
    def add_rolling_features(df_in: pd.DataFrame, windows):
        df = df_in.copy()
        exclude = {"label", "label_source", "label_notes", "date"}
        num_cols = [
            c for c in df.select_dtypes(include=[np.number]).columns if c not in exclude
        ]
        for w in windows:
            for c in num_cols:
                mname = f"{c}_r{w}_mean"
                sname = f"{c}_r{w}_std"
                dname = f"{c}_r{w}_delta"
                # rolling on past values (exclude current) -> shift then rolling
                try:
                    rolled_mean = (
                        df[c].shift(1).rolling(window=w, min_periods=1).mean().fillna(0)
                    )
                    rolled_std = (
                        df[c].shift(1).rolling(window=w, min_periods=1).std().fillna(0)
                    )
                    df[mname] = rolled_mean
                    df[sname] = rolled_std
                    df[dname] = (rolled_mean - rolled_mean.shift(1)).fillna(0)
                except Exception:
                    df[mname] = 0
                    df[sname] = 0
                    df[dname] = 0
        return df

    # helper: build sweep grid (small curated set)
    def build_sweep_grid(seq_grid, roll_grid):
        # keep single-window combos for 7 and 14 if present, and specific two-window combos
        combos = []
        if 7 in seq_grid:
            pass
        # seq_grid values as provided
        seqs = sorted(list(set(seq_grid)))
        # rolling combos limited to: [7], [14], [7,14], [14,28] when applicable
        rolls = []
        if 7 in roll_grid:
            rolls.append([7])
        if 14 in roll_grid:
            rolls.append([14])
        if 7 in roll_grid and 14 in roll_grid:
            rolls.append([7, 14])
        if 14 in roll_grid and 28 in roll_grid:
            rolls.append([14, 28])
        grid = []
        cid = 0
        for s in seqs:
            for r in rolls:
                cid += 1
                grid.append({"cfg_id": cid, "seq_len": int(s), "rolling_windows": r})
        return grid

    # Sweep mode: run multiple configs and collect summary
    if args.sweep:
        seq_grid = args.seq_len_grid if hasattr(args, "seq_len_grid") else [7, 14]
        roll_grid = args.rolling_grid if hasattr(args, "rolling_grid") else [7, 14, 28]
        grid = build_sweep_grid(seq_grid, roll_grid)
        sweep_rows = []
        best_row = None
        best_val = -np.inf
        for cfg in grid:
            cfg_id = cfg["cfg_id"]
            s_len = cfg["seq_len"]
            rws = cfg["rolling_windows"]
            print(
                f"INFO: Sweep cfg {cfg_id} seq_len={s_len} rolling={rws} use_class_weight={USE_CLASS_WEIGHT}"
            )
            # prepare a fresh df with rolling features
            df_cfg = add_rolling_features(df, rws)
            # set globals for LSTM
            LSTM_SEQ_LEN = int(s_len)
            # run baselines+LSTM
            agg_df, run_summary = run_baselines(df_cfg, features_path=str(features_p))
            # determine best model in this config
            bm = None
            bf = -np.inf
            bw = np.nan
            ba = np.nan
            bk = np.nan
            au = np.nan
            lat = None
            if (
                agg_df is not None
                and not agg_df.empty
                and "f1_macro_mean" in agg_df.columns
            ):
                for _, r in agg_df.iterrows():
                    v = r.get("f1_macro_mean", -np.inf)
                    if v > bf:
                        bf = v
                        bm = r["model"]
                        bw = r.get("f1_weighted_mean", np.nan)
                        ba = r.get("balanced_acc_mean", np.nan)
                        bk = r.get("kappa_mean", np.nan)
                        au = r.get("auroc_ovr_macro_mean", np.nan)
            # compare LSTM
            lstm_f1 = (
                run_summary.get("lstm_mean_f1") if run_summary is not None else None
            )
            if lstm_f1 is not None and lstm_f1 > bf:
                bm = "lstm"
                bf = lstm_f1
                lat = run_summary.get("lstm_mean_latency_ms")
            sweep_rows.append(
                {
                    "slug": slug,
                    "cfg_id": cfg_id,
                    "seq_len": s_len,
                    "rolling_windows": ";".join(map(str, rws)),
                    "use_class_weight": bool(USE_CLASS_WEIGHT),
                    "best_model": bm,
                    "f1_macro": (
                        float(bf) if bf is not None and not pd.isna(bf) else np.nan
                    ),
                    "f1_weighted": (
                        float(bw) if bw is not None and not pd.isna(bw) else np.nan
                    ),
                    "balanced_acc": (
                        float(ba) if ba is not None and not pd.isna(ba) else np.nan
                    ),
                    "kappa": (
                        float(bk) if bk is not None and not pd.isna(bk) else np.nan
                    ),
                    "auroc_ovr_macro": (
                        float(au) if au is not None and not pd.isna(au) else np.nan
                    ),
                    "latency_ms_per_sample": float(lat) if lat is not None else np.nan,
                    "n_train_total": (
                        int(run_summary.get("n_train_total", 0))
                        if run_summary is not None
                        else 0
                    ),
                    "n_val_total": (
                        int(run_summary.get("n_val_total", 0))
                        if run_summary is not None
                        else 0
                    ),
                }
            )
            if bf is not None and not pd.isna(bf) and bf > best_val:
                best_val = bf
                best_row = sweep_rows[-1]

        # write sweep summary and best config
        try:
            sweep_df = pd.DataFrame(sweep_rows)
            sweep_path = TAB_DIR / "nb2_sweep_summary.csv"
            sweep_df.to_csv(sweep_path, index=False)
            print("INFO: wrote sweep summary ->", sweep_path)
            if best_row is not None:
                best_cfg_p = OUT_ROOT / "best_config.json"
                with open(best_cfg_p, "w", encoding="utf8") as fh:
                    json.dump(best_row, fh, indent=2)
                print("INFO: wrote best config ->", best_cfg_p)
        except Exception as e:
            print("WARNING: failed to write sweep outputs ->", e)
        return 0

    # Batch mode: discover and loop
    if args.batch:
        root = args.kaggle_root if args.kaggle_mode else args.local_root
        items = discover_nb2_datasets(root)
        sel = select_datasets(
            items,
            f_part=args.filter_participant,
            f_snap=args.filter_snapshot,
            limit=args.limit,
        )
        if len(sel) == 0:
            print(f"ERROR: no datasets found under {root} matching filters")
            return 2
        summary_rows = []
        for it in sel:
            slug = it["slug"]
            features_p = it["features"]
            print(f"INFO: running NB2 for {slug} -> {features_p}")
            # per-dataset outdir
            outroot = Path("notebooks/outputs/NB2") / slug / RUN_TS
            prepare_output_dirs(outroot)
            print("INFO: outputs ->", OUT_ROOT)
            # load CSV
            try:
                parse_dates = (
                    ["date"]
                    if "date" in pd.read_csv(features_p, nrows=0).columns
                    else None
                )
                df = pd.read_csv(features_p, parse_dates=parse_dates)
            except Exception:
                df = pd.read_csv(features_p)
            if args.dry_run:
                print("INFO: dry-run requested; skipping models for this dataset")
                continue
            # run baselines (+ optional LSTM if ENABLE_LSTM)
            agg_df, run_summary = run_baselines(df, features_path=str(features_p))
            # collect summary row values
            # pick baseline best row
            bm = None
            bf = -np.inf
            bw = np.nan
            au = np.nan
            lat = None
            if not agg_df.empty and "f1_macro_mean" in agg_df.columns:
                for _, r in agg_df.iterrows():
                    v = r.get("f1_macro_mean", -np.inf)
                    if v > bf:
                        bf = v
                        bm = r["model"]
                        bw = r.get("f1_weighted_mean", np.nan)
                        au = r.get("auroc_ovr_macro_mean", np.nan)
            # If LSTM ran and beat baselines, try to capture it from printed BEST_MODEL or artifacts in OUT_ROOT
            # For quick summary, prefer baseline selection above (LSTM integration is handled inside run_baselines when enabled)
            summary_rows.append(
                {
                    "slug": slug,
                    "participant": it["participant"],
                    "snapshot": it["snapshot"],
                    "best_model": bm if bm is not None else "unknown",
                    "f1_macro": float(bf) if bf is not None else np.nan,
                    "f1_weighted": float(bw) if bw is not None else np.nan,
                    "auroc_ovr_macro": (
                        float(au) if au is not None and not pd.isna(au) else np.nan
                    ),
                    "latency_ms_per_sample": (
                        lat
                        if lat is not None
                        else (
                            run_summary.get("lstm_mean_latency_ms")
                            if run_summary is not None
                            else np.nan
                        )
                    ),
                    "out_dir": str(OUT_ROOT),
                }
            )

        # write aggregate summary
        agg_dir = Path("notebooks/outputs/NB2/_aggregate")
        agg_dir.mkdir(parents=True, exist_ok=True)
        agg_path = agg_dir / "nb2_batch_summary.csv"
        pd.DataFrame(summary_rows).to_csv(agg_path, index=False)
        print("INFO: wrote batch summary ->", agg_path)
        return 0

    # Single-run behavior (backward compatible): prefer --features; if not provided try discovery
    if args.features:
        features_p = Path(args.features).expanduser().resolve()
    else:
        root = args.kaggle_root if args.kaggle_mode else args.local_root
        items = discover_nb2_datasets(root)
        sel = select_datasets(items, limit=1)
        if len(sel) == 0:
            print("ERROR: no datasets discovered and --features not provided")
            return 2
        features_p = sel[-1]["features"]

    print(f"INFO: using dataset -> {features_p}")
    if not features_p.exists():
        print(f"ERROR: features dataset not found at {features_p}")
        raise SystemExit(2)

    # load CSV (attempt to parse date if header contains 'date')
    try:
        parse_dates = (
            ["date"] if "date" in pd.read_csv(features_p, nrows=0).columns else None
        )
        df = pd.read_csv(features_p, parse_dates=parse_dates)
    except Exception:
        df = pd.read_csv(features_p)

    print(
        f"INFO: loaded dataframe: rows={len(df)}, cols={len(df.columns)}, has_columns={list(df.columns)}"
    )

    # If dry-run, validate and exit 0
    if args.dry_run:
        print("INFO: dry-run requested; exiting without running models")
        return 0

    # prepare output directories
    outroot = (
        Path(args.outdir).expanduser().resolve()
        if args.outdir
        else Path("notebooks/outputs/NB2") / RUN_TS
    )
    prepare_output_dirs(outroot)
    print("INFO: outputs ->", OUT_ROOT)

    # Run baselines and write outputs
    agg_df, run_summary = run_baselines(df, features_path=str(features_p))
    print("INFO: Baselines complete. Models:", list(agg_df.get("model", [])))
    return 0


if __name__ == "__main__":
    sys.exit(main())
