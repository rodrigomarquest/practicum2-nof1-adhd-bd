import numpy as np
import warnings
from sklearn.metrics import (
    f1_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    roc_auc_score,
    brier_score_loss,
    classification_report,
)
from sklearn.preprocessing import label_binarize


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
