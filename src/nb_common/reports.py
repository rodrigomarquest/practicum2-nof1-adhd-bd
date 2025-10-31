import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import roc_curve, auc


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
    from sklearn.metrics import confusion_matrix

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
