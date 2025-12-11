"""
ML Metrics Extended Module

PhD-level evaluation metrics for ML6-Extended and ML7-Extended pipelines.
Provides:
- Naïve baselines (majority-class, stratified-random, persistence)
- Per-class metrics (precision, recall, F1, support)
- Confusion matrix aggregation across folds
- Structured export to results/metrics/<snapshot>/

Author: GitHub Copilot
Date: 2025-12-10
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union

import numpy as np
import pandas as pd
from dataclasses import dataclass, asdict, field

logger = logging.getLogger(__name__)


# =============================================================================
# Per-Class Metrics Dataclass
# =============================================================================

@dataclass
class PerClassMetrics:
    """Per-class classification metrics."""
    class_label: int
    precision: float
    recall: float
    f1: float
    support: int
    
    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ExtendedMetrics:
    """Extended metrics with per-class breakdown."""
    # Aggregate metrics
    f1_macro: float
    f1_weighted: float
    balanced_accuracy: float
    cohen_kappa: float
    accuracy: float
    
    # Per-class metrics
    per_class: List[PerClassMetrics]
    
    # Confusion matrix
    confusion_matrix: List[List[int]]
    class_labels: List[int]
    
    # Sample info
    n_samples: int
    n_classes: int
    
    def to_dict(self) -> dict:
        result = {
            'f1_macro': self.f1_macro,
            'f1_weighted': self.f1_weighted,
            'balanced_accuracy': self.balanced_accuracy,
            'cohen_kappa': self.cohen_kappa,
            'accuracy': self.accuracy,
            'per_class': [pc.to_dict() for pc in self.per_class],
            'confusion_matrix': self.confusion_matrix,
            'class_labels': self.class_labels,
            'n_samples': self.n_samples,
            'n_classes': self.n_classes,
        }
        return result


# =============================================================================
# Core Metrics Computation
# =============================================================================

def compute_metrics_extended(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_labels: Optional[List[int]] = None
) -> ExtendedMetrics:
    """
    Compute extended classification metrics including per-class breakdown.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        class_labels: Optional list of class labels (auto-detected if None)
    
    Returns:
        ExtendedMetrics with aggregate and per-class metrics
    """
    from sklearn.metrics import (
        f1_score, balanced_accuracy_score, cohen_kappa_score,
        precision_score, recall_score, accuracy_score,
        confusion_matrix, precision_recall_fscore_support
    )
    
    # Determine class labels
    if class_labels is None:
        class_labels = sorted(set(y_true.tolist()) | set(y_pred.tolist()))
    
    n_classes = len(class_labels)
    n_samples = len(y_true)
    
    # Aggregate metrics
    f1_macro = float(f1_score(y_true, y_pred, average='macro', zero_division=0))
    f1_weighted = float(f1_score(y_true, y_pred, average='weighted', zero_division=0))
    balanced_acc = float(balanced_accuracy_score(y_true, y_pred))
    kappa = float(cohen_kappa_score(y_true, y_pred))
    accuracy = float(accuracy_score(y_true, y_pred))
    
    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=class_labels, zero_division=0
    )
    
    per_class = []
    for i, label in enumerate(class_labels):
        per_class.append(PerClassMetrics(
            class_label=int(label),
            precision=float(precision[i]),
            recall=float(recall[i]),
            f1=float(f1[i]),
            support=int(support[i])
        ))
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=class_labels)
    
    return ExtendedMetrics(
        f1_macro=f1_macro,
        f1_weighted=f1_weighted,
        balanced_accuracy=balanced_acc,
        cohen_kappa=kappa,
        accuracy=accuracy,
        per_class=per_class,
        confusion_matrix=cm.tolist(),
        class_labels=[int(l) for l in class_labels],
        n_samples=n_samples,
        n_classes=n_classes,
    )


# =============================================================================
# Naïve Baseline Predictors
# =============================================================================

def predict_majority_class(
    y_train: np.ndarray,
    n_predict: int
) -> np.ndarray:
    """
    Majority-class baseline: always predict the most frequent class in training.
    
    Args:
        y_train: Training labels (to determine majority class)
        n_predict: Number of predictions to generate
    
    Returns:
        Array of predictions (all same class)
    """
    unique, counts = np.unique(y_train, return_counts=True)
    majority_class = unique[np.argmax(counts)]
    return np.full(n_predict, majority_class, dtype=y_train.dtype)


def predict_stratified_random(
    y_train: np.ndarray,
    n_predict: int,
    random_seed: int = 42
) -> np.ndarray:
    """
    Stratified-random baseline: predict class proportionally to training distribution.
    
    Uses deterministic RNG for reproducibility.
    
    Args:
        y_train: Training labels (to determine class proportions)
        n_predict: Number of predictions to generate
        random_seed: RNG seed for reproducibility
    
    Returns:
        Array of random predictions following training distribution
    """
    rng = np.random.RandomState(random_seed)
    unique, counts = np.unique(y_train, return_counts=True)
    probabilities = counts / counts.sum()
    return rng.choice(unique, size=n_predict, p=probabilities)


def predict_persistence(
    y: np.ndarray,
    val_idx: List[int]
) -> np.ndarray:
    """
    Persistence baseline: predict SoM[t] = SoM[t-1].
    
    For the first validation sample, uses the last training sample.
    
    Args:
        y: Full target array (including train and val)
        val_idx: Validation indices
    
    Returns:
        Array of persistence predictions for validation set
    """
    predictions = []
    
    for i, idx in enumerate(val_idx):
        if idx == 0:
            # Edge case: first sample has no previous
            # Use the sample itself (will count as "correct" if stable)
            predictions.append(y[idx])
        else:
            # Previous day's value
            predictions.append(y[idx - 1])
    
    return np.array(predictions, dtype=y.dtype)


def compute_naive_baselines(
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_full: np.ndarray,
    val_idx: List[int],
    class_labels: Optional[List[int]] = None,
    random_seed: int = 42
) -> Dict[str, ExtendedMetrics]:
    """
    Compute all three naïve baselines.
    
    Args:
        y_train: Training labels
        y_val: Validation labels
        y_full: Full target array (for persistence baseline)
        val_idx: Validation indices in y_full
        class_labels: Optional list of class labels
        random_seed: RNG seed for stratified-random baseline
    
    Returns:
        Dict mapping baseline name to ExtendedMetrics
    """
    n_val = len(y_val)
    
    baselines = {}
    
    # 1. Majority-class baseline
    y_pred_majority = predict_majority_class(y_train, n_val)
    baselines['majority_class'] = compute_metrics_extended(y_val, y_pred_majority, class_labels)
    
    # 2. Stratified-random baseline
    y_pred_stratified = predict_stratified_random(y_train, n_val, random_seed)
    baselines['stratified_random'] = compute_metrics_extended(y_val, y_pred_stratified, class_labels)
    
    # 3. Persistence baseline
    y_pred_persistence = predict_persistence(y_full, val_idx)
    baselines['persistence'] = compute_metrics_extended(y_val, y_pred_persistence, class_labels)
    
    return baselines


# =============================================================================
# Confusion Matrix Aggregation
# =============================================================================

def aggregate_confusion_matrices(
    fold_matrices: List[List[List[int]]],
    method: str = 'sum'
) -> List[List[float]]:
    """
    Aggregate confusion matrices across CV folds.
    
    Args:
        fold_matrices: List of confusion matrices (one per fold)
        method: 'sum' (total counts) or 'mean' (average counts)
    
    Returns:
        Aggregated confusion matrix
    """
    if not fold_matrices:
        return []
    
    # Stack and aggregate
    matrices = np.array(fold_matrices, dtype=float)
    
    if method == 'sum':
        aggregated = np.sum(matrices, axis=0)
    elif method == 'mean':
        aggregated = np.mean(matrices, axis=0)
    else:
        raise ValueError(f"Unknown aggregation method: {method}")
    
    return aggregated.tolist()


def aggregate_per_class_metrics(
    fold_metrics: List[List[PerClassMetrics]]
) -> List[Dict[str, Any]]:
    """
    Aggregate per-class metrics across CV folds.
    
    Args:
        fold_metrics: List of per-class metrics (one list per fold)
    
    Returns:
        List of dicts with mean and std for each metric per class
    """
    if not fold_metrics:
        return []
    
    # Get class labels from first fold
    class_labels = [m.class_label for m in fold_metrics[0]]
    
    aggregated = []
    for i, label in enumerate(class_labels):
        # Collect metrics for this class across folds
        precisions = [fm[i].precision for fm in fold_metrics if i < len(fm)]
        recalls = [fm[i].recall for fm in fold_metrics if i < len(fm)]
        f1s = [fm[i].f1 for fm in fold_metrics if i < len(fm)]
        supports = [fm[i].support for fm in fold_metrics if i < len(fm)]
        
        aggregated.append({
            'class_label': label,
            'precision_mean': float(np.mean(precisions)),
            'precision_std': float(np.std(precisions)),
            'recall_mean': float(np.mean(recalls)),
            'recall_std': float(np.std(recalls)),
            'f1_mean': float(np.mean(f1s)),
            'f1_std': float(np.std(f1s)),
            'support_total': int(np.sum(supports)),
            'support_mean': float(np.mean(supports)),
        })
    
    return aggregated


def aggregate_extended_metrics(
    fold_metrics: List[ExtendedMetrics]
) -> Dict[str, Any]:
    """
    Aggregate ExtendedMetrics across CV folds.
    
    Args:
        fold_metrics: List of ExtendedMetrics (one per fold)
    
    Returns:
        Dict with aggregated metrics
    """
    if not fold_metrics:
        return {}
    
    # Aggregate scalar metrics
    aggregated = {
        'mean_f1_macro': float(np.mean([m.f1_macro for m in fold_metrics])),
        'std_f1_macro': float(np.std([m.f1_macro for m in fold_metrics])),
        'mean_f1_weighted': float(np.mean([m.f1_weighted for m in fold_metrics])),
        'std_f1_weighted': float(np.std([m.f1_weighted for m in fold_metrics])),
        'mean_balanced_accuracy': float(np.mean([m.balanced_accuracy for m in fold_metrics])),
        'std_balanced_accuracy': float(np.std([m.balanced_accuracy for m in fold_metrics])),
        'mean_cohen_kappa': float(np.mean([m.cohen_kappa for m in fold_metrics])),
        'std_cohen_kappa': float(np.std([m.cohen_kappa for m in fold_metrics])),
        'mean_accuracy': float(np.mean([m.accuracy for m in fold_metrics])),
        'std_accuracy': float(np.std([m.accuracy for m in fold_metrics])),
    }
    
    # Aggregate confusion matrices
    aggregated['confusion_matrix_sum'] = aggregate_confusion_matrices(
        [m.confusion_matrix for m in fold_metrics], method='sum'
    )
    aggregated['confusion_matrix_mean'] = aggregate_confusion_matrices(
        [m.confusion_matrix for m in fold_metrics], method='mean'
    )
    
    # Aggregate per-class metrics
    aggregated['per_class'] = aggregate_per_class_metrics(
        [m.per_class for m in fold_metrics]
    )
    
    # Class labels (from first fold)
    aggregated['class_labels'] = fold_metrics[0].class_labels
    aggregated['n_folds'] = len(fold_metrics)
    aggregated['n_samples_total'] = sum(m.n_samples for m in fold_metrics)
    
    return aggregated


# =============================================================================
# Results Export Functions
# =============================================================================

def export_per_class_metrics_csv(
    metrics: Dict[str, Any],
    output_path: Path,
    model_name: str,
    target_name: str
) -> Path:
    """
    Export per-class metrics to CSV.
    
    Args:
        metrics: Aggregated metrics dict with 'per_class' key
        output_path: Output CSV path
        model_name: Model name for the file
        target_name: Target name (e.g., 'som_binary', 'som_3class')
    
    Returns:
        Path to written CSV file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    per_class = metrics.get('per_class', [])
    
    rows = []
    for pc in per_class:
        rows.append({
            'model': model_name,
            'target': target_name,
            'class_label': pc['class_label'],
            'precision_mean': pc['precision_mean'],
            'precision_std': pc['precision_std'],
            'recall_mean': pc['recall_mean'],
            'recall_std': pc['recall_std'],
            'f1_mean': pc['f1_mean'],
            'f1_std': pc['f1_std'],
            'support_total': pc['support_total'],
        })
    
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    
    logger.info(f"[Export] Per-class metrics: {output_path}")
    return output_path


def export_confusion_matrices_json(
    fold_matrices: List[List[List[int]]],
    class_labels: List[int],
    output_path: Path,
    model_name: str,
    target_name: str
) -> Path:
    """
    Export confusion matrices (per-fold and aggregated) to JSON.
    
    Args:
        fold_matrices: List of confusion matrices (one per fold)
        class_labels: Class label values
        output_path: Output JSON path
        model_name: Model name
        target_name: Target name
    
    Returns:
        Path to written JSON file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    data = {
        'model': model_name,
        'target': target_name,
        'class_labels': class_labels,
        'n_folds': len(fold_matrices),
        'per_fold': [
            {'fold_idx': i, 'matrix': cm}
            for i, cm in enumerate(fold_matrices)
        ],
        'aggregated_sum': aggregate_confusion_matrices(fold_matrices, method='sum'),
        'aggregated_mean': aggregate_confusion_matrices(fold_matrices, method='mean'),
        'generated': datetime.now().isoformat(),
    }
    
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    logger.info(f"[Export] Confusion matrices: {output_path}")
    return output_path


def export_baseline_comparison_csv(
    model_metrics: Dict[str, Any],
    baseline_metrics: Dict[str, Dict[str, Any]],
    output_path: Path,
    model_name: str,
    target_name: str
) -> Path:
    """
    Export baseline comparison table to CSV.
    
    Args:
        model_metrics: Aggregated metrics for the trained model
        baseline_metrics: Dict mapping baseline name to aggregated metrics
        output_path: Output CSV path
        model_name: Name of the trained model
        target_name: Target name
    
    Returns:
        Path to written CSV file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    rows = []
    
    # Add model row
    rows.append({
        'method': model_name,
        'type': 'model',
        'target': target_name,
        'f1_macro_mean': model_metrics.get('mean_f1_macro', 0),
        'f1_macro_std': model_metrics.get('std_f1_macro', 0),
        'balanced_accuracy_mean': model_metrics.get('mean_balanced_accuracy', 0),
        'balanced_accuracy_std': model_metrics.get('std_balanced_accuracy', 0),
        'cohen_kappa_mean': model_metrics.get('mean_cohen_kappa', 0),
        'cohen_kappa_std': model_metrics.get('std_cohen_kappa', 0),
    })
    
    # Add baseline rows
    for baseline_name, bl_metrics in baseline_metrics.items():
        rows.append({
            'method': baseline_name,
            'type': 'baseline',
            'target': target_name,
            'f1_macro_mean': bl_metrics.get('mean_f1_macro', 0),
            'f1_macro_std': bl_metrics.get('std_f1_macro', 0),
            'balanced_accuracy_mean': bl_metrics.get('mean_balanced_accuracy', 0),
            'balanced_accuracy_std': bl_metrics.get('std_balanced_accuracy', 0),
            'cohen_kappa_mean': bl_metrics.get('mean_cohen_kappa', 0),
            'cohen_kappa_std': bl_metrics.get('std_cohen_kappa', 0),
        })
    
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    
    logger.info(f"[Export] Baseline comparison: {output_path}")
    return output_path


def setup_metrics_output_dir(
    base_dir: Path,
    participant: str,
    snapshot: str
) -> Path:
    """
    Setup the results/metrics/<snapshot>/ directory structure.
    
    Args:
        base_dir: Base directory (usually 'results')
        participant: Participant ID
        snapshot: Snapshot date
    
    Returns:
        Path to metrics output directory
    """
    metrics_dir = Path(base_dir) / 'metrics' / participant / snapshot
    metrics_dir.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories
    (metrics_dir / 'per_class').mkdir(exist_ok=True)
    (metrics_dir / 'confusion_matrices').mkdir(exist_ok=True)
    (metrics_dir / 'baseline_comparisons').mkdir(exist_ok=True)
    
    logger.info(f"[Setup] Metrics directory: {metrics_dir}")
    return metrics_dir


# =============================================================================
# Sequence Model Baselines (for ML7)
# =============================================================================

def predict_persistence_sequence(
    y_seq: np.ndarray,
    seq_len: int
) -> np.ndarray:
    """
    Persistence baseline for sequence models: predict y[t] = y[t-1].
    
    For sequence models, the target is at the END of the sequence.
    Persistence predicts the second-to-last value in the sequence.
    
    Args:
        y_seq: Target values for each sequence (n_seq,)
        seq_len: Sequence length (used for logging only)
    
    Returns:
        Array of persistence predictions
    
    Note:
        For sequences constructed as [t-seq_len+1, ..., t],
        the label is y[t] and persistence predicts y[t-1].
        Since we don't have direct access to y[t-1] in sequence form,
        we use the fact that consecutive sequences share overlapping labels.
    """
    # For consecutive sequences, y_seq[i] = y[i + seq_len - 1] in original series
    # Persistence would predict y[i + seq_len - 2], which is the label of the previous sequence
    # Therefore: persistence_pred[i] = y_seq[i-1] for i > 0
    
    predictions = np.zeros_like(y_seq)
    predictions[0] = y_seq[0]  # No previous sequence, predict same value
    predictions[1:] = y_seq[:-1]  # Previous sequence's label
    
    return predictions


def compute_naive_baselines_sequence(
    y_train_seq: np.ndarray,
    y_val_seq: np.ndarray,
    seq_len: int,
    class_labels: Optional[List[int]] = None,
    random_seed: int = 42
) -> Dict[str, ExtendedMetrics]:
    """
    Compute naïve baselines for sequence models.
    
    Args:
        y_train_seq: Training sequence labels
        y_val_seq: Validation sequence labels
        seq_len: Sequence length
        class_labels: Optional class labels
        random_seed: RNG seed
    
    Returns:
        Dict mapping baseline name to ExtendedMetrics
    """
    n_val = len(y_val_seq)
    
    baselines = {}
    
    # 1. Majority-class baseline
    y_pred_majority = predict_majority_class(y_train_seq, n_val)
    baselines['majority_class'] = compute_metrics_extended(y_val_seq, y_pred_majority, class_labels)
    
    # 2. Stratified-random baseline
    y_pred_stratified = predict_stratified_random(y_train_seq, n_val, random_seed)
    baselines['stratified_random'] = compute_metrics_extended(y_val_seq, y_pred_stratified, class_labels)
    
    # 3. Persistence baseline (sequence-adapted)
    # Note: For validation set, we need to consider the last training sequence
    # Concatenate to handle boundary correctly
    y_combined = np.concatenate([y_train_seq[-1:], y_val_seq])
    y_pred_persistence = predict_persistence_sequence(y_combined, seq_len)[1:]  # Remove the prepended element
    baselines['persistence'] = compute_metrics_extended(y_val_seq, y_pred_persistence, class_labels)
    
    return baselines


# =============================================================================
# Utility Functions
# =============================================================================

def format_metrics_table(
    metrics: Dict[str, Any],
    title: str = "Classification Metrics"
) -> str:
    """
    Format metrics as a Markdown table.
    
    Args:
        metrics: Aggregated metrics dict
        title: Table title
    
    Returns:
        Markdown formatted table string
    """
    lines = [
        f"### {title}",
        "",
        "| Metric | Mean | Std |",
        "|--------|------|-----|",
    ]
    
    metric_names = [
        ('f1_macro', 'F1-Macro'),
        ('f1_weighted', 'F1-Weighted'),
        ('balanced_accuracy', 'Balanced Accuracy'),
        ('cohen_kappa', 'Cohen Kappa'),
        ('accuracy', 'Accuracy'),
    ]
    
    for key, display_name in metric_names:
        mean_key = f'mean_{key}'
        std_key = f'std_{key}'
        if mean_key in metrics:
            mean_val = metrics[mean_key]
            std_val = metrics.get(std_key, 0)
            lines.append(f"| {display_name} | {mean_val:.4f} | {std_val:.4f} |")
    
    return "\n".join(lines)


def format_per_class_table(
    per_class: List[Dict[str, Any]],
    title: str = "Per-Class Metrics"
) -> str:
    """
    Format per-class metrics as a Markdown table.
    
    Args:
        per_class: List of per-class metric dicts
        title: Table title
    
    Returns:
        Markdown formatted table string
    """
    lines = [
        f"### {title}",
        "",
        "| Class | Precision | Recall | F1 | Support |",
        "|-------|-----------|--------|----|---------| ",
    ]
    
    for pc in per_class:
        label = pc['class_label']
        prec = f"{pc['precision_mean']:.3f}±{pc['precision_std']:.3f}"
        rec = f"{pc['recall_mean']:.3f}±{pc['recall_std']:.3f}"
        f1 = f"{pc['f1_mean']:.3f}±{pc['f1_std']:.3f}"
        supp = pc['support_total']
        lines.append(f"| {label} | {prec} | {rec} | {f1} | {supp} |")
    
    return "\n".join(lines)


def format_baseline_comparison_table(
    model_metrics: Dict[str, Any],
    baseline_metrics: Dict[str, Dict[str, Any]],
    model_name: str,
    title: str = "Baseline Comparison"
) -> str:
    """
    Format baseline comparison as a Markdown table.
    
    Args:
        model_metrics: Aggregated model metrics
        baseline_metrics: Dict of baseline name -> metrics
        model_name: Name of the trained model
        title: Table title
    
    Returns:
        Markdown formatted table string
    """
    lines = [
        f"### {title}",
        "",
        "| Method | F1-Macro | Bal. Acc | Cohen κ |",
        "|--------|----------|----------|---------|",
    ]
    
    # Model row (highlighted)
    f1 = model_metrics.get('mean_f1_macro', 0)
    ba = model_metrics.get('mean_balanced_accuracy', 0)
    ck = model_metrics.get('mean_cohen_kappa', 0)
    lines.append(f"| **{model_name}** | **{f1:.4f}** | **{ba:.4f}** | **{ck:.4f}** |")
    
    # Baseline rows
    baseline_order = ['majority_class', 'stratified_random', 'persistence']
    display_names = {
        'majority_class': 'Majority Class',
        'stratified_random': 'Stratified Random',
        'persistence': 'Persistence',
    }
    
    for bl_name in baseline_order:
        if bl_name in baseline_metrics:
            bl = baseline_metrics[bl_name]
            f1 = bl.get('mean_f1_macro', 0)
            ba = bl.get('mean_balanced_accuracy', 0)
            ck = bl.get('mean_cohen_kappa', 0)
            display = display_names.get(bl_name, bl_name)
            lines.append(f"| {display} | {f1:.4f} | {ba:.4f} | {ck:.4f} |")
    
    return "\n".join(lines)


def format_confusion_matrix_md(
    cm: List[List[Union[int, float]]],
    class_labels: List[int],
    title: str = "Confusion Matrix"
) -> str:
    """
    Format confusion matrix as a Markdown table.
    
    Args:
        cm: Confusion matrix (rows=true, cols=pred)
        class_labels: Class label values
        title: Table title
    
    Returns:
        Markdown formatted table string
    """
    lines = [
        f"### {title}",
        "",
        "*Rows: True labels, Columns: Predicted labels*",
        "",
    ]
    
    # Header row
    header = "| True \\ Pred | " + " | ".join(str(l) for l in class_labels) + " |"
    lines.append(header)
    lines.append("|" + "---|" * (len(class_labels) + 1))
    
    # Data rows
    for i, row in enumerate(cm):
        row_str = f"| **{class_labels[i]}** | " + " | ".join(f"{v:.0f}" if isinstance(v, float) else str(v) for v in row) + " |"
        lines.append(row_str)
    
    return "\n".join(lines)
