"""
ML6 Baselines: 6-fold calendar-based temporal CV with 5 baselines.

Protocol:
    - 6 calendar folds: 4 months train / 2 months validation (strict calendar boundaries)
    - Baselines:
      1. Dummy (stratified random)
      2. Naive-Yesterday (d-1 label, fallback to 7-day mode)
      3. MovingAvg-7d (rolling mean → quantize at -0.33/+0.33)
      4. Rule-based Clinical (pbsi_score → map thresholds)
      5. Logistic Regression (L2, C∈{0.1,1,3}, class_weight=balanced)
    - Targets: 3-class (label_3cls) and 2-class (label_2cls)
    - Metrics:
      - 3-class: f1_macro, f1_weighted, balanced_acc, kappa
      - 2-class: above + roc_auc + mcnemar_p (LogReg vs Dummy)
    - Outputs:
      - ml6/baseline_stratified_3cls.csv (Dummy only)
      - ml6/baseline_stratified_2cls.csv (Dummy only)
      - ml6/baselines_label_3cls.csv (all 5 baselines)
      - ml6/baselines_label_2cls.csv (all 5 baselines + mcnemar_p)
      - ml6/confusion_matrices/*.png (per fold per model)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
import warnings
from datetime import timedelta
from scipy import stats

from sklearn.preprocessing import StandardScaler
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    f1_score, balanced_accuracy_score, cohen_kappa_score,
    roc_auc_score, confusion_matrix
)
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)
np.random.seed(42)


def calendar_cv_folds(
    df: pd.DataFrame, n_folds: int = 6, train_months: int = 4, val_months: int = 2
) -> List[Tuple]:
    """
    Generate calendar-based temporal CV folds (strict date boundaries).
    
    Returns: list of (fold_idx, train_df, val_df, train_start, train_end, val_start, val_end)
    """
    df_sorted = df.sort_values('date').reset_index(drop=True)
    dates = pd.to_datetime(df_sorted['date'])
    date_min = dates.min()
    
    total_days = (dates.max() - date_min).days
    fold_days = total_days // n_folds
    train_days_approx = (train_months / (train_months + val_months)) * fold_days
    
    folds = []
    for fold_idx in range(n_folds):
        fold_start = date_min + timedelta(days=fold_idx * fold_days)
        fold_end = fold_start + timedelta(days=fold_days)
        train_end = fold_start + timedelta(days=int(train_days_approx))
        
        train_mask = (dates >= fold_start) & (dates < train_end)
        val_mask = (dates >= train_end) & (dates < fold_end)
        
        if train_mask.sum() > 0 and val_mask.sum() > 0:
            df_train = df_sorted[train_mask].reset_index(drop=True)
            df_val = df_sorted[val_mask].reset_index(drop=True)
            
            train_str = f"{df_train['date'].min()} to {df_train['date'].max()}"
            val_str = f"{df_val['date'].min()} to {df_val['date'].max()}"
            
            logger.info(
                f"Fold {fold_idx}: train {train_str} ({len(df_train)} rows), "
                f"val {val_str} ({len(df_val)} rows)"
            )
            
            folds.append((fold_idx, df_train, df_val, train_str, val_str))
    
    logger.info(f"Generated {len(folds)} calendar-based folds")
    return folds


def prepare_features(x_train: pd.DataFrame, x_val: pd.DataFrame) -> Tuple:
    """Impute missing and standardize."""
    imputer = SimpleImputer(strategy='median')
    scaler = StandardScaler()
    
    x_train_imp = imputer.fit_transform(x_train)
    x_train_scaled = scaler.fit_transform(x_train_imp)
    
    x_val_imp = imputer.transform(x_val)
    x_val_scaled = scaler.transform(x_val_imp)
    
    return x_train_scaled, x_val_scaled


def compute_metrics_3cls(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    """Compute 3-class metrics."""
    return {
        'f1_macro': float(f1_score(y_true, y_pred, average='macro', zero_division=0)),
        'f1_weighted': float(f1_score(y_true, y_pred, average='weighted', zero_division=0)),
        'balanced_acc': float(balanced_accuracy_score(y_true, y_pred)),
        'kappa': float(cohen_kappa_score(y_true, y_pred)),
    }


def compute_metrics_2cls(
    y_true: np.ndarray, y_pred: np.ndarray, y_proba=None
) -> Dict:
    """Compute 2-class metrics."""
    metrics = compute_metrics_3cls(y_true, y_pred)
    
    roc_auc = np.nan
    if y_proba is not None and len(np.unique(y_true)) == 2:
        try:
            roc_auc = float(roc_auc_score(y_true, y_proba[:, 1]))
        except Exception:
            pass
    
    metrics['roc_auc'] = roc_auc
    return metrics


def fit_dummy(x_train: np.ndarray, y_train: np.ndarray) -> DummyClassifier:
    """Fit stratified dummy."""
    dummy = DummyClassifier(strategy='stratified', random_state=42)
    dummy.fit(x_train, y_train)
    return dummy


def fit_naive_yesterday(df_train: pd.DataFrame, y_train: np.ndarray) -> Dict:
    """Build mapping: date → label for Naive-Yesterday baseline."""
    return {'date_to_label': dict(zip(df_train['date'], y_train))}


def fit_moving_avg_7d(df_train: pd.DataFrame, y_train: np.ndarray) -> Dict:
    """Build training data for MA7 baseline."""
    return {'date_to_label': dict(zip(df_train['date'], y_train))}


def fit_rule_based() -> None:
    """Rule-based doesn't need training (uses pbsi_score directly)."""
    return None


def fit_logistic_regression(x_train: np.ndarray, y_train: np.ndarray) -> LogisticRegression:
    """Fit LR with hyperparameter tuning."""
    lr = LogisticRegression(
        penalty='l2', solver='liblinear', class_weight='balanced',
        random_state=42, max_iter=1000
    )
    
    try:
        grid = GridSearchCV(
            lr, {'C': [0.1, 1, 3]}, cv=min(3, len(np.unique(y_train))),
            scoring='f1_macro', n_jobs=-1
        )
        grid.fit(x_train, y_train)
        logger.info(f"  LR best C: {grid.best_params_['C']}")
        return grid.best_estimator_
    except Exception as e:
        logger.warning(f"GridSearch failed: {e}. Using default LR.")
        lr.fit(x_train, y_train)
        return lr


def predict_naive_yesterday(
    df_train: pd.DataFrame, df_val: pd.DataFrame, y_train: np.ndarray
) -> np.ndarray:
    """Predict using yesterday's label (fallback to 7-day mode)."""
    date_to_label = dict(zip(df_train['date'], y_train))
    dates_val = pd.to_datetime(df_val['date']).values
    
    preds = []
    for curr_date in dates_val:
        prev_date = curr_date - timedelta(days=1)
        
        # Try to find yesterday's label
        prev_date_str = pd.Timestamp(prev_date).strftime('%Y-%m-%d')
        if prev_date_str in date_to_label:
            preds.append(date_to_label[prev_date_str])
        else:
            # Fallback: use 7-day mode
            labels_near = []
            for d in range(1, 8):
                check_date = curr_date - timedelta(days=d)
                check_str = pd.Timestamp(check_date).strftime('%Y-%m-%d')
                if check_str in date_to_label:
                    labels_near.append(date_to_label[check_str])
            
            if labels_near:
                mode_val = max(set(labels_near), key=labels_near.count)
                preds.append(mode_val)
            else:
                # Last resort: mode of entire training set
                preds.append(max(set(y_train), key=list(y_train).count))
    
    return np.array(preds)


def predict_moving_avg_7d(
    df_train: pd.DataFrame, df_val: pd.DataFrame, y_train: np.ndarray
) -> np.ndarray:
    """Predict using 7-day rolling mean, quantized at -0.33/+0.33."""
    date_to_label = dict(zip(df_train['date'], y_train))
    dates_val = pd.to_datetime(df_val['date']).values
    
    preds = []
    for curr_date in dates_val:
        # Collect labels from past 7 days
        labels_window = []
        for d in range(0, 7):
            check_date = curr_date - timedelta(days=d)
            check_str = pd.Timestamp(check_date).strftime('%Y-%m-%d')
            if check_str in date_to_label:
                labels_window.append(float(date_to_label[check_str]))
        
        if labels_window:
            mean_val = np.mean(labels_window)
            # Quantize at thresholds
            if mean_val >= 0.33:
                pred = -1
            elif mean_val <= -0.33:
                pred = 1
            else:
                pred = 0
            preds.append(pred)
        else:
            # Fallback
            preds.append(0)
    
    return np.array(preds)


def predict_rule_based(pbsi_score_val: np.ndarray) -> np.ndarray:
    """
    Rule-based clinical prediction using pbsi_score.
    
    Mapping:
      pbsi_score >= +0.5 → -1 (unstable)
      pbsi_score <= -0.5 → +1 (stable)
      else → 0 (neutral)
    """
    preds = np.zeros_like(pbsi_score_val, dtype=int)
    preds[pbsi_score_val >= 0.5] = -1
    preds[pbsi_score_val <= -0.5] = 1
    return preds


def mcnemar_test(y_true: np.ndarray, y_pred_a: np.ndarray, y_pred_b: np.ndarray) -> float:
    """Compute McNemar test p-value (for 2-class only).
    
    McNemar test: chi2 = (b - c)^2 / (b + c), where:
      b = #cases correct in A but not B
      c = #cases correct in B but not A
    """
    if len(np.unique(y_true)) != 2:
        return np.nan
    
    # Disagreements: A correct vs B correct
    a_correct = y_pred_a == y_true
    b_correct = y_pred_b == y_true
    
    b = np.sum(a_correct & ~b_correct)  # A correct, B wrong
    c = np.sum(~a_correct & b_correct)  # A wrong, B correct
    
    if b + c == 0:
        return 1.0
    
    try:
        # Chi-squared test: (b - c)^2 / (b + c)
        chi2_stat = (b - c) ** 2 / (b + c)
        # P-value from chi-squared distribution with 1 df
        p_value = 1.0 - stats.chi2.cdf(chi2_stat, df=1)
        return float(p_value)
    except Exception:
        return np.nan


def save_confusion_matrix(
    y_true: np.ndarray, y_pred: np.ndarray, model_name: str,
    fold_idx: int, label_col: str, output_dir: Path
) -> None:
    """Save confusion matrix as PNG."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(f'{model_name} - Fold {fold_idx} - {label_col}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    out_file = output_dir / f'fold_{fold_idx}_{model_name}_{label_col}.png'
    plt.savefig(out_file, dpi=100, bbox_inches='tight')
    plt.close()
    logger.info(f"  Saved: {out_file}")


def _get_proba(model, x_val_scaled):
    """Safely get probabilities from model."""
    try:
        return model.predict_proba(x_val_scaled)
    except Exception:
        return None


def _eval_all_models(
    label_col, y_val, y_pred_dummy, y_pred_naive, y_pred_ma7,
    y_pred_rule, y_pred_lr, y_proba_dummy, y_proba_lr, fold_idx
) -> Dict:
    """Evaluate all models for a fold."""
    fold_result = {'fold': fold_idx}
    
    for model_name, y_pred, y_proba in [
        ('dummy', y_pred_dummy, y_proba_dummy),
        ('naive', y_pred_naive, None),
        ('ma7', y_pred_ma7, None),
        ('rule', y_pred_rule, None),
        ('lr', y_pred_lr, y_proba_lr),
    ]:
        if label_col == 'label_2cls':
            metrics = compute_metrics_2cls(y_val, y_pred, y_proba)
        else:
            metrics = compute_metrics_3cls(y_val, y_pred)
        
        for metric_name, metric_val in metrics.items():
            fold_result[f'{model_name}_{metric_name}'] = metric_val
        
        if label_col == 'label_2cls' and model_name != 'dummy':
            mcnemar_p = mcnemar_test(y_val, y_pred_lr, y_pred)
            fold_result[f'{model_name}_mcnemar_p'] = mcnemar_p
    
    if label_col == 'label_2cls':
        mcnemar_p_lr_vs_dummy = mcnemar_test(y_val, y_pred_lr, y_pred_dummy)
        fold_result['lr_mcnemar_p'] = mcnemar_p_lr_vs_dummy
    
    return fold_result


def _eval_dummy_only(label_col, y_val, y_pred_dummy, y_proba_dummy, fold_idx) -> Dict:
    """Evaluate Dummy model only."""
    fold_result = {'fold': fold_idx}
    
    if label_col == 'label_2cls':
        metrics = compute_metrics_2cls(y_val, y_pred_dummy, y_proba_dummy)
    else:
        metrics = compute_metrics_3cls(y_val, y_pred_dummy)
    
    for metric_name, metric_val in metrics.items():
        fold_result[f'dummy_{metric_name}'] = metric_val
    
    return fold_result


def run_temporal_cv(
    df_labeled: pd.DataFrame,
    feature_cols: List[str],
    label_col: str = 'label_3cls',
    n_folds: int = 6,
    output_dir: Optional[Path] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run temporal CV with all 5 baselines.
    
    Returns: (results_df_all_baselines, results_df_dummy_only)
    """
    logger.info(f"\n{'='*80}")
    logger.info(f"Temporal CV: {label_col} (calendar-based, 4m/2m folds)")
    logger.info('='*80)
    
    # Anti-degeneration check: global label distribution
    label_dist = df_labeled[label_col].value_counts()
    if len(label_dist) < 2:
        raise ValueError(
            f"Label {label_col} has only {len(label_dist)} class(es). "
            "Cannot run CV with degenerate labels."
        )
    logger.info(f"Global {label_col} distribution: {dict(label_dist)}")
    
    # Check quality
    if 'pbsi_quality' in df_labeled.columns:
        n_degraded = (df_labeled['pbsi_quality'] < 1.0).sum()
    else:
        n_degraded = 0
    logger.info(f"Days with pbsi_quality < 1.0: {n_degraded}/{len(df_labeled)}")
    
    folds = calendar_cv_folds(df_labeled, n_folds=n_folds)
    
    results_all = []
    results_dummy = []
    
    for fold_idx, df_train, df_val, train_str, val_str in folds:
        logger.info(f"\nFold {fold_idx}")
        
        y_train = df_train[label_col].values
        y_val = df_val[label_col].values
        
        # Anti-degeneration: check validation fold
        val_dist = pd.Series(y_val).value_counts()
        if len(val_dist) < 1 or (label_col == 'label_3cls' and len(val_dist) < 2):
            logger.error(
                f"  Degenerate validation fold: {dict(val_dist)}. Aborting."
            )
            raise ValueError(
                f"Fold {fold_idx} validation has {len(val_dist)} class(es). "
                "Cannot evaluate."
            )
        
        logger.info(f"  Train dist: {dict(pd.Series(y_train).value_counts())}")
        logger.info(f"  Val dist: {dict(pd.Series(y_val).value_counts())}")
        
        # Features
        x_train = df_train[feature_cols].copy()
        x_val = df_val[feature_cols].copy()
        x_train_scaled, x_val_scaled = prepare_features(x_train, x_val)
        
        # Fit all models
        dummy_model = fit_dummy(x_train_scaled, y_train)
        fit_naive_yesterday(df_train, y_train)  # Uses df_train/y_train internally
        fit_moving_avg_7d(df_train, y_train)  # Uses df_train/y_train internally
        fit_rule_based()
        lr_model = fit_logistic_regression(x_train_scaled, y_train)
        
        # Predictions
        y_pred_dummy = dummy_model.predict(x_val_scaled)
        y_pred_naive = predict_naive_yesterday(df_train, df_val, y_train)
        y_pred_ma7 = predict_moving_avg_7d(df_train, df_val, y_train)
        y_pred_rule = predict_rule_based(df_val['pbsi_score'].values)
        y_pred_lr = lr_model.predict(x_val_scaled)
        
        # Get probability estimates
        y_proba_dummy = _get_proba(dummy_model, x_val_scaled)
        y_proba_lr = _get_proba(lr_model, x_val_scaled)
        
        # Process all models and compute results
        fold_result_all = _eval_all_models(
            label_col, y_val, y_pred_dummy, y_pred_naive, y_pred_ma7,
            y_pred_rule, y_pred_lr, y_proba_dummy, y_proba_lr, fold_idx
        )
        results_all.append(fold_result_all)
        
        # Process Dummy only
        fold_result_dummy = _eval_dummy_only(
            label_col, y_val, y_pred_dummy, y_proba_dummy, fold_idx
        )
        results_dummy.append(fold_result_dummy)
        
        # Save confusion matrices
        if output_dir:
            for model_name, y_pred in [
                ('lr', y_pred_lr),
                ('rule', y_pred_rule),
                ('dummy', y_pred_dummy),
            ]:
                save_confusion_matrix(
                    y_val, y_pred, model_name, fold_idx, label_col, output_dir
                )
    
    if not results_all:
        raise ValueError(f"No valid folds for {label_col}")
    
    # Build DataFrames
    df_results_all = pd.DataFrame(results_all)
    df_results_dummy = pd.DataFrame(results_dummy)
    
    # Add mean/std
    df_results_all = _add_mean_std(df_results_all)
    df_results_dummy = _add_mean_std(df_results_dummy)
    
    logger.info(f"\n{label_col} Results (All Baselines):")
    logger.info(df_results_all.to_string(index=False))
    logger.info(f"\n{label_col} Results (Dummy Only):")
    logger.info(df_results_dummy.to_string(index=False))
    
    # Save
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # All baselines
        out_file_all = output_dir / f'baselines_label_{label_col}.csv'
        df_results_all.to_csv(out_file_all, index=False)
        logger.info(f"Saved: {out_file_all}")
        
        # Dummy only
        out_file_dummy = output_dir / f'baseline_stratified_{label_col}.csv'
        df_results_dummy.to_csv(out_file_dummy, index=False)
        logger.info(f"Saved: {out_file_dummy}")
    
    return df_results_all, df_results_dummy


def _add_mean_std(results_df: pd.DataFrame) -> pd.DataFrame:
    """Add mean and std rows."""
    numeric_cols = [c for c in results_df.columns if c != 'fold']
    mean_dict: Dict = {'fold': 'MEAN'}
    std_dict: Dict = {'fold': 'STD'}
    
    for col in numeric_cols:
        mean_dict[col] = float(results_df[col].mean())
        std_dict[col] = float(results_df[col].std())
    
    return pd.concat([
        results_df,
        pd.DataFrame([mean_dict]),
        pd.DataFrame([std_dict]),
    ], ignore_index=True)
