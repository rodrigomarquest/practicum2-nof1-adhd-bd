"""
ML Extended Utilities Module

Shared utilities for ML6-Extended and ML7-Extended pipelines.
Provides:
- Model selection loading
- Temporal CV fold creation
- Anti-leak column handling
- Scaler fitting logic (per-fold)

Author: GitHub Copilot
Date: 2025-12-08
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# =============================================================================
# Anti-Leak Column Definitions
# =============================================================================

# PBSI intermediate outputs - derived from labels, leak target info
ANTI_LEAK_COLUMNS = [
    'pbsi_quality',
    'sleep_sub',
    'cardio_sub',
    'activity_sub',
    # PBSI labels (old targets)
    'label_3cls',
    'label_2cls',
    'label_clinical',
]

# Direct targets - NEVER use as features
TARGET_COLUMNS = [
    'som_category_3class',
    'som_binary',
]

# Metadata - identifiers, not predictive signals
METADATA_COLUMNS = [
    'date',
    'segment_id',
    'som_vendor',
]

# Combined exclusion list
ALL_EXCLUDED_COLUMNS = ANTI_LEAK_COLUMNS + TARGET_COLUMNS + METADATA_COLUMNS


# =============================================================================
# Model Selection Loading
# =============================================================================

def load_model_selection(output_dir: Path) -> Dict[str, Any]:
    """
    Load model_selection.json from the snapshot directory.
    
    Args:
        output_dir: Base output directory (data/ai/<PID>/<SNAPSHOT>)
    
    Returns:
        Dict with model selection containing:
        - ml6.selected_fs: e.g., 'FS-B'
        - ml6.selected_target: e.g., 'binary'
        - ml6.features: list of feature column names
        - ml7.selected_config: e.g., 'CFG-3'
        - ml7.config_params.seq_len: sequence length
    
    Raises:
        FileNotFoundError: If model_selection.json doesn't exist
        ValueError: If required fields are missing
    """
    # model_selection.json is saved at the snapshot root level
    model_selection_path = output_dir / 'model_selection.json'
    
    if not model_selection_path.exists():
        # Fallback: check in ml6 subdirectory (older format)
        model_selection_path = output_dir / 'ml6' / 'model_selection.json'
    
    if not model_selection_path.exists():
        raise FileNotFoundError(f"model_selection.json not found at {output_dir}")
    
    with open(model_selection_path, 'r') as f:
        model_selection = json.load(f)
    
    # Validate required fields
    required_ml6 = ['selected_fs', 'selected_target', 'features']
    if 'ml6' not in model_selection:
        raise ValueError("model_selection.json missing 'ml6' section")
    
    for field in required_ml6:
        if field not in model_selection['ml6']:
            raise ValueError(f"model_selection.json missing 'ml6.{field}'")
    
    # Validate ML7 section
    if 'ml7' not in model_selection:
        logger.warning("model_selection.json missing 'ml7' section, using defaults")
        model_selection['ml7'] = {
            'selected_config': 'CFG-3',
            'config_params': {'seq_len': 14},
            'features': model_selection['ml6']['features']
        }
    
    # Ensure seq_len is present
    if 'config_params' not in model_selection['ml7']:
        model_selection['ml7']['config_params'] = {'seq_len': 14}
    if 'seq_len' not in model_selection['ml7']['config_params']:
        model_selection['ml7']['config_params']['seq_len'] = 14
    
    logger.info(f"[ML-Extended] Loaded model_selection.json")
    logger.info(f"  ML6: {model_selection['ml6']['selected_fs']} × {model_selection['ml6']['selected_target']}")
    logger.info(f"  ML7: {model_selection['ml7'].get('selected_config', 'CFG-3')} (seq_len={model_selection['ml7']['config_params']['seq_len']})")
    
    return model_selection


def get_target_column(selected_target: str) -> str:
    """
    Convert selected_target string to actual column name.
    
    Args:
        selected_target: 'binary' or '3class'
    
    Returns:
        Column name: 'som_binary' or 'som_category_3class'
    """
    target_map = {
        'binary': 'som_binary',
        '3class': 'som_category_3class',
        'som_binary': 'som_binary',
        'som_category_3class': 'som_category_3class',
    }
    return target_map.get(selected_target, 'som_binary')


# =============================================================================
# Anti-Leak Filtering
# =============================================================================

def apply_anti_leak_filter(df: pd.DataFrame, feature_cols: List[str]) -> List[str]:
    """
    Filter feature columns to remove any anti-leak columns.
    
    Args:
        df: DataFrame with all columns
        feature_cols: List of candidate feature columns
    
    Returns:
        Filtered list of safe feature columns
    """
    excluded = set(ALL_EXCLUDED_COLUMNS)
    safe_features = [f for f in feature_cols if f not in excluded and f in df.columns]
    
    removed = set(feature_cols) - set(safe_features)
    if removed:
        logger.warning(f"[Anti-Leak] Removed columns: {removed}")
    
    return safe_features


def validate_no_leakage(X: np.ndarray, feature_names: List[str]) -> bool:
    """
    Validate that no anti-leak columns are present in feature matrix.
    
    Args:
        X: Feature matrix
        feature_names: List of feature column names (same order as X columns)
    
    Returns:
        True if validation passes
    
    Raises:
        ValueError: If leakage detected
    """
    for col in feature_names:
        if col in TARGET_COLUMNS:
            raise ValueError(f"Target column '{col}' found in features - LEAKAGE!")
        if col in ANTI_LEAK_COLUMNS:
            raise ValueError(f"Anti-leak column '{col}' found in features - LEAKAGE!")
    
    return True


# =============================================================================
# Temporal CV Fold Creation
# =============================================================================

def create_temporal_folds(
    n_samples: int,
    n_folds: int = 6,
    min_train_samples: int = 10,
    min_val_samples: int = 3
) -> List[Tuple[List[int], List[int]]]:
    """
    Create temporal CV folds for time-series data.
    
    Uses simple temporal split where each fold's validation set is a 
    contiguous block, maintaining temporal order.
    
    Args:
        n_samples: Total number of samples
        n_folds: Number of folds (default: 6)
        min_train_samples: Minimum training samples per fold
        min_val_samples: Minimum validation samples per fold
    
    Returns:
        List of (train_indices, val_indices) tuples
    """
    if n_samples < n_folds * min_val_samples:
        # Fall back to fewer folds
        n_folds = max(2, n_samples // min_val_samples)
        logger.warning(f"[Temporal CV] Reduced to {n_folds} folds due to small sample size")
    
    fold_size = n_samples // n_folds
    folds = []
    
    for fold_idx in range(n_folds):
        val_start = fold_idx * fold_size
        val_end = min((fold_idx + 1) * fold_size, n_samples)
        
        val_idx = list(range(val_start, val_end))
        train_idx = [i for i in range(n_samples) if i not in val_idx]
        
        # Validate fold sizes
        if len(train_idx) < min_train_samples or len(val_idx) < min_val_samples:
            logger.warning(f"[Temporal CV] Fold {fold_idx} skipped: train={len(train_idx)}, val={len(val_idx)}")
            continue
        
        folds.append((train_idx, val_idx))
    
    logger.info(f"[Temporal CV] Created {len(folds)} folds from {n_samples} samples")
    return folds


# =============================================================================
# Per-Fold Scaler Fitting
# =============================================================================

def fit_scaler_per_fold(
    X: np.ndarray,
    train_idx: List[int],
    val_idx: List[int],
    scaler_class=None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fit scaler on training data only, then transform both train and val.
    
    This prevents data leakage from validation set into scaling parameters.
    
    Args:
        X: Full feature matrix (n_samples, n_features)
        train_idx: Training sample indices
        val_idx: Validation sample indices
        scaler_class: Scaler class (default: StandardScaler)
    
    Returns:
        (X_train_scaled, X_val_scaled)
    """
    if scaler_class is None:
        from sklearn.preprocessing import StandardScaler
        scaler_class = StandardScaler
    
    X_train = X[train_idx]
    X_val = X[val_idx]
    
    # Fit ONLY on training data
    scaler = scaler_class()
    scaler.fit(X_train)
    
    # Transform both
    X_train_scaled = scaler.transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    return X_train_scaled, X_val_scaled


# =============================================================================
# Model Leaderboard Management
# =============================================================================

def update_model_leaderboard(
    output_dir: Path,
    pipeline_name: str,
    best_model: str,
    f1_macro: float,
    additional_metrics: Optional[Dict] = None
) -> Dict:
    """
    Update the global model leaderboard with new results.
    
    Args:
        output_dir: Base output directory (ai/local/<PID>/<SNAPSHOT>)
        pipeline_name: 'ml6_extended' or 'ml7_extended'
        best_model: Name of best performing model
        f1_macro: Best F1-macro score
        additional_metrics: Optional dict of extra metrics
    
    Returns:
        Updated leaderboard dict
    """
    leaderboard_path = output_dir / 'model_leaderboard.json'
    
    # Load existing or create new
    if leaderboard_path.exists():
        with open(leaderboard_path, 'r') as f:
            leaderboard = json.load(f)
    else:
        leaderboard = {
            'created': datetime.now().isoformat(),
            'participant': output_dir.parent.name if output_dir.parent else 'unknown',
            'snapshot': output_dir.name,
        }
    
    # Update with new results
    leaderboard[pipeline_name] = {
        'best_model': best_model,
        'f1_macro': float(f1_macro),
        'updated': datetime.now().isoformat(),
    }
    
    if additional_metrics:
        leaderboard[pipeline_name].update(additional_metrics)
    
    # Save
    with open(leaderboard_path, 'w') as f:
        json.dump(leaderboard, f, indent=2)
    
    logger.info(f"[Leaderboard] Updated {leaderboard_path}: {pipeline_name} best={best_model} (F1={f1_macro:.4f})")
    
    return leaderboard


# =============================================================================
# Data Loading Utilities
# =============================================================================

def load_feature_universe(output_dir: Path) -> pd.DataFrame:
    """
    Load the feature universe CSV.
    
    Args:
        output_dir: Base output directory (data/ai/<PID>/<SNAPSHOT>)
    
    Returns:
        DataFrame with all features and targets
    """
    universe_path = output_dir / 'ml6' / 'features_daily_ml_universe.csv'
    
    if not universe_path.exists():
        # Fallback to standard ml6 features
        universe_path = output_dir / 'ml6' / 'features_daily_ml6.csv'
    
    if not universe_path.exists():
        raise FileNotFoundError(f"Feature universe not found at {universe_path}")
    
    df = pd.read_csv(universe_path)
    
    # Parse date column
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
    
    logger.info(f"[Data] Loaded feature universe: {len(df)} samples, {len(df.columns)} columns")
    
    return df


def prepare_features_and_target(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Prepare feature matrix and target vector with anti-leak filtering.
    
    Args:
        df: DataFrame with features and targets
        feature_cols: List of feature column names
        target_col: Target column name
    
    Returns:
        (X, y, safe_feature_cols)
    """
    # Apply anti-leak filter
    safe_features = apply_anti_leak_filter(df, feature_cols)
    
    # Validate no leakage
    validate_no_leakage(np.array([]), safe_features)
    
    # Extract X and y
    X = df[safe_features].values
    y = df[target_col].values
    
    # Handle NaN in target
    valid_mask = ~np.isnan(y)
    if not valid_mask.all():
        logger.warning(f"[Data] Removed {(~valid_mask).sum()} samples with NaN target")
        X = X[valid_mask]
        y = y[valid_mask]
    
    return X, y, safe_features


# =============================================================================
# Metrics Computation
# =============================================================================

def compute_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Compute standard classification metrics.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
    
    Returns:
        Dict with f1_macro, balanced_accuracy, cohen_kappa, precision_macro, recall_macro
    """
    from sklearn.metrics import (
        f1_score, balanced_accuracy_score, cohen_kappa_score,
        precision_score, recall_score
    )
    
    return {
        'f1_macro': float(f1_score(y_true, y_pred, average='macro', zero_division=0)),
        'balanced_accuracy': float(balanced_accuracy_score(y_true, y_pred)),
        'cohen_kappa': float(cohen_kappa_score(y_true, y_pred)),
        'precision_macro': float(precision_score(y_true, y_pred, average='macro', zero_division=0)),
        'recall_macro': float(recall_score(y_true, y_pred, average='macro', zero_division=0)),
    }


def aggregate_fold_metrics(fold_metrics: List[Dict[str, float]]) -> Dict[str, float]:
    """
    Aggregate metrics across folds.
    
    Args:
        fold_metrics: List of per-fold metric dicts
    
    Returns:
        Dict with mean and std for each metric
    """
    if not fold_metrics:
        return {}
    
    metrics = {}
    keys = fold_metrics[0].keys()
    
    for key in keys:
        values = [m[key] for m in fold_metrics if key in m]
        if values:
            metrics[f'mean_{key}'] = float(np.mean(values))
            metrics[f'std_{key}'] = float(np.std(values))
    
    return metrics
