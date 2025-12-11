"""
ML7-Extended Pipeline Module

Implements multiple sequence-based deep learning architectures for SoM classification:
- LSTM (CFG-3 configuration)
- GRU
- 1D-CNN (Conv1D)
- CNN-LSTM Hybrid

All models:
- Load selected feature set, target, and sequence length from model_selection.json
- Apply anti-leak column filtering BEFORE sequence creation
- Fit scaler ONLY on training data per fold (before sequence creation)
- Use consistent temporal CV

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

from src.etl.ml_extended_utils import (
    load_model_selection,
    load_feature_universe,
    get_target_column,
    apply_anti_leak_filter,
    validate_no_leakage,
    compute_classification_metrics,
    aggregate_fold_metrics,
    update_model_leaderboard,
)

from src.etl.ml_metrics_extended import (
    compute_metrics_extended,
    compute_naive_baselines_sequence,
    aggregate_extended_metrics,
    export_per_class_metrics_csv,
    export_confusion_matrices_json,
    export_baseline_comparison_csv,
    setup_metrics_output_dir,
    ExtendedMetrics,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Sequence Creation (Anti-Leak Safe)
# =============================================================================

def create_sequences(
    X: np.ndarray,
    y: np.ndarray,
    seq_len: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sequences for LSTM/GRU/CNN models.
    
    Uses sliding window approach where each sequence predicts the label
    of the last timestep.
    
    Args:
        X: Feature matrix (n_samples, n_features) - ALREADY SCALED
        y: Target vector (n_samples,)
        seq_len: Sequence length
    
    Returns:
        (X_seq, y_seq) where:
        - X_seq: (n_sequences, seq_len, n_features)
        - y_seq: (n_sequences,)
    """
    n_samples, n_features = X.shape
    
    if n_samples < seq_len:
        raise ValueError(f"Not enough samples ({n_samples}) for sequence length ({seq_len})")
    
    n_sequences = n_samples - seq_len + 1
    X_seq = np.zeros((n_sequences, seq_len, n_features))
    y_seq = np.zeros(n_sequences)
    
    for i in range(n_sequences):
        X_seq[i] = X[i:i + seq_len]
        y_seq[i] = y[i + seq_len - 1]  # Predict label of last timestep
    
    return X_seq, y_seq


def create_temporal_sequence_folds(
    n_samples: int,
    seq_len: int,
    n_folds: int = 6,
    min_train_seqs: int = 10,
    min_val_seqs: int = 3
) -> List[Tuple[List[int], List[int]]]:
    """
    Create temporal folds for sequence data.
    
    Folds are created at the sample level, then sequences are derived.
    This ensures no data leakage between folds.
    
    Args:
        n_samples: Total samples (before sequence creation)
        seq_len: Sequence length
        n_folds: Number of folds
        min_train_seqs: Minimum training sequences per fold
        min_val_seqs: Minimum validation sequences per fold
    
    Returns:
        List of (train_sample_indices, val_sample_indices)
    """
    # Calculate effective samples after sequence creation
    n_effective = n_samples - seq_len + 1
    
    if n_effective < n_folds * min_val_seqs:
        n_folds = max(2, n_effective // min_val_seqs)
        logger.warning(f"[Temporal CV] Reduced to {n_folds} folds for sequences")
    
    fold_size = n_effective // n_folds
    folds = []
    
    for fold_idx in range(n_folds):
        # Val indices in sequence space
        val_seq_start = fold_idx * fold_size
        val_seq_end = min((fold_idx + 1) * fold_size, n_effective)
        
        # Convert to sample space (for scaling)
        # Val samples include the full sequence window
        val_sample_start = val_seq_start
        val_sample_end = val_seq_end + seq_len - 1
        
        val_sample_idx = list(range(val_sample_start, min(val_sample_end, n_samples)))
        train_sample_idx = [i for i in range(n_samples) if i not in val_sample_idx]
        
        if len(train_sample_idx) < seq_len + min_train_seqs:
            logger.warning(f"[Temporal CV] Fold {fold_idx} skipped: insufficient train samples")
            continue
        
        folds.append((train_sample_idx, val_sample_idx))
    
    logger.info(f"[Temporal CV] Created {len(folds)} folds for {n_samples} samples (seq_len={seq_len})")
    return folds


# =============================================================================
# Model Architectures
# =============================================================================

def build_lstm_model(
    seq_len: int,
    n_features: int,
    n_classes: int,
    lstm_units: int = 32,
    dense_units: int = 32,
    dropout: float = 0.4
) -> Any:
    """
    Build LSTM model.
    
    Uses softmax + sparse_categorical_crossentropy for ALL n_classes values
    (including n_classes=2) for consistency with baseline ML7.
    """
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    
    model = Sequential([
        LSTM(lstm_units, input_shape=(seq_len, n_features), return_sequences=False),
        Dropout(dropout),
        Dense(dense_units, activation='relu'),
        Dropout(dropout),
        Dense(n_classes, activation='softmax'),  # Always softmax for consistency
    ])
    
    # Always use sparse_categorical_crossentropy (consistent with baseline ML7)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    return model


def build_gru_model(
    seq_len: int,
    n_features: int,
    n_classes: int,
    gru_units: int = 32,
    dense_units: int = 32,
    dropout: float = 0.4
) -> Any:
    """
    Build GRU model.
    
    Uses softmax + sparse_categorical_crossentropy for ALL n_classes values
    (including n_classes=2) for consistency with baseline ML7.
    """
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import GRU, Dense, Dropout
    
    model = Sequential([
        GRU(gru_units, input_shape=(seq_len, n_features), return_sequences=False),
        Dropout(dropout),
        Dense(dense_units, activation='relu'),
        Dropout(dropout),
        Dense(n_classes, activation='softmax'),  # Always softmax for consistency
    ])
    
    # Always use sparse_categorical_crossentropy (consistent with baseline ML7)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    return model


def build_conv1d_model(
    seq_len: int,
    n_features: int,
    n_classes: int,
    filters: int = 32,
    kernel_size: int = 3,
    dense_units: int = 32,
    dropout: float = 0.4
) -> Any:
    """
    Build 1D-CNN model.
    
    Uses softmax + sparse_categorical_crossentropy for ALL n_classes values
    (including n_classes=2) for consistency with baseline ML7.
    """
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Conv1D, GlobalMaxPooling1D, Dense, Dropout
    
    model = Sequential([
        Conv1D(filters, kernel_size, activation='relu', input_shape=(seq_len, n_features)),
        Conv1D(filters * 2, kernel_size, activation='relu'),
        GlobalMaxPooling1D(),
        Dropout(dropout),
        Dense(dense_units, activation='relu'),
        Dropout(dropout),
        Dense(n_classes, activation='softmax'),  # Always softmax for consistency
    ])
    
    # Always use sparse_categorical_crossentropy (consistent with baseline ML7)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    return model


def build_cnn_lstm_model(
    seq_len: int,
    n_features: int,
    n_classes: int,
    filters: int = 32,
    kernel_size: int = 3,
    lstm_units: int = 32,
    dense_units: int = 32,
    dropout: float = 0.4
) -> Any:
    """
    Build CNN-LSTM hybrid model.
    
    Uses softmax + sparse_categorical_crossentropy for ALL n_classes values
    (including n_classes=2) for consistency with baseline ML7.
    """
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Conv1D, LSTM, Dense, Dropout
    
    model = Sequential([
        Conv1D(filters, kernel_size, activation='relu', input_shape=(seq_len, n_features)),
        Dropout(dropout),
        LSTM(lstm_units, return_sequences=False),
        Dropout(dropout),
        Dense(dense_units, activation='relu'),
        Dropout(dropout),
        Dense(n_classes, activation='softmax'),  # Always softmax for consistency
    ])
    
    # Always use sparse_categorical_crossentropy (consistent with baseline ML7)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    return model


# =============================================================================
# Model Configurations
# =============================================================================

def get_model_configs(seq_len: int, n_features: int, n_classes: int) -> Dict[str, Dict]:
    """
    Get all ML7-Extended model configurations.
    
    Args:
        seq_len: Sequence length from model_selection.json
        n_features: Number of features
        n_classes: Number of target classes
    
    Returns:
        Dict mapping model names to config dicts
    """
    return {
        'LSTM': {
            'builder': build_lstm_model,
            'params': {
                'seq_len': seq_len,
                'n_features': n_features,
                'n_classes': n_classes,
                'lstm_units': 32,
                'dense_units': 32,
                'dropout': 0.4,
            },
        },
        'GRU': {
            'builder': build_gru_model,
            'params': {
                'seq_len': seq_len,
                'n_features': n_features,
                'n_classes': n_classes,
                'gru_units': 32,
                'dense_units': 32,
                'dropout': 0.4,
            },
        },
        'Conv1D': {
            'builder': build_conv1d_model,
            'params': {
                'seq_len': seq_len,
                'n_features': n_features,
                'n_classes': n_classes,
                'filters': 32,
                'kernel_size': 3,
                'dense_units': 32,
                'dropout': 0.4,
            },
        },
        'CNN_LSTM': {
            'builder': build_cnn_lstm_model,
            'params': {
                'seq_len': seq_len,
                'n_features': n_features,
                'n_classes': n_classes,
                'filters': 32,
                'kernel_size': 3,
                'lstm_units': 32,
                'dense_units': 32,
                'dropout': 0.4,
            },
        },
    }


# =============================================================================
# Training Functions
# =============================================================================

def compute_class_weights(y: np.ndarray) -> Dict[int, float]:
    """Compute class weights for imbalanced data."""
    from sklearn.utils.class_weight import compute_class_weight
    
    classes = np.unique(y)
    weights = compute_class_weight('balanced', classes=classes, y=y)
    return dict(zip(classes.astype(int), weights))


def train_sequence_model_cv(
    model_name: str,
    model_config: Dict,
    X: np.ndarray,
    y: np.ndarray,
    seq_len: int,
    folds: List[Tuple[List[int], List[int]]],
    epochs: int = 50,
    batch_size: int = 16,
    early_stopping: bool = True,
    class_labels: Optional[List[int]] = None
) -> Dict[str, Any]:
    """
    Train a sequence model using cross-validation with extended metrics.
    
    IMPORTANT: Scaler is fit ONLY on training data per fold.
    
    Args:
        model_name: Name of the model
        model_config: Model configuration
        X: Feature matrix (n_samples, n_features) - UNSCALED
        y: Target vector (n_samples,)
        seq_len: Sequence length
        folds: List of (train_sample_idx, val_sample_idx)
        epochs: Training epochs
        batch_size: Batch size
        early_stopping: Whether to use early stopping
        class_labels: Optional list of class labels for consistent metrics
    
    Returns:
        Dict with model results including extended metrics
    """
    from sklearn.preprocessing import StandardScaler
    
    # Determine class labels if not provided
    if class_labels is None:
        class_labels = sorted(set(y.tolist()))
    
    fold_metrics = []
    fold_extended_metrics = []
    fold_histories = []
    fold_confusion_matrices = []
    
    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        try:
            # =========================================================
            # CRITICAL: Fit scaler ONLY on training data
            # =========================================================
            X_train_raw = X[train_idx]
            X_val_raw = X[val_idx]
            y_train_raw = y[train_idx]
            y_val_raw = y[val_idx]
            
            scaler = StandardScaler()
            scaler.fit(X_train_raw)  # FIT ONLY ON TRAIN
            
            X_train_scaled = scaler.transform(X_train_raw)
            X_val_scaled = scaler.transform(X_val_raw)
            
            # =========================================================
            # Create sequences AFTER scaling
            # =========================================================
            if len(X_train_scaled) < seq_len:
                logger.warning(f"[ML7-Ext] {model_name} fold {fold_idx}: insufficient train samples for sequences")
                continue
            
            X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train_raw, seq_len)
            
            if len(X_val_scaled) < seq_len:
                logger.warning(f"[ML7-Ext] {model_name} fold {fold_idx}: insufficient val samples for sequences")
                continue
            
            X_val_seq, y_val_seq = create_sequences(X_val_scaled, y_val_raw, seq_len)
            
            # Check class distribution
            unique_train = np.unique(y_train_seq)
            if len(unique_train) < 2:
                logger.warning(f"[ML7-Ext] {model_name} fold {fold_idx}: Only {len(unique_train)} class in train")
                continue
            
            # Compute class weights
            class_weights = compute_class_weights(y_train_seq)
            
            # Build model
            model = model_config['builder'](**model_config['params'])
            
            # Setup callbacks
            callbacks = []
            if early_stopping:
                from tensorflow.keras.callbacks import EarlyStopping
                callbacks.append(EarlyStopping(
                    monitor='val_loss',
                    patience=10,
                    restore_best_weights=True,
                    verbose=0
                ))
            
            # Train
            history = model.fit(
                X_train_seq, y_train_seq,
                validation_data=(X_val_seq, y_val_seq),
                epochs=epochs,
                batch_size=batch_size,
                class_weight=class_weights,
                callbacks=callbacks,
                verbose=0
            )
            
            # Predict
            y_pred_proba = model.predict(X_val_seq, verbose=0)
            
            # Always use argmax since we use softmax for all n_classes (including n_classes=2)
            y_pred = np.argmax(y_pred_proba, axis=1)
            
            # Compute basic metrics (backward compatible)
            metrics = compute_classification_metrics(y_val_seq, y_pred)
            metrics['fold_idx'] = fold_idx
            metrics['n_train_seqs'] = len(y_train_seq)
            metrics['n_val_seqs'] = len(y_val_seq)
            metrics['epochs_trained'] = len(history.history['loss'])
            
            fold_metrics.append(metrics)
            
            # Compute extended metrics with per-class breakdown
            ext_metrics = compute_metrics_extended(y_val_seq, y_pred, class_labels)
            fold_extended_metrics.append(ext_metrics)
            
            # Store confusion matrix
            fold_confusion_matrices.append(ext_metrics.confusion_matrix)
            
            fold_histories.append({
                'fold_idx': fold_idx,
                'final_train_loss': float(history.history['loss'][-1]),
                'final_val_loss': float(history.history['val_loss'][-1]),
            })
            
            # Clear session to free memory
            from tensorflow.keras import backend as K
            K.clear_session()
            
        except Exception as e:
            logger.warning(f"[ML7-Ext] {model_name} fold {fold_idx} failed: {e}")
            continue
    
    if not fold_metrics:
        return {
            'model_name': model_name,
            'status': 'failed',
            'reason': 'All folds failed',
        }
    
    # Aggregate basic metrics (backward compatible)
    agg_metrics = aggregate_fold_metrics(fold_metrics)
    
    # Aggregate extended metrics
    agg_extended = aggregate_extended_metrics(fold_extended_metrics)
    
    return {
        'model_name': model_name,
        'status': 'success',
        'n_folds_completed': len(fold_metrics),
        'per_fold_metrics': fold_metrics,
        'aggregate_metrics': agg_metrics,
        'extended_metrics': agg_extended,
        'confusion_matrices': fold_confusion_matrices,
        'class_labels': class_labels,
        'training_history': fold_histories,
    }


# =============================================================================
# Drift Analysis
# =============================================================================

def compute_drift_analysis(
    model_results: Dict[str, Dict],
    output_dir: Path
) -> str:
    """
    Compute per-fold drift analysis for sequence models.
    
    Args:
        model_results: Dict of model name -> results
        output_dir: Output directory
    
    Returns:
        Path to drift analysis markdown file
    """
    drift_path = output_dir / 'drift_analysis.md'
    
    with open(drift_path, 'w', encoding='utf-8') as f:
        f.write("# ML7-Extended Drift Analysis\n\n")
        f.write(f"**Generated**: {datetime.now().isoformat()}\n\n")
        f.write("## Per-Fold Performance Trends\n\n")
        f.write("This analysis shows how model performance varies across temporal folds.\n")
        f.write("Degradation in later folds may indicate concept drift.\n\n")
        
        for model_name, result in model_results.items():
            if result.get('status') != 'success':
                continue
            
            f.write(f"### {model_name}\n\n")
            f.write("| Fold | F1-macro | Balanced Acc | Train Seqs | Val Seqs |\n")
            f.write("|------|----------|--------------|------------|----------|\n")
            
            for fold in result.get('per_fold_metrics', []):
                f1 = fold.get('f1_macro', 0)
                ba = fold.get('balanced_accuracy', 0)
                n_train = fold.get('n_train_seqs', 0)
                n_val = fold.get('n_val_seqs', 0)
                f.write(f"| {fold['fold_idx']} | {f1:.4f} | {ba:.4f} | {n_train} | {n_val} |\n")
            
            # Compute drift indicator
            f1_values = [m['f1_macro'] for m in result.get('per_fold_metrics', [])]
            if len(f1_values) >= 3:
                first_half = np.mean(f1_values[:len(f1_values)//2])
                second_half = np.mean(f1_values[len(f1_values)//2:])
                drift = second_half - first_half
                
                f.write(f"\n**Drift Indicator**: {drift:+.4f} ")
                if drift < -0.05:
                    f.write("(⚠️ Performance degradation detected)\n")
                elif drift > 0.05:
                    f.write("(✅ Performance improvement over time)\n")
                else:
                    f.write("(→ Stable performance)\n")
            
            f.write("\n")
    
    logger.info(f"[ML7-Ext] Drift analysis: {drift_path}")
    return str(drift_path)


# =============================================================================
# Main Pipeline
# =============================================================================

def run_ml7_extended(
    participant: str,
    snapshot: str,
    output_base: Optional[Path] = None,
    epochs: int = 50,
    batch_size: int = 16
) -> Dict[str, Any]:
    """
    Run the ML7-Extended pipeline.
    
    Args:
        participant: Participant ID (e.g., 'P000001')
        snapshot: Snapshot date (e.g., '2025-12-08')
        output_base: Base output directory (default: ai/local)
        epochs: Training epochs per model
        batch_size: Batch size
    
    Returns:
        Dict with all model results and best model selection
    """
    logger.info("=" * 70)
    logger.info("[ML7-Extended] Starting ML7-Extended Pipeline")
    logger.info("=" * 70)
    
    # Check TensorFlow availability
    try:
        import tensorflow as tf
        tf.get_logger().setLevel('ERROR')  # Suppress TF warnings
        logger.info(f"[ML7-Ext] TensorFlow version: {tf.__version__}")
    except ImportError:
        logger.error("[ML7-Extended] TensorFlow not installed - pipeline cannot run")
        return {
            'status': 'error',
            'error': 'TensorFlow not installed',
        }
    
    # Setup paths
    if output_base is None:
        output_base = Path('data/ai')
    
    output_dir = output_base / participant / snapshot
    ml7_ext_dir = output_dir / 'ml7_extended'
    ml7_ext_dir.mkdir(parents=True, exist_ok=True)
    
    results = {
        'participant': participant,
        'snapshot': snapshot,
        'pipeline': 'ml7_extended',
        'started': datetime.now().isoformat(),
        'models': {},
    }
    
    try:
        # =====================================================================
        # Load model selection
        # =====================================================================
        model_selection = load_model_selection(output_dir)
        
        selected_fs = model_selection['ml7'].get('selected_fs', model_selection['ml6']['selected_fs'])
        selected_target = model_selection['ml7'].get('selected_target', model_selection['ml6']['selected_target'])
        feature_cols = model_selection['ml7'].get('features', model_selection['ml6']['features'])
        seq_len = model_selection['ml7']['config_params']['seq_len']
        target_col = get_target_column(selected_target)
        
        results['selected_fs'] = selected_fs
        results['selected_target'] = selected_target
        results['target_col'] = target_col
        results['seq_len'] = seq_len
        results['n_features'] = len(feature_cols)
        
        logger.info(f"[ML7-Ext] Using: {selected_fs} × {selected_target}")
        logger.info(f"[ML7-Ext] Sequence length: {seq_len}")
        logger.info(f"[ML7-Ext] Features: {len(feature_cols)}")
        
        # =====================================================================
        # Load and prepare data
        # =====================================================================
        df = load_feature_universe(output_dir)
        
        # Apply anti-leak filter BEFORE any processing
        safe_features = apply_anti_leak_filter(df, feature_cols)
        validate_no_leakage(np.array([]), safe_features)
        
        results['safe_features'] = safe_features
        
        # Extract X, y (UNSCALED - scaling per fold)
        X = df[safe_features].values
        y = df[target_col].values
        
        # Remove NaN targets
        valid_mask = ~np.isnan(y)
        X = X[valid_mask]
        y = y[valid_mask]
        
        n_samples = len(y)
        n_features = len(safe_features)
        n_classes = len(np.unique(y))
        
        results['n_samples'] = n_samples
        results['n_classes'] = n_classes
        
        logger.info(f"[ML7-Ext] Samples: {n_samples}")
        logger.info(f"[ML7-Ext] Classes: {n_classes}")
        logger.info(f"[ML7-Ext] Target distribution: {dict(zip(*np.unique(y, return_counts=True)))}")
        
        # Check if we have enough data
        if n_samples < seq_len * 2:
            raise ValueError(f"Insufficient data: {n_samples} samples < {seq_len * 2} (2 × seq_len)")
        
        # =====================================================================
        # Create temporal folds
        # =====================================================================
        folds = create_temporal_sequence_folds(n_samples, seq_len, n_folds=6)
        results['n_folds'] = len(folds)
        
        # Determine class labels
        class_labels = sorted(set(y.tolist()))
        results['class_labels'] = class_labels
        
        # =====================================================================
        # Compute Naïve Baselines (sequence-adapted)
        # =====================================================================
        logger.info("[ML7-Ext] Computing naïve baselines...")
        
        from sklearn.preprocessing import StandardScaler
        
        baseline_fold_metrics = {
            'majority_class': [],
            'stratified_random': [],
            'persistence': [],
        }
        
        for fold_idx, (train_idx, val_idx) in enumerate(folds):
            # Scale for sequence creation
            X_train_raw = X[train_idx]
            X_val_raw = X[val_idx]
            y_train_raw = y[train_idx]
            y_val_raw = y[val_idx]
            
            scaler = StandardScaler()
            scaler.fit(X_train_raw)
            X_train_scaled = scaler.transform(X_train_raw)
            X_val_scaled = scaler.transform(X_val_raw)
            
            # Create sequences
            if len(X_train_scaled) < seq_len or len(X_val_scaled) < seq_len:
                continue
                
            _, y_train_seq = create_sequences(X_train_scaled, y_train_raw, seq_len)
            _, y_val_seq = create_sequences(X_val_scaled, y_val_raw, seq_len)
            
            # Compute baselines for this fold
            bl_metrics = compute_naive_baselines_sequence(
                y_train_seq=y_train_seq,
                y_val_seq=y_val_seq,
                seq_len=seq_len,
                class_labels=class_labels,
                random_seed=42 + fold_idx
            )
            
            for bl_name, bl_ext_metrics in bl_metrics.items():
                baseline_fold_metrics[bl_name].append(bl_ext_metrics)
        
        # Aggregate baseline metrics across folds
        baseline_results = {}
        for bl_name, fold_ext_list in baseline_fold_metrics.items():
            if fold_ext_list:
                agg_bl = aggregate_extended_metrics(fold_ext_list)
                baseline_results[bl_name] = agg_bl
                logger.info(f"  → {bl_name}: F1={agg_bl.get('mean_f1_macro', 0):.4f}")
        
        results['baselines'] = baseline_results
        
        # =====================================================================
        # Get model configurations
        # =====================================================================
        model_configs = get_model_configs(seq_len, n_features, n_classes)
        model_results = {}
        
        # =====================================================================
        # Train all models
        # =====================================================================
        for model_name, model_config in model_configs.items():
            logger.info(f"[ML7-Ext] Training {model_name}...")
            
            try:
                result = train_sequence_model_cv(
                    model_name, model_config,
                    X, y, seq_len, folds,
                    epochs=epochs,
                    batch_size=batch_size,
                    class_labels=class_labels
                )
                model_results[model_name] = result
                
                if result['status'] == 'success':
                    f1 = result['aggregate_metrics'].get('mean_f1_macro', 0)
                    logger.info(f"  → {model_name}: F1={f1:.4f}")
                else:
                    logger.info(f"  → {model_name}: {result['status']}")
                    
            except Exception as e:
                logger.error(f"[ML7-Ext] {model_name} failed: {e}")
                model_results[model_name] = {
                    'model_name': model_name,
                    'status': 'error',
                    'error': str(e),
                }
        
        results['models'] = model_results
        
        # =====================================================================
        # Select best model
        # =====================================================================
        successful_models = [
            (name, res) for name, res in model_results.items()
            if res.get('status') == 'success'
        ]
        
        if successful_models:
            # Sort by F1-macro (descending)
            successful_models.sort(
                key=lambda x: x[1]['aggregate_metrics'].get('mean_f1_macro', 0),
                reverse=True
            )
            
            best_name, best_result = successful_models[0]
            best_f1 = best_result['aggregate_metrics']['mean_f1_macro']
            
            results['best_model'] = {
                'name': best_name,
                'f1_macro': best_f1,
                'balanced_accuracy': best_result['aggregate_metrics'].get('mean_balanced_accuracy', 0),
                'cohen_kappa': best_result['aggregate_metrics'].get('mean_cohen_kappa', 0),
            }
            
            logger.info(f"[ML7-Ext] Best model: {best_name} (F1={best_f1:.4f})")
            
            # Update global leaderboard
            update_model_leaderboard(
                output_dir, 'ml7_extended', best_name, best_f1,
                {'n_models_tested': len(model_results), 'seq_len': seq_len}
            )
        else:
            results['best_model'] = None
            logger.warning("[ML7-Ext] No successful models")
        
        # =====================================================================
        # Drift Analysis
        # =====================================================================
        if model_results:
            drift_path = compute_drift_analysis(model_results, ml7_ext_dir)
            results['drift_analysis'] = drift_path
        
        # =====================================================================
        # Export Extended Metrics (PhD-level)
        # =====================================================================
        logger.info("[ML7-Ext] Exporting extended metrics...")
        
        metrics_dir = setup_metrics_output_dir(
            base_dir=Path('results'),
            participant=participant,
            snapshot=snapshot
        )
        
        results['metrics_exports'] = {}
        
        # Export for best model if available
        if results.get('best_model') and results['best_model']['name'] in model_results:
            best_name = results['best_model']['name']
            best_res = model_results[best_name]
            target_name = selected_target
            
            # Per-class metrics CSV
            if 'extended_metrics' in best_res:
                pc_path = export_per_class_metrics_csv(
                    metrics=best_res['extended_metrics'],
                    output_path=metrics_dir / 'per_class' / f'per_class_{best_name}_seq_{target_name}.csv',
                    model_name=best_name,
                    target_name=target_name
                )
                results['metrics_exports']['per_class_csv'] = str(pc_path)
            
            # Confusion matrices JSON
            if 'confusion_matrices' in best_res:
                cm_path = export_confusion_matrices_json(
                    fold_matrices=best_res['confusion_matrices'],
                    class_labels=class_labels,
                    output_path=metrics_dir / 'confusion_matrices' / f'cm_{best_name}_seq_{target_name}.json',
                    model_name=best_name,
                    target_name=target_name
                )
                results['metrics_exports']['confusion_matrices_json'] = str(cm_path)
            
            # Baseline comparison CSV
            if 'baselines' in results and 'extended_metrics' in best_res:
                bl_path = export_baseline_comparison_csv(
                    model_metrics=best_res['extended_metrics'],
                    baseline_metrics=results['baselines'],
                    output_path=metrics_dir / 'baseline_comparisons' / f'baseline_comparison_seq_{target_name}.csv',
                    model_name=best_name,
                    target_name=target_name
                )
                results['metrics_exports']['baseline_comparison_csv'] = str(bl_path)
        
        results['status'] = 'success'
        
    except Exception as e:
        logger.error(f"[ML7-Extended] Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        results['status'] = 'error'
        results['error'] = str(e)
    
    results['completed'] = datetime.now().isoformat()
    
    # =========================================================================
    # Save results
    # =========================================================================
    
    # Full results summary
    results_path = ml7_ext_dir / 'results_summary.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info(f"[ML7-Ext] Results saved: {results_path}")
    
    # Per-model metrics summary
    if results.get('models'):
        per_model = {}
        for model_name, model_res in results['models'].items():
            if isinstance(model_res, dict):
                per_model[model_name] = {
                    'status': model_res.get('status'),
                    'n_folds': model_res.get('n_folds_completed', 0),
                    'metrics': model_res.get('aggregate_metrics', {}),
                }
        
        per_model_path = ml7_ext_dir / 'per_model_metrics.json'
        with open(per_model_path, 'w') as f:
            json.dump(per_model, f, indent=2)
        
        logger.info(f"[ML7-Ext] Per-model metrics: {per_model_path}")
    
    # Best model summary
    if results.get('best_model'):
        best_summary_path = ml7_ext_dir / 'best_model_summary.md'
        best_name = results['best_model']['name']
        best_res = results['models'].get(best_name, {})
        
        with open(best_summary_path, 'w', encoding='utf-8') as f:
            f.write("# ML7-Extended Best Model Summary\n\n")
            f.write(f"**Generated**: {results['completed']}\n\n")
            f.write(f"## Configuration\n\n")
            f.write(f"- **Feature Set**: {results['selected_fs']}\n")
            f.write(f"- **Target**: {results['target_col']}\n")
            f.write(f"- **Sequence Length**: {results['seq_len']}\n")
            f.write(f"- **Samples**: {results['n_samples']}\n")
            f.write(f"- **Features**: {results['n_features']}\n")
            f.write(f"- **Classes**: {results['n_classes']}\n\n")
            f.write(f"## Best Model\n\n")
            f.write(f"| Metric | Value |\n")
            f.write(f"|--------|-------|\n")
            f.write(f"| Model | {results['best_model']['name']} |\n")
            f.write(f"| F1-macro | {results['best_model']['f1_macro']:.4f} |\n")
            f.write(f"| Balanced Accuracy | {results['best_model']['balanced_accuracy']:.4f} |\n")
            f.write(f"| Cohen's Kappa | {results['best_model']['cohen_kappa']:.4f} |\n")
            
            # Add baseline comparison section
            if 'baselines' in results:
                f.write("\n## Baseline Comparison\n\n")
                f.write("| Method | F1-Macro | Bal. Acc | Cohen κ |\n")
                f.write("|--------|----------|----------|--------|\n")
                f.write(f"| **{best_name}** | **{results['best_model']['f1_macro']:.4f}** | **{results['best_model']['balanced_accuracy']:.4f}** | **{results['best_model']['cohen_kappa']:.4f}** |\n")
                
                baseline_order = ['majority_class', 'stratified_random', 'persistence']
                baseline_names = {
                    'majority_class': 'Majority Class',
                    'stratified_random': 'Stratified Random',
                    'persistence': 'Persistence (seq)',
                }
                for bl_key in baseline_order:
                    if bl_key in results['baselines']:
                        bl = results['baselines'][bl_key]
                        f1 = bl.get('mean_f1_macro', 0)
                        ba = bl.get('mean_balanced_accuracy', 0)
                        kappa = bl.get('mean_cohen_kappa', 0)
                        f.write(f"| {baseline_names.get(bl_key, bl_key)} | {f1:.4f} | {ba:.4f} | {kappa:.4f} |\n")
            
            # Add per-class metrics section
            if best_res.get('extended_metrics', {}).get('per_class'):
                f.write("\n## Per-Class Metrics\n\n")
                f.write("| Class | Precision | Recall | F1 | Support |\n")
                f.write("|-------|-----------|--------|----|---------|\n")
                for pc in best_res['extended_metrics']['per_class']:
                    prec = f"{pc['precision_mean']:.3f}±{pc['precision_std']:.3f}"
                    rec = f"{pc['recall_mean']:.3f}±{pc['recall_std']:.3f}"
                    f1 = f"{pc['f1_mean']:.3f}±{pc['f1_std']:.3f}"
                    supp = pc['support_total']
                    f.write(f"| {pc['class_label']} | {prec} | {rec} | {f1} | {supp} |\n")
            
            # Add aggregated confusion matrix section
            if best_res.get('extended_metrics', {}).get('confusion_matrix_sum'):
                f.write("\n## Aggregated Confusion Matrix (Sum)\n\n")
                f.write("*Rows: True labels, Columns: Predicted labels*\n\n")
                class_labels_cm = best_res['extended_metrics'].get('class_labels', results.get('class_labels', []))
                cm = best_res['extended_metrics']['confusion_matrix_sum']
                
                header = "| True \\ Pred | " + " | ".join(str(l) for l in class_labels_cm) + " |"
                f.write(header + "\n")
                f.write("|" + "---|" * (len(class_labels_cm) + 1) + "\n")
                
                for i, row in enumerate(cm):
                    row_str = f"| **{class_labels_cm[i]}** | " + " | ".join(f"{v:.0f}" for v in row) + " |"
                    f.write(row_str + "\n")
            
            f.write(f"\n## All Models\n\n")
            f.write("| Model | Status | F1-macro |\n")
            f.write("|-------|--------|----------|\n")
            for model_name, model_res in sorted(
                results['models'].items(),
                key=lambda x: x[1].get('aggregate_metrics', {}).get('mean_f1_macro', 0) if x[1].get('status') == 'success' else -1,
                reverse=True
            ):
                status = model_res.get('status', 'unknown')
                f1 = model_res.get('aggregate_metrics', {}).get('mean_f1_macro', 'N/A')
                f1_str = f"{f1:.4f}" if isinstance(f1, float) else f1
                f.write(f"| {model_name} | {status} | {f1_str} |\n")
            
            # Add export references
            if results.get('metrics_exports'):
                f.write("\n## Extended Metrics Exports\n\n")
                for export_name, export_path in results['metrics_exports'].items():
                    f.write(f"- **{export_name}**: `{export_path}`\n")
        
        logger.info(f"[ML7-Ext] Best model summary: {best_summary_path}")
    
    logger.info("[ML7-Extended] Pipeline complete")
    logger.info("=" * 70)
    
    return results


# =============================================================================
# CLI Interface
# =============================================================================

def main():
    """Run ML7-Extended from command line."""
    import argparse
    
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(levelname)s: %(message)s'
    )
    
    parser = argparse.ArgumentParser(description='Run ML7-Extended Pipeline')
    parser.add_argument('--participant', '-p', default='P000001')
    parser.add_argument('--snapshot', '-s', default='2025-12-08')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=16)
    args = parser.parse_args()
    
    results = run_ml7_extended(
        args.participant,
        args.snapshot,
        epochs=args.epochs,
        batch_size=args.batch_size
    )
    
    if results.get('best_model'):
        print(f"\nBest Model: {results['best_model']['name']}")
        print(f"F1-macro: {results['best_model']['f1_macro']:.4f}")
    
    return 0 if results.get('status') == 'success' else 1


if __name__ == '__main__':
    exit(main())
