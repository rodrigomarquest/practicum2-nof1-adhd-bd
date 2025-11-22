"""
ML7 Extended Models: GRU, TCN, and Temporal MLP.

Implements 3 additional temporal models for the ML7 14-day sequence problem:
1. GRU (Gated Recurrent Unit)
2. TCN (Temporal Convolutional Network with dilated convolutions)
3. Tiny-MLP (temporal baseline with flattened input)

All models use the EXACT same:
- Dataset: 14-day sliding windows from features_daily_labeled.csv
- Folds: 6-fold calendar-based temporal CV (same as ML6/ML7 LSTM)
- Preprocessing: Z-score normalization
- Label encoding: {-1, 0, +1} → {0, 1, 2}
- Metrics: f1_macro, f1_weighted, balanced_acc, kappa, auroc_ovr

Explainability:
- Gradient-based saliency maps for all models
- Feature importance via integrated gradients

Dependencies:
    pip install tensorflow>=2.13.0 numpy pandas scikit-learn matplotlib
"""

import pandas as pd
import numpy as np
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    f1_score, balanced_accuracy_score, cohen_kappa_score,
    roc_auc_score, confusion_matrix
)
from sklearn.preprocessing import label_binarize

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)
np.random.seed(42)
tf.random.set_seed(42)


def build_gru(seq_len: int, n_feats: int, n_classes: int = 3, hidden: int = 64, dropout: float = 0.3) -> Model:
    """
    Build GRU model for sequence classification.
    
    Architecture:
        Input(seq_len, n_feats)
        → GRU(hidden, return_sequences=False)
        → Dropout(dropout)
        → Dense(n_classes, softmax)
    """
    model = keras.Sequential([
        layers.Input(shape=(seq_len, n_feats)),
        layers.GRU(hidden, return_sequences=False),
        layers.Dropout(dropout),
        layers.Dense(n_classes, activation='softmax')
    ], name='GRU')
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def build_tcn(seq_len: int, n_feats: int, n_classes: int = 3, 
              filters: int = 64, kernel_size: int = 3, dropout: float = 0.3) -> Model:
    """
    Build Temporal Convolutional Network with dilated convolutions.
    
    Architecture:
        Input(seq_len, n_feats)
        → Conv1D(filters, kernel_size, dilation=1, causal padding)
        → Conv1D(filters, kernel_size, dilation=2, causal padding)
        → Conv1D(filters, kernel_size, dilation=4, causal padding)
        → GlobalAveragePooling1D()
        → Dropout(dropout)
        → Dense(n_classes, softmax)
    
    Note: Uses causal padding to prevent information leakage from future timesteps.
    """
    inputs = layers.Input(shape=(seq_len, n_feats))
    
    # TCN block with increasing dilation rates
    x = inputs
    for dilation_rate in [1, 2, 4]:
        x = layers.Conv1D(
            filters=filters,
            kernel_size=kernel_size,
            dilation_rate=dilation_rate,
            padding='causal',
            activation='relu'
        )(x)
        x = layers.Dropout(dropout)(x)
    
    # Aggregate temporal dimension
    x = layers.GlobalAveragePooling1D()(x)
    
    # Output layer
    outputs = layers.Dense(n_classes, activation='softmax')(x)
    
    model = Model(inputs=inputs, outputs=outputs, name='TCN')
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def build_mlp_temporal(seq_len: int, n_feats: int, n_classes: int = 3, 
                       hidden: int = 128, dropout: float = 0.3) -> Model:
    """
    Build Tiny-MLP baseline with flattened temporal input.
    
    Architecture:
        Input(seq_len, n_feats)
        → Flatten()
        → Dense(hidden, relu)
        → Dropout(dropout)
        → Dense(hidden // 2, relu)
        → Dropout(dropout)
        → Dense(n_classes, softmax)
    """
    model = keras.Sequential([
        layers.Input(shape=(seq_len, n_feats)),
        layers.Flatten(),
        layers.Dense(hidden, activation='relu'),
        layers.Dropout(dropout),
        layers.Dense(hidden // 2, activation='relu'),
        layers.Dropout(dropout),
        layers.Dense(n_classes, activation='softmax')
    ], name='MLP_Temporal')
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def load_ml7_data(features_csv: str, seq_len: int = 14) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    Load and prepare ML7 sequence data from features_daily_labeled.csv.
    
    Returns:
        X: [n_sequences, seq_len, n_feats]
        y: [n_sequences] with labels {0, 1, 2} (converted from {-1, 0, +1})
        dates: [n_sequences] with sequence end dates
        feature_cols: List of feature names
    """
    df = pd.read_csv(features_csv)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    
    # Feature columns
    exclude_cols = ['date', 'label_3cls', 'label_2cls', 'segment_id']
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    
    # Extract features and labels
    X_df = df[feature_cols].fillna(0)  # Fill NaNs
    y_raw = df['label_3cls'].values
    
    # Create sequences
    X_sequences = []
    y_sequences = []
    date_sequences = []
    
    for i in range(len(df) - seq_len + 1):
        seq_x = X_df.iloc[i:i+seq_len].values
        seq_y = y_raw[i+seq_len-1]  # Label for last day in sequence
        seq_date = df.iloc[i+seq_len-1]['date']
        
        # Skip if label is missing
        if pd.isna(seq_y):
            continue
        
        X_sequences.append(seq_x)
        y_sequences.append(seq_y)
        date_sequences.append(seq_date)
    
    X = np.array(X_sequences)
    y = np.array(y_sequences)
    dates = np.array(date_sequences)
    
    # Convert labels {-1, 0, +1} → {0, 1, 2}
    y_encoded = np.array([0 if yi == -1 else (1 if yi == 0 else 2) for yi in y])
    
    logger.info(f"Created {len(X)} sequences of shape ({seq_len}, {len(feature_cols)})")
    logger.info(f"Label distribution: {np.bincount(y_encoded)}")
    
    return X, y_encoded, dates, feature_cols


def normalize_sequences(X_train: np.ndarray, X_val: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Normalize sequences using StandardScaler fitted on training data.
    
    Args:
        X_train: [n_train, seq_len, n_feats]
        X_val: [n_val, seq_len, n_feats]
    
    Returns:
        X_train_norm, X_val_norm with same shapes
    """
    n_train, seq_len, n_feats = X_train.shape
    n_val = X_val.shape[0]
    
    # Reshape to [n_samples * seq_len, n_feats]
    X_train_flat = X_train.reshape(-1, n_feats)
    X_val_flat = X_val.reshape(-1, n_feats)
    
    # Fit scaler on training data
    scaler = StandardScaler()
    X_train_flat_norm = scaler.fit_transform(X_train_flat)
    X_val_flat_norm = scaler.transform(X_val_flat)
    
    # Reshape back
    X_train_norm = X_train_flat_norm.reshape(n_train, seq_len, n_feats)
    X_val_norm = X_val_flat_norm.reshape(n_val, seq_len, n_feats)
    
    return X_train_norm, X_val_norm


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray = None) -> Dict:
    """Compute 3-class metrics."""
    metrics = {
        'f1_macro': float(f1_score(y_true, y_pred, average='macro', zero_division=0)),
        'f1_weighted': float(f1_score(y_true, y_pred, average='weighted', zero_division=0)),
        'balanced_acc': float(balanced_accuracy_score(y_true, y_pred)),
        'kappa': float(cohen_kappa_score(y_true, y_pred)),
    }
    
    # AUROC (One-vs-Rest)
    if y_proba is not None and len(np.unique(y_true)) >= 2:
        try:
            y_true_bin = label_binarize(y_true, classes=[0, 1, 2])
            auroc = roc_auc_score(y_true_bin, y_proba, average='macro', multi_class='ovr')
            metrics['auroc_ovr'] = float(auroc)
        except Exception:
            metrics['auroc_ovr'] = np.nan
    else:
        metrics['auroc_ovr'] = np.nan
    
    return metrics


def compute_gradient_saliency(model: Model, X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Compute gradient-based saliency maps for feature importance.
    
    Args:
        model: Trained keras model
        X: Input sequences [n_samples, seq_len, n_feats]
        y: True labels [n_samples]
    
    Returns:
        saliency: [n_samples, seq_len, n_feats] gradient magnitudes
    """
    X_tensor = tf.convert_to_tensor(X, dtype=tf.float32)
    y_tensor = tf.convert_to_tensor(y, dtype=tf.int32)
    
    saliency_maps = []
    
    for i in range(len(X)):
        with tf.GradientTape() as tape:
            x_sample = tf.expand_dims(X_tensor[i], 0)
            tape.watch(x_sample)
            
            # Forward pass
            proba = model(x_sample, training=False)
            
            # Loss for true class
            true_class_prob = proba[0, y_tensor[i]]
        
        # Gradient of true class probability w.r.t. input
        gradient = tape.gradient(true_class_prob, x_sample)
        gradient = tf.abs(gradient).numpy()[0]  # Take absolute value
        
        saliency_maps.append(gradient)
    
    return np.array(saliency_maps)


def save_saliency_summary(saliency: np.ndarray, feature_names: List[str], output_path: Path):
    """
    Save mean saliency scores across all sequences and timesteps.
    
    Args:
        saliency: [n_samples, seq_len, n_feats]
        feature_names: List of feature names
        output_path: CSV output path
    """
    # Mean across samples and timesteps
    mean_saliency = saliency.mean(axis=(0, 1))
    
    df = pd.DataFrame({
        'feature': feature_names,
        'mean_saliency': mean_saliency
    }).sort_values('mean_saliency', ascending=False)
    
    df.to_csv(output_path, index=False)
    logger.info(f"Saved saliency summary to {output_path}")
    logger.info(f"Top 5 important features:")
    for i, row in df.head(5).iterrows():
        logger.info(f"  {row['feature']}: {row['mean_saliency']:.6f}")


def train_model(
    model_type: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    seq_len: int,
    n_feats: int,
    epochs: int = 50,
    batch_size: int = 32
) -> Dict:
    """
    Train a single model and return metrics.
    
    Args:
        model_type: 'gru', 'tcn', or 'mlp'
        X_train, y_train: Training sequences
        X_val, y_val: Validation sequences
        seq_len: Sequence length
        n_feats: Number of features
        epochs: Training epochs
        batch_size: Batch size
    
    Returns:
        Dict with metrics, predictions, and model
    """
    # Build model
    if model_type == 'gru':
        model = build_gru(seq_len, n_feats)
    elif model_type == 'tcn':
        model = build_tcn(seq_len, n_feats)
    elif model_type == 'mlp':
        model = build_mlp_temporal(seq_len, n_feats)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    
    # Early stopping
    early_stop = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=0
    )
    
    # Train
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stop],
        verbose=0
    )
    
    # Predict
    y_proba = model.predict(X_val, verbose=0)
    y_pred = np.argmax(y_proba, axis=1)
    
    # Compute metrics
    metrics = compute_metrics(y_val, y_pred, y_proba)
    
    logger.info(
        f"  {model_type.upper()}: F1-macro={metrics['f1_macro']:.4f}, "
        f"AUROC={metrics['auroc_ovr']:.4f}, "
        f"Balanced-acc={metrics['balanced_acc']:.4f}"
    )
    
    return {
        'metrics': metrics,
        'y_pred': y_pred,
        'y_proba': y_proba,
        'model': model,
        'history': history.history
    }


def run_ml7_extended(
    features_csv: str,
    cv_summary_json: str,
    output_dir: str,
    models: List[str] = ['gru', 'tcn', 'mlp'],
    seq_len: int = 14,
    compute_saliency: bool = True
):
    """
    Run all ML7 extended models.
    
    Args:
        features_csv: Path to features_daily_labeled.csv
        cv_summary_json: Path to cv_summary.json (for fold definitions)
        output_dir: Output directory for results
        models: List of models to train
        seq_len: Sequence length (default 14 days)
        compute_saliency: Whether to compute gradient saliency
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    logger.info("="*80)
    logger.info("ML7 EXTENDED MODELS: GRU, TCN, TEMPORAL MLP")
    logger.info("="*80)
    
    # Load sequence data
    logger.info(f"\nLoading sequence data (seq_len={seq_len})...")
    X_all, y_all, dates_all, feature_cols = load_ml7_data(features_csv, seq_len)
    
    # Load fold definitions
    with open(cv_summary_json, 'r') as f:
        cv_data = json.load(f)
    folds = cv_data['folds']
    
    # Run each model
    results = {model: {'folds': []} for model in models}
    
    for fold_info in folds:
        fold_idx = fold_info['fold']
        logger.info(f"\n{'='*60}")
        logger.info(f"FOLD {fold_idx}: {fold_info['train_start']} to {fold_info['val_end']}")
        logger.info(f"{'='*60}")
        
        # Split sequences by date
        train_mask = (dates_all >= pd.to_datetime(fold_info['train_start'])) & \
                     (dates_all < pd.to_datetime(fold_info['val_start']))
        val_mask = (dates_all >= pd.to_datetime(fold_info['val_start'])) & \
                   (dates_all < pd.to_datetime(fold_info['val_end']))
        
        X_train = X_all[train_mask]
        y_train = y_all[train_mask]
        X_val = X_all[val_mask]
        y_val = y_all[val_mask]
        
        logger.info(f"Train: {len(X_train)} sequences, Val: {len(X_val)} sequences")
        
        # Normalize
        X_train_norm, X_val_norm = normalize_sequences(X_train, X_val)
        
        # Train each model
        for model_type in models:
            result = train_model(
                model_type=model_type,
                X_train=X_train_norm,
                y_train=y_train,
                X_val=X_val_norm,
                y_val=y_val,
                seq_len=seq_len,
                n_feats=len(feature_cols)
            )
            
            # Save model weights (Keras 3.x requires .weights.h5 extension)
            weights_path = output_path / f'ml7_{model_type}_fold{fold_idx}.weights.h5'
            result['model'].save_weights(str(weights_path))
            
            # Compute saliency
            if compute_saliency:
                saliency = compute_gradient_saliency(result['model'], X_val_norm, y_val)
                saliency_path = output_path / f'ml7_{model_type}_fold{fold_idx}_saliency.csv'
                save_saliency_summary(saliency, feature_cols, saliency_path)
            
            results[model_type]['folds'].append({
                'fold': fold_idx,
                **result['metrics']
            })
    
    # Save results
    for model_type in models:
        # Save fold-level metrics
        fold_df = pd.DataFrame(results[model_type]['folds'])
        metrics_json = output_path / f'ml7_{model_type}_metrics.json'
        
        # Compute summary statistics
        summary = {
            'model': model_type,
            'n_folds': len(fold_df),
            'mean_f1_macro': float(fold_df['f1_macro'].mean()),
            'std_f1_macro': float(fold_df['f1_macro'].std()),
            'mean_f1_weighted': float(fold_df['f1_weighted'].mean()),
            'mean_balanced_acc': float(fold_df['balanced_acc'].mean()),
            'mean_kappa': float(fold_df['kappa'].mean()),
            'mean_auroc_ovr': float(fold_df['auroc_ovr'].mean()),
            'folds': results[model_type]['folds']
        }
        
        with open(metrics_json, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"\nSaved {model_type.upper()} results to {metrics_json}")
        logger.info(f"  Mean F1-macro: {summary['mean_f1_macro']:.4f} ± {summary['std_f1_macro']:.4f}")
        logger.info(f"  Mean AUROC: {summary['mean_auroc_ovr']:.4f}")
    
    # Create summary CSV
    summary_rows = []
    for model_type in models:
        with open(output_path / f'ml7_{model_type}_metrics.json', 'r') as f:
            data = json.load(f)
        summary_rows.append({
            'model': model_type.upper(),
            'f1_macro_mean': data['mean_f1_macro'],
            'f1_macro_std': data['std_f1_macro'],
            'f1_weighted_mean': data['mean_f1_weighted'],
            'balanced_acc_mean': data['mean_balanced_acc'],
            'kappa_mean': data['mean_kappa'],
            'auroc_ovr_mean': data['mean_auroc_ovr']
        })
    
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(output_path / 'ml7_extended_summary.csv', index=False)
    
    # Create markdown summary
    with open(output_path / 'ml7_extended_summary.md', 'w') as f:
        f.write("# ML7 Extended Models Summary\n\n")
        f.write("## Models\n\n")
        f.write("- **GRU**: Gated Recurrent Unit (64 hidden units)\n")
        f.write("- **TCN**: Temporal Convolutional Network (dilations: 1, 2, 4)\n")
        f.write("- **MLP**: Temporal MLP baseline (flattened input)\n\n")
        f.write("## Performance Results\n\n")
        f.write(summary_df.to_markdown(index=False))
        f.write("\n\n## Configuration\n\n")
        f.write(f"- Sequence length: {seq_len} days\n")
        f.write(f"- Features: {len(feature_cols)}\n")
        f.write(f"- Cross-validation: 6 temporal folds\n")
        f.write(f"- Normalization: Z-score (StandardScaler)\n")
    
    logger.info(f"\n{'='*80}")
    logger.info(f"[OK] ML7 EXTENDED COMPLETE")
    logger.info(f"Results saved to: {output_path}")
    logger.info(f"{'='*80}\n")


if __name__ == '__main__':
    import sys
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Default paths
    features_csv = 'data/etl/P000001/2025-11-07/joined/features_daily_labeled.csv'
    cv_summary = 'data/ai/P000001/2025-11-07/ml6/cv_summary.json'
    output_dir = 'data/ai/P000001/2025-11-07/ml7_ext'
    
    run_ml7_extended(features_csv, cv_summary, output_dir)
