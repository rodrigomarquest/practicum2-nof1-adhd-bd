"""
ML7 Analysis Module
SHAP interpretability, Drift detection (ADWIN + KS), LSTM training

ML7 uses z-scored canonical features from the PBSI pipeline (Stage 3).
These features are segment-wise normalized to prevent data leakage.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import logging
import json
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


# ML7 Feature Set: Z-scored canonical features from PBSI pipeline
# These are segment-wise normalized (119 segments) to prevent leakage
ML7_FEATURE_COLS = [
    "z_sleep_total_h",       # Sleep duration (z-scored per segment)
    "z_sleep_efficiency",    # Sleep quality 0-1 scale (z-scored per segment)
    "z_hr_mean",       # Heart rate mean (z-scored per segment)
    "z_hrv_rmssd",     # HRV proxy: hr_std × 2 (z-scored per segment)
    "z_hr_max",        # Heart rate max (z-scored per segment)
    "z_steps",               # Activity steps (z-scored per segment)
    "z_exercise_min",        # Exercise estimate: active_energy ÷ 5 (z-scored per segment)
]

# Anti-leak columns: MUST NOT be used as predictors in ML7
ML7_ANTI_LEAK_COLS = [
    'pbsi_score',      # Target-derived composite score
    'pbsi_quality',    # Quality flag derived from labels
    'sleep_sub',       # PBSI subscore (intermediate calculation)
    'cardio_sub',      # PBSI subscore (intermediate calculation)
    'activity_sub',    # PBSI subscore (intermediate calculation)
    'label_2cls',      # Binary label (derived from label_3cls)
    'label_clinical',  # Clinical threshold label (derived)
]


def prepare_ml7_features(df_labeled: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare ML7 dataset with z-scored canonical features from PBSI pipeline.
    
    This function:
    1. Selects only z-scored features (segment-wise normalized) for ML7
    2. Removes anti-leak columns (pbsi_score, subscores, derived labels)
    3. Validates that no leakage columns are present
    4. Returns clean dataset for ML7 analysis (SHAP + LSTM)
    
    Args:
        df_labeled: Output from Stage 3 (features_daily_labeled.csv)
                    Contains raw features + z-scored features + PBSI outputs
    
    Returns:
        DataFrame with (date, 7 z-scored features, label_3cls)
        Shape: (n_days, 9 columns)
    
    Raises:
        AssertionError: If any anti-leak column found in output
        ValueError: If required z-scored features are missing
    """
    logger.info("[ML7 Prep] Preparing z-scored feature set from labeled data")
    
    # Verify all required z-features are present
    missing_features = [f for f in ML7_FEATURE_COLS if f not in df_labeled.columns]
    if missing_features:
        raise ValueError(f"Missing required z-features: {missing_features}")
    
    # Select ML7 features (z-scored) + date + label
    ml7_cols = ['date'] + ML7_FEATURE_COLS + ['label_3cls']
    df_ml7 = df_labeled[ml7_cols].copy()
    
    # Verify anti-leak safeguards
    for col in ML7_ANTI_LEAK_COLS:
        assert col not in df_ml7.columns, f"LEAK DETECTED: {col} found in ML7 features"
    
    logger.info(f"[ML7 Prep] Selected {len(ML7_FEATURE_COLS)} z-scored features:")
    for feat in ML7_FEATURE_COLS:
        logger.info(f"  - {feat}")
    
    logger.info(f"[ML7 Prep] Anti-leak verified: {len(ML7_ANTI_LEAK_COLS)} prohibited columns excluded")
    logger.info(f"[ML7 Prep] Output shape: {df_ml7.shape}")
    
    return df_ml7


def create_calendar_folds(df: pd.DataFrame, n_folds: int = 6, 
                          train_months: int = 4, val_months: int = 2) -> List[Dict]:
    """
    Create temporal CV folds based on calendar months.
    
```
    
    Args:
        df: DataFrame with 'date' column (sorted)
        n_folds: Number of folds
        train_months: Months for training
        val_months: Months for validation
    
    Returns:
        List of dicts with {train_idx, val_idx, train_dates, val_dates}
    """
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    
    # Find first date with at least 2 classes in a window
    min_date_orig = df['date'].min()
    max_date = df['date'].max()
    
    # Look for a good starting point (at least 2 classes in first 6 months)
    test_window = 180  # 6 months
    for i in range(len(df)):
        window_end_idx = min(i + test_window, len(df))
        window_labels = df['label_3cls'].iloc[i:window_end_idx].unique()
        if len(window_labels) >= 2:
            min_date = df['date'].iloc[i]
            logger.info(f"[Calendar CV] Starting from {min_date.date()} (first window with >=2 classes)")
            break
    else:
        min_date = min_date_orig
        logger.warning("[Calendar CV] Could not find window with >=2 classes, using full range")
    
    logger.info(f"[Calendar CV] Date range: {min_date.date()} to {max_date.date()}")
    logger.info(f"[Calendar CV] Creating {n_folds} folds: {train_months}mo train / {val_months}mo val")
    
    folds = []
    fold_months = train_months + val_months
    
    for fold_idx in range(n_folds):
        # Calculate fold boundaries
        fold_start = min_date + pd.DateOffset(months=fold_idx * fold_months)
        train_end = fold_start + pd.DateOffset(months=train_months)
        val_end = train_end + pd.DateOffset(months=val_months)
        
        # Stop if we've passed the max date
        if fold_start >= max_date:
            logger.warning(f"[Calendar CV] Fold {fold_idx}: Past end of data, stopping")
            break
        
        # Get indices
        train_mask = (df['date'] >= fold_start) & (df['date'] < train_end)
        val_mask = (df['date'] >= train_end) & (df['date'] < val_end)
        
        train_idx = df[train_mask].index.tolist()
        val_idx = df[val_mask].index.tolist()
        
        if len(train_idx) == 0 or len(val_idx) == 0:
            logger.warning(f"[Calendar CV] Fold {fold_idx}: Empty split, skipping")
            continue
        
        # Check if at least 2 classes in train
        train_classes = df.loc[train_idx, 'label_3cls'].unique()
        if len(train_classes) < 2:
            logger.warning(f"[Calendar CV] Fold {fold_idx}: Only 1 class in train, skipping")
            continue
        
        fold_info = {
            "fold": fold_idx,
            "train_idx": train_idx,
            "val_idx": val_idx,
            "train_start": fold_start.strftime("%Y-%m-%d"),
            "train_end": train_end.strftime("%Y-%m-%d"),
            "val_start": train_end.strftime("%Y-%m-%d"),
            "val_end": val_end.strftime("%Y-%m-%d"),
            "n_train": len(train_idx),
            "n_val": len(val_idx)
        }
        
        folds.append(fold_info)
        logger.info(f"  Fold {fold_idx}: Train {fold_start.date()}→{train_end.date()} "
                   f"({len(train_idx)}), Val {train_end.date()}→{val_end.date()} ({len(val_idx)})")
    
    return folds


def compute_shap_values(model, X_train, X_val, feature_names: List[str], 
                       fold_idx: int, output_dir: Path) -> Dict:
    """
    Compute SHAP values for a trained model.
    
    Args:
        model: Trained sklearn model
        X_train: Training features (DataFrame or array)
        X_val: Validation features (DataFrame or array)
        feature_names: List of feature names
        fold_idx: Fold number
        output_dir: Output directory for plots
    
    Returns:
        Dict with top features and SHAP values
    """
    try:
        import shap
        import matplotlib.pyplot as plt
        
        # Convert to numpy arrays
        if hasattr(X_train, 'values'):
            X_train_np = X_train.values.astype(np.float64)
        else:
            X_train_np = np.array(X_train, dtype=np.float64)
        
        if hasattr(X_val, 'values'):
            X_val_np = X_val.values.astype(np.float64)
        else:
            X_val_np = np.array(X_val, dtype=np.float64)
        
        # Use LinearExplainer for Logistic Regression (much faster)
        explainer = shap.LinearExplainer(model, X_train_np)
        
        # Calculate SHAP values (sample if too large)
        sample_size = min(200, len(X_val_np))
        sample_idx = np.random.choice(len(X_val_np), sample_size, replace=False)
        X_sample = X_val_np[sample_idx]
        
        shap_values = explainer.shap_values(X_sample)
        
        # Get mean absolute SHAP per feature (average across classes)
        # LinearExplainer for multi-class returns (n_samples, n_features, n_classes)
        if len(shap_values.shape) == 3:
            # Multi-class: average across samples and classes
            mean_shap = np.abs(shap_values).mean(axis=(0, 2))  # Mean over samples & classes
        else:
            # Binary: average across samples
            mean_shap = np.abs(shap_values).mean(axis=0)
        
        # Rank features
        feature_importance = pd.DataFrame({
            'feature': list(feature_names),
            'shap_importance': list(mean_shap)
        }).sort_values('shap_importance', ascending=False)
        
        top_5 = feature_importance.head(5)
        
        # Plot top 5
        plt.figure(figsize=(10, 6))
        plt.barh(top_5['feature'][::-1], top_5['shap_importance'][::-1])
        plt.xlabel('Mean |SHAP value|')
        plt.title(f'Fold {fold_idx}: Top 5 Features by SHAP Importance')
        plt.tight_layout()
        
        plot_path = output_dir / f"fold_{fold_idx}_top5.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"[SHAP] Fold {fold_idx} top-5: {', '.join(top_5['feature'].tolist())}")
        
        return {
            'fold': fold_idx,
            'top_features': feature_importance.to_dict('records'),
            'plot_path': str(plot_path)
        }
    
    except ImportError:
        logger.warning("[SHAP] shap library not available, skipping")
        return {'fold': fold_idx, 'error': 'shap not installed'}
    except Exception as e:
        logger.error(f"[SHAP] Error in fold {fold_idx}: {e}")
        return {'fold': fold_idx, 'error': str(e)}


def detect_drift_adwin(df: pd.DataFrame, score_col: str = 'pbsi_score', 
                      delta: float = 0.002, output_path: Path = None) -> Dict:
    """
    ADWIN drift detection on temporal sequence.
    
    Args:
        df: DataFrame with temporal data (sorted by date)
        score_col: Column to monitor ('pbsi_score' or 'proba_positive')
        delta: ADWIN confidence parameter
        output_path: Path to save changes CSV
    
    Returns:
        Dict with drift statistics
    """
    try:
        from river import drift
        
        df = df.copy().sort_values('date').reset_index(drop=True)
        
        if score_col not in df.columns:
            logger.warning(f"[ADWIN] Column '{score_col}' not found, skipping")
            return {'error': f'{score_col} not available'}
        
        # Initialize ADWIN
        adwin = drift.ADWIN(delta=delta)
        
        changes = []
        for idx, row in df.iterrows():
            value = row[score_col]
            if pd.isna(value):
                continue
            
            adwin.update(value)
            
            if adwin.drift_detected:
                changes.append({
                    'index': idx,
                    'date': row['date'],
                    'value': float(value),
                    'mean_before': float(adwin.estimation) if hasattr(adwin, 'estimation') else None
                })
                logger.info(f"[ADWIN] Drift detected at {row['date']} (idx={idx})")
        
        # Save changes
        if output_path and changes:
            pd.DataFrame(changes).to_csv(output_path, index=False)
            logger.info(f"[ADWIN] Saved {len(changes)} drift points to {output_path}")
        
        return {
            'delta': delta,
            'n_changes': len(changes),
            'changes': changes
        }
    
    except ImportError:
        logger.warning("[ADWIN] river library not available")
        return {'error': 'river not installed'}
    except Exception as e:
        logger.error(f"[ADWIN] Error: {e}")
        return {'error': str(e)}


def detect_drift_ks_segments(df: pd.DataFrame, segments_df: pd.DataFrame,
                             feature_cols: List[str], window_days: int = 14,
                             output_path: Path = None) -> Dict:
    """
    KS test at segment boundaries (±window_days).
    
    Args:
        df: Main dataframe with features (sorted by date)
        segments_df: segment_autolog.csv with segment boundaries
        feature_cols: Continuous features to test
        window_days: Days before/after boundary
        output_path: Path to save KS results CSV
    
    Returns:
        Dict with KS statistics
    """
    try:
        from scipy.stats import ks_2samp
        
        df = df.copy()
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)
        
        results = []
        
        for seg_idx, seg in segments_df.iterrows():
            if seg_idx == 0:
                continue  # Skip first segment (no before data)
            
            boundary_date = pd.to_datetime(seg['date_start'])
            
            # Get windows
            before_mask = (df['date'] >= boundary_date - pd.Timedelta(days=window_days)) & \
                         (df['date'] < boundary_date)
            after_mask = (df['date'] >= boundary_date) & \
                        (df['date'] < boundary_date + pd.Timedelta(days=window_days))
            
            df_before = df[before_mask]
            df_after = df[after_mask]
            
            if len(df_before) < 5 or len(df_after) < 5:
                continue  # Not enough data
            
            for feat in feature_cols:
                if feat not in df.columns:
                    continue
                
                # Drop NaNs
                before_vals = df_before[feat].dropna()
                after_vals = df_after[feat].dropna()
                
                if len(before_vals) < 5 or len(after_vals) < 5:
                    continue
                
                # KS test
                stat, pval = ks_2samp(before_vals, after_vals)
                
                results.append({
                    'segment_idx': seg_idx,
                    'boundary_date': boundary_date.strftime("%Y-%m-%d"),
                    'feature': feat,
                    'ks_statistic': float(stat),
                    'p_value': float(pval),
                    'significant': pval < 0.05,
                    'n_before': len(before_vals),
                    'n_after': len(after_vals)
                })
        
        # Save results
        if output_path and results:
            pd.DataFrame(results).to_csv(output_path, index=False)
            logger.info(f"[KS] Saved {len(results)} tests to {output_path}")
        
        n_significant = sum(1 for r in results if r['significant'])
        logger.info(f"[KS] {n_significant}/{len(results)} tests significant (p<0.05)")
        
        return {
            'n_tests': len(results),
            'n_significant': n_significant,
            'results': results
        }
    
    except ImportError:
        logger.warning("[KS] scipy not available")
        return {'error': 'scipy not installed'}
    except Exception as e:
        logger.error(f"[KS] Error: {e}")
        return {'error': str(e)}


def create_lstm_sequences(X: np.ndarray, y: np.ndarray, seq_len: int = 14) -> Tuple:
    """
    Create sequences for LSTM.
    
    Args:
        X: Features array (n_samples, n_features)
        y: Labels array (n_samples,)
        seq_len: Sequence length
    
    Returns:
        (X_seq, y_seq) where X_seq has shape (n_seq, seq_len, n_features)
    """
    X_seq, y_seq = [], []
    
    for i in range(len(X) - seq_len + 1):
        X_seq.append(X[i:i+seq_len])
        y_seq.append(y[i+seq_len-1])  # Predict last label in sequence
    
    return np.array(X_seq), np.array(y_seq)


def train_lstm_model(X_train, y_train, X_val, y_val, n_classes: int = 3,
                     seq_len: int = 14, n_features: int = 10) -> Dict:
    """
    Train LSTM model: LSTM(32) -> Dense(32) -> Dropout(0.2) -> Softmax.
    
    Args:
        X_train, y_train: Training sequences
        X_val, y_val: Validation sequences
        n_classes: Number of classes
        seq_len: Sequence length
        n_features: Number of input features
    
    Returns:
        Dict with model, history, metrics
    """
    try:
        import tensorflow as tf
        from tensorflow import keras
        from tensorflow.keras import layers
        from sklearn.metrics import f1_score
        
        # Remap labels from {-1, 0, 1} to {0, 1, 2} for Keras
        label_map = {-1: 0, 0: 1, 1: 2}
        y_train_mapped = np.array([label_map.get(y, y) for y in y_train])
        y_val_mapped = np.array([label_map.get(y, y) for y in y_val])
        
        # Build model
        model = keras.Sequential([
            layers.LSTM(32, input_shape=(seq_len, n_features), return_sequences=False),
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(n_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Train
        history = model.fit(
            X_train, y_train_mapped,
            validation_data=(X_val, y_val_mapped),
            epochs=20,
            batch_size=32,
            verbose=0
        )
        
        # Evaluate
        y_pred_proba = model.predict(X_val, verbose=0)
        y_pred_mapped = np.argmax(y_pred_proba, axis=1)
        
        # Reverse map for F1 calculation
        reverse_map = {0: -1, 1: 0, 2: 1}
        y_pred = np.array([reverse_map[y] for y in y_pred_mapped])
        
        f1 = f1_score(y_val, y_pred, average='macro', zero_division=0)
        
        return {
            'model': model,
            'history': history.history,
            'f1_macro': float(f1),
            'val_loss': float(history.history['val_loss'][-1]),
            'val_accuracy': float(history.history['val_accuracy'][-1])
        }
    
    except ImportError:
        logger.warning("[LSTM] TensorFlow not available")
        return {'error': 'tensorflow not installed'}
    except Exception as e:
        logger.error(f"[LSTM] Training error: {e}")
        return {'error': str(e)}


def convert_to_tflite(model, output_path: Path) -> bool:
    """
    Convert Keras model to TFLite.
    
    Args:
        model: Keras model
        output_path: Path to save .tflite file
    
    Returns:
        True if successful
    """
    try:
        import tensorflow as tf
        
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,  # enable TensorFlow Lite ops
            tf.lite.OpsSet.SELECT_TF_OPS  # enable TensorFlow ops (for LSTM)
        ]
        converter._experimental_lower_tensor_list_ops = False
        
        tflite_model = converter.convert()
        
        with open(output_path, 'wb') as f:
            f.write(tflite_model)
        
        logger.info(f"[TFLite] Model saved: {output_path} ({len(tflite_model)/1024:.1f} KB)")
        return True
    
    except Exception as e:
        logger.error(f"[TFLite] Conversion error: {e}")
        # Save a placeholder file
        with open(output_path, 'w') as f:
            f.write(f"TFLite conversion failed: {str(e)}\n")
        return False


def measure_latency(model_path: Path, X_sample: np.ndarray, n_runs: int = 100) -> Dict:
    """
    Measure TFLite inference latency.
    
    Args:
        model_path: Path to .tflite model
        X_sample: Sample input (1, seq_len, n_features)
        n_runs: Number of inference runs
    
    Returns:
        Dict with mean, p95 latency in ms
    """
    try:
        import tensorflow as tf
        import time
        
        # Load TFLite model
        interpreter = tf.lite.Interpreter(model_path=str(model_path))
        interpreter.allocate_tensors()
        
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        # Warm-up
        interpreter.set_tensor(input_details[0]['index'], X_sample.astype(np.float32))
        interpreter.invoke()
        
        # Measure
        latencies = []
        for _ in range(n_runs):
            start = time.time()
            interpreter.set_tensor(input_details[0]['index'], X_sample.astype(np.float32))
            interpreter.invoke()
            _ = interpreter.get_tensor(output_details[0]['index'])
            latencies.append((time.time() - start) * 1000)  # Convert to ms
        
        return {
            'mean_ms': float(np.mean(latencies)),
            'p95_ms': float(np.percentile(latencies, 95)),
            'p99_ms': float(np.percentile(latencies, 99)),
            'min_ms': float(np.min(latencies)),
            'max_ms': float(np.max(latencies)),
            'n_runs': n_runs
        }
    
    except Exception as e:
        logger.error(f"[Latency] Measurement error: {e}")
        return {'error': str(e)}
