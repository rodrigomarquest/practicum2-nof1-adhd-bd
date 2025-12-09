"""
Experiment Suite Module for Stage 5 Integration

This module provides a clean callable interface for running ML6/ML7 ablation
experiments from within Stage 5 of the pipeline.

The Experiment Suite:
1. Takes the Feature Universe DataFrame (post-MICE imputation)
2. Runs controlled ablation across feature sets and targets
3. Applies deterministic selection rules
4. Produces model_selection.json artifact

Usage (from stage_5_prep_ml6):
    from src.etl.experiment_suite import run_experiment_suite
    model_selection = run_experiment_suite(df_universe, participant, snapshot)
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# =============================================================================
# Feature Set Definitions (shared with ml6_som_experiments.py)
# =============================================================================

# FS-A: Baseline (10 features)
FS_A_FEATURES = [
    'sleep_hours', 'sleep_quality_score',
    'hr_mean', 'hr_min', 'hr_max', 'hr_std', 'hr_samples',
    'total_steps', 'total_distance', 'total_active_energy',
]

# FS-B: Baseline + HRV (15 features)
FS_B_FEATURES = FS_A_FEATURES + [
    'hrv_sdnn_mean', 'hrv_sdnn_median', 'hrv_sdnn_min', 'hrv_sdnn_max', 'n_hrv_sdnn',
]

# FS-C: FS-B + MEDS (18 features)
FS_C_FEATURES = FS_B_FEATURES + [
    'med_any', 'med_event_count', 'med_dose_total',
]

# FS-D: FS-C + PBSI (19 features)
FS_D_FEATURES = FS_C_FEATURES + [
    'pbsi_score',
]

FEATURE_SETS = {
    'FS-A': {'name': 'Baseline (Sleep+Cardio+Activity)', 'features': FS_A_FEATURES},
    'FS-B': {'name': 'Baseline + HRV', 'features': FS_B_FEATURES},
    'FS-C': {'name': 'FS-B + MEDS', 'features': FS_C_FEATURES},
    'FS-D': {'name': 'FS-C + PBSI', 'features': FS_D_FEATURES},
}

# Default fallback feature set
DEFAULT_FEATURE_SET = 'FS-B'
DEFAULT_FEATURES = FS_B_FEATURES


# =============================================================================
# Deterministic Selection Rules
# =============================================================================

def select_best_config(results: List[Dict], model_type: str = 'ml6') -> Dict:
    """
    Apply deterministic selection rules to choose best configuration.
    
    Rules (in order of priority):
    1. Highest F1-macro
    2. If tie: highest Cohen's kappa
    3. If tie: fewer features (simpler model)
    
    Args:
        results: List of experiment result dicts
        model_type: 'ml6' or 'ml7'
    
    Returns:
        Best configuration dict
    """
    if not results:
        return None
    
    # Filter to successful experiments
    successful = [r for r in results if r.get('status') == 'success' or r.get('success')]
    
    if not successful:
        return None
    
    # Sort by selection criteria
    def sort_key(r):
        f1 = r.get('f1_macro') or r.get('aggregate_metrics', {}).get('mean_f1_macro', 0)
        kappa = r.get('cohen_kappa') or r.get('aggregate_metrics', {}).get('mean_cohen_kappa', 0)
        n_features = r.get('n_features', 0)
        return (-f1, -kappa, n_features)
    
    sorted_results = sorted(successful, key=sort_key)
    return sorted_results[0]


# =============================================================================
# ML6 Ablation (LogisticRegression)
# =============================================================================

def run_ml6_ablation_quick(
    df: pd.DataFrame,
    n_folds: int = 6
) -> Dict:
    """
    Run quick ML6 ablation for feature set selection.
    
    This is a simplified version of ml6_som_experiments.run_ablation_study()
    that can be called from Stage 5 without external dependencies.
    
    Args:
        df: DataFrame with features and targets (post-MICE)
        n_folds: Number of CV folds
    
    Returns:
        Dict with results for all FS × target combinations
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score, balanced_accuracy_score, cohen_kappa_score
    
    results = []
    
    # Target configurations
    targets = {
        'binary': 'som_binary',
        '3class': 'som_category_3class'
    }
    
    for fs_id, fs_info in FEATURE_SETS.items():
        for target_key, target_col in targets.items():
            # Check if target exists
            if target_col not in df.columns:
                continue
            
            # Get available features
            features = [f for f in fs_info['features'] if f in df.columns]
            if len(features) < 3:
                continue
            
            # Prepare X, y
            X = df[features].values
            y = df[target_col].values
            
            # Check class distribution
            unique_classes = np.unique(y[~np.isnan(y)])
            if len(unique_classes) < 2:
                continue
            
            # Simple temporal CV
            n_samples = len(df)
            fold_size = n_samples // n_folds
            
            fold_metrics = []
            
            for fold_idx in range(n_folds):
                val_start = fold_idx * fold_size
                val_end = min((fold_idx + 1) * fold_size, n_samples)
                
                val_idx = list(range(val_start, val_end))
                train_idx = [i for i in range(n_samples) if i not in val_idx]
                
                if len(train_idx) < 5 or len(val_idx) < 2:
                    continue
                
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                
                # Check if train has enough classes
                if len(np.unique(y_train)) < 2:
                    continue
                
                try:
                    model = LogisticRegression(
                        class_weight='balanced',
                        max_iter=1000,
                        random_state=42
                    )
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_val)
                    
                    fold_metrics.append({
                        'f1_macro': f1_score(y_val, y_pred, average='macro', zero_division=0),
                        'balanced_accuracy': balanced_accuracy_score(y_val, y_pred),
                        'cohen_kappa': cohen_kappa_score(y_val, y_pred),
                    })
                except Exception:
                    continue
            
            if not fold_metrics:
                continue
            
            # Aggregate metrics
            result = {
                'feature_set': fs_id,
                'target': target_key,
                'target_col': target_col,
                'n_features': len(features),
                'features': features,
                'status': 'success',
                'n_folds': len(fold_metrics),
                'aggregate_metrics': {
                    'mean_f1_macro': float(np.mean([m['f1_macro'] for m in fold_metrics])),
                    'std_f1_macro': float(np.std([m['f1_macro'] for m in fold_metrics])),
                    'mean_balanced_accuracy': float(np.mean([m['balanced_accuracy'] for m in fold_metrics])),
                    'mean_cohen_kappa': float(np.mean([m['cohen_kappa'] for m in fold_metrics])),
                }
            }
            results.append(result)
    
    return results


# =============================================================================
# ML7 Ablation (LSTM) - Lightweight Version
# =============================================================================

def run_ml7_ablation_quick(
    df: pd.DataFrame,
    feature_set: str = 'FS-B',
    skip_lstm: bool = False
) -> Dict:
    """
    Run quick ML7 ablation for configuration selection.
    
    This is a simplified version that tests key configurations without
    running full grid search. If TensorFlow is not available or
    skip_lstm=True, returns placeholder with FS-B defaults.
    
    Args:
        df: DataFrame with features and targets (post-MICE)
        feature_set: Feature set to use (default: FS-B)
        skip_lstm: If True, skip LSTM training and return defaults
    
    Returns:
        Dict with ML7 selection results
    """
    # Get feature set
    if feature_set not in FEATURE_SETS:
        feature_set = DEFAULT_FEATURE_SET
    
    features = [f for f in FEATURE_SETS[feature_set]['features'] if f in df.columns]
    
    result = {
        'feature_set': feature_set,
        'features': features,
        'n_features': len(features),
        'status': 'success',
        'tested_configs': []
    }
    
    if skip_lstm or len(df) < 30:
        # Return defaults without running LSTM
        result['selected_config'] = 'CFG-3'
        result['selected_target'] = 'som_binary'
        result['config_params'] = {
            'seq_len': 14,
            'lstm_units': 32,
            'dense_units': 32,
            'dropout': 0.4,
            'early_stopping': True,
            'class_weights': True
        }
        result['metrics'] = {
            'f1_macro': 0.0,
            'note': 'LSTM skipped - insufficient data or skip_lstm=True'
        }
        result['status'] = 'skipped'
        return result
    
    # Try to import TensorFlow and run LSTM
    try:
        import tensorflow as tf
        from tensorflow import keras
        from tensorflow.keras import layers
        from sklearn.metrics import f1_score, balanced_accuracy_score
        from sklearn.utils.class_weight import compute_class_weight
        
        # Simplified config test: just CFG-3 (our known best)
        X = df[features].values
        y_binary = df['som_binary'].values
        
        # Create sequences (seq_len=14)
        seq_len = 14
        n_samples = len(df)
        n_sequences = n_samples - seq_len + 1
        
        if n_sequences < 10:
            result['status'] = 'skipped'
            result['metrics'] = {'note': f'Only {n_sequences} sequences possible'}
            return result
        
        X_seq = np.array([X[i:i+seq_len] for i in range(n_sequences)])
        y_seq = np.array([y_binary[i+seq_len-1] for i in range(n_sequences)])
        
        # 80/20 split
        split_idx = int(0.8 * len(X_seq))
        X_train, X_val = X_seq[:split_idx], X_seq[split_idx:]
        y_train, y_val = y_seq[:split_idx], y_seq[split_idx:]
        
        # Check class distribution
        if len(np.unique(y_train)) < 2:
            result['status'] = 'skipped'
            result['metrics'] = {'note': 'Train set has only 1 class'}
            return result
        
        # Build and train LSTM (CFG-3)
        n_features = X_train.shape[2]
        n_classes = len(np.unique(y_train))
        
        model = keras.Sequential([
            layers.LSTM(32, input_shape=(seq_len, n_features), return_sequences=False),
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.4),
            layers.Dense(n_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Class weights
        weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
        class_weight = dict(enumerate(weights))
        
        # Early stopping
        callbacks = [keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=3, restore_best_weights=True, verbose=0
        )]
        
        # Train
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=50,
            batch_size=16,
            class_weight=class_weight,
            callbacks=callbacks,
            verbose=0
        )
        
        # Evaluate
        y_pred = np.argmax(model.predict(X_val, verbose=0), axis=1)
        f1_macro = f1_score(y_val, y_pred, average='macro', zero_division=0)
        bal_acc = balanced_accuracy_score(y_val, y_pred)
        
        result['selected_config'] = 'CFG-3'
        result['selected_target'] = 'som_binary'
        result['config_params'] = {
            'seq_len': 14,
            'lstm_units': 32,
            'dense_units': 32,
            'dropout': 0.4,
            'early_stopping': True,
            'class_weights': True
        }
        result['metrics'] = {
            'f1_macro': float(f1_macro),
            'balanced_accuracy': float(bal_acc),
            'n_epochs_trained': len(history.history['loss']),
        }
        result['tested_configs'].append({
            'config': 'CFG-3',
            'target': 'som_binary',
            'f1_macro': float(f1_macro)
        })
        
    except ImportError:
        logger.warning("[Experiment Suite] TensorFlow not available, skipping LSTM")
        result['status'] = 'skipped'
        result['metrics'] = {'note': 'TensorFlow not available'}
        result['selected_config'] = 'CFG-3'
        result['selected_target'] = 'som_binary'
        result['config_params'] = {
            'seq_len': 14, 'lstm_units': 32, 'dense_units': 32,
            'dropout': 0.4, 'early_stopping': True, 'class_weights': True
        }
    except Exception as e:
        logger.warning(f"[Experiment Suite] LSTM training failed: {e}")
        result['status'] = 'error'
        result['error'] = str(e)
        result['selected_config'] = 'CFG-3'
        result['selected_target'] = 'som_binary'
        result['config_params'] = {
            'seq_len': 14, 'lstm_units': 32, 'dense_units': 32,
            'dropout': 0.4, 'early_stopping': True, 'class_weights': True
        }
    
    return result


# =============================================================================
# Main Experiment Suite Entry Point
# =============================================================================

def run_experiment_suite(
    df_universe: pd.DataFrame,
    participant: str,
    snapshot: str,
    output_dir: Optional[Path] = None,
    skip_ml7_lstm: bool = False
) -> Dict:
    """
    Run full Experiment Suite on Feature Universe.
    
    This is the main entry point called from Stage 5.
    
    Args:
        df_universe: Feature Universe DataFrame (post-MICE)
        participant: Participant ID (e.g., 'P000001')
        snapshot: Snapshot date (e.g., '2025-12-08')
        output_dir: Output directory for artifacts
        skip_ml7_lstm: If True, skip LSTM training (faster)
    
    Returns:
        model_selection dict ready to save as JSON
    """
    logger.info("[Experiment Suite] Starting feature selection ablation...")
    
    # Set up output directory
    if output_dir is None:
        output_dir = Path(f"ai/local/{participant}/{snapshot}")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize model selection structure
    model_selection = {
        'snapshot': snapshot,
        'participant': participant,
        'generated_at': datetime.now().isoformat(),
        'experiment_suite_version': '1.0.0',
        'n_samples': len(df_universe),
        'ml6': {},
        'ml7': {},
    }
    
    # =========================================================================
    # ML6 Ablation (LogisticRegression)
    # =========================================================================
    logger.info("[Experiment Suite] Running ML6 feature set ablation...")
    
    try:
        ml6_results = run_ml6_ablation_quick(df_universe)
        
        if ml6_results:
            # Select best based on deterministic rules
            best_ml6 = select_best_config(ml6_results, 'ml6')
            
            if best_ml6:
                model_selection['ml6'] = {
                    'selected_fs': best_ml6['feature_set'],
                    'selected_target': best_ml6['target'],
                    'features': best_ml6['features'],
                    'n_features': best_ml6['n_features'],
                    'metrics': {
                        'f1_macro': best_ml6['aggregate_metrics']['mean_f1_macro'],
                        'f1_macro_std': best_ml6['aggregate_metrics']['std_f1_macro'],
                        'balanced_accuracy': best_ml6['aggregate_metrics']['mean_balanced_accuracy'],
                        'cohen_kappa': best_ml6['aggregate_metrics']['mean_cohen_kappa'],
                    },
                    'selection_reason': 'Highest F1-macro among FS candidates',
                    'all_results': {
                        f"{r['feature_set']}_{r['target']}": {
                            'f1_macro': r['aggregate_metrics']['mean_f1_macro'],
                            'balanced_accuracy': r['aggregate_metrics']['mean_balanced_accuracy'],
                            'n_features': r['n_features'],
                        }
                        for r in ml6_results
                    }
                }
                logger.info(f"  [ML6] Best: {best_ml6['feature_set']} × {best_ml6['target']} "
                           f"(F1={best_ml6['aggregate_metrics']['mean_f1_macro']:.4f})")
            else:
                raise ValueError("No successful ML6 experiments")
        else:
            raise ValueError("ML6 ablation returned no results")
            
    except Exception as e:
        logger.warning(f"[Experiment Suite] ML6 ablation failed: {e}")
        logger.warning("[Experiment Suite] Using FS-B defaults for ML6")
        model_selection['ml6'] = {
            'selected_fs': 'FS-B',
            'selected_target': 'binary',
            'features': [f for f in FS_B_FEATURES if f in df_universe.columns],
            'n_features': len([f for f in FS_B_FEATURES if f in df_universe.columns]),
            'metrics': {'f1_macro': 0.0, 'note': f'Fallback due to: {str(e)}'},
            'selection_reason': 'Fallback to FS-B defaults',
            'fallback': True,
            'fallback_reason': str(e)
        }
    
    # =========================================================================
    # ML7 Ablation (LSTM)
    # =========================================================================
    logger.info("[Experiment Suite] Running ML7 configuration test...")
    
    try:
        # Use same feature set as ML6 for consistency
        ml6_fs = model_selection['ml6'].get('selected_fs', 'FS-B')
        ml7_result = run_ml7_ablation_quick(df_universe, feature_set=ml6_fs, skip_lstm=skip_ml7_lstm)
        
        model_selection['ml7'] = {
            'selected_config': ml7_result.get('selected_config', 'CFG-3'),
            'selected_fs': ml7_result.get('feature_set', ml6_fs),
            'selected_target': ml7_result.get('selected_target', 'som_binary'),
            'features': ml7_result.get('features', model_selection['ml6'].get('features', [])),
            'n_features': ml7_result.get('n_features', model_selection['ml6'].get('n_features', 0)),
            'config_params': ml7_result.get('config_params', {
                'seq_len': 14, 'lstm_units': 32, 'dense_units': 32,
                'dropout': 0.4, 'early_stopping': True, 'class_weights': True
            }),
            'metrics': ml7_result.get('metrics', {}),
            'status': ml7_result.get('status', 'unknown'),
            'selection_reason': 'CFG-3 (regularized) with same FS as ML6'
        }
        
        if ml7_result.get('status') == 'success':
            logger.info(f"  [ML7] {ml7_result['selected_config']} × {ml7_result['selected_target']} "
                       f"(F1={ml7_result['metrics'].get('f1_macro', 0):.4f})")
        else:
            logger.info(f"  [ML7] Status: {ml7_result.get('status')} - using defaults")
            
    except Exception as e:
        logger.warning(f"[Experiment Suite] ML7 test failed: {e}")
        model_selection['ml7'] = {
            'selected_config': 'CFG-3',
            'selected_fs': model_selection['ml6'].get('selected_fs', 'FS-B'),
            'selected_target': 'som_binary',
            'features': model_selection['ml6'].get('features', []),
            'n_features': model_selection['ml6'].get('n_features', 0),
            'config_params': {
                'seq_len': 14, 'lstm_units': 32, 'dense_units': 32,
                'dropout': 0.4, 'early_stopping': True, 'class_weights': True
            },
            'metrics': {'note': f'Fallback due to: {str(e)}'},
            'status': 'fallback',
            'selection_reason': 'Fallback to CFG-3 defaults'
        }
    
    # =========================================================================
    # Save model_selection.json
    # =========================================================================
    model_selection_path = output_dir / 'model_selection.json'
    with open(model_selection_path, 'w') as f:
        json.dump(model_selection, f, indent=2, default=str)
    
    logger.info(f"[Experiment Suite] Saved: {model_selection_path}")
    logger.info("[Experiment Suite] Complete.")
    
    return model_selection


# =============================================================================
# CLI Interface (for standalone testing)
# =============================================================================

def main():
    """Run Experiment Suite from command line."""
    import argparse
    
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(levelname)s: %(message)s'
    )
    
    parser = argparse.ArgumentParser(description='Run Experiment Suite')
    parser.add_argument('--participant', '-p', default='P000001')
    parser.add_argument('--snapshot', '-s', default='2025-12-08')
    parser.add_argument('--skip-lstm', action='store_true')
    args = parser.parse_args()
    
    # Load feature universe
    universe_path = Path(f"ai/local/{args.participant}/{args.snapshot}/ml6/features_daily_ml_universe.csv")
    
    # Fallback to ml6 data if universe doesn't exist
    if not universe_path.exists():
        universe_path = Path(f"ai/local/{args.participant}/{args.snapshot}/ml6/features_daily_ml6.csv")
    
    if not universe_path.exists():
        logger.error(f"Data not found: {universe_path}")
        return 1
    
    df = pd.read_csv(universe_path)
    df['date'] = pd.to_datetime(df['date'])
    
    # Run experiment suite
    result = run_experiment_suite(
        df, args.participant, args.snapshot,
        skip_ml7_lstm=args.skip_lstm
    )
    
    print(json.dumps(result, indent=2))
    return 0


if __name__ == '__main__':
    exit(main())
