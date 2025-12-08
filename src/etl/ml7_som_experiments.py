"""
ML7 SoM Experiments Module
LSTM architecture ablation study for SoM prediction

This module runs a controlled experiment grid comparing:
- CFG-1: Simple baseline (7d/16u/16d/0.2)
- CFG-2: Current/Legacy (14d/32u/32d/0.2)
- CFG-3: Regularized (14d/32u/32d/0.4 + early stopping + class weights)

Each configuration is evaluated on both som_3class and som_binary targets.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import logging
import json
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration Classes
# =============================================================================

@dataclass
class LSTMConfig:
    """Configuration for LSTM experiment."""
    name: str
    seq_len: int
    lstm_units: int
    dense_units: int
    dropout: float
    use_early_stopping: bool = False
    early_stopping_patience: int = 5
    use_class_weight: bool = False
    epochs: int = 30
    batch_size: int = 16
    
    def to_dict(self) -> dict:
        return asdict(self)


# Predefined configurations
CFG_1 = LSTMConfig(
    name="CFG-1 (Simple)",
    seq_len=7,
    lstm_units=16,
    dense_units=16,
    dropout=0.2,
    use_early_stopping=True,
    early_stopping_patience=5,
    use_class_weight=False,
    epochs=30,
    batch_size=16
)

CFG_2 = LSTMConfig(
    name="CFG-2 (Legacy)",
    seq_len=14,
    lstm_units=32,
    dense_units=32,
    dropout=0.2,
    use_early_stopping=False,
    use_class_weight=False,
    epochs=20,
    batch_size=32
)

CFG_3 = LSTMConfig(
    name="CFG-3 (Regularized)",
    seq_len=14,
    lstm_units=32,
    dense_units=32,
    dropout=0.4,
    use_early_stopping=True,
    early_stopping_patience=3,
    use_class_weight=True,
    epochs=50,
    batch_size=16
)

ALL_CONFIGS = [CFG_1, CFG_2, CFG_3]


# =============================================================================
# Feature Set Definition (FS-B from ML6)
# =============================================================================

FS_B_FEATURE_COLS = [
    # Sleep (2)
    'sleep_hours',
    'sleep_quality_score',
    # Cardio (5)
    'hr_mean',
    'hr_min',
    'hr_max',
    'hr_std',
    'hr_samples',
    # HRV (5)
    'hrv_sdnn_mean',
    'hrv_sdnn_median',
    'hrv_sdnn_min',
    'hrv_sdnn_max',
    'n_hrv_sdnn',
    # Activity (3)
    'total_steps',
    'total_distance',
    'total_active_energy',
]


# =============================================================================
# Sequence Creation
# =============================================================================

def create_sequences(X: np.ndarray, y: np.ndarray, seq_len: int) -> Tuple[np.ndarray, np.ndarray]:
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
        y_seq.append(y[i+seq_len-1])  # Predict label at END of sequence
    
    return np.array(X_seq), np.array(y_seq)


# =============================================================================
# LSTM Training
# =============================================================================

def build_lstm_model(config: LSTMConfig, n_features: int, n_classes: int):
    """Build LSTM model according to configuration."""
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    
    model = keras.Sequential([
        layers.LSTM(config.lstm_units, 
                   input_shape=(config.seq_len, n_features), 
                   return_sequences=False),
        layers.Dense(config.dense_units, activation='relu'),
        layers.Dropout(config.dropout),
        layers.Dense(n_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def train_lstm_with_config(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    config: LSTMConfig,
    n_classes: int
) -> Dict:
    """
    Train LSTM model with specified configuration.
    
    Returns dict with metrics and training history.
    """
    import tensorflow as tf
    from tensorflow import keras
    from sklearn.metrics import f1_score, balanced_accuracy_score, confusion_matrix
    from sklearn.utils.class_weight import compute_class_weight
    
    try:
        n_features = X_train.shape[2]
        
        # Map labels to 0-indexed for Keras
        unique_labels = sorted(set(y_train.tolist() + y_val.tolist()))
        label_map = {label: idx for idx, label in enumerate(unique_labels)}
        reverse_map = {idx: label for label, idx in label_map.items()}
        
        y_train_mapped = np.array([label_map[y] for y in y_train])
        y_val_mapped = np.array([label_map[y] for y in y_val])
        
        # Build model
        model = build_lstm_model(config, n_features, n_classes)
        
        # Callbacks
        callbacks = []
        if config.use_early_stopping:
            callbacks.append(keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=config.early_stopping_patience,
                restore_best_weights=True,
                verbose=0
            ))
        
        # Class weights
        class_weight = None
        if config.use_class_weight:
            weights = compute_class_weight(
                'balanced',
                classes=np.unique(y_train_mapped),
                y=y_train_mapped
            )
            class_weight = dict(enumerate(weights))
        
        # Train
        history = model.fit(
            X_train, y_train_mapped,
            validation_data=(X_val, y_val_mapped),
            epochs=config.epochs,
            batch_size=config.batch_size,
            class_weight=class_weight,
            callbacks=callbacks,
            verbose=0
        )
        
        # Evaluate
        y_pred_proba = model.predict(X_val, verbose=0)
        y_pred_mapped = np.argmax(y_pred_proba, axis=1)
        y_pred = np.array([reverse_map[y] for y in y_pred_mapped])
        
        # Metrics
        f1_macro = f1_score(y_val, y_pred, average='macro', zero_division=0)
        f1_weighted = f1_score(y_val, y_pred, average='weighted', zero_division=0)
        balanced_acc = balanced_accuracy_score(y_val, y_pred)
        conf_matrix = confusion_matrix(y_val, y_pred, labels=unique_labels).tolist()
        
        # Training stats
        n_epochs_trained = len(history.history['loss'])
        final_train_loss = history.history['loss'][-1]
        final_val_loss = history.history['val_loss'][-1]
        final_val_acc = history.history['val_accuracy'][-1]
        
        return {
            'success': True,
            'config_name': config.name,
            'n_features': n_features,
            'n_classes': n_classes,
            'n_train_seq': len(X_train),
            'n_val_seq': len(X_val),
            'f1_macro': float(f1_macro),
            'f1_weighted': float(f1_weighted),
            'balanced_accuracy': float(balanced_acc),
            'confusion_matrix': conf_matrix,
            'class_labels': unique_labels,
            'n_epochs_trained': n_epochs_trained,
            'final_train_loss': float(final_train_loss),
            'final_val_loss': float(final_val_loss),
            'final_val_accuracy': float(final_val_acc),
            'early_stopped': n_epochs_trained < config.epochs if config.use_early_stopping else False,
            'model': model
        }
        
    except Exception as e:
        logger.error(f"[LSTM] Training error: {e}")
        return {
            'success': False,
            'config_name': config.name,
            'error': str(e)
        }


# =============================================================================
# Experiment Runner
# =============================================================================

def run_single_experiment(
    df: pd.DataFrame,
    config: LSTMConfig,
    target_col: str,
    feature_cols: List[str] = None
) -> Dict:
    """
    Run a single LSTM experiment.
    
    Args:
        df: DataFrame with features and target
        config: LSTM configuration
        target_col: 'som_category_3class' or 'som_binary'
        feature_cols: List of feature columns (default: FS_B_FEATURE_COLS)
    
    Returns:
        Dict with experiment results
    """
    if feature_cols is None:
        feature_cols = FS_B_FEATURE_COLS
    
    # Validate features exist
    missing_features = [f for f in feature_cols if f not in df.columns]
    if missing_features:
        return {
            'success': False,
            'config_name': config.name,
            'target': target_col,
            'error': f"Missing features: {missing_features}"
        }
    
    # Validate target exists
    if target_col not in df.columns:
        return {
            'success': False,
            'config_name': config.name,
            'target': target_col,
            'error': f"Target column '{target_col}' not found"
        }
    
    # Prepare data
    X = df[feature_cols].values
    y = df[target_col].values
    
    # Get class info
    unique_classes = sorted(set(y))
    n_classes = len(unique_classes)
    
    # Check minimum samples
    n_samples = len(df)
    n_sequences = n_samples - config.seq_len + 1
    
    if n_sequences < 10:
        return {
            'success': False,
            'config_name': config.name,
            'target': target_col,
            'error': f"Only {n_sequences} sequences possible (need 10+)"
        }
    
    # Create sequences
    X_seq, y_seq = create_sequences(X, y, config.seq_len)
    
    # 80/20 temporal split
    split_idx = int(0.8 * len(X_seq))
    X_train, X_val = X_seq[:split_idx], X_seq[split_idx:]
    y_train, y_val = y_seq[:split_idx], y_seq[split_idx:]
    
    # Check class distribution in train/val
    train_classes = set(y_train)
    val_classes = set(y_val)
    
    if len(train_classes) < 2:
        return {
            'success': False,
            'config_name': config.name,
            'target': target_col,
            'error': f"Train set has only {len(train_classes)} class(es)"
        }
    
    # Train model
    result = train_lstm_with_config(X_train, y_train, X_val, y_val, config, n_classes)
    result['target'] = target_col
    result['seq_len'] = config.seq_len
    
    return result


def run_ablation_study(
    df: pd.DataFrame,
    configs: List[LSTMConfig] = None,
    targets: List[str] = None,
    output_dir: Path = None
) -> Dict:
    """
    Run full ablation study: all configs × all targets.
    
    Args:
        df: DataFrame with features and targets
        configs: List of configurations (default: ALL_CONFIGS)
        targets: List of target columns (default: both SoM targets)
        output_dir: Directory to save results
    
    Returns:
        Dict with all experiment results
    """
    if configs is None:
        configs = ALL_CONFIGS
    
    if targets is None:
        targets = ['som_category_3class', 'som_binary']
    
    logger.info("=" * 70)
    logger.info("ML7 LSTM ABLATION STUDY")
    logger.info("=" * 70)
    logger.info(f"Configurations: {len(configs)}")
    logger.info(f"Targets: {targets}")
    logger.info(f"Total experiments: {len(configs) * len(targets)}")
    logger.info("")
    
    results = []
    
    for config in configs:
        for target in targets:
            logger.info(f"Running: {config.name} × {target}")
            
            result = run_single_experiment(df, config, target)
            results.append(result)
            
            if result['success']:
                logger.info(f"  → F1-macro: {result['f1_macro']:.4f}, "
                           f"BA: {result['balanced_accuracy']:.4f}, "
                           f"Epochs: {result['n_epochs_trained']}")
            else:
                logger.warning(f"  → FAILED: {result.get('error', 'Unknown')}")
    
    # Find best result
    successful = [r for r in results if r['success']]
    if successful:
        best = max(successful, key=lambda r: r['f1_macro'])
        logger.info("")
        logger.info(f"Best: {best['config_name']} × {best['target']}")
        logger.info(f"  F1-macro: {best['f1_macro']:.4f}")
    
    # Package results
    study_results = {
        'timestamp': datetime.now().isoformat(),
        'n_experiments': len(results),
        'n_successful': len(successful),
        'configs': [c.to_dict() for c in configs],
        'targets': targets,
        'results': [
            {k: v for k, v in r.items() if k != 'model'}  # Exclude model
            for r in results
        ],
        'best_config': best['config_name'] if successful else None,
        'best_target': best['target'] if successful else None,
        'best_f1_macro': best['f1_macro'] if successful else None
    }
    
    # Save results
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        results_path = output_dir / 'lstm_ablation_results.json'
        with open(results_path, 'w') as f:
            json.dump(study_results, f, indent=2, default=str)
        logger.info(f"Results saved: {results_path}")
    
    return study_results


# =============================================================================
# Markdown Report Generation
# =============================================================================

def generate_markdown_report(
    study_results: Dict,
    participant: str,
    snapshot: str,
    output_path: Path
) -> str:
    """
    Generate PhD-grade markdown report from ablation study results.
    """
    lines = [
        f"# ML7 LSTM Ablation Study: {participant} / {snapshot}",
        "",
        f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Purpose**: Compare LSTM configurations for SoM prediction",
        "",
        "---",
        "",
        "## 1. Experiment Setup",
        "",
        "### Feature Set: FS-B (from ML6 ablation)",
        "",
        "| Category | Features | Count |",
        "|----------|----------|-------|",
        "| Sleep | `sleep_hours`, `sleep_quality_score` | 2 |",
        "| Cardio | `hr_mean`, `hr_min`, `hr_max`, `hr_std`, `hr_samples` | 5 |",
        "| HRV | `hrv_sdnn_*` (5 metrics) | 5 |",
        "| Activity | `total_steps`, `total_distance`, `total_active_energy` | 3 |",
        "| **Total** | | **15** |",
        "",
        "### Configurations Tested",
        "",
        "| Config | Seq Len | LSTM | Dense | Dropout | Early Stop | Class Wt |",
        "|--------|---------|------|-------|---------|------------|----------|",
    ]
    
    for cfg in study_results.get('configs', []):
        lines.append(
            f"| {cfg['name']} | {cfg['seq_len']} | {cfg['lstm_units']} | "
            f"{cfg['dense_units']} | {cfg['dropout']} | "
            f"{'Yes' if cfg['use_early_stopping'] else 'No'} | "
            f"{'Yes' if cfg['use_class_weight'] else 'No'} |"
        )
    
    lines.extend([
        "",
        "### Targets Tested",
        "",
        "- `som_category_3class`: 3-class (-1, 0, +1)",
        "- `som_binary`: Binary (0=stable, 1=unstable)",
        "",
        "---",
        "",
        "## 2. Results Summary",
        "",
        "| Config | Target | F1-Macro | F1-Weighted | Bal. Acc | Epochs | Status |",
        "|--------|--------|----------|-------------|----------|--------|--------|",
    ])
    
    for r in study_results.get('results', []):
        if r['success']:
            lines.append(
                f"| {r['config_name']} | {r['target']} | "
                f"**{r['f1_macro']:.4f}** | {r['f1_weighted']:.4f} | "
                f"{r['balanced_accuracy']:.4f} | {r['n_epochs_trained']} | ✓ |"
            )
        else:
            lines.append(
                f"| {r['config_name']} | {r['target']} | - | - | - | - | "
                f"✗ {r.get('error', 'Failed')[:20]} |"
            )
    
    # Find best per target
    results = study_results.get('results', [])
    successful = [r for r in results if r['success']]
    
    best_3class = max([r for r in successful if r['target'] == 'som_category_3class'], 
                      key=lambda r: r['f1_macro'], default=None)
    best_binary = max([r for r in successful if r['target'] == 'som_binary'], 
                      key=lambda r: r['f1_macro'], default=None)
    
    lines.extend([
        "",
        "### Best Per Target",
        "",
    ])
    
    if best_3class:
        lines.append(f"- **som_3class**: {best_3class['config_name']} (F1={best_3class['f1_macro']:.4f})")
    if best_binary:
        lines.append(f"- **som_binary**: {best_binary['config_name']} (F1={best_binary['f1_macro']:.4f})")
    
    # Overall best
    if successful:
        best_overall = max(successful, key=lambda r: r['f1_macro'])
        lines.extend([
            "",
            f"### Overall Best: {best_overall['config_name']} × {best_overall['target']}",
            "",
            f"- **F1-Macro**: {best_overall['f1_macro']:.4f}",
            f"- **F1-Weighted**: {best_overall['f1_weighted']:.4f}",
            f"- **Balanced Accuracy**: {best_overall['balanced_accuracy']:.4f}",
            f"- **Epochs Trained**: {best_overall['n_epochs_trained']}",
            f"- **Early Stopped**: {'Yes' if best_overall.get('early_stopped') else 'No'}",
        ])
    
    lines.extend([
        "",
        "---",
        "",
        "## 3. Analysis",
        "",
        "### Impact of Sequence Length",
        "",
    ])
    
    # Compare 7-day vs 14-day
    cfg1_results = [r for r in successful if 'Simple' in r.get('config_name', '')]
    cfg2_results = [r for r in successful if 'Legacy' in r.get('config_name', '')]
    
    if cfg1_results and cfg2_results:
        cfg1_f1 = np.mean([r['f1_macro'] for r in cfg1_results])
        cfg2_f1 = np.mean([r['f1_macro'] for r in cfg2_results])
        diff = cfg1_f1 - cfg2_f1
        lines.append(f"- **7-day (CFG-1)** avg F1: {cfg1_f1:.4f}")
        lines.append(f"- **14-day (CFG-2)** avg F1: {cfg2_f1:.4f}")
        lines.append(f"- **Difference**: {diff:+.4f} ({'shorter better' if diff > 0 else 'longer better'})")
    
    lines.extend([
        "",
        "### Impact of Regularization",
        "",
    ])
    
    # Compare CFG-2 vs CFG-3
    cfg3_results = [r for r in successful if 'Regularized' in r.get('config_name', '')]
    
    if cfg2_results and cfg3_results:
        cfg2_f1 = np.mean([r['f1_macro'] for r in cfg2_results])
        cfg3_f1 = np.mean([r['f1_macro'] for r in cfg3_results])
        diff = cfg3_f1 - cfg2_f1
        lines.append(f"- **No regularization (CFG-2)** avg F1: {cfg2_f1:.4f}")
        lines.append(f"- **With regularization (CFG-3)** avg F1: {cfg3_f1:.4f}")
        lines.append(f"- **Difference**: {diff:+.4f} ({'helps' if diff > 0 else 'hurts'})")
    
    lines.extend([
        "",
        "### Target Variable Comparison",
        "",
    ])
    
    if best_3class and best_binary:
        diff = best_binary['f1_macro'] - best_3class['f1_macro']
        lines.append(f"- **3-class best**: F1={best_3class['f1_macro']:.4f}")
        lines.append(f"- **Binary best**: F1={best_binary['f1_macro']:.4f}")
        lines.append(f"- **Difference**: {diff:+.4f} ({'binary better' if diff > 0 else '3-class better'})")
    
    lines.extend([
        "",
        "---",
        "",
        "## 4. Recommendations",
        "",
    ])
    
    if successful:
        best = max(successful, key=lambda r: r['f1_macro'])
        
        # Get config details
        best_cfg = None
        for cfg in study_results.get('configs', []):
            if cfg['name'] == best['config_name']:
                best_cfg = cfg
                break
        
        lines.extend([
            f"### Recommended Configuration: {best['config_name']}",
            "",
            f"- **Sequence Length**: {best_cfg['seq_len'] if best_cfg else 'N/A'} days",
            f"- **LSTM Units**: {best_cfg['lstm_units'] if best_cfg else 'N/A'}",
            f"- **Dense Units**: {best_cfg['dense_units'] if best_cfg else 'N/A'}",
            f"- **Dropout**: {best_cfg['dropout'] if best_cfg else 'N/A'}",
            f"- **Early Stopping**: {'Yes' if best_cfg and best_cfg['use_early_stopping'] else 'No'}",
            f"- **Class Weights**: {'Yes' if best_cfg and best_cfg['use_class_weight'] else 'No'}",
            f"- **Target**: `{best['target']}`",
            "",
            "### Rationale",
            "",
        ])
        
        # Generate rationale based on results
        if best_cfg and best_cfg['seq_len'] == 7:
            lines.append("- Shorter 7-day sequences reduce overfitting with limited SoM data")
        if best_cfg and best_cfg['dropout'] >= 0.4:
            lines.append("- Higher dropout (0.4) provides needed regularization")
        if best_cfg and best_cfg['use_class_weight']:
            lines.append("- Class weights address severe class imbalance (70% majority class)")
        if best['target'] == 'som_binary':
            lines.append("- Binary target provides cleaner signal than 3-class")
    
    lines.extend([
        "",
        "---",
        "",
        "## 5. Limitations",
        "",
        "1. **Small Dataset**: Only ~60-70 sequences available for training",
        "2. **Single Split**: 80/20 temporal split, no cross-validation",
        "3. **HRV Sparsity**: HRV features have <5% coverage, rely on MICE imputation",
        "4. **Temporal Drift**: SoM distribution changes over time (detected in drift analysis)",
        "",
        "---",
        "",
        f"**Report Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
    ])
    
    report = '\n'.join(lines)
    
    # Save report
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    logger.info(f"Report saved: {output_path}")
    
    return report


# =============================================================================
# CLI Interface
# =============================================================================

def main():
    """Run ML7 LSTM ablation study from command line."""
    import argparse
    
    parser = argparse.ArgumentParser(description='ML7 LSTM Ablation Study')
    parser.add_argument('--participant', required=True, help='Participant ID')
    parser.add_argument('--snapshot', required=True, help='Snapshot date')
    parser.add_argument('--data-dir', default='data/ai', help='AI data directory')
    parser.add_argument('--output-dir', default='docs/reports', help='Output directory')
    
    args = parser.parse_args()
    
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(levelname)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Load data
    data_path = Path(args.data_dir) / args.participant / args.snapshot / 'ml6' / 'features_daily_ml6.csv'
    
    if not data_path.exists():
        logger.error(f"Data not found: {data_path}")
        return 1
    
    df = pd.read_csv(data_path)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    
    logger.info(f"Loaded {len(df)} samples from {data_path}")
    
    # Run ablation
    output_dir = Path(args.output_dir)
    study_results = run_ablation_study(df, output_dir=output_dir / 'ablation')
    
    # Generate report
    report_path = output_dir / f'ML7_SOM_lstm_experiments_{args.participant}_{args.snapshot}.md'
    generate_markdown_report(study_results, args.participant, args.snapshot, report_path)
    
    return 0


if __name__ == '__main__':
    exit(main())
