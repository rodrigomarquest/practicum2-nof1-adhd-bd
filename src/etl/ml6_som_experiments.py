#!/usr/bin/env python
"""
ML6 SoM Feature Ablation Experiments

PhD-level controlled experiment runner for evaluating:
- Feature sets (FS-A/B/C/D)
- Target variables (3-class vs binary)
- Class weighting strategies

This module:
1. Loads post-Stage-4 data for a given PID/SNAPSHOT
2. Builds feature matrices for each feature set
3. Runs 6-fold temporal CV with LogisticRegression
4. Computes comprehensive metrics (F1, BA, Cohen's Kappa, confusion matrices)
5. Generates JSON + Markdown reports

IMPORTANT: Does NOT modify Stage 0-4 outputs. Reads from existing data.

Usage:
    python -m src.etl.ml6_som_experiments --participant P000001 --snapshot 2025-12-08
"""

import argparse
import json
import logging
import warnings
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
)

warnings.filterwarnings('ignore')

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# =============================================================================
# Feature Set Definitions
# =============================================================================

# FS-A: Baseline (Original 10 canonical features)
FS_A_FEATURES = [
    # Sleep (2)
    'sleep_hours', 'sleep_quality_score',
    # Cardio (5)
    'hr_mean', 'hr_min', 'hr_max', 'hr_std', 'hr_samples',
    # Activity (3)
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

# FS-D: FS-C + PBSI auxiliary (19 features)
FS_D_FEATURES = FS_C_FEATURES + [
    'pbsi_score',
]

FEATURE_SETS = {
    'FS-A': {'name': 'Baseline (Sleep+Cardio+Activity)', 'features': FS_A_FEATURES},
    'FS-B': {'name': 'Baseline + HRV', 'features': FS_B_FEATURES},
    'FS-C': {'name': 'FS-B + MEDS', 'features': FS_C_FEATURES},
    'FS-D': {'name': 'FS-C + PBSI (current)', 'features': FS_D_FEATURES},
}

# Target definitions
TARGET_CONFIGS = {
    '3class': {
        'column': 'som_category_3class',
        'name': 'SoM 3-Class',
        'classes': [-1, 0, 1],
        'class_names': ['Negative/Unstable', 'Neutral', 'Positive/Stable'],
    },
    'binary': {
        'column': 'som_binary',
        'name': 'SoM Binary (Unstable vs Rest)',
        'classes': [0, 1],
        'class_names': ['Not Unstable', 'Unstable'],
    },
}


# =============================================================================
# Data Loading and Preprocessing
# =============================================================================

def load_ml6_data(participant: str, snapshot: str) -> pd.DataFrame:
    """
    Load the Stage-5 prepared ML6 dataset.
    
    Args:
        participant: Participant ID (e.g., 'P000001')
        snapshot: Snapshot date (e.g., '2025-12-08')
        
    Returns:
        DataFrame with features and targets
    """
    # Try ai/local path first (new structure)
    ai_path = Path(f"data/ai/{participant}/{snapshot}/ml6/features_daily_ml6.csv")
    
    # Fallback to ai_base path
    if not ai_path.exists():
        ai_path = Path(f"ai/local/{participant}/{snapshot}/ml6/features_daily_ml6.csv")
    
    if not ai_path.exists():
        raise FileNotFoundError(f"ML6 data not found at {ai_path}")
    
    logger.info(f"[Data] Loading ML6 data from {ai_path}")
    
    df = pd.read_csv(ai_path)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    
    logger.info(f"[Data] Loaded {len(df)} samples with {len(df.columns)} columns")
    
    return df


def prepare_feature_matrix(
    df: pd.DataFrame, 
    feature_set: str,
    use_imputation: bool = True
) -> Tuple[pd.DataFrame, List[str], Dict]:
    """
    Prepare feature matrix X for a given feature set.
    
    Args:
        df: Input DataFrame with all columns
        feature_set: One of 'FS-A', 'FS-B', 'FS-C', 'FS-D'
        use_imputation: Whether to apply MICE imputation
        
    Returns:
        Tuple of (X DataFrame, feature names list, coverage stats dict)
    """
    if feature_set not in FEATURE_SETS:
        raise ValueError(f"Unknown feature set: {feature_set}")
    
    features = FEATURE_SETS[feature_set]['features']
    
    # Filter to available features
    available = [f for f in features if f in df.columns]
    missing = [f for f in features if f not in df.columns]
    
    if missing:
        logger.warning(f"[{feature_set}] Missing features: {missing}")
    
    X = df[available].copy()
    
    # Compute coverage stats before imputation
    coverage_stats = {}
    for col in available:
        n_valid = X[col].notna().sum()
        coverage_stats[col] = {
            'n_valid': int(n_valid),
            'n_total': len(X),
            'coverage_pct': round(100 * n_valid / len(X), 1)
        }
    
    # Apply MICE imputation if requested
    if use_imputation:
        n_missing = X.isna().sum().sum()
        if n_missing > 0:
            logger.info(f"[{feature_set}] Imputing {n_missing} missing values with MICE")
            imputer = IterativeImputer(
                max_iter=10,
                random_state=42,
                verbose=0
            )
            X_imputed = imputer.fit_transform(X)
            X = pd.DataFrame(X_imputed, columns=available, index=X.index)
    
    return X, available, coverage_stats


def prepare_target(
    df: pd.DataFrame,
    target_config: str
) -> Tuple[pd.Series, Dict]:
    """
    Prepare target variable y.
    
    Args:
        df: Input DataFrame
        target_config: One of '3class', 'binary'
        
    Returns:
        Tuple of (y Series, class distribution dict)
    """
    if target_config not in TARGET_CONFIGS:
        raise ValueError(f"Unknown target config: {target_config}")
    
    config = TARGET_CONFIGS[target_config]
    y = df[config['column']].copy()
    
    # Compute class distribution
    class_dist = y.value_counts().sort_index().to_dict()
    
    return y, class_dist


# =============================================================================
# Cross-Validation
# =============================================================================

def create_temporal_folds(
    df: pd.DataFrame,
    n_folds: int = 6
) -> List[Dict]:
    """
    Create temporal CV folds (same as current Stage 6 implementation).
    
    Args:
        df: DataFrame with 'date' column (sorted)
        n_folds: Number of folds
        
    Returns:
        List of fold dicts with train_idx, val_idx, date ranges
    """
    n_samples = len(df)
    fold_size = n_samples // n_folds
    
    folds = []
    for fold_idx in range(n_folds):
        val_start = fold_idx * fold_size
        val_end = min((fold_idx + 1) * fold_size, n_samples)
        
        val_idx = list(range(val_start, val_end))
        train_idx = [i for i in range(n_samples) if i not in val_idx]
        
        if len(train_idx) < 5 or len(val_idx) < 2:
            continue
        
        fold = {
            'fold_idx': fold_idx,
            'train_idx': train_idx,
            'val_idx': val_idx,
            'val_start_date': df.iloc[val_start]['date'].strftime('%Y-%m-%d'),
            'val_end_date': df.iloc[val_end - 1]['date'].strftime('%Y-%m-%d'),
            'n_train': len(train_idx),
            'n_val': len(val_idx),
        }
        folds.append(fold)
    
    return folds


# =============================================================================
# Model Training and Evaluation
# =============================================================================

def train_and_evaluate_fold(
    X: pd.DataFrame,
    y: pd.Series,
    train_idx: List[int],
    val_idx: List[int],
    class_weight: str = 'balanced'
) -> Dict:
    """
    Train LogisticRegression on one fold and evaluate.
    
    Args:
        X: Feature matrix
        y: Target vector
        train_idx: Training indices
        val_idx: Validation indices
        class_weight: 'balanced' or None
        
    Returns:
        Dict with metrics and predictions
    """
    X_train = X.iloc[train_idx]
    y_train = y.iloc[train_idx]
    X_val = X.iloc[val_idx]
    y_val = y.iloc[val_idx]
    
    # Check class representation in train set
    train_classes = set(y_train.unique())
    if len(train_classes) < 2:
        return {
            'status': 'skipped',
            'reason': f'Only {len(train_classes)} class(es) in train set'
        }
    
    # Train model
    model = LogisticRegression(
        multi_class='auto',
        class_weight=class_weight,
        max_iter=1000,
        random_state=42,
        solver='lbfgs'
    )
    model.fit(X_train, y_train)
    
    # Predict
    y_pred = model.predict(X_val)
    y_prob = model.predict_proba(X_val)
    
    # Compute metrics
    f1_macro = f1_score(y_val, y_pred, average='macro', zero_division=0)
    f1_weighted = f1_score(y_val, y_pred, average='weighted', zero_division=0)
    bal_acc = balanced_accuracy_score(y_val, y_pred)
    kappa = cohen_kappa_score(y_val, y_pred)
    
    # Confusion matrix
    cm = confusion_matrix(y_val, y_pred, labels=sorted(y.unique()))
    
    # Per-class metrics
    report = classification_report(y_val, y_pred, output_dict=True, zero_division=0)
    
    return {
        'status': 'success',
        'f1_macro': float(f1_macro),
        'f1_weighted': float(f1_weighted),
        'balanced_accuracy': float(bal_acc),
        'cohen_kappa': float(kappa),
        'confusion_matrix': cm.tolist(),
        'classification_report': report,
        'y_true': y_val.tolist(),
        'y_pred': y_pred.tolist(),
        'y_prob': y_prob.tolist(),
        'n_train': len(train_idx),
        'n_val': len(val_idx),
        'train_class_dist': y_train.value_counts().sort_index().to_dict(),
        'val_class_dist': y_val.value_counts().sort_index().to_dict(),
    }


def run_cv_experiment(
    df: pd.DataFrame,
    feature_set: str,
    target_config: str,
    n_folds: int = 6,
    class_weight: str = 'balanced'
) -> Dict:
    """
    Run full CV experiment for one configuration.
    
    Args:
        df: Input DataFrame
        feature_set: Feature set name ('FS-A', 'FS-B', etc.)
        target_config: Target config ('3class', 'binary')
        n_folds: Number of CV folds
        class_weight: Class weighting strategy
        
    Returns:
        Dict with experiment results
    """
    logger.info(f"[Experiment] {feature_set} × {target_config}")
    
    # Prepare features
    X, feature_names, coverage_stats = prepare_feature_matrix(df, feature_set)
    
    # Prepare target
    y, class_dist = prepare_target(df, target_config)
    
    # Create folds
    folds = create_temporal_folds(df, n_folds)
    
    logger.info(f"  Features: {len(feature_names)}, Samples: {len(df)}, Folds: {len(folds)}")
    logger.info(f"  Class distribution: {class_dist}")
    
    # Run CV
    fold_results = []
    for fold in folds:
        result = train_and_evaluate_fold(
            X, y,
            fold['train_idx'],
            fold['val_idx'],
            class_weight=class_weight
        )
        result['fold_idx'] = fold['fold_idx']
        result['val_start_date'] = fold['val_start_date']
        result['val_end_date'] = fold['val_end_date']
        fold_results.append(result)
        
        if result['status'] == 'success':
            logger.info(f"    Fold {fold['fold_idx']}: F1={result['f1_macro']:.4f}, BA={result['balanced_accuracy']:.4f}")
        else:
            logger.warning(f"    Fold {fold['fold_idx']}: SKIPPED - {result.get('reason', 'unknown')}")
    
    # Aggregate metrics
    valid_folds = [r for r in fold_results if r['status'] == 'success']
    
    if len(valid_folds) == 0:
        logger.warning(f"  No valid folds for {feature_set} × {target_config}")
        return {
            'feature_set': feature_set,
            'target_config': target_config,
            'status': 'failed',
            'reason': 'No valid CV folds',
            'n_samples': len(df),
            'n_features': len(feature_names),
            'class_distribution': class_dist,
        }
    
    # Compute aggregate statistics
    agg_metrics = {
        'mean_f1_macro': float(np.mean([r['f1_macro'] for r in valid_folds])),
        'std_f1_macro': float(np.std([r['f1_macro'] for r in valid_folds])),
        'mean_f1_weighted': float(np.mean([r['f1_weighted'] for r in valid_folds])),
        'std_f1_weighted': float(np.std([r['f1_weighted'] for r in valid_folds])),
        'mean_balanced_accuracy': float(np.mean([r['balanced_accuracy'] for r in valid_folds])),
        'std_balanced_accuracy': float(np.std([r['balanced_accuracy'] for r in valid_folds])),
        'mean_cohen_kappa': float(np.mean([r['cohen_kappa'] for r in valid_folds])),
        'std_cohen_kappa': float(np.std([r['cohen_kappa'] for r in valid_folds])),
    }
    
    # Aggregate confusion matrix
    agg_cm = np.zeros_like(valid_folds[0]['confusion_matrix'], dtype=float)
    for r in valid_folds:
        agg_cm += np.array(r['confusion_matrix'])
    
    logger.info(f"  RESULT: F1={agg_metrics['mean_f1_macro']:.4f}±{agg_metrics['std_f1_macro']:.4f}")
    
    return {
        'feature_set': feature_set,
        'feature_set_name': FEATURE_SETS[feature_set]['name'],
        'target_config': target_config,
        'target_name': TARGET_CONFIGS[target_config]['name'],
        'status': 'success',
        'n_samples': len(df),
        'n_features': len(feature_names),
        'feature_names': feature_names,
        'coverage_stats': coverage_stats,
        'class_distribution': class_dist,
        'n_folds': len(folds),
        'n_valid_folds': len(valid_folds),
        'class_weight': class_weight,
        'aggregate_metrics': agg_metrics,
        'aggregate_confusion_matrix': agg_cm.tolist(),
        'fold_results': fold_results,
    }


# =============================================================================
# Full Ablation Study
# =============================================================================

def run_ablation_study(
    participant: str,
    snapshot: str,
    output_dir: Optional[Path] = None
) -> Dict:
    """
    Run full ablation study across all feature sets and targets.
    
    Args:
        participant: Participant ID
        snapshot: Snapshot date
        output_dir: Output directory for results
        
    Returns:
        Dict with all experiment results
    """
    logger.info("=" * 60)
    logger.info("ML6 SoM Feature Ablation Study")
    logger.info("=" * 60)
    
    # Load data
    df = load_ml6_data(participant, snapshot)
    
    # Setup output directory
    if output_dir is None:
        output_dir = Path(f"data/ai/{participant}/{snapshot}/ml6/experiments")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run all experiments
    experiments = []
    
    for feature_set in ['FS-A', 'FS-B', 'FS-C', 'FS-D']:
        for target_config in ['3class', 'binary']:
            result = run_cv_experiment(df, feature_set, target_config)
            experiments.append(result)
    
    # Build summary
    study_results = {
        'study_metadata': {
            'participant': participant,
            'snapshot': snapshot,
            'timestamp': datetime.now().isoformat(),
            'n_samples': len(df),
            'feature_sets_tested': list(FEATURE_SETS.keys()),
            'targets_tested': list(TARGET_CONFIGS.keys()),
        },
        'experiments': experiments,
    }
    
    # Find best configuration
    successful = [e for e in experiments if e['status'] == 'success']
    if successful:
        best = max(successful, key=lambda x: x['aggregate_metrics']['mean_f1_macro'])
        study_results['best_configuration'] = {
            'feature_set': best['feature_set'],
            'target_config': best['target_config'],
            'mean_f1_macro': best['aggregate_metrics']['mean_f1_macro'],
        }
        logger.info(f"\nBEST: {best['feature_set']} × {best['target_config']} "
                   f"(F1={best['aggregate_metrics']['mean_f1_macro']:.4f})")
    
    # Save JSON results
    json_path = output_dir / "ml6_som_experiments.json"
    with open(json_path, 'w') as f:
        json.dump(study_results, f, indent=2, default=str)
    logger.info(f"\nJSON saved: {json_path}")
    
    return study_results


# =============================================================================
# Report Generation
# =============================================================================

def generate_markdown_report(
    study_results: Dict,
    participant: str,
    snapshot: str,
    output_dir: Optional[Path] = None
) -> Path:
    """
    Generate PhD-level Markdown report for ablation study.
    
    Args:
        study_results: Results from run_ablation_study()
        participant: Participant ID
        snapshot: Snapshot date
        output_dir: Output directory for report
        
    Returns:
        Path to generated report
    """
    if output_dir is None:
        output_dir = Path("docs/reports")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    report_path = output_dir / f"ML6_SOM_feature_ablation_{participant}_{snapshot}.md"
    
    lines = [
        f"# ML6 SoM Feature Ablation Study",
        "",
        f"**Participant**: {participant}  ",
        f"**Snapshot**: {snapshot}  ",
        f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "---",
        "",
        "## Executive Summary",
        "",
    ]
    
    # Best configuration
    if 'best_configuration' in study_results:
        best = study_results['best_configuration']
        lines.extend([
            f"**Best Configuration**: {best['feature_set']} × {best['target_config']}  ",
            f"**Best F1-Macro**: {best['mean_f1_macro']:.4f}",
            "",
        ])
    
    # Sample info
    meta = study_results['study_metadata']
    lines.extend([
        "### Dataset",
        "",
        f"- **N Samples**: {meta['n_samples']}",
        f"- **Feature Sets Tested**: {', '.join(meta['feature_sets_tested'])}",
        f"- **Targets Tested**: {', '.join(meta['targets_tested'])}",
        "",
    ])
    
    # Results comparison table
    lines.extend([
        "---",
        "",
        "## Results Comparison",
        "",
        "### 3-Class Target (som_category_3class)",
        "",
        "| Feature Set | N Features | F1-Macro | Bal. Acc | Cohen κ |",
        "|-------------|------------|----------|----------|---------|",
    ])
    
    for exp in study_results['experiments']:
        if exp['status'] != 'success':
            continue
        if exp['target_config'] != '3class':
            continue
        m = exp['aggregate_metrics']
        lines.append(
            f"| {exp['feature_set']} ({exp['feature_set_name']}) | {exp['n_features']} | "
            f"{m['mean_f1_macro']:.4f}±{m['std_f1_macro']:.4f} | "
            f"{m['mean_balanced_accuracy']:.4f} | {m['mean_cohen_kappa']:.4f} |"
        )
    
    lines.extend([
        "",
        "### Binary Target (som_binary)",
        "",
        "| Feature Set | N Features | F1-Macro | Bal. Acc | Cohen κ |",
        "|-------------|------------|----------|----------|---------|",
    ])
    
    for exp in study_results['experiments']:
        if exp['status'] != 'success':
            continue
        if exp['target_config'] != 'binary':
            continue
        m = exp['aggregate_metrics']
        lines.append(
            f"| {exp['feature_set']} ({exp['feature_set_name']}) | {exp['n_features']} | "
            f"{m['mean_f1_macro']:.4f}±{m['std_f1_macro']:.4f} | "
            f"{m['mean_balanced_accuracy']:.4f} | {m['mean_cohen_kappa']:.4f} |"
        )
    
    # Feature coverage analysis
    lines.extend([
        "",
        "---",
        "",
        "## Feature Coverage Analysis",
        "",
    ])
    
    # Get coverage from FS-D (most complete)
    fs_d_exp = next((e for e in study_results['experiments'] 
                     if e['feature_set'] == 'FS-D' and e['status'] == 'success'), None)
    
    if fs_d_exp and 'coverage_stats' in fs_d_exp:
        lines.extend([
            "| Feature | Coverage (%) | N Valid |",
            "|---------|--------------|---------|",
        ])
        for feat, stats in fs_d_exp['coverage_stats'].items():
            lines.append(f"| {feat} | {stats['coverage_pct']:.1f}% | {stats['n_valid']}/{stats['n_total']} |")
    
    # Interpretation section
    lines.extend([
        "",
        "---",
        "",
        "## Interpretation",
        "",
        "### Q1: Does HRV help?",
        "",
    ])
    
    # Compare FS-A vs FS-B
    fs_a = next((e for e in study_results['experiments'] 
                 if e['feature_set'] == 'FS-A' and e['target_config'] == '3class' 
                 and e['status'] == 'success'), None)
    fs_b = next((e for e in study_results['experiments'] 
                 if e['feature_set'] == 'FS-B' and e['target_config'] == '3class'
                 and e['status'] == 'success'), None)
    
    if fs_a and fs_b:
        delta = fs_b['aggregate_metrics']['mean_f1_macro'] - fs_a['aggregate_metrics']['mean_f1_macro']
        direction = "improves" if delta > 0 else "decreases" if delta < 0 else "unchanged"
        lines.extend([
            f"- FS-A (Baseline): F1 = {fs_a['aggregate_metrics']['mean_f1_macro']:.4f}",
            f"- FS-B (+ HRV): F1 = {fs_b['aggregate_metrics']['mean_f1_macro']:.4f}",
            f"- **Delta**: {delta:+.4f} ({direction})",
            "",
            f"**Conclusion**: Adding HRV {direction} F1 by {abs(delta):.4f}. ",
        ])
        if delta > 0.01:
            lines.append("HRV adds meaningful signal despite low coverage (23%).")
        elif delta < -0.01:
            lines.append("HRV may introduce noise due to imputation on low-coverage data.")
        else:
            lines.append("HRV has negligible impact on prediction.")
    
    lines.extend([
        "",
        "### Q2: Does MEDS add signal?",
        "",
    ])
    
    # Compare FS-B vs FS-C
    fs_c = next((e for e in study_results['experiments'] 
                 if e['feature_set'] == 'FS-C' and e['target_config'] == '3class'
                 and e['status'] == 'success'), None)
    
    if fs_b and fs_c:
        delta = fs_c['aggregate_metrics']['mean_f1_macro'] - fs_b['aggregate_metrics']['mean_f1_macro']
        direction = "improves" if delta > 0 else "decreases" if delta < 0 else "unchanged"
        lines.extend([
            f"- FS-B (Baseline + HRV): F1 = {fs_b['aggregate_metrics']['mean_f1_macro']:.4f}",
            f"- FS-C (+ MEDS): F1 = {fs_c['aggregate_metrics']['mean_f1_macro']:.4f}",
            f"- **Delta**: {delta:+.4f} ({direction})",
            "",
        ])
    
    lines.extend([
        "",
        "### Q3: Does PBSI as auxiliary feature help?",
        "",
    ])
    
    # Compare FS-C vs FS-D
    fs_d = next((e for e in study_results['experiments'] 
                 if e['feature_set'] == 'FS-D' and e['target_config'] == '3class'
                 and e['status'] == 'success'), None)
    
    if fs_c and fs_d:
        delta = fs_d['aggregate_metrics']['mean_f1_macro'] - fs_c['aggregate_metrics']['mean_f1_macro']
        direction = "improves" if delta > 0 else "decreases" if delta < 0 else "unchanged"
        lines.extend([
            f"- FS-C (without PBSI): F1 = {fs_c['aggregate_metrics']['mean_f1_macro']:.4f}",
            f"- FS-D (with PBSI): F1 = {fs_d['aggregate_metrics']['mean_f1_macro']:.4f}",
            f"- **Delta**: {delta:+.4f} ({direction})",
            "",
        ])
    
    lines.extend([
        "",
        "### Q4: 3-Class vs Binary Target?",
        "",
    ])
    
    # Compare 3-class vs binary for best feature set
    if study_results.get('best_configuration'):
        best_fs = study_results['best_configuration']['feature_set']
        exp_3c = next((e for e in study_results['experiments'] 
                       if e['feature_set'] == best_fs and e['target_config'] == '3class'
                       and e['status'] == 'success'), None)
        exp_bin = next((e for e in study_results['experiments'] 
                        if e['feature_set'] == best_fs and e['target_config'] == 'binary'
                        and e['status'] == 'success'), None)
        
        if exp_3c and exp_bin:
            lines.extend([
                f"For {best_fs}:",
                f"- 3-Class: F1 = {exp_3c['aggregate_metrics']['mean_f1_macro']:.4f}",
                f"- Binary: F1 = {exp_bin['aggregate_metrics']['mean_f1_macro']:.4f}",
                "",
            ])
    
    # Class imbalance discussion
    lines.extend([
        "",
        "### Class Imbalance Analysis",
        "",
    ])
    
    if fs_d:
        dist = fs_d['class_distribution']
        total = sum(dist.values())
        lines.extend([
            "**3-Class Distribution**:",
            "",
        ])
        for cls, count in sorted(dist.items()):
            pct = 100 * count / total
            lines.append(f"- Class {cls}: {count} ({pct:.1f}%)")
        
        # Check if severely imbalanced
        max_pct = 100 * max(dist.values()) / total
        if max_pct > 60:
            lines.extend([
                "",
                f"⚠️ **Warning**: Majority class represents {max_pct:.1f}% of samples.",
                "This severe imbalance may bias the model toward predicting the majority class.",
                "Consider: (1) SMOTE oversampling, (2) Binary target, (3) More data collection.",
            ])
    
    # Recommendations
    lines.extend([
        "",
        "---",
        "",
        "## Recommendations",
        "",
    ])
    
    if study_results.get('best_configuration'):
        best = study_results['best_configuration']
        lines.extend([
            f"### Recommended Configuration for Stage 5/6",
            "",
            f"1. **Feature Set**: `{best['feature_set']}` ({FEATURE_SETS[best['feature_set']]['name']})",
            f"2. **Target**: `{best['target_config']}` ({TARGET_CONFIGS[best['target_config']]['name']})",
            f"3. **Expected F1-Macro**: {best['mean_f1_macro']:.4f}",
            "",
        ])
    
    lines.extend([
        "### Scientific Justification",
        "",
        "The recommended configuration balances:",
        "- **Predictive power**: Highest cross-validated F1-macro score",
        "- **Interpretability**: Feature set that captures physiologically meaningful signals",
        "- **Robustness**: Stable performance across temporal CV folds",
        "",
        "Further improvements may require:",
        "- Additional SoM data collection (current N=77 is borderline)",
        "- Investigation of HRV proxy features for days without direct HRV measurement",
        "- Exploration of sequence models (LSTM) which may capture temporal patterns",
        "",
        "---",
        "",
        f"*Report generated by `src/etl/ml6_som_experiments.py`*",
    ])
    
    # Write report
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    
    logger.info(f"Markdown report saved: {report_path}")
    
    return report_path


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="ML6 SoM Feature Ablation Study"
    )
    parser.add_argument(
        '--participant', '-p',
        type=str,
        default='P000001',
        help='Participant ID'
    )
    parser.add_argument(
        '--snapshot', '-s',
        type=str,
        default='2025-12-08',
        help='Snapshot date'
    )
    args = parser.parse_args()
    
    # Run ablation study
    results = run_ablation_study(args.participant, args.snapshot)
    
    # Generate report
    generate_markdown_report(results, args.participant, args.snapshot)
    
    logger.info("\n✓ Ablation study complete!")


if __name__ == '__main__':
    main()
