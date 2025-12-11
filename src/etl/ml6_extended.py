"""
ML6-Extended Pipeline Module

Implements multiple classical ML algorithms for SoM classification:
- LogisticRegression
- RandomForestClassifier
- GradientBoostingClassifier
- XGBoost (optional)
- SVM (Linear, RBF)
- GaussianNB
- KNN (k=3, 5, 7)

All models:
- Load selected feature set and target from model_selection.json
- Apply anti-leak column filtering
- Fit scaler ONLY on training data per fold
- Use consistent 6-fold temporal CV

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
    create_temporal_folds,
    fit_scaler_per_fold,
    compute_classification_metrics,
    aggregate_fold_metrics,
    update_model_leaderboard,
)

from src.etl.ml_metrics_extended import (
    compute_metrics_extended,
    compute_naive_baselines,
    aggregate_extended_metrics,
    export_per_class_metrics_csv,
    export_confusion_matrices_json,
    export_baseline_comparison_csv,
    setup_metrics_output_dir,
    format_metrics_table,
    format_per_class_table,
    format_baseline_comparison_table,
    format_confusion_matrix_md,
    ExtendedMetrics,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Model Definitions
# =============================================================================

def get_model_configs() -> Dict[str, Dict]:
    """
    Get all ML6-Extended model configurations.
    
    Returns:
        Dict mapping model names to config dicts with:
        - 'class': Model class (imported dynamically)
        - 'params': Model hyperparameters
        - 'needs_scaling': Whether features need standardization
        - 'optional': Whether to skip if import fails
    """
    return {
        'LogisticRegression': {
            'module': 'sklearn.linear_model',
            'class_name': 'LogisticRegression',
            'params': {
                'class_weight': 'balanced',
                'max_iter': 1000,
                'random_state': 42,
                'solver': 'lbfgs',
            },
            'needs_scaling': True,
            'optional': False,
        },
        'RandomForest': {
            'module': 'sklearn.ensemble',
            'class_name': 'RandomForestClassifier',
            'params': {
                'n_estimators': 100,
                'max_depth': 10,
                'class_weight': 'balanced',
                'random_state': 42,
                'n_jobs': -1,
            },
            'needs_scaling': False,
            'optional': False,
        },
        'GradientBoosting': {
            'module': 'sklearn.ensemble',
            'class_name': 'GradientBoostingClassifier',
            'params': {
                'n_estimators': 100,
                'max_depth': 5,
                'learning_rate': 0.1,
                'random_state': 42,
            },
            'needs_scaling': False,
            'optional': False,
        },
        'XGBoost': {
            'module': 'xgboost',
            'class_name': 'XGBClassifier',
            'params': {
                'n_estimators': 100,
                'max_depth': 5,
                'learning_rate': 0.1,
                'random_state': 42,
                'use_label_encoder': False,
                'eval_metric': 'mlogloss',
            },
            'needs_scaling': False,
            'optional': True,  # Skip if xgboost not installed
        },
        'SVM_Linear': {
            'module': 'sklearn.svm',
            'class_name': 'SVC',
            'params': {
                'kernel': 'linear',
                'class_weight': 'balanced',
                'random_state': 42,
                'probability': True,
            },
            'needs_scaling': True,
            'optional': False,
        },
        'SVM_RBF': {
            'module': 'sklearn.svm',
            'class_name': 'SVC',
            'params': {
                'kernel': 'rbf',
                'class_weight': 'balanced',
                'random_state': 42,
                'probability': True,
            },
            'needs_scaling': True,
            'optional': False,
        },
        'GaussianNB': {
            'module': 'sklearn.naive_bayes',
            'class_name': 'GaussianNB',
            'params': {},
            'needs_scaling': False,  # NB is scale-invariant
            'optional': False,
        },
        'KNN_3': {
            'module': 'sklearn.neighbors',
            'class_name': 'KNeighborsClassifier',
            'params': {
                'n_neighbors': 3,
                'weights': 'distance',
            },
            'needs_scaling': True,
            'optional': False,
        },
        'KNN_5': {
            'module': 'sklearn.neighbors',
            'class_name': 'KNeighborsClassifier',
            'params': {
                'n_neighbors': 5,
                'weights': 'distance',
            },
            'needs_scaling': True,
            'optional': False,
        },
        'KNN_7': {
            'module': 'sklearn.neighbors',
            'class_name': 'KNeighborsClassifier',
            'params': {
                'n_neighbors': 7,
                'weights': 'distance',
            },
            'needs_scaling': True,
            'optional': False,
        },
    }


def load_model_class(config: Dict) -> Optional[type]:
    """
    Dynamically import and return a model class.
    
    Args:
        config: Model config dict with 'module' and 'class_name'
    
    Returns:
        Model class or None if import fails
    """
    try:
        import importlib
        module = importlib.import_module(config['module'])
        return getattr(module, config['class_name'])
    except (ImportError, AttributeError) as e:
        if config.get('optional'):
            logger.warning(f"[ML6-Ext] Optional model import failed: {config['class_name']} - {e}")
            return None
        else:
            raise


# =============================================================================
# Training Functions
# =============================================================================

def train_model_cv(
    model_name: str,
    model_config: Dict,
    X: np.ndarray,
    y: np.ndarray,
    folds: List[Tuple[List[int], List[int]]],
    class_labels: Optional[List[int]] = None
) -> Dict[str, Any]:
    """
    Train a single model using cross-validation with extended metrics.
    
    Args:
        model_name: Name of the model
        model_config: Model configuration dict
        X: Feature matrix (n_samples, n_features)
        y: Target vector (n_samples,)
        folds: List of (train_idx, val_idx) tuples
        class_labels: Optional list of class labels for consistent metrics
    
    Returns:
        Dict with model results including per-fold and aggregate metrics,
        confusion matrices, and per-class breakdowns
    """
    # Load model class
    model_class = load_model_class(model_config)
    if model_class is None:
        return {
            'model_name': model_name,
            'status': 'skipped',
            'reason': 'Import failed (optional dependency)',
        }
    
    from sklearn.preprocessing import StandardScaler
    
    # Determine class labels if not provided
    if class_labels is None:
        class_labels = sorted(set(y.tolist()))
    
    fold_metrics = []
    fold_extended_metrics = []
    fold_predictions = []
    fold_confusion_matrices = []
    
    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        try:
            # Scale if needed - FIT ONLY ON TRAINING DATA
            if model_config.get('needs_scaling', False):
                X_train, X_val = fit_scaler_per_fold(X, train_idx, val_idx, StandardScaler)
            else:
                X_train, X_val = X[train_idx], X[val_idx]
            
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Check class distribution in training set
            unique_train = np.unique(y_train)
            if len(unique_train) < 2:
                logger.warning(f"[ML6-Ext] {model_name} fold {fold_idx}: Only {len(unique_train)} class in train")
                continue
            
            # Create and train model
            model = model_class(**model_config['params'])
            
            # Handle XGBoost multi-class
            if model_name == 'XGBoost':
                n_classes = len(np.unique(y))
                if n_classes > 2:
                    model.set_params(objective='multi:softmax', num_class=n_classes)
            
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            
            # Compute basic metrics (for backward compatibility)
            metrics = compute_classification_metrics(y_val, y_pred)
            metrics['fold_idx'] = fold_idx
            fold_metrics.append(metrics)
            
            # Compute extended metrics with per-class breakdown
            ext_metrics = compute_metrics_extended(y_val, y_pred, class_labels)
            fold_extended_metrics.append(ext_metrics)
            
            # Store confusion matrix
            fold_confusion_matrices.append(ext_metrics.confusion_matrix)
            
            fold_predictions.append({
                'fold_idx': fold_idx,
                'y_true': y_val.tolist(),
                'y_pred': y_pred.tolist(),
            })
            
        except Exception as e:
            logger.warning(f"[ML6-Ext] {model_name} fold {fold_idx} failed: {e}")
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
        'predictions': fold_predictions,
    }


# =============================================================================
# SHAP Analysis (for tree-based models)
# =============================================================================

def compute_shap_summary(
    model_name: str,
    model_config: Dict,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    output_dir: Path
) -> Optional[str]:
    """
    Compute SHAP feature importance for tree-based models.
    
    Args:
        model_name: Name of the model
        model_config: Model configuration
        X: Feature matrix
        y: Target vector
        feature_names: List of feature names
        output_dir: Directory to save SHAP summary
    
    Returns:
        Path to SHAP summary file or None if SHAP not available
    """
    # Only for tree-based models
    tree_models = ['RandomForest', 'GradientBoosting', 'XGBoost']
    if model_name not in tree_models:
        return None
    
    try:
        import shap
    except ImportError:
        logger.warning("[ML6-Ext] SHAP not installed, skipping SHAP analysis")
        return None
    
    try:
        # Train model on full data for SHAP
        model_class = load_model_class(model_config)
        if model_class is None:
            return None
        
        model = model_class(**model_config['params'])
        model.fit(X, y)
        
        # Compute SHAP values
        if model_name == 'XGBoost':
            explainer = shap.TreeExplainer(model)
        else:
            explainer = shap.TreeExplainer(model)
        
        shap_values = explainer.shap_values(X)
        
        # Handle multi-class (take absolute mean across classes)
        if isinstance(shap_values, list):
            shap_importance = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)
        else:
            shap_importance = np.abs(shap_values).mean(axis=0)
        
        # Create summary
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'shap_importance': shap_importance,
        }).sort_values('shap_importance', ascending=False)
        
        # Save markdown summary
        shap_path = output_dir / f'shap_summary_{model_name}.md'
        with open(shap_path, 'w') as f:
            f.write(f"# SHAP Feature Importance: {model_name}\n\n")
            f.write(f"**Generated**: {datetime.now().isoformat()}\n\n")
            f.write("## Top Features\n\n")
            f.write("| Rank | Feature | SHAP Importance |\n")
            f.write("|------|---------|----------------|\n")
            for rank, row in enumerate(importance_df.head(10).itertuples(), 1):
                f.write(f"| {rank} | {row.feature} | {row.shap_importance:.4f} |\n")
        
        logger.info(f"[ML6-Ext] SHAP summary saved: {shap_path}")
        return str(shap_path)
        
    except Exception as e:
        logger.warning(f"[ML6-Ext] SHAP analysis failed for {model_name}: {e}")
        return None


# =============================================================================
# Main Pipeline
# =============================================================================

def run_ml6_extended(
    participant: str,
    snapshot: str,
    output_base: Optional[Path] = None,
    skip_shap: bool = False
) -> Dict[str, Any]:
    """
    Run the ML6-Extended pipeline.
    
    Args:
        participant: Participant ID (e.g., 'P000001')
        snapshot: Snapshot date (e.g., '2025-12-08')
        output_base: Base output directory (default: ai/local)
        skip_shap: If True, skip SHAP analysis
    
    Returns:
        Dict with all model results and best model selection
    """
    logger.info("=" * 70)
    logger.info("[ML6-Extended] Starting ML6-Extended Pipeline")
    logger.info("=" * 70)
    
    # Setup paths
    if output_base is None:
        output_base = Path('data/ai')
    
    output_dir = output_base / participant / snapshot
    ml6_ext_dir = output_dir / 'ml6_extended'
    ml6_ext_dir.mkdir(parents=True, exist_ok=True)
    
    results = {
        'participant': participant,
        'snapshot': snapshot,
        'pipeline': 'ml6_extended',
        'started': datetime.now().isoformat(),
        'models': {},
    }
    
    try:
        # =====================================================================
        # Load model selection
        # =====================================================================
        model_selection = load_model_selection(output_dir)
        
        selected_fs = model_selection['ml6']['selected_fs']
        selected_target = model_selection['ml6']['selected_target']
        feature_cols = model_selection['ml6']['features']
        target_col = get_target_column(selected_target)
        
        results['selected_fs'] = selected_fs
        results['selected_target'] = selected_target
        results['target_col'] = target_col
        results['n_features'] = len(feature_cols)
        
        logger.info(f"[ML6-Ext] Using: {selected_fs} × {selected_target}")
        logger.info(f"[ML6-Ext] Features: {len(feature_cols)}")
        
        # =====================================================================
        # Load and prepare data
        # =====================================================================
        df = load_feature_universe(output_dir)
        
        # Apply anti-leak filter
        safe_features = apply_anti_leak_filter(df, feature_cols)
        validate_no_leakage(np.array([]), safe_features)
        
        results['safe_features'] = safe_features
        
        # Extract X, y
        X = df[safe_features].values
        y = df[target_col].values
        
        # Remove NaN targets
        valid_mask = ~np.isnan(y)
        X = X[valid_mask]
        y = y[valid_mask]
        
        n_samples = len(y)
        results['n_samples'] = n_samples
        
        logger.info(f"[ML6-Ext] Samples: {n_samples}")
        logger.info(f"[ML6-Ext] Target distribution: {dict(zip(*np.unique(y, return_counts=True)))}")
        
        # =====================================================================
        # Create temporal folds
        # =====================================================================
        folds = create_temporal_folds(n_samples, n_folds=6)
        results['n_folds'] = len(folds)
        
        # Determine class labels
        class_labels = sorted(set(y.tolist()))
        results['class_labels'] = class_labels
        
        # =====================================================================
        # Compute Naïve Baselines
        # =====================================================================
        logger.info("[ML6-Ext] Computing naïve baselines...")
        
        baseline_fold_metrics = {
            'majority_class': [],
            'stratified_random': [],
            'persistence': [],
        }
        
        for fold_idx, (train_idx, val_idx) in enumerate(folds):
            y_train = y[train_idx]
            y_val = y[val_idx]
            
            # Compute baselines for this fold
            bl_metrics = compute_naive_baselines(
                y_train=y_train,
                y_val=y_val,
                y_full=y,
                val_idx=val_idx,
                class_labels=class_labels,
                random_seed=42 + fold_idx  # Vary seed per fold for stratified-random
            )
            
            for bl_name, bl_ext_metrics in bl_metrics.items():
                baseline_fold_metrics[bl_name].append(bl_ext_metrics)
        
        # Aggregate baseline metrics across folds
        baseline_results = {}
        for bl_name, fold_ext_list in baseline_fold_metrics.items():
            agg_bl = aggregate_extended_metrics(fold_ext_list)
            baseline_results[bl_name] = agg_bl
            logger.info(f"  → {bl_name}: F1={agg_bl.get('mean_f1_macro', 0):.4f}")
        
        results['baselines'] = baseline_results
        
        # =====================================================================
        # Train all models
        # =====================================================================
        model_configs = get_model_configs()
        model_results = {}
        
        for model_name, model_config in model_configs.items():
            logger.info(f"[ML6-Ext] Training {model_name}...")
            
            try:
                result = train_model_cv(model_name, model_config, X, y, folds, class_labels)
                model_results[model_name] = result
                
                if result['status'] == 'success':
                    f1 = result['aggregate_metrics'].get('mean_f1_macro', 0)
                    logger.info(f"  → {model_name}: F1={f1:.4f}")
                else:
                    logger.info(f"  → {model_name}: {result['status']}")
                    
            except Exception as e:
                logger.error(f"[ML6-Ext] {model_name} failed: {e}")
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
            
            logger.info(f"[ML6-Ext] Best model: {best_name} (F1={best_f1:.4f})")
            
            # Update global leaderboard
            update_model_leaderboard(
                output_dir, 'ml6_extended', best_name, best_f1,
                {'n_models_tested': len(model_results)}
            )
        else:
            results['best_model'] = None
            logger.warning("[ML6-Ext] No successful models")
        
        # =====================================================================
        # SHAP Analysis (tree-based models)
        # =====================================================================
        if not skip_shap:
            results['shap_summaries'] = {}
            for model_name in ['RandomForest', 'GradientBoosting', 'XGBoost']:
                if model_name in model_results and model_results[model_name].get('status') == 'success':
                    shap_path = compute_shap_summary(
                        model_name, model_configs[model_name],
                        X, y, safe_features, ml6_ext_dir
                    )
                    if shap_path:
                        results['shap_summaries'][model_name] = shap_path
        
        # =====================================================================
        # Export Extended Metrics (PhD-level)
        # =====================================================================
        logger.info("[ML6-Ext] Exporting extended metrics...")
        
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
                    output_path=metrics_dir / 'per_class' / f'per_class_{best_name}_{target_name}.csv',
                    model_name=best_name,
                    target_name=target_name
                )
                results['metrics_exports']['per_class_csv'] = str(pc_path)
            
            # Confusion matrices JSON
            if 'confusion_matrices' in best_res:
                cm_path = export_confusion_matrices_json(
                    fold_matrices=best_res['confusion_matrices'],
                    class_labels=class_labels,
                    output_path=metrics_dir / 'confusion_matrices' / f'cm_{best_name}_{target_name}.json',
                    model_name=best_name,
                    target_name=target_name
                )
                results['metrics_exports']['confusion_matrices_json'] = str(cm_path)
            
            # Baseline comparison CSV
            if 'baselines' in results and 'extended_metrics' in best_res:
                bl_path = export_baseline_comparison_csv(
                    model_metrics=best_res['extended_metrics'],
                    baseline_metrics=results['baselines'],
                    output_path=metrics_dir / 'baseline_comparisons' / f'baseline_comparison_{target_name}.csv',
                    model_name=best_name,
                    target_name=target_name
                )
                results['metrics_exports']['baseline_comparison_csv'] = str(bl_path)
        
        results['status'] = 'success'
        
    except Exception as e:
        logger.error(f"[ML6-Extended] Pipeline failed: {e}")
        results['status'] = 'error'
        results['error'] = str(e)
    
    results['completed'] = datetime.now().isoformat()
    
    # =========================================================================
    # Save results
    # =========================================================================
    
    # Full results summary
    results_path = ml6_ext_dir / 'results_summary.json'
    with open(results_path, 'w') as f:
        # Remove predictions for smaller file
        results_for_save = results.copy()
        if 'models' in results_for_save:
            for model_name, model_res in results_for_save['models'].items():
                if isinstance(model_res, dict) and 'predictions' in model_res:
                    del model_res['predictions']
        json.dump(results_for_save, f, indent=2, default=str)
    
    logger.info(f"[ML6-Ext] Results saved: {results_path}")
    
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
        
        per_model_path = ml6_ext_dir / 'per_model_metrics.json'
        with open(per_model_path, 'w') as f:
            json.dump(per_model, f, indent=2)
        
        logger.info(f"[ML6-Ext] Per-model metrics: {per_model_path}")
    
    # Best model summary
    if results.get('best_model'):
        best_summary_path = ml6_ext_dir / 'best_model_summary.md'
        best_name = results['best_model']['name']
        best_res = results['models'].get(best_name, {})
        
        with open(best_summary_path, 'w', encoding='utf-8') as f:
            f.write("# ML6-Extended Best Model Summary\n\n")
            f.write(f"**Generated**: {results['completed']}\n\n")
            f.write(f"## Configuration\n\n")
            f.write(f"- **Feature Set**: {results['selected_fs']}\n")
            f.write(f"- **Target**: {results['target_col']}\n")
            f.write(f"- **Samples**: {results['n_samples']}\n")
            f.write(f"- **Features**: {results['n_features']}\n\n")
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
                    'persistence': 'Persistence',
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
                class_labels = best_res['extended_metrics'].get('class_labels', results.get('class_labels', []))
                cm = best_res['extended_metrics']['confusion_matrix_sum']
                
                header = "| True \\ Pred | " + " | ".join(str(l) for l in class_labels) + " |"
                f.write(header + "\n")
                f.write("|" + "---|" * (len(class_labels) + 1) + "\n")
                
                for i, row in enumerate(cm):
                    row_str = f"| **{class_labels[i]}** | " + " | ".join(f"{v:.0f}" for v in row) + " |"
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
        
        logger.info(f"[ML6-Ext] Best model summary: {best_summary_path}")
    
    logger.info("[ML6-Extended] Pipeline complete")
    logger.info("=" * 70)
    
    return results


# =============================================================================
# CLI Interface
# =============================================================================

def main():
    """Run ML6-Extended from command line."""
    import argparse
    
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(levelname)s: %(message)s'
    )
    
    parser = argparse.ArgumentParser(description='Run ML6-Extended Pipeline')
    parser.add_argument('--participant', '-p', default='P000001')
    parser.add_argument('--snapshot', '-s', default='2025-12-08')
    parser.add_argument('--skip-shap', action='store_true', help='Skip SHAP analysis')
    args = parser.parse_args()
    
    results = run_ml6_extended(
        args.participant,
        args.snapshot,
        skip_shap=args.skip_shap
    )
    
    if results.get('best_model'):
        print(f"\nBest Model: {results['best_model']['name']}")
        print(f"F1-macro: {results['best_model']['f1_macro']:.4f}")
    
    return 0 if results.get('status') == 'success' else 1


if __name__ == '__main__':
    exit(main())
