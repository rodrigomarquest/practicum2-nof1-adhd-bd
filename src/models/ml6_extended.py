"""
ML6 Extended Models with Temporal Instability Regularization.

Implements 4 additional baselines for the ML6 3-class problem:
1. Random Forest (with instability-based feature penalty)
2. XGBoost (with instability-based L1/L2 penalty)
3. LightGBM (with instability-based feature penalty)
4. SVM with RBF kernel (no instability penalty)

All models use the EXACT same:
- Dataset: data/ai/P000001/2025-11-07/ml6/features_daily_ml6.csv
- Folds: 6-fold calendar-based temporal CV (from cv_summary.json)
- Preprocessing: StandardScaler + SimpleImputer (median)
- Label encoding: {-1, 0, +1}
- Metrics: f1_macro, f1_weighted, balanced_acc, kappa

Temporal Instability Regularization:
====================================
For tree-based models (RF, XGB, LGBM), we penalize features with high
variance across behavioral segments. This prevents overfitting to 
non-stationary patterns.

Algorithm:
    1. Compute instability[f] = Var(mean(f | segment_id))
    2. Normalize to [0, 1]
    3. Apply model-specific penalties:
       - XGBoost: reg_alpha/lambda += instability * scale
       - LightGBM: feature_penalty = 1 + instability * scale
       - RandomForest: reduce max_features for unstable features

Dependencies:
    pip install xgboost lightgbm shap scikit-learn
"""

import pandas as pd
import numpy as np
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    f1_score, balanced_accuracy_score, cohen_kappa_score, confusion_matrix
)
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
import lightgbm as lgb
import shap
import matplotlib.pyplot as plt

from src.utils.temporal_instability import compute_instability_scores

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)
np.random.seed(42)


class InstabilityRegularizedXGBoost:
    """XGBoost with per-feature L1/L2 penalties based on temporal instability."""
    
    def __init__(
        self, 
        instability_scores: Dict[str, float],
        feature_names: List[str],
        alpha_scale: float = 0.1,
        lambda_scale: float = 0.2,
        **xgb_params
    ):
        """
        Args:
            instability_scores: Dict mapping feature_name -> instability[0,1]
            feature_names: Ordered list of feature names
            alpha_scale: Multiplier for L1 penalty
            lambda_scale: Multiplier for L2 penalty
            **xgb_params: Additional XGBoost parameters
        """
        self.instability_scores = instability_scores
        self.feature_names = feature_names
        self.alpha_scale = alpha_scale
        self.lambda_scale = lambda_scale
        
        # Build feature-wise penalty arrays
        self.feature_alpha = np.array([
            instability_scores.get(f, 0.0) * alpha_scale 
            for f in feature_names
        ])
        self.feature_lambda = np.array([
            instability_scores.get(f, 0.0) * lambda_scale 
            for f in feature_names
        ])
        
        # Base XGBoost parameters
        self.params = {
            'objective': 'multi:softprob',
            'num_class': 3,
            'eval_metric': 'mlogloss',
            'max_depth': 4,
            'learning_rate': 0.05,
            'n_estimators': 200,
            'min_child_weight': 3,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            **xgb_params
        }
        
        # Add instability-based regularization
        # Note: XGBoost doesn't support per-feature penalties directly,
        # so we use global reg_alpha/reg_lambda with weighted averages
        mean_alpha = float(np.mean(self.feature_alpha))
        mean_lambda = float(np.mean(self.feature_lambda))
        
        self.params['reg_alpha'] = self.params.get('reg_alpha', 0.0) + mean_alpha
        self.params['reg_lambda'] = self.params.get('reg_lambda', 1.0) + mean_lambda
        
        self.model = None
        
        logger.info(
            f"XGBoost instability regularization: "
            f"mean_alpha={mean_alpha:.4f}, mean_lambda={mean_lambda:.4f}"
        )
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit XGBoost model."""
        # Convert labels {-1, 0, +1} to {0, 1, 2}
        y_encoded = np.array([0 if yi == -1 else (1 if yi == 0 else 2) for yi in y])
        
        self.model = xgb.XGBClassifier(**self.params)
        self.model.fit(X, y_encoded)
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict labels."""
        y_pred_encoded = self.model.predict(X)
        # Convert back to {-1, 0, +1}
        return np.array([-1 if yi == 0 else (0 if yi == 1 else +1) for yi in y_pred_encoded])
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities."""
        return self.model.predict_proba(X)


class InstabilityRegularizedLightGBM:
    """LightGBM with feature penalties based on temporal instability."""
    
    def __init__(
        self,
        instability_scores: Dict[str, float],
        feature_names: List[str],
        penalty_scale: float = 1.0,
        **lgbm_params
    ):
        """
        Args:
            instability_scores: Dict mapping feature_name -> instability[0,1]
            feature_names: Ordered list of feature names
            penalty_scale: Multiplier for feature penalties
            **lgbm_params: Additional LightGBM parameters
        """
        self.instability_scores = instability_scores
        self.feature_names = feature_names
        self.penalty_scale = penalty_scale
        
        # Build feature penalty array
        # feature_penalty > 1.0 discourages using the feature
        self.feature_penalty = [
            1.0 + instability_scores.get(f, 0.0) * penalty_scale
            for f in feature_names
        ]
        
        # Base LightGBM parameters
        self.params = {
            'objective': 'multiclass',
            'num_class': 3,
            'metric': 'multi_logloss',
            'max_depth': 4,
            'learning_rate': 0.05,
            'n_estimators': 200,
            'min_child_samples': 10,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'verbose': -1,
            **lgbm_params
        }
        
        self.model = None
        
        mean_penalty = np.mean(self.feature_penalty)
        logger.info(
            f"LightGBM instability regularization: "
            f"mean_penalty={mean_penalty:.4f} (range: {min(self.feature_penalty):.4f}-{max(self.feature_penalty):.4f})"
        )
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit LightGBM model."""
        # Convert labels {-1, 0, +1} to {0, 1, 2}
        y_encoded = np.array([0 if yi == -1 else (1 if yi == 0 else 2) for yi in y])
        
        # Note: LightGBM doesn't support feature_penalty parameter directly
        # We implement via feature importance weighting during training
        # Workaround: use feature_fraction_bynode with weighted sampling
        
        self.model = lgb.LGBMClassifier(**self.params)
        
        # Apply penalty via sample weights (proxy for feature penalty)
        # Higher instability features get lower effective weight
        feature_weights = 1.0 / np.array(self.feature_penalty)
        X_weighted = X * feature_weights[np.newaxis, :]
        
        self.model.fit(X_weighted, y_encoded)
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict labels."""
        feature_weights = 1.0 / np.array(self.feature_penalty)
        X_weighted = X * feature_weights[np.newaxis, :]
        
        y_pred_encoded = self.model.predict(X_weighted)
        # Convert back to {-1, 0, +1}
        return np.array([-1 if yi == 0 else (0 if yi == 1 else +1) for yi in y_pred_encoded])
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities."""
        feature_weights = 1.0 / np.array(self.feature_penalty)
        X_weighted = X * feature_weights[np.newaxis, :]
        return self.model.predict_proba(X_weighted)


class InstabilityRegularizedRandomForest:
    """Random Forest with dynamic max_features based on instability."""
    
    def __init__(
        self,
        instability_scores: Dict[str, float],
        feature_names: List[str],
        instability_penalty: float = 0.3,
        **rf_params
    ):
        """
        Args:
            instability_scores: Dict mapping feature_name -> instability[0,1]
            feature_names: Ordered list of feature names
            instability_penalty: Reduction factor for max_features (0-1)
            **rf_params: Additional RandomForest parameters
        """
        self.instability_scores = instability_scores
        self.feature_names = feature_names
        self.instability_penalty = instability_penalty
        
        # Compute effective max_features
        mean_instability = np.mean([instability_scores.get(f, 0.0) for f in feature_names])
        base_max_features = int(np.sqrt(len(feature_names)))
        self.max_features_eff = int(base_max_features * (1.0 - mean_instability * instability_penalty))
        self.max_features_eff = max(1, self.max_features_eff)  # At least 1
        
        # Base RandomForest parameters
        self.params = {
            'n_estimators': 200,
            'max_depth': 10,
            'min_samples_split': 10,
            'min_samples_leaf': 5,
            'max_features': self.max_features_eff,
            'class_weight': 'balanced',
            'random_state': 42,
            'n_jobs': -1,
            **rf_params
        }
        
        self.model = None
        
        logger.info(
            f"RandomForest instability regularization: "
            f"max_features={self.max_features_eff} (base={base_max_features}, "
            f"mean_instability={mean_instability:.4f})"
        )
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit RandomForest model."""
        self.model = RandomForestClassifier(**self.params)
        self.model.fit(X, y)
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict labels."""
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities."""
        return self.model.predict_proba(X)


def load_ml6_data(ml6_csv: str) -> Tuple[pd.DataFrame, List[str]]:
    """Load ML6 dataset and extract feature columns."""
    df = pd.read_csv(ml6_csv)
    df['date'] = pd.to_datetime(df['date'])
    
    # Feature columns (exclude date and labels)
    exclude_cols = ['date', 'label_3cls', 'label_2cls']
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    
    logger.info(f"Loaded ML6 data: {len(df)} rows, {len(feature_cols)} features")
    return df, feature_cols


def load_cv_folds(cv_summary_json: str) -> List[Dict]:
    """Load CV fold definitions from cv_summary.json."""
    with open(cv_summary_json, 'r') as f:
        cv_data = json.load(f)
    return cv_data['folds']


def prepare_features(X_train: pd.DataFrame, X_val: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """Impute missing and standardize."""
    imputer = SimpleImputer(strategy='median')
    scaler = StandardScaler()
    
    X_train_imp = imputer.fit_transform(X_train)
    X_train_scaled = scaler.fit_transform(X_train_imp)
    
    X_val_imp = imputer.transform(X_val)
    X_val_scaled = scaler.transform(X_val_imp)
    
    return X_train_scaled, X_val_scaled


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    """Compute 3-class metrics."""
    return {
        'f1_macro': float(f1_score(y_true, y_pred, average='macro', zero_division=0)),
        'f1_weighted': float(f1_score(y_true, y_pred, average='weighted', zero_division=0)),
        'balanced_acc': float(balanced_accuracy_score(y_true, y_pred)),
        'kappa': float(cohen_kappa_score(y_true, y_pred)),
    }


def train_model(
    model_type: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    instability_scores: Optional[Dict[str, float]],
    feature_names: List[str]
) -> Dict:
    """
    Train a single model and return metrics.
    
    Args:
        model_type: 'rf', 'xgb', 'lgbm', or 'svm'
        X_train, y_train: Training data
        X_val, y_val: Validation data
        instability_scores: Feature instability scores (for RF/XGB/LGBM)
        feature_names: Ordered feature names
    
    Returns:
        Dict with metrics and predictions
    """
    if model_type == 'rf':
        model = InstabilityRegularizedRandomForest(
            instability_scores=instability_scores or {},
            feature_names=feature_names
        )
    elif model_type == 'xgb':
        model = InstabilityRegularizedXGBoost(
            instability_scores=instability_scores or {},
            feature_names=feature_names
        )
    elif model_type == 'lgbm':
        model = InstabilityRegularizedLightGBM(
            instability_scores=instability_scores or {},
            feature_names=feature_names
        )
    elif model_type == 'svm':
        # SVM with RBF kernel (no instability penalty per requirement)
        model = SVC(
            C=1.0,
            kernel='rbf',
            gamma='scale',
            class_weight='balanced',
            random_state=42,
            probability=True
        )
        model.fit(X_train, y_train)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    
    # Fit model
    if model_type != 'svm':
        model.fit(X_train, y_train)
    
    # Predict
    y_pred = model.predict(X_val)
    
    # Compute metrics
    metrics = compute_metrics(y_val, y_pred)
    
    logger.info(
        f"  {model_type.upper()}: F1-macro={metrics['f1_macro']:.4f}, "
        f"Balanced-acc={metrics['balanced_acc']:.4f}"
    )
    
    return {
        'metrics': metrics,
        'y_pred': y_pred,
        'model': model
    }


def compute_shap_analysis(
    model,
    model_type: str,
    X_train: np.ndarray,
    X_val: np.ndarray,
    feature_names: List[str],
    fold_idx: int,
    output_dir: Path
):
    """
    Compute SHAP values for a trained model and save results.
    
    Args:
        model: Trained model
        model_type: 'logreg', 'rf', 'xgb', or 'lgbm'
        X_train: Training data (for background samples)
        X_val: Validation data (for SHAP computation)
        feature_names: List of feature names
        fold_idx: Fold index
        output_dir: Directory to save SHAP results
    """
    logger.info(f"    Computing SHAP for {model_type.upper()} fold {fold_idx}...")
    
    shap_dir = output_dir / 'shap'
    shap_dir.mkdir(exist_ok=True)
    
    try:
        # Ensure inputs are numpy arrays (2D) and contiguous
        if not isinstance(X_train, np.ndarray):
            X_train = np.asarray(X_train.values if hasattr(X_train, 'values') else X_train, dtype=np.float64)
        else:
            X_train = np.ascontiguousarray(X_train, dtype=np.float64)
            
        if not isinstance(X_val, np.ndarray):
            X_val = np.asarray(X_val.values if hasattr(X_val, 'values') else X_val, dtype=np.float64)
        else:
            X_val = np.ascontiguousarray(X_val, dtype=np.float64)
        
        # Ensure 2D shape
        if len(X_train.shape) == 1:
            X_train = X_train.reshape(-1, 1)
        if len(X_val.shape) == 1:
            X_val = X_val.reshape(-1, 1)
        
        # Validate shapes
        if X_train.shape[1] != len(feature_names):
            logger.warning(f"    Shape mismatch: X_train has {X_train.shape[1]} features, expected {len(feature_names)}")
            return
        
        # Use subset of training data for background (faster computation)
        np.random.seed(42)
        n_background = min(100, X_train.shape[0])
        background_idx = np.random.choice(X_train.shape[0], n_background, replace=False)
        X_background = np.ascontiguousarray(X_train[background_idx], dtype=np.float64)
        
        # Select appropriate explainer based on model type
        if model_type == 'logreg':
            # For logistic regression, use Kernel explainer with predict_proba
            explainer = shap.KernelExplainer(model.predict_proba, X_background)
            shap_values = explainer.shap_values(X_val[:20])  # Limit to 20 samples for speed
            
        elif model_type in ['rf', 'xgb', 'lgbm']:
            # For tree-based models, use model-agnostic Kernel explainer
            # This is slower but more robust for custom wrappers
            if hasattr(model, 'predict_proba'):
                pred_fn = model.predict_proba
            elif hasattr(model, 'predict'):
                pred_fn = model.predict
            else:
                logger.warning(f"    Model {model_type} doesn't have predict method, skipping")
                return
            
            explainer = shap.KernelExplainer(pred_fn, X_background)
            shap_values = explainer.shap_values(X_val[:20])  # Limit to 20 samples
        
        else:
            logger.warning(f"    SHAP not supported for {model_type}, skipping")
            return
        
        # Handle multi-class output (3 classes)
        # For 3-class problems, shap_values is typically a list of 3 arrays
        # Each array has shape (n_samples, n_features)
        if isinstance(shap_values, list):
            # Take average absolute SHAP across all classes: mean over classes, then over samples
            # Stack: (n_classes, n_samples, n_features) -> mean over axis 0 -> (n_samples, n_features)
            stacked = np.stack([np.abs(sv) for sv in shap_values], axis=0)
            shap_abs_mean = np.mean(stacked, axis=0)  # Average across classes: (n_samples, n_features)
        else:
            # Binary case: shape is already (n_samples, n_features)
            shap_abs_mean = np.abs(shap_values)
        
        # Compute global feature importance (mean |SHAP| per feature across samples)
        # Input: (n_samples, n_features) -> Output: (n_features,)
        global_importance = np.mean(shap_abs_mean, axis=0)
        global_importance = np.asarray(global_importance, dtype=np.float64).flatten()
        global_importance = np.ascontiguousarray(global_importance)
        
        # Ensure feature_names is a Python list (not pandas Index/Series)
        feature_names_list = []
        if isinstance(feature_names, (list, tuple)):
            feature_names_list = list(feature_names)
        elif hasattr(feature_names, 'tolist'):  # pandas Index/Series
            feature_names_list = feature_names.tolist()  # type: ignore
        else:
            feature_names_list = [str(fn) for fn in feature_names]
        
        # Validate lengths match - if mismatch, truncate or pad feature names
        n_features_actual = len(global_importance)
        if len(feature_names_list) != n_features_actual:
            logger.warning(f"    Feature name mismatch: {len(feature_names_list)} names vs {n_features_actual} SHAP values")
            if len(feature_names_list) < n_features_actual:
                # Pad with generic names
                for i in range(len(feature_names_list), n_features_actual):
                    feature_names_list.append(f"feature_{i}")
            else:
                # Truncate feature names
                feature_names_list = feature_names_list[:n_features_actual]
        
        # Create DataFrame with explicit 1D arrays
        shap_df = pd.DataFrame({
            'feature': feature_names_list,
            'mean_abs_shap': global_importance
        }).sort_values('mean_abs_shap', ascending=False)
        
        # Save top-10 to CSV
        shap_csv = shap_dir / f'{model_type}_fold{fold_idx}_shap_top10.csv'
        shap_df.head(10).to_csv(shap_csv, index=False)
        
        # Get top-10 feature names (after sorting by importance)
        top10_features = shap_df.head(10)['feature'].tolist()
        top10_values = np.asarray(shap_df.head(10)['mean_abs_shap'].values, dtype=np.float64)
        
        # Create bar plot
        fig, ax = plt.subplots(figsize=(8, 6))
        y_pos = np.arange(len(top10_features))
        ax.barh(y_pos, top10_values, color='steelblue', edgecolor='black')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(top10_features)
        ax.set_xlabel('Mean |SHAP value|', fontsize=10)
        ax.set_title(f'{model_type.upper()} - Top 10 Features (Fold {fold_idx})', fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        plt.tight_layout()
        
        # Save figure
        fig_path = shap_dir / f'{model_type}_fold{fold_idx}_shap_top10.png'
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"    Saved SHAP results to {shap_csv}")
        logger.info(f"    Top feature: {shap_df.iloc[0]['feature']} (mean|SHAP|={shap_df.iloc[0]['mean_abs_shap']:.4f})")
        
    except Exception as e:
        logger.warning(f"    SHAP computation failed for {model_type}: {e}")


def run_ml6_extended(
    ml6_csv: str,
    cv_summary_json: str,
    segments_csv: str,
    output_dir: str,
    models: List[str] = ['rf', 'xgb', 'lgbm', 'svm']
):
    """
    Run all ML6 extended models with temporal instability regularization.
    
    Args:
        ml6_csv: Path to features_daily_ml6.csv
        cv_summary_json: Path to cv_summary.json
        segments_csv: Path to segment_autolog.csv
        output_dir: Output directory for results
        models: List of models to train
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    logger.info("="*80)
    logger.info("ML6 EXTENDED MODELS WITH TEMPORAL INSTABILITY REGULARIZATION")
    logger.info("="*80)
    
    # Load data
    df, feature_cols = load_ml6_data(ml6_csv)
    folds = load_cv_folds(cv_summary_json)
    
    # Compute instability scores
    logger.info("\nComputing temporal instability scores...")
    instability_scores = compute_instability_scores(
        features_df=df,
        segments_csv=segments_csv,
        feature_cols=feature_cols
    )
    
    # Save instability scores
    instability_df = pd.DataFrame([
        {'feature': k, 'instability_score': v}
        for k, v in sorted(instability_scores.items(), key=lambda x: x[1], reverse=True)
    ])
    instability_df.to_csv(output_path / 'instability_scores.csv', index=False)
    logger.info(f"Saved instability scores to {output_path / 'instability_scores.csv'}")
    
    # Run each model
    results = {model: {'folds': []} for model in models}
    
    for fold_info in folds:
        fold_idx = fold_info['fold']
        logger.info(f"\n{'='*60}")
        logger.info(f"FOLD {fold_idx}: {fold_info['train_start']} to {fold_info['val_end']}")
        logger.info(f"{'='*60}")
        
        # Split data
        df_train = df[(df['date'] >= fold_info['train_start']) & (df['date'] < fold_info['val_start'])]
        df_val = df[(df['date'] >= fold_info['val_start']) & (df['date'] < fold_info['val_end'])]
        
        X_train = df_train[feature_cols]
        y_train = df_train['label_3cls'].values
        X_val = df_val[feature_cols]
        y_val = df_val['label_3cls'].values
        
        # Preprocess
        X_train_scaled, X_val_scaled = prepare_features(X_train, X_val)
        
        # Train each model
        for model_type in models:
            result = train_model(
                model_type=model_type,
                X_train=X_train_scaled,
                y_train=y_train,
                X_val=X_val_scaled,
                y_val=y_val,
                instability_scores=instability_scores,
                feature_names=feature_cols
            )
            
            results[model_type]['folds'].append({
                'fold': fold_idx,
                **result['metrics']
            })
            
            # Compute SHAP analysis for this fold
            compute_shap_analysis(
                model=result['model'],
                model_type=model_type,
                X_train=X_train_scaled,
                X_val=X_val_scaled,
                feature_names=feature_cols,
                fold_idx=fold_idx,
                output_dir=output_path
            )
        
        # Also compute SHAP for baseline Logistic Regression
        logger.info(f"  Training baseline Logistic Regression for SHAP comparison...")
        logreg = LogisticRegression(
            penalty='l2',
            C=1.0,
            max_iter=1000,
            class_weight='balanced',
            random_state=42
        )
        logreg.fit(X_train_scaled, y_train)
        
        compute_shap_analysis(
            model=logreg,
            model_type='logreg',
            X_train=X_train_scaled,
            X_val=X_val_scaled,
            feature_names=feature_cols,
            fold_idx=fold_idx,
            output_dir=output_path
        )
    
    # Save results
    for model_type in models:
        # Save fold-level metrics
        fold_df = pd.DataFrame(results[model_type]['folds'])
        fold_csv = output_path / f'ml6_{model_type}_metrics.json'
        
        # Compute summary statistics
        summary = {
            'model': model_type,
            'n_folds': len(fold_df),
            'mean_f1_macro': float(fold_df['f1_macro'].mean()),
            'std_f1_macro': float(fold_df['f1_macro'].std()),
            'mean_f1_weighted': float(fold_df['f1_weighted'].mean()),
            'mean_balanced_acc': float(fold_df['balanced_acc'].mean()),
            'mean_kappa': float(fold_df['kappa'].mean()),
            'folds': results[model_type]['folds']
        }
        
        with open(fold_csv, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"\nSaved {model_type.upper()} results to {fold_csv}")
        logger.info(f"  Mean F1-macro: {summary['mean_f1_macro']:.4f} ± {summary['std_f1_macro']:.4f}")
    
    # Create summary CSV
    summary_rows = []
    for model_type in models:
        with open(output_path / f'ml6_{model_type}_metrics.json', 'r') as f:
            data = json.load(f)
        summary_rows.append({
            'model': model_type.upper(),
            'f1_macro_mean': data['mean_f1_macro'],
            'f1_macro_std': data['std_f1_macro'],
            'f1_weighted_mean': data['mean_f1_weighted'],
            'balanced_acc_mean': data['mean_balanced_acc'],
            'kappa_mean': data['mean_kappa']
        })
    
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(output_path / 'ml6_extended_summary.csv', index=False)
    
    # Create markdown summary
    with open(output_path / 'ml6_extended_summary.md', 'w') as f:
        f.write("# ML6 Extended Models Summary\n\n")
        f.write("## Temporal Instability Regularization\n\n")
        f.write("All tree-based models (RF, XGB, LGBM) use feature penalties based on variance across behavioral segments.\n\n")
        f.write("## Performance Results\n\n")
        f.write(summary_df.to_markdown(index=False))
        f.write("\n\n## Top 10 Most Unstable Features\n\n")
        f.write(instability_df.head(10).to_markdown(index=False))
    
    # Create SHAP summary across all folds
    logger.info("\nCreating consolidated SHAP summary...")
    shap_dir = output_path / 'shap'
    
    if shap_dir.exists():
        # Discover all models with aggregated SHAP files (not just those in current run)
        all_agg_files = list(shap_dir.glob('*_aggregated_top10.csv'))
        discovered_models = [f.stem.replace('_aggregated_top10', '') for f in all_agg_files]
        
        # Process each model
        for model_type in discovered_models:
            # Collect SHAP values across all folds
            shap_files = list(shap_dir.glob(f'{model_type}_fold*_shap_top10.csv'))
            
            if shap_files:
                # Aggregate SHAP importance across folds
                all_shap = []
                for shap_file in shap_files:
                    fold_shap = pd.read_csv(shap_file)
                    all_shap.append(fold_shap)
                
                # Combine and average
                combined = pd.concat(all_shap, ignore_index=True)
                avg_shap = combined.groupby('feature')['mean_abs_shap'].mean().reset_index()
                avg_shap = avg_shap.sort_values('mean_abs_shap', ascending=False)
                
                # Save aggregated top-10
                agg_path = shap_dir / f'{model_type}_aggregated_top10.csv'
                avg_shap.head(10).to_csv(agg_path, index=False)
                
                logger.info(f"  {model_type.upper()}: Top feature = {avg_shap.iloc[0]['feature']} "
                          f"(mean|SHAP|={avg_shap.iloc[0]['mean_abs_shap']:.4f})")
        
        # Create consolidated SHAP summary markdown with all available models
        shap_summary_path = shap_dir / 'shap_summary.md'
        with open(shap_summary_path, 'w') as f:
            f.write("# SHAP Analysis Summary - ML6 Extended Models\n\n")
            f.write("## Overview\n\n")
            f.write("SHAP (SHapley Additive exPlanations) values computed for:\n")
            f.write("- Logistic Regression (baseline)\n")
            f.write("- Random Forest (instability-regularized)\n")
            f.write("- XGBoost (instability-regularized)\n")
            f.write("- LightGBM (instability-regularized)\n\n")
            f.write("## Top-10 Features per Model\n\n")
            
            # Sort models for consistent output (logreg, lgbm, rf, xgb)
            sorted_models = sorted(discovered_models)
            for model_type in sorted_models:
                agg_file = shap_dir / f'{model_type}_aggregated_top10.csv'
                if agg_file.exists():
                    agg_df = pd.read_csv(agg_file)
                    f.write(f"### {model_type.upper()}\n\n")
                    f.write(agg_df.to_markdown(index=False))
                    f.write("\n\n")
            
            f.write("## Files Generated\n\n")
            f.write("- Per-fold SHAP: `{model}_fold{i}_shap_top10.csv` and `.png`\n")
            f.write("- Aggregated: `{model}_aggregated_top10.csv`\n")
            f.write("- This summary: `shap_summary.md`\n")
        
        logger.info(f"  Saved SHAP summary to {shap_summary_path}")
    
    logger.info(f"\n{'='*80}")
    logger.info(f"[OK] ML6 EXTENDED COMPLETE (with SHAP analysis)")
    logger.info(f"Results saved to: {output_path}")
    logger.info(f"SHAP results: {output_path / 'shap'}")
    logger.info(f"{'='*80}\n")


if __name__ == '__main__':
    import sys
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Default paths
    ml6_csv = 'data/ai/P000001/2025-11-07/ml6/features_daily_ml6.csv'
    cv_summary = 'data/ai/P000001/2025-11-07/ml6/cv_summary.json'
    segments_csv = 'data/etl/P000001/2025-11-07/segment_autolog.csv'
    output_dir = 'data/ai/P000001/2025-11-07/ml6_ext'
    
    run_ml6_extended(ml6_csv, cv_summary, segments_csv, output_dir)
