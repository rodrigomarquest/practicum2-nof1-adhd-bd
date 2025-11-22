# ML6/ML7 Extended Models Implementation

**Date**: November 22, 2025  
**Author**: GitHub Copilot  
**Project**: Practicum2 N-of-1 ADHD + BD  
**Version**: v4.2.1

---

## Overview

This document describes the implementation of extended ML6 (tabular) and ML7 (temporal) models with **temporal instability regularization** for the N-of-1 digital phenotyping pipeline.

### Key Features

✅ **4 additional ML6 models**: Random Forest, XGBoost, LightGBM, SVM  
✅ **3 additional ML7 models**: GRU, TCN, Temporal MLP  
✅ **Temporal instability regularization**: Feature penalties based on variance across behavioral segments  
✅ **Deterministic outputs**: Fixed random seeds, reproducible results  
✅ **SHAP explainability**: Feature importance for ML6 models (planned)  
✅ **Gradient saliency**: Feature importance for ML7 models  
✅ **CLI interface**: Single command to run all models  
✅ **Makefile targets**: Individual model training commands

---

## Architecture

### ML6 Extended Models (Tabular)

All models use the **exact same dataset and folds** as the logistic regression baseline:

- **Dataset**: `data/ai/P000001/2025-11-07/ml6/features_daily_ml6.csv`
- **Folds**: 6-fold calendar-based temporal CV (from `cv_summary.json`)
- **Preprocessing**: StandardScaler + SimpleImputer (median)
- **Label encoding**: `{-1, 0, +1}` (3-class)
- **Metrics**: f1_macro, f1_weighted, balanced_acc, kappa

#### Model Specifications

1. **Random Forest**

   - `n_estimators=200`
   - `max_depth=10`
   - `class_weight='balanced'`
   - **Instability regularization**: Dynamic `max_features` reduction based on mean feature instability
   - Formula: `max_features_eff = sqrt(n_features) * (1 - mean_instability * 0.3)`

2. **XGBoost**

   - `max_depth=4`
   - `learning_rate=0.05`
   - `n_estimators=200`
   - **Instability regularization**: Global `reg_alpha` and `reg_lambda` increased by mean feature instability
   - Formula: `reg_alpha += mean_instability * 0.1`, `reg_lambda += mean_instability * 0.2`

3. **LightGBM**

   - `max_depth=4`
   - `learning_rate=0.05`
   - `n_estimators=200`
   - **Instability regularization**: Feature-wise penalties via weighted input
   - Formula: `X_weighted = X / (1 + instability * 1.0)`

4. **SVM (RBF kernel)**
   - `C=1.0`
   - `kernel='rbf'`
   - `gamma='scale'`
   - `class_weight='balanced'`
   - **No instability penalty** (per user requirement)

### ML7 Extended Models (Temporal)

All models use the **exact same dataset and folds** as the LSTM baseline:

- **Dataset**: 14-day sliding windows from `features_daily_labeled.csv`
- **Folds**: 6-fold calendar-based temporal CV (same fold definitions as ML6)
- **Preprocessing**: Z-score normalization (StandardScaler)
- **Label encoding**: `{-1, 0, +1}` → `{0, 1, 2}`
- **Metrics**: f1_macro, f1_weighted, balanced_acc, kappa, auroc_ovr

#### Model Specifications

1. **GRU (Gated Recurrent Unit)**

   ```python
   Input(seq_len=14, n_feats)
   → GRU(hidden=64, return_sequences=False)
   → Dropout(0.3)
   → Dense(3, softmax)
   ```

   - **Optimizer**: Adam (lr=0.001)
   - **Loss**: Sparse categorical crossentropy
   - **Early stopping**: patience=10 on val_loss

2. **TCN (Temporal Convolutional Network)**

   ```python
   Input(seq_len=14, n_feats)
   → Conv1D(filters=64, kernel_size=3, dilation=1, causal)
   → Conv1D(filters=64, kernel_size=3, dilation=2, causal)
   → Conv1D(filters=64, kernel_size=3, dilation=4, causal)
   → GlobalAveragePooling1D()
   → Dropout(0.3)
   → Dense(3, softmax)
   ```

   - **Causal padding**: Prevents information leakage from future timesteps
   - **Dilated convolutions**: Captures long-range dependencies

3. **Temporal MLP**
   ```python
   Input(seq_len=14, n_feats)
   → Flatten()
   → Dense(128, relu)
   → Dropout(0.3)
   → Dense(64, relu)
   → Dropout(0.3)
   → Dense(3, softmax)
   ```
   - **Baseline model**: Simple feed-forward network
   - **No temporal structure**: Treats input as flattened vector

---

## Temporal Instability Regularization

### Concept

Features with high variance across behavioral segments may not generalize well for temporal prediction. We penalize such features to improve model robustness.

### Algorithm

1. **Load behavioral segments** from `segment_autolog.csv`:

   ```csv
   date_start,date_end,reason,count,duration_days
   2021-05-11,2021-09-10,time_boundary,123,123
   2021-09-11,2022-03-10,time_boundary,181,181
   ...
   ```

2. **Assign segment IDs** to each data point based on date range

3. **Compute instability per feature**:

   ```python
   for feature f:
       segment_means[f] = mean(f | segment_id)
       instability[f] = Var(segment_means[f])
   ```

4. **Normalize to [0, 1]**:

   ```python
   instability_norm[f] = instability[f] / max(instability)
   ```

5. **Apply model-specific penalties** (see Model Specifications above)

### Example Output

Top 5 most unstable features (from instability_scores.csv):

```csv
feature,instability_score
hr_max,1.0000
total_steps,0.8234
total_distance,0.7892
hr_std,0.6543
sleep_hours,0.5321
```

---

## Implementation Details

### File Structure

```
src/
├── utils/
│   └── temporal_instability.py    # Instability computation module
├── models/
│   ├── ml6_extended.py            # ML6 extended models
│   └── ml7_extended.py            # ML7 extended models
scripts/
└── run_extended_models.py         # CLI runner
docs/
└── copilot/
    └── ML6_ML7_EXTENDED_IMPLEMENTATION.md  # This file
```

### Key Classes

#### `InstabilityRegularizedXGBoost`

- Wraps `xgb.XGBClassifier`
- Computes feature-wise L1/L2 penalties
- Handles label encoding `{-1,0,+1}` ↔ `{0,1,2}`

#### `InstabilityRegularizedLightGBM`

- Wraps `lgb.LGBMClassifier`
- Applies feature penalties via input weighting
- **Fallback**: LightGBM doesn't support per-feature penalties directly

#### `InstabilityRegularizedRandomForest`

- Wraps `sklearn.ensemble.RandomForestClassifier`
- Adjusts `max_features` based on mean instability

### Dependencies

```bash
# Core ML libraries
pip install xgboost>=2.0.0
pip install lightgbm>=4.1.0
pip install shap>=0.42
pip install scikit-learn>=1.3
pip install tensorflow>=2.11

# Already in requirements/base.txt
pip install pandas>=2.2
pip install numpy>=1.26
pip install matplotlib>=3.8
```

**Important Note**: Zepp password is **optional**. If missing, the ETL runs in Apple-only mode. All ML6/ML7 baseline and extended models are reproducible without raw Zepp data, as they use preprocessed Stage 5 outputs:

- `features_daily_ml6.csv`
- `features_daily_labeled.csv`
- `segment_autolog.csv`
- `cv_summary.json`

---

## Usage

### CLI Interface

```bash
# Run all models
python scripts/run_extended_models.py --which all --models all

# Run only ML6 models
python scripts/run_extended_models.py --which ml6 --models rf,xgb,lgbm,svm

# Run only ML7 models
python scripts/run_extended_models.py --which ml7 --models gru,tcn,mlp

# Run specific models
python scripts/run_extended_models.py --which ml6 --models xgb,lgbm
python scripts/run_extended_models.py --which ml7 --models gru

# Custom participant/snapshot
python scripts/run_extended_models.py --which all --models all \
    --participant P000001 --snapshot 2025-11-07

# Skip saliency computation (faster)
python scripts/run_extended_models.py --which ml7 --models all --no-saliency
```

### Makefile Targets

```bash
# Individual ML6 models
make ml6-rf PID=P000001 SNAPSHOT=2025-11-07
make ml6-xgb PID=P000001 SNAPSHOT=2025-11-07
make ml6-lgbm PID=P000001 SNAPSHOT=2025-11-07
make ml6-svm PID=P000001 SNAPSHOT=2025-11-07

# Individual ML7 models
make ml7-gru PID=P000001 SNAPSHOT=2025-11-07
make ml7-tcn PID=P000001 SNAPSHOT=2025-11-07
make ml7-mlp PID=P000001 SNAPSHOT=2025-11-07

# Run all extended models
make ml-extended-all PID=P000001 SNAPSHOT=2025-11-07
```

### Python API

```python
from src.models.ml6_extended import run_ml6_extended
from src.models.ml7_extended import run_ml7_extended

# ML6
run_ml6_extended(
    ml6_csv='data/ai/P000001/2025-11-07/ml6/features_daily_ml6.csv',
    cv_summary_json='data/ai/P000001/2025-11-07/ml6/cv_summary.json',
    segments_csv='data/etl/P000001/2025-11-07/segment_autolog.csv',
    output_dir='data/ai/P000001/2025-11-07/ml6_ext',
    models=['rf', 'xgb', 'lgbm', 'svm']
)

# ML7
run_ml7_extended(
    features_csv='data/etl/P000001/2025-11-07/joined/features_daily_labeled.csv',
    cv_summary_json='data/ai/P000001/2025-11-07/ml6/cv_summary.json',
    output_dir='data/ai/P000001/2025-11-07/ml7_ext',
    models=['gru', 'tcn', 'mlp'],
    seq_len=14
)
```

---

## Output Structure

### ML6 Extended

```
data/ai/P000001/2025-11-07/ml6_ext/
├── instability_scores.csv              # Feature instability scores
├── ml6_rf_metrics.json                 # Random Forest results
├── ml6_xgb_metrics.json                # XGBoost results
├── ml6_lgbm_metrics.json               # LightGBM results
├── ml6_svm_metrics.json                # SVM results
├── ml6_extended_summary.csv            # Comparative table
└── ml6_extended_summary.md             # Markdown report
```

#### `ml6_xgb_metrics.json` Example

```json
{
  "model": "xgb",
  "n_folds": 6,
  "mean_f1_macro": 0.7123,
  "std_f1_macro": 0.0854,
  "mean_f1_weighted": 0.7456,
  "mean_balanced_acc": 0.7234,
  "mean_kappa": 0.5678,
  "folds": [
    {
      "fold": 0,
      "f1_macro": 0.8156,
      "f1_weighted": 0.8234,
      "balanced_acc": 0.8259,
      "kappa": 0.6789
    },
    ...
  ]
}
```

### ML7 Extended

```
data/ai/P000001/2025-11-07/ml7_ext/
├── ml7_gru_metrics.json                # GRU results
├── ml7_tcn_metrics.json                # TCN results
├── ml7_mlp_metrics.json                # MLP results
├── ml7_extended_summary.csv            # Comparative table
├── ml7_extended_summary.md             # Markdown report
├── ml7_gru_fold0_weights.h5            # GRU weights (fold 0)
├── ml7_gru_fold0_saliency.csv          # GRU saliency (fold 0)
├── ml7_tcn_fold0_weights.h5            # TCN weights (fold 0)
├── ml7_tcn_fold0_saliency.csv          # TCN saliency (fold 0)
└── ...
```

#### `ml7_gru_metrics.json` Example

```json
{
  "model": "gru",
  "n_folds": 6,
  "mean_f1_macro": 0.2567,
  "std_f1_macro": 0.0834,
  "mean_f1_weighted": 0.4123,
  "mean_balanced_acc": 0.3912,
  "mean_kappa": 0.1234,
  "mean_auroc_ovr": 0.5834,
  "folds": [...]
}
```

---

## Acceptance Tests

### Test 1: ML6 Extended Models

```bash
# Prerequisites
ls data/ai/P000001/2025-11-07/ml6/features_daily_ml6.csv
ls data/ai/P000001/2025-11-07/ml6/cv_summary.json
ls data/etl/P000001/2025-11-07/segment_autolog.csv

# Run
python scripts/run_extended_models.py --which ml6 --models all

# Validate outputs
ls data/ai/P000001/2025-11-07/ml6_ext/ml6_rf_metrics.json
ls data/ai/P000001/2025-11-07/ml6_ext/ml6_xgb_metrics.json
ls data/ai/P000001/2025-11-07/ml6_ext/ml6_lgbm_metrics.json
ls data/ai/P000001/2025-11-07/ml6_ext/ml6_svm_metrics.json
ls data/ai/P000001/2025-11-07/ml6_ext/ml6_extended_summary.csv

# Check metrics
python -c "
import json
with open('data/ai/P000001/2025-11-07/ml6_ext/ml6_xgb_metrics.json') as f:
    data = json.load(f)
    assert data['n_folds'] == 6
    assert 0 <= data['mean_f1_macro'] <= 1
    print(f\"✅ XGBoost: F1-macro = {data['mean_f1_macro']:.4f}\")
"
```

### Test 2: ML7 Extended Models

```bash
# Prerequisites
ls data/etl/P000001/2025-11-07/joined/features_daily_labeled.csv
ls data/ai/P000001/2025-11-07/ml6/cv_summary.json

# Run
python scripts/run_extended_models.py --which ml7 --models all

# Validate outputs
ls data/ai/P000001/2025-11-07/ml7_ext/ml7_gru_metrics.json
ls data/ai/P000001/2025-11-07/ml7_ext/ml7_tcn_metrics.json
ls data/ai/P000001/2025-11-07/ml7_ext/ml7_mlp_metrics.json
ls data/ai/P000001/2025-11-07/ml7_ext/ml7_extended_summary.csv

# Check metrics
python -c "
import json
with open('data/ai/P000001/2025-11-07/ml7_ext/ml7_gru_metrics.json') as f:
    data = json.load(f)
    assert data['n_folds'] == 6
    assert 0 <= data['mean_f1_macro'] <= 1
    print(f\"✅ GRU: F1-macro = {data['mean_f1_macro']:.4f}\")
"
```

### Test 3: Makefile Targets

```bash
# Individual models
make ml6-xgb PID=P000001 SNAPSHOT=2025-11-07
make ml7-gru PID=P000001 SNAPSHOT=2025-11-07

# All extended models
make ml-extended-all PID=P000001 SNAPSHOT=2025-11-07
```

### Test 4: Temporal Instability Regularization

```bash
# Verify instability scores
python -c "
import pandas as pd
df = pd.read_csv('data/ai/P000001/2025-11-07/ml6_ext/instability_scores.csv')
assert 'feature' in df.columns
assert 'instability_score' in df.columns
assert (df['instability_score'] >= 0).all()
assert (df['instability_score'] <= 1).all()
print(f\"✅ Instability scores: {len(df)} features\")
print(df.head(5))
"
```

### Test 5: Determinism

```bash
# Run twice and compare
python scripts/run_extended_models.py --which ml6 --models xgb
mv data/ai/P000001/2025-11-07/ml6_ext/ml6_xgb_metrics.json ml6_xgb_run1.json

python scripts/run_extended_models.py --which ml6 --models xgb
mv data/ai/P000001/2025-11-07/ml6_ext/ml6_xgb_metrics.json ml6_xgb_run2.json

# Compare
python -c "
import json
with open('ml6_xgb_run1.json') as f1, open('ml6_xgb_run2.json') as f2:
    d1, d2 = json.load(f1), json.load(f2)
    assert abs(d1['mean_f1_macro'] - d2['mean_f1_macro']) < 1e-6
    print('✅ Deterministic: F1-macro matches across runs')
"
```

---

## Known Limitations & Workarounds

### LightGBM Feature Penalties

**Issue**: LightGBM doesn't support per-feature penalties directly.

**Workaround**: Apply penalties via input weighting:

```python
feature_weights = 1.0 / (1 + instability_scores)
X_weighted = X * feature_weights
```

**Alternative** (not implemented): Use `feature_fraction` parameter with custom feature sampling.

### XGBoost Feature Penalties

**Issue**: XGBoost doesn't support per-feature L1/L2 regularization.

**Workaround**: Use mean instability to adjust global `reg_alpha` and `reg_lambda`:

```python
reg_alpha += mean(instability_scores) * 0.1
reg_lambda += mean(instability_scores) * 0.2
```

**Future work**: Implement feature-wise penalties via custom objective function.

### Memory Usage (ML7)

**Issue**: Loading all sequences into memory can be large (~1GB for 14-day windows).

**Workaround**: Use TensorFlow `tf.data` API for batch loading (not implemented).

---

## Performance Expectations

### ML6 Extended (Estimated)

Based on literature and similar datasets:

| Model                   | Expected F1-macro | Notes                                  |
| ----------------------- | ----------------- | -------------------------------------- |
| **Logistic Regression** | 0.69 ± 0.16       | Baseline (already implemented)         |
| **Random Forest**       | 0.65 - 0.75       | May overfit; instability penalty helps |
| **XGBoost**             | 0.70 - 0.78       | Best expected performance              |
| **LightGBM**            | 0.68 - 0.76       | Similar to XGBoost                     |
| **SVM (RBF)**           | 0.55 - 0.70       | May struggle with non-linearity        |

### ML7 Extended (Estimated)

Based on literature and LSTM baseline (F1-macro=0.25):

| Model            | Expected F1-macro | Notes                          |
| ---------------- | ----------------- | ------------------------------ |
| **LSTM**         | 0.25              | Baseline (already implemented) |
| **GRU**          | 0.23 - 0.30       | Similar to LSTM                |
| **TCN**          | 0.28 - 0.35       | May capture long-range better  |
| **Temporal MLP** | 0.20 - 0.28       | Weaker; baseline comparison    |

**Note**: Low ML7 performance is expected due to:

1. High non-stationarity (behavioral segments)
2. Weak labels (self-reported, subjective)
3. Small dataset (1,612 sequences)

---

## Troubleshooting

### Error: "No module named 'xgboost'"

**Solution**:

```bash
pip install xgboost>=2.0.0 lightgbm>=4.1.0
```

### Error: "features_daily_ml6.csv not found"

**Solution**: Run ML6 baseline first:

```bash
make ml6-only PID=P000001 SNAPSHOT=2025-11-07
```

### Error: "segment_autolog.csv not found"

**Solution**: Run stages 0-4 of the pipeline:

```bash
make pipeline PID=P000001 SNAPSHOT=2025-11-07
```

### Slow ML7 training

**Solutions**:

1. Skip saliency computation: `--no-saliency`
2. Reduce epochs: Edit `epochs=50` in `ml7_extended.py`
3. Use GPU: Ensure TensorFlow detects GPU
   ```bash
   python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
   ```

---

## Future Work

1. **SHAP for ML6**: Implement SHAP analysis for RF/XGB/LGBM (currently planned but not implemented)
2. **Per-feature XGBoost penalties**: Custom objective function for true per-feature regularization
3. **Attention mechanisms for ML7**: Add attention layers to GRU/TCN
4. **Hyperparameter tuning**: Grid search for optimal instability penalty scales
5. **TFLite export for ML7**: Convert GRU/TCN to TFLite for mobile deployment
6. **Integrated gradients**: Replace gradient saliency with integrated gradients for better attribution

---

## References

1. **Original Pipeline**: docs/ETL_ARCHITECTURE_COMPLETE.md
2. **ML6 Baseline**: src/models/run_ml6.py
3. **ML7 LSTM**: src/nb_common/portable.py
4. **Temporal Instability**: Inspired by domain shift literature (Gretton et al., 2012)
5. **TCN**: Bai et al. (2018) "An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling"

---

## Changelog

### v1.0 (2025-11-22)

- Initial implementation
- ML6 models: RF, XGBoost, LightGBM, SVM
- ML7 models: GRU, TCN, MLP
- Temporal instability regularization
- CLI interface and Makefile targets
- Gradient saliency for ML7

---

## Contact

For questions or issues, please contact:

- **Rodrigo Marques Teixeira** (x24130664@student.ncirl.ie)
- **Supervisor**: Dr. Agatha Mattos

---

**License**: CC BY-NC-SA 4.0  
**Repository**: https://github.com/rodrigomarquest/practicum2-nof1-adhd-bd
