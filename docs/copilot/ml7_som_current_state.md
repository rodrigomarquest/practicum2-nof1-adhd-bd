# ML7 SoM-Centric: Current State Analysis

**Date**: 2025-12-08  
**Participant**: P000001  
**Snapshot**: 2025-12-08  
**Purpose**: Document current ML7 implementation before LSTM/SHAP/Drift refinement

---

## 1. Current Implementation Overview

### Source Files

| File                           | Function                     | Description                                   |
| ------------------------------ | ---------------------------- | --------------------------------------------- |
| `scripts/run_full_pipeline.py` | `stage_7_ml7()`              | SHAP analysis, drift detection, LSTM training |
| `scripts/run_full_pipeline.py` | `stage_8_tflite()`           | TFLite model export + latency measurement     |
| `src/etl/ml7_analysis.py`      | `compute_shap_values()`      | SHAP on LogisticRegression                    |
| `src/etl/ml7_analysis.py`      | `create_lstm_sequences()`    | Builds (X_seq, y_seq) for LSTM                |
| `src/etl/ml7_analysis.py`      | `train_lstm_model()`         | LSTM(32) → Dense(32) → Dropout(0.2) → Softmax |
| `src/etl/ml7_analysis.py`      | `convert_to_tflite()`        | Keras → TFLite conversion                     |
| `src/etl/ml7_analysis.py`      | `detect_drift_adwin()`       | ADWIN drift detection (river)                 |
| `src/etl/ml7_analysis.py`      | `detect_drift_ks_segments()` | KS-test at segment boundaries                 |

### Data Flow

```
Stage 5 output → features_daily_ml6.csv (77 rows, 19 cols)
                          ↓
              [Load + Sort by Date]
                          ↓
              [SHAP: LogReg on 80/20 split]
                          ↓
              [Drift: First vs Second half dist]
                          ↓
              [LSTM: seq_len=14, 80/20 split]
                          ↓
Stage 7 output → shap_summary.md, drift_report.md, lstm_report.md
                          ↓
Stage 8 output → best_model.tflite (39.6 KB)
```

---

## 2. Feature Set (Input Dimension)

### Current Features (15, from ML6 FS-B)

Stage 7 inherits features from Stage 5 output (`features_daily_ml6.csv`):

| Category         | Feature               | Coverage    |
| ---------------- | --------------------- | ----------- |
| **Sleep (2)**    | `sleep_hours`         | 92.2%       |
|                  | `sleep_quality_score` | 92.2%       |
| **Cardio (5)**   | `hr_mean`             | 100%        |
|                  | `hr_min`              | 100%        |
|                  | `hr_max`              | 100%        |
|                  | `hr_std`              | 100%        |
|                  | `hr_samples`          | 100%        |
| **HRV (5)**      | `hrv_sdnn_mean`       | 1.3% (1/77) |
|                  | `hrv_sdnn_median`     | 1.3%        |
|                  | `hrv_sdnn_min`        | 1.3%        |
|                  | `hrv_sdnn_max`        | 1.3%        |
|                  | `n_hrv_sdnn`          | 1.3%        |
| **Activity (3)** | `total_steps`         | 100%        |
|                  | `total_distance`      | 100%        |
|                  | `total_active_energy` | 100%        |

**Total Input Dimension**: 15 features per time step

**Note**: HRV coverage is extremely low (1.3%) but SHAP still ranks HRV features as most important. This is due to MICE imputation creating synthetic patterns that the model can learn from.

### Feature Exclusions

The following are explicitly excluded in Stage 7:

- `date` (identifier)
- `som_category_3class` (target)
- `som_binary` (target)
- `segment_id` (metadata)

---

## 3. Target Variable

### Current Target Selection Logic

```python
# From stage_7_ml7() in run_full_pipeline.py
class_dist_3 = df['som_category_3class'].value_counts()
use_3class = class_dist_3.min() >= 3  # At least 3 samples per class

if use_3class:
    y = df['som_category_3class'].values  # {-1, 0, +1}
    target_name = "som_3class"
else:
    y = df['som_binary'].values  # {0, 1}
    target_name = "som_binary"
```

### Current Target Distribution (som_3class)

| Class     | Meaning           | Count  | Percentage |
| --------- | ----------------- | ------ | ---------- |
| -1        | Negative/Unstable | 12     | 15.6%      |
| 0         | Neutral           | 11     | 14.3%      |
| +1        | Positive/Stable   | 54     | 70.1%      |
| **Total** |                   | **77** | 100%       |

**Decision**: Uses `som_3class` because min class (11) ≥ 3.

### Label Mapping for Keras

```python
# In train_lstm_model()
label_map = {-1: 0, 0: 1, 1: 2}  # {-1,0,+1} → {0,1,2}
# Reverse after prediction for F1 calculation
```

---

## 4. Sequence Configuration

### Current Settings

| Parameter         | Value   | Notes                    |
| ----------------- | ------- | ------------------------ |
| Sequence Length   | 14 days | Fixed in `stage_7_ml7()` |
| Min Days for LSTM | 30      | Minimum total samples    |
| N Sequences       | 64      | 77 - 14 + 1 = 64         |
| Train Sequences   | 51      | 80% of 64                |
| Val Sequences     | 13      | 20% of 64                |

### Sequence Creation

```python
# From create_lstm_sequences() in ml7_analysis.py
def create_lstm_sequences(X, y, seq_len=14):
    for i in range(len(X) - seq_len + 1):
        X_seq.append(X[i:i+seq_len])
        y_seq.append(y[i+seq_len-1])  # Predict LAST label in sequence
    return np.array(X_seq), np.array(y_seq)
```

**Note**: The model predicts the label at the END of the sequence (day 14), not aggregating across the window.

---

## 5. LSTM Architecture

### Current Architecture

```python
# From train_lstm_model() in ml7_analysis.py
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
```

### Training Configuration

| Parameter      | Value                           |
| -------------- | ------------------------------- |
| LSTM Units     | 32                              |
| Dense Units    | 32                              |
| Dropout Rate   | 0.2                             |
| Optimizer      | Adam (default lr=0.001)         |
| Loss           | Sparse Categorical Crossentropy |
| Epochs         | 20 (fixed)                      |
| Batch Size     | 32                              |
| Early Stopping | None                            |
| Class Weights  | None                            |

---

## 6. Current Performance Metrics

### LSTM Results (Most Recent Run)

| Metric       | Value  |
| ------------ | ------ |
| F1-Macro     | 0.5020 |
| Val Loss     | 0.9047 |
| Val Accuracy | 0.6154 |

### SHAP Feature Ranking (Top 10)

| Rank | Feature               | SHAP Importance |
| ---- | --------------------- | --------------- |
| 1    | `hrv_sdnn_mean`       | 2.6945          |
| 2    | `n_hrv_sdnn`          | 1.5580          |
| 3    | `hrv_sdnn_max`        | 1.1179          |
| 4    | `hrv_sdnn_min`        | 0.9498          |
| 5    | `hrv_sdnn_median`     | 0.4087          |
| 6    | `total_steps`         | 0.1856          |
| 7    | `hr_samples`          | 0.0159          |
| 8    | `total_distance`      | 0.0066          |
| 9    | `total_active_energy` | 0.0009          |
| 10   | `hr_max`              | 0.0000          |

**Observation**: HRV dominates despite 1.3% coverage. This is likely due to MICE imputation creating learned synthetic patterns.

### Drift Detection (SoM Distribution)

| Period      | Class +1 | Class 0 | Class -1 |
| ----------- | -------- | ------- | -------- |
| First Half  | 89.5%    | 2.6%    | 7.9%     |
| Second Half | 51.3%    | 25.6%   | 23.1%    |

**Observation**: Strong drift detected. Second half has much more class variability.

---

## 7. Issues and Limitations

### Known Issues

1. **No Early Stopping**: Training for fixed 20 epochs without early stopping risks overfitting.

2. **No Class Weights**: Severe class imbalance (70% class +1) not addressed in LSTM.

3. **Small Dataset**: 64 sequences total, 13 for validation → high variance.

4. **SHAP on Wrong Model**: SHAP is computed on LogisticRegression, not LSTM.

5. **Drift Detection Simplistic**: Only compares first vs second half, ignores temporal dynamics.

6. **Fixed Sequence Length**: 14 days may be too long given SoM sparsity (77 days over ~2 years).

### Performance Context

From ML6 ablation study:

- LogReg on som_binary achieved F1=0.4898 (best)
- LogReg on som_3class achieved F1=0.2916

LSTM on som_3class achieved F1=0.5020 → better than LogReg, but:

- Single train/val split (no CV)
- No regularization
- High variance expected

---

## 8. ML6 Feature Set Decision (Baseline for ML7)

From the ML6 ablation study, the optimal configuration was:

**FS-B × som_binary**:

- F1-macro: 0.4623 → 0.4898 (in production)
- 15 features (no MEDS, no PBSI)

**Features to Use in ML7** (same as ML6 FS-B):

```python
ML7_FEATURE_COLS = [
    # Sleep (2)
    'sleep_hours', 'sleep_quality_score',
    # Cardio (5)
    'hr_mean', 'hr_min', 'hr_max', 'hr_std', 'hr_samples',
    # HRV (5)
    'hrv_sdnn_mean', 'hrv_sdnn_median', 'hrv_sdnn_min', 'hrv_sdnn_max', 'n_hrv_sdnn',
    # Activity (3)
    'total_steps', 'total_distance', 'total_active_energy',
]
```

**Target Consideration**:

- ML6 showed `som_binary` performs better than `som_3class`
- However, for LSTM, 3-class may capture more temporal patterns
- Will evaluate both in ML7 experiments

---

## 9. Recommendations for ML7 Refinement

### Architecture Changes

1. **Add Early Stopping**: Patience=5, monitor='val_loss'
2. **Add Class Weights**: `class_weight='balanced'` equivalent for Keras
3. **Consider Shorter Sequences**: 7 days may reduce overfitting

### Experiment Grid (Proposed)

| Config | Seq Len | LSTM | Dense | Dropout | Notes           |
| ------ | ------- | ---- | ----- | ------- | --------------- |
| CFG-1  | 7       | 16   | 16    | 0.2     | Simple baseline |
| CFG-2  | 14      | 32   | 32    | 0.2     | Current/Legacy  |
| CFG-3  | 14      | 32   | 32    | 0.4     | Regularized     |

### SHAP Improvements

1. Compute SHAP on LSTM (using GradientExplainer or DeepExplainer)
2. Focus on temporal importance (which time steps matter)

### Drift Improvements

1. Use ADWIN on SoM predictions (if available)
2. KS-test at actual segment boundaries (S1-S6)

---

## 10. Files to Create/Modify

| File                                               | Action | Purpose                           |
| -------------------------------------------------- | ------ | --------------------------------- |
| `src/etl/ml7_som_experiments.py`                   | CREATE | Ablation runner for CFG-1/2/3     |
| `src/etl/ml7_analysis.py`                          | MODIFY | Add early stopping, class weights |
| `scripts/run_full_pipeline.py`                     | MODIFY | Use best config in stage_7_ml7()  |
| `docs/reports/ML7_SOM_lstm_experiments_*.md`       | CREATE | Experiment comparison report      |
| `docs/reports/ML7_SOM_drift_and_shap_summary_*.md` | CREATE | SHAP + Drift analysis             |

---

## 11. ML7 Feature Set Alignment (STEP 2)

### Decision: Use ML6 FS-B Feature Set

Based on ML6 ablation study results, ML7 will use the **FS-B** feature set (15 features):

| Category  | Features                                                                         | Count  |
| --------- | -------------------------------------------------------------------------------- | ------ |
| Sleep     | `sleep_hours`, `sleep_quality_score`                                             | 2      |
| Cardio    | `hr_mean`, `hr_min`, `hr_max`, `hr_std`, `hr_samples`                            | 5      |
| HRV       | `hrv_sdnn_mean`, `hrv_sdnn_median`, `hrv_sdnn_min`, `hrv_sdnn_max`, `n_hrv_sdnn` | 5      |
| Activity  | `total_steps`, `total_distance`, `total_active_energy`                           | 3      |
| **Total** |                                                                                  | **15** |

### Excluded Features (from ML6 findings)

| Feature           | Reason                                    |
| ----------------- | ----------------------------------------- |
| `med_any`         | MEDS adds marginal signal (+0.02 F1)      |
| `med_event_count` | MEDS adds marginal signal                 |
| `med_dose_total`  | MEDS adds marginal signal                 |
| `pbsi_score`      | Neutral to negative impact on performance |

### Anti-Leak Columns (must never be features)

- `som_category_3class` - Target variable
- `som_binary` - Target variable
- `label_3cls`, `label_2cls` - PBSI labels (target leakage)
- `pbsi_quality`, `sleep_sub`, `cardio_sub`, `activity_sub` - PBSI subscores

### Target Variable for ML7

**Primary**: `som_category_3class` (3-class: -1, 0, +1)

- Rationale: LSTM may capture temporal patterns better with more class granularity
- Will compare with `som_binary` in experiments

**Fallback**: `som_binary` (binary: 0, 1)

- Use if class imbalance prevents meaningful 3-class training

---

**Status**: STEP 2 Complete - ML7 feature alignment documented.

**Next**: STEP 3 - Create LSTM experiment grid.
