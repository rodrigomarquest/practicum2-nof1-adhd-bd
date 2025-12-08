# ML Stage Map (Pre-SoM Refactor)

**Date**: 2025-12-08  
**Purpose**: Document current Stage 5-8 implementation before refactoring to SoM-centric ML

---

## Stage 5: Prep ML6 (`stage_5_prep_ml6`)

### Current Behavior

**Input**: `features_daily_labeled.csv` from Stage 3

**Temporal Filter**:

- Cutoff: `>= 2021-05-11` (Amazfit GTR 2 start date)
- Rationale: Inter-device consistency (cardio data doesn't exist before this)

**Anti-Leak Blacklist** (columns dropped):

- `pbsi_score`
- `pbsi_quality`
- `sleep_sub`, `cardio_sub`, `activity_sub`

**Feature Columns** (X):

```python
feature_cols = [
    'sleep_hours', 'sleep_quality_score',
    'hr_mean', 'hr_min', 'hr_max', 'hr_std', 'hr_samples',
    'total_steps', 'total_distance', 'total_active_energy'
]
```

**Target** (y):

- `label_3cls` — PBSI-derived 3-class label (-1, 0, +1)

**Imputation**:

- MICE (`IterativeImputer`) with `max_iter=10`, `random_state=42`
- Segment-aware: imputes per `segment_id` if available
- Only imputes feature columns (X), NOT the target

**Output**:

- `ai/local/<PID>/<SNAPSHOT>/ml6/features_daily_ml6.csv`
- Columns: `date`, `label_3cls`, + 10 raw feature columns
- All NaN imputed (verified with assertion)

**Issues for SoM Refactor**:

1. Uses `label_3cls` (PBSI) as target — needs to change to SoM
2. Missing HRV features (`hrv_sdnn_*`)
3. Missing MEDS features (`med_any`, `med_event_count`, `med_dose_total`)
4. No SoM columns included

---

## Stage 6: ML6 Training (`stage_6_ml6`)

### Current Behavior

**Input**: `features_daily_ml6.csv` from Stage 5

**Target**: `label_3cls` (PBSI 3-class: -1, 0, +1)

**Model**:

- `LogisticRegression(multi_class='multinomial', class_weight='balanced', max_iter=1000)`
- Deterministic: `random_state=42`

**Cross-Validation**:

- `create_calendar_folds()` — 6-fold temporal calendar CV
- Train: 4 months, Validation: 2 months
- Non-overlapping temporal splits

**Metrics**:

- F1-macro (primary)
- Balanced accuracy
- Per-fold results logged

**Output**:

- `ai/local/<PID>/<SNAPSHOT>/ml6/cv_summary.json`
- Contains: model type, CV type, mean F1, std F1, per-fold results

**Graceful Skip**:

- If no valid folds (class imbalance), logs warning and returns `True` to continue pipeline

**Issues for SoM Refactor**:

1. Uses `label_3cls` (PBSI) — needs to switch to `som_category_3class` or `som_binary`
2. No fallback logic for extreme SoM class imbalance
3. No tracking of which target was actually used

---

## Stage 7: ML7 Analysis (`stage_7_ml7`)

### Current Behavior

**Input**: `features_daily_ml6.csv` from Stage 5

**Feature Transformation**:

- Applies z-scoring to raw features → z-scored features
- Maps raw columns to `ML7_FEATURE_COLS`:
  ```python
  ML7_FEATURE_COLS = [
      "z_sleep_total_h",
      "z_sleep_efficiency",
      "z_hr_mean",
      "z_hrv_rmssd",      # hr_std × 2 as HRV proxy
      "z_hr_max",
      "z_steps",
      "z_exercise_min",   # active_energy ÷ 5
  ]
  ```

**Target**: `label_3cls` (PBSI 3-class)

### Components

#### 1. SHAP Analysis

- Trains `LogisticRegression` per fold
- Computes SHAP values using `shap.LinearExplainer`
- Outputs:
  - Per-fold SHAP values
  - Global top-10 feature ranking
  - `shap_summary.md`

#### 2. Drift Detection

- **ADWIN**: Detects concept drift in `pbsi_score` — CURRENTLY SKIPPED (pbsi_score not in z-scored dataset)
- **KS Test**: At segment boundaries — CURRENTLY SKIPPED
- Outputs: `drift_report.md`

#### 3. LSTM Training

- **Architecture**: LSTM(32) → Dense(32) → Dropout(0.2) → Softmax(n_classes)
- **Sequence Length**: 14 days
- **Training**: Per-fold, same calendar folds as Stage 6
- **Metrics**: F1-macro, val_loss, val_accuracy
- **Output**: `lstm_report.md`, stores `best_model` in context

**Issues for SoM Refactor**:

1. Uses `label_3cls` (PBSI) as target
2. Drift detection references `pbsi_score`
3. Missing real HRV features (currently uses `hr_std` as proxy)
4. No fallback for insufficient SoM sequences

---

## Stage 8: TFLite Export (`stage_8_tflite`)

### Current Behavior

**Input**: `best_lstm_model` from Stage 7 context

**Actions**:

1. Converts Keras LSTM model to TFLite format
2. Measures inference latency (100 runs, p95)

**Output**:

- `ai/local/<PID>/<SNAPSHOT>/ml7/models/best_model.tflite`
- `ai/local/<PID>/<SNAPSHOT>/ml7/latency_stats.json`

**Graceful Skip**:

- If no LSTM model available, logs warning and returns `True`

**Issues for SoM Refactor**:

1. Minor: Log messages reference PBSI implicitly
2. No changes needed for core logic

---

## Stage 9: RUN Report (`stage_9_report`)

### Current Behavior

**Input**: Multiple files:

- `features_daily_labeled.csv`
- `ml6/cv_summary.json`
- Context results (SHAP, drift, LSTM)

**Report Content**:

1. Data Summary (date range, total rows)
2. Label Distribution (`label_3cls` — PBSI)
3. ML6 Results (F1, BA, per-fold)
4. SHAP Top-10 Features
5. Drift Detection Results (ADWIN, KS)
6. LSTM Results (per-fold F1)
7. TFLite Latency
8. Artifact Paths

**Output**:

- `docs/reports/RUN_<PID>_<SNAPSHOT>_stages<X-Y>_<TIMESTAMP>.md`

**Issues for SoM Refactor**:

1. Reports PBSI label distribution — needs SoM distribution
2. No SoM coverage metrics
3. No MEDS/HRV coverage metrics
4. No indication of which target was used (3-class vs binary)
5. No warnings about fallback or skip conditions

---

## Summary: Changes Needed for SoM Refactor

| Stage | Current Target      | New Target                                     | Key Changes                                          |
| ----- | ------------------- | ---------------------------------------------- | ---------------------------------------------------- |
| 5     | `label_3cls` (PBSI) | `som_category_3class`, `som_binary`            | Add HRV/MEDS features, filter by SoM validity        |
| 6     | `label_3cls` (PBSI) | `som_category_3class` (fallback: `som_binary`) | Add class imbalance fallback logic                   |
| 7     | `label_3cls` (PBSI) | Same as Stage 6                                | Update drift to use SoM, add data sufficiency checks |
| 8     | N/A                 | N/A                                            | Minor log updates                                    |
| 9     | PBSI distribution   | SoM distribution                               | Add SoM/MEDS/HRV coverage metrics                    |

---

## Feature Set Evolution

### Current (Stage 5)

```
sleep_hours, sleep_quality_score,
hr_mean, hr_min, hr_max, hr_std, hr_samples,
total_steps, total_distance, total_active_energy
```

Total: 10 features

### Proposed (SoM Refactor)

```
# Sleep (2)
sleep_hours, sleep_quality_score,

# Cardio + HRV (10)
hr_mean, hr_min, hr_max, hr_std, hr_samples,
hrv_sdnn_mean, hrv_sdnn_median, hrv_sdnn_min, hrv_sdnn_max, n_hrv_sdnn,

# Activity (3)
total_steps, total_distance, total_active_energy,

# Meds (3)
med_any, med_event_count, med_dose_total,

# PBSI as auxiliary feature (1, optional)
pbsi_score
```

Total: 18-19 features

---

_Document created: 2025-12-08_  
_Author: GitHub Copilot_
