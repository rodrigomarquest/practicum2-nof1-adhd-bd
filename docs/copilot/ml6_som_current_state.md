# ML6 SoM-Centric: Current State Analysis

**Date**: 2025-12-08  
**Participant**: P000001  
**Snapshot**: 2025-12-08  
**Purpose**: Document current ML6 implementation as baseline for feature/target refinement

---

## 1. Current Implementation Overview

### Source Files

| File                           | Function                  | Description                                         |
| ------------------------------ | ------------------------- | --------------------------------------------------- |
| `scripts/run_full_pipeline.py` | `stage_5_prep_ml6()`      | Prepares features + target, applies MICE imputation |
| `scripts/run_full_pipeline.py` | `stage_6_ml6()`           | Trains LogisticRegression, runs temporal CV         |
| `src/etl/ml7_analysis.py`      | `create_calendar_folds()` | Creates calendar-based CV folds                     |

### Data Flow

```
Stage 4 output → features_daily_labeled.csv (2868 rows)
                          ↓
                   [Temporal Filter: >= 2021-05-11]
                          ↓
                   [SoM Filter: som_vendor == 'apple_autoexport']
                          ↓
Stage 5 output → features_daily_ml6.csv (77 rows, 23 cols)
                          ↓
                   [Feature Selection + CV]
                          ↓
Stage 6 output → cv_summary.json (F1=0.2916)
```

---

## 2. Feature Matrix (X)

### Current Feature Set (19 features)

| Category         | Feature               | Description                 | Coverage      |
| ---------------- | --------------------- | --------------------------- | ------------- |
| **Sleep (2)**    | `sleep_hours`         | Total sleep duration        | 100%          |
|                  | `sleep_quality_score` | Quality metric (0-100)      | 100%          |
| **Cardio (5)**   | `hr_mean`             | Daily mean heart rate       | 100%          |
|                  | `hr_min`              | Daily min heart rate        | 100%          |
|                  | `hr_max`              | Daily max heart rate        | 100%          |
|                  | `hr_std`              | Daily HR standard deviation | 100%          |
|                  | `hr_samples`          | Number of HR samples        | 100%          |
| **HRV (5)**      | `hrv_sdnn_mean`       | Mean SDNN (ms)              | 23.4% (18/77) |
|                  | `hrv_sdnn_median`     | Median SDNN (ms)            | 23.4%         |
|                  | `hrv_sdnn_min`        | Min SDNN (ms)               | 23.4%         |
|                  | `hrv_sdnn_max`        | Max SDNN (ms)               | 23.4%         |
|                  | `n_hrv_sdnn`          | Number of HRV samples       | 23.4%         |
| **Activity (3)** | `total_steps`         | Daily step count            | 100%          |
|                  | `total_distance`      | Distance (meters)           | 100%          |
|                  | `total_active_energy` | Active calories             | 100%          |
| **Meds (3)**     | `med_any`             | Binary med indicator        | 100%          |
|                  | `med_event_count`     | Count of med events         | 100%          |
|                  | `med_dose_total`      | Total dosage                | 100%          |
| **PBSI (1)**     | `pbsi_score`          | Auxiliary composite score   | 100%          |

**Total**: 19 features (after MICE imputation)

### Feature Exclusions (Anti-Leak)

The following columns are explicitly excluded:

- `pbsi_quality` - Derived from PBSI labels
- `sleep_sub`, `cardio_sub`, `activity_sub` - PBSI subscores
- `label_3cls`, `label_2cls` - PBSI labels (would leak target)

---

## 3. Target Variable (y)

### Primary Target: `som_category_3class`

| Class     | Meaning           | Count  | Percentage |
| --------- | ----------------- | ------ | ---------- |
| -1        | Negative/Unstable | 12     | 15.6%      |
| 0         | Neutral           | 11     | 14.3%      |
| +1        | Positive/Stable   | 54     | 70.1%      |
| **Total** |                   | **77** | 100%       |

### Derived Binary: `som_binary`

| Class | Meaning                              | Count | Percentage |
| ----- | ------------------------------------ | ----- | ---------- |
| 1     | Unstable (som_category_3class == -1) | 12    | 15.6%      |
| 0     | Not Unstable                         | 65    | 84.4%      |

### Target Selection Logic (Stage 6)

```python
MIN_SAMPLES_PER_CLASS = 5

# Try 3-class first
if min(class_counts) >= MIN_SAMPLES_PER_CLASS:
    use 3-class
else:
    fallback to binary

# If binary also imbalanced, skip gracefully
```

**Current**: 3-class is used (min class = 11 ≥ 5)

---

## 4. Cross-Validation Strategy

### Current: Temporal 6-Fold CV

| Parameter       | Value                    |
| --------------- | ------------------------ |
| N Folds         | 6                        |
| Split Method    | Temporal (by date order) |
| Fold Size       | ~12-13 samples           |
| Train/Val Ratio | ~85/15 per fold          |

### Fold Details (from cv_summary.json)

| Fold | Val Range               | N Train | N Val | F1-Macro | Bal. Acc |
| ---- | ----------------------- | ------- | ----- | -------- | -------- |
| 0    | 2023-11-19 → 2024-03-20 | 65      | 12    | 0.2857   | 0.4091   |
| 1    | 2024-05-06 → 2024-07-01 | 65      | 12    | 0.2593   | 0.2333   |
| 2    | 2024-07-07 → 2024-09-06 | 65      | 12    | 0.3000   | 0.4091   |
| 3    | 2024-09-07 → 2025-05-26 | 65      | 12    | 0.2456   | 0.3889   |
| 4    | 2025-05-28 → 2025-11-08 | 65      | 12    | 0.3205   | 0.3333   |
| 5    | 2025-11-09 → 2025-11-28 | 65      | 12    | 0.3385   | 0.3333   |

**Note**: Current implementation uses simple temporal splits, NOT calendar-based (4mo train / 2mo val). The `create_calendar_folds()` function in `ml7_analysis.py` is designed for calendar-based splits but is not currently used in Stage 6.

---

## 5. Model Configuration

### Current: LogisticRegression

```python
model = LogisticRegression(
    multi_class='auto',
    class_weight='balanced',  # Addresses class imbalance
    max_iter=1000,
    random_state=42
)
```

**Key Settings**:

- `class_weight='balanced'`: Automatically adjusts weights inversely proportional to class frequencies
- `multi_class='auto'`: Uses multinomial softmax for 3-class
- `random_state=42`: Ensures reproducibility

---

## 6. Current Performance Metrics

### Aggregate Metrics (6-fold CV)

| Metric                 | Value           |
| ---------------------- | --------------- |
| Mean F1-Macro          | 0.2916 ± 0.0324 |
| Mean Balanced Accuracy | 0.3512          |
| N Samples              | 77              |
| N Features             | 19              |

### Interpretation

- **F1-Macro = 0.29**: Below 0.33 random baseline for 3-class, indicating model struggles
- **Balanced Accuracy = 0.35**: Slightly above 0.33 random, but barely
- **High class imbalance**: 70% class +1 dominates predictions

### SHAP Feature Importance (from ML7)

Top-10 features by mean |SHAP| importance:

| Rank | Feature               | SHAP Importance |
| ---- | --------------------- | --------------- |
| 1    | `hrv_sdnn_min`        | 0.7547          |
| 2    | `hrv_sdnn_max`        | 0.2693          |
| 3    | `hrv_sdnn_mean`       | 0.1268          |
| 4    | `n_hrv_sdnn`          | 0.1093          |
| 5    | `total_steps`         | 0.0936          |
| 6    | `hr_samples`          | 0.0380          |
| 7    | `hrv_sdnn_median`     | 0.0247          |
| 8    | `total_distance`      | 0.0133          |
| 9    | `total_active_energy` | 0.0008          |
| 10   | `hr_max`              | 0.0000          |

**Key Insight**: HRV features dominate SHAP importance (top 4), but only 23% of samples have HRV data. This suggests HRV is highly predictive when available, but the model may overfit to imputed HRV values.

---

## 7. Known Issues and Limitations

### 7.1 Class Imbalance

- Class +1 (Positive) represents 70.1% of samples
- Class 0 (Neutral) and -1 (Negative) are minority classes
- Even with `class_weight='balanced'`, the model biases toward majority class

### 7.2 HRV Coverage Gap

- Only 18/77 days (23.4%) have real HRV data
- MICE imputation fills the gap, but imputed values may not reflect true physiology
- SHAP shows HRV as top predictor, which may be misleading

### 7.3 Small Sample Size

- N=77 samples is borderline for robust 6-fold CV
- With 12 samples per fold, class representation varies significantly across folds
- Some folds may have 0 samples of a minority class in validation set

### 7.4 PBSI as Auxiliary Feature

- `pbsi_score` is included as a feature (not target)
- PBSI is itself a composite score derived from sleep/cardio/activity
- This may introduce feature redundancy or leakage if PBSI subscores correlate with other features

---

## 8. Candidate Feature Sets for Ablation Study

### FS-A: Baseline (Original 10 Features)

The original canonical features from pre-SoM era, without HRV/MEDS/PBSI:

```
Sleep (2):     sleep_hours, sleep_quality_score
Cardio (5):    hr_mean, hr_min, hr_max, hr_std, hr_samples
Activity (3):  total_steps, total_distance, total_active_energy
```

**Total**: 10 features

### FS-B: Baseline + HRV (15 Features)

FS-A plus HRV features:

```
FS-A (10):     [all 10 baseline features]
HRV (5):       hrv_sdnn_mean, hrv_sdnn_median, hrv_sdnn_min, hrv_sdnn_max, n_hrv_sdnn
```

**Total**: 15 features

**Hypothesis**: HRV may add signal for SoM prediction, but low coverage (23%) may limit benefit or introduce noise through imputation.

### FS-C: FS-B + MEDS (18 Features)

FS-B plus medication features:

```
FS-B (15):     [all 15 FS-B features]
MEDS (3):      med_any, med_event_count, med_dose_total
```

**Total**: 18 features

**Hypothesis**: Medication patterns may correlate with SoM states (e.g., taking meds on bad days), but MEDS coverage (15.8% of all days) may be sparse within the 77-day SoM window.

### FS-D: FS-C + PBSI Auxiliary (19 Features)

FS-C plus PBSI composite score:

```
FS-C (18):     [all 18 FS-C features]
PBSI (1):      pbsi_score
```

**Total**: 19 features (current implementation)

**Hypothesis**: PBSI is an auxiliary "expert system" score that may capture patterns not visible in individual features. However, it may also introduce redundancy.

---

## 9. Experimental Questions

### Q1: Does HRV actually help, or is it noise from imputation?

- Compare FS-A vs FS-B
- If F1 drops with HRV, imputed values may be harmful

### Q2: Does MEDS add signal?

- Compare FS-B vs FS-C
- Check if med\_\* features have non-zero SHAP importance

### Q3: Does PBSI as feature improve or hurt?

- Compare FS-C vs FS-D
- PBSI may capture holistic patterns, or may leak information

### Q4: Is 3-class appropriate, or should we use binary?

- Run all experiments with both `som_category_3class` and `som_binary`
- Binary may be more robust with severe imbalance

### Q5: Would class-rebalancing techniques help?

- Test with/without `class_weight='balanced'`
- Consider SMOTE if sample size permits

---

## 10. Next Steps

1. **Create `src/etl/ml6_som_experiments.py`** - Controlled experiment runner
2. **Run ablation study**: FS-A/B/C/D × (3-class, binary)
3. **Generate PhD-level report** with metrics tables and interpretation
4. **Update Stage 5/6** with best-performing configuration
5. **Document final choice** with scientific justification

---

## Appendix: Raw Data Samples

### features_daily_ml6.csv (first 3 rows)

```
date,sleep_hours,sleep_quality_score,hr_mean,...,pbsi_score,som_category_3class,som_binary,segment_id
2023-11-19,5.38,100.0,78.45,...,0.502,1.0,0,global
2023-11-20,7.56,100.0,100.05,...,1.218,1.0,0,global
2023-11-23,5.59,100.0,96.55,...,0.648,1.0,0,global
```

### cv_summary.json (structure)

```json
{
  "model": "LogisticRegression",
  "target": "som_3class",
  "n_classes": 3,
  "class_distribution": {"-1.0": 12, "0.0": 11, "1.0": 54},
  "cv_type": "temporal_6fold",
  "mean_f1_macro": 0.2916,
  "std_f1_macro": 0.0324,
  "mean_balanced_accuracy": 0.3512,
  "n_samples": 77,
  "n_features": 19,
  "folds": [...]
}
```
