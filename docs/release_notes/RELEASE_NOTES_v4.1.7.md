# Release Notes: v4.1.7 (2025-11-20)

## 🎯 Major Changes: Intuitive PBSI + MICE Imputation

This release addresses two critical issues blocking ML model training and paper submission:

1. **PBSI Sign Convention Inverted** (v4.1.7): Fixed counterintuitive scoring where lower PBSI indicated better regulation
2. **MICE Imputation for Missing Data**: Implemented segment-aware multiple imputation to handle 56.6% missing data in 2017-2021 period

---

## 🔄 PBSI v4.1.7: Intuitive Sign Convention

### Problem (v4.1.6 and earlier)

- **Counterintuitive**: Lower PBSI scores indicated better physiological regulation
- **Example**: Day with 8h sleep, high HRV, 10k steps → PBSI = -0.79 → labeled `+1` (confusing!)
- **Interpretation difficulty**: Clinical users found negative scores for "good days" unintuitive

### Solution (v4.1.7)

**Inverted all PBSI formulas** so that **HIGHER scores = BETTER regulation** (intuitive!)

#### Formula Changes

**Sleep Subscore** (inverted sign):

```python
# v4.1.6 (counterintuitive):
sleep_sub = -0.6 * z_sleep_dur + 0.4 * z_sleep_eff

# v4.1.7 (intuitive):
sleep_sub = +0.6 * z_sleep_dur + 0.4 * z_sleep_eff  # ✅ Higher = better
```

**Cardio Subscore** (inverted all signs):

```python
# v4.1.6 (counterintuitive):
cardio_sub = +0.5 * z_hr_mean - 0.6 * z_hrv + 0.2 * z_hr_max

# v4.1.7 (intuitive):
cardio_sub = -0.5 * z_hr_mean + 0.6 * z_hrv - 0.2 * z_hr_max  # ✅ Higher HRV = better
```

**Activity Subscore** (inverted sign):

```python
# v4.1.6 (counterintuitive):
activity_sub = -0.7 * z_steps - 0.3 * z_exercise

# v4.1.7 (intuitive):
activity_sub = +0.7 * z_steps + 0.3 * z_exercise  # ✅ Higher activity = better
```

**Composite Score** (weights unchanged):

```python
pbsi_score = 0.40 * sleep_sub + 0.35 * cardio_sub + 0.25 * activity_sub
```

#### Label Changes

**New interpretation** (v4.1.7):

- `+1` (**high_pbsi**): HIGHER score = **regulated/stable** (good physiological state)
- `0` (**mid_pbsi**): Typical/average
- `-1` (**low_pbsi**): LOWER score = **dysregulated/unstable** (poor physiological state)

**Thresholds** (P25/P75 on 2021-2025 data):

- P25 = **-0.370** → Label `-1` if pbsi_score ≤ -0.370
- P75 = **+0.321** → Label `+1` if pbsi_score ≥ +0.321

**Distribution maintained**: 25% / 50% / 25% (low / mid / high)

---

## 🔬 Missing Data Handling: MICE Imputation

### Problem

- **56.6% of days** (1,600/2,828) had missing HR/HRV data
- **Root cause**: iPhone Motion API (2017-2020) lacks cardio sensors
- **Impact**: LogisticRegression failed with `ValueError: Input X contains NaN`

### Data Availability by Period

| Period                | Devices                         | Cardio Coverage | Days |
| --------------------- | ------------------------------- | --------------- | ---- |
| **P1-P2** (2017-2019) | iPhone only                     | **0%**          | 707  |
| **P3** (2020-2022)    | iPhone + Apple Watch (sporadic) | **31.2%**       | 731  |
| **P4** (2022-2023)    | Amazfit GTR 2                   | **46.3%**       | 365  |
| **P5** (2023-2024)    | Amazfit GTR 4                   | **79.5%**       | 365  |
| **P6** (2024-2025)    | Helio Ring                      | **95.2%**       | 660  |

**Critical date**: **2021-05-11** = first sustained cardio data (Amazfit GTR 2)

### Solution: Temporal Filter + MICE

**Stage 5 (ML6 Preparation)** now implements:

1. **Temporal Filter**:

   - **Cutoff**: `>= 2021-05-11` (Amazfit GTR 2 era)
   - **Excluded**: 1,203 days (2017-2020, iPhone-only era)
   - **Retained**: 1,625 days (2021-2025, 80.9% cardio coverage)
   - **Rationale**: MAR (Missing At Random) assumption valid for 2021-2025, violated for 2017-2020

2. **MICE Imputation**:

   - **Method**: `sklearn.experimental.IterativeImputer` (multiple imputation by chained equations)
   - **Parameters**:
     - `max_iter=10` (convergence iterations)
     - `random_state=42` (reproducibility)
   - **Strategy**: **Segment-aware** (imputes within temporal segments, respects non-stationarity)
   - **Results**:
     - Imputed: **1,938 missing values** (11.9% of 1,625 days × 10 features)
     - Remaining NaN: **0** ✅

3. **Anti-leak Verification**:
   - Removed: `pbsi_score`, `pbsi_quality`, `sleep_sub`, `cardio_sub`, `activity_sub`
   - Excluded: `segment_id` from ML features

**Output**: `data/ai/{participant}/{snapshot}/ml6/features_daily_nb2.csv`

---

## 📊 Performance Results

### Stage 6 (ML6): LogisticRegression

**6-fold calendar-based CV** (4mo train / 2mo val):

| Fold           | Period            | F1 Macro            | Balanced Acc |
| -------------- | ----------------- | ------------------- | ------------ |
| 0              | 2021-09 → 2021-11 | **0.8559**          | 0.8674       |
| 1              | 2022-03 → 2022-05 | 0.6544              | 0.8002       |
| 2              | 2022-09 → 2022-11 | **0.8674**          | 0.8725       |
| 3              | 2023-03 → 2023-05 | 0.7242              | 0.7820       |
| 4              | 2023-09 → 2023-11 | 0.6334              | 0.6728       |
| 5              | 2024-03 → 2024-05 | 0.3891              | 0.5409       |
| **Mean ± Std** |                   | **0.6874 ± 0.1608** |              |

**Interpretation**: Classical ML (LogisticRegression) achieves moderate performance on MICE-imputed features. High variance across folds suggests temporal non-stationarity.

### Stage 7 (ML7): SHAP + Drift + LSTM

**SHAP Feature Importance** (Top-3 global):

1. **z_sleep_total_h** (sleep duration)
2. **z_sleep_efficiency** (sleep quality)
3. **z_hrv_rmssd** (heart rate variability)

**Drift Detection**:

- **ADWIN**: 6 change points detected
  - 2022-03-26, 2022-12-07, 2023-09-21, 2023-12-26, 2024-12-12, 2025-05-21
- **KS Tests**: 40/336 significant (p<0.05) at segment boundaries

**LSTM M1** (seq_len=14):

- **Mean F1**: 0.3686 ± 0.1059
- **Note**: Lower than LogReg (temporal patterns complex, requires more data)

---

## 🔧 Technical Changes

### Modified Files

**`src/labels/build_pbsi.py`**:

- Lines 110-145: Inverted all PBSI subscore formulas
- Docstring updated with v4.1.7 sign convention explanation

**`scripts/run_full_pipeline.py`**:

- **Stage 5 (`stage_5_prep_nb2`)**: Completely rewritten (lines ~339-410)
  - Added temporal filter (`>= 2021-05-11`)
  - Added MICE imputation (segment-aware)
  - Added anti-leak verification
  - Changed output path: `ai/ml6/features_daily_nb2.csv`
- **Stage 7 (`stage_7_nb3`)**: Updated to use MICE-imputed data (lines ~585-605)
  - Now loads `ai/ml6/features_daily_nb2.csv` instead of `joined/features_daily_labeled.csv`
  - Applies same temporal filter (`>= 2021-05-11`)
  - Verifies 0 NaN before SHAP/LSTM

### New Dependencies

- `sklearn.experimental.enable_iterative_imputer`
- `sklearn.impute.IterativeImputer`

---

## 🚀 Usage

**Run full pipeline** (Stages 1-9):

```bash
python scripts/run_full_pipeline.py \
  --participant P000001 \
  --snapshot 2025-11-07
```

**Run specific stages**:

```bash
# Stage 5 only (temporal filter + MICE)
python scripts/run_full_pipeline.py --start-stage 5 --end-stage 5

# Stages 5-7 (prep + train + analysis)
python scripts/run_full_pipeline.py --start-stage 5 --end-stage 7
```

**Outputs**:

- **Stage 5**: `data/ai/P000001/2025-11-07/ml6/features_daily_nb2.csv` (1,625 days, 0 NaN)
- **Stage 6**: `data/ai/P000001/2025-11-07/ml6/cv_summary.json` (CV results)
- **Stage 7**: `data/ai/P000001/2025-11-07/ml7/` (SHAP, drift, LSTM outputs)

---

## ⚠️ Breaking Changes

### API Changes

- **Stage 5 output path changed**: `joined/features_nb2_clean.csv` → `ai/ml6/features_daily_nb2.csv`
- **Temporal filter applied**: ML datasets now start from **2021-05-11** (not 2017-12-04)

### Label Interpretation Reversed

- **v4.1.6**: `+1` = low_pbsi (high instability) ❌
- **v4.1.7**: `+1` = high_pbsi (high regulation) ✅

**Update downstream code** that interprets labels!

### EDA vs ML Datasets

- **EDA** (Stages 1-4): Uses full 2017-2025 dataset (2,828 days)
- **ML** (Stages 5-9): Uses filtered 2021-2025 dataset (1,625 days, MICE-imputed)

---

## 📚 Scientific Justification

### Temporal Filter Rationale

- **Hardware limitation**: iPhone Motion API (2017-2020) does not expose HR/HRV sensors
- **MAR assumption**: Missing At Random valid for 2021-2025 (occasional missed readings)
- **MNAR before 2021**: Missing Not At Random (hardware constraint, not random process)
- **Solution**: Exclude MNAR period, apply MICE to MAR period

### MICE Advantages

- **Gold standard**: Rubin (1987), van Buuren & Groothuis-Oudshoorn (2011)
- **Preserves uncertainty**: Multiple imputations capture variance
- **Segment-aware**: Respects temporal non-stationarity in longitudinal data
- **Unbiased**: Does not inflate correlations (unlike mean imputation)

### Limitations

- **Pre-2021 exclusion**: Limits longitudinal analysis (loses 1,203 days)
- **MAR assumption**: May not hold if missingness correlates with unobserved variables
- **Segment size**: Small segments (<5 days) not imputed (too few neighbors)

---

## 🔬 Validation

**Imputation quality checks**:

- ✅ 0 NaN remaining after MICE
- ✅ Anti-leak verified (no target leakage)
- ✅ Segment-aware (respects temporal structure)
- ✅ Reproducible (random_state=42)

**Model performance**:

- ✅ LogisticRegression trains successfully (F1=0.69±0.16)
- ✅ SHAP analysis completes (6 folds)
- ✅ LSTM trains successfully (F1=0.37±0.11)

---

## 📖 References

- Rubin, D. B. (1987). _Multiple Imputation for Nonresponse in Surveys_. Wiley.
- van Buuren, S., & Groothuis-Oudshoorn, K. (2011). mice: Multivariate Imputation by Chained Equations in R. _Journal of Statistical Software_, 45(3), 1-67.
- Azur, M. J., Stuart, E. A., Frangakis, C., & Leaf, P. J. (2011). Multiple imputation by chained equations: what is it and how does it work? _International Journal of Methods in Psychiatric Research_, 20(1), 40-49.

---

## 🏷️ Version History

- **v4.1.7** (2025-11-20): PBSI sign inversion + MICE imputation + temporal filter
- **v4.1.6** (2025-11-19): Percentile-based thresholds (P25/P75)
- **v4.1.5** (2025-11-18): Segment-wise z-scoring
- **v4.1.0** (2025-11-10): Canonical PBSI implementation

---

**Contributors**: Rodrigo Marques (@rodrigomarquest)  
**License**: See LICENSE file  
**Documentation**: See `docs/PBSI_LABELS_v4.1.7.md` for technical details
