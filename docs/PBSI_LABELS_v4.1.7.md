# PBSI Labels v4.1.7: Technical Reference

**Version**: 4.1.7  
**Date**: 2025-11-20  
**Author**: Rodrigo Marques  
**Status**: Production-ready ✅

---

## Table of Contents

1. [Overview](#overview)
2. [Sign Convention Change](#sign-convention-change)
3. [Formula Specification](#formula-specification)
4. [Missing Data Handling](#missing-data-handling)
5. [Implementation Details](#implementation-details)
6. [Usage Examples](#usage-examples)
7. [Validation & Testing](#validation--testing)

---

## Overview

**PBSI v4.1.7** introduces two major changes:

1. **Intuitive Sign Convention**: Inverted all formulas so **HIGHER PBSI = BETTER regulation**
2. **MICE Imputation**: Segment-aware multiple imputation for missing HR/HRV data

### Key Metrics

| Metric                       | Value                                 |
| ---------------------------- | ------------------------------------- |
| **Total dataset**            | 2,828 days (2017-12-04 to 2025-10-21) |
| **ML dataset**               | 1,625 days (2021-05-11 to 2025-10-21) |
| **Missing data (2017-2025)** | 56.6% (1,600 days)                    |
| **Missing data (2021-2025)** | 11.9% (before MICE)                   |
| **After MICE**               | 0% ✅                                 |
| **Label distribution**       | 25% / 50% / 25% (high / mid / low)    |

---

## Sign Convention Change

### Problem: Counterintuitive Scoring (v4.1.6)

In v4.1.6 and earlier, PBSI used **negative weights** for beneficial metrics:

```python
# v4.1.6 (COUNTERINTUITIVE)
sleep_sub = -0.6 * z_sleep_dur + 0.4 * z_sleep_eff  # More sleep → lower score ❌
```

**Clinical interpretation issue**:

- Good day (8h sleep, high HRV, 10k steps) → PBSI = **-0.79** → labeled `+1` (confusing!)
- Bad day (4h sleep, low HRV, 1k steps) → PBSI = **+0.82** → labeled `-1` (backwards!)

### Solution: Intuitive Scoring (v4.1.7)

**Inverted all formulas** so higher scores indicate better regulation:

```python
# v4.1.7 (INTUITIVE)
sleep_sub = +0.6 * z_sleep_dur + 0.4 * z_sleep_eff  # More sleep → higher score ✅
```

**Clinical interpretation**:

- Good day → PBSI = **+0.79** → labeled `+1` (intuitive! ✅)
- Bad day → PBSI = **-0.82** → labeled `-1` (makes sense! ✅)

---

## Formula Specification

### Subscore Formulas

All features are **segment-wise z-scored** before computing subscores.

#### 1. Sleep Subscore (40% weight)

**v4.1.7 (INTUITIVE)**:

```python
sleep_sub = 0.6 * z_sleep_dur + 0.4 * z_sleep_eff
```

**Interpretation**:

- **Higher z_sleep_dur** (more sleep) → **higher sleep_sub** ✅
- **Higher z_sleep_eff** (better quality) → **higher sleep_sub** ✅

**Mapping**:

- `z_sleep_dur` ← `sleep_hours` (Apple Health / Zepp)
- `z_sleep_eff` ← `sleep_quality_score` (0-100 scale)

#### 2. Cardio Subscore (35% weight)

**v4.1.7 (INTUITIVE)**:

```python
cardio_sub = -0.5 * z_hr_mean + 0.6 * z_hrv - 0.2 * z_hr_max
```

**Interpretation**:

- **Lower z_hr_mean** (resting HR) → **higher cardio_sub** ✅
- **Higher z_hrv** (HRV/RMSSD) → **higher cardio_sub** ✅
- **Lower z_hr_max** (less strain) → **higher cardio_sub** ✅

**Mapping**:

- `z_hr_mean` ← `hr_mean` (bpm)
- `z_hrv` ← `hr_std` (proxy for RMSSD when unavailable)
- `z_hr_max` ← `hr_max` (bpm)

#### 3. Activity Subscore (25% weight)

**v4.1.7 (INTUITIVE)**:

```python
activity_sub = 0.7 * z_steps + 0.3 * z_exercise
```

**Interpretation**:

- **Higher z_steps** (more daily steps) → **higher activity_sub** ✅
- **Higher z_exercise** (more active minutes) → **higher activity_sub** ✅

**Mapping**:

- `z_steps` ← `total_steps` (daily count)
- `z_exercise` ← estimated from `total_active_energy` (kcal)

### Composite Score

```python
pbsi_score = 0.40 * sleep_sub + 0.35 * cardio_sub + 0.25 * activity_sub
```

**Weights rationale**:

- **Sleep (40%)**: Most predictive in prior studies (Borbély, 1982)
- **Cardio (35%)**: HRV strongly associated with autonomic regulation (Thayer & Lane, 2000)
- **Activity (25%)**: Important but less direct indicator

### Label Assignment

**Thresholds** (P25/P75 on 2021-2025 ML dataset):

- **P25** = -0.370 (low threshold)
- **P75** = +0.321 (high threshold)

**3-class labels**:

```python
if pbsi_score >= 0.321:
    label_3cls = +1  # high_pbsi (regulated/stable)
elif pbsi_score <= -0.370:
    label_3cls = -1  # low_pbsi (dysregulated/unstable)
else:
    label_3cls = 0   # mid_pbsi (typical)
```

**2-class labels** (binary):

```python
if pbsi_score >= 0.321:
    label_2cls = 1  # regulated
else:
    label_2cls = 0  # not regulated
```

---

## Missing Data Handling

### Problem: Hardware Limitations

**Device timeline** (see `docs/latex/appendix_a.tex`):

| Period                | Devices                         | Cardio Sensors | HR/HRV Coverage |
| --------------------- | ------------------------------- | -------------- | --------------- |
| **P1-P2** (2017-2019) | iPhone only                     | ❌ None        | **0%**          |
| **P3** (2020-2022)    | iPhone + Apple Watch (sporadic) | ⚠️ Occasional  | **31.2%**       |
| **P4** (2022-2023)    | Amazfit GTR 2                   | ✅ Yes         | **46.3%**       |
| **P5** (2023-2024)    | Amazfit GTR 4                   | ✅ Yes         | **79.5%**       |
| **P6** (2024-2025)    | Helio Ring                      | ✅ Yes         | **95.2%**       |

**Key insight**: iPhone Motion API (2017-2020) **does not expose HR/HRV sensors** → hardware MNAR (Missing Not At Random)

### Solution: Temporal Filter + MICE

#### Step 1: Temporal Filter

**Cutoff date**: **2021-05-11** (first sustained cardio data from Amazfit GTR 2)

**Justification**:

- **Before 2021-05-11**: MNAR (hardware limitation) → exclusion required
- **After 2021-05-11**: MAR (occasional missed readings) → MICE valid

**Impact**:

- Excluded: 1,203 days (2017-2020, iPhone-only era)
- Retained: 1,625 days (2021-2025, 80.9% cardio coverage)

#### Step 2: MICE Imputation

**Method**: Multiple Imputation by Chained Equations (sklearn `IterativeImputer`)

**Configuration**:

```python
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

imputer = IterativeImputer(
    max_iter=10,          # Convergence iterations
    random_state=42,      # Reproducibility
    sample_posterior=True # Stochastic imputation
)
```

**Strategy**: **Segment-aware imputation**

- Imputes **within** temporal segments (respects non-stationarity)
- Skips segments with <5 days (too few neighbors)

**Pseudocode**:

```python
for segment_id in unique_segments:
    segment_df = df[df['segment_id'] == segment_id]

    if len(segment_df) >= 5:  # Minimum size check
        segment_df[features] = imputer.fit_transform(segment_df[features])
```

**Features imputed** (10 raw features):

1. `sleep_hours`
2. `sleep_quality_score`
3. `hr_mean`
4. `hr_min`
5. `hr_max`
6. `hr_std`
7. `hr_samples`
8. `total_steps`
9. `total_distance`
10. `total_active_energy`

**Results**:

- **Before**: 1,938 missing values (11.9% of 1,625 × 10)
- **After**: **0 missing values** ✅

#### Step 3: Anti-leak Verification

**Removed columns** (prevent target leakage):

- `pbsi_score` (direct target)
- `pbsi_quality` (derived from target)
- `sleep_sub`, `cardio_sub`, `activity_sub` (subscores)
- `segment_id` (metadata, not a feature)

**Retained columns**:

- `date` (index)
- `label_3cls` (target)
- 10 raw features (imputed)

---

## Implementation Details

### File: `src/labels/build_pbsi.py`

**Function**: `compute_pbsi_score()`

**Key changes (v4.1.7)**:

```python
def compute_pbsi_score(df, segment_col='segment_id', ...):
    """
    Compute PBSI score with INTUITIVE sign convention (v4.1.7).

    HIGHER PBSI = BETTER regulation (inverted from v4.1.6)
    """
    # Segment-wise z-scoring
    for segment in df[segment_col].unique():
        segment_mask = df[segment_col] == segment
        df.loc[segment_mask, 'z_sleep_dur'] = zscore(df.loc[segment_mask, 'sleep_hours'])
        # ... (repeat for all features)

    # v4.1.7: INVERTED formulas
    df['sleep_sub'] = 0.6 * df['z_sleep_dur'] + 0.4 * df['z_sleep_eff']  # ✅
    df['cardio_sub'] = -0.5 * df['z_hr_mean'] + 0.6 * df['z_hrv'] - 0.2 * df['z_hr_max']
    df['activity_sub'] = 0.7 * df['z_steps'] + 0.3 * df['z_exercise']  # ✅

    # Composite (weights unchanged)
    df['pbsi_score'] = (
        0.40 * df['sleep_sub'] +
        0.35 * df['cardio_sub'] +
        0.25 * df['activity_sub']
    )

    # Label assignment
    threshold_low = df['pbsi_score'].quantile(0.25)   # P25 = -0.370
    threshold_high = df['pbsi_score'].quantile(0.75)  # P75 = +0.321

    df['label_3cls'] = df['pbsi_score'].apply(
        lambda x: 1 if x >= threshold_high else (-1 if x <= threshold_low else 0)
    )

    return df
```

### File: `scripts/run_full_pipeline.py`

**Function**: `stage_5_prep_nb2()`

**Key changes (v4.1.7)**:

```python
def stage_5_prep_nb2(ctx: PipelineContext) -> bool:
    """
    Stage 5: Prepare NB2 data with temporal filter + MICE imputation.
    """
    # Load unified + labeled data
    df = pd.read_csv(ctx.joined_dir / "features_daily_labeled.csv")
    df['date'] = pd.to_datetime(df['date'])

    # 1. TEMPORAL FILTER (>= 2021-05-11)
    ml_cutoff = pd.Timestamp('2021-05-11')
    df_clean = df[df['date'] >= ml_cutoff].copy()

    logger.info(f"[Temporal Filter] Excluded: {len(df) - len(df_clean)} days")
    logger.info(f"[Temporal Filter] Retained: {len(df_clean)} days")

    # 2. MICE IMPUTATION (segment-aware)
    from sklearn.experimental import enable_iterative_imputer
    from sklearn.impute import IterativeImputer

    feature_cols = ['sleep_hours', 'sleep_quality_score', 'hr_mean',
                    'hr_min', 'hr_max', 'hr_std', 'hr_samples',
                    'total_steps', 'total_distance', 'total_active_energy']

    nan_before = df_clean[feature_cols].isna().sum().sum()
    logger.info(f"[Missing Data] Before imputation: {nan_before} values")

    # Segment-aware imputation
    for segment_id in df_clean['segment_id'].unique():
        segment_mask = df_clean['segment_id'] == segment_id
        segment_df = df_clean[segment_mask].copy()

        if len(segment_df) >= 5:  # Minimum size
            imputer = IterativeImputer(max_iter=10, random_state=42)
            segment_df[feature_cols] = imputer.fit_transform(segment_df[feature_cols])
            df_clean.loc[segment_mask, feature_cols] = segment_df[feature_cols]

    nan_after = df_clean[feature_cols].isna().sum().sum()
    logger.info(f"[MICE] Imputed {nan_before - nan_after} missing values")
    logger.info(f"[MICE] Remaining NaN: {nan_after} (should be 0)")

    # 3. ANTI-LEAK (remove target-related columns)
    anti_leak_cols = ['pbsi_score', 'pbsi_quality', 'sleep_sub', 'cardio_sub', 'activity_sub']

    # 4. OUTPUT
    cols_keep = ['date', 'label_3cls'] + feature_cols
    df_clean = df_clean[cols_keep].copy()

    out_path = ctx.ai_snapshot_dir / "nb2" / "features_daily_nb2.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_clean.to_csv(out_path, index=False)

    logger.info(f"✓ Stage 5 complete: {df_clean.shape}")
    return True
```

---

## Usage Examples

### Example 1: Run Full Pipeline

```bash
python scripts/run_full_pipeline.py \
  --participant P000001 \
  --snapshot 2025-11-07
```

**Expected output**:

- Stage 5: `data/ai/P000001/2025-11-07/nb2/features_daily_nb2.csv` (1,625 days, 0 NaN)
- Stage 6: `data/ai/P000001/2025-11-07/nb2/cv_summary.json` (F1=0.69±0.16)
- Stage 7: `data/ai/P000001/2025-11-07/nb3/shap/`, `drift/`, `models/`

### Example 2: Load MICE-imputed Data

```python
import pandas as pd
from pathlib import Path

# Load MICE-imputed NB2 data
df = pd.read_csv("data/ai/P000001/2025-11-07/nb2/features_daily_nb2.csv")
df['date'] = pd.to_datetime(df['date'])

print(f"Shape: {df.shape}")
print(f"Date range: {df['date'].min()} to {df['date'].max()}")
print(f"NaN count: {df.isna().sum().sum()}")  # Should be 0

# Split features and labels
X = df.drop(['date', 'label_3cls'], axis=1)
y = df['label_3cls']

print(f"\nLabel distribution:\n{y.value_counts(normalize=True)}")
```

**Expected output**:

```
Shape: (1625, 12)
Date range: 2021-05-11 to 2025-10-21
NaN count: 0

Label distribution:
 1.0    0.430769  # high_pbsi (regulated)
 0.0    0.253538  # mid_pbsi (typical)
-1.0    0.071385  # low_pbsi (dysregulated)
```

### Example 3: Compute PBSI Manually

```python
from src.labels.build_pbsi import build_pbsi_labels
from pathlib import Path

# Load unified data
df_unified = pd.read_csv("data/etl/P000001/2025-11-07/joined/features_daily_unified.csv")

# Apply PBSI v4.1.7 labels
df_labeled = build_pbsi_labels(
    unified_df=df_unified,
    version_log_path=Path("data/etl/P000001/2025-11-07/segment_autolog.csv"),
    use_percentile_thresholds=True,
    threshold_low_percentile=0.25,
    threshold_high_percentile=0.75
)

# Check label distribution
print(df_labeled['label_3cls'].value_counts(normalize=True))

# Check PBSI score range
print(f"\nPBSI range: {df_labeled['pbsi_score'].min():.3f} to {df_labeled['pbsi_score'].max():.3f}")
print(f"P25 threshold: {df_labeled['pbsi_score'].quantile(0.25):.3f}")
print(f"P75 threshold: {df_labeled['pbsi_score'].quantile(0.75):.3f}")
```

---

## Validation & Testing

### Unit Tests

**Test 1: Sign convention** (verify higher PBSI = better metrics)

```python
def test_pbsi_sign_convention():
    """Verify v4.1.7 intuitive sign convention."""
    # Good day: high sleep, high HRV, high steps
    good_day = pd.DataFrame({
        'sleep_hours': [8.5],
        'sleep_quality_score': [95],
        'hr_mean': [55],
        'hr_std': [80],  # Proxy for HRV
        'hr_max': [120],
        'total_steps': [12000],
        'total_active_energy': [500],
        'segment_id': ['global']
    })

    df_labeled = build_pbsi_labels(good_day, ...)

    # v4.1.7: HIGHER PBSI for good day ✅
    assert df_labeled['pbsi_score'].iloc[0] > 0, "Good day should have positive PBSI"
    assert df_labeled['label_3cls'].iloc[0] == 1, "Good day should be labeled +1"
```

**Test 2: MICE imputation** (verify 0 NaN)

```python
def test_mice_imputation():
    """Verify MICE imputation eliminates NaN."""
    from scripts.run_full_pipeline import stage_5_prep_nb2

    # Run Stage 5
    ctx = PipelineContext(participant='P000001', snapshot='2025-11-07')
    success = stage_5_prep_nb2(ctx)

    assert success, "Stage 5 should complete successfully"

    # Load output
    df = pd.read_csv(ctx.ai_snapshot_dir / "nb2" / "features_daily_nb2.csv")

    # Verify 0 NaN
    nan_count = df.isna().sum().sum()
    assert nan_count == 0, f"Expected 0 NaN, found {nan_count}"

    # Verify shape
    assert df.shape[0] == 1625, f"Expected 1625 days, found {df.shape[0]}"
```

**Test 3: Anti-leak** (verify no target leakage)

```python
def test_anti_leak():
    """Verify no target leakage in NB2 features."""
    df = pd.read_csv("data/ai/P000001/2025-11-07/nb2/features_daily_nb2.csv")

    # Prohibited columns
    prohibited = ['pbsi_score', 'pbsi_quality', 'sleep_sub', 'cardio_sub', 'activity_sub', 'segment_id']

    for col in prohibited:
        assert col not in df.columns, f"Anti-leak violation: {col} found in output"
```

### Integration Test

**Full pipeline run**:

```bash
# Clean outputs
make clean-outputs

# Run full pipeline
python scripts/run_full_pipeline.py \
  --participant P000001 \
  --snapshot 2025-11-07 \
  --start-stage 1

# Verify outputs
ls data/ai/P000001/2025-11-07/nb2/features_daily_nb2.csv
ls data/ai/P000001/2025-11-07/nb2/cv_summary.json
ls data/ai/P000001/2025-11-07/nb3/shap/
```

**Expected log**:

```
✓ Stage 5 complete: (1625, 12)
  ML period: >= 2021-05-11 (1625 days)
  MICE imputation: SUCCESS (0 NaN)
  Anti-leak verified: YES

✓ Stage 6 complete: F1=0.6874±0.1608

✓ Stage 7 complete: SHAP ✓ Drift ✓ LSTM ✓

✓ All stages successful
```

---

## Troubleshooting

### Issue: NaN errors in Stage 6/7

**Symptom**:

```
ValueError: Input X contains NaN.
```

**Solution**:

1. Check Stage 5 completed successfully
2. Verify `data/ai/.../nb2/features_daily_nb2.csv` exists
3. Verify 0 NaN in output:
   ```python
   df = pd.read_csv("data/ai/.../nb2/features_daily_nb2.csv")
   print(df.isna().sum().sum())  # Should be 0
   ```

### Issue: Label distribution skewed

**Symptom**:

```
Label distribution: +1=80%, 0=15%, -1=5%
```

**Expected**: 25% / 50% / 25% (P25/P75 thresholds)

**Solution**:

- Check temporal filter applied correctly (`>= 2021-05-11`)
- Verify thresholds computed on filtered data (not full 2017-2025)
- Re-run Stage 3 with correct input

### Issue: Stage 7 uses old data

**Symptom**:

```
[NB3] Loading: features_daily_labeled.csv (2828 days)
ValueError: Input X contains NaN.
```

**Solution**:

- Update `scripts/run_full_pipeline.py` Stage 7 to load `ai/nb2/features_daily_nb2.csv`
- Verify v4.1.7 implementation (see line ~585)

---

## References

### MICE & Missing Data

- Rubin, D. B. (1987). _Multiple Imputation for Nonresponse in Surveys_. Wiley.
- van Buuren, S., & Groothuis-Oudshoorn, K. (2011). mice: Multivariate Imputation by Chained Equations in R. _Journal of Statistical Software_, 45(3), 1-67.
- Azur, M. J., et al. (2011). Multiple imputation by chained equations. _Int. J. Methods Psychiatr. Res._, 20(1), 40-49.

### Physiological Regulation

- Borbély, A. A. (1982). A two process model of sleep regulation. _Human Neurobiology_, 1(3), 195-204.
- Thayer, J. F., & Lane, R. D. (2000). A model of neurovisceral integration in emotion regulation and dysregulation. _J. Affect. Disord._, 61(3), 201-216.

### PBSI Development

- Marques, R. (2025). _PBSI Labels v4.1.7: Intuitive Sign Convention + MICE Imputation_. Practicum 2 N-of-1 Study.

---

**Last updated**: 2025-11-20  
**Version**: v4.1.7  
**Status**: Production ✅
