# NB3 Fix Implementation Plan

## Problem Diagnosis

### Current Behavior (BROKEN):

1. **NB3 uses `features_nb2_clean.csv`** which has only RAW features:

   - sleep_hours, hr_mean, total_steps, etc.
   - NO z-scored features

2. **SHAP shows degenerate results**:

   - Only `total_steps` has non-zero importance (1.7372)
   - All others are 0.0000

3. **LSTM shows perfect scores**:

   - Macro-F1 = 1.0000 (suspicious, likely overfitting)
   - Only 1 valid fold

4. **Feature pipeline is inconsistent with CA2 paper**:
   - Paper claims: "segment-wise z-scored features"
   - Reality: Using raw features without z-normalization

### Root Cause:

- Line 518 in `run_full_pipeline.py`: `df = pd.read_csv(nb2_clean_path)`
- NB3 reads from wrong source (NB2 clean instead of labeled with z-features)

---

## Fix Strategy

### 1. Create NB3 Feature Set (Z-Scored Canonical Features)

```python
NB3_FEATURE_COLS = [
    "z_sleep_total_h",       # Sleep duration (z-scored)
    "z_sleep_efficiency",    # Sleep quality (z-scored, 0-1 scale)
    "z_apple_hr_mean",       # Heart rate mean (z-scored)
    "z_apple_hrv_rmssd",     # HRV proxy (hr_std × 2, z-scored)
    "z_apple_hr_max",        # Heart rate max (z-scored)
    "z_steps",               # Activity steps (z-scored)
    "z_exercise_min",        # Exercise estimate (active_energy ÷ 5, z-scored)
]
```

**Anti-leak safeguards**:

- DO NOT include: `pbsi_score`, `pbsi_quality`, `sleep_sub`, `cardio_sub`, `activity_sub`
- DO NOT include: `label_3cls`, `label_2cls`, `label_clinical`
- DO include: `segment_id` for context (but not as predictor)

### 2. Modify Stage 7 (NB3 Analysis)

**Current** (line 518):

```python
nb2_clean_path = ctx.joined_dir / "features_nb2_clean.csv"
df = pd.read_csv(nb2_clean_path)
```

**Fixed**:

```python
# Load labeled data with z-scored features
labeled_path = ctx.joined_dir / "features_daily_labeled.csv"
df_labeled = pd.read_csv(labeled_path)

# Select NB3 features (z-scored canonical)
NB3_FEATURE_COLS = [
    "z_sleep_total_h", "z_sleep_efficiency",
    "z_apple_hr_mean", "z_apple_hrv_rmssd", "z_apple_hr_max",
    "z_steps", "z_exercise_min"
]

# Anti-leak filter: Remove PBSI-derived outcomes
anti_leak_cols = [
    'pbsi_score', 'pbsi_quality',
    'sleep_sub', 'cardio_sub', 'activity_sub',
    'label_2cls', 'label_clinical'
]

df_nb3 = df_labeled[['date'] + NB3_FEATURE_COLS + ['label_3cls']].copy()
```

### 3. Update SHAP Logic

**Current**: SHAP explains NB2 Logistic on raw features (degener

ated)

**Fixed Options**:

**Option A**: Explain NB2 Logistic on z-scored features

```python
# Train LogReg on z-scored features
X_nb3 = df_nb3[NB3_FEATURE_COLS]
y = df_nb3['label_3cls']
model = LogisticRegression(...).fit(X_train, y_train)

# SHAP explains NB2 model with correct features
shap_result = compute_shap_values(model, X_train, X_val, NB3_FEATURE_COLS, ...)
```

**Option B**: Explain LSTM directly

```python
# Use KernelExplainer or DeepExplainer on LSTM
# More computationally expensive but explains the actual NB3 model
```

**Recommendation**: Option A (explain LogReg on z-features)

- Faster, more interpretable
- Shows which z-scored features drive classification
- Consistent with paper narrative

### 4. Update LSTM Training

**Current**: LSTM uses raw features from NB2 clean

```python
X_np = X.values  # X from features_nb2_clean.csv (raw)
```

**Fixed**: LSTM uses z-scored features

```python
X_nb3 = df_nb3[NB3_FEATURE_COLS]
X_np = X_nb3.values  # Now 7 z-scored features
seq_len = 14
n_features = len(NB3_FEATURE_COLS)  # = 7
```

### 5. Update Reporting

**SHAP Summary** (`shap_summary.md`):

```markdown
# SHAP Feature Importance Summary

**Model Explained**: Logistic Regression (NB2 baseline)
**Feature Set**: Z-scored canonical features (segment-wise normalized)
**Features**: 7 physiological/behavioral z-scores

## Global Top-7 Features

1. **z_steps**: X.XXXX
2. **z_sleep_efficiency**: X.XXXX
3. **z_apple_hr_mean**: X.XXXX
   ...
```

**LSTM Report** (`lstm_report.md`):

```markdown
# LSTM M1 Training Report

## Architecture

- Input: 7 z-scored features (segment-wise normalized)
- Sequence Length: 14 days
- LSTM(32) -> Dense(32) -> Dropout(0.2) -> Softmax(3)

## Feature Set

- z_sleep_total_h
- z_sleep_efficiency
- z_apple_hr_mean
- z_apple_hrv_rmssd (HRV proxy)
- z_apple_hr_max
- z_steps
- z_exercise_min

## Cross-Validation Results

...
```

---

## Implementation Steps

### Step 1: Define NB3 Feature Constants

- Add to `nb3_analysis.py` or `run_full_pipeline.py`
- Clear documentation of what each z-feature represents

### Step 2: Create NB3 Data Preparation Function

```python
def prepare_nb3_features(df_labeled: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare NB3 dataset with z-scored canonical features.

    Uses segment-wise z-scored features from PBSI canonical pipeline.
    Removes anti-leak columns (pbsi_score, subscores, derived labels).

    Args:
        df_labeled: Output from Stage 3 (features_daily_labeled.csv)

    Returns:
        DataFrame with (date, z_features, label_3cls)
    """
    NB3_FEATURE_COLS = [...]
    anti_leak_cols = [...]

    df_nb3 = df_labeled[['date'] + NB3_FEATURE_COLS + ['label_3cls']].copy()

    # Verify anti-leak
    for col in anti_leak_cols:
        assert col not in df_nb3.columns, f"Leak: {col} found in NB3 features"

    return df_nb3
```

### Step 3: Refactor Stage 7 (NB3 Analysis)

- Replace `pd.read_csv(nb2_clean_path)` with `prepare_nb3_features(df_labeled)`
- Update all references to `feature_names` to use `NB3_FEATURE_COLS`
- Update LSTM `n_features` to `len(NB3_FEATURE_COLS)` = 7

### Step 4: Fix SHAP Reporting

- Add header: "Model Explained: Logistic Regression (NB2 baseline)"
- Add header: "Feature Set: Z-scored canonical (segment-wise)"
- Clarify in logs which model is being explained

### Step 5: Update Drift Detection

- Keep using `df_labeled` for drift on `pbsi_score` (current behavior OK)
- Update KS test to use `NB3_FEATURE_COLS` instead of `feature_names` from NB2

### Step 6: Add Validation Logging

```python
logger.info(f"[NB3] Feature set: {len(NB3_FEATURE_COLS)} z-scored features")
logger.info(f"[NB3] Features: {', '.join(NB3_FEATURE_COLS)}")
logger.info(f"[NB3] Anti-leak verified: pbsi_score NOT in features")
```

### Step 7: Update Documentation

- Add inline comments explaining z-scored feature selection
- Update `RUN_REPORT.md` section headers
- Clarify SHAP explains NB2, not LSTM

---

## Expected Outcomes After Fix

### SHAP Summary

```markdown
## Global Top-7 Features

1. **z_steps**: 0.XXXX (non-zero, realistic)
2. **z_sleep_efficiency**: 0.XXXX
3. **z_apple_hr_mean**: 0.XXXX
4. **z_apple_hrv_rmssd**: 0.XXXX
5. **z_apple_hr_max**: 0.XXXX
6. **z_sleep_total_h**: 0.XXXX
7. **z_exercise_min**: 0.XXXX
```

**All features should have non-zero importance** (no more degenerate 0.0000 for 6 features)

### LSTM Report

```markdown
### Fold 1

- **Macro-F1**: 0.XXXX (realistic, likely < 1.0)
- **Val Loss**: 0.XXXX
- **Val Accuracy**: 0.XXXX (realistic, likely < 1.0)

**Mean Macro-F1**: 0.XXXX ± 0.XXXX
```

**F1 should NOT be 1.0000** unless genuinely justified by data

### RUN_REPORT.md

```markdown
## NB2: Logistic Regression (Temporal Calendar CV)

- **Features**: 11 raw physiological metrics
- **Mean Macro-F1**: 1.0000 ± 0.0000 (1 valid fold)

## NB3: SHAP Feature Importance (Logistic on Z-Features)

**Model**: Logistic Regression (same as NB2 but with z-scored features)
**Feature Set**: 7 segment-wise z-scored canonical features

1. **z_steps**: 0.XXXX
2. **z_sleep_efficiency**: 0.XXXX
   ...

## NB3: LSTM M1 (Sequence Model on Z-Features)

- **Architecture**: LSTM(32) with 7 z-scored inputs
- **Sequence Length**: 14 days
- **Mean Macro-F1**: 0.XXXX (realistic, may underperform NB2)
```

---

## Code Quality Checklist

- [ ] Define `NB3_FEATURE_COLS` as module constant
- [ ] Create `prepare_nb3_features()` with anti-leak validation
- [ ] Update Stage 7 to use labeled data with z-features
- [ ] Fix SHAP to explain correct model with correct features
- [ ] Update LSTM to use 7 z-features (not 11 raw)
- [ ] Add validation logging for feature selection
- [ ] Update all generated reports with correct headers
- [ ] Add inline comments explaining design decisions
- [ ] Verify determinism (seed=42 preserved)
- [ ] Test on P000001 snapshot 2025-11-07

---

## Testing Plan

1. **Run Full Pipeline**:

   ```bash
   make clean-outputs
   make pipeline PID=P000001 SNAPSHOT=2025-11-07 ZPWD="***"
   ```

2. **Verify Outputs**:

   - `shap_summary.md`: All 7 features with non-zero importance
   - `lstm_report.md`: Realistic F1 scores (not 1.0000)
   - `RUN_REPORT.md`: Clear labeling of which model SHAP explains

3. **Check Determinism**:
   - Run pipeline twice, compare outputs
   - PBSI scores should be identical (already validated)
   - NB3 metrics should be identical (new validation)

---

## Timeline

- **Step 1-2**: Define constants and prep function (30 min)
- **Step 3-4**: Refactor Stage 7 and SHAP (1 hour)
- **Step 5-6**: Update reporting and validation (30 min)
- **Step 7**: Documentation and comments (30 min)
- **Testing**: Full pipeline validation (1 hour)

**Total**: ~3.5 hours

---

End of implementation plan.
