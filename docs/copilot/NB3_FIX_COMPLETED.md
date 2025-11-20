# ML7 Fix Implementation - COMPLETED

**Date**: 2025-01-XX (placeholder)  
**Status**: ✅ **IMPLEMENTATION COMPLETE** - Ready for Testing

---

## Problem Summary

### Critical Issues Discovered in ML7 Pipeline

1. **Degenerate SHAP Results**:

   - Only 1 feature (total_steps: 1.7372) had non-zero importance
   - All other 9 features: 0.0000 (DEGENERATE)

2. **Suspicious LSTM Perfect Scores**:

   - Macro-F1: 1.0000 (unrealistic for this problem)
   - Val Accuracy: 1.0000 (red flag for overfitting or trivial problem)

3. **Wrong Feature Source**:

   - ML7 was reading from `features_nb2_clean.csv` (10 raw features)
   - Should read from `features_daily_labeled.csv` (7 z-scored features)

4. **Inconsistent with CA2 Paper**:
   - Paper claims: "segment-wise z-scored features"
   - Code used: raw features (no z-scoring)

---

## Root Cause Identified

**File**: `scripts/run_full_pipeline.py`, **Line 518**

**BEFORE (WRONG)**:

```python
# Load data
nb2_clean_path = ctx.joined_dir / "features_nb2_clean.csv"
df = pd.read_csv(nb2_clean_path)  # RAW features only

# Prepare features
X = df.drop(columns=['date', 'label_3cls'])  # 10 raw features
y = df['label_3cls']
feature_names = X.columns.tolist()  # Raw feature names

# LSTM
n_features = X_np.shape[1]  # = 10 (raw)
```

**AFTER (FIXED)**:

```python
# Load labeled data with z-scored features
from src.etl.ml7_analysis import prepare_nb3_features, NB3_FEATURE_COLS

labeled_path = ctx.joined_dir / "features_daily_labeled.csv"
df_labeled = pd.read_csv(labeled_path)
df = prepare_nb3_features(df_labeled)  # Anti-leak validation

# Prepare features (7 z-scored)
X = df[NB3_FEATURE_COLS].copy()  # 7 z-scored features
y = df['label_3cls'].copy()
feature_names = NB3_FEATURE_COLS  # Explicit z-feature names

# LSTM
n_features = len(NB3_FEATURE_COLS)  # = 7 (z-scored)
```

---

## Implementation Details

### 1. Added ML7 Constants (`src/etl/ml7_analysis.py`)

**NB3_FEATURE_COLS** (7 z-scored canonical features):

```python
NB3_FEATURE_COLS = [
    "z_sleep_total_h",       # Sleep duration (segment-wise z-scored)
    "z_sleep_efficiency",    # Sleep quality (segment-wise z-scored)
    "z_apple_hr_mean",       # Heart rate mean (segment-wise z-scored)
    "z_apple_hrv_rmssd",     # HRV proxy (segment-wise z-scored)
    "z_apple_hr_max",        # Heart rate max (segment-wise z-scored)
    "z_steps",               # Activity steps (segment-wise z-scored)
    "z_exercise_min",        # Exercise estimate (segment-wise z-scored)
]
```

**NB3_ANTI_LEAK_COLS** (7 prohibited columns):

```python
NB3_ANTI_LEAK_COLS = [
    'pbsi_score',      # Target-derived composite score
    'pbsi_quality',    # Quality flag derived from labels
    'sleep_sub',       # PBSI subscore (intermediate calculation)
    'cardio_sub',      # PBSI subscore (intermediate calculation)
    'activity_sub',    # PBSI subscore (intermediate calculation)
    'label_2cls',      # Binary label (derived from label_3cls)
    'label_clinical',  # Clinical threshold label (derived)
]
```

### 2. Added `prepare_nb3_features()` Function

**Purpose**: Validate and select z-scored features with anti-leak safeguards

**Functionality**:

- Verifies all 7 z-scored features are present in `df_labeled`
- Selects only `date`, `NB3_FEATURE_COLS`, and `label_3cls`
- Validates that no anti-leak columns are present
- Logs feature selection for audit trail

**Returns**: DataFrame with shape `(n_days, 9)` - date + 7 z-features + label

### 3. Refactored Stage 7 in `run_full_pipeline.py`

**Changes**:

1. Import `prepare_nb3_features` and `NB3_FEATURE_COLS` from `ml7_analysis`
2. Load `features_daily_labeled.csv` (has z-scored features)
3. Call `prepare_nb3_features(df_labeled)` to get validated ML7 dataset
4. Use explicit `NB3_FEATURE_COLS` for X matrix
5. Update LSTM to use `len(NB3_FEATURE_COLS)` (7 z-features, not 10 raw)
6. Enhanced logging for feature selection and validation

### 4. Enhanced SHAP Report (`shap_summary.md`)

**Added Headers**:

```markdown
**Model Explained**: Logistic Regression (multinomial, class_weight='balanced')

**Feature Set**: Z-scored canonical features from PBSI pipeline

- 7 features: z_sleep_total_h, z_sleep_efficiency, ...
- Segment-wise normalized (119 temporal segments) to prevent leakage

**Note**: SHAP explains the LogisticRegression baseline, NOT the LSTM model.
```

### 5. Enhanced LSTM Report (`lstm_report.md`)

**Added Sections**:

```markdown
## Architecture

- Input Features: 7 z-scored canonical features
  - z_sleep_total_h, z_sleep_efficiency, ...
- LSTM(32) -> Dense(32) -> Dropout(0.2) -> Softmax
- Classes: 3

## Feature Set

**Z-scored Canonical Features** (segment-wise normalized, 119 segments):

1. `z_sleep_total_h`
2. `z_sleep_efficiency`
   ... (all 7 features listed)

**Note**: Features are segment-wise z-scored to prevent temporal leakage.
```

---

## Expected Outcomes After Fix

### SHAP (Non-Degenerate Results)

**Before**:

```
1. total_steps: 1.7372     ← ONLY non-zero
2. total_distance: 0.0000  ← DEGENERATE
3. hr_mean: 0.0000         ← DEGENERATE
... (all others 0.0000)
```

**After** (Expected):

```
1. z_steps: 0.XXXX              ← Non-zero (realistic)
2. z_sleep_efficiency: 0.XXXX   ← Non-zero (realistic)
3. z_apple_hr_mean: 0.XXXX      ← Non-zero (realistic)
4. z_apple_hrv_rmssd: 0.XXXX    ← Non-zero (realistic)
5. z_apple_hr_max: 0.XXXX       ← Non-zero (realistic)
6. z_sleep_total_h: 0.XXXX      ← Non-zero (realistic)
7. z_exercise_min: 0.XXXX       ← Non-zero (realistic)
```

### LSTM (Realistic Performance)

**Before**:

```
Macro-F1: 1.0000  ← PERFECT (suspicious)
Val Accuracy: 1.0000  ← PERFECT (red flag)
Valid Folds: 1/6  ← Only 1 fold
```

**After** (Expected):

```
Macro-F1: 0.XXXX ± 0.XXXX  ← Realistic (likely < 1.0, may underperform ML6)
Valid Folds: 1-6  ← Multiple folds (if data permits)
```

**Note**: Due to extreme class imbalance (7.5% stable, 2.3% unstable), we may still see some single-class folds being skipped (expected behavior).

---

## Files Modified

1. **src/etl/ml7_analysis.py** (2 edits):

   - Added `NB3_FEATURE_COLS` constant (7 z-scored features)
   - Added `NB3_ANTI_LEAK_COLS` constant (7 prohibited columns)
   - Added `prepare_nb3_features()` function with validation
   - Fixed syntax error (duplicate `]` removed)

2. **scripts/run_full_pipeline.py** (4 edits):
   - Updated imports to include `prepare_nb3_features` and `NB3_FEATURE_COLS`
   - Changed data source from `features_nb2_clean.csv` to `features_daily_labeled.csv`
   - Updated feature matrix preparation to use explicit `NB3_FEATURE_COLS`
   - Updated LSTM to use `len(NB3_FEATURE_COLS)` instead of `X_np.shape[1]`
   - Enhanced SHAP report generation (model explained, feature set documentation)
   - Enhanced LSTM report generation (architecture, feature list, notes)
   - Added validation logging

---

## Testing Plan

### Step 1: Run Full Pipeline

```bash
# Clean previous outputs
make clean-outputs

# Run full pipeline with ML7 fix
make pipeline PID=P000001 SNAPSHOT=2025-11-07 ZPWD="your_password"
```

### Step 2: Verify Non-Degenerate Results

**Check SHAP Summary**:

```bash
cat data/ai/P000001/2025-11-07/ml7/shap_summary.md
```

**Expected**: All 7 z-features should have non-zero importance (not just 1 feature).

**Check LSTM Report**:

```bash
cat data/ai/P000001/2025-11-07/ml7/lstm_report.md
```

**Expected**:

- Feature list shows 7 z-features
- F1 scores are realistic (not 1.0000)
- Multiple valid folds (if data permits)

**Check RUN_REPORT.md**:

```bash
cat RUN_REPORT.md | grep -A 10 "ML7"
```

**Expected**: Stage 7 logs show validation messages, no degenerate warnings.

### Step 3: Validate Determinism

```bash
# Run 1
make clean-outputs
make pipeline PID=P000001 SNAPSHOT=2025-11-07

# Run 2
make clean-outputs
make pipeline PID=P000001 SNAPSHOT=2025-11-07

# Compare outputs: Should be IDENTICAL
diff -u data/ai/P000001/2025-11-07/ml7/shap_summary.md <(same from Run 1)
diff -u data/ai/P000001/2025-11-07/ml7/lstm_report.md <(same from Run 1)
```

**Expected**: Bit-for-bit identical outputs (determinism preserved).

---

## Timeline

| Phase                   | Duration       | Status             |
| ----------------------- | -------------- | ------------------ |
| Problem Diagnosis       | 2 hours        | ✅ COMPLETE        |
| Implementation Planning | 1 hour         | ✅ COMPLETE        |
| Code Implementation     | 1.5 hours      | ✅ COMPLETE        |
| Testing & Validation    | 2 hours        | ⏳ PENDING         |
| Documentation & Release | 1 hour         | ⏳ PENDING         |
| **TOTAL**               | **~7.5 hours** | **🔄 IN PROGRESS** |

---

## Next Steps

### Immediate (Next 2 Hours)

1. **Run Full Pipeline Test**:

   ```bash
   make clean-outputs
   make pipeline PID=P000001 SNAPSHOT=2025-11-07 ZPWD="password"
   ```

2. **Verify Results**:

   - SHAP: Check for non-zero importance across all 7 features
   - LSTM: Check for realistic F1 scores (not 1.0000)
   - Logs: Verify validation messages appear

3. **Address Issues** (if any):
   - If SHAP still degenerate: Investigate LogReg training, check class balance
   - If LSTM still F1=1.0: Check CV fold validity, verify label distribution

### Short-term (Next 4 Hours)

4. **Validate Determinism**:

   - Run pipeline twice
   - Compare outputs (should be identical)

5. **Update Documentation**:

   - Create `NB3_VALIDATION_REPORT.md` with before/after comparison
   - Update `DETERMINISM_VALIDATION_REPORT.md` with ML7 results

6. **Re-publish v4.1.4**:
   - Update CHANGELOG.md with ML7 fix notes
   - Create new git commit
   - Update git tag v4.1.4 (force) or create v4.1.5
   - Publish GitHub release

---

## Validation Checklist

- [ ] Pipeline runs without errors
- [ ] SHAP shows all 7 features with non-zero importance
- [ ] LSTM shows realistic F1 scores (not 1.0000)
- [ ] Multiple CV folds valid (if data permits)
- [ ] Validation logs present in output
- [ ] Reports document feature set and model clearly
- [ ] Determinism preserved (2 runs identical)
- [ ] No anti-leak columns in ML7 features
- [ ] Z-scored features used (not raw)

---

## Success Criteria

### ✅ Fix is considered successful if:

1. **SHAP Non-Degenerate**: At least 5 out of 7 features have non-zero importance
2. **LSTM Realistic**: Macro-F1 < 1.0 (may underperform ML6, but should be reasonable)
3. **Determinism**: 2 independent runs produce identical outputs
4. **Transparency**: Reports clearly document which model SHAP explains and which features are used
5. **Anti-leak**: No prohibited columns (pbsi_score, subscores) in ML7 features
6. **Consistency**: Z-scored features align with CA2 paper claims

### ⚠️ Acceptable Outcomes (Not Failures):

- **Single-class folds skipped**: Expected due to extreme class imbalance
- **LSTM underperforms ML6**: Acceptable (may not be suitable for this problem)
- **Some features have low importance**: Acceptable (not all features need to be equally important)

### ❌ Failure Criteria (Require Re-investigation):

- SHAP still shows only 1-2 features non-zero
- LSTM still shows F1=1.0000 across all folds
- Anti-leak columns detected in ML7 features
- Raw features detected (not z-scored)
- Determinism broken (2 runs differ)

---

## Implementation Sign-off

**Implementation Date**: 2025-01-XX  
**Implementer**: GitHub Copilot  
**Code Review**: Pending  
**Testing**: Pending  
**Status**: ✅ **CODE COMPLETE** - Ready for Testing

**Files Changed**: 2  
**Lines Added**: ~70  
**Lines Modified**: ~50  
**Tests Added**: 0 (smoke test recommended)

**Next Action**: Run `make pipeline PID=P000001 SNAPSHOT=2025-11-07` to validate fix.

---

**END OF IMPLEMENTATION REPORT**
