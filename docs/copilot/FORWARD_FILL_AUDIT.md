# Forward-Fill / Back-Fill Audit Report

**Project**: practicum2-nof1-adhd-bd  
**Release**: v4.1.4  
**Audit Date**: 2025-11-19  
**Auditor**: PhD-level Data Engineer

---

## Executive Summary

This report documents all instances of forward-fill (ffill) and back-fill (bfill) operations found in the codebase. Forward-fill and back-fill are data imputation techniques that propagate previous/next non-null values to fill gaps in time series data.

**Total Instances Found**: 9 distinct locations with forward/back-fill logic

**Risk Classification**:

- **HIGH-RISK**: 2 instances (ETL unified join, label auto-segmentation)
- **MEDIUM-RISK**: 1 instance (enriched post-join)
- **LOW-RISK**: 6 instances (notebook model prep, utility functions)

---

## Detailed Inventory

### HIGH-RISK USAGES

These affect core physiological signals at the ETL or label computation stage, potentially masking genuine missing data and creating spurious patterns.

#### 1. ETL Unified Daily Join - CRITICAL

| Property             | Value                                                                                                                                                                                                                 |
| -------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **File**             | `src/etl/stage_unify_daily.py`                                                                                                                                                                                        |
| **Function**         | `DailyUnifier.unify_all()`                                                                                                                                                                                            |
| **Line**             | 197                                                                                                                                                                                                                   |
| **Direction**        | ffill → bfill (chained)                                                                                                                                                                                               |
| **Columns Affected** | ALL numeric columns (sleep_hours, sleep_quality_score, hr_mean, hr_max, hr_min, hr_resting, hrv_rmssd, total_steps, distance_km, active_energy_kcal, stand_hours, exercise_minutes, and all vendor-specific variants) |
| **Stage**            | Stage 2: Unified Join (ETL)                                                                                                                                                                                           |
| **Window/Limit**     | None (fills entire time series)                                                                                                                                                                                       |
| **Code**             | `df_unified[numeric_cols] = df_unified[numeric_cols].fillna(method="ffill").fillna(method="bfill")`                                                                                                                   |

**Rationale (from code comment)**:

```python
# Forward fill NaN values (common in sensor data)
```

**Problem**:
This is the most critical usage in the entire pipeline. It applies unlimited forward-fill followed by back-fill to ALL numeric columns in the unified daily dataset. This means:

1. **Sleep data**: A day with genuinely no sleep (device not worn) gets filled with the previous day's sleep values → cannot distinguish real sleepless nights from missing sensors
2. **Heart rate**: Days without HR data get filled with stale HR from previous days → HR variability analysis is compromised
3. **Activity**: Missing step counts propagated → activity patterns artificially smoothed
4. **Multi-day gaps**: If user didn't wear device for 7 days, all 7 days get filled with the last known values → 100% domain coverage is misleading (as documented in ETL_AUDIT_REPORT.md)

**Impact**:

- **Data Quality**: Severe - masks ~36% missing sleep data, ~19% missing cardio data
- **Scientific Validity**: Violated - physiological signals on Day N are NOT related to Day N-1
- **Modeling**: Biased - models learn spurious continuity patterns
- **Interpretability**: Compromised - SHAP/feature importance may attribute effects to forward-filled values

**Recommendation**: **REMOVE** this forward-fill and preserve NaN values. Downstream modeling can handle missing data explicitly (e.g., masking, imputation with domain knowledge, or using models that support missing data natively).

---

#### 2. Label Auto-Segmentation - CRITICAL

| Property             | Value                                                  |
| -------------------- | ------------------------------------------------------ |
| **File**             | `src/labels/auto_segment.py`                           |
| **Function**         | `detect_physiological_triggers()`                      |
| **Lines**            | 82, 94, 106 (3 separate occurrences)                   |
| **Direction**        | ffill → bfill (chained)                                |
| **Columns Affected** | `apple_hr_mean`, `apple_hrv_rmssd`, `sleep_efficiency` |
| **Stage**            | Stage 3: Label Computation                             |
| **Window/Limit**     | None (fills entire time series)                        |
| **Code**             |                                                        |

```python
# Line 82
hr = df["apple_hr_mean"].ffill().bfill()

# Line 94
hrv = df["apple_hrv_rmssd"].ffill().bfill()

# Line 106
sleep = df["sleep_efficiency"].ffill().bfill()
```

**Rationale (inferred)**:
The function attempts to detect physiological triggers (HR changes, HRV changes, sleep efficiency changes) by comparing sliding window means. Forward-fill is used to ensure no NaN values when computing means.

**Problem**:

1. **HR/HRV triggers**: If HR data is missing for days, forward-filled values mask the true variability. A trigger detected from forward-filled data is spurious.
2. **Sleep efficiency**: Forward-filled sleep efficiency may trigger false segment boundaries
3. **Cascade effect**: These triggers are used to auto-segment participant data into time periods, which then affects label propagation

**Impact**:

- **Label Quality**: Moderate-High - false triggers → incorrect segment boundaries → mislabeled days
- **Auto-segmentation**: Compromised - cannot distinguish real physiological shifts from data gaps
- **Reproducibility**: Hidden bias - segment boundaries depend on fill strategy

**Recommendation**:

- **Option A**: Skip days with missing HR/HRV/sleep when computing triggers (i.e., only compare windows with real data)
- **Option B**: Use a more sophisticated trigger detection that explicitly handles missing data (e.g., require minimum data density in window)
- **Option C**: Remove auto-segmentation and rely only on manual labels (conservative)

---

### MEDIUM-RISK USAGES

These affect enriched/derived metrics, not core physiological signals, but may still introduce bias.

#### 3. Enriched Post-Join Interpolation

| Property             | Value                                                |
| -------------------- | ---------------------------------------------------- |
| **File**             | `src/domains/enriched/post/postjoin_enricher.py`     |
| **Function**         | `_handle_missing_domains()`                          |
| **Line**             | 86                                                   |
| **Direction**        | ffill only                                           |
| **Columns Affected** | Numeric columns (after linear interpolation attempt) |
| **Stage**            | Enrichment (post-join, Phase 3 architecture)         |
| **Window/Limit**     | None                                                 |
| **Code**             | `df[col] = df[col].ffill()`                          |

**Rationale (from code comment)**:

```python
"""Fill missing domain data with linear interpolation + forward fill.

Strategy: Try to infer missing dates from date column, interpolate, then ffill residual gaps.
This is a light touch - only for daily metrics when we have sparse coverage.
"""
```

**Problem**:

1. **Two-stage imputation**: First tries linear interpolation (reasonable), then forward-fills any remaining NaNs
2. **Applies to enriched metrics**: Not clear which columns are affected (depends on what's in enriched layer)
3. **"Light touch" claim**: Forward-fill with no limit is NOT light touch

**Impact**:

- **Enriched Metrics**: Moderate - derived metrics (correlations, z-scores, rolling windows) may be affected
- **Phase 3 Architecture**: Unclear - this is part of the Phase 3 enrichment pipeline, impact depends on usage

**Recommendation**:

- **Audit which columns** are actually processed by this function
- **Add limit parameter** to forward-fill (e.g., `ffill(limit=3)` to only fill 3 consecutive days max)
- **Or remove** forward-fill entirely and rely on interpolation + explicit NaN handling

---

### LOW-RISK USAGES

These occur in notebook/model preparation code, not in ETL. They affect model inputs but are more transparent and easier to replace.

#### 4. NB2 Baseline Model Preparation

| Property             | Value                                                                   |
| -------------------- | ----------------------------------------------------------------------- |
| **File**             | `notebooks/NB2_Baseline.py`                                             |
| **Function**         | `run_baselines()`                                                       |
| **Line**             | 52                                                                      |
| **Direction**        | ffill, then fillna(0)                                                   |
| **Columns Affected** | All numeric features (excluding label, label_source, label_notes, date) |
| **Stage**            | Notebook: Model preparation                                             |
| **Window/Limit**     | None                                                                    |
| **Code**             | `Xnum = df[numeric_cols].ffill().fillna(0)`                             |

**Rationale**: Ensure no NaN values before fitting scikit-learn models (LogisticRegression).

**Problem**: Forward-fill + zero-fill creates artificial feature values. Models trained on this data learn from imputed values.

**Impact**:

- **Baseline Models**: Low-Moderate - affects LogisticRegression baseline performance
- **Transparency**: High - code is in notebook, easy to see and modify
- **Scope**: Limited - only affects NB2 baseline comparisons, not production pipeline

**Recommendation**: Replace with explicit imputation strategy:

- Use `SimpleImputer(strategy='median')` or `KNNImputer`
- Or use models that handle missing data natively (e.g., LightGBM, XGBoost with `missing` parameter)

---

#### 5. NB3 Deep Learning Model Preparation

| Property             | Value                                                   |
| -------------------- | ------------------------------------------------------- |
| **File**             | `notebooks/NB3_DeepLearning.py`                         |
| **Function**         | `main()` (within sweep loop)                            |
| **Line**             | 95                                                      |
| **Direction**        | ffill, then fillna(0)                                   |
| **Columns Affected** | All numeric features (excluding label, date)            |
| **Stage**            | Notebook: Model preparation                             |
| **Window/Limit**     | None                                                    |
| **Code**             | `Xnum_full = df[numeric_cols].ffill().fillna(0).values` |

**Rationale**: Prepare sequences for LSTM/GRU models (cannot have NaN in input tensors).

**Problem**: Same as NB2 - forward-fill + zero-fill creates artificial sequences.

**Impact**:

- **Deep Learning Models**: Low-Moderate - affects LSTM/GRU performance
- **Sequence Learning**: Compromised - models may learn from padded/filled sequences
- **Scope**: Limited - only affects NB3 experiments

**Recommendation**:

- Use **masking layers** in Keras/TensorFlow (e.g., `Masking(mask_value=0.0)`)
- Or use **attention mechanisms** that can skip missing timesteps
- Or **explicit imputation** with domain knowledge (e.g., use rolling medians for missing HR)

---

#### 6. Portable Feature Engineering (Rolling Windows)

| Property             | Value                                                                           |
| -------------------- | ------------------------------------------------------------------------------- |
| **File**             | `src/nb_common/portable.py`                                                     |
| **Function**         | `apply_rolling()`                                                               |
| **Lines**            | 155, 158, 162 (within rolling window computation)                               |
| **Direction**        | fillna(0) after rolling operations                                              |
| **Columns Affected** | Rolling mean, rolling std, rolling diff                                         |
| **Stage**            | Feature engineering (notebook utility)                                          |
| **Window/Limit**     | Applied to rolling window results                                               |
| **Code**             | `.rolling(...).mean().fillna(0)`, `.rolling(...).std().fillna(0)`, `.fillna(0)` |

**Rationale**: Rolling window operations at the start of the series produce NaN (not enough history). Fill with 0 to avoid NaN in features.

**Problem**: Minimal - this is standard practice for rolling window features. Zero is a reasonable default for "no history yet".

**Impact**:

- **Rolling Features**: Low - only affects first few days of each rolling window
- **Feature Engineering**: Standard approach - not a concern

**Recommendation**: No change needed. This is acceptable practice.

---

#### 7. Feature Engineering Module (Identical to #6)

| Property             | Value                                          |
| -------------------- | ---------------------------------------------- |
| **File**             | `src/nb_common/features.py`                    |
| **Function**         | `apply_rolling()`                              |
| **Lines**            | 21, 24, 28 (within rolling window computation) |
| **Direction**        | fillna(0) after rolling operations             |
| **Columns Affected** | Rolling mean, rolling std, rolling diff        |
| **Stage**            | Feature engineering (notebook utility)         |
| **Window/Limit**     | Applied to rolling window results              |

**Rationale**: Same as #6.

**Impact**: Low - same as #6.

**Recommendation**: No change needed.

---

#### 8. Baseline Model Training Utility

| Property             | Value                                              |
| -------------------- | -------------------------------------------------- |
| **File**             | `src/models/baseline_train.py`                     |
| **Function**         | `run_logreg_with_xval()` and `run_xgb_with_xval()` |
| **Lines**            | 370, 496                                           |
| **Direction**        | fillna(0)                                          |
| **Columns Affected** | All numeric features                               |
| **Stage**            | Model training utility                             |
| **Window/Limit**     | None                                               |
| **Code**             | `X = df[numeric].astype(float).fillna(0.0)`        |

**Rationale**: Prepare features for scikit-learn models.

**Problem**: Simple zero-fill, no forward-fill. Better than ffill but still crude.

**Impact**:

- **Model Training**: Low - affects baseline model training scripts
- **Scope**: Limited to model training utilities

**Recommendation**: Use `SimpleImputer` or similar for more principled imputation.

---

#### 9. Activity Feature Engineering

| Property             | Value                                                                  |
| -------------------- | ---------------------------------------------------------------------- |
| **File**             | `src/domains/activity/activity_features.py`                            |
| **Function**         | Various feature computation functions                                  |
| **Lines**            | 50, 66 (multiple occurrences), 102                                     |
| **Direction**        | fillna(0)                                                              |
| **Columns Affected** | Activity-related features (kcal, active_min, stand hours, sleep hours) |
| **Stage**            | Feature computation                                                    |
| **Window/Limit**     | None                                                                   |
| **Code**             | `.fillna(0)` in various expressions                                    |

**Rationale**: When computing derived activity features (e.g., sedentary time = total_min - sleep - active - stand), missing values default to 0.

**Problem**: Minimal - using 0 for missing activity is reasonable (no activity = 0).

**Impact**:

- **Activity Features**: Very Low - zero is a sensible default for activity counts
- **Domain Semantics**: Reasonable - missing steps/kcal/stand hours ≈ 0

**Recommendation**: No change needed. This is acceptable.

---

## Summary by Pipeline Stage

| Stage                           | Files                        | Risk Level | Impact                                                               |
| ------------------------------- | ---------------------------- | ---------- | -------------------------------------------------------------------- |
| **ETL: Unified Join**           | `stage_unify_daily.py`       | 🔴 HIGH    | ALL core physiological signals forward-filled                        |
| **ETL: Label Computation**      | `auto_segment.py`            | 🔴 HIGH    | Auto-segmentation triggers computed from forward-filled HR/HRV/sleep |
| **Enrichment: Post-Join**       | `postjoin_enricher.py`       | 🟡 MEDIUM  | Enriched metrics forward-filled after interpolation                  |
| **Notebook: NB2 Baseline**      | `NB2_Baseline.py`            | 🟢 LOW     | Baseline model inputs forward-filled + zero-filled                   |
| **Notebook: NB3 Deep Learning** | `NB3_DeepLearning.py`        | 🟢 LOW     | LSTM/GRU inputs forward-filled + zero-filled                         |
| **Feature Engineering**         | `portable.py`, `features.py` | 🟢 LOW     | Rolling window edges zero-filled (standard)                          |
| **Model Training**              | `baseline_train.py`          | 🟢 LOW     | Model inputs zero-filled                                             |
| **Activity Features**           | `activity_features.py`       | 🟢 LOW     | Missing activity defaults to 0 (reasonable)                          |

---

## Risk Classification Summary

### HIGH-RISK: Remove or Redesign (PRIORITY 1)

**Files**:

1. `src/etl/stage_unify_daily.py` (line 197)
2. `src/labels/auto_segment.py` (lines 82, 94, 106)

**Why High-Risk**:

- Affects **core physiological signals** (sleep, HR, HRV, steps) at **ETL level**
- Creates **artificial continuity** in time series data
- **Masks missing data** (36% missing sleep, 19% missing cardio)
- **Scientifically invalid**: sleep on Day N ≠ sleep on Day N-1
- **Cascades to all downstream analysis**: modeling, PBSI, interpretability

**Action Required**:

- [ ] Remove forward-fill from `stage_unify_daily.py` and preserve NaN
- [ ] Redesign `detect_physiological_triggers()` to handle missing data explicitly
- [ ] Re-run ETL pipeline and validate with `sleep_hourly_audit.py` (Task 2)
- [ ] Update all downstream code to handle NaN values (modeling, PBSI, notebooks)

---

### MEDIUM-RISK: Audit and Limit (PRIORITY 2)

**Files**:

1. `src/domains/enriched/post/postjoin_enricher.py` (line 86)

**Why Medium-Risk**:

- Affects **derived/enriched metrics**, not raw physiological signals
- Uses two-stage imputation (interpolation + forward-fill)
- Scope unclear (depends on which columns are enriched)

**Action Required**:

- [ ] Audit which columns are processed by `_handle_missing_domains()`
- [ ] Add `limit` parameter to forward-fill (e.g., `ffill(limit=3)`)
- [ ] Or remove forward-fill and rely only on interpolation

---

### LOW-RISK: Document and Consider Alternatives (PRIORITY 3)

**Files**:

1. `notebooks/NB2_Baseline.py` (line 52)
2. `notebooks/NB3_DeepLearning.py` (line 95)
3. `src/nb_common/portable.py` (lines 155, 158, 162)
4. `src/nb_common/features.py` (lines 21, 24, 28)
5. `src/models/baseline_train.py` (lines 370, 496)
6. `src/domains/activity/activity_features.py` (lines 50, 66, 102)

**Why Low-Risk**:

- Notebook/model preparation code (transparent, easy to modify)
- Rolling window edge cases (standard practice)
- Activity features with zero defaults (semantically reasonable)

**Action Optional**:

- [ ] Replace `ffill().fillna(0)` in notebooks with `SimpleImputer` or model-native missing data handling
- [ ] Add masking layers for deep learning models
- [ ] Document imputation strategy in notebook markdown cells

---

## Hidden Forward-Fill Logic

**None found**. All forward-fill operations use explicit pandas methods (`.ffill()`, `.fillna(method="ffill")`). No custom loops or manual propagation logic detected.

---

## Recommended Action Plan

### Phase 1: ETL Fix (CRITICAL - Do Before Release)

1. **Remove forward-fill from unified join**:

   ```python
   # src/etl/stage_unify_daily.py, line 197
   # DELETE:
   # df_unified[numeric_cols] = df_unified[numeric_cols].fillna(method="ffill").fillna(method="bfill")

   # ADD:
   logger.info("[Unify] Preserving NaN values (no forward-fill)")
   # Optional: Log missing data stats per column
   ```

2. **Redesign auto-segmentation triggers**:

   ```python
   # src/labels/auto_segment.py
   # REPLACE: hr = df["apple_hr_mean"].ffill().bfill()
   # WITH: hr = df["apple_hr_mean"]  # Keep NaN
   # THEN: Only compute triggers on windows with sufficient non-NaN values
   ```

3. **Add data quality provenance column** (as recommended in ETL_AUDIT_REPORT.md):

   ```python
   # Track which sources contributed to each day
   df_unified["data_quality"] = ""
   df_unified.loc[df_sleep["date"].isin(...), "data_quality"] += "S"
   df_unified.loc[df_cardio["date"].isin(...), "data_quality"] += "C"
   df_unified.loc[df_activity["date"].isin(...), "data_quality"] += "A"
   ```

4. **Re-run full pipeline** (P000001/2025-11-07):

   ```bash
   make pipeline PID=P000001 SNAPSHOT=2025-11-07 ZPWD="..."
   ```

5. **Validate with sleep hourly audit** (Task 2):
   ```bash
   python -m src.etl.sleep_hourly_audit P000001 2025-11-07
   ```

### Phase 2: Model Code Updates (HIGH)

6. **Update NB2 baseline preparation**:

   ```python
   # Replace: Xnum = df[numeric_cols].ffill().fillna(0)
   # With: from sklearn.impute import SimpleImputer
   #       imputer = SimpleImputer(strategy='median')
   #       Xnum = pd.DataFrame(imputer.fit_transform(df[numeric_cols]),
   #                           columns=numeric_cols)
   ```

7. **Update NB3 deep learning preparation**:
   ```python
   # Add masking layer to LSTM/GRU models
   # Or use explicit imputation with domain knowledge
   ```

### Phase 3: Enrichment Audit (MEDIUM)

8. **Audit enrichment forward-fill**:
   - Determine which columns are processed by `_handle_missing_domains()`
   - Add `ffill(limit=3)` to prevent long-distance propagation
   - Document interpolation + ffill strategy

---

## Appendix: Search Patterns Used

```bash
# Pattern 1: Explicit ffill/bfill
grep -rn "ffill\|bfill" src/ notebooks/

# Pattern 2: fillna with method
grep -rn "fillna.*method" src/ notebooks/

# Pattern 3: Custom forward-fill logic (manual loops)
grep -rn "for.*in.*range.*:\[i-1\]" src/ notebooks/
```

---

**Audit Completed**: 2025-11-19  
**Next Steps**:

1. Implement Phase 1 fixes (ETL + auto-segmentation)
2. Run Task 2 (sleep hourly audit) to validate
3. Update modeling code (Phase 2)
4. Document changes in CHANGELOG.md

**Sign-off**: PhD-level Data Engineer

---

## ACTION PLAN (v4.1.5)

### Locations to Modify (HIGH-RISK)

1. **`src/etl/stage_unify_daily.py:197`** - `DailyUnifier.unify_all()`

   - **Columns**: ALL numeric (sleep, HR, HRV, steps, distance, energy, etc.)
   - **Stage**: ETL Stage 2 (Unified Join)
   - **Action**: REMOVE `fillna(method="ffill").fillna(method="bfill")`
   - **Rationale**: Core physiological signals must not be forward-filled

2. **`src/labels/auto_segment.py:82,94,106`** - `detect_physiological_triggers()`
   - **Columns**: `apple_hr_mean`, `apple_hrv_rmssd`, `sleep_efficiency`
   - **Stage**: Label Computation (auto-segmentation)
   - **Action**: REMOVE forward-fill, handle NaN explicitly in trigger detection
   - **Rationale**: Triggers from forward-filled data are spurious

### Locations to Leave Unchanged (LOW-RISK)

1. **`notebooks/NB2_Baseline.py:52`** - Model preparation
   - **Reason**: Notebook-level imputation, transparent, can be improved later
2. **`notebooks/NB3_DeepLearning.py:95`** - Model preparation

   - **Reason**: Notebook-level imputation, can add masking layers later

3. **`src/nb_common/portable.py:155,158,162`** - Rolling window edges

   - **Reason**: Standard practice, fillna(0) for initial window edges is acceptable

4. **`src/nb_common/features.py:21,24,28`** - Rolling window edges

   - **Reason**: Same as above

5. **`src/models/baseline_train.py:370,496`** - Model training utility

   - **Reason**: Limited scope, can improve with SimpleImputer later

6. **`src/domains/activity/activity_features.py:50,66,102`** - Activity features
   - **Reason**: Zero defaults for missing activity are semantically reasonable

### Medium-Risk Item (TO AUDIT)

1. **`src/domains/enriched/post/postjoin_enricher.py:86`** - `_handle_missing_domains()`
   - **Action**: Audit which columns are affected, consider adding `ffill(limit=3)`
   - **Priority**: After high-risk fixes are validated

---

## HIGH-RISK FIXES APPLIED (v4.1.5)

**Date**: 2025-11-19  
**Status**: ✅ Implemented

### 1. ETL Unified Join - FIXED

**File**: `src/etl/stage_unify_daily.py:197`  
**Function**: `DailyUnifier.unify_all()`

**Change**:

```python
# REMOVED (v4.1.4):
# df_unified[numeric_cols] = df_unified[numeric_cols].fillna(method="ffill").fillna(method="bfill")

# ADDED (v4.1.5):
# NOTE (v4.1.5): Forward-fill removed for scientific integrity.
# Missing values are kept as NaN to avoid inventing sleep/cardio/activity data.
logger.info(f"[Unify] Preserving NaN values (no forward-fill)")

# Log missing data statistics per column (added for transparency)
```

**Impact**:

- ✅ Core physiological signals (sleep, HR, HRV, steps) no longer forward-filled
- ✅ NaN values preserved in `features_daily_unified.csv`
- ✅ Missing data statistics logged for transparency
- ⚠️ Downstream code must now handle NaN explicitly

---

### 2. Auto-Segmentation Triggers - FIXED

**File**: `src/labels/auto_segment.py:82,94,106`  
**Function**: `detect_physiological_triggers()`

**Change**:

```python
# REMOVED (v4.1.4):
# hr = df["apple_hr_mean"].ffill().bfill()
# hrv = df["apple_hrv_rmssd"].ffill().bfill()
# sleep = df["sleep_efficiency"].ffill().bfill()

# ADDED (v4.1.5):
# Keep NaN, do not forward-fill
hr = df["apple_hr_mean"]
hrv = df["apple_hrv_rmssd"]
sleep = df["sleep_efficiency"]

# Only compute triggers on windows with ≥70% data density
min_data_density = 0.7
prev_density = prev_window.notna().sum() / len(prev_window)
curr_density = curr_window.notna().sum() / len(curr_window)
if prev_density >= min_data_density and curr_density >= min_data_density:
    # ... compute trigger
```

**Impact**:

- ✅ Triggers only detected when 70%+ of data is present in both windows
- ✅ False triggers from forward-filled data eliminated
- ⚠️ Fewer triggers may be detected in periods with sparse data (expected behavior)

---

### Documentation Updated

1. **`DETERMINISM_AND_NAN_POLICY_UPDATE.md`** (NEW):

   - Comprehensive documentation of NaN policy
   - Comparison of v4.1.4 vs v4.1.5
   - Downstream impact analysis
   - Validation and testing plan

2. **`FORWARD_FILL_AUDIT.md`** (THIS FILE):

   - Added ACTION PLAN section
   - Added HIGH-RISK FIXES APPLIED section

3. **Source Code Comments**:
   - Added `# NOTE (v4.1.5): ...` comments at each fix location
   - Explained rationale for removal

---

### Validation Status

**Pipeline Test**: ✅ **PASSED** (2025-11-19 03:33 UTC)

```bash
make clean-outputs
make pipeline PID=P000001 SNAPSHOT=2025-11-07 ZPWD="qqQKwnhY"
```

**Results**:

- ✅ ETL completed successfully
- ✅ NaN preserved in `features_daily_unified.csv`:
  - Sleep: 960/2828 NaN (34.0%)
  - HR: 1513/2828 NaN (53.5%)
  - Steps: 98/2828 NaN (3.5%)
- ✅ Z-scores preserve NaN (same counts as raw features)
- ✅ PBSI computed successfully (handles NaN by using available sub-scores)
- ✅ Labels assigned successfully
- ✅ NB2/NB3 generated outputs

**Key Finding**: NaN policy working as designed. Missing data is now explicitly represented at all pipeline stages.

**Next Steps**:

1. ✅ Run full pipeline validation - **COMPLETE**
2. ⏳ Run sleep hourly audit to verify NaN policy
3. ⏳ Update NB2/NB3 notebooks for NaN handling (if needed)
4. ⏳ Update PBSI documentation to mention NaN policy
