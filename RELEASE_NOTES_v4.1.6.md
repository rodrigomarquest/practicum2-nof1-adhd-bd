# Release Notes: v4.1.6 - PBSI Label Improvements

**Date**: November 20, 2025  
**Type**: Feature Enhancement + Scientific Improvement  
**Impact**: Resolves class imbalance, improves clinical coherence, maintains backward compatibility

---

## 🎯 Summary

Version 4.1.6 implements **percentile-based thresholds** for PBSI labels, resolving extreme class imbalance (93% neutral → balanced 25/50/25 split) and renaming labels to **descriptive, non-clinical terms** (low_pbsi/mid_pbsi/high_pbsi) with explicit disclaimers about clinical validity.

---

## ✨ Key Changes

### 1. Percentile-Based Thresholds (Default)

**Before (v4.1.5)**:

```python
# Fixed thresholds
stable:   pbsi ≤ -0.5  →  176 days (6.2%)
neutral:  -0.5 < pbsi < 0.5  → 2,643 days (93.5%)
unstable: pbsi ≥ 0.5  →    9 days (0.3%)
```

**After (v4.1.6)**:

```python
# Percentile-based thresholds (P25/P75)
low_pbsi:  pbsi ≤ P25 (-0.117)  →  707 days (25%)
mid_pbsi:  P25 < pbsi < P75  → 1,414 days (50%)
high_pbsi: pbsi ≥ P75 (0.172)  →  707 days (25%)
```

**Impact**:

- ✅ **Balanced classes**: All 3 classes have sufficient samples for ML training
- ✅ **Cross-validation viable**: Stage 6 (NB2) no longer skips training
- ✅ **Adaptive**: Thresholds adjust to each participant's data distribution
- ✅ **Scientifically defensible**: Quartile-based interpretation

### 2. Renamed Labels (Clinical Coherence)

**Before (v4.1.5)**:

- `+1`: "stable" (❌ vague, implies psychiatric state)
- `0`: "neutral" (❌ clinically ambiguous)
- `-1`: "unstable" (❌ does not map to DSM-5 diagnoses)

**After (v4.1.6)**:

- `+1`: **"low_pbsi (regulated)"** - Physiologically regulated patterns
- `0`: **"mid_pbsi (typical)"** - Typical physiological patterns
- `-1`: **"high_pbsi (dysregulated)"** - Physiologically dysregulated patterns

**Impact**:

- ✅ **Descriptive, not prescriptive**: Labels describe data, not diagnoses
- ✅ **Avoids psychiatric terminology**: Does not claim to detect mania/depression/ADHD
- ✅ **Transparent**: Clear that these are composite physiological indices

### 3. Clinical Validity Disclaimers

Added explicit documentation that PBSI labels:

- ❌ Are **NOT validated** against mood diaries, clinician ratings, or DSM-5 criteria
- ❌ Do **NOT represent** psychiatric states (mania, depression, ADHD severity)
- ✅ Are **exploratory** physiological composites requiring validation
- ✅ Should be cross-referenced with clinical context (medications, life events)

**Where**:

- Module docstring (`src/labels/build_pbsi.py`)
- Function docstrings
- Logger output
- Documentation (`docs/PBSI_LABELS_v4.1.6.md`, `docs/CLINICAL_COHERENCE_ANALYSIS.md`)

### 4. Backward Compatibility

**Maintained**:

- Legacy fixed thresholds available via flag: `use_percentile_thresholds=False`
- Label integer values unchanged (+1, 0, -1)
- Column names unchanged (`label_3cls`, `label_2cls`)
- Pipeline integration unchanged (Stage 3)

**Deprecated**:

- `label_clinical` column (replaced by `label_3cls`)
- "stable/neutral/unstable" terminology (documentation only)

---

## 📊 Results (P000001, Snapshot 2025-11-07)

### Class Distribution

| Metric                       | v4.1.5 (Fixed) | v4.1.6 (Percentile) | Improvement          |
| ---------------------------- | -------------- | ------------------- | -------------------- |
| **Low PBSI (regulated)**     | 176 (6.2%)     | 707 (25%)           | +4.0x samples        |
| **Mid PBSI (typical)**       | 2,643 (93.5%)  | 1,414 (50%)         | -46.5% (balanced)    |
| **High PBSI (dysregulated)** | 9 (0.3%)       | 707 (25%)           | +78.5x samples       |
| **Min class size**           | 9 days         | 707 days            | Viable for 6-fold CV |
| **Class ratio**              | 1:294:20       | 1:2:1               | Balanced             |

### Training Impact

| Pipeline Stage       | v4.1.5     | v4.1.6  | Notes                      |
| -------------------- | ---------- | ------- | -------------------------- |
| **Stage 6 (NB2)**    | ❌ Skipped | ✅ Runs | Sufficient class diversity |
| **Stage 7 (NB3)**    | ⚠️ Partial | ✅ Full | LSTM training enabled      |
| **Stage 8 (TFLite)** | ❌ Skipped | ✅ Runs | Model export enabled       |

---

## 🔧 API Changes

### `build_pbsi_labels()` Function

**New Parameters**:

```python
def build_pbsi_labels(
    unified_df: pd.DataFrame,
    version_log_path: Optional[Path] = None,
    output_path: Optional[Path] = None,
    # New in v4.1.6
    use_percentile_thresholds: bool = True,           # ← Default: percentile-based
    threshold_low_percentile: float = 0.25,           # ← P25
    threshold_high_percentile: float = 0.75,          # ← P75
    threshold_low_fixed: float = -0.5,                # ← Fallback
    threshold_high_fixed: float = 0.5,                # ← Fallback
) -> pd.DataFrame:
```

**Usage Examples**:

```python
# Default (v4.1.6 behavior - percentile-based)
df_labeled = build_pbsi_labels(unified_df)

# Legacy fixed thresholds (v4.1.5 behavior)
df_labeled = build_pbsi_labels(unified_df, use_percentile_thresholds=False)

# Custom percentiles (20/60/20 split)
df_labeled = build_pbsi_labels(
    unified_df,
    threshold_low_percentile=0.20,
    threshold_high_percentile=0.80
)
```

---

## 📝 Documentation Updates

### New Files

- **`docs/PBSI_LABELS_v4.1.6.md`**: Technical reference for v4.1.6 labels
- **`docs/CLINICAL_COHERENCE_ANALYSIS.md`**: Full analysis of clinical validity issues
- **`docs/PBSI_THRESHOLD_ANALYSIS.md`**: Statistical rationale for percentile approach

### Updated Files

- **`src/labels/build_pbsi.py`**: Implementation with percentile logic
- **`README.md`**: (to be updated) Mention v4.1.6 improvements
- **Paper draft**: (to be updated) Add clinical validity disclaimer

---

## 🚀 Migration Guide

### For Existing Analyses

**Option 1: Adopt v4.1.6 (Recommended)**

```bash
# Re-run pipeline with new defaults
make pipeline PID=P000001 SNAPSHOT=2025-11-07

# Results will have balanced classes
# All stages (0-9) will complete successfully
```

**Option 2: Keep v4.1.5 Behavior**

```python
# In Python scripts
df_labeled = build_pbsi_labels(
    unified_df,
    use_percentile_thresholds=False,  # Use old thresholds
    threshold_low_fixed=-0.5,
    threshold_high_fixed=0.5
)
```

**Option 3: Custom Thresholds**

```python
# For specific research questions
df_labeled = build_pbsi_labels(
    unified_df,
    threshold_low_percentile=0.10,  # More extreme low
    threshold_high_percentile=0.90  # More extreme high
)
```

### For Papers / Reports

**Update Terminology**:

- ❌ "periods of stability/instability"
- ✅ "periods of physiological regulation/dysregulation"

**Add Disclaimer** (suggested text):

> The PBSI labels (low/mid/high) represent composite physiological indices derived from sleep, cardiovascular, and activity patterns. These labels have not been validated against psychiatric ground truth (mood diaries, clinician ratings, or DSM-5 diagnostic criteria) and should not be interpreted as direct proxies for psychiatric states (mania, depression, ADHD severity). They are exploratory constructs suitable for hypothesis generation, requiring prospective validation with ecological momentary assessments (EMA) and clinical interviews.

---

## 🔬 Scientific Rationale

### Why Percentile-Based?

1. **Participant-Specific**: Each N-of-1 participant has unique physiological ranges
2. **Data-Driven**: Thresholds emerge from actual distribution, not arbitrary values
3. **Reproducible**: P25/P75 are well-established statistical quantiles
4. **ML-Friendly**: Ensures trainable class sizes for supervised learning

### Why Rename Labels?

1. **Clinical Accuracy**: "Stable" implies psychiatric state (not measured)
2. **Transparency**: "Low PBSI" describes what's computed, not what's inferred
3. **Ethics**: Avoids misleading clinical interpretations
4. **Scientific Integrity**: Acknowledges limitations of exploratory indices

### Future Validation (v5.x)

Planned for v5.0.0:

- **Ground Truth Collection**: Retrospective mood diary + medication timeline
- **Clinical Mapping**: Validate PBSI patterns against documented psychiatric states
- **State-Specific Biomarkers**: Separate indices for mania, depression, ADHD
- **Multi-Participant**: Test generalizability (N=10-20)

---

## 🐛 Known Issues

1. **Segment Info Loading**: `segment_autolog.csv` format (date_start/date_end) not yet fully integrated

   - **Workaround**: Currently uses global z-scores
   - **Fix**: Planned for v4.1.7

2. **Lint Warnings**: Type checking errors in label_name.get()
   - **Impact**: None (cosmetic only)
   - **Fix**: Will suppress in next patch

---

## 🙏 Acknowledgments

This release addresses critical feedback about clinical coherence and class balancing. Special thanks to:

- Scientific rigor > convenience
- Transparent limitations > overstated claims
- Data-driven thresholds > arbitrary values

---

## 📚 References

- **Internal Docs**:

  - `docs/PBSI_LABELS_v4.1.6.md` - Technical specs
  - `docs/CLINICAL_COHERENCE_ANALYSIS.md` - Validity analysis
  - `docs/PBSI_THRESHOLD_ANALYSIS.md` - Threshold rationale

- **Code**:
  - `src/labels/build_pbsi.py` - Implementation
  - `src/etl_pipeline.py` - Stage 3 integration

---

**Version**: 4.1.6  
**Release Date**: 2025-11-20  
**Status**: Production-ready, clinical validation pending
