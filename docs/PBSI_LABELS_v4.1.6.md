# PBSI Labels - Technical Reference (v4.1.6)

**Version**: 4.1.6  
**Date**: November 20, 2025  
**Status**: Production (with clinical validation pending)

---

## Overview

The **Physio-Behavioral Stability Index (PBSI)** is a composite score derived from wearable sensor data that quantifies physiological regulation patterns. It combines sleep, cardiovascular, and activity metrics into a single daily index.

**Key Changes in v4.1.6**:
- ✅ Percentile-based thresholds (P25/P75) for balanced class distribution
- ✅ Renamed labels to descriptive terms (low_pbsi/mid_pbsi/high_pbsi)
- ✅ Added clinical validity disclaimers
- ✅ Maintained backward compatibility with v4.1.5

---

## Label Definitions (v4.1.6)

### 3-Class Labels (`label_3cls`)

| Value | Name | Interpretation | PBSI Range |
|-------|------|---------------|------------|
| **+1** | `low_pbsi` (regulated) | Physiologically regulated patterns: good sleep, high HRV, active | pbsi ≤ P25 (~-0.12) |
| **0** | `mid_pbsi` (typical) | Typical physiological patterns within participant's normal range | P25 < pbsi < P75 |
| **-1** | `high_pbsi` (dysregulated) | Physiologically dysregulated patterns: poor sleep, low HRV, sedentary | pbsi ≥ P75 (~+0.17) |

### 2-Class Labels (`label_2cls`)

| Value | Name | Interpretation |
|-------|------|---------------|
| **1** | Regulated | Physiologically regulated (corresponds to label_3cls = +1) |
| **0** | Not regulated | Typical or dysregulated (corresponds to label_3cls = 0 or -1) |

---

## PBSI Computation

### Formula

```python
pbsi_score = 0.40 * sleep_sub + 0.35 * cardio_sub + 0.25 * activity_sub
```

### Subscores

#### Sleep Subscore
```python
sleep_sub = -0.6 * z_sleep_duration + 0.4 * z_sleep_efficiency
```
- More sleep + better efficiency → lower subscore (regulated)
- Less sleep + poor efficiency → higher subscore (dysregulated)

#### Cardio Subscore
```python
cardio_sub = 0.5 * z_hr_mean - 0.6 * z_hrv + 0.2 * z_hr_max
```
- Lower HR + higher HRV → lower subscore (regulated)
- Higher HR + lower HRV → higher subscore (dysregulated)

#### Activity Subscore
```python
activity_sub = -0.7 * z_steps - 0.3 * z_exercise_minutes
```
- More steps + more exercise → lower subscore (regulated)
- Less steps + less exercise → higher subscore (dysregulated)

---

## Sign Convention

**Lower PBSI = More Regulated** (counterintuitive but by design)

- **pbsi ≤ -0.12** (low): Well-rested, high HRV, active → **+1 (regulated)**
- **-0.12 < pbsi < +0.17** (mid): Normal variance → **0 (typical)**
- **pbsi ≥ +0.17** (high): Poor sleep, low HRV, sedentary → **-1 (dysregulated)**

---

## Thresholds (v4.1.6)

### Percentile-Based (Default)

Thresholds adapt to participant's data distribution:

```python
threshold_low = pbsi_scores.quantile(0.25)   # P25
threshold_high = pbsi_scores.quantile(0.75)  # P75
```

**For P000001 (2025-11-07)**:
- P25 = -0.117
- P75 = +0.172

**Resulting Distribution**:
- Low PBSI (regulated): 25% of days (n=707)
- Mid PBSI (typical): 50% of days (n=1,414)
- High PBSI (dysregulated): 25% of days (n=707)

### Fixed Thresholds (Legacy, v4.1.5)

Original thresholds (deprecated but available via flag):

```python
threshold_low = -0.5
threshold_high = +0.5
```

**Problem**: Too strict for most participants, results in extreme class imbalance (93% neutral).

---

## Clinical Validity Status

### ⚠️ Important Disclaimers

**PBSI labels are NOT clinically validated**:

1. **No Ground Truth**: Labels not validated against:
   - Mood diaries / self-reports
   - Clinician ratings (YMRS, MADRS, ASRS)
   - Diagnostic interviews (DSM-5/ICD-11)

2. **Not Psychiatric States**: PBSI does NOT map to:
   - Mania/hypomania (BD)
   - Depression (BD/MDD)
   - ADHD symptom severity
   - Euthymia vs episode

3. **Composite Index**: PBSI is a physiological composite that may not capture:
   - Subjective mood states
   - Cognitive symptoms (attention, concentration)
   - Life context (stress, medication changes)

### Appropriate Uses

✅ **Valid for**:
- Exploratory analysis of physiological variance
- Feature engineering for ML models
- Identifying periods of physiological deviation
- Hypothesis generation for future validation

❌ **Invalid for**:
- Clinical diagnosis or treatment decisions
- Direct interpretation as psychiatric states
- Substitute for clinical assessment

---

## Future Validation (v5.x)

### Planned Enhancements

1. **Ground Truth Collection**:
   - Daily mood tracking (EMA)
   - Retrospective clinical timeline
   - Medication/life event annotations

2. **Clinical Mapping**:
   - Validate patterns against psychiatric states
   - Identify state-specific biomarkers
   - Develop early warning algorithms

3. **Multi-Participant Validation**:
   - Test generalizability (N=10-20)
   - Cross-participant pattern analysis
   - Population norms for ADHD/BD

---

## Usage Examples

### In Python

```python
from src.labels.build_pbsi import build_pbsi_labels

# Load unified data
df = pd.read_csv('data/etl/P000001/2025-11-07/joined/features_daily_unified.csv')

# Build labels with percentile thresholds (v4.1.6)
df_labeled = build_pbsi_labels(
    unified_df=df,
    version_log_path=Path('data/etl/P000001/2025-11-07/segment_autolog.csv'),
    output_path=Path('data/etl/P000001/2025-11-07/joined/features_daily_labeled.csv'),
    use_percentile_thresholds=True,  # v4.1.6 default
    threshold_low_percentile=0.25,   # P25
    threshold_high_percentile=0.75,  # P75
)

# Access labels
print(df_labeled['label_3cls'].value_counts())
# +1 (low_pbsi):  707 (25%)
# 0 (mid_pbsi):   1414 (50%)
# -1 (high_pbsi): 707 (25%)
```

### In Pipeline

```bash
# Run labeling stage (uses v4.1.6 defaults)
make label PID=P000001 SNAPSHOT=2025-11-07

# Use legacy fixed thresholds (not recommended)
python src/etl_pipeline.py label \
  --participant P000001 \
  --snapshot 2025-11-07 \
  --use-fixed-thresholds
```

---

## Interpretation Guidelines

### When Analyzing Results

1. **Report as Physiological Patterns**:
   - ✅ "Periods of physiological dysregulation (high PBSI)"
   - ❌ "Periods of instability" (vague)
   - ❌ "Manic episodes" (unvalidated)

2. **Cross-Reference with Context**:
   - Medication changes
   - Life events (travel, stress)
   - Device changes (firmware updates)
   - Data quality issues

3. **Acknowledge Limitations**:
   - In papers: Include clinical validity disclaimer
   - In presentations: Clarify exploratory nature
   - To clinicians: Do not use for treatment decisions

---

## References

### Internal Docs
- `docs/CLINICAL_COHERENCE_ANALYSIS.md` - Full clinical validity analysis
- `docs/PBSI_THRESHOLD_ANALYSIS.md` - Threshold selection rationale
- `docs/ETL_ARCHITECTURE_COMPLETE.md` - Pipeline integration

### Code
- `src/labels/build_pbsi.py` - Implementation (v4.1.6)
- `src/etl_pipeline.py` - Stage 3 integration
- `config/label_rules.yaml` - Configuration (deprecated for PBSI)

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| v4.1.6 | 2025-11-20 | Percentile thresholds (P25/P75), renamed labels, clinical disclaimers |
| v4.1.5 | 2025-11-07 | Fixed thresholds (±0.5), original "stable/unstable" labels |
| v4.1.0 | 2025-10-15 | Segment-wise z-scores, PBSI formula finalized |

---

**For Questions**: Contact research team or refer to `docs/CLINICAL_COHERENCE_ANALYSIS.md`
