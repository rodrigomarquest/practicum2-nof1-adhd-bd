# Anti-Leak Columns Reference

**Date**: 2025-12-08  
**Purpose**: Document columns that MUST NOT be used as ML features to prevent target leakage

---

## Overview

Target leakage occurs when information about the target variable "leaks" into the feature set, artificially inflating model performance and creating models that fail in production.

For the SoM-centric pipeline, we must carefully separate:

- **Features**: Signals available BEFORE the outcome is known
- **Targets**: What we're trying to predict (SoM)
- **Intermediate outputs**: Calculations that encode target information

---

## Anti-Leak Column Categories

### Category 1: Direct Targets

These ARE the targets - never use as features:

| Column                | Description                     | Reason                  |
| --------------------- | ------------------------------- | ----------------------- |
| `som_category_3class` | SoM target (-1, 0, +1)          | **Is the target**       |
| `som_binary`          | SoM binary (0=rest, 1=unstable) | **Derived from target** |

### Category 2: PBSI Intermediate Outputs

These are calculated FROM features using formulas that could encode arbitrary patterns:

| Column         | Description                   | Reason                        |
| -------------- | ----------------------------- | ----------------------------- |
| `pbsi_quality` | Quality flag from PBSI labels | Derived from label_3cls       |
| `sleep_sub`    | PBSI sleep subscore           | Intermediate PBSI calculation |
| `cardio_sub`   | PBSI cardio subscore          | Intermediate PBSI calculation |
| `activity_sub` | PBSI activity subscore        | Intermediate PBSI calculation |

### Category 3: PBSI Labels (Legacy)

These were the OLD targets from PBSI-centric pipeline:

| Column           | Description              | Reason                           |
| ---------------- | ------------------------ | -------------------------------- |
| `label_3cls`     | PBSI 3-class label       | Old target, encodes PBSI outcome |
| `label_2cls`     | PBSI binary label        | Derived from label_3cls          |
| `label_clinical` | Clinical threshold label | Derived from PBSI                |

### Category 4: Metadata (Not Features)

These are identifiers/metadata, not predictive signals:

| Column       | Description        | Reason                             |
| ------------ | ------------------ | ---------------------------------- |
| `date`       | Calendar date      | Identifier, not a feature          |
| `segment_id` | Treatment segment  | Used for CV splits, not prediction |
| `som_vendor` | Source of SoM data | QC metadata                        |

---

## Special Case: `pbsi_score`

**`pbsi_score` is NOT in the anti-leak list.**

Rationale:

- It's a composite wellness index derived from raw features
- We want to test whether it adds predictive value beyond raw features
- It's included in **FS-D** (Feature Set D) for ablation experiments
- The ablation study will show if PBSI helps or not

From ML6 ablation (P000001/2025-12-08):

- FS-B (no PBSI): F1 = 0.4623
- FS-D (with PBSI): F1 = 0.4623

**Finding**: PBSI adds no value beyond FS-B features for SoM prediction.

---

## Implementation

### Anti-Leak Constants

```python
# In ml6_som_experiments.py and stage_5_prep_ml6()

ANTI_LEAK_COLUMNS = [
    # PBSI intermediate outputs
    'pbsi_quality',
    'sleep_sub',
    'cardio_sub',
    'activity_sub',
    # PBSI labels (old targets)
    'label_3cls',
    'label_2cls',
    'label_clinical',
]

# Note: pbsi_score is NOT here - it's a candidate feature in FS-D

TARGET_COLUMNS = [
    'som_category_3class',
    'som_binary',
]

METADATA_COLUMNS = [
    'date',
    'segment_id',
    'som_vendor',
]
```

### Feature Extraction Logic

```python
def get_feature_columns(df: pd.DataFrame, feature_set: str) -> List[str]:
    """
    Get feature columns for a given feature set, ensuring no leakage.
    """
    all_cols = set(df.columns)

    # Remove anti-leak columns
    excluded = set(ANTI_LEAK_COLUMNS + TARGET_COLUMNS + METADATA_COLUMNS)
    candidate_features = all_cols - excluded

    # Select based on feature set
    fs_features = FEATURE_SETS[feature_set]['features']
    return [f for f in fs_features if f in candidate_features]
```

---

## Verification Checklist

Before training any ML model, verify:

- [ ] `som_category_3class` not in X
- [ ] `som_binary` not in X
- [ ] `pbsi_quality` not in X
- [ ] `sleep_sub`, `cardio_sub`, `activity_sub` not in X
- [ ] `label_3cls`, `label_2cls` not in X
- [ ] If using FS-A/B/C, `pbsi_score` not in X

---

## References

- `src/etl/ml6_som_experiments.py`: Feature set definitions
- `src/etl/ml7_analysis.py`: ML7_SOM_ANTI_LEAK_COLS
- `scripts/run_full_pipeline.py`: stage_5_prep_ml6() anti-leak logic
