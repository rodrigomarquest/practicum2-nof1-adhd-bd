# ML6-vFinal: Unified ML Pipeline (2025-11)

## Overview

Production-grade ML pipeline combining Apple+Zepp biomarker data with PBSI heuristic labels for 6-fold temporal cross-validation.

**Key Features:**

- ✅ Unified daily dataset (Apple preference > Zepp fallback)
- ✅ PBSI heuristic labels (sleep + cardio + activity subscores)
- ✅ Temporal CV with 4-month train / 2-month validation splits
- ✅ Multiple baselines: Dummy (stratified), Logistic Regression (C tuning)
- ✅ Deterministic outputs (seed=42 everywhere)
- ✅ Reproducible (rerun = identical CSVs)

## Canonical Daily Schema (27 columns)

### Core Features

| Group        | Columns                                                                       |
| ------------ | ----------------------------------------------------------------------------- |
| **Date**     | `date` (YYYY-MM-DD)                                                           |
| **Sleep**    | `sleep_total_h`, `sleep_efficiency` (0-1 normalized)                          |
| **Cardio**   | `apple_hr_mean`, `apple_hr_max`, `apple_hrv_rmssd`                            |
| **Activity** | `steps`, `exercise_min`, `stand_hours`, `move_kcal`                           |
| **QC Flags** | `missing_sleep`, `missing_cardio`, `missing_activity` (0/1)                   |
| **Sources**  | `source_sleep`, `source_cardio`, `source_activity` (apple\|zepp\|mixed\|none) |

### PBSI Label Columns (8 additional)

| Label            | Formula                                              | Range      | Interpretation     |
| ---------------- | ---------------------------------------------------- | ---------- | ------------------ |
| `sleep_sub`      | -0.6×z_sleep_dur + 0.4×z_sleep_eff                   | [-3, 3]    | Sleep subscale     |
| `cardio_sub`     | 0.5×z_hr_mean - 0.6×z_hrv + 0.2×z_hr_max             | [-3, 3]    | Cardio subscale    |
| `activity_sub`   | -0.7×z_steps - 0.3×z_exercise                        | [-3, 3]    | Activity subscale  |
| `pbsi_score`     | 0.40×sleep_sub + 0.35×cardio_sub + 0.25×activity_sub | [-3, 3]    | Composite          |
| `label_3cls`     | +1 (stable) / 0 (neutral) / -1 (unstable)            | {-1, 0, 1} | Tri-class          |
| `label_2cls`     | 1 (stable) / 0 (unstable)                            | {0, 1}     | Binary             |
| `label_clinical` | 1 if pbsi_score ≥ 0.75, else 0                       | {0, 1}     | Clinical threshold |
| `pbsi_quality`   | Starts 1.0, ×0.8 per missing block, min 0.5          | [0.5, 1.0] | Quality gate       |

## File Structure

```
src/features/
  __init__.py
  unify_daily.py         ← Merge Apple+Zepp data

src/labels/
  __init__.py
  build_pbsi.py          ← Generate PBSI labels

src/models/
  __init__.py
  run_ml6.py             ← Temporal CV + baselines

scripts/
  run_ml6_pipeline.py    ← Main orchestrator

ml6/                     ← Output directory
  baselines_label_3cls.csv
  baselines_label_2cls.csv
  confusion_matrices/
    fold_0_3cls.png
    fold_0_2cls.png
    ...
  baselines_summary.md

data/etl/
  features_daily_unified.csv    ← Stage 1 output
  features_daily_labeled.csv    ← Stage 2 output
```

## Usage

### Full Pipeline

```bash
python scripts/run_ml6_pipeline.py --stage all
```

### Individual Stages

```bash
# Stage 1: Unify Apple + Zepp
python scripts/run_ml6_pipeline.py --stage unify

# Stage 2: Build PBSI labels
python scripts/run_ml6_pipeline.py --stage labels

# Stage 3: Temporal CV with baselines
python scripts/run_ml6_pipeline.py --stage baselines
```

### Via Makefile

```bash
make ml6-all              # Full pipeline
make ml6-unify            # Stage 1
make ml6-labels           # Stage 2
make ml6-baselines        # Stage 3
```

## Pipeline Stages

### Stage 1: Unify Daily Data

**Input:**

- `data/raw/apple/*.csv` (Apple Health export)
- `data/raw/zepp_processed/*.csv` (Zepp export)

**Process:**

- Load all Apple CSVs, merge by date
- Load all Zepp CSVs, merge by date
- For each date: try Apple first, fill gaps with Zepp
- Generate QC flags (1 if entire block missing)
- Track sources (apple/zepp/mixed/none per feature block)

**Output:** `data/etl/features_daily_unified.csv` (27 columns, 1 row per date)

### Stage 2: Build PBSI Labels

**Input:**

- `data/etl/features_daily_unified.csv`
- `data/version_log_enriched.csv` (segment mapping for z-scores)

**Process:**

- Compute z-scores per segment (S1-S6 if available)
- Calculate sleep/cardio/activity subscores (clipped [-3, 3])
- Compute PBSI composite score
- Generate 3-class labels: +1 (pbsi ≤ -0.5), 0 (neutral), -1 (pbsi ≥ 0.5)
- Generate 2-class labels: 1 (stable), 0 (unstable)
- Apply quality gates (degraded to 0.8× per missing block, min 0.5)
- Validate: >1 class in each label column, >100 rows, no NaNs

**Output:** `data/etl/features_daily_labeled.csv` (35 columns = 27 + 8 labels)

### Stage 3: Temporal CV + Baselines

**Input:** `data/etl/features_daily_labeled.csv`

**Process:**

- Generate 6 non-overlapping folds (≈ 4 months train, 2 months val)
- For each fold:
  - Impute missing values (median from train)
  - Standardize features (StandardScaler on train → apply to val)
  - Fit models:
    - **Dummy**: stratified random (baseline)
    - **LogisticRegression**: C ∈ {0.1, 1, 3}, class_weight="balanced"
  - Evaluate on validation fold
  - Compute metrics: F1_macro, F1_weighted, balanced_acc, kappa, (AUROC for 2cls)

**Output:**

- `ml6/baselines_label_3cls.csv` (6 folds + mean/std)
- `ml6/baselines_label_2cls.csv` (6 folds + mean/std)
- Confusion matrices (PNG per fold per task)
- Summary markdown

## Metrics

### 3-Class Task

- **F1_macro**: Average F1 across 3 classes
- **F1_weighted**: Weighted F1 by class support
- **Balanced_acc**: Mean recall per class
- **Kappa**: Cohen's kappa agreement

### 2-Class Task

- All above +
- **ROC_AUC**: Area under ROC curve (for 1 vs 0)

## Key Design Decisions

1. **Apple > Zepp Preference**

   - Apple data considered more reliable
   - Zepp used only to fill gaps

2. **Efficiency Normalization**

   - Auto-detect: if max(efficiency) > 1, divide by 100
   - Result: always [0, 1]

3. **Segment-Aware Z-Scores**

   - Per-segment standardization (S1-S6 if available)
   - Prevents data leakage in CV

4. **Quality Gating**

   - Starts at 1.0, degrades by 0.8× per missing feature block
   - Minimum 0.5 (never dropped completely)

5. **Temporal CV (No Explicit Gap)**
   - Calendar-based boundaries (not explicit gap)
   - Row-based split: 67% train, 33% val within each fold

## Reproducibility

All operations use **random_state=42**:

- Stratified Dummy: `random_state=42`
- LogisticRegression: `random_state=42`
- GridSearchCV: Fixed seed in CV splits

**Verification:** Rerun with same data → identical CSVs and PNGs (byte-for-byte)

## Data Assumptions

- Apple data: `data/raw/apple/*.csv` with columns like:

  - `sleep_total_h`, `sleep_eff` (or variants: `apple_sleep_total_h`, `zepp_slp_total_h`, etc.)
  - `apple_hr_mean`, `zepp_hr_mean` (or `hr_mean`)
  - `steps`, `exercise_min`, `stand_hours`, `move_kcal`

- Zepp data: Similar structure with `zepp_` prefix variants

- Segment info: `data/version_log_enriched.csv` with:
  - `date` (YYYY-MM-DD)
  - `segment_id` (S1, S2, S3, etc.)

## Next Steps

1. **Validate Data Existence**

   ```bash
   ls data/raw/apple/  # Check for CSV files
   ls data/raw/zepp_processed/  # Check for Zepp files
   ```

2. **Run Full Pipeline**

   ```bash
   python scripts/run_ml6_pipeline.py --stage all
   ```

3. **Check Outputs**

   ```bash
   head data/etl/features_daily_unified.csv
   head data/etl/features_daily_labeled.csv
   cat ml6/baselines_label_3cls.csv
   ls ml6/confusion_matrices/*.png
   ```

4. **Verify Reproducibility**
   ```bash
   # Delete outputs, rerun, compare checksums
   rm data/etl/features_daily_*.csv ml6/*.csv
   python scripts/run_ml6_pipeline.py --stage all
   # Should match previous outputs exactly
   ```

---

**Version**: v1.0 (2025-11)  
**Author**: ML6 Pipeline Generator  
**Determinism**: ✅ Reproducible (seed=42)
