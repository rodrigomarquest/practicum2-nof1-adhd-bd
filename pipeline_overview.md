N-of-1 ADHD + Bipolar – Wearable-Based Digital Phenotyping

**Pipeline Version**: v4.1.5  
**Last Updated**: November 20, 2025

## 1. Purpose & Scope

This document provides a high-level overview of the deterministic end-to-end data pipeline used in the N-of-1 ADHD + Bipolar Disorder digital phenotyping study.

### Goals

- Integrate multi-modal wearable data (Apple Health, Zepp/Amazfit) into a single, **100% reproducible** pipeline
- Produce daily, participant-level features suitable for time-series modeling and clinical interpretation
- Support **research reproducibility**: ETL transparency, feature provenance, and modeling traceability (NB0–NB3)
- Enable **deterministic execution** with fixed random seeds across all stages

The focus of this document is on what each stage does and how the pieces connect. For implementation details, see `docs/ETL_ARCHITECTURE_COMPLETE.md`.

## 2. Data Sources

### 2.1 Participants & Study Design

- **Design**: N-of-1 longitudinal study (8 years)
- **Primary participant**: P000001 (self)
- **Timeline**: December 2017 – October 2025 (~2,828 days)
- **Segments**: 119 behavioral segments detected automatically via calendar boundaries + gaps >1 day

### 2.2 Sensors and Platforms

**Apple Health** (iPhone + Apple Watch)

- Heart Rate (HR)
- Heart Rate Variability (HRV)
- Sleep stages and durations
- Activity (steps, distance, active energy)
- Screen Time (iOS usage)
- State of Mind / mood check-ins (iOS 17+)

**Zepp / Amazfit** (GTR 4)

- Heart Rate / HRV
- Sleep (total duration, stages, sleep score)
- Activity (steps, PAI score, workouts)
- Stress metrics

**Metadata**

- Device context logs (timezone changes, device transitions)
- Segmentation autologs (119 segments with calendar boundaries)
- Pipeline provenance (ETL manifests, QC reports)

## 3. Repository Structure (v4.1.5)

```
practicum2-nof1-adhd-bd/
├── data/
│   ├── raw/                              # Original exports (not in git)
│   ├── etl/P000001/2025-11-07/          # Snapshot-based ETL outputs
│   │   ├── extracted/                    # Stage 0-1: Per-metric daily CSVs
│   │   ├── joined/                       # Stage 2-4: Unified features + segments
│   │   └── qc/                           # Quality control reports
│   └── ai/P000001/2025-11-07/           # Modeling outputs
│       ├── nb2/                          # Stage 6: Logistic regression
│       └── nb3/                          # Stage 7: LSTM models
├── notebooks/
│   ├── NB0_DataRead.ipynb               # Stage detection & readiness
│   ├── NB1_EDA.ipynb                    # Exploratory data analysis
│   ├── NB2_Baseline.ipynb               # Baseline model results
│   ├── NB3_DeepLearning.ipynb           # LSTM evaluation
│   └── archive/                         # Deprecated notebooks
├── src/
│   ├── etl_pipeline.py                  # Stage 0: Extraction orchestrator
│   ├── make_labels.py                   # PBSI label construction
│   └── models_nb2.py, models_nb3.py     # Modeling scripts
├── scripts/
│   └── run_full_pipeline.py             # Main orchestrator (10 stages)
├── config/
│   ├── settings.yaml                    # Pipeline configuration
│   ├── label_rules.yaml                 # PBSI threshold definitions
│   └── participants.yaml                # Participant metadata
├── docs/
│   ├── notebooks_overview.md            # Canonical notebooks guide
│   ├── ETL_ARCHITECTURE_COMPLETE.md     # Technical architecture
│   ├── QUICK_REFERENCE.md               # Pipeline cheat sheet
│   ├── latex/                           # Research paper sources
│   └── copilot/                         # AI-assisted development docs
└── Makefile                             # Convenience targets
```

## 4. Snapshot Philosophy

All ETL operations are run per **snapshot** (a consistent view of data at a given date):

- **Path structure**: `data/etl/P000001/2025-11-07/...`
- **Immutability**: All downstream files are frozen for that snapshot
- **New data = new snapshot**: Never overwriting previous snapshots
- **Canonical snapshot**: `2025-11-07` is the reference for v4.1.5 release

This supports:

- **100% reproducibility** for research validation
- **Temporal analysis** across 8 years
- **Behavioral segment comparison** (119 segments)

## 5. Pipeline Stages (0-9)

The pipeline consists of **10 deterministic stages** orchestrated by `scripts/run_full_pipeline.py`:

| Stage | Name          | Input            | Output                       | Description                               |
| ----- | ------------- | ---------------- | ---------------------------- | ----------------------------------------- |
| 0     | **Ingest**    | `data/raw/`      | `extracted/`                 | Extract from Apple Health XML + Zepp ZIPs |
| 1     | **Aggregate** | Raw CSVs         | `daily_*.csv`                | Aggregate to daily per-metric files       |
| 2     | **Unify**     | Per-metric CSVs  | `features_daily_unified.csv` | Merge all metrics by date                 |
| 3     | **Label**     | Unified features | `features_daily_labeled.csv` | Apply PBSI labels (3-class)               |
| 4     | **Segment**   | Labeled features | `segment_autolog.csv`        | Detect behavioral segments (2 rules)      |
| 5     | **Prep-NB2**  | Segments         | Training arrays              | Prepare data with anti-leak safeguards    |
| 6     | **NB2**       | Arrays           | `data/ai/.../nb2/`           | Train logistic regression (6-fold CV)     |
| 7     | **NB3**       | Sequences        | `data/ai/.../nb3/`           | Train LSTM models (sequence)              |
| 8     | **TFLite**    | NB3 model        | `model.tflite`               | Export to mobile format                   |
| 9     | **Report**    | All outputs      | `RUN_REPORT.md`              | Generate execution summary                |

**Segmentation Rules** (Stage 4):

1. Calendar boundaries (month/year transitions)
2. Gaps greater than 1 day

See `docs/ETL_ARCHITECTURE_COMPLETE.md` for detailed stage specifications.

## 6. Key Features by Domain

### 6.1 Sleep Domain

- `sleep_total_h` - Total sleep duration (hours)
- `sleep_deep_h`, `sleep_rem_h`, `sleep_light_h` - Sleep stage durations
- `sleep_onset_time` - Sleep start time (minutes from midnight)
- `sleep_midpoint_time` - Sleep midpoint (circadian anchor)
- `sleep_efficiency` - Ratio of time asleep / time in bed

**Sources**: Apple Sleep Analysis + Zepp Sleep Summary

### 6.2 Cardiovascular Domain

- `hr_mean`, `hr_std`, `hr_min`, `hr_max` - Heart rate statistics
- `hrv_rmssd`, `hrv_sdnn` - HRV time-domain metrics (when available)
- `hr_n` - Number of HR measurements per day

**Sources**: Apple HealthKit HR + Zepp Cardio

### 6.3 Activity Domain

- `steps` - Daily step count
- `active_energy_kcal` - Active energy expenditure
- `distance_km` - Total distance traveled
- `exercise_min` - Minutes of exercise

**Sources**: Apple Activity + Zepp Activity

### 6.4 Screen Time Domain

- `screen_min` - Total screen time (minutes)
- `screen_pickups` - Number of device pickups

**Sources**: Apple Screen Time (iOS)

### 6.5 Behavioral Stability Index (PBSI)

- `pbsi_score` - Composite stability score (-1 to +1)
- `pbsi_label` - 3-class label (stable=+1, neutral=0, unstable=-1)

**Construction**: Segment-wise z-scored composite of sleep regularity, HR variability, activity consistency. See `config/label_rules.yaml` for threshold definitions.

7. Label Construction

Labels are built in a dedicated labeling stage to ensure clear separation between features and outcomes.

7.1 Label Sources

EMA / clinical mood scales, when present.

Apple “State of Mind” entries (daily mood).

Zepp / Helio Ring Emotion exports (when API/export is available).

7.2 Heuristic Mapping

A rule-based mapping compresses raw mood/affect into a ternary label:

+1 – positive / “better than usual”

0 – neutral / “about the same”

-1 – negative / “worse than usual”

For example:

High negative affect (or low mood score) → -1.

Clear positive affect → +1.

Anything in the “middle band” → 0.

This rule-based mapping is documented in Appendix D (feature glossary & label rules).

7.3 Final Label Attachment

Labels are joined on date (and optionally participant_id).

Output: features_daily_labeled.csv, used by NB2/NB3.

8. Modeling Pipelines (NB1–NB3)
   8.1 NB1 – Exploratory Data Analysis (EDA)

Purpose: Understand basic properties of the dataset before modeling.

Inputs

features_daily.csv (without labels; can also use labeled version).

Key Outputs

nb1_eda_summary.md – narrative summary (ranges, missingness, distributions).

nb1_feature_stats.csv – per-feature statistics.

Quick plots:

Time series of key features (steps, HR, sleep).

Histograms / boxplots for distribution sanity checks.

8.2 NB2 – Baseline Models (Daily Classification)

Purpose: Provide interpretable baselines for daily mood/label prediction.

Inputs

features_daily_labeled.csv (one row per day, with label).

Cross-Validation

Temporal CV with 6 folds:

Each fold: 4 months training / 2 months validation.

Strict calendar boundaries (no leakage).

All folds respect temporal order.

Baselines Implemented

Dummy (Sklearn stratified)

Random predictions respecting class distribution.

Naïve Yesterday

Predict today’s label as yesterday’s.

Fallback to 7-day modal label when previous day missing.

Moving Avg 7-day

Use rolling window over last 7 days.

Quantize mean to ternary label thresholds (±0.33).

Rule-based Clinical

Uses a composite score (e.g., pbsi_score) derived from mood-related items.

Thresholds:

≥ 0.5 → -1

≤ -0.5 → +1

else → 0

Logistic Regression

Multi-class (one-vs-rest).

L2 penalty, C ∈ {0.1, 1, 3} via grid search.

Class-weighted, solver liblinear.

Metrics: F1 (macro/weighted), AUROC (OvR), Balanced Accuracy, Cohen κ, McNemar p.

Outputs

Fold-wise metrics and aggregated tables (nb2_results.csv).

Confusion matrices per fold.

Feature importance (coefficients) for logistic regression.

8.3 NB3 – Sequence Models & Drift (LSTM / CNN1D)

Purpose: Model temporal dependencies and evaluate performance under concept drift.

Inputs

Sequence windows built from features_daily_labeled.csv (e.g., 14-day windows).

Preprocessed tensors with:

Segment-aware standardization.

Train/val splits respecting temporal CV.

Models

LSTM / GRU sequence classifiers.

Optional 1D-CNN configurations.

Evaluation

Same 6-fold temporal CV as NB2.

Early stopping (patience=10 on validation loss).

Drift & robustness:

ADWIN (δ = 0.002) for stream-based drift detection.

KS tests at version / segment boundaries.

SHAP-based drift: >10% change in feature importance flagged.

## 7. Canonical Notebooks (v4.1.5)

The pipeline includes 4 canonical Jupyter notebooks for reproducible analysis:

### NB0 – Data Readiness (`notebooks/NB0_DataRead.ipynb`)

- **Purpose**: Pipeline stage detection and file verification
- **Runtime**: <5 seconds
- **Outputs**: Status table, missing stage hints, quick inventory

### NB1 – Exploratory Analysis (`notebooks/NB1_EDA.ipynb`)

- **Purpose**: 8-year behavioral pattern analysis
- **Runtime**: 30-60 seconds
- **Key visualizations**: Temporal trends, missingness, distributions, segments (119)
- **Paper figures**: Fig 3(a,b), Fig 4

### NB2 – Baseline Models (`notebooks/NB2_Baseline.ipynb`)

- **Purpose**: Logistic regression performance evaluation
- **Model**: L2-regularized logistic regression
- **CV**: 5-fold calendar-based (chronological splits)
- **Performance**: Macro F1 ~0.81
- **Paper figures**: Fig 5, Table 3

### NB3 – Deep Learning (`notebooks/NB3_DeepLearning.ipynb`)

- **Purpose**: LSTM sequence model evaluation
- **Architecture**: BiLSTM(64) + Dense(32) + softmax(3)
- **Input**: 14-day windows, 7 features
- **Performance**: Macro F1 ~0.79-0.87 (fold-dependent)
- **Paper figures**: Fig 6, Table 3

See `docs/notebooks_overview.md` for complete documentation.

## 8. Orchestration

### 8.1 Main Pipeline Script

```bash
# Run full pipeline (stages 0-9)
python -m scripts.run_full_pipeline \
  --participant P000001 \
  --snapshot 2025-11-07 \
  --start-stage 0 \
  --end-stage 9
```

### 8.2 Makefile Targets

```bash
make etl-extract    # Stage 0: Extract from raw/
make etl-all        # Stages 0-4: Full ETL
make nb2            # Stage 6: Train logistic regression
make nb3            # Stage 7: Train LSTM
make notebooks      # Open Jupyter notebooks
```

## 9. Quality Control & Reproducibility

### 9.1 Determinism Guarantees

**Fixed Seeds** across all stages:

- Python: `random.seed(42)`
- NumPy: `np.random.seed(42)`
- TensorFlow: `tf.random.set_seed(42)`
- Sklearn: `random_state=42`

**Result**: 100% bit-exact reproducibility across runs

### 9.2 QC Reports

Generated per stage in `data/etl/.../qc/`:

- `qc_report.json` - Dataset statistics, missingness, outliers
- ETL provenance logs with file checksums
- Segment detection logs (119 segments)

### 9.3 Anti-Leak Safeguards

**Segment-wise Normalization** (Stage 5):

- StandardScaler fit independently per segment
- Prevents data leakage across behavioral boundaries
- Critical for temporal CV validity

**Calendar-based CV**:

- Chronological fold splits (no shuffling)
- Strict temporal boundaries
- Test data always in future relative to train

### 9.4 Pipeline Provenance

All outputs include:

- Snapshot ID (`2025-11-07`)
- Pipeline version (`v4.1.5`)
- Execution timestamp
- Input file checksums (SHA-256)

See `RUN_REPORT.md` (generated by Stage 9) for complete execution summary.

## 10. Key Design Decisions

### 10.1 Two Segmentation Rules Only

Behavioral segments are detected using exactly 2 rules:

1. **Calendar boundaries**: Month/year transitions
2. **Gaps > 1 day**: Missing data spans

**Rationale**: Simple, deterministic, data-driven (no subjective "behavioral shifts")

### 10.2 PBSI Label Construction

**Method**: Segment-wise z-scored composite of:

- Sleep regularity (coefficient of variation)
- HR variability (within-day std)
- Activity consistency (day-to-day correlation)

**Thresholds** (from `config/label_rules.yaml`):

- PBSI ≥ 0.5 → Unstable (-1)
- PBSI ≤ -0.5 → Stable (+1)
- Otherwise → Neutral (0)

**Advantage**: Weak supervision without requiring daily clinical labels

### 10.3 Snapshot Immutability

Once a snapshot is created, it is **never modified**:

- New data → new snapshot
- Enables reproducible research
- Supports temporal comparison

## 11. Known Limitations & Future Work

### Limitations

1. **Single participant**: N-of-1 design limits generalizability
2. **Weak supervision**: PBSI labels from heuristics (not clinical gold standard)
3. **Missing data**: ~15-30% missingness in some features
4. **Long-term shifts**: 8-year timeline includes life events, device changes

### Future Work

1. **Multi-participant**: Extend to P000002, P000003 (federated learning)
2. **Clinical validation**: Compare predictions with clinical assessments (PHQ-9, MDQ)
3. **Feature engineering**: Add circadian rhythm features, sleep architecture
4. **Explainability**: SHAP values, attention weights for LSTM
5. **Real-time**: Deploy TFLite model for on-device inference

## 12. Documentation Index

- **This file** (`pipeline_overview.md`): High-level architecture
- `docs/notebooks_overview.md`: Canonical notebooks guide (470 lines)
- `docs/ETL_ARCHITECTURE_COMPLETE.md`: Technical ETL specifications
- `docs/QUICK_REFERENCE.md`: Pipeline cheat sheet
- `docs/latex/main.tex`: Research paper (100% code-paper aligned)
- `docs/copilot/`: AI-assisted development documentation (53 files)

---

**Version**: v4.1.5  
**Last Updated**: November 20, 2025  
**Status**: ✅ Production-ready, publication-ready
