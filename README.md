# N-of-1 Study – ADHD + Bipolar Disorder (Practicum Part 2)

![Version](https://img.shields.io/badge/Version-v4.1.8-blue.svg)
![Last Updated](https://img.shields.io/badge/Last%20Updated-November%202025-brightgreen.svg)
![Pipeline](https://img.shields.io/badge/Pipeline-v4.1.7-orange.svg)

**Author:** Rodrigo Marques Teixeira  
**Supervisor:** Dr. Agatha Mattos  
**Course:** MSc Artificial Intelligence for Business – National College of Ireland  
**Student ID:** 24130664  
**Period:** Jan 2023 – Jan 2026

## 📘 About the Project

This repository contains the complete deterministic N-of-1 pipeline for multimodal digital phenotyping in ADHD + Bipolar Disorder comorbidity, developed as part of Practicum Part 2 for the MSc in Artificial Intelligence for Business at NCI.

The pipeline integrates **8 years** of longitudinal data (2017-2025) from wearable sensors (Apple Health, Amazfit GTR 4, Helio Ring) and implements a fully reproducible ETL → ML6/ML7 workflow with segment-wise normalization, MICE imputation, and comprehensive drift analysis.

### Key Features

- ✅ **Deterministic ETL:** Fully reproducible from raw exports (seed=42)
- ✅ **ML6 Baseline:** Logistic regression with calendar-based 6-fold CV (Macro-F1: 0.69 ± 0.16)
- ✅ **ML7 LSTM:** 14-day sequence model with SHAP explainability (Macro-F1: 0.25)
- ✅ **Drift Analysis:** ADWIN (6 events) + KS tests (93 shifts) + SHAP monitoring
- ✅ **Public Reproducibility:** ETL snapshot enables stages 3-9 replication without raw data
- ✅ **Automated QC:** HR, sleep, activity validation with fabrication detection

### Dataset Summary

- **Total timeline:** 2,828 days (2017-12-04 to 2025-10-21)
- **ML-filtered:** 1,625 days (>= 2021-05-11, Amazfit-only era)
- **ML7 sequences:** 1,612 (14-day sliding windows)
- **Labels:** 322/596/707 (19.8% dysregulated / 36.7% typical / 43.5% regulated)
- **Segments:** 119 (full timeline), 48 (ML-filtered)
- **Missing data:** 0 (post-MICE imputation)

## 📊 Performance Results (v4.1.7)

| Model   | Macro-F1 | Std Dev | Best Fold | Worst Fold | Algorithm             |
| ------- | -------- | ------- | --------- | ---------- | --------------------- |
| **ML6** | **0.69** | ±0.16   | 0.87 (F2) | 0.39 (F5)  | Logistic Regression   |
| **ML7** | 0.25     | ±0.08   | -         | -          | LSTM (14-day windows) |

**Key Finding:** ML6 baseline significantly outperforms ML7 sequence model, consistent with literature showing simple models excel under high non-stationarity and weak labels in N-of-1 settings.

**Top Predictors (SHAP):**

1. Sleep efficiency
2. HRV RMSSD (autonomic variability)
3. Normalized heart rate mean

## 📓 Notebooks (v4.1.8)

**Canonical Jupyter notebooks for reproducible analysis:**

| Notebook                                                   | Purpose                     | Runtime | Outputs                 |
| ---------------------------------------------------------- | --------------------------- | ------- | ----------------------- |
| [NB0_DataRead.ipynb](notebooks/NB0_DataRead.ipynb)         | Pipeline readiness check    | <5s     | -                       |
| [NB1_EDA.ipynb](notebooks/NB1_EDA.ipynb)                   | 8-year exploratory analysis | 30-60s  | 9 figures (300 DPI)     |
| [ML6_Baseline.ipynb](notebooks/ML6_Baseline.ipynb)         | Logistic regression results | 10-20s  | CV metrics, SHAP        |
| [ML7_DeepLearning.ipynb](notebooks/ML7_DeepLearning.ipynb) | LSTM evaluation + drift     | 15-30s  | Sequence metrics, ADWIN |

See [📖 Notebooks Overview](docs/notebooks_overview.md) for detailed documentation.

**Quick Start**:

```bash
# Install dependencies
pip install -r requirements/base.txt

# Run full pipeline
python -m scripts.run_full_pipeline --participant P000001 --snapshot 2025-11-07

# Open notebooks
jupyter notebook notebooks/
```

�📊 Repository Structure (Canonical)

```
notebooks/
  NB0_DataRead.ipynb           # Stage detection & readiness
  NB1_EDA.ipynb                # Comprehensive 8-year EDA
  ML6_Baseline.ipynb           # Logistic regression results
  ML7_DeepLearning.ipynb       # LSTM evaluation
  archive/                     # Deprecated notebooks (pre-v4.1.5)
data/
  raw/                         # única fonte persistente (Apple Health exports, Zepp data)
  etl/                         # gerado; outputs canônicos (joined/qc/segment_autolog)
  ai/                          # gerado; artefatos ML6/ML7 por snapshot
src/
  etl/                         # ETL pipeline stages (aggregation, unify, labels)
  modeling/                    # ML6, ML7, LSTM training modules
  utils/                       # zip extraction, IO guards, paths, logger
scripts/
  run_full_pipeline.py         # orquestrador determinístico 10 stages (0-9)
docs/
  notebooks_overview.md        # 📖 Canonical notebooks guide (NEW)
  latex/                       # Research paper LaTeX sources
  copilot/                     # AI-assisted development documentation
  QUICK_REFERENCE.md           # Pipeline usage cheat sheet
  ETL_ARCHITECTURE_COMPLETE.md # Technical architecture
tests/
  test_etl_consistency.py
  test_anti_leak.py
  test_pipeline_reproducibility.py
config/
  participants.yaml            # configuração de participantes
  label_rules.yaml             # regras de rotulação
archive/                       # versões antigas preservadas (sem deleção)
```

⚙️ Reproducibility and Environment

### Full Pipeline Execution (With Raw Data)

```bash
# Install dependencies
pip install -r requirements/base.txt

# Run full deterministic pipeline (stages 0-9)
python scripts/run_full_pipeline.py --participant P000001 --snapshot 2025-11-07

# Outputs:
#   data/etl/P000001/2025-11-07/joined/features_daily_unified.csv
#   data/ai/P000001/2025-11-07/ml6/cv_summary.json
#   data/ai/P000001/2025-11-07/ml7/shap_summary.md
#   RUN_REPORT.md
```

### 🔓 Public Reproducibility (Without Raw Data)

**NEW**: External researchers can now reproduce the full pipeline using the **public ETL snapshot** included in this repository, without requiring access to private raw data!

```bash
# Reproduce stages 3-9 from public ETL snapshot
python scripts/run_full_pipeline.py \
  --participant P000001 \
  --snapshot 2025-11-07 \
  --start-from-etl \
  --end-stage 9
```

This command:

- ✅ Skips raw data extraction (stages 0-2)
- ✅ Loads pre-processed ETL snapshot from `data/etl/P000001/2025-11-07/`
- ✅ Executes stages 3-9: Labeling → Segmentation → ML6 → ML7 → TFLite → Report
- ✅ Produces **identical results** (deterministic pipeline, seed=42)

**Public ETL Snapshot Contents**:

- **Participant**: P000001
- **Date Range**: 2023-12-01 to 2025-09-29 (669 days)
- **Features**: 47 daily features (cardio, activity, sleep, screen time)
- **Pipeline Version**: v4.1.8 (scientific: v4.1.7)
- **Location**: `data/etl/P000001/2025-11-07/`
- **Access**: OneDrive for NCI evaluators (medical privacy constraints)

For detailed instructions, see: **[📖 Reproducing with ETL Snapshot](docs/REPRODUCING_WITH_ETL_SNAPSHOT.md)**

**Pipeline Stages (v4.1.7)**:

- **Stage 0**: Ingest (Apple Health + Zepp extraction)
- **Stage 1**: CSV Aggregation (daily metrics)
- **Stage 2**: Unify Daily (canonical features)
- **Stage 3**: Apply Labels (3-class classification)
- **Stage 4**: Segmentation (period boundaries)
- **Stage 5**: Prep ML6 (anti-leak safeguards)
- **Stage 6**: ML6 Training (Logistic Regression, temporal CV)
- **Stage 7**: ML7 Analysis (SHAP + Drift + LSTM)
- **Stage 8**: TFLite Export (model conversion + latency)
- **Stage 9**: Generate Report (RUN_REPORT.md)

**Pipeline Outputs (v4.1.7)**:

- `features_daily_unified.csv` – 2,828 rows (2017-12-04 to 2025-10-21)
- `cv_summary.json` – 6 temporal folds with dates, F1 scores
- `shap_summary.md` – Top-10 global feature importances
- `drift_report.md` – ADWIN + KS drift detection results (6 + 93 events)
- `best_model.tflite` – LSTM M1 exported for mobile deployment
- `RUN_REPORT.md` – Comprehensive pipeline execution summary

**Data Processing**: Z-score scaling, MICE imputation, segment normalization, anti-leak safeguards.

⚡ Performance Optimization: Parquet Caching

### Dual-Cache Strategy for Apple Health HR Parsing

**Problem**: Parsing 1.5 GB Apple Health `export.xml` with `ET.findall()` was slow (~5-10 minutes).

**Solution**: Binary regex streaming + dual Parquet caching (implemented Nov 2025).

#### Cache Architecture

```
First Run (3m7s):
  export.xml (1.5 GB)
    ↓ Binary regex streaming (~500 MB/sec)
    ├─ Event-level collection (4.6M HR records)
    ├─ Daily aggregation (1,315 days)
    └─ Save dual caches:
       ├─ export_apple_hr_events.parquet (23 MB, 4.6M records)  # QC verification
       └─ export_apple_hr_daily.parquet (30 KB, 1.3K days)      # Fast loading

Subsequent Runs (<1 second):
  Load export_apple_hr_daily.parquet directly
  → 180x speedup
```

#### Performance Results

| Metric              | Value                        |
| ------------------- | ---------------------------- |
| XML size            | 1.5 GB                       |
| Event cache         | 23.4 MB (4,677,088 records)  |
| Daily cache         | 30 KB (1,315 days)           |
| Compression ratio   | 64x (event), 50,000x (daily) |
| First run           | ~3 minutes                   |
| Cached run          | <1 second                    |
| **Overall speedup** | **180x**                     |

#### Quality Control (QC) Module

The event-level Parquet enables rigorous academic verification:

```bash
# Run QC verification (v4.1.7+)
python etl_tools/hr_daily_aggregation_consistency_check.py \
    --participant P000001 --snapshot 2025-11-07 \
    --start-date 2024-01-01 --end-date 2024-03-01

# Outputs:
#   data_etl/P000001/2025-11-07/qc/hr_daily_aggregation_diff.csv
#   data_etl/P000001/2025-11-07/qc/hr_daily_aggregation_consistency_report.md
```

**QC Workflow**:

1. Load event-level Parquet (4.6M HR records)
2. Re-aggregate to daily (reference "ground truth")
3. Load official `daily_cardio.csv` (fast pipeline output)
4. Compare metrics: `hr_mean`, `hr_samples`, `hr_min`, `hr_max`, `hr_std`
5. Apply thresholds: ±5 records, ±1 bpm, 5% relative difference
6. Generate comprehensive report with consistency statistics

**Verified Results** (60 days, Jan-Mar 2024):

- ✅ **100% consistency** for `hr_n_records` (0 mismatches)
- ✅ **100% consistency** for `hr_mean` (max diff: 2.8e-14 bpm - floating-point precision)
- ✅ Binary regex optimization maintains **full accuracy**

#### Cache Locations

```
data/etl/{PID}/{SNAPSHOT}/extracted/apple/apple_health_export/.cache/
  ├─ export_apple_hr_events.parquet   # Event-level (for QC re-aggregation)
  └─ export_apple_hr_daily.parquet    # Daily aggregates (for fast loading)
```

#### Dependencies

```bash
pip install pyarrow>=14.0.0  # Parquet support
```

## 🔷 Modeling and Explainability

### ML6 Logistic Regression Baseline

- **Architecture:** Regularized logistic regression (C=1.0, balanced class weights)
- **Cross-validation:** 6 folds, calendar-based (4 months train, 2 months test)
- **Features:** Single-day segment-normalized predictors (anti-leak safeguards)
- **Performance:** Macro-F1 = 0.69 ± 0.16 (range: 0.39-0.87)

### ML7 LSTM Sequence Model

- **Architecture:** Single-layer LSTM (64 units) + Dense head
- **Input:** 14-day sliding windows (1,612 sequences)
- **Temporal filter:** >= 2021-05-11 (Amazfit-only era, 1,625 days)
- **Performance:** Macro-F1 = 0.25, AUROC = 0.58

### Drift Analysis

- **ADWIN:** 6 long-term distributional shifts detected
- **KS tests:** 93 significant distribution changes (p < 0.05)
- **SHAP monitoring:** Feature importance evolution across temporal folds

**Temporal CV:** 6 folds (4 months train / 2 months validation)  
**Metrics:** F1-macro/weighted, AUROC-OvR, Balanced Accuracy, Confusion Matrix

🧠 Ethics and Data Protection

No identifiable data committed.

Raw exports encrypted locally.

/docs contains consent forms and data management plan.

GDPR + NCI Ethics compliance with anonymisation.

Phases

Self-data (Apple Health, Amazfit, Helio Ring).

Future opt-in collection from family/friends with consent.

## 🧮 Version Control and Releases

| Tag               | Description                                                | Date     |
| ----------------- | ---------------------------------------------------------- | -------- |
| **v4.1.8**        | **ML6/ML7 pipeline finalization & public reproducibility** | Nov 2025 |
| v4.1.7            | ML7 temporal filter + MICE imputation                      | Nov 2025 |
| v4.1.5            | Canonical notebooks restructure                            | Nov 2025 |
| v2.3-final-report | CA3 / dissertation submission                              | Jan 2026 |

**Latest Release:** [v4.1.8](https://github.com/rodrigomarquest/practicum2-nof1-adhd-bd/releases/tag/v4.1.8)

## 🧭 Project Timeline

| Phase       | Period            | Milestone                                  |
| ----------- | ----------------- | ------------------------------------------ |
| **Phase 1** | Jan-Jun 2023      | Initial data collection + Practicum Part 1 |
| **Phase 2** | Jul-Dec 2023      | ETL architecture + ethics approval         |
| **Phase 3** | Jan-Jun 2024      | ML6 baseline + feature engineering         |
| **Phase 4** | Jul-Oct 2024      | ML7 LSTM + drift analysis                  |
| **Phase 5** | Nov 2024-Jan 2025 | Dissertation writing + final validation    |
| **Defense** | Jan 2026          | Viva defense and repository archival       |

## 🌐 Citation

```bibtex
@mastersthesis{teixeira2025nof1,
  author  = {Teixeira, Rodrigo Marques},
  title   = {A Deterministic N-of-1 Pipeline for Multimodal Digital Phenotyping in ADHD and Bipolar Disorder},
  school  = {National College of Ireland},
  year    = {2025},
  type    = {MSc Dissertation},
  url     = {https://github.com/rodrigomarquest/practicum2-nof1-adhd-bd}
}
```

**APA Style:**  
Teixeira, R. M. (2025). _A Deterministic N-of-1 Pipeline for Multimodal Digital Phenotyping in ADHD and Bipolar Disorder_ [Master's dissertation, National College of Ireland]. GitHub. https://github.com/rodrigomarquest/practicum2-nof1-adhd-bd

## 🔒 License

Licensed under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) (Attribution-NonCommercial-ShareAlike).  
Reuse allowed for academic purposes with proper credit.

## 📞 Contact

**Rodrigo Marques Teixeira** – MSc AI for Business (NCI)  
**Supervisor:** Dr. Agatha Mattos  
**Email:** x24130664@student.ncirl.ie

---

**Version:** v4.1.8 | **Last updated:** November 2025 | **Pipeline:** v4.1.7
