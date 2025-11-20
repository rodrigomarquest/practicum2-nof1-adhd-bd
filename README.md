N-of-1 Study – ADHD + Bipolar Disorder (Practicum Part 2)

![Last Updated](https://img.shields.io/badge/Last%20Updated-October 2025-brightgreen.svg)

Author: Rodrigo Marques Teixeira
Supervisor: Dr. Agatha Mattos
Course: MSc Artificial Intelligence for Business – National College of Ireland
Student ID: 24130664
Period: Jan 2023 – Jan 2026

📘 About the Project

This repository consolidates the full technical and ethical framework developed during Practicum Part 2 of the MSc in Artificial Intelligence for Business at NCI.
It integrates data pre-processing (ETL), feature engineering, time-series modeling (baselines + LSTM), SHAP explainability, and complete GDPR/ethics documentation.

The project extends the previous Practicum (Part 1) phase and focuses on reprocessing and modelling data from an N-of-1 longitudinal study on comorbidity ADHD + Bipolar Disorder, collected from wearable sensors (Apple Health, Amazfit GTR 4, Helio Ring) and self-reports (EMA / State of Mind).

� Notebooks (v4.1.5)

**NEW**: Canonical Jupyter notebooks for reproducible analysis!

| Notebook                                                   | Purpose                     | Runtime | Figures        |
| ---------------------------------------------------------- | --------------------------- | ------- | -------------- |
| [NB0_DataRead.ipynb](notebooks/NB0_DataRead.ipynb)         | Pipeline readiness check    | <5s     | -              |
| [NB1_EDA.ipynb](notebooks/NB1_EDA.ipynb)                   | 8-year exploratory analysis | 30-60s  | Fig 3, 4       |
| [ML6_Baseline.ipynb](notebooks/ML6_Baseline.ipynb)         | Logistic regression results | 10-20s  | Fig 5, Table 3 |
| [ML7_DeepLearning.ipynb](notebooks/ML7_DeepLearning.ipynb) | LSTM evaluation             | 15-30s  | Fig 6, Table 3 |

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

Pipeline Stages:

- Stage 0: Ingest (Apple Health + Zepp extraction)
- Stage 1: CSV Aggregation (daily metrics)
- Stage 2: Unify Daily (canonical features)
- Stage 3: Apply Labels (3-class classification)
- Stage 4: Segmentation (period boundaries)
- Stage 5: Prep ML6 (anti-leak safeguards)
- Stage 6: ML6 Training (Logistic Regression, temporal CV)
- Stage 7: ML7 Analysis (SHAP + Drift + LSTM)
- Stage 8: TFLite Export (model conversion + latency)
- Stage 9: Generate Report (RUN_REPORT.md)

Outputs

features_daily_unified.csv (2828 rows, 2017-2025)

cv_summary.json (6 folds with dates, F1 scores)

shap_summary.md (top-10 global features)

drift_report.md (ADWIN + KS drift detection)

best_model.tflite (LSTM M1 exported)

Includes z-score scaling, missing-value handling, and segment normalization.

🧩 iOS Screen Time / Usage Extraction

A dedicated sub-module (ios_extract/) automates the pipeline from encrypted local iTunes backup → decrypted Manifest → Screen Time/KnowledgeC data.

ios_extract/
├─ decrypt_manifest.py # decrypts Manifest.db and validates SQLite
├─ quick_post_backup_probe.py # probes blobs present (flags=1)
├─ extract_plist_screentime.py # extracts DeviceActivity & ScreenTimeAgent plists
├─ smart_extract_plists.py # adaptive extractor (multi-strategy)
├─ plist_to_usage_csv.py # parses plists → usage_daily_from_plists.csv
├─ extract_knowledgec.py # extracts CoreDuet/KnowledgeC.db when present
└─ parse_knowledgec_usage.py # parses KnowledgeC.db → usage_daily_from_knowledgec.csv

Typical workflow

python decrypt_manifest.py
python quick_post_backup_probe.py
python smart_extract_plists.py
python plist_to_usage_csv.py

Expected outputs

decrypted_output/
├─ Manifest_decrypted.db
├─ screentime_plists/
│ ├─ DeviceActivity.plist
│ ├─ ScreenTimeAgent.plist
│ └─ usage_daily_from_plists.csv
└─ knowledgec/
└─ KnowledgeC.db → usage_daily_from_knowledgec.csv

If only plists exist, the CSV may be empty (settings-only snapshot).
Once KnowledgeC.db appears, the parser aggregates daily app-level usage.

Dependencies

python -m pip install iphone-backup-decrypt==0.9.0 pycryptodome

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
# Run QC verification
python -m src.etl.hr_daily_aggregation_consistency_check P000001 2025-11-07 \
    --start-date 2024-01-01 --end-date 2024-03-01

# Outputs:
#   data/ai/{PID}/{SNAPSHOT}/qc/hr_daily_aggregation_diff.csv
#   data/ai/{PID}/{SNAPSHOT}/qc/hr_daily_aggregation_consistency_report.md
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

🔷 Modeling and Explainability
Notebook Focus Output
01_feature_engineering.ipynb Daily feature aggregation 97 features (27 engineered)
02_model_training.ipynb Baselines (Naïve, MA, LogReg) + LSTM M1–M3 best_model.tflite + metrics
03_shap_analysis.ipynb SHAP drift + top-5 features shap_top5_features.csv
04_rule_based_baseline.ipynb Clinical heuristic baseline Rule-based comparison

Temporal CV: 6 folds (4 m train / 2 m val)
Metrics: F1-macro/weighted, AUROC-OvR, Balanced ACC, Cohen κ, McNemar p.

🧠 Ethics and Data Protection

No identifiable data committed.

Raw exports encrypted locally.

/docs contains consent forms and data management plan.

GDPR + NCI Ethics compliance with anonymisation.

Phases

Self-data (Apple Health, Amazfit, Helio Ring).

Future opt-in collection from family/friends with consent.

🧮 Version Control and Tagging
Tag Description
v2.0-pre-ethics ETL + docs before ethics approval
v2.1-ethics-approved Ethics approved revision
v2.2-modeling-complete Final models and SHAP results
v2.3-final-report CA3 / dissertation submission
🧭 Execution Timeline (Oct → Jan)
Week Milestone
W1 – Oct 2025 KnowledgeC integration + ETL update
W2 – Nov 2025 Model retraining + LSTM drift analysis
W3 – Dec 2025 Final LaTeX report + appendices
Jan 2026 Viva defense and repository archival
🌐 Citation

Teixeira, R. M. (2025). N-of-1 Study – ADHD + Bipolar Disorder (Practicum Part 2).
National College of Ireland. GitHub repository: https://github.com/rodrigomarques/practicum2-nof1-adhd-bd

🔒 License

Licensed under CC BY-NC-SA 4.0 (Attribution-NonCommercial-ShareAlike).
Reuse allowed for academic purposes with proper credit.

📞 Contact

Rodrigo Marques Teixeira – MSc AI for Business (NCI)
Supervisor: Dr. Agatha Mattos
Email: x24130664@student.ncirl.ie

Version: v2.0-pre-ethics  Last updated: October 2025
