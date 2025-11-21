# Reproducing the Pipeline from Public ETL Snapshot

This document explains how external researchers can reproduce the deterministic N-of-1 ADHD+BD pipeline using the **public ETL snapshot** included in this repository, without requiring access to private raw data.

## Overview

The pipeline consists of 10 stages (0-9):

| Stage | Name      | Description                                    | Raw Data Required?     |
| ----- | --------- | ---------------------------------------------- | ---------------------- |
| **0** | Ingest    | Extract Apple Health export.xml and Zepp ZIPs  | ✅ YES                 |
| **1** | Aggregate | Parse XML/CSVs → daily feature CSVs            | ✅ YES                 |
| **2** | Unify     | Join all daily features into single dataframe  | ⚠️ Uses stage 1 output |
| **3** | Label     | Apply PBSI behavioral labels from diary        | ❌ NO                  |
| **4** | Segment   | Detect behavioral periods (stable/treatment)   | ❌ NO                  |
| **5** | Prep ML6  | Clean and prepare data for supervised learning | ❌ NO                  |
| **6** | ML6       | Train supervised baseline models (RF, XGBoost) | ❌ NO                  |
| **7** | ML7       | Train LSTM sequence models                     | ❌ NO                  |
| **8** | TFLite    | Export models to TensorFlow Lite               | ❌ NO                  |
| **9** | Report    | Generate markdown reports and provenance       | ❌ NO                  |

**Stages 0-2** require raw data (Apple Health export, Zepp wearable ZIPs), which are **not included** in the public repository due to privacy constraints.

**Stages 3-9** operate on the processed ETL snapshot and can be **fully reproduced** by external researchers.

## Public ETL Snapshot

The repository includes a single public ETL snapshot for reproducibility:

```
data/etl/P000001/2025-11-07/
├── extracted/                 # Daily CSVs from stage 1
│   ├── daily_steps.csv
│   ├── daily_heart_rate.csv
│   ├── daily_sleep.csv
│   └── ...
├── joined/                    # Unified dataframes from stage 2
│   ├── features_daily.csv
│   ├── features_daily_labeled.csv  # After stage 3
│   ├── features_ml6_clean.csv      # After stage 5
│   └── ...
└── qc/                        # Quality control reports
    ├── missing_data_report.csv
    └── ...
```

This snapshot contains:

- **Participant**: P000001
- **Snapshot date**: 2025-11-07
- **Pipeline version**: v4.1.8
- **Date range**: 2023-12-01 to 2025-09-29 (669 days)
- **Features**: 47 daily features (cardio, activity, sleep, screen time)
- **Labels**: PBSI behavioral annotations (stable/crisis periods)

## Reproducing the Full Pipeline (Stages 3-9)

### Prerequisites

1. **Clone the repository**:

   ```bash
   git clone https://github.com/your-org/practicum2-nof1-adhd-bd.git
   cd practicum2-nof1-adhd-bd
   ```

2. **Set up Python environment** (Python 3.10+):

   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   pip install -r requirements/base.txt
   ```

3. **Verify ETL snapshot exists**:
   ```bash
   ls -la data/etl/P000001/2025-11-07/extracted/
   ls -la data/etl/P000001/2025-11-07/joined/
   ```

### Run the Pipeline from ETL Snapshot

Use the `--start-from-etl` flag to skip raw data extraction (stages 0-2):

```bash
python -m scripts.run_full_pipeline \
  --participant P000001 \
  --snapshot 2025-11-07 \
  --start-from-etl \
  --end-stage 9
```

This will execute:

- ✅ Stage 3: Apply PBSI labels
- ✅ Stage 4: Segment behavioral periods
- ✅ Stage 5: Prepare ML6 dataset
- ✅ Stage 6: Train supervised models (RF, XGBoost, SVM)
- ✅ Stage 7: Train LSTM sequence models
- ✅ Stage 8: Export TensorFlow Lite models
- ✅ Stage 9: Generate provenance reports

### Expected Output

After successful execution, you will find:

```
data/ai/P000001/2025-11-07/
├── ml6/                          # Supervised learning outputs
│   ├── rf_model.pkl              # Random Forest
│   ├── xgb_model.pkl             # XGBoost
│   ├── svm_model.pkl             # SVM
│   ├── ml6_metrics.json          # Accuracy, F1, etc.
│   └── ml6_confusion_matrix.png
├── ml7/                          # LSTM sequence outputs
│   ├── lstm_model.h5             # Trained LSTM
│   ├── ml7_metrics.json          # Loss, accuracy curves
│   └── ml7_predictions.csv
├── tflite/                       # Mobile deployment
│   ├── rf_model.tflite
│   └── lstm_model.tflite
└── reports/
    ├── RUN_REPORT.md             # Full execution log
    └── etl_provenance_report.csv # Data lineage
```

### Reproducing Individual Stages

You can also run specific stages only:

```bash
# Only label and segment (stages 3-4)
python -m scripts.run_full_pipeline \
  --participant P000001 \
  --snapshot 2025-11-07 \
  --start-from-etl \
  --start-stage 3 \
  --end-stage 4

# Only ML6 modeling (stage 6)
python -m scripts.run_full_pipeline \
  --participant P000001 \
  --snapshot 2025-11-07 \
  --start-from-etl \
  --start-stage 6 \
  --end-stage 6

# Only generate reports (stage 9)
python -m scripts.run_full_pipeline \
  --participant P000001 \
  --snapshot 2025-11-07 \
  --start-from-etl \
  --start-stage 9 \
  --end-stage 9
```

## Understanding the ETL Snapshot Structure

### extracted/

Contains daily-aggregated CSVs from raw data sources:

| File                      | Source          | Features                                |
| ------------------------- | --------------- | --------------------------------------- |
| `daily_steps.csv`         | Apple Health    | steps, distance                         |
| `daily_heart_rate.csv`    | Zepp wearable   | resting_hr, avg_hr, max_hr              |
| `daily_sleep.csv`         | Zepp wearable   | sleep_duration, deep_sleep%, awakenings |
| `daily_active_energy.csv` | Apple Health    | active_calories                         |
| `daily_screen_time.csv`   | iOS Screen Time | total_minutes, pickups                  |

Each CSV has columns: `date`, `participant`, `<feature_name>`.

### joined/

Contains unified dataframes ready for modeling:

| File                         | Description                                      |
| ---------------------------- | ------------------------------------------------ |
| `features_daily.csv`         | All 47 features joined by date (stage 2 output)  |
| `features_daily_labeled.csv` | With PBSI behavioral labels (stage 3 output)     |
| `features_ml6_clean.csv`     | Cleaned for ML (MICE imputation, stage 5 output) |

### qc/

Quality control reports for data provenance:

- `missing_data_report.csv`: Per-feature missingness rates
- `outlier_report.csv`: Detected outliers (IQR method)
- `adwin_changes.csv`: ADWIN drift detection results

## Determinism and Reproducibility

The pipeline is **fully deterministic** from stage 3 onwards:

1. **Random seeds**: Fixed in `config/settings.yaml` (seed=42)
2. **MICE imputation**: `random_state=42` for missingness
3. **ML models**: All use `random_state=42` (scikit-learn, XGBoost)
4. **LSTM training**: `tf.random.set_seed(42)` in ML7 stage
5. **Train/test split**: Temporal split (no randomness)

Running the same ETL snapshot should produce **identical results** across machines.

## Limitations

### What Cannot Be Reproduced

- **Stages 0-1**: Raw data extraction/aggregation (requires private exports)
- **Stage 2**: Unification (uses stage 1 outputs, already in ETL snapshot)
- **Custom snapshots**: Only `2025-11-07` snapshot is public

### Generating New Snapshots

To create your own ETL snapshot from raw data:

```bash
# Full pipeline from scratch (requires raw data)
python -m scripts.run_full_pipeline \
  --participant P000001 \
  --snapshot 2025-12-01 \
  --start-stage 0 \
  --end-stage 9 \
  --zepp-password YOUR_PASSWORD
```

This requires:

- `data/raw/P000001/apple/export/*.zip` - Apple Health export
- `data/raw/P000001/zepp/*.zip` - Zepp wearable exports

## Troubleshooting

### Error: "ETL snapshot not found"

```bash
[FATAL] --start-from-etl specified but ETL snapshot not found:
  Expected: data/etl/P000001/2025-11-07
```

**Solution**: Ensure you've cloned the repository correctly. The public ETL snapshot should be tracked in Git. Check:

```bash
git status
ls -la data/etl/P000001/2025-11-07/
```

### Error: "Raw data required for stage 0"

```bash
[FATAL] Raw data required for stage 0 but not found:
  Expected: data/raw/P000001
```

**Solution**: You're trying to run stage 0-2 without raw data. Use `--start-from-etl`:

```bash
python -m scripts.run_full_pipeline \
  --participant P000001 \
  --snapshot 2025-11-07 \
  --start-from-etl  # Add this flag
```

### Missing Python Dependencies

```bash
ModuleNotFoundError: No module named 'sklearn'
```

**Solution**: Install requirements:

```bash
pip install -r requirements/base.txt
```

## Citation

If you use this public ETL snapshot in your research, please cite:

```bibtex
@phdthesis{yourname2025,
  title={N-of-1 Digital Phenotyping for ADHD and Bipolar Disorder: A Multimodal Time Series Analysis},
  author={Your Name},
  year={2025},
  school={Your University}
}
```

## Contact

For questions about reproducing the pipeline:

- Open an issue: https://github.com/your-org/practicum2-nof1-adhd-bd/issues
- Email: your.email@university.edu

## Changelog

| Version | Date       | Changes                             |
| ------- | ---------- | ----------------------------------- |
| v4.1.8  | 2025-11-07 | Initial public ETL snapshot release |
