# NB3 Pipeline — Quick Reference

## Overview

**NB3** implements:

1. **Logistic Regression** across 6 calendar-based folds (same CV as NB2)
2. **SHAP Explainability**: Top-5 features per fold + global importance ranking
3. **Drift Detection**:
   - **ADWIN** (δ=0.002): Detects changes in per-sample loss within validation folds
   - **KS tests** (p<0.01): Feature distribution shifts at segment boundaries
   - **SHAP drift** (>10%): Per-segment SHAP importance change
4. **LSTM M1**: Sequence-to-label model, best fold exported as TFLite
5. **Latency Profiling**: 200 TFLite inference runs, compute mean/p50/p95/std

## Data Requirements

- **Input**: `data/etl/features_daily_labeled.csv` (from NB2)
- **Columns required**:
  - `date` (YYYY-MM-DD, datetime)
  - `label_3cls` or `label_2cls` (class labels)
  - `segment_id` (for drift detection; optional but recommended)
  - 27+ numeric feature columns (Apple + Zepp merged)

## Quick Start

### Run full NB3 pipeline

```bash
make nb3-run
```

### Or directly

```bash
python scripts/run_nb3_pipeline.py \
  --csv data/etl/features_daily_labeled.csv \
  --outdir nb3 \
  --label_col label_3cls
```

### Run NB2 → NB3 end-to-end

```bash
make nb3-all
```

## Output Structure

```
nb3/
├── models/
│   └── best_model.tflite         # TFLite-quantized LSTM (best by F1-macro)
├── plots/
│   ├── shap_top5_fold1.png       # Per-fold SHAP bar charts (≤6)
│   ├── shap_top5_fold2.png
│   ├── adwin_fold1.png           # ADWIN changepoint visualizations (≤6)
│   ├── adwin_fold2.png
│   └── ...
├── shap_summary.md               # Per-fold top-5 + global SHAP importance
├── drift_report.md               # ADWIN findings + KS hits + segment drift
├── lstm_report.md                # Best fold, F1-macro, latency summary
└── latency_stats.json            # {mean_ms, p50_ms, p95_ms, std_ms, runs: 200}
```

## Output Files in Detail

### shap_summary.md

- Per-fold SHAP top-5 features (mean |SHAP| values)
- Global ranking across all folds
- Interpretation: higher values = more important for Logistic predictions

### drift_report.md

- **ADWIN**: # change points detected per fold (streaming loss monitoring)
- **KS tests**: Features with p<0.01 at segment boundaries
- **SHAP drift**: Implicit via per-fold top-5 (can be expanded for exact segment comparison)

### lstm_report.md

- Best fold ID (highest F1-macro on validation)
- F1-macro score for best fold
- TFLite path and latency statistics:
  - Mean inference time (ms)
  - Median (P50) and 95th percentile (P95)
  - Standard deviation across 200 runs

### latency_stats.json

```json
{
  "runs": 200,
  "mean_ms": 3.45,
  "p50_ms": 3.4,
  "p95_ms": 4.2,
  "std_ms": 0.32
}
```

## Configuration Options

| Option          | Default                               | Description                  |
| --------------- | ------------------------------------- | ---------------------------- |
| `--csv`         | `data/etl/features_daily_labeled.csv` | Input labeled CSV            |
| `--outdir`      | `nb3`                                 | Output directory             |
| `--label_col`   | `label_3cls`                          | Label column name            |
| `--date_col`    | `date`                                | Date column name             |
| `--segment_col` | `segment_id`                          | Segment column (drift tests) |
| `--seq_len`     | `14`                                  | LSTM sequence length (days)  |

## Key Parameters

### Logistic Regression

- **Penalty**: L2 (Ridge)
- **C**: 1.0 (inverse regularization strength)
- **Class weight**: balanced (auto-weight for imbalanced classes)
- **Solver**: liblinear (efficient for binary/small multiclass)

### ADWIN (Drift)

- **Delta**: 0.002 (drift significance threshold)
- **Loss**: 1 - P(true_class) per sample
- Output: indices where concept drift detected

### KS Test (Distribution Shift)

- **Threshold**: p < 0.01 (reject null hypothesis of equal distributions)
- **Scope**: Segment boundaries (e.g., clinical phases, stimulation periods)

### LSTM Architecture

```
Input (seq_len=14, n_features=27)
  → LSTM(32 units, return_sequences=False)
  → Dense(32, ReLU)
  → Dropout(0.2)
  → Dense(n_classes, softmax)
```

### LSTM Training

- **CV**: Same 6 calendar folds as NB2
- **Early Stopping**: patience=10, monitor='val_accuracy'
- **Optimizer**: Adam
- **Loss**: sparse_categorical_crossentropy
- **Best Model**: Highest F1-macro on validation fold

## Dependencies

```python
sklearn              # LogisticRegression, StandardScaler, metrics
scipy.stats          # ks_2samp (KS test)
river                # ADWIN drift detector
shap                 # LinearExplainer for SHAP
tensorflow           # Keras LSTM, TFLite conversion
matplotlib           # Plotting
pandas, numpy        # Data handling
```

Install all:

```bash
pip install -r requirements/base.txt
```

## Interpretation Guide

### SHAP Top-5

- **Feature appears in top-5 across multiple folds**: Robust predictor
- **High SHAP value**: Large contribution to model output (positive or negative)
- **Global ranking**: Aggregated importance across all 6 folds

### ADWIN Changes

- **No changes detected**: Stable loss during validation period
- **Multiple change points**: Possible distribution shift or concept drift
- **Action**: If significant drift, consider online learning or model retraining

### KS Test Hits

- **p < 0.01**: Feature distribution significantly different across segment boundary
- **Common causes**: Clinical phase transition, data collection method change, intervention
- **Action**: Feature engineering (e.g., segment-specific scaling) may help

### Latency (TFLite)

- **Mean < 5ms**: Suitable for real-time mobile/embedded deployment
- **P95**: Expected latency for 95% of inferences
- **Std**: Variability (ideally low for predictable performance)

## Determinism & Reproducibility

All operations use fixed seeds:

- `np.random.seed(42)`
- `tf.random.set_seed(42)` (for LSTM)
- ADWIN and KS tests are deterministic (no randomness)
- Rerun produces identical outputs (same data, same parameters)

## Common Issues

### "Missing column 'X'"

- Check `--date_col`, `--label_col`, `--segment_col` arguments
- Verify CSV structure: `python -c "import pandas as pd; df = pd.read_csv('data/etl/features_daily_labeled.csv'); print(df.columns.tolist())"`

### "Not enough data for seq_len=14"

- Some folds have <14 rows after splitting
- Reduce `--seq_len` (e.g., `--seq_len 7`)
- Extend training data range or adjust calendar fold boundaries

### LSTM training hangs

- Check TensorFlow installation: `python -c "import tensorflow as tf; print(tf.__version__)"`
- Use small test run first: `--seq_len 7` on subset

### TFLite latency very high (>50ms)

- Normal for CPU-only inference on large models
- Consider: quantization, model compression, or hardware acceleration

## Next Steps

After NB3 completes:

1. **Validate outputs**:

   ```bash
   ls -la nb3/
   python -c "import json; print(json.load(open('nb3/latency_stats.json')))"
   ```

2. **Review markdown reports**:

   ```bash
   cat nb3/shap_summary.md
   cat nb3/drift_report.md
   ```

3. **Compare with NB2 baselines**:

   - LSTM F1-macro vs Logistic F1 from NB2
   - Latency profile for production feasibility

4. **Integration**:
   - Export TFLite model for mobile app
   - Deploy SHAP insights to clinical dashboard
   - Monitor for drift in production (ADWIN callback)

## References

- **SHAP**: Lundberg & Lee (2017) "A Unified Approach to Interpreting Model Predictions"
- **ADWIN**: Bifet & Gavaldà (2007) "Learning from Time-Changing Data"
- **TFLite**: https://www.tensorflow.org/lite
