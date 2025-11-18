# RUN_REPORT.md - Pipeline Execution Summary

**Generated**: 2025-11-18 22:30:49
**Participant**: P000001
**Snapshot**: 2025-11-07

## Data Summary

- **Date Range**: 2017-12-04 to 2025-10-21
- **Total Rows**: 2828
- **Missing Values**: 0

## Label Distribution

- **Label -1 (Unstable)**: 65 (2.3%)
- **Label +0 (Neutral)**: 2552 (90.2%)
- **Label +1 (Stable)**: 211 (7.5%)

## NB2: Logistic Regression (Temporal Calendar CV)

- **CV Type**: temporal_calendar_6fold
- **Train/Val**: 4mo / 2mo
- **Mean Macro-F1**: 1.0000 ± 0.0000

### Per-Fold Results

- **Fold 1** (2019-01-19 → 2019-03-19): F1=1.0000, BA=1.0000

## NB3: SHAP Feature Importance (Global Top-10)

1. **total_steps**: 1.7372
2. **total_distance**: 0.0000
3. **hr_mean**: 0.0000
4. **hr_std**: 0.0000
5. **hr_max**: 0.0000
6. **hr_min**: 0.0000
7. **sleep_hours**: 0.0000
8. **hr_samples**: 0.0000
9. **total_active_energy**: 0.0000
10. **sleep_quality_score**: 0.0000

## NB3: Drift Detection

- **ADWIN Changes Detected (δ=0.002)**: 5
- **KS Significant Tests (p<0.05)**: 102

## NB3: LSTM M1

- **Architecture**: LSTM(32) -> Dense(32) -> Dropout(0.2) -> Softmax(3)
- **Sequence Length**: 14 days
- **Mean Macro-F1**: 1.0000

### Per-Fold LSTM Results

- **Fold 1**: F1=1.0000, Loss=0.3481, Acc=1.0000

## Model Export & Latency

- **Best Model**: data\ai\P000001\2025-11-07\nb3\models\best_model.tflite
- **Inference Latency (p95)**: 0.00 ms

## Artifact Paths

- **Unified**: data\etl\P000001\2025-11-07\joined\features_daily_unified.csv
- **Labeled**: data\etl\P000001\2025-11-07\joined\features_daily_labeled.csv
- **NB2 Clean**: data\etl\P000001\2025-11-07\joined\features_nb2_clean.csv
- **Segments**: data\etl\P000001\2025-11-07\segment_autolog.csv
- **NB2 CV Summary**: data\ai\P000001\2025-11-07\nb2\cv_summary.json
- **SHAP Summary**: data\ai\P000001\2025-11-07\nb3\shap_summary.md
- **Drift Report**: data\ai\P000001\2025-11-07\nb3\drift_report.md
- **LSTM Report**: data\ai\P000001\2025-11-07\nb3\lstm_report.md
- **Latency Stats**: data\ai\P000001\2025-11-07\nb3\latency_stats.json

## Status

✅ **PIPELINE COMPLETE**
