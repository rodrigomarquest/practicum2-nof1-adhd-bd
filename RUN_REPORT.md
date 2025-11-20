# RUN_REPORT.md - Pipeline Execution Summary

**Generated**: 2025-11-20 07:40:28
**Participant**: P000001
**Snapshot**: 2025-11-07

## Data Summary

- **Date Range**: 2017-12-04 to 2025-10-21
- **Total Rows**: 2828
- **Missing Values**: 0

## Label Distribution

- **Label -1 (Dysregulated)**: 755 (26.7%)
- **Label +0 (Typical)**: 1366 (48.3%)
- **Label +1 (Regulated)**: 707 (25.0%)

## ML6: Logistic Regression (Temporal Calendar CV)

- **CV Type**: temporal_calendar_6fold
- **Train/Val**: 4mo / 2mo
- **Mean Macro-F1**: 0.7099 ± 0.0906

### Per-Fold Results

- **Fold 0** (2021-09-11 → 2021-11-11): F1=0.8156, BA=0.8259
- **Fold 1** (2022-03-11 → 2022-05-11): F1=0.6441, BA=0.7912
- **Fold 2** (2022-09-11 → 2022-11-11): F1=0.8539, BA=0.8601
- **Fold 3** (2023-03-11 → 2023-05-11): F1=0.6794, BA=0.7519
- **Fold 4** (2023-09-11 → 2023-11-11): F1=0.6471, BA=0.6835
- **Fold 5** (2024-03-11 → 2024-05-11): F1=0.6195, BA=0.8742

## ML7: SHAP Feature Importance (Global Top-10)

1. **z_sleep_total_h**: 0.7878
2. **z_sleep_efficiency**: 0.7300
3. **z_hrv_rmssd**: 0.6188
4. **z_hr_mean**: 0.3851
5. **z_steps**: 0.3747
6. **z_hr_max**: 0.2651
7. **z_exercise_min**: 0.2585

## ML7: Drift Detection

- **ADWIN Changes Detected (δ=0.002)**: 6
- **KS Significant Tests (p<0.05)**: 40

## ML7: LSTM M1

- **Architecture**: LSTM(32) -> Dense(32) -> Dropout(0.2) -> Softmax(3)
- **Sequence Length**: 14 days
- **Mean Macro-F1**: 0.4120

### Per-Fold LSTM Results

- **Fold 0**: F1=0.4464, Loss=0.8320, Acc=0.6279
- **Fold 1**: F1=0.4286, Loss=0.8518, Acc=0.7500
- **Fold 2**: F1=0.5238, Loss=1.0869, Acc=0.4286
- **Fold 3**: F1=0.2714, Loss=0.9756, Acc=0.3824
- **Fold 4**: F1=0.3351, Loss=0.8628, Acc=0.5610
- **Fold 5**: F1=0.4667, Loss=0.3821, Acc=0.8750

## Model Export & Latency

- **Best Model**: data\ai\P000001\2025-11-07\ml7\models\best_model.tflite
- **Inference Latency (p95)**: 0.00 ms

## Artifact Paths

- **Unified**: data\etl\P000001\2025-11-07\joined\features_daily_unified.csv
- **Labeled**: data\etl\P000001\2025-11-07\joined\features_daily_labeled.csv
- **ML6 Clean**: data\etl\P000001\2025-11-07\joined\features_ml6_clean.csv
- **Segments**: data\etl\P000001\2025-11-07\segment_autolog.csv
- **ML6 CV Summary**: data\ai\P000001\2025-11-07\ml6\cv_summary.json
- **SHAP Summary**: data\ai\P000001\2025-11-07\ml7\shap_summary.md
- **Drift Report**: data\ai\P000001\2025-11-07\ml7\drift_report.md
- **LSTM Report**: data\ai\P000001\2025-11-07\ml7\lstm_report.md
- **Latency Stats**: data\ai\P000001\2025-11-07\ml7\latency_stats.json

## Status

✅ **PIPELINE COMPLETE**
