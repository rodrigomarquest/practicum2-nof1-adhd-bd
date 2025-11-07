# RUN_REPORT.md - Pipeline Execution Summary

**Generated**: 2025-11-07 16:39:01
**Participant**: P000001
**Snapshot**: 2025-11-07

## Data Summary

- **Date Range**: 2017-12-04 to 2025-10-21
- **Total Rows**: 2828
- **Missing Values**: 0

## Label Distribution

- **Label -1 (Unstable)**: 1737 (61.4%)
- **Label +0 (Neutral)**: 284 (10.0%)
- **Label +1 (Stable)**: 807 (28.5%)

## NB2: Logistic Regression (Temporal Calendar CV)

- **CV Type**: temporal_calendar_6fold
- **Train/Val**: 4mo / 2mo
- **Mean Macro-F1**: 0.7066 ± 0.1699

### Per-Fold Results

- **Fold 1** (2021-09-16 → 2021-11-16): F1=0.7938, BA=0.7516
- **Fold 2** (2022-03-16 → 2022-05-16): F1=0.8611, BA=0.8611
- **Fold 3** (2022-09-16 → 2022-11-16): F1=0.8644, BA=0.8949
- **Fold 4** (2023-03-16 → 2023-05-16): F1=0.4433, BA=0.5833
- **Fold 5** (2023-09-16 → 2023-11-16): F1=0.5705, BA=0.5465

## NB3: SHAP Feature Importance (Global Top-10)

1. **sleep_quality_score**: 18.5418
2. **total_steps**: 11.0671
3. **total_active_energy**: 6.6438
4. **hr_samples**: 5.7602
5. **total_distance**: 5.2239
6. **hr_max**: 2.2310
7. **hr_mean**: 2.0244
8. **hr_min**: 1.5292
9. **sleep_hours**: 0.7664
10. **hr_std**: 0.3785

## NB3: Drift Detection

- **ADWIN Changes Detected (δ=0.002)**: 11
- **KS Significant Tests (p<0.05)**: 102

## NB3: LSTM M1

- **Architecture**: LSTM(32) -> Dense(32) -> Dropout(0.2) -> Softmax(3)
- **Sequence Length**: 14 days
- **Mean Macro-F1**: 0.2812

### Per-Fold LSTM Results

- **Fold 1**: F1=0.3655, Loss=1.0107, Acc=0.5208
- **Fold 2**: F1=0.2946, Loss=0.5493, Acc=0.7917
- **Fold 3**: F1=0.2029, Loss=1.2193, Acc=0.4375
- **Fold 4**: F1=0.3030, Loss=0.6400, Acc=0.8333
- **Fold 5**: F1=0.2400, Loss=1.0850, Acc=0.5625

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
