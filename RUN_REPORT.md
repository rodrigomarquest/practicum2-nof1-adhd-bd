# RUN_REPORT.md - Pipeline Execution Summary

**Generated**: 2025-11-22 02:46:42
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
- **Mean Macro-F1**: 0.6874 ± 0.1608

### Per-Fold Results

- **Fold 0** (2021-09-11 → 2021-11-11): F1=0.8559, BA=0.8674
- **Fold 1** (2022-03-11 → 2022-05-11): F1=0.6544, BA=0.8002
- **Fold 2** (2022-09-11 → 2022-11-11): F1=0.8674, BA=0.8725
- **Fold 3** (2023-03-11 → 2023-05-11): F1=0.7242, BA=0.7820
- **Fold 4** (2023-09-11 → 2023-11-11): F1=0.6334, BA=0.6728
- **Fold 5** (2024-03-11 → 2024-05-11): F1=0.3891, BA=0.5409

## ML7: Drift Detection

- **ADWIN Changes Detected (δ=0.002)**: 0
- **KS Significant Tests (p<0.05)**: 0

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
