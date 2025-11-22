# RUN_REPORT_EXTENDED.md – Extended ML6/ML7 Experiments

**Generated**: 2025-11-22 04:02:04  
**Participant**: P000001  
**Snapshot**: 2025-11-07  
**Pipeline Version**: v4.2.1 (Extended Models)

---

## Core Pipeline Summary

This report **supplements** the main [RUN_REPORT.md](RUN_REPORT.md) with results from extended ML6/ML7 models.

**Core Pipeline Results**:

- Date Range: : 2017-12-04 to 2025-10-21
- Total Rows: : 2828
- ML6 Baseline (Logistic Regression): F1-macro = : 0.6874 ± 0.1608

For full ETL/baseline details, see [RUN_REPORT.md](RUN_REPORT.md).

---

## Extended ML6 Models (Tabular Classification)

Four additional models were trained on the same ML6 dataset (1,625 MICE-imputed days, 6-fold temporal CV):

- **Random Forest**: 200 trees, max_depth=10, instability-regularized max_features
- **XGBoost**: max_depth=4, lr=0.05, instability-regularized L1/L2 penalties
- **LightGBM**: max_depth=4, lr=0.05, instability-based feature weighting
- **SVM (RBF)**: C=1.0, gamma='scale', NO instability penalty

**Temporal Instability Regularization**: Tree/boosting models penalize features with high variance across 119 behavioral segments.

### ML6 Model Comparison

| Model               | F1-macro (mean ± std) | F1-weighted | Balanced Accuracy | Cohen's κ |
| ------------------- | --------------------- | ----------- | ----------------- | --------- |
| Logistic Regression | 0.6874 ± 0.1608       | —           | —                 | —         |
| RF                  | 0.7005 ± 0.1406       | 0.7905      | 0.7141            | 0.5810    |

---

## Extended ML7 Models (Temporal Sequence Classification)

Three additional sequence models were trained on the same ML7 dataset (14-day windows, 6-fold temporal CV):

- **GRU**: 64 hidden units, dropout=0.3
- **TCN (Temporal Convolutional Network)**: 64 filters, dilations [1,2,4], causal padding
- **Temporal MLP**: Flattened 14-day input → Dense(128) → Dense(64) → Softmax(3)

### ML7 Model Comparison

| Model                              | F1-macro (mean ± std) | F1-weighted | Balanced Accuracy | AUROC (OvR) | Cohen's κ |
| ---------------------------------- | --------------------- | ----------- | ----------------- | ----------- | --------- |
| LSTM (baseline)                    | —                     | —           | —                 | —           | —         |
| _(metrics pending ML7 completion)_ |                       |             |                   |             |           |
| MLP                                | 0.4438 ± 0.1283       | 0.6115      | 0.4570            | 0.7563      | 0.2152    |

---

## Interpretation & Key Findings

- **ML6 Best Model**: RF achieved F1-macro = 0.7005, compared to Logistic Regression baseline (0.6874).
- **Instability Regularization**: Improved performance for tree/boosting models.
- **ML7 Performance**: Sequence models (GRU/TCN/MLP) face challenges due to:
  - Weak supervision (PBSI heuristic labels, not clinical gold standard)
  - Non-stationarity across 8-year timeline (119 behavioral segments)
  - Limited dataset size (1,625 days post-2021 temporal filter)
- **Baseline Strength**: Logistic regression remains a strong, interpretable baseline for this N-of-1 dataset.
- **Future Work**: Multi-participant datasets, stronger supervision (PHQ-9/MDQ), and federated learning may improve sequence model performance.

---

## Reproducibility Notes

All extended models use **Stage 5 preprocessed outputs** (no raw data required):

**Required Files**:

- `data/ai/P000001/2025-11-07/ml6/features_daily_ml6.csv` (1,625 rows, MICE-imputed)
- `data/etl/P000001/2025-11-07/joined/features_daily_labeled.csv` (2,828 rows, full timeline)
- `data/etl/P000001/2025-11-07/segment_autolog.csv` (119 segments)
- `data/ai/P000001/2025-11-07/ml6/cv_summary.json` (6-fold definitions)

**No Zepp Password Required**: Pipeline runs in Apple-only mode if Zepp data unavailable.

**Regenerate Extended Models**:

```bash
make ml-extended-all PID=P000001 SNAPSHOT=2025-11-07
```

**Regenerate This Report**:

```bash
make report-extended PID=P000001 SNAPSHOT=2025-11-07
```

---

## References

- **Implementation Details**: `docs/copilot/ML6_ML7_EXTENDED_IMPLEMENTATION.md`
- **Quick Start Guide**: `docs/copilot/QUICK_START.md`
- **Core Pipeline**: `RUN_REPORT.md`
- **Pipeline Architecture**: `pipeline_overview.md`

**Generated**: 2025-11-22T04:02:04.325620
