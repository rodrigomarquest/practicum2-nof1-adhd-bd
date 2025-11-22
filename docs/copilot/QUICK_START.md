# ML6/ML7 Extended Models - Quick Start Guide

**Date**: November 22, 2025  
**Status**: ✅ **IMPLEMENTATION COMPLETE**

---

## 🎯 What Was Built

Extended the deterministic N-of-1 pipeline with:

- **4 ML6 models**: Random Forest, XGBoost, LightGBM, SVM
- **3 ML7 models**: GRU, TCN, Temporal MLP
- **Temporal instability regularization**: Feature penalties based on behavioral segment variance
- **Complete CLI interface**: One command to run all models
- **Makefile targets**: Individual model execution

---

## 🚀 Quick Start

### Prerequisites

```bash
# Install new dependencies
pip install xgboost>=2.0.0 lightgbm>=4.1.0

# Verify existing data
ls data/ai/P000001/2025-11-07/ml6/features_daily_ml6.csv
ls data/ai/P000001/2025-11-07/ml6/cv_summary.json
ls data/etl/P000001/2025-11-07/segment_autolog.csv
ls data/etl/P000001/2025-11-07/joined/features_daily_labeled.csv
```

**Important**: Zepp password is **optional**. If missing, the pipeline runs in Apple-only mode. All ML6/ML7 baseline and extended models are reproducible without raw Zepp data, as they use preprocessed Stage 5 outputs.

### Run All Models (Recommended)

```bash
# Via Makefile (NO Zepp password required)
make ml-extended-all PID=P000001 SNAPSHOT=2025-11-07

# Via CLI
python scripts/run_extended_models.py --which all --models all
```

### Run Individual Models

```bash
# ML6 models
make ml6-xgb PID=P000001 SNAPSHOT=2025-11-07
make ml6-lgbm PID=P000001 SNAPSHOT=2025-11-07
make ml6-rf PID=P000001 SNAPSHOT=2025-11-07
make ml6-svm PID=P000001 SNAPSHOT=2025-11-07

# ML7 models
make ml7-gru PID=P000001 SNAPSHOT=2025-11-07
make ml7-tcn PID=P000001 SNAPSHOT=2025-11-07
make ml7-mlp PID=P000001 SNAPSHOT=2025-11-07
```

---

## 📊 Output Structure

### ML6 Extended

```
data/ai/P000001/2025-11-07/ml6_ext/
├── instability_scores.csv              # Feature instability [0,1]
├── ml6_rf_metrics.json                 # Random Forest results
├── ml6_xgb_metrics.json                # XGBoost results
├── ml6_lgbm_metrics.json               # LightGBM results
├── ml6_svm_metrics.json                # SVM results
├── ml6_extended_summary.csv            # Comparative table
└── ml6_extended_summary.md             # Report
```

### ML7 Extended

```
data/ai/P000001/2025-11-07/ml7_ext/
├── ml7_gru_metrics.json                # GRU results
├── ml7_tcn_metrics.json                # TCN results
├── ml7_mlp_metrics.json                # MLP results
├── ml7_extended_summary.csv            # Comparative table
├── ml7_extended_summary.md             # Report
├── ml7_gru_fold{0-5}_weights.h5        # Model weights
└── ml7_gru_fold{0-5}_saliency.csv      # Feature importance
```

---

## 📈 Check Results

```bash
# View ML6 summary
cat data/ai/P000001/2025-11-07/ml6_ext/ml6_extended_summary.csv

# View ML7 summary
cat data/ai/P000001/2025-11-07/ml7_ext/ml7_extended_summary.csv

# Check XGBoost metrics
python -c "
import json
with open('data/ai/P000001/2025-11-07/ml6_ext/ml6_xgb_metrics.json') as f:
    data = json.load(f)
    print(f'XGBoost F1-macro: {data[\"mean_f1_macro\"]:.4f} ± {data[\"std_f1_macro\"]:.4f}')
"

# Check GRU metrics
python -c "
import json
with open('data/ai/P000001/2025-11-07/ml7_ext/ml7_gru_metrics.json') as f:
    data = json.load(f)
    print(f'GRU F1-macro: {data[\"mean_f1_macro\"]:.4f} ± {data[\"std_f1_macro\"]:.4f}')
"

# Check instability scores
head -10 data/ai/P000001/2025-11-07/ml6_ext/instability_scores.csv
```

---

## 📚 Documentation

- **Complete guide**: `docs/copilot/ML6_ML7_EXTENDED_IMPLEMENTATION.md` (685 lines)
  - Architecture details
  - Temporal instability algorithm
  - Usage examples
  - Acceptance tests
  - Troubleshooting

---

## 🔧 Troubleshooting

### Missing dependencies

```bash
pip install xgboost>=2.0.0 lightgbm>=4.1.0
```

### Missing data files

```bash
# Run ML6 baseline first
make ml6-only PID=P000001 SNAPSHOT=2025-11-07

# Or run full pipeline
make pipeline PID=P000001 SNAPSHOT=2025-11-07
```

### Slow ML7 training

```bash
# Skip saliency computation
python scripts/run_extended_models.py --which ml7 --models all --no-saliency
```

---

## ✅ Acceptance Test

```bash
# Full test
python scripts/run_extended_models.py --which all --models all \
    --participant P000001 --snapshot 2025-11-07

# Verify outputs exist
ls data/ai/P000001/2025-11-07/ml6_ext/ml6_xgb_metrics.json
ls data/ai/P000001/2025-11-07/ml7_ext/ml7_gru_metrics.json

# Check metrics are valid
python -c "
import json
with open('data/ai/P000001/2025-11-07/ml6_ext/ml6_xgb_metrics.json') as f:
    data = json.load(f)
    assert data['n_folds'] == 6
    assert 0 <= data['mean_f1_macro'] <= 1
    print('✅ ML6 XGBoost: PASS')

with open('data/ai/P000001/2025-11-07/ml7_ext/ml7_gru_metrics.json') as f:
    data = json.load(f)
    assert data['n_folds'] == 6
    assert 0 <= data['mean_f1_macro'] <= 1
    print('✅ ML7 GRU: PASS')
"
```

---

## 📞 Support

For detailed documentation, see:

- `docs/copilot/ML6_ML7_EXTENDED_IMPLEMENTATION.md`

For issues:

- Rodrigo Marques Teixeira (x24130664@student.ncirl.ie)
- Dr. Agatha Mattos (supervisor)

---

**Status**: ✅ **READY FOR USE**
