# ML6-vFinal Testing Guide

## Quick Start

### 1. Run Full Pipeline

```bash
# Single command
python scripts/run_ml6_pipeline.py --stage all

# Or via Makefile
make ml6-all
make ml6-summary
```

### 2. Expected Outputs

**Data Files** (data/etl/):

- `features_daily_unified.csv` - Apple+Zepp merged (27 cols, 1 row/date)
- `features_daily_labeled.csv` - PBSI labels added (35 cols)

**Results Files** (ml6/):

- `baselines_label_3cls.csv` - All 5 baselines (3-class)
- `baselines_label_2cls.csv` - All 5 baselines (2-class, with mcnemar_p)
- `baseline_stratified_3cls.csv` - Dummy only (3-class, backward compat)
- `baseline_stratified_2cls.csv` - Dummy only (2-class, backward compat)
- `baselines_summary.md` - Markdown comparison table
- `confusion_matrices/fold_*_*.png` - Confusion matrices (18 PNGs: 6 folds × 3 models)

### 3. Validate Outputs

```bash
# Check row counts (should be 6 data + 2 summary rows)
wc -l ml6/baselines_label_3cls.csv   # 9 rows
wc -l ml6/baselines_label_2cls.csv   # 9 rows

# Check columns (all 5 baselines)
head -1 ml6/baselines_label_3cls.csv
# Expected: fold, dummy_f1_macro, dummy_f1_weighted, ..., lr_kappa

# Check McNemar column (2-class only)
head -1 ml6/baselines_label_2cls.csv | grep mcnemar
# Expected: lr_mcnemar_p, naive_mcnemar_p, ma7_mcnemar_p, rule_mcnemar_p

# Check confusion matrices
ls -la ml6/confusion_matrices/ | wc -l   # Should be 20 (18 PNG + 2 dirs)

# Check summary markdown
cat ml6/baselines_summary.md | head -30
```

### 4. Verify Determinism (Rerun Reproducibility)

```bash
# Delete outputs
rm data/etl/features_daily_*.csv ml6/baselines_*.csv ml6/confusion_matrices/*.png

# Rerun
python scripts/run_ml6_pipeline.py --stage all

# Compare checksums (should match previous run)
md5sum ml6/baselines_label_3cls.csv
# Run again and compare
```

## Expected Log Output

### Stage 1: Unify

```
[INFO] Starting daily unification
[INFO] ✓ Merged data: NNN unique dates
[INFO]
[INFO] ================================================================================
[INFO] UNIFICATION REPORT
[INFO] ================================================================================
[INFO]
[INFO] Date span: YYYY-MM-DD to YYYY-MM-DD
[INFO] Total days: NNN
[INFO] Unique dates: NNN
[INFO]
[INFO] Missing data:
[INFO]   missing_sleep: N (X.X%)
[INFO]   missing_cardio: N (X.X%)
[INFO]   missing_activity: N (X.X%)
[INFO]
[INFO] Data sources:
[INFO]   source_sleep:
[INFO]     apple: NNN (X.X%)
[INFO]     zepp: NNN (X.X%)
[INFO]     mixed: NNN (X.X%)
[INFO]     none: NNN (X.X%)
[...and so on for cardio, activity...]
[WARNING] ⚠️  Zepp sleep data found after 2024-06: YYYY-MM-DD  [IF APPLICABLE]
```

### Stage 2: Labels

```
[INFO] Building PBSI labels from unified data
[INFO] Loading version log: data/version_log_enriched.csv
[INFO] Computing z-scores by segment (S1-S6)
[INFO] Generating PBSI subscores and labels
[INFO] Days with >1 class in label_3cls: YYYY
[INFO] Days with >1 class in label_2cls: YYYY
[INFO] Days with pbsi_quality < 1.0: NNN
[INFO] ✓ Saved labeled data to data/etl/features_daily_labeled.csv
```

### Stage 3: Baselines

```
[INFO] ================================================================================
[INFO] Temporal CV: label_3cls (calendar-based, 4m/2m folds)
[INFO] ================================================================================
[INFO] Generated 6 calendar-based folds
[INFO] Global label_3cls distribution: {+1: NNN, 0: NNN, -1: NNN}
[INFO] Days with pbsi_quality < 1.0: NNN/NNN

[INFO] Fold 0: train 2023-MM-DD to 2023-MM-DD (NNN rows), val 2023-MM-DD to 2023-MM-DD (NNN rows)
[INFO] Fold 1: train 2023-MM-DD to 2023-MM-DD (NNN rows), val 2023-MM-DD to 2023-MM-DD (NNN rows)
[...6 folds...]
[INFO]
[INFO] Fold 0
[INFO]   Train dist: {...}
[INFO]   Val dist: {...}
[INFO]   LR best C: 1
[INFO]   Saved: ml6/confusion_matrices/fold_0_lr_label_3cls.png
[INFO]   Saved: ml6/confusion_matrices/fold_0_rule_label_3cls.png
[INFO]   Saved: ml6/confusion_matrices/fold_0_dummy_label_3cls.png
[...6 folds...]

[INFO] RESULTS (3-class):
[fold | dummy_f1_macro | dummy_f1_weighted | ... | lr_kappa]
[0    | 0.XXXX        | 0.XXXX           | ... | 0.XXXX  ]
[...]
[MEAN | 0.XXXX        | 0.XXXX           | ... | 0.XXXX  ]
[STD  | 0.XXXX        | 0.XXXX           | ... | 0.XXXX  ]

[INFO] Saved: ml6/baselines_label_3cls.csv
[INFO] Saved: ml6/baseline_stratified_3cls.csv

[...similar for 2-class task...]
[INFO] Saved: ml6/baselines_label_2cls.csv
[INFO] Saved: ml6/baseline_stratified_2cls.csv

[INFO] Generate baselines summary markdown
```

## Verification Checklist

- [ ] 6 folds reported with actual date ranges
- [ ] No "Degenerate validation fold" errors
- [ ] Naive baseline successfully computes predictions
- [ ] MA7 baseline successfully quantizes rolling mean
- [ ] Rule-based baseline uses pbsi_score thresholds
- [ ] McNemar p-values computed for 2-class task
- [ ] Confusion matrices saved (18 PNGs)
- [ ] CSVs have correct row count (6 + 2 summary)
- [ ] baselines_summary.md readable and contains 5 baselines
- [ ] Zepp warning logged if data > 2024-06
- [ ] Quality degradation logged (pbsi_quality < 1.0)
- [ ] Rerun produces identical CSVs (seed=42)

## Troubleshooting

### "Degenerate validation fold" Error

**Cause**: Validation fold has only 1 class (e.g., all +1, no -1, no 0)

**Fix**:

1. Check if data is naturally imbalanced
2. Verify PBSI label generation (should have mix of classes)
3. Increase data size or adjust fold boundaries

### McNemar p-values are NaN

**Expected**: OK for 3-class task or when Dummy and LR make identical predictions

**Fix**: Check if 2-class validation set actually has 2 classes

### Missing confusion_matrices/ Directory

**Cause**: output_dir not passed correctly

**Check**:

```bash
ls -la ml6/confusion_matrices/
```

Should have files like `fold_0_lr_label_3cls.png`

### Different CSV on Rerun

**Cause**: seed not set, or random.seed() called after sklearn imports

**Fix**: Ensure `np.random.seed(42)` at module level (already done)

## Advanced Usage

### Single Fold Test

```python
from src.models.run_ml6 import calendar_cv_folds, run_temporal_cv
import pandas as pd

df = pd.read_csv("data/etl/features_daily_labeled.csv", parse_dates=['date'])

# Test calendar CV folding
folds = calendar_cv_folds(df, n_folds=6)
print(f"Generated {len(folds)} folds")
for fold_idx, df_train, df_val, train_str, val_str in folds:
    print(f"Fold {fold_idx}: {train_str} | {val_str}")
```

### Custom Thresholds

Edit `predict_rule_based()` in `src/models/run_ml6.py`:

```python
def predict_rule_based(pbsi_score_val: np.ndarray) -> np.ndarray:
    preds = np.zeros_like(pbsi_score_val, dtype=int)
    preds[pbsi_score_val >= 0.75] = -1  # Change from 0.5
    preds[pbsi_score_val <= -0.75] = 1  # Change from -0.5
    return preds
```

---

**Version**: ML6-vFinal (2025-11)
**Last Updated**: 2025-11-07
