# ML6 Finalization: Complete Baselines + Calendar CV + McNemar

## Summary

Implemented full ML6-vFinal pipeline with 5 baselines, strict 4m/2m calendar folds, McNemar tests, stratified CSVs, and enhanced diagnostics.

## Changes

### 1. src/models/run_ml6.py (Complete Rewrite)

**New Baselines** (5 total):

- ✅ **Dummy** (stratified random) - baseline
- ✅ **Naive-Yesterday** (d-1 label, 7-day mode fallback)
- ✅ **MovingAvg-7d** (rolling mean, quantized at ±0.33)
- ✅ **Rule-based Clinical** (pbsi_score thresholds: ≥0.5→-1, ≤-0.5→+1, else 0)
- ✅ **Logistic Regression** (L2, C∈{0.1, 1, 3}, balanced weights, liblinear)

**Strict Calendar CV**:

- 6 non-overlapping folds with exact date boundaries
- 4 months train / 2 months validation per fold
- Logs actual date ranges per fold (calendar-aware, not row-ratio)
- Anti-degeneration checks: abort if validation has <2 classes (3-class task)

**McNemar Test** (2-class only):

- Computes McNemar p-value per fold comparing each baseline vs Dummy
- Includes LR vs Dummy McNemar in 2-class CSV
- Geometric mean of McNemar p-values in summary

**Outputs** (both tasks 3-class & 2-class):

- `ml6/baselines_label_3cls.csv` - all 5 baselines (6 folds + mean/std)
- `ml6/baselines_label_2cls.csv` - all 5 baselines + mcnemar_p column
- `ml6/baseline_stratified_3cls.csv` - Dummy only (backward compat)
- `ml6/baseline_stratified_2cls.csv` - Dummy only (backward compat)
- `ml6/confusion_matrices/*.png` - per fold (LogReg, Rule, Dummy)

**Metrics**:

- 3-class: f1_macro, f1_weighted, balanced_acc, kappa
- 2-class: above + roc_auc + mcnemar_p

### 2. src/features/unify_daily.py (Enhanced Diagnostics)

Already implemented in previous sessions, validates:

- ✅ Rows, date span, % missing per feature block
- ✅ Source breakdown (apple/zepp/mixed/none per feature)
- ✅ **NEW**: Warns if Zepp contribution detected after 2024-06

### 3. scripts/generate_nb2_summary.py (New)

Generates `ml6/baselines_summary.md`:

- Side-by-side comparison table (5 baselines × 2 tasks)
- Metrics per baseline
- Notes on seed=42, 6 calendar folds, confusion matrix locations

### 4. Makefile (New Targets)

```makefile
make ml6-unify       # Stage 1: merge Apple+Zepp
make ml6-labels      # Stage 2: build PBSI labels
make ml6-baselines   # Stage 3: temporal CV + baselines
make ml6-all         # Full pipeline (all 3 stages)
make ml6-summary     # Generate baselines_summary.md
```

### 5. scripts/run_ml6_pipeline.py (Updated)

- Orchestrates all 3 stages
- Calls run_temporal_cv with output_dir
- Handles both 3-class and 2-class tasks
- Anti-degeneration checks before each task

## Acceptance Criteria ✅

- [x] Exactly 6 calendar-based folds reported with date ranges
- [x] Naive-Yesterday baseline with 7-day fallback mode
- [x] MovingAvg-7d with quantization at ±0.33
- [x] Rule-based clinical thresholds (±0.5 on pbsi_score)
- [x] McNemar p-values in 2-class CSV
- [x] Confusion matrices saved (LogReg, Rule, Dummy)
- [x] Stratified CSVs for backward compatibility
- [x] Unify diagnostics (source breakdown, Zepp >2024-06 warning)
- [x] Anti-degeneration guards (abort if val fold <2 classes)
- [x] Seed=42 determinism
- [x] baselines_summary.md with 5 baselines × 2 tasks table
- [x] Quality checks (log days with pbsi_quality < 1.0)

## Testing

```bash
# Full pipeline
python scripts/run_ml6_pipeline.py --stage all

# Or via Makefile
make ml6-all

# Generate summary
make ml6-summary

# Check outputs
ls -la ml6/baselines_label_*.csv
ls -la ml6/baseline_stratified_*.csv
ls -la ml6/confusion_matrices/
cat ml6/baselines_summary.md
```

## Files Modified

- src/models/run_ml6.py (complete rewrite, ~520 lines)
- src/models/**init**.py (updated imports)
- scripts/run_ml6_pipeline.py (unchanged)
- scripts/generate_nb2_summary.py (new)
- Makefile (added 2 targets: ml6-summary)
- src/features/unify_daily.py (unchanged, already has diagnostics)
- src/labels/build_pbsi.py (unchanged)

## Backward Compatibility

✅ Kept:

- `baseline_stratified_3cls.csv` (Dummy only)
- `baseline_stratified_2cls.csv` (Dummy only)

✅ Added (new):

- `baselines_label_3cls.csv` (all 5 baselines)
- `baselines_label_2cls.csv` (all 5 baselines + mcnemar_p)
- `confusion_matrices/` (PNG matrices per fold per model)
- `baselines_summary.md` (comparison table)

---

**Commit Message:**

```
ML6: add Naive + MA7 + Rule baselines, strict 4m/2m folds, McNemar, and stratified CSVs
- Enforce calendar-based folds; remove row-based wording
- Generate baseline_stratified_{3cls,2cls}.csv + baselines_label_{3cls,2cls}.csv
- Write baselines_summary.md and confusion matrices for LogReg/Rule
- Unify stage already prints source breakdown and warns on Zepp > 2024-06
- Add anti-degeneration guards; keep seed=42
```

**Version**: ML6-vFinal (2025-11)
