🚀 v4.1.8 – Notebook Fixes & Documentation Updates

Release date: 2025-11-20  
Branch: `main`  
Author: Rodrigo Marques Teixeira  
Project: MSc AI for Business – Practicum Part 2 (N-of-1 ADHD + BD)

---

## 🔧 Summary

This release focuses on fixing notebook execution errors and improving documentation consistency following the ML6/ML7 refactoring. All notebooks are now executable without errors or warnings, with corrected metric loading, proper PBSI v4.1.7 label terminology, and comprehensive visualizations of ML pipeline results.

All changes maintain pipeline compatibility while ensuring reproducible notebook execution and clear documentation.

---

## 🧩 Highlights

### 📓 Notebook Fixes

• **NB2_Baseline.ipynb** (ML6 Stage 6)

- Fixed to read `cv_summary.json` instead of non-existent CSVs
- Adjusted metric names (`f1_macro`, `balanced_accuracy`)
- Added informative messages for missing files
- Updated confusion matrix labels to v4.1.7 terminology (`regulated`/`typical`/`dysregulated`)
- All 6 cells execute successfully without warnings

• **NB3_DeepLearning.ipynb** (ML7 Stage 7)

- **Setup**: Changed `NB2_DIR` → `ML6_DIR` (fixed path references)
- **Metrics**: Parse `lstm_report.md` with regex (extracts 6 folds)
- **Visualization**: Plot fold-wise metrics (F1, Accuracy, Loss)
- **SHAP**: Read `shap_summary.md` and display PNG files
- **Comparison**: Fixed NameError, read `cv_summary.json` from ML6
- **Drift**: Visualize `adwin_changes.csv` (6 points) and `ks_segment_boundaries.csv` (338 tests)

### 📊 Performance Results

| Model   | Macro F1  | Std Dev | Folds | Algorithm                      |
| ------- | --------- | ------- | ----- | ------------------------------ |
| **ML6** | **0.710** | ±0.099  | 6     | Logistic Regression (baseline) |
| **ML7** | 0.412     | ±0.084  | 6     | LSTM Sequence Model            |

**Conclusion**: ML6 baseline superior (expected for N=1 single-subject dataset with limited temporal patterns)

### 🔄 Code Quality

• **Makefile Cleanup**

- Removed `print-version` target (unused)
- Maintained release workflow targets
- Cleaner `.PHONY` declarations

---

## 🧪 Testing & Validation

### Modified Files

1. **notebooks/NB2_Baseline.ipynb**

   - Cell 1 (Setup): Added ML6_DIR path definition
   - Cell 2 (Load): Read `cv_summary.json` with JSON parsing
   - Cell 3 (Metrics): Adjusted column names (`f1_macro`)
   - Cell 4 (CM): Updated label names to v4.1.7
   - Cell 5-6 (Viz): Added informative messages for missing files

2. **notebooks/NB3_DeepLearning.ipynb**

   - Cell 1 (Setup): `NB2_DIR` → `ML6_DIR`
   - Cell 2 (Load): Regex parsing of `lstm_report.md`
   - Cell 3 (Training): Fold-wise metric plots
   - Cell 4 (SHAP): Read markdown + display PNGs
   - Cell 5 (Compare): Fixed NameError with ML6_DIR
   - Cell 6 (Drift): CSV visualization (ADWIN + KS tests)

3. **Makefile**

   - Removed `print-version` from `.PHONY`
   - Removed `print-version` from `help-release`
   - Removed `print-version` target definition

4. **docs/release_notes/release_notes_v4.1.8.md**
   - Created comprehensive release notes
   - Documented all notebook fixes
   - Performance comparison table

### Expected Behavior

```bash
# ✅ NB2 Execution (should complete without errors):
jupyter notebook notebooks/NB2_Baseline.ipynb
# All cells execute → ML6 F1 = 0.710 ± 0.099

# ✅ NB3 Execution (should complete without errors):
jupyter notebook notebooks/NB3_DeepLearning.ipynb
# All cells execute → ML7 F1 = 0.412 ± 0.084
# Drift visualization shows 6 ADWIN points, 338 KS boundaries
```

---

## 🧠 Next Steps

- Validate notebooks with fresh ML6/ML7 pipeline runs
- Add unit tests for notebook cell execution
- Document ML6 vs ML7 performance trade-offs in paper
- Consider ensemble methods (ML6 + ML7) for improved predictions
- Add notebook execution to CI/CD pipeline

---

## 📝 Commits Included

- `652dc60`: release: v4.1.8 - Notebook fixes and documentation updates
- `89b3a55`: refactor: Remove print-version target from Makefile
- `12a53b9`: fix: Corrigir notebooks NB2 e NB3 para ML6/ML7
- `bfcb8fb`: refactor: rename NB2→ML6, NB3→ML7 to avoid confusion
- `58658c0`: refactor: NB2→ML6, NB3→ML7 complete (code + docs + notebooks)

---

## 🧾 Citation

Teixeira, R. M. (2025). _N-of-1 Study – ADHD + Bipolar Disorder (Practicum Part 2)_. National College of Ireland. GitHub repository: https://github.com/rodrigomarquest/practicum2-nof1-adhd-bd

---

⚖️ **License**: CC BY-NC-SA 4.0 https://creativecommons.org/licenses/by-nc-sa/4.0/  
**Supervisor**: Dr. Agatha Mattos  
**Student ID**: 24130664  
**Maintainer**: Rodrigo Marques Teixeira

---
