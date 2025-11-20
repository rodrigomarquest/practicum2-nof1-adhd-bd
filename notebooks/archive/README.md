# Deprecated Notebooks

**Warning**: These notebooks are obsolete and replaced by canonical v4.1.5 notebooks.

## Replacements

| Old File                  | Replaced By                                 | Notes                                 |
| ------------------------- | ------------------------------------------- | ------------------------------------- |
| NB0_Test.ipynb            | NB0_DataRead.ipynb                          | Complete rewrite with stage detection |
| NB1_EDA_daily.ipynb / .py | NB1_EDA.ipynb                               | Comprehensive 8-year analysis         |
| NB2_Baselines_LSTM.ipynb  | NB2_Baseline.ipynb + NB3_DeepLearning.ipynb | Split into separate notebooks         |
| NB2_Baseline.py           | NB2_Baseline.ipynb                          | Converted from script to notebook     |
| NB3_DeepLearning.py       | NB3_DeepLearning.ipynb                      | Converted from script to notebook     |

## Why Deprecated?

- Mixed formats (.ipynb + .py)
- Inconsistent naming conventions
- Hardcoded absolute paths
- Missing error handling
- No graceful handling of missing data
- Non-standard visualization styles

## v4.1.5 Improvements

✅ Self-contained (runnable from repo root)  
✅ Relative paths only  
✅ Standard libraries (no custom imports)  
✅ Graceful error handling with actionable hints  
✅ Publication-quality visualizations  
✅ Comprehensive documentation  
✅ 100% reproducible with fixed seeds

**Do not use these files for new analysis. See `docs/notebooks_overview.md` for canonical notebooks.**

---

**Archive Date**: 2025-11-20  
**Pipeline Version**: v4.1.5
