# Nomenclature Update: NB2→ML6, NB3→ML7

**Date**: 2025-11-20  
**File**: `docs/latex/main.tex`  
**Status**: ✅ Complete (100% replaced)

---

## Summary

Updated dissertation nomenclature to align with codebase refactoring:
- **NB2** (Notebook 2) → **ML6** (Machine Learning pipeline #6)
- **NB3** (Notebook 3) → **ML7** (Machine Learning pipeline #7)

This change reflects the transition from Jupyter notebook-based prototypes to production-grade modular ML pipelines implemented in `src/models_ml6.py` and `src/models_ml7.py`.

---

## Sections Updated

### Abstract (Lines 35-42)
- "NB2, a leak-free supervised baseline" → "ML6, a leak-free supervised baseline"
- "NB3, a sequence model" → "ML7, a sequence model"

### Introduction (Lines 53-79)
- Workflow stage (7): "NB2 modelling" → "ML6 modelling"
- Workflow stage (8): "NB3 analysis" → "ML7 analysis"
- "regularised logistic regression baseline (NB2)" → "baseline (ML6)"
- "14-day LSTM sequence model (NB3)" → "sequence model (ML7)"
- Research objective: "Compare... (NB2) with... (NB3)" → "(ML6) with... (ML7)"
- Paper structure: "results for NB2 and NB3" → "results for ML6 and ML7"

### Methodology (Lines 152-700)
- Pipeline stages: "anti-leakage feature filtering for NB2" → "for ML6"
- Pipeline stages: "baseline modelling (NB2)" → "baseline modelling (ML6)"
- Pipeline stages: "sequence modelling (NB3)" → "sequence modelling (ML7)"
- Feature exclusions: "NB2/NB3 models" → "ML6/ML7 models" (4 occurrences)
- Anti-leak mechanisms: "both NB2 and NB3" → "both ML6 and ML7"
- Feature filtering: "NB2 therefore uses..." → "ML6 therefore uses..."
- "NB3 restricts its temporal context" → "ML7 restricts its temporal context"
- "no NB3 input window crosses" → "no ML7 input window crosses"
- Cross-validation: "Both NB2 and NB3 employ" → "Both ML6 and ML7 employ"
- "For NB3, the 14-day input window" → "For ML7, the 14-day input window"
- **Subsection title**: `\subsubsection{NB2: Logistic Regression Baseline}` → `{ML6: ...}`
- "against which NB3 is compared" → "against which ML7 is compared"
- **Subsection title**: `\subsubsection{NB3: LSTM Sequence Model}` → `{ML7: ...}`
- "The NB3 model consists of" → "The ML7 model consists of"
- "NB3 is trained under" → "ML7 is trained under"
- Explainability: "SHAP values... for NB2 predictions" → "for ML6 predictions"
- Provenance: "labelling, segmentation, NB2/NB3 modelling" → "ML6/ML7 modelling"
- Artefacts: "NB2/NB3 modelling artefacts" → "ML6/ML7 modelling artefacts"

### Results (Lines 871-1092)
- Section intro: "baseline modelling (NB2), sequence modelling (NB3)" → "(ML6)... (ML7)"
- Temporal filter: "ML stages (NB2/NB3)" → "ML stages (ML6/ML7)"
- Z-scored features: "excluded from NB3 analysis" → "excluded from ML7 analysis"
- Segmentation: "48 segments remained for NB2/NB3 training" → "for ML6/ML7 training"
- **Subsection title**: `\subsection{NB2 Baseline Performance}` → `{ML6 Baseline Performance}`
- "The NB2 model consists of" → "The ML6 model consists of"
- Performance table: `Table~\ref{tab:nb2results}` → `tab:ml6results`
- **Table caption**: "NB2 logistic regression performance" → "ML6 logistic regression performance"
- **Table label**: `\label{tab:nb2results}` → `\label{tab:ml6results}`
- Results text: "Despite variability, NB2 provides" → "ML6 provides"
- SHAP figure: "underlying NB2 predictions" → "underlying ML6 predictions"
- **Subsection title**: `\subsection{NB3 LSTM Performance}` → `{ML7 LSTM Performance}`
- "The NB3 model uses" → "The ML7 model uses"
- Performance table: `Table~\ref{tab:nb3results}` → `tab:ml7results`
- **Table caption**: "NB3 LSTM classification performance" → "ML7 LSTM classification performance"
- **Table label**: `\label{tab:nb3results}` → `\label{tab:ml7results}`
- Interpretation: "the poor NB3 performance provides evidence against overfitting in NB2" → "ML7... ML6"
- Summary: "NB2 provides strong, interpretable performance" → "ML6 provides..."
- Summary: "NB3 underperforms due to non-stationarity" → "ML7 underperforms..."

### Discussion (Lines 1097-1306)
- Overview: "temporal modelling through NB2 and NB3" → "through ML6 and ML7"
- Temporal modelling: "NB2 and NB3 exploit calendar-based CV" → "ML6 and ML7 exploit..."
- "NB2 emphasises interpretable baselines" → "ML6 emphasises..."
- "while NB3 leverages a multi-day window" → "while ML7 leverages..."
- Interpretation: "The logistic regression baseline (NB2) achieved" → "(ML6) achieved"
- "The success of NB2 supports existing literature" → "The success of ML6 supports..."
- "the LSTM model (NB3) achieved substantially lower" → "(ML7) achieved..."
- "Because NB3 attempts to learn" → "Because ML7 attempts..."
- "NB2, in contrast, relies on segment-wise normalised values" → "ML6, in contrast, relies..."
- "The divergence between NB2 and NB3 reinforces" → "between ML6 and ML7 reinforces"
- Leakage prevention: "NB2 would appear artificially stronger" → "ML6 would appear..."
- Conclusion: "The strong NB2 baseline, the weaker NB3 results" → "ML6... ML7..."

### Conclusion (Lines 1312-1320)
- Pipeline summary: "baseline temporal models (NB2 and NB3)" → "(ML6 and ML7)"

---

## Cross-Reference Labels Updated

### Tables
- `\label{tab:nb2results}` → `\label{tab:ml6results}`
- `\label{tab:nb3results}` → `\label{tab:ml7results}`

### References (to be updated in text if needed)
- `Table~\ref{tab:nb2results}` → `Table~\ref{tab:ml6results}` (already updated)
- `Table~\ref{tab:nb3results}` → `Table~\ref{tab:ml7results}` (already updated)

---

## Validation

### Grep Check (Post-Update)
```bash
grep -E '\bNB[23]\b' docs/latex/main.tex
# Result: No matches found ✅
```

### Appendices Check
```bash
grep -E 'NB2|NB3' docs/latex/appendix*.tex
# Result: No matches found ✅
```

---

## Rationale

**Why ML6/ML7 instead of NB2/NB3?**

1. **Codebase alignment**: Production code uses `src/models_ml6.py` and `src/models_ml7.py`
2. **Module consistency**: ETL pipeline references `ml6/` and `ml7/` output directories
3. **Semantic clarity**: "ML6" indicates machine learning pipeline #6 (more descriptive than "Notebook 2")
4. **Professional naming**: Reflects transition from prototype notebooks to production-grade modules
5. **Versioning transparency**: ML6/ML7 align with pipeline evolution (v4.1.x series introduced ML6 as baseline)

---

## Files Modified

- ✅ `docs/latex/main.tex` (1,462 lines)
  - 80+ replacements across all sections
  - 2 subsection titles updated
  - 2 table labels updated
  - 2 table captions updated

- ⏭️ `docs/latex/appendix*.tex` (no changes needed - no NB2/NB3 references)

---

## Next Steps

1. ✅ **Compile LaTeX**: Verify PDF generation with updated nomenclature
   ```bash
   cd docs/latex
   pdflatex main.tex
   bibtex main
   pdflatex main.tex
   pdflatex main.tex
   ```

2. ⏭️ **Update README**: If project README mentions NB2/NB3, update to ML6/ML7

3. ⏭️ **Update Makefile targets**: Ensure `make ml6` and `make ml7` targets exist (if applicable)

---

**Status**: ✅ Nomenclature update complete! Dissertation now uses consistent ML6/ML7 naming throughout all sections. ���
