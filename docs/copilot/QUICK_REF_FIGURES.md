# Extended Models Figures - Quick Reference

## 📊 New Figures Generated

| #   | Filename                                   | Description                             | Best Model    |
| --- | ------------------------------------------ | --------------------------------------- | ------------- |
| 10  | `fig10_extended_ml6_models.{pdf,png}`      | ML6 tabular models (RF, XGB, LGBM, SVM) | SVM: 0.702    |
| 11  | `fig11_extended_ml7_models.{pdf,png}`      | ML7 sequence models (GRU, TCN, MLP)     | GRU: 0.801 ⭐ |
| 12  | `fig12_instability_scores.{pdf,png}`       | Top-15 unstable features                | total_steps   |
| 13  | `fig13_extended_models_combined.{pdf,png}` | Side-by-side ML6 + ML7                  | -             |

## 🚀 Quick Commands

```bash
# Generate ALL figures (core 1-9 + extended 10-13)
python scripts/generate_dissertation_figures.py

# If extended model data missing:
make ml6-rf ml6-xgb ml6-lgbm ml6-svm PID=P000001 SNAPSHOT=2025-11-07
make ml7-gru ml7-tcn ml7-mlp PID=P000001 SNAPSHOT=2025-11-07

# Verify figures:
ls docs/latex/figures/fig1*.pdf
```

## 📝 LaTeX Quick Include

```latex
% Figure 10: ML6 Extended
\begin{figure}[htbp]
    \centering
    \includegraphics[width=0.8\textwidth]{figures/fig10_extended_ml6_models.pdf}
    \caption{Extended ML6 tabular models comparison (SVM best: 0.702 ± 0.131).}
    \label{fig:extended_ml6_models}
\end{figure}

% Figure 11: ML7 Extended
\begin{figure}[htbp]
    \centering
    \includegraphics[width=0.8\textwidth]{figures/fig11_extended_ml7_models.pdf}
    \caption{Extended ML7 sequence models (GRU best: 0.801 ± 0.153).}
    \label{fig:extended_ml7_models}
\end{figure}

% Figure 12: Instability
\begin{figure}[htbp]
    \centering
    \includegraphics[width=0.9\textwidth]{figures/fig12_instability_scores.pdf}
    \caption{Top-15 temporally unstable features.}
    \label{fig:instability_scores}
\end{figure}

% Figure 13: Combined
\begin{figure}[htbp]
    \centering
    \includegraphics[width=\textwidth]{figures/fig13_extended_models_combined.pdf}
    \caption{Combined ML6 + ML7 extended models comparison.}
    \label{fig:extended_models_combined}
\end{figure}
```

## 📈 Key Results

**ML6 Best**: SVM (0.702 ± 0.131)  
**ML7 Best**: GRU (0.801 ± 0.153) ⭐  
**Most Unstable**: total_steps (1.000)

## 📚 Documentation

- **Full Guide**: `docs/FIGURES_GUIDE.md` (224 lines)
- **Implementation**: `docs/copilot/FIGURE_GENERATION_SUMMARY.md` (180 lines)
- **Task Completion**: `docs/copilot/TASK_COMPLETION_FIGURE_GENERATION.md` (370 lines)

## ✅ Status

- [x] Script extended (+285 lines)
- [x] 4 new figures generated (8 files: PDF+PNG)
- [x] Documentation complete
- [x] Tested and verified
- [x] Ready for LaTeX inclusion

**Output**: `docs/latex/figures/` (13 PDF + 13 PNG at 300 DPI)
