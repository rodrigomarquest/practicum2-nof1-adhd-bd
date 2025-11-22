# Dissertation Figures Guide

## Overview

The dissertation figure generation pipeline produces 13 publication-quality figures for inclusion in the LaTeX dissertation document. All figures are generated at 300 DPI in both PDF (vector) and PNG (raster) formats.

## Generating Figures

To regenerate all figures (core + extended models):

```bash
python scripts/generate_dissertation_figures.py
```

The script is **idempotent** and will safely overwrite existing figures.

## Output Directory

All figures are saved to:

```
docs/latex/figures/
```

## Generated Figures

### Core Pipeline Figures (1-9)

| Figure | Filename                                      | Description                                 |
| ------ | --------------------------------------------- | ------------------------------------------- |
| 1      | `fig01_pbsi_timeline.{pdf,png}`               | PBSI behavioral labels timeline (2017-2025) |
| 2      | `fig02_cardio_distributions.{pdf,png}`        | HR/HRV distributions by behavioral label    |
| 3      | `fig03_sleep_activity_boxplots.{pdf,png}`     | Sleep and activity patterns by label        |
| 4      | `fig04_segmentwise_normalization.{pdf,png}`   | Segment-wise z-score normalization demo     |
| 5      | `fig05_missing_data_pattern.{pdf,png}`        | Cardiovascular data availability timeline   |
| 6      | `fig06_label_distribution_timeline.{pdf,png}` | Label distribution over time (stacked)      |
| 7      | `fig07_correlation_heatmap.{pdf,png}`         | Feature correlation matrix                  |
| 8      | `fig08_adwin_drift.{pdf,png}`                 | ADWIN drift detection events                |
| 9      | `fig09_shap_importance.{pdf,png}`             | SHAP feature importance (6 folds)           |

### Extended Models Figures (10-13)

| Figure | Filename                                   | Description                                      |
| ------ | ------------------------------------------ | ------------------------------------------------ |
| 10     | `fig10_extended_ml6_models.{pdf,png}`      | ML6 extended tabular models (RF, XGB, LGBM, SVM) |
| 11     | `fig11_extended_ml7_models.{pdf,png}`      | ML7 extended sequence models (GRU, TCN, MLP)     |
| 12     | `fig12_instability_scores.{pdf,png}`       | Top-15 temporally unstable features              |
| 13     | `fig13_extended_models_combined.{pdf,png}` | Side-by-side ML6 + ML7 comparison                |

## LaTeX Inclusion

### Basic Usage

Use `\includegraphics` to include figures in your LaTeX document:

```latex
\begin{figure}[htbp]
    \centering
    \includegraphics[width=0.8\textwidth]{figures/fig10_extended_ml6_models.pdf}
    \caption{Extended ML6 tabular models performance comparison using 6-fold temporal cross-validation. Error bars represent standard deviation across folds.}
    \label{fig:extended_ml6_models}
\end{figure}
```

### Extended Models Section Example

```latex
\section{Extended Modelling Experiments}

\subsection{Temporal Instability Regularization}

We hypothesized that features with high temporal variance across behavioral segments
might lead to overfitting in tree-based models. Figure~\ref{fig:instability_scores}
shows the top-15 most temporally unstable features.

\begin{figure}[htbp]
    \centering
    \includegraphics[width=0.9\textwidth]{figures/fig12_instability_scores.pdf}
    \caption{Top-15 temporally unstable features ranked by variance of segment-wise means.
    Features with high instability scores (e.g., \texttt{total\_steps}, \texttt{hr\_samples})
    exhibit significant non-stationarity across behavioral phases.}
    \label{fig:instability_scores}
\end{figure}

\subsection{Extended ML6 Models}

Four additional tabular models were evaluated: Random Forest, XGBoost, LightGBM
(all with instability regularization), and SVM without regularization.
Results are shown in Figure~\ref{fig:extended_ml6_models}.

\begin{figure}[htbp]
    \centering
    \includegraphics[width=0.8\textwidth]{figures/fig10_extended_ml6_models.pdf}
    \caption{Extended ML6 models performance using 6-fold temporal cross-validation.
    SVM achieved the highest macro-F1 score (0.702 ± 0.131), followed closely by
    Random Forest (0.701 ± 0.141).}
    \label{fig:extended_ml6_models}
\end{figure}

\subsection{Extended ML7 Sequence Models}

Three deep learning architectures were evaluated on 14-day sequence windows:
GRU, TCN (Temporal Convolutional Network), and Temporal MLP.
Figure~\ref{fig:extended_ml7_models} presents the results.

\begin{figure}[htbp]
    \centering
    \includegraphics[width=0.8\textwidth]{figures/fig11_extended_ml7_models.pdf}
    \caption{Extended ML7 sequence models performance. GRU substantially outperformed
    other architectures (F1-macro=0.801 ± 0.153, AUROC=0.957), while TCN struggled
    with the dataset characteristics (F1-macro=0.251 ± 0.038).}
    \label{fig:extended_ml7_models}
\end{figure}

\subsection{Comparative Analysis}

Figure~\ref{fig:extended_models_combined} provides a side-by-side comparison of all
extended models across both modelling paradigms.

\begin{figure}[htbp]
    \centering
    \includegraphics[width=\textwidth]{figures/fig13_extended_models_combined.pdf}
    \caption{Combined comparison of extended ML6 (tabular) and ML7 (sequence) models.
    The GRU sequence model achieved the highest overall performance, suggesting that
    temporal dependencies are valuable for behavioral state prediction.}
    \label{fig:extended_models_combined}
\end{figure}
```

## Figure Specifications

- **Resolution**: 300 DPI (publication quality)
- **Format**: Both PDF (vector) and PNG (raster)
- **Style**: Grayscale-friendly, high-contrast
- **Fonts**: Serif family (consistent with academic standards)
- **Line widths**: 0.8-1.5 pt (clear at print scale)
- **Grid**: Alpha=0.3 (subtle, non-distracting)

## Data Sources

### Core Figures (1-9)

- **Source**: `data/etl/P000001/2025-11-07/joined/features_daily_labeled.csv`
- **ML6 Data**: `data/ai/P000001/2025-11-07/ml6/features_daily_ml6.csv`
- **ML7 Data**: `data/ai/P000001/2025-11-07/ml7/` (SHAP, drift)
- **Date Range**: 2017-12-04 to 2025-10-21 (full dataset)
- **ML-filtered**: 2021-05-11 to 2025-10-21 (1,625 days)

### Extended Figures (10-13)

- **ML6 Extended**: `data/ai/P000001/2025-11-07/ml6_ext/ml6_extended_summary.csv`
- **ML7 Extended**: `data/ai/P000001/2025-11-07/ml7_ext/ml7_extended_summary.csv`
- **Instability**: `data/ai/P000001/2025-11-07/ml6_ext/instability_scores.csv`

## Reproducibility

All figures are generated deterministically from the processed data. To ensure reproducibility:

1. **Run ETL pipeline** (if raw data changes):

   ```bash
   make etl PID=P000001 SNAPSHOT=2025-11-07
   ```

2. **Run extended models** (if model results need updating):

   ```bash
   make ml6-rf ml6-xgb ml6-lgbm ml6-svm PID=P000001 SNAPSHOT=2025-11-07
   make ml7-gru ml7-tcn ml7-mlp PID=P000001 SNAPSHOT=2025-11-07
   ```

3. **Generate figures**:
   ```bash
   python scripts/generate_dissertation_figures.py
   ```

## Troubleshooting

### Missing Extended Model Data

If figures 10-13 show warnings like `[WARN] Extended ML6 summary not found`, run:

```bash
# Generate ML6 extended results
make ml6-rf ml6-xgb ml6-lgbm ml6-svm PID=P000001 SNAPSHOT=2025-11-07

# Generate ML7 extended results
make ml7-gru ml7-tcn ml7-mlp PID=P000001 SNAPSHOT=2025-11-07

# Then regenerate figures
python scripts/generate_dissertation_figures.py
```

### Figure Quality Issues

If figures appear blurry or low-resolution:

- Use the **PDF versions** for LaTeX (vector graphics scale infinitely)
- PNG versions are provided for web/preview purposes only
- All figures are generated at 300 DPI by default

### Customizing Figure Appearance

To modify figure styling, edit the matplotlib configuration in `scripts/generate_dissertation_figures.py`:

```python
# Configure publication-quality style (lines 20-28)
plt.style.use('seaborn-v0_8-paper')
sns.set_context('paper', font_scale=1.2)
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.family'] = 'serif'
```

## Version Information

- **Script**: `scripts/generate_dissertation_figures.py`
- **Participant**: P000001
- **Snapshot**: 2025-11-07
- **Python**: 3.9+
- **Dependencies**: pandas, numpy, matplotlib, seaborn, Pillow

## Contact

For questions about figure generation or modifications, refer to:

- Implementation documentation: `docs/copilot/IMPLEMENTATION_COMPLETE.md`
- Extended models report: `RUN_REPORT_EXTENDED.md`
- Interactive notebook: `notebooks/NB4_Extended_Models.ipynb`
