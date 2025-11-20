# Notebooks Overview (v4.1.5)

**Pipeline**: practicum2-nof1-adhd-bd  
**Version**: v4.1.5  
**Participant**: P000001  
**Snapshot**: 2025-11-07  
**Date Range**: 2017-12-04 to 2025-10-21 (8 years, ~2,828 days)

## Purpose

This document describes the 4 canonical Jupyter notebooks that demonstrate the complete N-of-1 digital phenotyping pipeline for ADHD and bipolar disorder behavioral stability analysis. These notebooks are designed for:

1. **Thesis examination**: Reproducible results for academic defense
2. **Research paper**: Publication-quality figures (IEEE format)
3. **Code review**: Clear documentation for examiners/reviewers
4. **Future replication**: Template for other N-of-1 studies

All notebooks follow strict guidelines:

- ✅ Self-contained (runnable from repository root)
- ✅ Relative paths only (no hardcoded absolute paths)
- ✅ Standard libraries (pandas, numpy, matplotlib, seaborn)
- ✅ Graceful error handling with actionable hints
- ✅ Read-only (never modify pipeline outputs)
- ✅ Publication-quality visualizations

---

## Pipeline Architecture

The pipeline consists of **10 stages** (0-9):

| Stage | Name          | Input              | Output                           | Script                         |
| ----- | ------------- | ------------------ | -------------------------------- | ------------------------------ |
| 0     | **Ingest**    | `data/raw/`        | `data/etl/.../extracted/`        | `etl_pipeline.py extract`      |
| 1     | **Aggregate** | XML + CSV exports  | `daily_*.csv` (per-metric files) | `run_full_pipeline.py` stage 1 |
| 2     | **Unify**     | Per-metric CSVs    | `features_daily_unified.csv`     | `run_full_pipeline.py` stage 2 |
| 3     | **Label**     | Unified features   | `features_daily_labeled.csv`     | `run_full_pipeline.py` stage 3 |
| 4     | **Segment**   | Labeled features   | `segment_autolog.csv`            | `run_full_pipeline.py` stage 4 |
| 5     | **Prep-NB2**  | Segments           | `X_train.npy`, `y_train.npy`     | `run_full_pipeline.py` stage 5 |
| 6     | **NB2**       | Training arrays    | `data/ai/.../nb2/`               | `run_full_pipeline.py` stage 6 |
| 7     | **NB3**       | Training sequences | `data/ai/.../nb3/`               | `run_full_pipeline.py` stage 7 |
| 8     | **TFLite**    | NB3 model          | `model.tflite`                   | `run_full_pipeline.py` stage 8 |
| 9     | **Report**    | All outputs        | `report.pdf`                     | `run_full_pipeline.py` stage 9 |

**Data Structure**:

```
data/
  raw/                              # Private exports (not committed)
    apple/export.zip
    zepp/*.zip
  etl/P000001/2025-11-07/
    extracted/                      # Stage 0-1 outputs
      daily_sleep.csv
      daily_hr.csv
      daily_hrv.csv
      daily_steps.csv
      daily_screen_time.csv
    joined/                         # Stage 2-4 outputs
      features_daily_unified.csv
      features_daily_labeled.csv
      segment_autolog.csv
    qc/                             # Quality control reports
      qc_report.json
  ai/P000001/2025-11-07/
    nb2/                            # Stage 6 outputs
      metrics_summary.csv
      confusion_matrix.csv
      predictions.csv
    nb3/                            # Stage 7 outputs
      training_log.json
      metrics_summary.csv
      predictions.csv
```

---

## Notebook 0: Data Readiness Check

**File**: `notebooks/NB0_DataRead.ipynb`  
**Lines**: 326  
**Purpose**: Pipeline readiness validation and stage detection

### What It Does

1. **File Existence Checks**: Verifies all expected outputs from stages 0-9
2. **Stage Detection**: Identifies which stages have been completed
3. **Status Report**: Generates actionable table with missing files
4. **Quick Inventory**: Shows data shape, date range, label distribution

### Key Functions

```python
check_file_exists(path)               # Verify file and check if non-empty
check_directory_has_files(path, pat)  # Pattern-based directory validation
```

### Output Example

```
Stage | Status | Key Files | Exists | Action
------|--------|-----------|--------|-------
0-1   | ✓      | daily_*.csv | Yes | -
2-3   | ✓      | features_daily_*.csv | Yes | -
4     | ✗      | segment_autolog.csv | No | Run: python -m scripts.run_full_pipeline --start-stage 4
```

### When to Use

- **Before analysis**: Verify pipeline has been run
- **After crashes**: Check which stages need re-running
- **Onboarding**: Understand pipeline structure
- **Debugging**: Identify missing intermediate outputs

### Expected Runtime

< 5 seconds (file existence checks only)

---

## Notebook 1: Exploratory Data Analysis

**File**: `notebooks/NB1_EDA.ipynb`  
**Lines**: 537  
**Purpose**: Comprehensive 8-year behavioral pattern analysis

### What It Does

Generates **15+ publication-quality visualizations** across 10 analysis sections:

1. **Data Loading**: Read unified + labeled CSVs with graceful error handling
2. **Dataset Overview**: Feature groups (sleep, cardiovascular, activity, screen, labels)
3. **Missingness Analysis**: % missing per feature, bar chart with thresholds
4. **Temporal Trends**: 4-panel time series (sleep, HR, steps, screen time)
5. **Yearly Summaries**: Mean ± std bar charts by year (2018-2025)
6. **Distribution Analyses**: Histograms + KDE + mean/median markers
7. **PBSI Label Analysis**: Pie chart, score distribution, time series evolution
8. **Segment Analysis**: 119 segments, size distribution, timeline plot
9. **Correlation Analysis**: Heatmap for 7 key features
10. **Behavioral Insights**: Trend summaries, extreme episodes (7-day moving averages)

### Key Visualizations

- **Figure 3(a)** [Paper]: Missingness bar chart → Shows data quality over 8 years
- **Figure 3(b)** [Paper]: Yearly summary 4-panel → Documents temporal trends
- **Figure 4** [Paper]: Segment timeline → Illustrates calendar-based segmentation

### Configuration

```python
PARTICIPANT = "P000001"
SNAPSHOT = "2025-11-07"
REPO_ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
```

### When to Use

- **Paper writing**: Generate IEEE-format figures
- **Data validation**: Identify anomalies, outliers, missingness patterns
- **Hypothesis generation**: Discover behavioral clusters, seasonal effects
- **Thesis defense**: Visual evidence of 8-year longitudinal study

### Expected Runtime

30-60 seconds (depends on plotting complexity)

### Important Notes

- Uses `seaborn-v0_8-darkgrid` style for publication consistency
- Handles missing data gracefully (informative errors + hints)
- All dates in ISO format (YYYY-MM-DD)
- Requires stages 0-3 completed

---

## Notebook 2: Baseline Models

**File**: `notebooks/NB2_Baseline.ipynb`  
**Lines**: ~400  
**Purpose**: Logistic regression performance evaluation

### What It Does

1. **Load NB2 Outputs**: Per-fold metrics, confusion matrices, predictions
2. **Performance Metrics**: Visualize macro F1, balanced accuracy, Cohen's kappa across folds
3. **Confusion Matrix**: Heatmaps (raw counts + normalized)
4. **Baseline Comparison**: NB2 vs dummy/naive/rule-based models
5. **Time Series Predictions**: Ground truth vs predictions over 8 years

### Key Visualizations

- **Figure 5** [Paper]: Confusion matrix (normalized) → Shows classification performance
- **Table 3** [Paper]: NB2 vs baselines comparison → Demonstrates model superiority

### NB2 Architecture

- **Model**: Regularized logistic regression (L2 penalty, C=1.0)
- **Features**: 7 physiological signals (sleep, HR, HRV, steps, screen time)
- **Labels**: 3-class PBSI (stable=+1, neutral=0, unstable=-1)
- **Cross-Validation**: Calendar-based (5 folds, chronological splits)
- **Normalization**: Segment-wise StandardScaler (anti-leak safeguard)

### Expected Performance

| Model                         | Macro F1  | Notes                   |
| ----------------------------- | --------- | ----------------------- |
| Dummy (most frequent)         | ~0.20     | Always predicts neutral |
| Naive (previous day)          | ~0.35     | Simple persistence      |
| Rule-based (thresholds)       | ~0.50     | Hand-crafted rules      |
| **NB2 (Logistic Regression)** | **~0.81** | **Learned from data**   |

### When to Use

- **Model validation**: Verify NB2 training succeeded
- **Performance reporting**: Extract metrics for paper
- **Baseline establishment**: Before comparing with NB3
- **Confusion analysis**: Identify systematic misclassifications

### Expected Runtime

10-20 seconds (depends on fold count)

### Important Notes

- Requires stage 6 completed
- Confusion matrix uses `['Stable', 'Neutral', 'Unstable']` labels
- Per-fold metrics stored in `fold_*_metrics.csv` or `metrics_summary.csv`

---

## Notebook 3: Deep Learning Models

**File**: `notebooks/NB3_DeepLearning.ipynb`  
**Lines**: ~500  
**Purpose**: LSTM sequence model evaluation and NB2 comparison

### What It Does

1. **Load NB3 Outputs**: Training logs, metrics, predictions
2. **Training Curves**: Loss + accuracy over epochs (train/val)
3. **Performance Metrics**: Per-fold macro F1, balanced accuracy, Cohen's kappa
4. **NB2 vs NB3 Comparison**: Side-by-side bar chart with improvement %
5. **Sequence Predictions**: 3-panel visualization (truth, predictions, errors)

### Key Visualizations

- **Figure 6** [Paper]: Training curves → Shows LSTM convergence
- **Table 3** [Paper]: NB2 vs NB3 comparison → Quantifies deep learning benefit

### NB3 Architecture

- **Model**: Bidirectional LSTM with dropout
- **Input**: 14-day windows (7-day stride)
- **Features**: 7 physiological signals × 14 timesteps
- **Architecture**: BiLSTM(64) → Dropout(0.3) → Dense(32, ReLU) → Dense(3, softmax)
- **Optimization**: Adam (lr=1e-3), class weights for imbalance
- **Training**: Early stopping (patience=10, min_delta=0.001)
- **Determinism**: Fixed seeds (TensorFlow, NumPy, Python)

### Expected Performance

| Scenario         | NB3 Macro F1 | vs NB2 | Interpretation             |
| ---------------- | ------------ | ------ | -------------------------- |
| **Best case**    | 0.83-0.87    | +3-5%  | Temporal patterns captured |
| **Typical case** | 0.79-0.82    | ±2%    | Parity with baseline       |
| **Worst case**   | < 0.79       | -3%+   | Overfitting (use NB2)      |

### When to Use

- **Model selection**: Decide between NB2 and NB3 for deployment
- **Training validation**: Verify no overfitting (val_loss stable)
- **Performance reporting**: Extract metrics for paper/thesis
- **Sequence analysis**: Understand multi-day behavioral patterns

### Expected Runtime

15-30 seconds (depends on prediction file size)

### Important Notes

- Requires stage 7 completed
- Training log stored in `training_log.json` or `history.json`
- If NB3 < NB2, recommend using NB2 as final model
- TFLite export (stage 8) uses best performing model

---

## Running the Notebooks

### Prerequisites

```bash
# Install dependencies
pip install -r requirements/base.txt

# Run full pipeline (stages 0-7)
python -m scripts.run_full_pipeline \
  --participant P000001 \
  --snapshot 2025-11-07 \
  --start-stage 0 \
  --end-stage 7
```

### Execution Order

1. **NB0_DataRead.ipynb** → Verify pipeline readiness
2. **NB1_EDA.ipynb** → Explore data patterns (requires stages 0-3)
3. **NB2_Baseline.ipynb** → Evaluate logistic regression (requires stage 6)
4. **NB3_DeepLearning.ipynb** → Evaluate LSTM (requires stage 7)

### Common Issues

**Issue**: `FileNotFoundError: features_daily_unified.csv not found`  
**Fix**: Run `python -m scripts.run_full_pipeline --start-stage 0 --end-stage 3`

**Issue**: `Empty directory: data/etl/.../joined/`  
**Fix**: Pipeline not run to completion. Check QC reports for errors.

**Issue**: `KeyError: 'macro_f1' not found`  
**Fix**: Metrics file format changed. Check `metrics_summary.csv` columns.

---

## Publication Checklist

### Figures for Paper (main.tex)

| Figure      | Source           | Notebook | Cell      | Description                |
| ----------- | ---------------- | -------- | --------- | -------------------------- |
| Figure 3(a) | Missingness      | NB1      | Section 3 | Data quality bar chart     |
| Figure 3(b) | Yearly trends    | NB1      | Section 5 | 4-panel time series        |
| Figure 4    | Segments         | NB1      | Section 8 | Timeline with 119 segments |
| Figure 5    | Confusion matrix | NB2      | Section 3 | Normalized heatmap         |
| Figure 6    | Training curves  | NB3      | Section 2 | Loss + accuracy            |

### Tables for Paper

| Table   | Source           | Notebook  | Cell          | Description                 |
| ------- | ---------------- | --------- | ------------- | --------------------------- |
| Table 2 | Dataset summary  | NB1       | Section 2     | Feature groups, missingness |
| Table 3 | Model comparison | NB2 + NB3 | Sections 4, 4 | NB2 vs NB3 metrics          |

### Exporting Figures

```python
# In notebook cells
plt.savefig(REPO_ROOT / "reports" / "plots" / "fig3a_missingness.pdf",
            dpi=300, bbox_inches='tight')
plt.savefig(REPO_ROOT / "reports" / "plots" / "fig3a_missingness.png",
            dpi=300, bbox_inches='tight')
```

### Recommended Figure Format

- **Vector**: PDF (preferred for LaTeX)
- **Raster**: PNG 300 DPI (fallback)
- **Size**: 14cm width (IEEE single-column)
- **Font**: 10pt (matching paper body text)
- **Style**: `seaborn-v0_8-darkgrid` or `seaborn-v0_8-whitegrid`

---

## Reproducibility Notes

### Determinism Guarantees

All notebooks are **100% reproducible** given:

1. ✅ Same input data (P000001, snapshot 2025-11-07)
2. ✅ Same Python version (3.11.x)
3. ✅ Same package versions (`requirements/base.txt`)
4. ✅ Fixed random seeds (42 everywhere)

### Random Seed Locations

- **NB2**: `np.random.seed(42)`, `random.seed(42)` in `run_full_pipeline.py` stage 6
- **NB3**: `tf.random.set_seed(42)`, `np.random.seed(42)`, `random.seed(42)` in stage 7
- **Segmentation**: Deterministic (calendar boundaries + gaps)
- **Labeling**: Deterministic (PBSI thresholds)

### Versioning

- **Pipeline**: v4.1.5
- **Paper**: Final submitted version (100% code-paper alignment)
- **Notebooks**: Canonical release (replaces all previous versions)

### Validation Tests

```bash
# Verify determinism (run twice, compare outputs)
python -m scripts.run_full_pipeline --participant P000001 --snapshot 2025-11-07 --start-stage 0 --end-stage 7
python -m scripts.run_full_pipeline --participant P000001 --snapshot 2025-11-07 --start-stage 0 --end-stage 7

# Compare metrics (should be identical)
diff data/ai/P000001/2025-11-07/nb2/metrics_summary.csv \
     data/ai/P000001/2025-11-07/nb2/metrics_summary.csv.bak
```

---

## Future Work

### Planned Enhancements (v4.2.x)

- [ ] **NB4_Clinical.ipynb**: Correlate predictions with diary entries
- [ ] **NB5_Intervention.ipynb**: Simulate counterfactual scenarios
- [ ] **Interactive dashboards**: Plotly/Dash for web visualization
- [ ] **Multi-participant**: Extend to P000002, P000003 (federated learning)

### Known Limitations

1. **Single subject**: N-of-1 design limits generalizability
2. **Weak supervision**: PBSI labels from heuristics (not clinical gold standard)
3. **Missing data**: ~15-30% missingness in some features (Zepp sync issues)
4. **Long-term shifts**: 8-year timeline includes life events (relocation, pandemic)

### Recommended Extensions

- **External validation**: Compare with clinical assessments (PHQ-9, MDQ)
- **Feature engineering**: Add circadian rhythm features, sleep architecture
- **Ensemble models**: Combine NB2 + NB3 predictions (voting/stacking)
- **Explainability**: SHAP values, attention weights, saliency maps

---

## Contact & Contribution

**Author**: [Participant P000001]  
**Institution**: [University Name]  
**Thesis**: N-of-1 Digital Phenotyping for ADHD and Bipolar Disorder  
**Date**: 2025-11-07

**Repository**: https://github.com/[username]/practicum2-nof1-adhd-bd  
**License**: [Specify license]

For questions or contributions:

1. Open an issue on GitHub
2. Fork repository and submit pull request
3. Contact author via email

---

## Appendix: File Inventory

### Notebook Files

```
notebooks/
  NB0_DataRead.ipynb          # 326 lines - Stage detection
  NB1_EDA.ipynb               # 537 lines - Exploratory analysis
  NB2_Baseline.ipynb          # ~400 lines - Logistic regression
  NB3_DeepLearning.ipynb      # ~500 lines - LSTM evaluation
  README.md                   # This overview document
```

### Configuration Files

```
config/
  settings.yaml               # Pipeline parameters
  label_rules.yaml            # PBSI threshold definitions
  participants.yaml           # Participant metadata
```

### Key Scripts

```
scripts/
  run_full_pipeline.py        # Main pipeline orchestrator (1,048 lines)
  etl_runner.py               # Legacy ETL wrapper
src/
  etl_pipeline.py             # Stage 0 extraction logic
  make_labels.py              # PBSI labeling logic
  models_nb2.py               # Logistic regression training
  models_nb3.py               # LSTM training
```

### Documentation

```
docs/
  notebooks_overview.md       # This document
  QUICK_REFERENCE.md          # Pipeline cheat sheet
  ETL_ARCHITECTURE_COMPLETE.md # Technical architecture
  FINAL_VALIDATION_REPORT.md  # Code-paper alignment report
  MAKEFILE_GUIDE.md           # Make commands reference
```

---

**Version**: 1.0.0  
**Date**: 2025-11-07  
**Status**: ✅ Production-ready
