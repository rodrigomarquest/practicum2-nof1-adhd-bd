# Notebooks Overview

This folder contains a small set of curated notebooks used during the
development and validation of the N-of-1 ETL and modeling pipeline.

For **reproducible runs**, prefer the CLI + Makefile entrypoints
(e.g. `make etl-all`, `make nb1-eda`, `make nb2-baselines`). The notebooks
here are mainly for exploration, visualization and teaching.

## Available notebooks

- **NB0_Test.ipynb**  
  Quick smoke test for the environment. Verifies that core modules under
  `src/` can be imported and that a minimal ETL step runs without errors.

- **NB1_EDA_daily.ipynb / NB1_EDA_daily.py**  
  Exploratory Data Analysis on `features_daily.csv`. Generates:

  - summary statistics,
  - coverage plots,
  - `nb1_eda_summary.md` and `nb1_feature_stats.csv` artifacts
    (also available under `archive/root_artifacts`).

- **NB2_Baseline.py**  
  Script-style notebook file used to prototype NB2 baselines. The canonical
  training is now implemented in `src/models/run_nb2.py` and wired into
  the Makefile target `make nb2-baselines`.

- **NB2_Baselines_LSTM.ipynb**  
  Experimental notebook combining NB2 baselines with early sequence models
  (LSTM / CNN1D). Useful for ad-hoc experiments; the production-grade NB3
  sequence pipeline lives under `src/models_nb3.py` and `src/nb3_run.py`.

- **NB3_DeepLearning.py**  
  Early deep learning experiments for NB3 (LSTM / CNN1D). The current
  production-ready setup is reflected in the source code and in
  `docs/NB3_QUICK_REFERENCE.md`.

## Recommended entrypoints

For day-to-day work:

- Use `make etl-all` to regenerate features for a snapshot.
- Use `make nb1-eda` to update the EDA summary.
- Use `make nb2-baselines` to train and evaluate baseline models.
- Use `make nb3-seq` (or the configured equivalent) for sequence models.

The notebooks are optional and may evolve or be replaced as the project
progresses.
