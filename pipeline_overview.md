N-of-1 ADHD + Bipolar – Wearable-Based Digital Phenotyping

1. Purpose & Scope

This document gives a high-level overview of the end-to-end data pipeline used in the N-of-1 ADHD + Bipolar Disorder project.

Goals

Integrate multi-modal wearable data (Apple Health, Zepp / Amazfit, Helio Ring) into a single, reproducible pipeline.

Produce daily, participant-level features suitable for time-series modeling and clinical interpretation.

Support the needs of the Practicum CA2/CA3: ETL reproducibility, feature transparency, and modeling traceability (NB1–NB3).

The focus of this document is on what each stage does and how the pieces connect, not on implementation details of each function.

2. Data Sources
   2.1 Participants & Study Design

Design: N-of-1 longitudinal study (24 months).

Primary participant: P000001 (self).

Timeline: approx. 2023–2025 in multiple segments S1–S6 (treatment/medication regimes defined in version_log_enriched.csv).

2.2 Sensors and Platforms

Apple Health (iPhone + Apple Watch / Helio Ring bridge)

Heart Rate (HR)

HRV (if available)

Sleep stages and durations

Activity (steps, distance, active energy)

State of Mind / mood check-ins (for labels)

Zepp / Amazfit (GTR2 → GTR4 + Helio Ring via Zepp)

Heart Rate / HRV

Sleep (total duration, stages, sleep score)

Activity (steps, activity intensity, workouts)

Emotion / Stress (when exportable via Zepp Cloud or in-app export)

Meta & Logs

version_log_enriched.csv: medication / regime segments (S1–S6)

EMA, subjective mood reports (when available)

Device / export metadata (timestamps, snapshot dates, etc.)

3. Repository & Directory Structure (High-Level)

Exact paths may vary slightly; this is the conceptual layout.

.
├── data/
│ ├── raw/ # Original exports (zip/xml/csv) from Apple, Zepp
│ ├── extracted/ # Unzipped / parsed intermediate files
│ ├── etl/
│ │ └── P000001/
│ │ └── snapshots/
│ │ └── YYYY-MM-DD/
│ │ ├── cardio/
│ │ ├── sleep/
│ │ ├── activity/
│ │ ├── joined/
│ │ └── ai*input/
│ └── meta/
│ ├── version_log_enriched.csv
│ └── ...
├── src/
│ ├── etl/
│ │ ├── cardio*_.py
│ │ ├── sleep\__.py
│ │ ├── activity*\*.py
│ │ └── join*\*.py
│ ├── features/
│ │ └── daily_features.py
│ ├── labels/
│ │ └── label_builder.py
│ ├── models/
│ │ ├── run_nb1_eda.py
│ │ ├── run_nb2_baselines.py
│ │ └── run_nb3_lstm.py
│ └── utils/
│ └── io_guards.py, qc_utils.py, ...
├── notebooks/
│ ├── 01_nb1_eda.ipynb
│ ├── 02_nb2_baselines.ipynb
│ └── 03_nb3_lstm.ipynb
├── reports/
│ ├── nb1_eda_summary.md
│ ├── nb1_feature_stats.csv
│ ├── nb2_results.csv
│ └── nb3_results.csv
├── docs/
│ ├── pipeline_overview.md # THIS FILE
│ ├── appendix_c_sensor_mapping.md
│ └── appendix_d_feature_glossary.md
└── Makefile

4. Snapshot Philosophy

All ETL operations are run per snapshot:

A snapshot is a consistent view of raw+extracted data at a given date:
data/etl/P000001/snapshots/2025-10-22/…

All downstream files (features_daily.csv, features_daily_labeled.csv, model-ready tensors) are immutable for that snapshot.

New data = new snapshot, never overwriting previous ones.
This supports:

Reproducibility for CA2/CA3.

Temporal drift analysis.

Comparison across segments S1–S6.

5. ETL Stages
   5.1 Common ETL Pattern

All modalities follow a similar stage flow:

Raw → Extracted

Unzip exports (export.zip, Zepp data bundles).

Parse XML/CSV to normalized tables with UTC-aligned timestamps.

Extracted → Normalized

Map vendor-specific fields into a canonical schema.

Apply basic QC filters (e.g., plausible HR ranges, non-negative sleep durations).

Normalized → Processed (Daily Aggregates)

Resample / aggregate to daily level (date index).

Compute summary statistics (mean, std, min, max, counts).

Processed → Joined

Outer-join across modalities (cardio, sleep, activity) by date and participant_id.

Optional: join meta (segment_id from version_log_enriched.csv).

Joined → Features Daily

Derive higher-level features (rolling windows, z-scores, ratios).

Output: features_daily.csv (without labels) and features_daily_labeled.csv (with label).

Output Layer Naming (typical)

cardio_daily.csv – aggregated per-day cardio metrics.

sleep_daily.csv – aggregated per-day sleep metrics.

activity_daily.csv – per-day steps and activity load.

joined_features_daily.csv – multi-modal join, sometimes still row-per-record.

features_daily.csv – clean 1-row-per-day feature table.

features_daily_labeled.csv – same as above, with final label.

6. Modality Pipelines
   6.1 Cardio (HR / HRV)

Input

Apple Health export.xml / apple_heart_rate.csv

Zepp/Amazfit cardio CSVs
(e.g., heartrate_points.csv, hrv_summary.csv)

Core Steps

Streaming parse Apple HR

Extract HKQuantityTypeIdentifierHeartRate.

Convert timestamps to local time (Dublin) with fallback to UTC.

Store intermediate in extracted/apple_hr.csv.

Parse Zepp HR / HRV

Harmonize timestamps and units.

Remove implausible values (e.g., HR < 30 or > 250 bpm, if configured).

Daily aggregation

apple_hr_mean, apple_hr_std, apple_hr_max, apple_n_hr.

zepp_hr_mean, zepp_hr_std, zepp_hr_max, zepp_n_hr.

Optional HRV metrics (rmssd, sdnn, etc.) when available.

Segment-aware z-scores

Join with version_log_enriched.csv to get segment_id (S1–S6).

Compute z-scores within each segment to control for treatment/medication shifts:

e.g., z_apple_hr_mean_Sk.

QC

Write QC summary, e.g. cardio_qc_summary.csv:

Number of days

Missingness per feature

Min/Median/Max.

6.2 Sleep

Input

Apple Sleep (Sleep Analysis, Sleep Stages).

Zepp Sleep (total sleep, light/deep/REM, score).

Core Steps

Convert episodes to sleep periods anchored to the main night.

Compute per-day metrics such as:

sleep_total_h

sleep_deep_h, sleep_rem_h, sleep_light_h

sleep_onset_time, sleep_midpoint_time

sleep_efficiency (ratio of time asleep / time in bed).

Merge Apple + Zepp where both exist:

Prioritize consistency, keep vendor-specific columns separate (apple*, zepp* prefixes).

Segment-aware z-score normalization (S1–S6).

QC summary file (sleep_qc_summary.csv).

6.3 Activity

Input

Apple Activity / Steps / Workouts.

Zepp steps, activity intensity, workouts.

Core Steps

Aggregate to daily activity metrics:

steps_total, active_energy_kcal, distance_km.

Activity sporadic vs. workouts (if available).

Compute derived features:

Moving averages (7-day, 14-day).

Relative load vs. personal segment baseline (e.g., z_steps_Sk).

Join with cardio/sleep per day.

7. Label Construction

Labels are built in a dedicated labeling stage to ensure clear separation between features and outcomes.

7.1 Label Sources

EMA / clinical mood scales, when present.

Apple “State of Mind” entries (daily mood).

Zepp / Helio Ring Emotion exports (when API/export is available).

7.2 Heuristic Mapping

A rule-based mapping compresses raw mood/affect into a ternary label:

+1 – positive / “better than usual”

0 – neutral / “about the same”

-1 – negative / “worse than usual”

For example:

High negative affect (or low mood score) → -1.

Clear positive affect → +1.

Anything in the “middle band” → 0.

This rule-based mapping is documented in Appendix D (feature glossary & label rules).

7.3 Final Label Attachment

Labels are joined on date (and optionally participant_id).

Output: features_daily_labeled.csv, used by NB2/NB3.

8. Modeling Pipelines (NB1–NB3)
   8.1 NB1 – Exploratory Data Analysis (EDA)

Purpose: Understand basic properties of the dataset before modeling.

Inputs

features_daily.csv (without labels; can also use labeled version).

Key Outputs

nb1_eda_summary.md – narrative summary (ranges, missingness, distributions).

nb1_feature_stats.csv – per-feature statistics.

Quick plots:

Time series of key features (steps, HR, sleep).

Histograms / boxplots for distribution sanity checks.

8.2 NB2 – Baseline Models (Daily Classification)

Purpose: Provide interpretable baselines for daily mood/label prediction.

Inputs

features_daily_labeled.csv (one row per day, with label).

Cross-Validation

Temporal CV with 6 folds:

Each fold: 4 months training / 2 months validation.

Strict calendar boundaries (no leakage).

All folds respect temporal order.

Baselines Implemented

Dummy (Sklearn stratified)

Random predictions respecting class distribution.

Naïve Yesterday

Predict today’s label as yesterday’s.

Fallback to 7-day modal label when previous day missing.

Moving Avg 7-day

Use rolling window over last 7 days.

Quantize mean to ternary label thresholds (±0.33).

Rule-based Clinical

Uses a composite score (e.g., pbsi_score) derived from mood-related items.

Thresholds:

≥ 0.5 → -1

≤ -0.5 → +1

else → 0

Logistic Regression

Multi-class (one-vs-rest).

L2 penalty, C ∈ {0.1, 1, 3} via grid search.

Class-weighted, solver liblinear.

Metrics: F1 (macro/weighted), AUROC (OvR), Balanced Accuracy, Cohen κ, McNemar p.

Outputs

Fold-wise metrics and aggregated tables (nb2_results.csv).

Confusion matrices per fold.

Feature importance (coefficients) for logistic regression.

8.3 NB3 – Sequence Models & Drift (LSTM / CNN1D)

Purpose: Model temporal dependencies and evaluate performance under concept drift.

Inputs

Sequence windows built from features_daily_labeled.csv (e.g., 14-day windows).

Preprocessed tensors with:

Segment-aware standardization.

Train/val splits respecting temporal CV.

Models

LSTM / GRU sequence classifiers.

Optional 1D-CNN configurations.

Evaluation

Same 6-fold temporal CV as NB2.

Early stopping (patience=10 on validation loss).

Drift & robustness:

ADWIN (δ = 0.002) for stream-based drift detection.

KS tests at version / segment boundaries.

SHAP-based drift: >10% change in feature importance flagged.

Outputs

Best models per fold.

Optional TFLite export (best_model.tflite) for on-device evaluation.

Drift reports and SHAP top-5 feature summaries.

9. Orchestration with Makefile

The Makefile coordinates common tasks:

9.1 ETL Targets (examples)

# Run cardio ETL for latest snapshot

make etl-cardio

# Run sleep ETL

make etl-sleep

# Run activity ETL

make etl-activity

# Full ETL (all modalities + join + features)

make etl-all

Typical flow (per new snapshot):

Place raw exports under data/raw/.

Run make etl-all to:

Extract + normalize all modalities.

Aggregate to daily.

Join.

Generate features_daily.csv.

Run label builder (if not integrated in etl-all) to obtain features_daily_labeled.csv.

9.2 Modeling Targets

# NB1 – EDA

make nb1-eda

# NB2 – Baselines

make nb2-baselines

# NB3 – Sequence models

make nb3-seq

Outputs are written under reports/ and notebooks/outputs/, with timestamped folders to ensure reproducibility.

9.3 Release & Provenance (Optional)

For the Practicum and GitHub releases:

make release-notes – build release_notes_vX.Y.Z.md.

make release-draft – assemble assets and dry-run release.

make release-assets – collect ETL manifests, QC reports, and model outputs.

make publish-release – final tag and GitHub Release (optionally with CA2 bundle).

10. Quality Control & Provenance

Across the pipeline, QC and provenance are first-class concerns:

QC files per modality (cardio, sleep, activity) summarizing:

Date ranges.

Missingness.

Outlier counts.

ETL manifests (CSV/JSON) capturing:

Snapshot ID.

Raw → processed file mapping.

SHA-256 checksums.

Version log (version_log_enriched.csv) linking periods of time to:

Medication changes.

Life events / segments (S1–S6).

Device changes (GTR2 → GTR4, Helio Ring introduction).

These artifacts support auditability, reproducibility, and interpretability for CA2/CA3 and future multi-participant scaling.

11. Known Limitations & Next Steps

Incomplete Emotion Exports

Zepp / Helio Ring emotion data may still depend on Zepp Cloud API or partial exports.

The pipeline is designed to integrate these as soon as stable exports are available.

Single-Participant Focus

Current implementation is optimized for P000001.

Generalization to multiple participants will require:

Robust handling of different device combinations.

Participant-specific configuration and consent.

Label Noise

Labels based on heuristic mapping of mood / State of Mind may contain noise.

Collaboration with psychology / clinical supervision is planned for refinement.

Future Work

Stabilize NB3 models and finalize best_model.tflite export.

Expand Appendix C (sensor mapping) and Appendix D (feature glossary) to fully align with this pipeline overview.

Evaluate feasibility of adding more behavioral signals (screen time, app usage) in later phases.

This pipeline_overview.md is meant to evolve alongside the codebase and Practicum requirements. Whenever a major change is made to the ETL, labeling rules, or modeling strategy, this document should be updated accordingly.
