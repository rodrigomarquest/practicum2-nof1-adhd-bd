🧭 Release Notes – v4.0.0

Date: 2025-10-31
Branch: v4-main
Tags: v3.9.9-pristine, v4.0.0
Commit: b616b62

🎯 Summary

Version 4.0.0 marks the complete repository restructuring for the N-of-1 ADHD + BD longitudinal study, consolidating all ETL, EDA and modeling scripts into a clean, reproducible and drift-aware architecture.

This release transforms the previous experimental layout (v3.x) into a modular, production-ready research environment — separating Local ETL + EDA execution from Kaggle-based modeling (NB2/NB3) while preserving full data provenance and academic traceability.

🧩 Structural Highlights
Area Change Outcome
Repository Layout Introduced canonical folders src/, config/, notebooks/outputs/, archive/. Flat, deterministic structure for analysis and publication.
ETL Pipeline Moved to src/etl_pipeline.py with segment-aware z-score normalization (segment_id S1–S∞). Robust drift handling across device/firmware versions.
Label Generation Added src/make_labels.py + config/label_rules.yaml (heuristic_v1, drift-aware pseudo-labels). Scientific transparency for weak-supervision of mood states.
Modeling Code Baselines → src/models_nb2.py; Deep Learning → src/models_nb3.py. Decoupled, reusable, Kaggle-ready modules.
EDA & QC Unified into src/eda.py + notebook NB1_EDA_daily.ipynb. Single-source exploratory workflow.
Configuration New config/settings.yaml (CV windows, seed, rolling lengths). Centralized experimental parameters.
Version Control Tags v3.9.9-pristine (snapshot) → v4.0.0. Historical continuity and auditability.
Reproducibility Makefile v4 + .gitignore rebuilt; provenance freeze via pip_freeze_YYYY-MM-DD.txt. Clean reproducible environment under Python 3.13 (local) / 3.11 (Kaggle).
⚗️ Functional Validation
Test Result Notes
Import smoke-test ✅ IMPORT_OK All src modules import cleanly.
Label Rules YAML ✅ YAML_OK True Non-empty heuristic_v1 structure validated.
Notebooks ✅ NB1_EDA_daily and NB2_Baselines_LSTM include env/version cell; no broken imports.
🧠 Research Integrity

The pseudo-label framework (heuristic_v1) introduces clinically grounded mood state heuristics from Cardio + Sleep + Activity signals, normalized per segment (S1–S∞).

The modular separation ensures reproducibility, weak-supervision transparency, and drift awareness — key academic requirements for N-of-1 digital phenotyping studies.

ETL, labeling, and modeling scripts are now publishable artifacts, aligned with FAIR and open-science principles.

🧾 Definition of Done (DoD v4)

✅ features_daily_labeled.csv ≥ 90 % coverage

✅ Temporal CV (120/10/60 days) working baseline (NB2)

✅ Exported best_model.tflite pipeline scaffold (NB3)

✅ Drift (ADWIN/KS) + SHAP explainability modules integrated

✅ Provenance freeze + reproducible environment documentation

🚀 Next Milestones
Milestone Target
NB2 Baseline Evaluation Run full 6-fold temporal CV on Kaggle with heuristic v1 labels.
NB3 Sequence Models Fine-tune LSTM/CNN variants, measure latency + drift stability.
Literature Integration Formalize label_rules with citations for HRV / Sleep / Activity heuristics.
EMA Integration Optional SoM/EMA labels for validation phase.
Publication Prep Export cleaned notebooks and documentation for CA2 appendices.
📚 Citation

“Teixeira R. M., 2025. N-of-1 Digital Phenotyping in ADHD + Bipolar Disorder:
A longitudinal, drift-aware approach using wearable and mobile data.
National College of Ireland, MSc AI for Business (Practicum Part 2).”

Tag Summary

v3.9.9-pristine — Pre-refactor snapshot (legacy layout).

v4.0.0 — Clean modular architecture, drift-aware ETL → Kaggle modeling pipeline.
