# 🚀 0.2.0-provenance-alpha – Data Provenance Sprint â€“ 0.2.0-provenance-alpha

**Release date:** 2025-10-22T20:35:48.958411+00:00  
**Branch:** `main`  
**Author:** Rodrigo Marques Teixeira  
**Project:** MSc AI for Business – Practicum Part 2 (N-of-1 ADHD + BD)

---

## 🔧 Summary

Release 0.2.0-provenance-alpha

This release strengthens the end-to-end reproducibility and auditability of the N-of-1 ETL → Modeling pipeline.  
All new scripts follow the **atomic write**, **manifest-based provenance**, and **idempotent rerun** guarantees.

---

## 🧩 Highlights

### 🧱 ETL & QC

- Implemented new deterministic Apple In-App ETL stages:
  - `apple_inapp_parse.py` (normalized CSVs)
  - `apple_inapp_qc.py` (QC metrics & markdown report)
  - `apple_inapp_daily.py` (daily aggregates)
  - Orchestrator target `etl-apple` (parse → qc → daily)
- Added atomic manifest writing, per-run logging, and progress visualization.
- Introduced robust dry-run and idempotence testing via:
  - `make idempotence-check`
  - `make atomicity-sim`

### ⚙️ Build & Provenance

- Added new `io_utils.py` primitives for atomic writes and schema hashing.
- Extended `migrate_layout.py` and `intake_zip.py` to standardize raw → extracted structure.
- Integrated provenance audit (`make provenance`) for run-level data integrity.

### 🧠 CI / Dev Improvements

- Added Makefile lint and structure checks (`lint-layout`, `lint-deprecated-exports`).
- Improved cross-platform compatibility (Windows / Git Bash / Linux).
- Simplified developer UX with `make help-layout` and `make venv-shell`.

---

## 🧪 Testing & Validation

- Dry-run and idempotence checks verified on Apple In-App sample exports.
- All ETL stages validated for atomicity and deterministic manifests.
- Provenance reports correctly summarize normalized → processed data transitions.

---

## 🧠 Next Steps

- Extend idempotence to Zepp and iOS ETL stages (B1–B3, Z1–Z3).
- Add integration tests and GitHub Actions CI for `etl-apple` smoke runs.
- Automate release generation via `make release-draft` and `make release-publish`.
- Begin model retraining using multi-snapshot data (Phase M1).

---

## 🧾 Citation

Teixeira, R. M. (2025). _N-of-1 Study – ADHD + Bipolar Disorder (Practicum Part 2)._  
National College of Ireland. GitHub repository:  
[https://github.com/rodrigomarquest/practicum2-nof1-adhd-bd](https://github.com/rodrigomarquest/practicum2-nof1-adhd-bd)

---

⚖️ **License:** [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)  
Supervisor: **Dr. Agatha Mattos**  
Student ID: **24130664**  
Maintainer: **Rodrigo Marques Teixeira**

---


# 🚀 v2.1.3 – Kaggle Baseline Modelling (preview) & Repo Hygiene

**Release date:** 2025-10-21  
**Branch:** `main`  
**Author:** Rodrigo Marques Teixeira  
**Project:** MSc AI for Business – Practicum Part 2 (N-of-1 ADHD + BD)

- Freeze do ambiente de modelagem (notebooks + ETL estável para baseline).
- Repo hygiene: remoção segura de temporários; preserva `data_ai/` e `data_etl/`.
- Ajustes no .gitignore (sem duplicatas perigosas); Makefile normalizado.
- Preparado para próxima fase (features*daily*\* e labeling).

* � v2.1.2 – Kaggle Baseline Modelling (preview): environment freeze, repo hygiene, Makefile fix, notebooks added (0d79d6c)
* chore: guard Continue configs (secrets safe) (297aa1d)
* chore: guard Continue configs (secrets safe) (f7ad161)
* chore(git): stop tracking local-only script openchat.ps1 (9881c4d)
* chore(cleanup): untrack generated data & notebook outputs per .gitignore (cc67736)
* chore(cleanup): untrack generated data & notebook outputs per .gitignore (ea18757)
* chore(gitignore): tighten ignores for data_ai, data_etl, notebooks outputs (28191bf)
* chore(all): v2.1.1 – cardio stabilization, atomic IO, EDA path fix (a9e4868)
* � v2.1.1 – Cardio stabilization, atomic I/O, and EDA path fix (e71adde)
* feat(etl): cardio minimal implementation (apple/zepp loaders, join, features) (f5cb196)
* chore: modular ETL structure (cardiovascular domain stubs) (f820da9)
* release v2.1.0 – Zepp Integration & Hybrid Join (0419a59)
* fix(makefile): minor adjustments before v2.0.3 tag (2a9c0b6)
* chore(git): allow tracking of requirements\_\*.txt in .gitignore (109efbe)
* chore(docs): add CHANGELOG v2.0.3 – dependency & Makefile standardization (0ec18da)
* chore(docs): add CHANGELOG v2.0.3 – dependency & Makefile standardization (473adf3)
* docs(changelog): add v2.0.2 structural consolidation and cleanup (a51e40c)
* chore(repo): fix extract_knowledgec typo and consolidate all iOS extraction scripts under ios_extract/ (136a150)
* chore(repo): finalize move of probe and plist CSV scripts into ios_extract/; update Makefile (19bb3d3)
* chore(cleanup): ignore decrypted_output correctly, remove temporary manifests and local PDF; update Makefile (7c5d35c)
* fix(structure): move main ETL pipeline back to project root (962b9f9)
* chore(repo): finalize migration of ETL and extraction scripts into ios_extract/ (91f06ae)
* chore(repo): reorganize iOS extraction into ios_extract/, add Makefile & CHANGELOG, restore root requirements, drop tracked PDF; update README/.gitignore (42e8cd3)
* docs(ios_extract): add runbook for encrypted backup → Screen Time/KnowledgeC extraction (8bf8009)
* chore: merge iOS extraction + ETL ignore rules (secure PII, cleaned repo layout) (a43a8c0)
* Add academic diagrams (System Architecture & ETL Pipeline) (39c1163)
* Add academic diagrams (System Architecture & ETL Pipeline) (d717420)
* Sync Configuration Manual from Overleaf (c0ea120)
* Add custom .gitignore for ETL, notebooks, and ethical data handling (f1738bf)
* Add LICENSE and requirements for Practicum2 project (52523eb)
* Update README.md (16124c8)
* Update README.md (5d2581c)
* Initial commit (4af1743)


# 🚀 2.1.5 – Data Provenance Sprint â€“ 2.1.5

**Release date:** 2025-10-22T21:01:27.632173+00:00  
**Branch:** `main`  
**Author:** Rodrigo Marques Teixeira  
**Project:** MSc AI for Business – Practicum Part 2 (N-of-1 ADHD + BD)

---

## 🔧 Summary

Release 2.1.5

This release strengthens the end-to-end reproducibility and auditability of the N-of-1 ETL → Modeling pipeline.  
All new scripts follow the **atomic write**, **manifest-based provenance**, and **idempotent rerun** guarantees.

---

## 🧩 Highlights

### 🧱 ETL & QC

- Implemented new deterministic Apple In-App ETL stages:
  - `apple_inapp_parse.py` (normalized CSVs)
  - `apple_inapp_qc.py` (QC metrics & markdown report)
  - `apple_inapp_daily.py` (daily aggregates)
  - Orchestrator target `etl-apple` (parse → qc → daily)
- Added atomic manifest writing, per-run logging, and progress visualization.
- Introduced robust dry-run and idempotence testing via:
  - `make idempotence-check`
  - `make atomicity-sim`

### ⚙️ Build & Provenance

- Added new `io_utils.py` primitives for atomic writes and schema hashing.
- Extended `migrate_layout.py` and `intake_zip.py` to standardize raw → extracted structure.
- Integrated provenance audit (`make provenance`) for run-level data integrity.

### 🧠 CI / Dev Improvements

- Added Makefile lint and structure checks (`lint-layout`, `lint-deprecated-exports`).
- Improved cross-platform compatibility (Windows / Git Bash / Linux).
- Simplified developer UX with `make help-layout` and `make venv-shell`.

---

## 🧪 Testing & Validation

- Dry-run and idempotence checks verified on Apple In-App sample exports.
- All ETL stages validated for atomicity and deterministic manifests.
- Provenance reports correctly summarize normalized → processed data transitions.

---

## 🧠 Next Steps

- Extend idempotence to Zepp and iOS ETL stages (B1–B3, Z1–Z3).
- Add integration tests and GitHub Actions CI for `etl-apple` smoke runs.
- Automate release generation via `make release-draft` and `make release-publish`.
- Begin model retraining using multi-snapshot data (Phase M1).

---

## 🧾 Citation

Teixeira, R. M. (2025). _N-of-1 Study – ADHD + Bipolar Disorder (Practicum Part 2)._  
National College of Ireland. GitHub repository:  
[https://github.com/rodrigomarquest/practicum2-nof1-adhd-bd](https://github.com/rodrigomarquest/practicum2-nof1-adhd-bd)

---

⚖️ **License:** [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)  
Supervisor: **Dr. Agatha Mattos**  
Student ID: **24130664**  
Maintainer: **Rodrigo Marques Teixeira**

---


# 🚀 2.1.7 – Data Provenance Sprint â€“ 2.1.7

**Release date:** 2025-10-22T21:16:39.126643+00:00  
**Branch:** `main`  
**Author:** Rodrigo Marques Teixeira  
**Project:** MSc AI for Business – Practicum Part 2 (N-of-1 ADHD + BD)

---

## 🔧 Summary

Release 2.1.7

This release strengthens the end-to-end reproducibility and auditability of the N-of-1 ETL → Modeling pipeline.  
All new scripts follow the **atomic write**, **manifest-based provenance**, and **idempotent rerun** guarantees.

---

## 🧩 Highlights

### 🧱 ETL & QC

- Implemented new deterministic Apple In-App ETL stages:
  - `apple_inapp_parse.py` (normalized CSVs)
  - `apple_inapp_qc.py` (QC metrics & markdown report)
  - `apple_inapp_daily.py` (daily aggregates)
  - Orchestrator target `etl-apple` (parse → qc → daily)
- Added atomic manifest writing, per-run logging, and progress visualization.
- Introduced robust dry-run and idempotence testing via:
  - `make idempotence-check`
  - `make atomicity-sim`

### ⚙️ Build & Provenance

- Added new `io_utils.py` primitives for atomic writes and schema hashing.
- Extended `migrate_layout.py` and `intake_zip.py` to standardize raw → extracted structure.
- Integrated provenance audit (`make provenance`) for run-level data integrity.

### 🧠 CI / Dev Improvements

- Added Makefile lint and structure checks (`lint-layout`, `lint-deprecated-exports`).
- Improved cross-platform compatibility (Windows / Git Bash / Linux).
- Simplified developer UX with `make help-layout` and `make venv-shell`.

---

## 🧪 Testing & Validation

- Dry-run and idempotence checks verified on Apple In-App sample exports.
- All ETL stages validated for atomicity and deterministic manifests.
- Provenance reports correctly summarize normalized → processed data transitions.

---

## 🧠 Next Steps

- Extend idempotence to Zepp and iOS ETL stages (B1–B3, Z1–Z3).
- Add integration tests and GitHub Actions CI for `etl-apple` smoke runs.
- Automate release generation via `make release-draft` and `make release-publish`.
- Begin model retraining using multi-snapshot data (Phase M1).

---

## 🧾 Citation

Teixeira, R. M. (2025). _N-of-1 Study – ADHD + Bipolar Disorder (Practicum Part 2)._  
National College of Ireland. GitHub repository:  
[https://github.com/rodrigomarquest/practicum2-nof1-adhd-bd](https://github.com/rodrigomarquest/practicum2-nof1-adhd-bd)

---

⚖️ **License:** [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)  
Supervisor: **Dr. Agatha Mattos**  
Student ID: **24130664**  
Maintainer: **Rodrigo Marques Teixeira**

---


# 🚀 3.0.1 – Tooling & Provenance Refactor

**Release date:** 2025-10-24T03:59:24.565088+00:00  
**Branch:** `main`  
**Author:** Rodrigo Marques Teixeira  
**Project:** MSc AI for Business – Practicum Part 2 (N-of-1 ADHD + BD)

---

## 🔧 Summary

Release 3.0.1

This release strengthens the end-to-end reproducibility and auditability of the N-of-1 ETL → Modeling pipeline.  
All new scripts follow the **atomic write**, **manifest-based provenance**, and **idempotent rerun** guarantees.

---

## 🧩 Highlights

### 🧱 ETL & QC

- Implemented new deterministic Apple In-App ETL stages:
  - `apple_inapp_parse.py` (normalized CSVs)
  - `apple_inapp_qc.py` (QC metrics & markdown report)
  - `apple_inapp_daily.py` (daily aggregates)
  - Orchestrator target `etl-apple` (parse → qc → daily)
- Added atomic manifest writing, per-run logging, and progress visualization.
- Introduced robust dry-run and idempotence testing via:
  - `make idempotence-check`
  - `make atomicity-sim`

### ⚙️ Build & Provenance

- Added new `io_utils.py` primitives for atomic writes and schema hashing.
- Extended `migrate_layout.py` and `intake_zip.py` to standardize raw → extracted structure.
- Integrated provenance audit (`make provenance`) for run-level data integrity.

### 🧠 CI / Dev Improvements

- Added Makefile lint and structure checks (`lint-layout`, `lint-deprecated-exports`).
- Improved cross-platform compatibility (Windows / Git Bash / Linux).
- Simplified developer UX with `make help-layout` and `make venv-shell`.

---

## 🧪 Testing & Validation

- Dry-run and idempotence checks verified on Apple In-App sample exports.
- All ETL stages validated for atomicity and deterministic manifests.
- Provenance reports correctly summarize normalized → processed data transitions.

---

## 🧠 Next Steps

- Extend idempotence to Zepp and iOS ETL stages (B1–B3, Z1–Z3).
- Add integration tests and GitHub Actions CI for `etl-apple` smoke runs.
- Automate release generation via `make release-draft` and `make release-publish`.
- Begin model retraining using multi-snapshot data (Phase M1).

---

## 🧾 Citation

Teixeira, R. M. (2025). _N-of-1 Study – ADHD + Bipolar Disorder (Practicum Part 2)._  
National College of Ireland. GitHub repository:  
[https://github.com/rodrigomarquest/practicum2-nof1-adhd-bd](https://github.com/rodrigomarquest/practicum2-nof1-adhd-bd)

---

⚖️ **License:** [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)  
Supervisor: **Dr. Agatha Mattos**  
Student ID: **24130664**  
Maintainer: **Rodrigo Marques Teixeira**

---


# 🚀 3.0.2 – Tooling & Provenance Refactor

**Release date:** 2025-10-26T15:20:22.794212+00:00  
**Branch:** `main`  
**Author:** Rodrigo Marques Teixeira  
**Project:** MSc AI for Business – Practicum Part 2 (N-of-1 ADHD + BD)

---

## 🔧 Summary

Release 3.0.2

This release strengthens the end-to-end reproducibility and auditability of the N-of-1 ETL → Modeling pipeline.  
All new scripts follow the **atomic write**, **manifest-based provenance**, and **idempotent rerun** guarantees.

---

## 🧩 Highlights

### 🧱 ETL & QC

- Implemented new deterministic Apple In-App ETL stages:
  - `apple_inapp_parse.py` (normalized CSVs)
  - `apple_inapp_qc.py` (QC metrics & markdown report)
  - `apple_inapp_daily.py` (daily aggregates)
  - Orchestrator target `etl-apple` (parse → qc → daily)
- Added atomic manifest writing, per-run logging, and progress visualization.
- Introduced robust dry-run and idempotence testing via:
  - `make idempotence-check`
  - `make atomicity-sim`

### ⚙️ Build & Provenance

- Added new `io_utils.py` primitives for atomic writes and schema hashing.
- Extended `migrate_layout.py` and `intake_zip.py` to standardize raw → extracted structure.
- Integrated provenance audit (`make provenance`) for run-level data integrity.

### 🧠 CI / Dev Improvements

- Added Makefile lint and structure checks (`lint-layout`, `lint-deprecated-exports`).
- Improved cross-platform compatibility (Windows / Git Bash / Linux).
- Simplified developer UX with `make help-layout` and `make venv-shell`.

---

## 🧪 Testing & Validation

- Dry-run and idempotence checks verified on Apple In-App sample exports.
- All ETL stages validated for atomicity and deterministic manifests.
- Provenance reports correctly summarize normalized → processed data transitions.

---

## 🧠 Next Steps

- Extend idempotence to Zepp and iOS ETL stages (B1–B3, Z1–Z3).
- Add integration tests and GitHub Actions CI for `etl-apple` smoke runs.
- Automate release generation via `make release-draft` and `make release-publish`.
- Begin model retraining using multi-snapshot data (Phase M1).

---

## 🧾 Citation

Teixeira, R. M. (2025). _N-of-1 Study – ADHD + Bipolar Disorder (Practicum Part 2)._  
National College of Ireland. GitHub repository:  
[https://github.com/rodrigomarquest/practicum2-nof1-adhd-bd](https://github.com/rodrigomarquest/practicum2-nof1-adhd-bd)

---

⚖️ **License:** [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)  
Supervisor: **Dr. Agatha Mattos**  
Student ID: **24130664**  
Maintainer: **Rodrigo Marques Teixeira**

---


# 🚀 v3.0.3 – AutoExport SoM Labels & Manifest

**Release date:** 2025-10-29 UTC  
**Branch:** `main`  
**Participant/Snapshot:** P000001 / 2025-09-29

## 🔧 Summary

- Added Apple Health **AutoExport State of Mind (SoM)** ingestion and daily label composition.
- Introduced manifest field `real_labels` and expanded counts by source.
- No breaking changes; re-run `make etl-extract` and `make labels` to refresh labels.

## 🧩 Highlights

- AutoExport SoM parser detects latest `StateOfMind-*.csv` under `data/raw/.../apple/autoexport/`.
- Writes normalized file `apple_state_of_mind_autoexport.csv` atomically.
- Label precedence: **apple_autoexport → apple_cda → zepp → synthetic**.
- `labels_manifest.json` now includes:
  - `real_labels`
  - `counts_by_source`
  - `files_used`
- ASCII-only logs; idempotent writes preserved.

## 📊 Label Stats (from labels_manifest.json)

- `dates_total`: **1313**
- `real_labels`: **54**
- `counts_by_source`: {apple_autoexport: 54, apple_cda: 0, zepp: 0, synthetic: 1259}

## 📁 Artifacts

| File                               | Exists |        Size | SHA256                                                             |
| ---------------------------------- | -----: | ----------: | ------------------------------------------------------------------ |
| apple_state_of_mind_autoexport.csv |     ✅ |  5750 bytes | `2bac8a1a7ae8e0153b386a3f0b1787bde39a10fb01bee88882597484f9878716` |
| apple_state_of_mind.csv            |     ✅ | 18533 bytes | `73564f468820e30be9ea94cc541d99a466d59be16518f621626feacfe23bc961` |

## 🧪 Impact on NB2

- Baselines unchanged; 54 real labels now available for training.
- Expect fewer single-class folds and improved class distribution metrics.

## 🔄 Upgrade Notes

1. Copy `StateOfMind-*.csv` to `data/raw/P000001/apple/autoexport/`.
2. Run:

```
make etl-extract PARTICIPANT=P000001 SNAPSHOT=2025-09-29
make labels PARTICIPANT=P000001 SNAPSHOT=2025-09-29
make nb2-run PARTICIPANT=P000001
```

3. Verify manifest at `data/etl/P000001/snapshots/2025-09-29/labels_manifest.json`.

## 📝 Changelog

Commits since `v3.0.2` (up to 5):

```
64bdc75 chore: Extract SoM from Auto Extract app
1fec1cc chore: checkpoint before enduring SoM extratction
e8e197e chore: checkpoint before implementing labels subcommand (safe state post-full)
8eea5b1 chore: checkpoint before Copilot interventions
c1fef18 chore(makefile): stabilize ETL → AI → NB2 workflow targets
```

---

_Generated by automation on 2025-10-29 UTC — release draft v3.0.3._


# 🚀 v3.0.4 – Release Draft

**Release date:** 2025-10-29 UTC  
**Branch:** `main`

This is a small follow-up release draft to allow publishing. See v3.0.3 for the main AutoExport SoM labels changes.

## Notes

- Minor packaging/maintenance release. No code changes affecting the AutoExport SoM behavior beyond v3.0.3.

_Generated automatically to satisfy release workflow._


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


# 🚀 4.0.2 – Environment centralization & Developer Guide

**Release date:** 2025-10-31T10:49:12.207521+00:00  
**Branch:** `v4-main`  
**Author:** Rodrigo Marques Teixeira  
**Project:** MSc AI for Business – Practicum Part 2 (N-of-1 ADHD + BD)

---

## 🔧 Summary

Consolidated Makefile, requirements cleanup, and full developer documentation.

This release strengthens the end-to-end reproducibility and auditability of the N-of-1 ETL → Modeling pipeline.  
All new scripts follow the **atomic write**, **manifest-based provenance**, and **idempotent rerun** guarantees.

---

## 🧩 Highlights

### 🧱 ETL & QC

- Implemented new deterministic Apple In-App ETL stages:
  - `apple_inapp_parse.py` (normalized CSVs)
  - `apple_inapp_qc.py` (QC metrics & markdown report)
  - `apple_inapp_daily.py` (daily aggregates)
  - Orchestrator target `etl-apple` (parse → qc → daily)
- Added atomic manifest writing, per-run logging, and progress visualization.
- Introduced robust dry-run and idempotence testing via:
  - `make idempotence-check`
  - `make atomicity-sim`

### ⚙️ Build & Provenance

- Added new `io_utils.py` primitives for atomic writes and schema hashing.
- Extended `migrate_layout.py` and `intake_zip.py` to standardize raw → extracted structure.
- Integrated provenance audit (`make provenance`) for run-level data integrity.

### 🧠 CI / Dev Improvements

- Added Makefile lint and structure checks (`lint-layout`, `lint-deprecated-exports`).
- Improved cross-platform compatibility (Windows / Git Bash / Linux).
- Simplified developer UX with `make help-layout` and `make venv-shell`.

---

## 🧪 Testing & Validation

- Dry-run and idempotence checks verified on Apple In-App sample exports.
- All ETL stages validated for atomicity and deterministic manifests.
- Provenance reports correctly summarize normalized → processed data transitions.

---

## 🧠 Next Steps

- Extend idempotence to Zepp and iOS ETL stages (B1–B3, Z1–Z3).
- Add integration tests and GitHub Actions CI for `etl-apple` smoke runs.
- Automate release generation via `make release-draft` and `make release-publish`.
- Begin model retraining using multi-snapshot data (Phase M1).

---

## 🧾 Citation

Teixeira, R. M. (2025). _N-of-1 Study – ADHD + Bipolar Disorder (Practicum Part 2)._  
National College of Ireland. GitHub repository:  
[https://github.com/rodrigomarquest/practicum2-nof1-adhd-bd](https://github.com/rodrigomarquest/practicum2-nof1-adhd-bd)

---

⚖️ **License:** [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)  
Supervisor: **Dr. Agatha Mattos**  
Student ID: **24130664**  
Maintainer: **Rodrigo Marques Teixeira**

---
