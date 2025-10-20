# Changelog — N-of-1 Study (Practicum Part 2)

All notable changes to this project will be documented in this file.  
This project adheres to Semantic Versioning where applicable.

## [Unreleased]

- KnowledgeC integration (device-specific schema) and `parse_knowledgec_usage.py`.
- Notebook 02 re-run with rule-based baseline, SHAP top-5, drift metrics.
- Export `best_model.tflite` and latency measurements.
- Finalise LaTeX `main.tex` with updated figures + Appendices C–D.

🧾 CHANGELOG – v2.1.1

Release date: 2025-10-19
Author: Rodrigo Marques Teixeira
Project: MSc AI for Business – Practicum Part 2 (N-of-1 ADHD + BD)
Previous: v2.1.0 – ETL Convergence & EDA Baseline

Current: v2.1.1 – Cardio Stabilization & EDA Path Fix

🚀 Overview

This minor release finalizes the cardiovascular stage stabilization and ensures end-to-end reproducibility of the new ETL architecture.
It introduces atomic write guarantees, UTC-safe datetime normalization, runtime profiling, and dynamic path resolution in the EDA notebooks.
The pipeline is now robust, idempotent, and aligned with the research’s reproducibility standards.

🧱 Core ETL Changes
Type Component Description
🧩 Refactor etl_pipeline.py Simplified orchestration; modular sub-commands (extract, cardio, full) with improved logging.
⚙️ Enhancement cardio_etl.py Integrated the new Timer utility for execution profiling.
⚙️ Enhancement cardio_features.py Fixed tz-aware datetime conversion (utc=True) preventing ValueError at high sample counts.
🧰 Utility helpers/\_write_atomic_csv/json Introduced idempotent atomic writes using temporary files + replace pattern.
📁 Pathing Directory schema Unified under data_ai/Pxxxxxx/snapshots/YYYY-MM-DD (applied repo-wide).
💓 Cardiovascular Stage

Aggregates heart-rate, HRV (SDNN), and sleep intervals into daily features.

Outputs validated for participant P000001 snapshot 2025-09-29.

Files generated:

features_cardiovascular.csv ≈ 86 kB

features_daily_updated.csv ≈ 276 kB

Runtime: ≈ 230 seconds on Windows 10 (local Python 3.13).

Internal manifest tracking (cardio_manifest.json) confirmed.

📊 EDA & Visualization
Type Component Description
🧭 Fix 03_eda_cardio_plus.ipynb Corrected relative paths (removed “/notebooks/” prefix).
🧩 Feature Notebook logic Auto-detects repo root and resolves data_ai/... dynamically.
💾 Output Artifacts Generated charts saved to eda_outputs/ under each snapshot.
📈 Rendering Plotly offline Activated pio.renderers.default = 'notebook' for seamless local use.
🧪 Validation Summary
Check Status Notes
ETL runtime ✅ Stable; measured 230 s on local test.
Atomic write ✅ Temp → final rename verified.
Date parsing ✅ No tz-aware errors post-fix.
Output presence ✅ 2 feature CSVs generated with valid content.
Notebook ✅ Runs end-to-end with figures rendered.
🧭 Next Milestones

Add new participants (P000002–P000003).

Implement 04_modeling_baseline.ipynb for first N-of-1 forecasting experiments.

Build feature_catalogue.md for documentation of engineered variables.

Start CA3 draft (Nov 2025): methods + EDA + baseline results.

🧠 Acknowledgment

This release was produced within the Practicum Part 2 module of the MSc AI for Business, under supervision of Dr. Agatha Mattos, and represents the first stable public milestone of the N-of-1 Longitudinal Phenotyping Pipeline.

⚖️ License

This project remains licensed under the
Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0) license.
© 2025 Rodrigo Marques Teixeira. All rights reserved.

## [v2.0.3] – 2025-10-17

### 🔧 Dependency & Build Standardization

**Summary:**  
This version consolidates all dependency management into a unified and modular structure, adds precise Makefile targets for each environment, and ensures compatibility across Python 3.10 → 3.13 (Windows + Kaggle GPU).

### Added

- **New dependency structure:**
  - `requirements_etl.txt` – Core ETL pipeline (Py 3.13 safe)
  - `requirements_ios.txt` – iOS backup decryption / Screen Time extraction
  - `requirements_ai_kaggle.txt` – Modeling + SHAP explainability (Kaggle GPU)
  - `requirements_dev.txt` – Development / lint / Jupyter utilities
  - Root `requirements.txt` now includes all modular references.
- **Makefile targets:**
  - `install` → ETL-only environment
  - `install-ios` → iOS extraction stack
  - `install-ai` → Kaggle / modeling environment
  - `install-dev` → dev / test environment
  - `install-all` → full installation (aggregated)
- Added clear comments for Python version compatibility and environment isolation.

### Changed

- Replaced old `etl/requirements.txt` include with modular root requirements.
- Refactored `Makefile` to install from root-level requirements and added help docs.
- Updated `.gitignore` to explicitly ignore all decrypted outputs and iOS temp folders.
- Moved ETL-agnostic extraction scripts into `ios_extract/` sub-directory for clarity.

### Fixed

- Compatibility of `iphone-backup-decrypt==0.9.0` with Windows / Python 3.10–3.12.
- Ensured all ETL dependencies compile cleanly on Python 3.13.
- Unified CRLF/LF handling to prevent Git newline warnings on Windows.

### Notes

- Recommended to maintain two venv environments:  
  • `venv-etl` (Python 3.13) for standard ETL/modeling  
  • `venv-ios` (Python 3.10–3.12) for encrypted iOS backup extraction
- This version completes the project structure stabilization phase.  
  Next release (`v2.0.4`) will focus on full Kaggle modeling reproducibility and drift-detection notebooks.

---

## [v2.0.2] – October 2025

**Status:** Structural consolidation complete

### Summary

This release finalises the repository’s folder architecture and naming consistency for the N-of-1 ADHD + BD Practicum Part 2 project.  
All iOS extraction scripts are now fully consolidated under `ios_extract/`, while the global ETL pipeline remains at the project root.

### Added

- Centralised iOS extraction utilities:
  - `ios_extract/decrypt_manifest.py`
  - `ios_extract/export_screentime.py`
  - `ios_extract/extract_deviceactivity.py`
  - `ios_extract/extract_knowledgec.py`
  - `ios_extract/extract_plist_screentime.py`
  - `ios_extract/plist_to_usage.py`
  - `ios_extract/probe_deviceactivity_blobs.py`
  - `ios_extract/quick_post_backup_probe.py`
  - `ios_extract/screentime_ios_backup.py`
  - `ios_extract/smart_extract_plists.py`

### Changed

- Fixed filename typo: `extract_knowledgegc.py` → `extract_knowledgec.py`
- Updated `Makefile` targets to match new script paths
- Improved `.gitignore` with explicit exceptions for `.keep` placeholders  
  and consistent ignoring of decrypted output and manifests

### Removed / Cleaned

- Deleted temporary `manifest_*.tsv` and local generated PDF (`Configuration_Manual_Full.pdf`)
- Removed duplicate `decrypted_output/` under `ios_extract/`
- Eliminated redundant ETL scripts from project root (migrated to `ios_extract/`)

### Notes

- **ETL pipeline:** remains at root (`etl_pipeline.py`)
- **Next milestone:** add `parse_knowledgec_usage.py` once `KnowledgeC.db` schema is confirmed
- Repository is now compliant with the Practicum CA3 submission layout and ready for academic archiving

## [v2.0-pre-ethics] — 2025-10-17

### Added

- **ios_extract/** module:
  - `decrypt_manifest.py` — decrypts Manifest and validates SQLite.
  - `quick_post_backup_probe.py` — probes candidates w/ `flags=1` and on-disk blobs.
  - `smart_extract_plists.py` — adaptive extraction of `DeviceActivity.plist` and `ScreenTimeAgent.plist` (handles API variations in `iphone-backup-decrypt==0.9.0`).
  - `plist_to_usage_csv.py` — heuristics to export daily usage from plists (settings-only snapshots produce empty CSV for provenance).
  - `extract_knowledgec.py` — pulls `KnowledgeC.db` when available.
- **Makefile** with targets: `venv`, `install`, `decrypt`, `probe`, `extract-plists`, `plist-csv`, `extract-knowledgec`, `parse-knowledgec`, `etl`, `clean`, `deepclean`.
- Extended **README.md** documenting iOS extraction workflow and integration with ETL.
- Hardened **.gitignore** to exclude decrypted outputs (`decrypted_output/`, `.plist`, `.db`, `.sqlite*`, etc.) and secrets.

### Changed

- Repository structure updated to include `ios_extract/` and keep PII out of version control.
- ETL docs clarified (segment normalisation S1–S6 and time-zone cutover).

### Security

- Explicit guidance to keep backup passphrases out of code.
- Default `deepclean` target to remove decrypted outputs locally.
