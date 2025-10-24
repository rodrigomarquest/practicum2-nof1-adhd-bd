# 🚀 3.0.1 – Tooling & Provenance Refactor

**Release date:** 2025-10-24T03:36:14.840117+00:00  
**Branch:** `main`  
**Author:** Rodrigo Marques Teixeira  
**Project:** MSc AI for Business – Practicum Part 2 (N-of-1 ADHD + BD)

---

## 🔧 Summary

Release 3.0.1 — Tooling & Provenance Refactor

This release focuses on reproducibility, provenance, and repo hygiene. Highlights include a small but impactful refactor of developer tooling and release engineering, stronger provenance artifacts for ETL runs, and several bugfixes that improve cross-platform behavior.

Key points:

- Modularized operational scripts: `make_scripts/` was reorganized into domain packages (`make_scripts/apple`, `make_scripts/ios`, `make_scripts/zepp`, `make_scripts/utils`) to improve discoverability and maintainability. Top-level shims and backups were kept during staged migration.
- Deterministic migration plans: Added `tools/audit/reorg_import_plan.json` and `tools/audit/reorg_plan.json` to make reorganization deterministic and reviewable.
- Provenance & manifest tooling: Added and canonicalized manifest helpers under `make_scripts/utils/` (snapshot locking, manifest building and printing). Release asset manifests are produced for every release.
- Cleanups & safety: Executed an audited `clean_data` run with backups; repaired installer scripts and several shell shims; improved Makefile canonicalization and lint checks.
- Tests & verification: Ran focused import smoke tests and Makefile linting after changes to validate behavior before publishing.

All new or refactored scripts adhere to the project's guarantees: **atomic write**, **manifest-based provenance**, and **idempotent rerun**.

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
