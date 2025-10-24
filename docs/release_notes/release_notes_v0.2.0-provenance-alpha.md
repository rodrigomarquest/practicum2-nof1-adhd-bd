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
