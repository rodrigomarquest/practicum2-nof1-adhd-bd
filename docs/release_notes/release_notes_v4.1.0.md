# Ì∫Ä v4.1.0 ‚Äì Fase 3 ETL Consolidation & Production-Ready Analytics

**Release date:** 2025-11-07T00:00:00Z  
**Branch:** `release/v4.1.0`  
**Author:** Rodrigo Marques Teixeira  
**Project:** MSc AI for Business ‚Äì Practicum Part 2 (N-of-1 ADHD + BD)

---

## Ì∑© Summary

**v4.1.0** represents a major consolidation milestone, completing **Fase 3** of the ETL‚ÜíModeling pipeline with production-ready analytics, unified extract infrastructure for Apple/Zepp variants, and deterministic CLI entrypoints for cardio/sleep/activity domain seeds.

This release **resolves 4 critical data quality issues** (#8, #9, #10, #13), standardizes the snapshot directory layout, and introduces atomic writes with QC validation. All scripts now follow consistent provenance patterns and support dry-run verification.

**Key achievement:** End-to-end reproducibility from raw device exports ‚Üí normalized domain datasets ‚Üí feature engineering, with full audit trail via manifest-based QC.

---

## Ìºø Highlights

### Ì∑± ETL & Data Quality

- **Unified Extract Orchestration**:
  - Consolidated Apple Health export, iTunes backup, AutoExport, and Zepp imports into single discovery pipeline.
  - Snapshot resolution with conflict detection and manifest logging.
  - Support for ZEPP_ZIP_PASSWORD-protected archives.
  
- **Snapshot Path Normalization** (**Closes #9**):
  - Removed redundant `snapshots/` directory nesting.
  - Canonical path: `data/etl/<PID>/<YYYY-MM-DD>/`
  - Migration helper included for legacy layouts.

- **Cardio Output Fix** (**Closes #10**):
  - Corrected PID extraction from snapshot directory.
  - Cardio outputs now write to correct `data/etl/<PID>/<SNAP>/joined/`.

- **Deprecated Cardio Step Cleanup** (**Closes #8**):
  - Removed standalone cardio orchestrator (now integrated into full ETL).
  - Streamlined pipeline architecture.

### Ì∑™ Fase 3 Analytics Pipeline (Production-Ready)

- **Join Coalescence & QC**:
  - Per-domain joins (cardio HR/HRV, activity steps, sleep intervals).
  - Post-join enrichments with validation checksums.
  - Atomic CSV writes with schema hashing.

- **CLI Domain Entrypoints**:
  - `etl_runner activity` ‚Äî seed activity features from extracted data.
  - `etl_runner cardio` ‚Äî join cardiovascular metrics with QC.
  - `etl_runner sleep` ‚Äî aggregate sleep intervals (stub).
  - Dry-run mode with idempotence checks.

- **Notebook Integration**:
  - NB1_EDA_daily.ipynb refactored with Fase 3 architecture.
  - Cross-domain enrichments and validation workflows.

### ‚öôÔ∏è Infrastructure & Developer Experience

- **ETL Namespace & Makefile Modernization**:
  - Unified `make etl` orchestrator with subcommands.
  - Fixed cross-platform shell compatibility (Windows/Bash/Linux).
  - Normalized line endings (LF).

- **Optional Progress Visualization**:
  - `ETL_TQDM=1` environment variable enables tqdm bars for large files.
  - Especially useful for Apple Health export parsing (4M+ records).

- **Activity Import Robustness**:
  - Discovers Apple export.xml under extracted structure.
  - Fallback to daily CSVs if XML unavailable.
  - Home timezone profile support for multi-device scenarios.

- **CLI Dry-Run Semantics**:
  - Harmonized exit codes: dry-run=0, empty real run=2.
  - Better CI/automation compatibility.

### Ì≥ä Data Snapshot Date Resolution (**Closes #13**)

- **Manifest-Based Provenance**:
  - All extract stages log source file, size, modification timestamp.
  - Explicit snapshot date validation against source metadata.
  - Warnings if mismatch detected between `--snapshot` parameter and actual file dates.

### Ì≥ö Documentation & Migration

- **NB1_EDA_MIGRATION.md**:
  - Consolidation report documenting Fase 1‚Üí3 analytics workflow.
  - Migration guides for legacy snapshot layouts.
  - Best practices for cross-participant analysis.

---

## Ì∑™ Testing & Validation

- ‚úÖ Cardio/sleep seed tests pass (atomic writes, manifest validation).
- ‚úÖ Extract QC validates against 30-day sample from P000001.
- ‚úÖ Zepp parsing tested with AutoExport CSVs and export zips.
- ‚úÖ Cross-platform Makefile validated on Windows (bash.exe) and Linux.
- ‚úÖ Dry-run mode tested; idempotence checks pass for re-runs.

---

## Ì≥à Statistics

- **Commits since v4.0.4:** 30 commits
- **Main contributors:** Rodrigo Teixeira (27), Rodrigo M Teixeira (3)
- **Issues closed:** 4 (#8, #9, #10, #13)
- **Domains implemented:** 3 (activity, cardio, sleep stubs)
- **Notebooks refactored:** 1 (NB1_EDA_daily)
- **Lines of ETL code:** ~2,500 across new domain modules

---

## Ì¥¨ Next Steps

- **v4.2.0:** Extend idempotence guarantees to Zepp and iOS stages (B1‚ÄìB3, Z1‚ÄìZ3).
- **v4.2.0:** Remove legacy etl_modules shims after full migration to `src/domains`.
- **Q4 2025:** GitHub Actions CI for ETL smoke runs on each push.
- **Phase M1:** Model retraining with multi-snapshot stacked features.

---

## Ì∂ò Known Limitations

- Issue #7 (Mandatory Release Publish Title): Partially addressed. RELEASE_TITLE now enforced for PR/publish targets.
- Issue #11 (make clean-all): Pending verification; may be environment-specific (Windows path handling).
- Issue #14 (Legacy shim cleanup): Scheduled for v4.2.0; backward compatibility maintained for now.

---

## Ì¥ê Reproducibility & Audit

- **Provenance:** `provenance/etl_provenance_report.csv` includes per-file source hashes.
- **Data audit:** `provenance/data_audit_summary.md` documents QC metrics.
- **Dependencies:** Frozen via `pip freeze` snapshot at release time.
- **Atomic writes:** All ETL outputs verified via SHA256 manifest.

---

## Ì∑æ Citation

Teixeira, R. M. (2025). _N-of-1 Study ‚Äì ADHD + Bipolar Disorder (Practicum Part 2)._  
National College of Ireland, MSc AI for Business. GitHub repository:  
[https://github.com/rodrigomarquest/practicum2-nof1-adhd-bd](https://github.com/rodrigomarquest/practicum2-nof1-adhd-bd)

Release v4.1.0. Retrieved from: https://github.com/rodrigomarquest/practicum2-nof1-adhd-bd/releases/tag/v4.1.0

---

‚öñÔ∏è **License:** [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)  
**Supervisor:** Dr. Agatha Mattos  
**Student ID:** 24130664  
**Maintainer:** Rodrigo Marques Teixeira
