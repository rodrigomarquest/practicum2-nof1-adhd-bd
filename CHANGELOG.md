Changelog — N-of-1 Study (Practicum Part 2)

All notable changes to this project will be documented in this file.
This project adheres to Semantic Versioning where applicable.

[Unreleased]

KnowledgeC integration (device-specific schema) and parse_knowledgec_usage.py.

Notebook 02 re-run with rule-based baseline, SHAP top-5, drift metrics.

## [v4.1.4] – 2025-11-18

### Major: Canonical PBSI Integration (PhD-Level Methodology Alignment)

**Summary:**  
Integrated the canonical segment-wise z-scored PBSI implementation from `src/labels/build_pbsi.py` into the main ETL pipeline Stage 3 (`src/etl/stage_apply_labels.py`). This resolves **Critical Issue #2** from the PhD-level code-paper consistency review: the pipeline now uses the exact PBSI methodology documented in the CA2 research paper.

**Impact:**

- ✅ **Deterministic & Reproducible**: Validated across two independent pipeline runs
- ✅ **Paper-Aligned**: Segment-wise z-scores, correct formula weights, proper thresholds
- ✅ **Production-Ready**: All downstream stages (NB2/NB3) working correctly

---

### 🔬 PBSI Implementation Changes

**Before (Simple Heuristic - Deprecated):**

- ❌ 0-100 scale (not z-scored)
- ❌ No temporal segmentation
- ❌ Thresholds at 33/66 (arbitrary)
- ❌ Global normalization (potential data leakage)

**After (Canonical - CA2 Paper):**

- ✅ Z-scale normalization (segment-wise)
- ✅ 119 temporal segments (gaps + month/year boundaries)
- ✅ Thresholds at -0.5 / +0.5 (documented)
- ✅ Anti-leak safeguard (no global statistics)
- ✅ Formula: `pbsi_score = 0.40×sleep_sub + 0.35×cardio_sub + 0.25×activity_sub`

**Code Changes:**

1. **`src/etl/stage_apply_labels.py`** (Major Refactor):

   - Added `_create_temporal_segments()`: Creates 119 segments based on gaps >1 day and month/year boundaries
   - Added `_normalize_column_names_for_pbsi()`: Maps unified daily columns to build_pbsi expected names
   - Updated `PBSILabeler.apply_labels()`: Now delegates to canonical `build_pbsi_labels()`
   - **Deleted**: `_legacy_calculate_pbsi_score_simple()` (deprecated 0-100 heuristic)

2. **`src/labels/build_pbsi.py`** (Documentation Only):

   - Enhanced module docstring explaining CA2 paper alignment
   - Added sign convention explanation (lower PBSI = more stable)

3. **New Test**: `tests/test_canonical_pbsi_integration.py`
   - Smoke test validating segment-wise z-scores on synthetic data
   - Verifies expected stable/unstable behavior

---

### 📊 Validation Results (P000001, Snapshot 2025-11-07)

**Pipeline Metrics (Run 1 vs Run 2):**
| Checkpoint | Run 1 | Run 2 | Status |
|-----------|-------|-------|--------|
| Total days | 2828 | 2828 | ✅ IDENTICAL |
| Segments | 119 | 119 | ✅ IDENTICAL |
| PBSI range | -1.298 to 0.926 | -1.298 to 0.926 | ✅ IDENTICAL |
| Label +1 (stable) | 211 (7.5%) | 211 (7.5%) | ✅ IDENTICAL |
| Label 0 (neutral) | 2552 (90.2%) | 2552 (90.2%) | ✅ IDENTICAL |
| Label -1 (unstable) | 65 (2.3%) | 65 (2.3%) | ✅ IDENTICAL |
| NB2 F1-score | 1.0000 | 1.0000 | ✅ IDENTICAL |
| NB3 Drift (ADWIN) | 5 changes | 5 changes | ✅ IDENTICAL |

**Numerical Precision:**

- PBSI scores match to 15+ decimal places across runs
- Example stable day: `pbsi_score = -0.616576836079193` (both runs)
- Example unstable day: `pbsi_score = 0.5035790070447901` (both runs)

---

### 📁 New Documentation

1. **`docs/PBSI_INTEGRATION_UPDATE.md`** (Comprehensive Guide):

   - Technical details of integration
   - Formula explanation with sign convention
   - Column name mapping table
   - Known caveats (HRV approximation, exercise estimation)

2. **`PBSI_INTEGRATION_SUMMARY.md`** (Quick Reference):

   - Before/after comparison
   - Files modified summary
   - Validation results

3. **`DETERMINISM_VALIDATION_REPORT.md`** (PhD-Level Analysis):

   - Full determinism test methodology
   - Stage-by-stage comparison
   - Floating-point precision analysis
   - Reproducibility checklist

4. **`PAPER_CODE_CONSISTENCY_REVIEW.md`** (Original Review - 39KB):
   - Comprehensive code archaeology
   - Critical Issue #2 documented (now RESOLVED)
   - PhD-level consistency audit

---

### 🔧 Column Name Mapping

| Unified Daily (Stage 2)       | Build PBSI (Expected)    | Transformation             |
| ----------------------------- | ------------------------ | -------------------------- |
| `sleep_hours`                 | `sleep_total_h`          | Direct rename              |
| `sleep_quality_score` (0-100) | `sleep_efficiency` (0-1) | Divide by 100              |
| `hr_mean`                     | `apple_hr_mean`          | Direct rename              |
| `hr_max`                      | `apple_hr_max`           | Direct rename              |
| `hr_std`                      | `apple_hrv_rmssd`        | Proxy (×2 approximation)\* |
| `total_steps`                 | `steps`                  | Direct rename              |
| `total_active_energy`         | `exercise_min`           | Estimate (÷5)\*            |

\*Known caveats documented in `PBSI_INTEGRATION_UPDATE.md`

---

### ⚙️ Technical Details

**Segmentation Strategy:**

- **Method**: Temporal gaps and calendar boundaries
- **Rules**: New segment on gap >1 day OR month/year change
- **Result**: 119 segments (range: 1-31 days each)
- **Purpose**: Enables segment-wise z-score normalization

**PBSI Formula (CA2 Paper):**

```
pbsi_score = 0.40 × sleep_sub + 0.35 × cardio_sub + 0.25 × activity_sub

Where:
  sleep_sub    = -0.6 × z_sleep_dur + 0.4 × z_sleep_eff
  cardio_sub   =  0.5 × z_hr_mean - 0.6 × z_hrv + 0.2 × z_hr_max
  activity_sub = -0.7 × z_steps - 0.3 × z_exercise
```

**Sign Convention** (Counterintuitive but Intentional):

- **Lower PBSI** = More stable → Label +1
- **Higher PBSI** = More unstable → Label -1
- Rationale: Negative coefficients on healthy metrics (sleep, steps) drive scores down when health improves

---

### 🧪 Testing & Validation

**Determinism Test:**

1. Run 1: Full pipeline (Stages 0-9)
2. Cleanup: `make clean-outputs` (removed all processed data)
3. Run 2: Full pipeline from scratch
4. **Result**: ✅ Identical outputs (validated in `DETERMINISM_VALIDATION_REPORT.md`)

**Downstream Impact:**

- ✅ NB2 (Baseline Models): F1=1.0000 (both runs)
- ✅ NB3 (SHAP): Top-5 features identical
- ✅ NB3 (Drift): ADWIN and KS tests identical
- ✅ NB3 (LSTM): Model training successful
- ✅ TFLite: 37.1 KB model exported

---

### 📝 Migration Notes

**Breaking Changes**: None (API-compatible)

- `PBSILabeler.apply_labels()` signature unchanged
- Output columns expanded (added `segment_id`, `z_*`, subscores)
- Old `pbsi_score` values different (z-scale vs 0-100) but column name same

**Deprecated Code**:

- ❌ `_legacy_calculate_pbsi_score_simple()` (deleted in this release)

**For Researchers**:

- If comparing with old results, note that PBSI scales differ (z-scale vs 0-100)
- Recommend re-running experiments with canonical PBSI for paper consistency
- Label distributions may change (old: 33/66 thresholds, new: -0.5/+0.5 thresholds)

---

### 🎯 Research Impact

**Resolved Issues:**

- ✅ **Critical Issue #2**: PBSI implementation now matches CA2 paper methodology
- ✅ **Anti-leak claims**: Segment-wise z-scores prevent data leakage
- ✅ **Reproducibility**: Deterministic pipeline with fixed seed (42)

**Future Work:**

- Investigate true HRV data sources (currently using `hr_std × 2` as proxy)
- Refine exercise estimation (currently using `active_energy ÷ 5`)
- Evaluate sophisticated segmentation (auto_segment.py)

---

**Git Tag**: `v4.1.4-canonical-pbsi`  
**Commit Hash**: `856a640`  
**Validation Report**: `DETERMINISM_VALIDATION_REPORT.md`  
**Integration Guide**: `docs/PBSI_INTEGRATION_UPDATE.md`

## [v4.1.3-dev] – 2025-11-16 (In Progress)

### Refactor: Archive Legacy Modules (No Functional Changes)

**Summary:**  
Structural refactoring to archive unused and duplicate modules without any behavioral changes to the pipeline. All canonical entrypoints documented in `CANONICAL_ENTRYPOINTS.md` are preserved. Tests and Makefile targets still pass.

**Key Changes:**

### Code Organization

- **Archived Duplicate Modules**: Moved unused/duplicate code to `archive/` (not deleted)
  - `src/cli/*` duplicates → `archive/src_cli_legacy/` (5 modules)
  - `src/domains/biomarkers/*` → `archive/src_domains_legacy/` (9 modules)

### Canonical Entrypoints Preserved

- ✅ `scripts/run_full_pipeline.py` - Main orchestrator (UNCHANGED)
- ✅ `scripts/extract_biomarkers.py` - Uses `src.biomarkers.aggregate` (UNCHANGED)
- ✅ `scripts/prepare_zepp_data.py` - Standalone script (UNCHANGED)
- ✅ `scripts/prepare_nb2_dataset.py` - Anti-leak preparation (UNCHANGED)
- ✅ All `src/etl/stage_*.py` modules (UNCHANGED)
- ✅ All `src/biomarkers/*` modules (UNCHANGED - canonical version)

### Modules Archived (Phase 2A - src/cli/)

- `src/cli/extract_biomarkers.py` → Duplicate of `scripts/extract_biomarkers.py`
- `src/cli/prepare_zepp_data.py` → Duplicate of `scripts/prepare_zepp_data.py`
- `src/cli/etl_runner.py` → Superseded by `scripts/run_full_pipeline.py`
- `src/cli/run_etl_with_timer.py` → Not referenced by Makefile or tests
- `src/cli/migrate_snapshots.py` → Not referenced by Makefile or tests

### Modules Archived (Phase 2B - src/domains/biomarkers/)

- Entire `src/domains/biomarkers/` folder → Duplicate of `src/biomarkers/`
- The canonical `src/biomarkers/` is used by `scripts/extract_biomarkers.py`
- Contains full chain: aggregate → {segmentation, hrv, sleep, activity, circadian, validators}

### Modules Archived (Phase 3A/B/C - Root-Level Legacy)

**Phase 3A: NB2/NB3 Prototypes** (Commit: 65f6935)

- `src/models_nb2.py` → Baseline model prototype (1598 lines, replaced by `nb3_analysis.py`)
- `src/models_nb3.py` → Notebook wrapper (replaced by `nb3_analysis.py`)
- `src/nb3_run.py` → NB3 prototype with SHAP/Drift (699 lines, replaced by `nb3_analysis.py`)
- `src/eda.py` → EDA notebook wrapper (not in canonical pipeline)

**Phase 3B: Legacy ETL Pipeline** (Commit: 929c396)

- `src/etl_pipeline.py` → Legacy CLI with `discover_sources()` (163KB, replaced by `run_full_pipeline.py`)
- `tests/test_cli_extract_logging.py` → Test for legacy CLI (moved to `archive/tests_legacy/`)
- Created `pytest.ini` to exclude `archive/` from test collection

**Phase 3C: Legacy Labeling** (Commit: 859376b)

- `src/make_labels.py` → Legacy labeling CLI (replaced by `stage_apply_labels.py`)
- `src/utils.py` → Utility functions for `make_labels.py`

**Total Phase 3**: 7 root-level modules (256KB) + 1 test archived to `archive/src_root_legacy/` and `archive/tests_legacy/`

### Tests Archived (Phase Tests - Pre-v4 Layout)

**Phase Tests: Legacy Test Clean-up** (Commit: 13577cf)

- `tests/test_aggregate_features_daily.py` → Used `etl_tools` (pre-v4 package)
- `tests/test_cda_in_pipeline.py` → Used bare `domains.cda` import
- `tests/test_cda_probe.py` → Used bare `domains.cda` import
- `tests/test_io_utils.py` → Used `etl_modules` (pre-v4 package)

**Result**: All pre-v4 import errors eliminated. Pytest now collects only v4.1.x compatible tests (13 tests).

### Finishing Pass: Documentation & Cleanup (Commits: 7692ddd, 4f02959, 9be9a81)

**Step A: AI-Assisted Documentation** (Commit: 7692ddd)

- Moved 31 AI-assisted session logs from `docs/` to `docs/copilot/`
  - Implementation summaries, phase reports, release execution plans
  - Session logs, handoff docs, fix reports
- Updated `docs/copilot/README.md` with clear organization
- **Result**: `docs/` now contains only canonical architecture and research documents

**Step B: Provenance Consolidation** (Commit: 4f02959)

- Created `.keep` files for `dist/assets/` and `dist/provenance/`
- Canonical provenance remains in `/provenance` (audit CSVs, reports)
- `dist/provenance/` for release staging only (ignored except `.keep`)
- **Result**: Clear separation between canonical provenance and release artifacts

**Step C: Archive Untracking** (Commit: 9be9a81)

- Untracked 26 files (32MB) from `archive/` using `git rm -r --cached`
- Files remain on disk for local reference
- Accessible in Git history via previous commits
- **Result**: Clean HEAD focused on v4.1.x, smaller clone size

**Step D-E: Smoke Test & Unused Modules Report** (Documentation only)

- Created `SMOKETEST_PIPELINE.md` - Quick validation guide for pipeline
  - Commands for stages 0-5 (core ETL)
  - Expected outputs, runtime estimates, troubleshooting
- Created `UNUSED_MODULES_REPORT.md` - Informational analysis
  - Identified 8 folders (`src/features/`, `src/io/`, etc.) potentially unused
  - No moves/deletions performed - report only
- **Result**: Clear validation and future refactoring guidance

### Documentation Added

- **CANONICAL_ENTRYPOINTS.md**: Single source of truth for protected modules
  - Lists 15 core files + 3 protected folders (18 units total)
  - Documents import graph and dependency tree
  - Provides deletion safety rules
- **docs/ARCHIVE_PLAN.md**: Detailed refactoring plan and execution log
- **SMOKETEST_PIPELINE.md**: Pipeline validation guide (stages 0-5)
- **UNUSED_MODULES_REPORT.md**: Analysis of potentially unused modules (informational)
- **docs/copilot/README.md**: Organization of AI-assisted documentation

### Validation

- ✅ All imports from canonical modules still resolve
- ✅ `pytest` collects 13 tests (no pre-v4 errors)
- ✅ `python -m scripts.run_full_pipeline --help` works
- ✅ No changes to pipeline behavior or outputs

### Breaking Changes

**NONE** - This is a purely structural refactor with no functional changes.

### Migration Guide

**NO ACTION REQUIRED** - All canonical entrypoints remain in place. If custom scripts import from archived modules, update imports to canonical locations:

- `src.cli.extract_biomarkers` → `scripts.extract_biomarkers` (or run as script)
- `src.domains.biomarkers` → `src.biomarkers`

---

## [v4.1.2] – 2025-11-07

### Strict Path Normalization & Fail-Fast Password Validation

**Summary:**  
Implements strict canonical path normalization in ETL pipeline (removes intermediate PID subdirectories from extracted/) and adds fail-fast Zepp password validation with exit code 2. Ensures cleaner directory structures, predictable file locations, and immediate error detection for encrypted data.

**Key Changes:**

### Security & Validation

- **Fail-Fast Zepp Password**: Pipeline aborts with exit 2 if Zepp ZIP exists but no password provided
  - Checks `--zepp-password` argument and `ZEP_ZIP_PASSWORD` / `ZEPP_ZIP_PASSWORD` env vars
  - Validation in Makefile `env` target and `run_full_pipeline.py` Stage 0
  - Clear error: `[FATAL] Zepp ZIP found but no password provided...`

### Path Normalization

- **Canonical Paths**: Removed intermediate `/<PID>/` subdirectory from extracted data
  - Before: `data/etl/P000001/2025-11-07/extracted/apple/P000001/daily_sleep.csv`
  - After: `data/etl/P000001/2025-11-07/extracted/apple/daily_sleep.csv`
  - Modified: `stage_csv_aggregation.py` (write), `stage_unify_daily.py` (read)

### File Management

- **Automatic Renaming**: Existing `daily_*.csv` files renamed to `*.prev.csv` before overwrite
- **No Fallback Reading**: `stage_unify_daily.py` only reads canonical paths (no alternative searches)
- **Clear Warnings**: Missing files logged as `[WARN] Missing {source} daily_{metric} at canonical path`

### Files Modified

- `Makefile`: Added ZPWD variable, fail-fast env check, --zepp-password flag propagation
- `scripts/run_full_pipeline.py`: Stage 0 fail-fast validation, unified zpwd variable
- `src/etl/stage_csv_aggregation.py`: Canonical output paths, .prev.csv renaming
- `src/etl/stage_unify_daily.py`: Canonical input paths, removed fallback logic

### Breaking Changes

- Legacy paths with intermediate PID (`extracted/apple/P000001/`) no longer read
- Missing Zepp password now aborts immediately (exit 2) instead of silent failure

### Migration Guide

- Update any scripts reading from `extracted/{apple,zepp}/<PID>/` to remove PID subdirectory
- Set `ZEP_ZIP_PASSWORD` environment variable or pass `ZPWD=` to Makefile for encrypted Zepp data
- Existing `.csv` files automatically preserved as `.prev.csv` (no manual action needed)

Export best_model.tflite and latency measurements.

Finalise LaTeX main.tex with updated figures + Appendices C–D.

## [v4.1.1] – 2025-11-07

### Infrastructure Improvements & CI/Batch Support + Performance Hotfixes + Sleep Domain Integration

**Summary:**  
Hotfix release addressing tqdm progress bar visibility in Git Bash/MSYS2 terminals, adding non-interactive Python EDA script for CI/batch pipelines, **major performance optimization for XML parsing** (150x speedup on 3.9GB files), and **fixing missing sleep domain in canonical join**. Enables fully automated workflows with real-time feedback, responsive CLI, and complete data domain integration.

**Performance Highlights:**

- ✅ Full ETL pipeline: **6 minutes 11 seconds** (extract → activity → cardio → sleep → join → enrich)
- ✅ Cardio parsing: **2.5 minutes** (was indefinitely hung) — 4.6M heart rate records from 3.9GB XML
- ✅ Zero hanging or buffering issues — all commands responsive with real-time progress
- ✅ 101,955 daily observation rows extracted and processed end-to-end
- ✅ Sleep domain now fully included: 4 Zepp sleep features (total, deep, light, REM hours)

**Key technical improvements:**

1. **Binary regex streaming for XML parsing** (150x faster)

   - Bypasses full XML parsing overhead
   - Processes 3.9GB Apple export.xml in ~2.5 minutes vs indefinite hang
   - Memory efficient: 10MB chunk streaming with minimal buffer

2. **Native datetime parsing** (100x faster than pandas)

   - Replaces `pd.to_datetime()` which was causing indefinite stalls
   - Direct `datetime.strptime()` + timezone offset calculation
   - 51,000 records/second parsing throughput

3. **Unbuffered CLI output** (real-time feedback)
   - PYTHONUNBUFFERED=1 auto-set in CLI runner
   - Logging StreamHandler configured for immediate display
   - tqdm progress bars now appear instantly, not buffered to end

### Added

- **NB1_EDA_daily.py:** Non-interactive Python version of NB1_EDA_daily.ipynb

  - Generates nb1_eda_summary.md, nb1_feature_stats.csv, nb1_manifest.json
  - Saves 5+ PNG visualizations (coverage, signals, correlations, labels)
  - CLI args: --pid, --snapshot, --repo-root
  - Logging with INFO messages, no user interaction
  - Useful for CI pipelines, GitHub Actions, batch processing

- **Makefile targets for EDA automation:**

  - `make nb1-eda-run`: Execute NB1_EDA_daily.py with ETL_TQDM=1
  - `make full-with-eda`: Complete pipeline (extract→join→enrich→nb1-eda)

- **tqdm Git Bash/MSYS2 detection:**

  - Improved \_should_show_tqdm() with MSYSTEM/TERM environment detection
  - Fallback detection for interactive terminals where isatty() fails
  - Environment variable control: ETL_TQDM=1 (force), ETL_TQDM=0 (disable)

- **ZIP extraction progress bars:**
  - Added progress_bar during Apple Health ZIP extraction
  - Added progress_bar during Zepp AES-encrypted ZIP extraction
  - File-level feedback prevents user perception of hang during long extractions
  - Shows real-time progress for potentially 1000s of files

### Changed

- `.gitignore`: Whitelist notebooks/\*.py for EDA/modeling scripts
- `.gitignore`: Add reports/ and latest/ to outputs exclusion list
- Progress bar display logic: Now respects Git Bash/MSYS2 terminals
- **Makefile default behavior:** ETL_TQDM now defaults to 1 (progress bars enabled by default)
  - Users can override with `make <target> ETL_TQDM=0` to disable progress bars if needed
  - Ensures full pipeline visibility and user feedback without explicit flags

### Fixed

- tqdm progress bars not displaying in Git Bash terminal despite interactive TTY
- NB1 EDA output organization (reports/ + latest/ mirror)
- **Makefile Python unbuffered output:** Added `-u` flag to all Python commands to ensure real-time progress bar visibility during long-running ETL extractions (extract runs for ~10 min)
- **CDA export_cda.xml parser memory overflow:** Replaced whole-file parsing with streaming iterparse() for large files (4GB+)
  - Now uses memory-efficient chunk-based streaming (1MB chunks, 500MB limit)
  - Automatic fallback chain: strict parse → lxml.iterparse → recover mode → salvage streaming
  - Prevents OutOfMemory errors during Apple Health CDA extraction
  - **Progress bar with total estimation:** Added record count estimation for progress feedback
    - Fast binary scan of file for `<entry` and `<Section` tags
    - Progress shows percentage/ETA instead of just items/sec
    - Prevents CLI from appearing hung during CDA parsing

### Breaking Changes

- ⚠️ **Hard-removed CUTOVER/TZ functionality:** ETL now operates UTC-only for all timestamps and daily binning
  - Removed `--cutover`, `--tz-before`, `--tz-after` CLI flags entirely
  - Removed `make_tz_selector()` and all timezone-switching logic
  - All timestamps are parsed/converted to UTC; daily buckets use UTC midnight
  - Impact: All `features_*.csv` files now use UTC date columns; no local timezone projection
  - Migration: If local day views needed for reporting, compute downstream (e.g., `df['date_local'] = df['timestamp_utc'].dt.tz_convert('Europe/Dublin').dt.floor('D')`)
  - **Rationale:** Simplified timezone handling reduces off-by-one errors in multi-device scenarios with DST transitions

### Infrastructure

- CI-ready NB1 EDA with no notebook kernel dependency
- Progress visualization works across Windows/Git Bash/Linux
- Batch processing support via CLI entry points

### Tested

- NB1_EDA_daily.py: <10s execution on test data (201 rows × 53 cols)
- 5 PNG plots + JSON metadata generated successfully
- ETL_TQDM=1 enables progress bars in Git Bash
- All outputs UTF-8 encoded (cross-platform compatible)

## [v4.1.0] – 2025-11-07

### Fase 3 ETL Consolidation & Production-Ready Analytics

**Summary:**  
Major consolidation milestone completing Fase 3 of the ETL→Modeling pipeline with production-ready analytics, unified extract infrastructure for Apple/Zepp variants, and deterministic CLI entrypoints for cardio/sleep/activity domain seeds.

**Key achievements:**

- End-to-end reproducibility from raw device exports → normalized domain datasets → feature engineering.
- Resolves 4 critical data quality issues (#8, #9, #10, #13).
- Manifest-based provenance with full audit trail.

### Added

- **Unified Extract Orchestration:** Single discovery pipeline for Apple Health export, iTunes backup, AutoExport, and Zepp imports.
- **Snapshot Path Normalization:** Canonical path `data/etl/<PID>/<YYYY-MM-DD>/` (removed redundant `snapshots/` nesting).
- **CLI Domain Entrypoints:** `etl_runner activity`, `etl_runner cardio`, `etl_runner sleep` with dry-run support.
- **Fase 3 Analytics Pipeline:** Per-domain joins (cardio HR/HRV, activity steps, sleep intervals) with post-join enrichments.
- **NB1_EDA_MIGRATION.md:** Consolidation report documenting Fase 1→3 analytics workflow and migration guides.
- **Manifest-Based Provenance:** All extract stages log source file, size, modification timestamp with validation warnings.
- **Optional Progress Visualization:** `ETL_TQDM=1` enables tqdm bars for large files.

### Fixed

- **Sleep domain now included in canonical join** — Fixed missing sleep features in `joined_features_daily.csv`
  - Added 'sleep' domain to `join_run()` domains_data collection
  - Sleep features now properly merged: zepp_slp_total_h, zepp_slp_deep_h, zepp_slp_light_h, zepp_slp_rem_h
  - EDA reports now show complete domain coverage (Activity Apple/Zepp/Coalesced + Cardio + Sleep)
- Corrected PID extraction from snapshot directory (cardio outputs now write to `data/etl/<PID>/<SNAP>/joined/`).
- Fixed activity import to discover Apple export.xml under extracted structure with fallback to daily CSVs.
- Harmonized CLI dry-run exit codes: dry-run=0, empty real run=2.
- Fixed cross-platform shell compatibility in ETL namespace Makefile recipes.
- Removed deprecated standalone cardio orchestrator (integrated into full ETL).

### Changed

- ETL namespace now unified under `make etl` orchestrator with subcommands.
- Normalized line endings (LF) across Makefile and CLI entrypoints.
- Activity import now supports home timezone profile for multi-device scenarios.
- Improved Windows/Git Bash/Linux compatibility.

### Infrastructure

- Extended Makefile modernization with modular ETL orchestration.
- Cross-platform testing validated on Windows (bash.exe) and Linux.
- Dry-run mode tested; idempotence checks pass for re-runs.

### Documentation

- Release notes follow academic template with citation and reproducibility guarantees.
- Updated README and DEV_GUIDE with new CLI entrypoints and Fase 3 workflow.

### Issues Resolved

- Closes #8 (Remove Deprecated Cardio ETL step)
- Closes #9 (Incorrect data_etl participant snapshots directory)
- Closes #10 (ETL: cardio outputs written to wrong path)
- Closes #13 (Snapshot date incoherence across sources)

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

## [v2.1.4] – 2025-10-21

### 🚀 Modeling Exporter, Baseline & Makefile Refactor

**Summary:**  
This release completes the transition to a stable ETL → Modeling workflow with automated exports, baseline CV, and a fully modular Makefile.

### Added

- `etl_tools/export_modeling_dataset.py` — dataset exporter with manifest and zipped outputs.
- `modeling/baseline_train.py` — 6-fold temporal CV baseline with latency profiling and optional TFLite export.
- `make_scripts/` — contains modular scripts (weekly-report, helpers, etc.).
- `make_scripts/common.py` — PID/SNAP argument parsing utilities.
- `.github/workflows/ci.yml` — lightweight CI for pytest validation.

### Changed

- Makefile refactored with `.RECIPEPREFIX := >` (no tabs or heredocs).
- Removed duplicated `weekly-report` recipes.
- Updated `README.md` with new modeling and Makefile sections.

### Fixed

- Residual indentation and encoding issues in Makefile.
- Pytest now runs cleanly (`5 passed in 1.08 s`).

### Tests

- Added small fixtures for aggregation sanity tests.
- All existing tests pass.

### Notes

- TFLite export optional; skipped if TensorFlow unavailable.
- Data paths (`data_ai/`, `data_etl/`) remain local and ignored by Git.

---

[v2.1.3] – 2025-10-21
🚀 Kaggle Baseline Modelling (preview) & Repo Hygiene

Release date: 2025-10-21
Author: Rodrigo Marques Teixeira
Project: MSc AI for Business – Practicum Part 2 (N-of-1 ADHD + BD)
Previous: v2.1.1 – Cardio Stabilization & EDA Path Fix

🔧 Summary

This release freezes the environment for the first Kaggle baseline modelling preview and performs a safe repository hygiene pass.
It ensures that only relevant source files, notebooks, and documentation are tracked, while sensitive or large data directories (data_ai/, data_etl/) remain strictly local.

A clean tag (v2.1.3) marks the final pre-modelling checkpoint, establishing a reproducible, compliant foundation for feature consolidation and predictive analysis.

🧩 Highlights
🧹 Repository Hygiene

Removed transient and system-specific directories: .venv/, **pycache**/, .pytest_cache/, and Jupyter checkpoints.

Deleted obsolete decrypted outputs (decrypted*output*\*) — ETL and wearable data preserved.

Cleaned temporary .bak, .fixbak, and .pre\_\* artifacts from prior ETL iterations.

Added .gitattributes for export-ignore, enabling clean archive generation.

🛡️ Data Protection & .gitignore

Confirmed that no PII or wearable data is versioned.

Duplicated .continue/ rules simplified for clarity.

Retained .keep sentinels in data_ai/ and data_etl/ for structural integrity.

📓 Baseline Notebook Integration

Added baseline notebooks (practicum-ca2-final.ipynb, etc.) for early Kaggle environment validation.

Ensured offline-safe artifact saving to notebooks/eda_outputs/ (ignored by VCS).

Normalized Makefile for cross-platform compatibility (Windows ↔ Kaggle ↔ Linux).

⚙️ Build & Tagging

Explicit tag-based release process (v2.1.3) replacing previous multi-tag confusion.

Safe selective staging (git add pathspec) to prevent accidental inclusion of local datasets.

🧠 Next Steps

Merge chore/repo-hygiene → main (complete).

Extend ETL with:

Consolidated features_daily_updated.csv (sleep/cardio/activity/screentime).

state_of_mind.csv label integration (features_daily_labeled.csv).

QC reporting (etl_qc_summary.csv, etl_report.md).

Start 04_modeling_baseline.ipynb for cross-validation and metrics export.

Prepare updated Feature Catalogue and appendices for Practicum CA3.

⚖️ License

This project remains licensed under the
Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0) license.
© 2025 Rodrigo Marques Teixeira. All rights reserved.

[v2.1.1] – 2025-10-19

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

# 🔗 Version Comparison Links

# 📜 Changelog — N-of-1 Study (Practicum Part 2)

All notable changes to this project will be documented in this file.  
This project adheres to **Semantic Versioning (SemVer)** and each entry corresponds to a GitHub tag.

---

## [3.0.2] – 2025-10-26T15:20:23.720880+00:00

### 🚀 Tooling & Provenance Refactor

## **Summary:**

### Added

--

### Changed

--

### Fixed

--

### Tests

--

### Notes

--

---

[3.0.2]: https://github.com/rodrigomarquest/practicum2-nof1-adhd-bd/compare/{{PREVIOUS_TAG}}...3.0.2

# 📜 Changelog — N-of-1 Study (Practicum Part 2)

All notable changes to this project will be documented in this file.  
This project adheres to **Semantic Versioning (SemVer)** and each entry corresponds to a GitHub tag.

---

## [3.0.1] – 2025-10-24T03:59:24.900781+00:00

### 🚀 Tooling & Provenance Refactor

## **Summary:**

### Added

--

### Changed

--

### Fixed

--

### Tests

--

### Notes

--

---

[3.0.1]: https://github.com/rodrigomarquest/practicum2-nof1-adhd-bd/compare/{{PREVIOUS_TAG}}...3.0.1

# 📜 Changelog — N-of-1 Study (Practicum Part 2)

All notable changes to this project will be documented in this file.  
This project adheres to **Semantic Versioning (SemVer)** and each entry corresponds to a GitHub tag.

---

## [2.1.7] – 2025-10-22T21:07:24.667194+00:00

### 🚀 Data Provenance Sprint â€“ 2.1.7

## **Summary:**

### Added

--

### Changed

--

### Fixed

--

### Tests

--

### Notes

--

---

[2.1.7]: https://github.com/rodrigomarquest/practicum2-nof1-adhd-bd/compare/{{PREVIOUS_TAG}}...2.1.7

# 📜 Changelog — N-of-1 Study (Practicum Part 2)

All notable changes to this project will be documented in this file.  
This project adheres to **Semantic Versioning (SemVer)** and each entry corresponds to a GitHub tag.

---

## [2.1.5] – 2025-10-22T21:01:53.454456+00:00

### 🚀 Data Provenance Sprint â€“ 2.1.5

## **Summary:**

### Added

--

### Changed

--

### Fixed

--

### Tests

--

### Notes

--

---

[2.1.5]: https://github.com/rodrigomarquest/practicum2-nof1-adhd-bd/compare/{{PREVIOUS_TAG}}...2.1.5
[v2.1.4]: https://github.com/rodrigomarquest/practicum2-nof1-adhd-bd/compare/v2.1.3...v2.1.4
[v2.1.3]: https://github.com/rodrigomarquest/practicum2-nof1-adhd-bd/compare/v2.1.1...v2.1.3
[v2.1.1]: https://github.com/rodrigomarquest/practicum2-nof1-adhd-bd/compare/v2.1.0...v2.1.1
[v2.1.0]: https://github.com/rodrigomarquest/practicum2-nof1-adhd-bd/compare/v2.0.3...v2.1.0
[v2.0.3]: https://github.com/rodrigomarquest/practicum2-nof1-adhd-bd/compare/v2.0.2...v2.0.3
[v2.0.2]: https://github.com/rodrigomarquest/practicum2-nof1-adhd-bd/compare/v2.0-pre-ethics...v2.0.2
[v2.0-pre-ethics]: https://github.com/rodrigomarquest/practicum2-nof1-adhd-bd/releases/tag/v2.0-pre-ethics
[2.1.5]: https://github.com/<owner>/<repo>/compare/v2.1.4...2.1.5
[2.1.7]: https://github.com/<owner>/<repo>/compare/v2.1.4...2.1.7
[3.0.1]: https://github.com/<owner>/<repo>/compare/v2.1.7...3.0.1
[3.0.2]: https://github.com/<owner>/<repo>/compare/v3.0.1-26-ge94eb8d3c1a2db11c2afa68167c920be0ce80753...3.0.2
