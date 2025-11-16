# Archive Plan - Legacy Module Identification

**Generated:** 2025-11-16  
**Purpose:** Identify non-canonical modules for safe archival (NO deletions)  
**Reference:** CANONICAL_ENTRYPOINTS.md

---

## Methodology

1. **Canonical Check:** Compare against CANONICAL_ENTRYPOINTS.md
2. **Import Analysis:** Check if imported by canonical entrypoints
3. **Test Coverage:** Verify if used by tests/
4. **Makefile Check:** Verify if referenced in Makefile

---

## CANONICAL MODULES (PROTECTED - DO NOT MOVE)

### Scripts (4 files)
- ✅ `scripts/run_full_pipeline.py` ⭐ MAIN ORCHESTRATOR
- ✅ `scripts/prepare_nb2_dataset.py`
- ✅ `scripts/extract_biomarkers.py`
- ✅ `scripts/prepare_zepp_data.py`

### Core Stages (4 files)
- ✅ `src/etl/stage_csv_aggregation.py`
- ✅ `src/etl/stage_unify_daily.py`
- ✅ `src/etl/stage_apply_labels.py`
- ✅ `src/etl/nb3_analysis.py`

### Domain Loaders (4 files)
- ✅ `src/domains/parse_zepp_export.py`
- ✅ `src/domains/sleep/sleep_from_extracted.py`
- ✅ `src/domains/cardiovascular/cardio_from_extracted.py`
- ✅ `src/domains/activity/zepp_activity.py`

### Infrastructure (3 files)
- ✅ `src/etl/config.py`
- ✅ `src/etl/io_utils.py`
- ✅ `src/biomarkers/aggregate.py` (used by scripts/extract_biomarkers.py)

### Supporting Folders (3 complete)
- ✅ `src/etl/iphone_backup/` (FULL)
- ✅ `src/etl/cardiovascular/` (FULL)
- ✅ `src/etl/common/` (FULL)

---

## DUPLICATE ANALYSIS

### Family 1: scripts/*.py vs src/cli/*.py

#### scripts/ (4 files - ALL CANONICAL)
- ✅ `scripts/run_full_pipeline.py` - Makefile: all pipeline targets
- ✅ `scripts/prepare_nb2_dataset.py` - Makefile: prep-nb2 target
- ✅ `scripts/extract_biomarkers.py` - Not in Makefile but canonical
- ✅ `scripts/prepare_zepp_data.py` - Not in Makefile but canonical

#### src/cli/ (6 files - CHECK FOR DUPLICATES)
- ❓ `src/cli/etl_runner.py` - CANDIDATE (duplicate of run_full_pipeline.py?)
- ❓ `src/cli/extract_biomarkers.py` - DUPLICATE of scripts/extract_biomarkers.py
- ❓ `src/cli/migrate_snapshots.py` - CANDIDATE (not referenced)
- ❓ `src/cli/prepare_zepp_data.py` - DUPLICATE of scripts/prepare_zepp_data.py
- ❓ `src/cli/run_etl_with_timer.py` - CANDIDATE (not referenced)
- ✅ `src/cli/__init__.py` - Keep (package marker)

**Action Plan:**
- Compare `scripts/extract_biomarkers.py` vs `src/cli/extract_biomarkers.py`
- Compare `scripts/prepare_zepp_data.py` vs `src/cli/prepare_zepp_data.py`
- If scripts/ version is used → archive src/cli/ versions
- If src/cli/ version is used → move scripts/ to archive (unlikely)

---

### Family 2: src/biomarkers/* vs src/domains/biomarkers/*

#### src/biomarkers/ (9 files)
- ✅ `aggregate.py` - **CANONICAL** (used by scripts/extract_biomarkers.py)
- ❓ `activity.py` - Check if imported by aggregate.py
- ❓ `circadian.py` - Check if imported
- ❓ `extract.py` - Check if imported
- ❓ `hrv.py` - Check if imported
- ❓ `segmentation.py` - Check if imported
- ❓ `sleep.py` - Check if imported
- ❓ `validators.py` - Check if imported
- ✅ `__init__.py` - Keep

#### src/domains/biomarkers/ (needs listing)
- 🔍 **TODO:** List contents of src/domains/biomarkers/
- 🔍 **TODO:** Check for duplicates with src/biomarkers/

**Action Plan:**
- List src/domains/biomarkers/ contents
- Compare with src/biomarkers/ modules
- Determine which is canonical (likely src/biomarkers/ due to CANONICAL_ENTRYPOINTS.md)
- Archive the unused family

---

### Family 3: Root-level src/*.py modules

#### Potentially Legacy Modules
- ❓ `src/make_labels.py` - Check if used (vs stage_apply_labels.py)
- ❓ `src/utils.py` - Check imports
- ❓ `src/eda.py` - Check if referenced
- ❓ `src/etl_pipeline.py` - Check if used (vs run_full_pipeline.py)
- ❓ `src/models_nb2.py` - Check if used by nb3_analysis.py
- ❓ `src/models_nb3.py` - Check if used by nb3_analysis.py
- ❓ `src/nb3_run.py` - Check if used

**Action Plan:**
- Check imports from canonical modules
- Archive unused root-level modules

---

### Family 4: src/domains/* vs src/etl/* overlap

#### src/domains/ structure
```
domains/
├── activity/
├── apple_raw_to_per_metric.py (DUPLICATE?)
├── biomarkers/
├── cardiovascular/
├── cda/
├── common/
├── config.py (DUPLICATE?)
├── enriched/
├── extract_screen_time.py
├── features/
├── io_utils.py (DUPLICATE?)
├── iphone_backup/
├── join/
├── parse_zepp_export.py (CANONICAL)
├── sleep/
└── zepp_join.py (DUPLICATE?)
```

#### Potential Duplicates
- ❓ `src/domains/apple_raw_to_per_metric.py` vs `src/etl/apple_raw_to_per_metric.py`
- ❓ `src/domains/config.py` vs `src/etl/config.py`
- ❓ `src/domains/io_utils.py` vs `src/etl/io_utils.py`
- ❓ `src/domains/zepp_join.py` vs `src/etl/zepp_join.py`

**Action Plan:**
- Check which version is imported by canonical modules
- Archive the unused copies

---

### Family 5: src/tools/* (utility scripts)

#### Files in src/tools/
- ❓ `aggregate_joined.py`
- ❓ `audit/` (folder)
- ❓ `cda_probe.py`
- ❓ `check_zips.py`
- ❓ `etl_paths.py`
- ❓ `extract_usage_data.py`
- ❓ `generate_provenance_report.py`
- ❓ `import_sweep_by_domain.py`
- ❓ `import_sweep_clean.py`
- ❓ `pack_kaggle.py`
- ❓ `px8_lite_pass2.py`
- ❓ `render_release_from_templates.py`
- ❓ `run_apple_per_metric.py`
- ❓ `templates/` (folder)
- ❓ `update_changelog.py`
- ❓ `_import_sweep.py`

**Action Plan:**
- Check if any are imported by canonical modules
- Check if any are referenced in Makefile
- Archive unused tools

---

### Family 6: Other src/ subfolders

#### Potentially Legacy Folders
- ❓ `src/features/` - Check contents and usage
- ❓ `src/io/` - Check vs src/etl/io_utils.py
- ❓ `src/labels/` - Check vs stage_apply_labels.py
- ❓ `src/lib/` - Check if df_utils.py, io_guards.py are used
- ❓ `src/modeling/` - Check contents
- ❓ `src/models/` - Check contents
- ❓ `src/nb_common/` - Check contents
- ❓ `src/utils/` - Check contents

**Action Plan:**
- List contents of each folder
- Check imports from canonical modules
- Archive unused folders

---

## ARCHIVE DESTINATIONS

### Proposed Archive Structure
```
archive/
├── root_scripts_legacy/          # Unused scripts/*.py
├── src_cli_legacy/               # Duplicate src/cli/*.py modules
├── src_root_legacy/              # Unused src/*.py (make_labels, utils, etc.)
├── src_biomarkers_legacy/        # If src/domains/biomarkers is canonical
├── src_domains_duplicates/       # Duplicate src/domains/*.py files
├── src_tools_legacy/             # Unused src/tools/*.py
└── src_folders_legacy/           # Unused src/* folders
    ├── features/
    ├── io/
    ├── labels/
    ├── lib/
    ├── modeling/
    ├── models/
    ├── nb_common/
    └── utils/
```

---

## EXECUTION PLAN

### Phase 1: Import Analysis (READ-ONLY)
1. ✅ Create this ARCHIVE_PLAN.md
2. 🔄 Scan imports in canonical modules
3. 🔄 Check Makefile references
4. 🔄 Check test imports
5. 🔄 Classify each module as CANONICAL or CANDIDATE

### Phase 2: Duplicate Resolution
1. Compare scripts/*.py vs src/cli/*.py
2. Compare src/biomarkers/* vs src/domains/biomarkers/*
3. Compare src/domains/*.py vs src/etl/*.py
4. Document canonical choice in this file

### Phase 3: Safe Archival (INCREMENTAL COMMITS)
1. Move Family 1 duplicates (src/cli/)
2. Test: `pytest` + `make pipeline PID=P000001 SNAPSHOT=auto`
3. Move Family 2 duplicates (biomarkers)
4. Test again
5. Move Family 3 (root-level src/*.py)
6. Test again
7. Move Family 4 (domains duplicates)
8. Test again
9. Move Family 5 (tools)
10. Test again
11. Move Family 6 (unused folders)
12. Final test

### Phase 4: Documentation
1. Update CHANGELOG.md
2. Update CANONICAL_ENTRYPOINTS.md if needed
3. Create commit message with full summary

---

## SAFETY CHECKS

### Before Each Move
- [ ] Module NOT in CANONICAL_ENTRYPOINTS.md protected list
- [ ] Module NOT imported by canonical entrypoints
- [ ] Module NOT referenced in Makefile
- [ ] Module NOT used by tests/

### After Each Batch
- [ ] `pytest` passes
- [ ] `make pipeline` runs (or minimal ETL target)
- [ ] No import errors in canonical modules
- [ ] Commit with clear message

---

## STATUS: � ANALYSIS COMPLETE

**Next Action:** Begin Phase 2 - Duplicate Resolution

---

## IMPORT ANALYSIS RESULTS

### ✅ CONFIRMED CANONICAL (based on actual imports)

#### From scripts/run_full_pipeline.py:
- `src.etl.stage_csv_aggregation.run_csv_aggregation`
- `src.etl.stage_unify_daily.run_unify_daily`
- `src.etl.stage_apply_labels.run_apply_labels`
- `src.etl.nb3_analysis.*` (lazy imports for NB3 stages)

#### From scripts/extract_biomarkers.py:
- `src.biomarkers.aggregate` ✅ **CANONICAL**
  - Imports: `segmentation`, `hrv`, `sleep`, `activity`, `circadian`, `validators`
  - **ALL 8 modules in src/biomarkers/ are CANONICAL** (part of the aggregate chain)

#### From src/etl_pipeline.py (legacy module, but used by tests):
- `src.etl.common.progress.Timer`
- `src.domains.common.io.migrate_from_data_ai_if_present`
- `src.domains.common.io.etl_snapshot_root`
- `src.domains.enriched.pre.enrich_prejoin_run`
- `lib.io_guards` (not src/)

### ❌ DUPLICATES FOUND

#### Family 1: scripts/*.py vs src/cli/*.py

**DUPLICATES (ARCHIVE src/cli/ versions):**
1. `scripts/extract_biomarkers.py` (canonical) vs `src/cli/extract_biomarkers.py`
   - **Difference:** scripts/ imports `src.biomarkers`, src/cli/ imports `src.domains.biomarkers`
   - **Action:** ARCHIVE `src/cli/extract_biomarkers.py` → `archive/src_cli_legacy/`

2. `scripts/prepare_zepp_data.py` (canonical) vs `src/cli/prepare_zepp_data.py`
   - **Difference:** Nearly identical standalone scripts
   - **Action:** ARCHIVE `src/cli/prepare_zepp_data.py` → `archive/src_cli_legacy/`

**CANDIDATES (not referenced):**
3. `src/cli/etl_runner.py` - Uses `src.etl_pipeline` (legacy module)
   - **Action:** ARCHIVE → `archive/src_cli_legacy/` (superseded by run_full_pipeline.py)

4. `src/cli/run_etl_with_timer.py` - Not referenced
   - **Action:** ARCHIVE → `archive/src_cli_legacy/`

5. `src/cli/migrate_snapshots.py` - Not referenced
   - **Action:** ARCHIVE → `archive/src_cli_legacy/`

#### Family 2: src/biomarkers/* vs src/domains/biomarkers/*

**CANONICAL:** `src/biomarkers/` (ALL 9 files)
- ✅ Used by `scripts/extract_biomarkers.py`
- ✅ Internal chain: aggregate → {segmentation, hrv, sleep, activity, circadian, validators}

**DUPLICATE:** `src/domains/biomarkers/` (9 files)
- ❌ Only referenced by `src/cli/extract_biomarkers.py` (which is itself a duplicate)
- **Action:** ARCHIVE entire `src/domains/biomarkers/` → `archive/src_domains_legacy/biomarkers/`

---

## DETAILED EXECUTION PLAN

### Phase 2A: Archive src/cli/ duplicates

**Batch 1: Move 5 src/cli/ modules**
```bash
mkdir -p archive/src_cli_legacy
git mv src/cli/extract_biomarkers.py archive/src_cli_legacy/
git mv src/cli/prepare_zepp_data.py archive/src_cli_legacy/
git mv src/cli/etl_runner.py archive/src_cli_legacy/
git mv src/cli/run_etl_with_timer.py archive/src_cli_legacy/
git mv src/cli/migrate_snapshots.py archive/src_cli_legacy/
git commit -m "refactor: archive duplicate src/cli/ modules (scripts/ is canonical)"
```

**Test:** `pytest && make verify`

### Phase 2B: Archive src/domains/biomarkers/

**Batch 2: Move biomarkers duplicate**
```bash
mkdir -p archive/src_domains_legacy
git mv src/domains/biomarkers archive/src_domains_legacy/
git commit -m "refactor: archive src/domains/biomarkers (src/biomarkers is canonical)"
```

**Test:** `pytest && make verify`

---

**Maintainer:** Rodrigo Marques Teixeira  
**Last Update:** 2025-11-16
