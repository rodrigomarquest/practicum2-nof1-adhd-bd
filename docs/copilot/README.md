# Copilot & AI-assisted Documentation

This folder contains **AI-assisted session logs, completion reports, implementation summaries, and execution plans** generated with the help of GitHub Copilot, Claude, and ChatGPT during the v4.0-v4.1.x development cycles.

---

## ⚠️ IMPORTANT: File Location Policy

**ALL Copilot/AI-assisted documentation MUST be created in `docs/copilot/`.**

- ❌ Do NOT create AI work files in repository root
- ❌ Do NOT create AI work files in `docs/` root
- ✅ Always use `docs/copilot/` for session logs, audits, summaries, and reports
- ✅ Reference files with full path: `docs/copilot/FILENAME.md`

**Examples:**
- ✅ `docs/copilot/TESTS_CLEANUP_AUDIT.md`
- ✅ `docs/copilot/PAPER_CODE_AUDIT_REPORT.md`
- ❌ `TESTS_CLEANUP_AUDIT.md` (root level)
- ❌ `docs/PAPER_CODE_AUDIT.md` (docs root)

---

## Purpose

These documents serve as:

- **Development provenance** – tracking decisions, iterations, and problem-solving sessions
- **Execution logs** – detailed records of refactoring phases, fixes, and releases
- **Historical context** – understanding how the codebase evolved over time

## What's Here

**40+ AI-assisted documents** covering:

### Paper-Code Alignment (Nov 2025)

- `PAPER_CODE_AUDIT_REPORT.md` - Comprehensive audit (1,847 lines)
- `PAPER_CODE_CONSISTENCY_REVIEW.md` - PhD-level technical review
- `AUDIT_EXECUTIVE_SUMMARY.md` - Prioritized action items
- `FIX_SEGMENTATION_SECTION.md` - Actionable fix guide
- `FINAL_VALIDATION_REPORT.md` - 100% alignment confirmation

### Pipeline Development & Audits

- `ETL_AUDIT_IMPLEMENTATION_PLAN.md` - Regression test suite plan
- `HR_FEATURE_INTEGRITY_AUDIT.md` - Heart rate feature validation
- `DETERMINISM_VALIDATION_REPORT.md` - Reproducibility verification
- `PBSI_INTEGRATION_SUMMARY.md` - Canonical PBSI implementation

### Release Management

- `RELEASE_v4.1.0_FINAL_REPORT.md` - v4.1.0 release summary
- `RELEASE_v4.1.0_INDEX.md` - Release documentation index
- `RELEASE_SUMMARY_v4.1.2.md` - v4.1.2 deployment summary
- `FINAL_COMPLETION_REPORT.md` - Project completion milestone

### Notebook Development (v4.1.5)

- `NB1_EDA_MIGRATION.md` - EDA notebook refactoring
- `NB2_FINALIZATION.md` - Baseline model implementation
- `NB3_FIX_COMPLETED.md` - LSTM model fixes
- `cleanup_decisions.md` - Repository reorganization plan

### Technical Implementations

- Phase summaries (Phase 2, Phase 3, Phase 7, etc.)
- Period Expansion implementation (Vendor/Variant, Biomarkers)
- Sleep audit and forward-fill strategies
- ETL fixes and optimization reports
- Session completion and handoff reports
- Module usage analysis (`UNUSED_MODULES_REPORT.md`)

## Important Note

⚠️ These are **NOT canonical design documents**. They document the _process_ of building the pipeline, not the _result_.

For **canonical architecture and research documentation**, see:

- `docs/ETL_ARCHITECTURE_COMPLETE.md` – Complete ETL architecture
- `docs/ETL_EDA_MODELING_PLAN.md` – EDA and modeling plan
- `docs/PHASE2_ENRICHED_PREJOIN_ARCHITECTURE.md` – Phase 2 design
- `docs/PHASE3_ENRICHED_GLOBAL_ARCHITECTURE.md` – Phase 3 design
- `docs/DEV_GUIDE.md` – Developer guide
- `docs/MAKEFILE_GUIDE.md` – Build system reference
- `pipeline_overview.md` – High-level pipeline overview

## Organization

All files in this folder follow naming conventions:

- `*_SUMMARY.md` – Phase or feature summaries
- `*_REPORT.md` – Completion or execution reports
- `*_IMPLEMENTATION.md` – Implementation details
- `HANDOFF_*.md` – Context handoffs between sessions
- `SESSION_*.md`, `TASK_*.md` – Session logs

Last updated: 2025-11-16 (Finishing Pass - Step A)
