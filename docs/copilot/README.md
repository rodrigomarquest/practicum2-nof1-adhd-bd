# Copilot & AI-assisted Documentation

This folder contains **AI-assisted session logs, completion reports, implementation summaries, and execution plans** generated with the help of GitHub Copilot, Claude, and ChatGPT during the v4.0-v4.1.x development cycles.

## Purpose

These documents serve as:
- **Development provenance** – tracking decisions, iterations, and problem-solving sessions
- **Execution logs** – detailed records of refactoring phases, fixes, and releases
- **Historical context** – understanding how the codebase evolved over time

## What's Here

**32+ AI-assisted documents** covering:
- Phase summaries (Phase 2, Phase 3, Phase 7, etc.)
- Implementation reports (Period Expansion, Vendor/Variant, Biomarkers)
- Release execution plans (v4.1.0)
- Notebook migrations and finalization (NB1, NB2, NB3)
- ETL fixes and cleanup summaries
- Session completion and handoff reports

## Important Note

⚠️ These are **NOT canonical design documents**. They document the *process* of building the pipeline, not the *result*.

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
