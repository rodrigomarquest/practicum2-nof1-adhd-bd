# make_scripts — extracted idempotent helpers

## Purpose

This directory contains small, idempotent Python scripts that were extracted from Makefile heredocs and long shell recipes. They serve two purposes:

- `create_*` scripts: write canonical module files used by the ETL pipeline. They are safe to run multiple times and will overwrite target files.
- `patch_*` scripts: apply conservative, regex-based patches to existing source files (create backups before modifying files).

## Usage

Run the scripts directly with the repository's Python interpreter. For example:

```bash
python make_scripts/patch_etl_pipeline_helpers.py
```

## Notes

- Patcher scripts create `.bak.<TIMESTAMP>` backups of any file they modify.
- Keep these scripts executable and review them before running in CI.
- The Makefile was refactored to call these scripts from single-line recipes using `.RECIPEPREFIX := >` to avoid literal TABs and heredoc fragility.
