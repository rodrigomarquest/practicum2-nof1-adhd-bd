Tooling & Provenance Refactor

This release centralizes release tooling, tightens provenance collection, and fixes a few environment and encoding issues to make releases more reproducible and maintainable.

- ETL & Provenance updates — standardized provenance outputs and collection points; added checks to require `etl_provenance_report.csv`, `data_audit_summary.md`, and a pip freeze snapshot before asset packaging.
- Makefile standardization — introduced a LATEST/NEXT layer, unified recipe prefixes to the repo RECIPEPREFIX, and improved `print-version` output for clearer CI/automation use.
- Modeling baseline fixes — small fixes to modeling helpers and baseline scripts to ensure deterministic behavior across environments (no large model changes in this release).
- Notebook NB1 recovery and environment auto-detection — recovered an important EDA notebook and improved environment detection to better locate virtualenv interpreters on Windows and POSIX shells.
- Version guard moved to make_scripts/release — the pre-release checks (clean tree, remote tag checks, allow-dirty opt) were moved under `make_scripts/release/` for clearer separation.
- Renderer and Changelog UTF-8 support — the release notes renderer was centralized (stdlib-only) and the changelog updater now writes UTF-8 on Windows to avoid encoding errors.
- Docs: added `docs/RELEASE_PROCESS.md` — documents the new flow, provenance requirements, and human/AI roles for preparing releases.

Next Steps:

- Finalize provenance CSV and audit reports
- Run full release (non-dry)
- Prepare CA2 LaTeX update with new metrics
