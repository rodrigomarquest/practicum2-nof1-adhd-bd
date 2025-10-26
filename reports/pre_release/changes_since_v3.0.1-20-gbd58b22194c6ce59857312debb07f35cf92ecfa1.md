Summary

No substantive code changes since tag v3.0.1-20-gbd58b22194c6ce59857312debb07f35cf92ecfa1. This release prepares a small tooling and provenance update that centralizes release helpers, hardens provenance checks, and updates documentation (release process). The change set includes Makefile fixes, relocated release scripts under make_scripts/release/, and the new docs/RELEASE_PROCESS.md.

Highlights:

- Centralized release helpers into `make_scripts/release/` (render, version_guard, changelog updater).
- Introduced a stable LATEST / NEXT layer in the Makefile and improved `print-version` output.
- Added strict pre-release summary and provenance checks for `release-notes` and `release-assets`.

Next Steps:

- Verify provenance files under `provenance/` (etl*provenance_report.csv, data_audit_summary.md, pip_freeze*\*.txt) are present and up-to-date.
- Run `make release-draft RELEASE_DRY_RUN=1 ALLOW_DIRTY=1` to validate the draft flow; review the generated manifest and release notes.
- After verification, run the non-dry release flow with a chosen `VERSION` if a real tag should be created and published.
