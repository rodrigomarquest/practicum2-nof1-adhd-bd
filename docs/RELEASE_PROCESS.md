# Release Process – N-of-1 / Practicum Project

Overview

This document describes the standardized release flow used by the Practicum2 repository. It defines the LATEST/NEXT versioning layer, provenance requirements, release artifacts, and the Makefile targets involved in producing, staging and publishing a release.

Core Concepts

- LATEST_TAG_RAW: the most recent Git tag (falls back to v0.0.0 if none).
- LATEST_VERSION: numeric version derived from LATEST_TAG_RAW (strip leading `v`).
- NEXT_VERSION: auto-computed patch bump over LATEST_VERSION by default; can be manually overridden.
- NEXT_TAG: `v$(NEXT_VERSION)` — tag used for the new release.

Makefile Targets (summary)

- `make print-version` — prints LATEST/NEXT fields to help decide versions.
- `make release-notes` — renders academic-style release notes from the pre-release summary.
- `make changelog-update` — updates `CHANGELOG.md` with the new entry.
- `make release-assets` — collects provenance and builds `dist/assets/<TAG>/manifest.json` (SHA256 included).
- `make release-draft` — orchestrates `version-guard -> release-notes -> changelog-update -> release-assets` and creates a draft GitHub release.
- `make release-publish` — publishes the release to GitHub and updates the `latest` tag.
- `make release-pack` — builds a tgz bundle for one-file submissions.
- `make release-all` — deprecated alias; use `release-draft` + `release-pack`.

Provenance Integration

Release artifacts require provenance. The following files are mandatory for `make release-assets`:

- `provenance/etl_provenance_report.csv`
- `provenance/data_audit_summary.md`
- `provenance/pip_freeze_*.txt` (latest)

If any of these are missing the make target will fail with a clear NEED message describing the missing items.

Output Structure

- Release notes: `docs/release_notes/release_notes_<TAG>.md`
- Changelog: `CHANGELOG.md` (updated by `make changelog-update`)
- Assets: `dist/assets/<TAG>/` with `manifest.json` containing `path`, `size_bytes`, and `sha256` for each file.
- Latest mirror: `git tag -f latest <TAG>` is updated by `make release-publish`.

Semantic Rules

- All console output in the release flow is English-only.
- Pre-release summaries are authored by the human or AI (VADER) and placed at `reports/pre_release/changes_since_<LATEST_TAG>.md`.
- `release-notes` will fail if the pre-release summary is missing, printing a NEED message asking for the summary.

Styling Standards (academic template)

Release notes are rendered to follow an academic-style template with the following sections:

- Title with tag, release date, branch, author, project
- Summary (short prose)
- Highlights (bullet items)
- Next Steps (bullet items)
- Citation and license block

This template is used by `make_scripts/release/render_release_from_templates.py` and is embedded in the renderer to avoid external template dependencies.

Human–AI Roles

- Human: Review diffs, author pre-release summary if not using AI, verify provenance artifacts, and publish the release.
- VADER (optional AI role): Generate `reports/pre_release/changes_since_<LATEST_TAG>.md` summarizing commits and PRs since the latest tag.

Validation Checklist

Before running the full release flow ensure:

- Working tree is committed or `ALLOW_DIRTY=1` is explicitly set (careful).
- `reports/pre_release/changes_since_<LATEST_TAG>.md` exists and is complete.
- Provenance files exist under `provenance/` as listed above.
- You are authenticated with GitHub CLI (`gh auth login`) and have push rights.

Future Automation Goals

- Add an optional GitHub Action to auto-generate draft releases from push events to `main`.
- Integrate VADER into a CI step that proposes pre-release summaries as PRs for human review.
- Add a verification step to validate that manifest SHA256 sums match uploaded assets on GH release.

---

This file is canonical documentation for the project's release flow.
