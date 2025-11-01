# ================================
# Practicum2 — N-of-1 ADHD + BD
# Canonical Makefile (v4-series, 2025-10-31)
# Goal: readable, minimal, and portable for researchers
# Notes:
# - Uses `.RECIPEPREFIX := >` so every recipe line starts with `> ` (space after >).
# - Detects Windows venv (.venv/Scripts/python.exe) vs POSIX (.venv/bin/python).
# - All echoes are ASCII to avoid terminal encoding quirks.
# ================================

.RECIPEPREFIX := >

# -------- Environment --------
# Auto-detect venv Python (Windows vs POSIX); fallback to "python" if venv not found.
# GNU make "wildcard" returns empty if path doesn't exist.
PYTHON := $(if $(wildcard .venv/Scripts/python.exe),.venv/Scripts/python.exe,$(if $(wildcard .venv/bin/python),.venv/bin/python,python))
VENV_DIR := .venv
PID ?= P000001
SNAPSHOT ?= auto
ETL_CMD ?= full
# Fixed defaults requested
CUTOVER ?= 2024-03-11
TZ_BEFORE ?= America/Sao_Paulo
TZ_AFTER  ?= Europe/Dublin

# -------- Installation (centralized requirements/) --------
.PHONY: install-base install-dev install-kaggle install-local

etl:
> echo ">>> etl: running src.etl_pipeline ($(ETL_CMD))"
> if [ "$(ETL_CMD)" = "extract" ]; then \
>   # Run extract under our Timer wrapper so the terminal shows the standard header/footer
>   PYTHONPATH="$$PWD" $(PYTHON) scripts/run_etl_with_timer.py extract \
>     --participant $(PID) \
>     --snapshot $(SNAPSHOT) \
>     --cutover $(CUTOVER) \
>     --tz_before $(TZ_BEFORE) \
>     --tz_after $(TZ_AFTER); \
> else \
>   # For full runs ensure extracted data exists before proceeding
>   if [ "$(ETL_CMD)" = "full" ]; then \
>     if [ ! -d "data/etl/$(PID)/$(SNAPSHOT)/extracted" ]; then \
>       echo "ERROR: extracted data not found for $(PID)/$(SNAPSHOT). Run 'make etl ETL_CMD=extract PID=$(PID) SNAPSHOT=$(SNAPSHOT)' first."; \
>       exit 1; \
>     fi; \
>   fi; \
>   PYTHONPATH="$$PWD" $(PYTHON) -m src.etl_pipeline $(ETL_CMD) \
>     --participant $(PID) \
>     --snapshot $(SNAPSHOT) \
>     --cutover $(CUTOVER) \
>     --tz_before $(TZ_BEFORE) \
>     --tz_after $(TZ_AFTER); \
> fi
clean-data:
> echo ">>> clean-data: removing ETL outputs and AI results"
> rm -rf notebooks/outputs dist/assets logs backups processed 2>/dev/null || true
> rm -rf data/etl data/ai 2>/dev/null || true
> echo "[OK] data outputs removed"

clean-provenance:
> echo ">>> clean-provenance: removing transient provenance artifacts (keep reports)"
> find provenance -type f \( \
>   -name "pip_freeze_*.txt" -o \
>   -name "hash_snapshot_*.json" -o \
>   -name "migrate_layout_*.json" -o \
>   -name "cleanup_log_*.txt" \
> \) -exec rm -f {} + 2>/dev/null || true
> echo "[OK] provenance transient files removed"

clean-all: clean clean-data clean-provenance
> echo ">>> clean-all: full cleanup done"

# -------- Core workflows (ETL, labels, QC, packaging) --------
.PHONY: etl labels qc pack-kaggle

etl:
> echo ">>> etl: running src.etl_pipeline"
> PYTHONPATH="$$PWD" $(PYTHON) -m src.etl_pipeline $(ETL_CMD) \
>   --participant $(PID) \
>   --snapshot $(SNAPSHOT) \
>   --cutover $(CUTOVER) \
>   --tz_before $(TZ_BEFORE) \
>   --tz_after $(TZ_AFTER)
> if [ "$(ETL_CMD)" = "extract" ]; then \
>   # Run extract under our Timer wrapper so the terminal shows the standard header/footer
>   PYTHONPATH="$$PWD" $(PYTHON) scripts/run_etl_with_timer.py extract \
>     --participant $(PID) \
>     --snapshot $(SNAPSHOT) \
>     --cutover $(CUTOVER) \
>     --tz_before $(TZ_BEFORE) \
>     --tz_after $(TZ_AFTER); \
> else \
>   # For full runs ensure extracted data exists before proceeding
>   if [ "$(ETL_CMD)" = "full" ]; then \
>     if [ ! -d "data/etl/$(PID)/$(SNAPSHOT)/extracted" ]; then \
>       echo "ERROR: extracted data not found for $(PID)/$(SNAPSHOT). Run 'make etl ETL_CMD=extract PID=$(PID) SNAPSHOT=$(SNAPSHOT)' first."; \
>       exit 1; \
>     fi; \
>   fi; \
>   PYTHONPATH="$$PWD" $(PYTHON) -m src.etl_pipeline $(ETL_CMD) \
>     --participant $(PID) \
>     --snapshot $(SNAPSHOT) \
>     --cutover $(CUTOVER) \
>     --tz_before $(TZ_BEFORE) \
>     --tz_after $(TZ_AFTER); \
> fi

# Labels usam PARTICIPANT/SNAPSHOT (defaults em config/settings.yaml)
labels:
> @echo ">>> labels: running src.make_labels for $(PID) @ $(SNAPSHOT)"
> @PYTHONPATH="$$PWD" $(PYTHON) -m src.make_labels \
>   --rules config/label_rules.yaml \
>   --in data/etl/$(PID)/$(SNAPSHOT)/joined/features_daily.csv \
>   --out data/etl/$(PID)/$(SNAPSHOT)/joined/features_daily_labeled.csv



# ----------------------------------------------------------------------
# Release pipeline (v4-friendly helpers)
# - Defaults are non-invasive; do not push automatically. Use release-push to push.
# - Uses $(PYTHON) (auto-detected earlier) and preserves .RECIPEPREFIX := >
# ----------------------------------------------------------------------

RELEASE_VERSION ?= 4.0.2
RELEASE_TAG ?= v$(RELEASE_VERSION)
RELEASE_BRANCH ?= v4-main

.PHONY: release-verify release-summary release-draft release-freeze release-tag release-push release-publish release-final help-release

release-verify:
> @echo ">>> verify: tree clean, tag free, semver"
> @test -z "$$(git status --porcelain)" || (echo "Working tree not clean" && exit 1)
> @test -z "$$(git tag -l $(RELEASE_TAG))" || (echo "Tag $(RELEASE_TAG) already exists" && exit 1)
> @echo "$(RELEASE_VERSION)" | grep -Eq '^[0-9]+\.[0-9]+\.[0-9]+$$' || (echo "Invalid SemVer: $(RELEASE_VERSION)" && exit 1)
> @echo "[ok] verification passed"

release-summary:
> echo ">>> summary: collect commits since last tag"
> mkdir -p dist/changelog
> LAST_TAG=$$(git describe --tags --abbrev=0 2>/dev/null || echo ""); \
> if [ -n "$$LAST_TAG" ]; then \
>   git log --pretty=oneline $$LAST_TAG..HEAD > dist/changelog/CHANGES_SINCE_LAST_TAG.txt; \
> else \
>   git log --pretty=oneline > dist/changelog/CHANGES_SINCE_LAST_TAG.txt; \
> fi; \
> echo "[ok] wrote dist/changelog/CHANGES_SINCE_LAST_TAG.txt"

release-draft:
> echo ">>> draft: release notes + changelog (dry-run)"
> mkdir -p docs/release_notes dist/changelog
> if [ -z "$(RELEASE_TITLE)" ]; then \
>   echo "ERROR: RELEASE_TITLE must be provided for a meaningful release (see issue #7)."; \
>   echo "Example: make release-draft RELEASE_VERSION=4.1.0 LAST_TAG=v4.0.4 RELEASE_TITLE=\"My release title\""; \
>   exit 1; \
> fi
> $(PYTHON) -m src.tools.render_release_from_templates \
>   --version $(RELEASE_VERSION) \
>   --tag $(RELEASE_TAG) \
>   --title "Release $(RELEASE_TAG) – $(RELEASE_TITLE)" \
>   --summary "Auto-generated draft. See CHANGES_SINCE_LAST_TAG.txt." \
>   --branch $(RELEASE_BRANCH)
> echo "[ok] draft prepared under docs/release_notes and dist/changelog"

release-freeze:
> echo ">>> freeze: pip freeze snapshot"
> @mkdir -p dist/provenance
> $(PYTHON) -m pip freeze > dist/provenance/pip_freeze_$$(date -u +%F).txt
> @echo "[ok] freeze written"

release-tag:
> echo ">>> tagging $(RELEASE_TAG)"
> @git add docs/release_notes dist/changelog dist/provenance || true
> @git commit -m "chore(release): finalize artifacts for $(RELEASE_TAG)" || echo "(no changes to commit)"
> @git tag -a $(RELEASE_TAG) -m "Release $(RELEASE_TAG)"
> @echo "[ok] commit+tag created"

release-publish:
release-publish:
> echo ">>> gh release create $(RELEASE_TAG)"
> echo "(gh CLI optional)"
> mkdir -p dist
> PUBLISH_NOTES=dist/release_notes_for_publish_$(RELEASE_VERSION).md; \
> cat docs/release_notes/release_notes_v$(RELEASE_VERSION).md > $$PUBLISH_NOTES; \
> if [ -n "$(ISSUES)" ]; then \
>   for i in $(ISSUES); do echo "\nCloses #$$i" >> $$PUBLISH_NOTES; done; \
> fi; \
> if [ -n "$(RELEASE_TITLE)" ]; then TITLE="$(RELEASE_TAG) - $(RELEASE_TITLE)"; else TITLE="$(RELEASE_TAG)"; fi; \
> gh release create $(RELEASE_TAG) \
>   --title "$$TITLE" \
>   --notes-file "$$PUBLISH_NOTES" \
>   dist/changelog/CHANGELOG.dryrun.md \
>   dist/provenance/pip_freeze_$$(date -u +%F).txt dist/assets/$(RELEASE_VERSION)/* || true
> echo "[ok] GitHub Release created (or gh not available)"

release-final: release-verify release-summary release-draft release-freeze release-tag
> @echo "[ok] local release finalized: $(RELEASE_TAG)"

release-assets:
> @echo ">>> Assembling release package for $(RELEASE_VERSION)"
> @mkdir -p dist/assets/$(RELEASE_VERSION)
> @cp docs/release_notes/release_notes_v$(RELEASE_VERSION).md dist/assets/$(RELEASE_VERSION)/
> @cp dist/changelog/CHANGELOG.dryrun.md dist/assets/$(RELEASE_VERSION)/CHANGELOG.md || true
> @cp provenance/etl_provenance_report.csv dist/assets/$(RELEASE_VERSION)/ || true
> @echo "[OK] Assembled assets for v$(RELEASE_VERSION)"

release-pr:
> @echo ">>> release-pr: create branch and open PR targeting main"
> if [ -z "$(RELEASE_TITLE)" ]; then \
>   echo "ERROR: RELEASE_TITLE is required for release-pr (see issue #7)."; \
>   echo "Example: make release-pr RELEASE_VERSION=4.1.0 RELEASE_TITLE=\"My title\" ISSUES=\"1 4\""; \
>   exit 1; \
> fi; \
> @set -e; \
BR=release/v$(RELEASE_VERSION); \
if ! git rev-parse --verify --quiet $$BR >/dev/null; then \
	git switch -c $$BR; \
else \
	git switch $$BR; \
fi; \
	git add docs/release_notes/release_notes_v$(RELEASE_VERSION).md || true; \
	git commit -m "chore(release): add release notes for v$(RELEASE_VERSION)" || true; \
	git push -u origin $$BR || true; \
	mkdir -p dist; \
	cat docs/release_notes/release_notes_v$(RELEASE_VERSION).md > dist/release_pr_body_$(RELEASE_VERSION).md; \
	echo "\nCloses #1\nCloses #2" >> dist/release_pr_body_$(RELEASE_VERSION).md; \
	if gh auth status >/dev/null 2>&1; then \
		gh pr create --base main --head $$BR --title "Release v$(RELEASE_VERSION) – $(RELEASE_TITLE)" --body-file dist/release_pr_body_$(RELEASE_VERSION).md || echo "gh pr create failed; you can open a PR manually"; \
	else \
		echo "gh CLI not authenticated or not available. Create PR manually using: https://github.com/$(shell git config --get remote.origin.url | sed -e 's/.*:\/\///' -e 's/\.git$$//')/compare/main...$$BR"; \
	fi

help-release:
> echo "Release targets:"
> echo "  make release-verify    # tree clean, tag livre, SemVer"
> echo "  make release-summary   # gera CHANGES_SINCE_LAST_TAG.txt"
> echo "  make release-draft     # gera release_notes + changelog draft"
> echo "  make release-freeze    # pip freeze -> dist/provenance"
> echo "  make release-tag       # commit + tag local"
> echo "  make release-push      # push branch + tags"
> echo "  make release-publish   # cria GitHub Release (gh CLI)"
> echo "  make release-final     # encadeado local (sem push/publish)"
> echo "  make release-pr        # create a release PR targeting main (gh CLI)"
> echo "Note: release-draft and release-pr require RELEASE_TITLE to be set (see issue #7)."

qc:
> echo ">>> qc: running src.eda"
> $(PYTHON) -m src.eda

PACK_OUT := dist/assets

pack-kaggle:
> @echo ">>> pack-kaggle: packaging Kaggle dataset snapshot for $(PID) $(SNAPSHOT)"
> @PYTHONPATH="$$PWD" $(PYTHON) -m src.tools.pack_kaggle

# -------- Help --------
.PHONY: help
help:
> echo "Available targets:"
> echo "  install-base       - Install base requirements"
> echo "  install-dev        - Install development requirements"
> echo "  install-kaggle     - Install Kaggle requirements"
> echo "  install-local      - Install local env requirements"
> echo "  clean              - Remove cache/logs"
> echo "  clean-data         - Remove ETL/AI outputs"
> echo "  clean-provenance   - Remove transient provenance artifacts"
> echo "  clean-all          - Run all clean targets"
> echo "  etl                - Run ETL pipeline (src.etl_pipeline)"
> echo "  labels             - Generate heuristic labels (src.make_labels)"
> echo "  qc                 - Run EDA/QC (src.eda)"
> echo "  pack-kaggle        - Zip dataset for Kaggle (uses PID/SNAPSHOT)"
