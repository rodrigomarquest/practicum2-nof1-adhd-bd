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
DRY_RUN ?= 0
REPO_ROOT ?= .
# Compute platform path separator via the chosen Python executable and build a
# PYTHONPATH value that works on both POSIX (:) and Windows (;) hosts. This
# ensures child processes can import `src` when Make spawns a new Python.
PATHSEP := $(shell $(PYTHON) -c "import os; print(os.pathsep)")
PYTHONPATH_ETL := $(REPO_ROOT)$(PATHSEP)$(REPO_ROOT)/src
# Fixed defaults requested
CUTOVER ?= 2024-03-11
TZ_BEFORE ?= America/Sao_Paulo
TZ_AFTER  ?= Europe/Dublin

ifeq ($(DRY_RUN),1)
DRY_FLAG := --dry-run
else
DRY_FLAG :=
endif

# -------- Installation (centralized requirements/) --------
.PHONY: install-base install-dev install-kaggle install-local

install-base:
> echo ">>> install-base: requirements/base.txt"
> $(PYTHON) -m pip install -U pip
> $(PYTHON) -m pip install -r requirements/base.txt

install-dev:
> echo ">>> install-dev: requirements/dev.txt"
> $(PYTHON) -m pip install -U pip
> $(PYTHON) -m pip install -r requirements/dev.txt

install-kaggle:
> echo ">>> install-kaggle: requirements/kaggle.txt"
> python -m pip install -U pip
> python -m pip install -r requirements/kaggle.txt

install-local:
> echo ">>> install-local: requirements/local.txt"
> $(PYTHON) -m pip install -U pip
> $(PYTHON) -m pip install -r requirements/local.txt

# -------- Clean-up (safe, portable) --------
.PHONY: clean clean-data clean-provenance clean-all

clean:
> echo ">>> clean: removing caches and logs"
> find . -name "__pycache__" -type d -prune -exec rm -rf {} + 2>/dev/null || true
> find . -name ".ipynb_checkpoints" -type d -prune -exec rm -rf {} + 2>/dev/null || true
> find . -name "*.pyc" -delete 2>/dev/null || true
> find . -name "*.log" -delete 2>/dev/null || true
> echo "[OK] caches/logs removed"

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

# Defaults (mantém compatibilidade)
PID ?= P000001
SNAPSHOT ?= auto
DRY_RUN ?= 0
REPO_ROOT ?= .

# Namespace 'etl' para estilo: make etl <subcomando>
.PHONY: etl
etl:
> @echo "[ETL] namespace loaded (use: make etl extract|sleep|cardio|activity|full)"

# Subcomandos (nível 2) -----------------------------

# 1) extract (já existente no projeto – manter apenas eco aqui)
.PHONY: extract
extract:
	> PYTHONPATH="$(PYTHONPATH_ETL)" \
	> $(PYTHON) -m cli.etl_runner extract \
>   --participant "$(PID)" \
>   --snapshot "$(SNAPSHOT)" \
>   --cutover "$(CUTOVER)" \
>   --tz_before "$(TZ_BEFORE)" \
>   --tz_after "$(TZ_AFTER)" \
>   $(DRY_FLAG)

# 2) sleep (stub provisório)
.PHONY: sleep
sleep:
> @echo "[ETL] sleep (stub) — será implementado em prompt separado"

# 3) cardio (stub provisório)
.PHONY: cardio
cardio:
> @echo "[ETL] cardio (stub) — será implementado em prompt separado"

# 4) activity (NOVO: chama o módulo de features)
.PHONY: activity
activity:
> PYTHONPATH=src \
> python -m domains.activity.activity_features \
>   --pid $(PID) \
>   --repo-root $(REPO_ROOT) \
>   --snapshot $(SNAPSHOT) \
>   --dry-run $(DRY_RUN)

# 5) full — encadeia extract → cardio → activity
.PHONY: full
full: extract cardio activity
> @echo "[OK] FULL ETL completed for PID=$(PID) SNAPSHOT=$(SNAPSHOT)"

# Labels usam PARTICIPANT/SNAPSHOT (defaults em config/settings.yaml)
labels:
> @echo ">>> labels: running src.make_labels for $(PID) @ $(SNAPSHOT)"
> @PYTHONPATH="$$PWD" $(PYTHON) -m src.make_labels \
>   --rules config/label_rules.yaml \
>   --in data/etl/$(PID)/snapshots/$(SNAPSHOT)/joined/features_daily.csv \
>   --out data/etl/$(PID)/snapshots/$(SNAPSHOT)/joined/features_daily_labeled.csv



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
> @mkdir -p dist/changelog
> @LAST_TAG=$$(git describe --tags --abbrev=0 2>/dev/null || echo ""); \
> if [ -n "$$LAST_TAG" ]; then \
>   git log --pretty=oneline $$LAST_TAG..HEAD > dist/changelog/CHANGES_SINCE_LAST_TAG.txt; \
> else \
>   git log --pretty=oneline > dist/changelog/CHANGES_SINCE_LAST_TAG.txt; \
> fi; \
	git add docs/release_notes/release_notes_v$(RELEASE_VERSION).md || true; \
	git commit -m "chore(release): add release notes for v$(RELEASE_VERSION)" || true; \
	git push -u origin $$BR || true; \
	mkdir -p dist; \
	PR_BODY=dist/extra$(RELEASE_VERSION).md; \
	cat docs/release_notes/release_notes_v$(RELEASE_VERSION).md > $$PR_BODY; \
	if [ -n "$(ISSUES)" ]; then \
	  for i in $(ISSUES); do echo "\nCloses #$$i" >> $$PR_BODY; done; \
	else \
	  echo "\nCloses #1\nCloses #2" >> $$PR_BODY; \
	fi; \
	if gh auth status >/dev/null 2>&1; then \
		gh pr create --base main --head $$BR --title "Release v$(RELEASE_VERSION) – $(RELEASE_TITLE)" --body-file $$PR_BODY || echo "gh pr create failed; you can open a PR manually"; \
	else \
		echo "gh CLI not authenticated or not available. Create PR manually using: https://github.com/$(shell git config --get remote.origin.url | sed -e 's/.*:\/\///' -e 's/\.git$$//')/compare/main...$$BR"; \
	fi
> @echo "[ok] draft prepared under docs/release_notes and dist/changelog"

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
> echo ">>> gh release create $(RELEASE_TAG)"
> echo "(gh CLI optional)"
> mkdir -p dist
> PUBLISH_NOTES=dist/release_notes_for_publish_$(RELEASE_VERSION).md; \
> cat docs/release_notes/release_notes_v$(RELEASE_VERSION).md > $$PUBLISH_NOTES; \
> if [ -n "$(ISSUES)" ]; then \
>   for i in $(ISSUES); do echo "\nCloses #$$i" >> $$PUBLISH_NOTES; done; \
> fi; \
> gh release create $(RELEASE_TAG) \
>   --title "$(RELEASE_TAG)" \
>   --notes-file "$$PUBLISH_NOTES" \
>   dist/changelog/CHANGELOG.dryrun.md \
>   dist/provenance/pip_freeze_$$(date -u +%F).txt dist/assets/$(RELEASE_VERSION)/* || true
> echo "[ok] GitHub Release created (or gh not available)"
> gh release create $(RELEASE_TAG) \
>   --title "$(RELEASE_TAG)" \
>   --notes-file "docs/release_notes/release_notes_$(RELEASE_VERSION).md" \
>   dist/changelog/CHANGELOG.dryrun.md \
>   dist/provenance/pip_freeze_$$(date -u +%F).txt || true
> @echo "[ok] GitHub Release created (or gh not available)"

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
