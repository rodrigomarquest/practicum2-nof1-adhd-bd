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
SNAPSHOT ?= 2025-09-29

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

etl:
> echo ">>> etl: running src.etl_pipeline"
> PYTHONPATH="$$PWD" $(PYTHON) -m src.etl_pipeline

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
> @echo "[ok] wrote dist/changelog/CHANGES_SINCE_LAST_TAG.txt"

release-draft:
> echo ">>> draft: release notes + changelog (dry-run)"
> @mkdir -p docs/release_notes dist/changelog
> $(PYTHON) -m src.tools.render_release_from_templates \
>   --version $(RELEASE_VERSION) \
>   --tag $(RELEASE_TAG) \
>   --title "Release $(RELEASE_TAG)" \
>   --summary "Auto-generated draft. See CHANGES_SINCE_LAST_TAG.txt." \
>   --branch $(RELEASE_BRANCH)
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

release-push:
> echo ">>> pushing branch + tags"
> @git push origin $$(git rev-parse --abbrev-ref HEAD)
> @git push origin --tags
> @echo "[ok] pushed"

release-publish:
> echo ">>> gh release create $(RELEASE_TAG)"
> echo "(gh CLI optional)"
> gh release create $(RELEASE_TAG) \
>   --title "$(RELEASE_TAG)" \
>   --notes-file "docs/release_notes/release_notes_$(RELEASE_VERSION).md" \
>   dist/changelog/CHANGELOG.dryrun.md \
>   dist/provenance/pip_freeze_$$(date -u +%F).txt || true
> @echo "[ok] GitHub Release created (or gh not available)"

release-final: release-verify release-summary release-draft release-freeze release-tag
> @echo "[ok] local release finalized: $(RELEASE_TAG)"

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

qc:
> echo ">>> qc: running src.eda"
> $(PYTHON) -m src.eda

PACK_OUT := dist/assets

pack-kaggle:
> @echo ">>> pack-kaggle: packaging Kaggle dataset snapshot for $(PID) $(SNAPSHOT)"
> @PID=$(PID) SNAPSHOT=$(SNAPSHOT) $(PYTHON) - <<'PY'
> import zipfile, pathlib, sys, os
> pid = os.environ.get("PID", "$(PID)")
> snap = os.environ.get("SNAPSHOT", "$(SNAPSHOT)")
> base = pathlib.Path(f"data/etl/{pid}/snapshots/{snap}/joined")
> files = list(base.glob("features_daily*.csv"))
> v = base / "version_log_enriched.csv"
> if v.exists(): files.append(v)
> readme = pathlib.Path("README.md")
> if readme.exists(): files.append(readme)
> if not files:
>     print("No files found to package from", base)
>     sys.exit(1)
> outdir = pathlib.Path("dist/assets")
> outdir.mkdir(parents=True, exist_ok=True)
> zipfn = outdir / f"{pid}_{snap}_ai.zip"
> with zipfile.ZipFile(zipfn, "w", compression=zipfile.ZIP_DEFLATED) as z:
>     for f in files:
>         z.write(f, arcname=f.name)
> print("Wrote", zipfn)
> PY

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
