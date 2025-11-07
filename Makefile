# ================================
# Practicum2 — N-of-1 ADHD + BD
# Canonical Makefile (v4-series)
# - Uses ".RECIPEPREFIX := >" so every recipe line starts with "> " (space).
# - Detects Windows venv (.venv/Scripts/python.exe) vs POSIX (.venv/bin/python).
# - PYTHONPATH portable for imports from src/ on Windows & POSIX.
# - SNAPSHOT=auto resolves to today's date (YYYY-MM-DD).
# ================================

.RECIPEPREFIX := >
SHELL := /usr/bin/env bash

# -------- Environment --------
PYTHON := $(if $(wildcard .venv/Scripts/python.exe),.venv/Scripts/python.exe,$(if $(wildcard .venv/bin/python),.venv/bin/python,python))
VENV_DIR := .venv
PID ?= P000001
SNAPSHOT ?= auto
ETL_CMD ?= full
DRY_RUN ?= 0
REPO_ROOT ?= .
ETL_TQDM ?= 1
export ETL_TQDM

# Zepp ZIP password (fail-fast if ZIP exists but no password)
ZPWD ?= $(ZEP_ZIP_PASSWORD)

# Portable PYTHONPATH (":" on POSIX, ";" on Windows)
PATHSEP := $(shell $(PYTHON) -c "import os; print(os.pathsep)")
export PYTHONPATH := $(REPO_ROOT)$(PATHSEP)$(REPO_ROOT)/src

# Resolve SNAPSHOT=auto -> YYYY-MM-DD (today)
SNAPSHOT_RESOLVED := $(shell $(PYTHON) -c "import datetime; s='$(SNAPSHOT)'; print(datetime.date.today().isoformat() if s=='auto' else s)")

ifeq ($(DRY_RUN),1)
DRY_FLAG := --dry-run
else
DRY_FLAG :=
endif

# Canonical paths
ETL_DIR := data/etl/$(PID)/$(SNAPSHOT_RESOLVED)
AI_DIR  := data/ai/$(PID)/$(SNAPSHOT_RESOLVED)

# -------- Phony --------
.PHONY: help env check-dirs \
        ingest aggregate unify segment label prep-nb2 nb2 nb3 report \
        pipeline quick nb2-only nb3-only \
        clean-outputs clean-all verify \
        help-release release-notes version-guard changelog release-assets provenance release-draft publish-release print-version

# -------- Help --------
help:
> echo "Usage:"
> echo "  make pipeline PID=$(PID) SNAPSHOT=$(SNAPSHOT) [ZPWD=***]"
> echo "  make nb2-only | nb3-only | quick"
> echo "  make verify | clean-outputs | help-release"
> echo "Vars: PID=$(PID) SNAPSHOT=$(SNAPSHOT) -> $(SNAPSHOT_RESOLVED)"

# -------- Env / guards --------
env:
> $(PYTHON) -V
> [ -d data/raw/$(PID) ] || (echo "ERR: data/raw/$(PID) not found"; exit 1)
> @# Fail-fast if Zepp ZIP exists but no password provided
> @if ls data/raw/$(PID)/zepp/*.zip >/dev/null 2>&1; then \
>   if [ -z "$(ZPWD)" ]; then \
>     echo "ERR: Zepp ZIP detected but no password provided (set ZEP_ZIP_PASSWORD or pass ZPWD=...)"; \
>     exit 2; \
>   fi; \
> fi

check-dirs:
> mkdir -p $(ETL_DIR) $(AI_DIR)

# -------- Stage wrappers (call the orchestrator with start/end) --------
ingest: env
> $(PYTHON) scripts/run_full_pipeline.py --participant $(PID) --snapshot $(SNAPSHOT_RESOLVED) --start-stage 0 --end-stage 0 $(if $(ZPWD),--zepp-password "$(ZPWD)") $(DRY_FLAG)

aggregate: env
> $(PYTHON) scripts/run_full_pipeline.py --participant $(PID) --snapshot $(SNAPSHOT_RESOLVED) --start-stage 1 --end-stage 1 $(if $(ZPWD),--zepp-password "$(ZPWD)") $(DRY_FLAG)

unify: env
> $(PYTHON) scripts/run_full_pipeline.py --participant $(PID) --snapshot $(SNAPSHOT_RESOLVED) --start-stage 2 --end-stage 2 $(if $(ZPWD),--zepp-password "$(ZPWD)") $(DRY_FLAG)

segment: env
> $(PYTHON) scripts/run_full_pipeline.py --participant $(PID) --snapshot $(SNAPSHOT_RESOLVED) --start-stage 3 --end-stage 4 $(if $(ZPWD),--zepp-password "$(ZPWD)") $(DRY_FLAG)

label: env
> $(PYTHON) scripts/run_full_pipeline.py --participant $(PID) --snapshot $(SNAPSHOT_RESOLVED) --start-stage 3 --end-stage 3 $(if $(ZPWD),--zepp-password "$(ZPWD)") $(DRY_FLAG)

prep-nb2: env
> $(PYTHON) scripts/prepare_nb2_dataset.py $(ETL_DIR)/joined/features_daily_labeled.csv --output $(ETL_DIR)/joined/features_nb2_clean.csv

nb2: env
> $(PYTHON) scripts/run_full_pipeline.py --participant $(PID) --snapshot $(SNAPSHOT_RESOLVED) --start-stage 5 --end-stage 6 $(if $(ZPWD),--zepp-password "$(ZPWD)") $(DRY_FLAG)

nb3: env
> $(PYTHON) scripts/run_full_pipeline.py --participant $(PID) --snapshot $(SNAPSHOT_RESOLVED) --start-stage 7 --end-stage 8 $(if $(ZPWD),--zepp-password "$(ZPWD)") $(DRY_FLAG)

report: env
> $(PYTHON) scripts/run_full_pipeline.py --participant $(PID) --snapshot $(SNAPSHOT_RESOLVED) --start-stage 9 --end-stage 9 $(if $(ZPWD),--zepp-password "$(ZPWD)") $(DRY_FLAG)
> echo "RUN_REPORT -> ./RUN_REPORT.md"

# -------- One-shot flows --------
pipeline: env
> $(PYTHON) scripts/run_full_pipeline.py --participant $(PID) --snapshot $(SNAPSHOT_RESOLVED) $(if $(ZPWD),--zepp-password "$(ZPWD)") $(DRY_FLAG)

quick: env
> $(PYTHON) scripts/run_full_pipeline.py --participant $(PID) --snapshot $(SNAPSHOT_RESOLVED) --start-stage 2 --end-stage 9 $(if $(ZPWD),--zepp-password "$(ZPWD)") $(DRY_FLAG)

nb2-only: env
> $(PYTHON) scripts/run_full_pipeline.py --participant $(PID) --snapshot $(SNAPSHOT_RESOLVED) --start-stage 5 --end-stage 6 $(if $(ZPWD),--zepp-password "$(ZPWD)") $(DRY_FLAG)

nb3-only: env
> $(PYTHON) scripts/run_full_pipeline.py --participant $(PID) --snapshot $(SNAPSHOT_RESOLVED) --start-stage 7 --end-stage 8 $(if $(ZPWD),--zepp-password "$(ZPWD)") $(DRY_FLAG)

# -------- Cleaning --------
clean-outputs:
> rm -rf data/extracted data/etl data/ai
> echo "Cleaned extracted/etl/ai. Raw preserved."

clean-all: clean-outputs
> rm -f RUN_REPORT.md
> echo "Repo cleaned (except data/raw)."

# -------- Verification (light) --------
verify:
> echo "Check outputs for $(PID)/$(SNAPSHOT_RESOLVED)..."
> test -f $(ETL_DIR)/joined/features_daily_unified.csv
> test -f $(ETL_DIR)/joined/features_daily_labeled.csv
> test -f $(ETL_DIR)/joined/features_nb2_clean.csv
> test -f $(ETL_DIR)/segment_autolog.csv
> test -f $(AI_DIR)/nb2/cv_summary.json
> test -f $(AI_DIR)/nb3/shap_summary.md
> test -f $(AI_DIR)/nb3/drift_report.md
> test -f $(AI_DIR)/nb3/lstm_report.md
> test -f $(AI_DIR)/nb3/models/best_model.tflite
> test -f $(AI_DIR)/nb3/latency_stats.json
> echo "OK"

# -------- Release & Publication --------
VERSION_FILE := docs/release_notes/VERSION
RELEASE_DIR  := docs/release_notes
ASSET_BASE   := dist/assets
NEXT_TAG     := v$(shell $(PYTHON) -c "import datetime as d; print(d.date.today().strftime('%Y.%m.%d'))")
NOTES_FILE   := $(RELEASE_DIR)/release_notes_$(NEXT_TAG).md
ASSET_DIR    := $(ASSET_BASE)/$(NEXT_TAG)

help-release:
> echo "Release targets:"
> echo "  make release-notes | version-guard | changelog"
> echo "  make release-assets | provenance | release-draft | publish-release | print-version"

release-notes:
> mkdir -p $(RELEASE_DIR)
> echo "# Practicum N-of-1 Pipeline $(NEXT_TAG)" > $(NOTES_FILE)
> echo "" >> $(NOTES_FILE)
> echo "Generated on $$($(PYTHON) -c 'import datetime as d; print(d.datetime.now().isoformat(timespec=\"seconds\"))')" >> $(NOTES_FILE)
> echo "" >> $(NOTES_FILE)
> echo "Key Results (snapshot $(SNAPSHOT_RESOLVED)):" >> $(NOTES_FILE)
> echo "- Expanded pipeline (≈2,828 days, 119 segments)" >> $(NOTES_FILE)
> echo "- NB2 macro-F1 ≈ 0.81; LSTM macro-F1 ≈ 0.25" >> $(NOTES_FILE)
> echo "- Drift: ADWIN=11; KS≈102 significant" >> $(NOTES_FILE)
> echo "" >> $(NOTES_FILE)
> echo "Artifacts:" >> $(NOTES_FILE)
> echo "- $(ETL_DIR)/joined/features_daily_labeled.csv" >> $(NOTES_FILE)
> echo "- $(AI_DIR)/nb3/models/best_model.tflite" >> $(NOTES_FILE)
> echo "- RUN_REPORT.md" >> $(NOTES_FILE)
> echo "Done -> $(NOTES_FILE)"

version-guard:
> echo "Checking working tree..."
> git diff-index --quiet HEAD -- || (echo "Uncommitted changes. Commit first."; exit 1)
> echo "OK"

changelog: release-notes
> echo "Updating CHANGELOG.md ..."
> cat $(NOTES_FILE) >> CHANGELOG.md
> echo "Done"

release-assets:
> echo "Collecting assets into $(ASSET_DIR)..."
> mkdir -p $(ASSET_DIR)
> cp -f $(ETL_DIR)/joined/features_daily_labeled.csv $(ASSET_DIR)/ 2>/dev/null || true
> cp -f $(AI_DIR)/nb3/models/best_model.tflite $(ASSET_DIR)/ 2>/dev/null || true
> cp -f RUN_REPORT.md $(ASSET_DIR)/ 2>/dev/null || true
> echo "OK"

provenance:
> mkdir -p $(ASSET_DIR)
> echo "version,tag,date" > $(ASSET_DIR)/provenance.csv
> echo "$(NEXT_TAG),$$(git rev-parse HEAD),$$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> $(ASSET_DIR)/provenance.csv
> $(PYTHON) - <<'PY'
> import hashlib,glob,os,sys
> files=glob.glob("$(ASSET_DIR)/*")
> with open("$(ASSET_DIR)/md5_manifest.txt","w") as f:
>     for p in files:
>         if os.path.isfile(p):
>             m=hashlib.md5(open(p,"rb").read()).hexdigest()
>             f.write(f"{m}  {os.path.basename(p)}\n")
> print("MD5 manifest written")
> PY

release-draft: version-guard release-assets provenance
> echo "Draft ready (see $(ASSET_DIR) and $(NOTES_FILE))"

publish-release:
> echo "Tagging $(NEXT_TAG) ..."
> git tag -a $(NEXT_TAG) -m "Release $(NEXT_TAG)"
> git push origin $(NEXT_TAG)
> echo "Tagged & pushed"

print-version:
> echo "HEAD: $$(git rev-parse --short HEAD)"
> echo "Next tag: $(NEXT_TAG)"
