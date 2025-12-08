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

# Zepp ZIP password (optional; if missing, Zepp extraction skipped)
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
        ingest aggregate unify segment label prep-ml6 nb2 nb3 report report-extended \
        pipeline quick ml6-only ml7-only \
        ml6-rf ml6-xgb ml6-lgbm ml6-svm ml7-gru ml7-tcn ml7-mlp ml-extended-all \
        qc-cardio qc-activity qc-sleep qc-meds qc-som qc-unified-ext qc-labels qc-all qc-etl \
        clean-outputs clean-all verify \
        help-release release-notes version-guard changelog release-assets provenance release-draft publish-release

# -------- Help --------
help:
> echo "Usage:"
> echo "  make pipeline PID=$(PID) SNAPSHOT=$(SNAPSHOT) [ZPWD=***]"
> echo "  make ml6-only | ml7-only | quick"
> echo "  make ml6-rf | ml6-xgb | ml6-lgbm | ml6-svm"
> echo "  make ml7-gru | ml7-tcn | ml7-mlp | ml-extended-all"
> echo "  make qc-cardio | qc-activity | qc-sleep | qc-meds | qc-som | qc-all"
> echo "  make verify | clean-outputs | help-release"
> echo "Vars: PID=$(PID) SNAPSHOT=$(SNAPSHOT) -> $(SNAPSHOT_RESOLVED)"

# -------- Env / guards --------
env:
> $(PYTHON) -V
> [ -d data/raw/$(PID) ] || (echo "ERR: data/raw/$(PID) not found"; exit 1)
> @# Warn if Zepp ZIP exists but no password provided (non-fatal)
> @if ls data/raw/$(PID)/zepp/*.zip >/dev/null 2>&1; then \
>   if [ -z "$(ZPWD)" ]; then \
>     echo "[WARN] Zepp ZIP detected but password not provided. Skipping Zepp extraction."; \
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

prep-ml6: env
> $(PYTHON) scripts/prepare_ml6_dataset.py $(ETL_DIR)/joined/features_daily_labeled.csv --output $(ETL_DIR)/joined/features_ml6_clean.csv

ml6: env
> $(PYTHON) scripts/run_full_pipeline.py --participant $(PID) --snapshot $(SNAPSHOT_RESOLVED) --start-stage 5 --end-stage 6 $(if $(ZPWD),--zepp-password "$(ZPWD)") $(DRY_FLAG)

ml7: env
> $(PYTHON) scripts/run_full_pipeline.py --participant $(PID) --snapshot $(SNAPSHOT_RESOLVED) --start-stage 7 --end-stage 8 $(if $(ZPWD),--zepp-password "$(ZPWD)") $(DRY_FLAG)

report: env
> $(PYTHON) scripts/run_full_pipeline.py --participant $(PID) --snapshot $(SNAPSHOT_RESOLVED) --start-stage 9 --end-stage 9 $(if $(ZPWD),--zepp-password "$(ZPWD)") $(DRY_FLAG)
> echo "RUN_REPORT -> ./RUN_REPORT.md"

report-extended:
> @echo "Generating extended model report..."
> $(PYTHON) scripts/generate_extended_report.py --participant $(PID) --snapshot $(SNAPSHOT_RESOLVED)
> echo "[OK] RUN_REPORT_EXTENDED.md -> ./RUN_REPORT_EXTENDED.md"

# -------- QC / Audits --------
qc-cardio:
> @echo "Running Cardio (HR) feature integrity audit..."
> $(PYTHON) -m src.etl.etl_audit --participant $(PID) --snapshot $(SNAPSHOT_RESOLVED) --domain cardio

qc-activity:
> @echo "Running Activity (Steps) feature integrity audit..."
> $(PYTHON) -m src.etl.etl_audit --participant $(PID) --snapshot $(SNAPSHOT_RESOLVED) --domain activity

qc-sleep:
> @echo "Running Sleep feature integrity audit..."
> $(PYTHON) -m src.etl.etl_audit --participant $(PID) --snapshot $(SNAPSHOT_RESOLVED) --domain sleep

qc-meds:
> @echo "Running Meds feature integrity audit..."
> $(PYTHON) -m src.etl.etl_audit --participant $(PID) --snapshot $(SNAPSHOT_RESOLVED) --domain meds

qc-som:
> @echo "Running SoM feature integrity audit..."
> $(PYTHON) -m src.etl.etl_audit --participant $(PID) --snapshot $(SNAPSHOT_RESOLVED) --domain som

qc-unified-ext:
> @echo "Running Unified extension audit..."
> $(PYTHON) -m src.etl.etl_audit --participant $(PID) --snapshot $(SNAPSHOT_RESOLVED) --domain unified_ext

qc-labels:
> @echo "Running Labels layer audit..."
> $(PYTHON) -m src.etl.etl_audit --participant $(PID) --snapshot $(SNAPSHOT_RESOLVED) --domain labels

qc-all: qc-cardio qc-activity qc-sleep qc-meds qc-som qc-unified-ext qc-labels
> @echo "All domain audits complete"

# Alias for backward compatibility
qc-etl: qc-cardio

# -------- One-shot flows --------
pipeline: env
> $(PYTHON) scripts/run_full_pipeline.py --participant $(PID) --snapshot $(SNAPSHOT_RESOLVED) $(if $(ZPWD),--zepp-password "$(ZPWD)") $(DRY_FLAG)

quick: env
> $(PYTHON) scripts/run_full_pipeline.py --participant $(PID) --snapshot $(SNAPSHOT_RESOLVED) --start-stage 2 --end-stage 9 $(if $(ZPWD),--zepp-password "$(ZPWD)") $(DRY_FLAG)

ml6-only: env
> $(PYTHON) scripts/run_full_pipeline.py --participant $(PID) --snapshot $(SNAPSHOT_RESOLVED) --start-stage 5 --end-stage 6 $(if $(ZPWD),--zepp-password "$(ZPWD)") $(DRY_FLAG)

ml7-only: env
> $(PYTHON) scripts/run_full_pipeline.py --participant $(PID) --snapshot $(SNAPSHOT_RESOLVED) --start-stage 7 --end-stage 8 $(if $(ZPWD),--zepp-password "$(ZPWD)") $(DRY_FLAG)

# -------- ML6/ML7 Extended Models --------
# NOTE: Extended models use preprocessed Stage 5 outputs (no data/raw or Zepp password needed)
ml6-rf:
> @echo "Running ML6 Random Forest with instability regularization..."
> $(PYTHON) scripts/run_extended_models.py --which ml6 --models rf --participant $(PID) --snapshot $(SNAPSHOT_RESOLVED)

ml6-xgb:
> @echo "Running ML6 XGBoost with instability regularization..."
> $(PYTHON) scripts/run_extended_models.py --which ml6 --models xgb --participant $(PID) --snapshot $(SNAPSHOT_RESOLVED)

ml6-lgbm:
> @echo "Running ML6 LightGBM with instability regularization..."
> $(PYTHON) scripts/run_extended_models.py --which ml6 --models lgbm --participant $(PID) --snapshot $(SNAPSHOT_RESOLVED)

ml6-svm:
> @echo "Running ML6 SVM (no instability penalty)..."
> $(PYTHON) scripts/run_extended_models.py --which ml6 --models svm --participant $(PID) --snapshot $(SNAPSHOT_RESOLVED)

ml7-gru:
> @echo "Running ML7 GRU..."
> $(PYTHON) scripts/run_extended_models.py --which ml7 --models gru --participant $(PID) --snapshot $(SNAPSHOT_RESOLVED)

ml7-tcn:
> @echo "Running ML7 TCN (Temporal Convolutional Network)..."
> $(PYTHON) scripts/run_extended_models.py --which ml7 --models tcn --participant $(PID) --snapshot $(SNAPSHOT_RESOLVED)

ml7-mlp:
> @echo "Running ML7 Temporal MLP..."
> $(PYTHON) scripts/run_extended_models.py --which ml7 --models mlp --participant $(PID) --snapshot $(SNAPSHOT_RESOLVED)

ml-extended-all:
> @echo "Running ALL extended ML6/ML7 models..."
> $(PYTHON) scripts/run_extended_models.py --which all --models all --participant $(PID) --snapshot $(SNAPSHOT_RESOLVED)

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
> test -f $(ETL_DIR)/joined/features_ml6_clean.csv
> test -f $(ETL_DIR)/segment_autolog.csv
> test -f $(AI_DIR)/ml6/cv_summary.json
> test -f $(AI_DIR)/ml7/shap_summary.md
> test -f $(AI_DIR)/ml7/drift_report.md
> test -f $(AI_DIR)/ml7/lstm_report.md
> test -f $(AI_DIR)/ml7/models/best_model.tflite
> test -f $(AI_DIR)/ml7/latency_stats.json
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
> echo "  make release-assets | provenance | release-draft | publish-release"

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
> echo "- $(AI_DIR)/ml7/models/best_model.tflite" >> $(NOTES_FILE)
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
> cp -f $(AI_DIR)/ml7/models/best_model.tflite $(ASSET_DIR)/ 2>/dev/null || true
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
