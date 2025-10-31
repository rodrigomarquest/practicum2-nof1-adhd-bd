# -------- Makefile v4.0.0 --------
PY_WIN := py -3.13
PY_UNIX := python3.13
PY := $(PY_UNIX)
OS := $(shell uname 2>/dev/null)
ifdef COMSPEC
  PY := $(PY_WIN)
endif

VENV=.venv
PIP=$(VENV)/bin/pip
PYTHON=$(VENV)/bin/python
ifeq ($(OS),)
  PIP=$(VENV)/Scripts/pip.exe
  PYTHON=$(VENV)/Scripts/python.exe
endif

.PHONY: venv-create venv-recreate freeze etl labels qc clean pack-kaggle clean-data clean-all

venv-create:
	@echo ">>> Creating venv"
	$(PY) -m venv $(VENV)
	$(PYTHON) -m pip install --upgrade pip setuptools wheel
	$(PIP) install -r requirements.txt -r requirements_dev.txt || \
		$(PIP) install -r requirements_etl.txt -r requirements_dev.txt
	$(MAKE) freeze

venv-recreate:
	@echo ">>> Recreating venv"
	- rm -rf $(VENV)
	$(MAKE) venv-create

freeze:
	@echo ">>> Freezing environment"
	$(PYTHON) -c "import platform; print(platform.python_version())"
	$(PIP) freeze > provenance/pip_freeze_$$(date +%F).txt || $(PIP) freeze > provenance/pip_freeze.txt

etl:
	@echo ">>> Running ETL pipeline"
	$(PYTHON) -m src.etl_pipeline

labels:
	@echo ">>> Applying label rules"
	$(PYTHON) -m src.make_labels --rules config/label_rules.yaml --in data/etl/FEATURES_PATH/features_daily.csv --out data/etl/FEATURES_PATH/features_daily_labeled.csv

qc:
	@echo ">>> Running EDA/QC"
	$(PYTHON) -m src.eda

pack-kaggle:
	@echo ">>> Packing dataset for Kaggle"
	# Implement: zip features_daily(_labeled).csv + version_log_enriched.csv + README into dist/assets/<slug>.zip

clean:
	@echo ">>> clean: caches and temp files"
	- find . -name "__pycache__" -type d -prune -exec rm -rf {} + 2>/dev/null || true
	- find . -name ".ipynb_checkpoints" -type d -prune -exec rm -rf {} + 2>/dev/null || true
	- find . -name "*.pyc" -delete 2>/dev/null || true
	- find . -name "*.log" -delete 2>/dev/null || true
	@echo "✔ caches/logs removed"

# Remove dataset outputs (but keep source code and configs)
clean-data:
	@echo ">>> clean-data: ETL/AI outputs and reports"
	- rm -rf notebooks/outputs dist/assets logs backups processed 2>/dev/null || true
	- rm -rf data/etl data/ai 2>/dev/null || true
	@echo "✔ data outputs removed"

# Full sweep: clean + data + common leftovers introduced pre-v4
clean-all: clean
	@echo ">>> clean-all: legacy dirs (pre-v4) and provenance artifacts"
	- rm -rf data_ai data_ai_legacy_* reports 2>/dev/null || true
	- rm -rf etl_tools make_scripts processed apple_etl_cache 2>/dev/null || true
	- rm -rf notebooks/eda_outputs 2>/dev/null || true
	@echo "✔ everything swept (safe)"

# PROJECT RULES FOR MAKE (READ FIRST)
# 1) Never use heredoc (<<EOF) in Make recipes.
# 2) Never put multi-line shell/python inline in Make; put logic in Python files under make_scripts/ and call them.
# 3) Do not change .RECIPEPREFIX; use standard TAB recipes.
# 4) All new command logic must live in make_scripts/*.py with a CLI (argparse) and be called from Make with env vars.
# 5) Keep Make targets thin: pass env -> call script -> print summary.
#
# The lint target `lint-make` will fail if heredoc tokens are present.
# ============================================================
# Practicum2 – N-of-1 ADHD + BD  ·  Makefile
# Targets for venv, install, iOS extraction, ETL and docs
# Usage examples:
#   make venv install
#   make decrypt probe extract-plists plist-csv
#   make extract-knowledgec parse-knowledgec
#   make etl
#   make help
# ============================================================

# --- Config --------------------------------------------------
.RECIPEPREFIX := > 
SHELL := /usr/bin/env bash
PY    ?= python
# Absolute path to the Python interpreter used to run Make/PY
PY_ABS := $(shell $(PY) -c "import sys,os;print(os.path.abspath(sys.executable))")
DRY_RUN ?= 0
DRY_RUN ?= 0
NON_INTERACTIVE ?= 0
RELEASE_DRY_RUN ?= 0
PIP   ?= $(PY) -m pip
# Resolve venv python interpreter path (prefer Windows Scripts, then POSIX bin). Falls back to $(PY)
VENV_PY ?= $(shell if [ -x .venv/Scripts/python.exe ]; then echo $(abspath .venv/Scripts/python.exe); elif [ -x .venv/Scripts/python ]; then echo $(abspath .venv/Scripts/python); elif [ -x .venv/bin/python ]; then echo $(abspath .venv/bin/python); else echo $(PY_ABS); fi)
PID ?= P000001
TZ  ?= Europe/Dublin

# Participant alias (some older scripts use PARTICIPANT)
PARTICIPANT ?= $(PID)
SNAP ?= $(SNAPSHOT_DATE)

# Consolidated directory layout (defaults)
# RAW_DIR, ETL_DIR and AI_DIR include the participant subfolder
RAW_DIR := data/raw/$(PARTICIPANT)
ETL_DIR := data/etl/$(PARTICIPANT)
AI_DIR  := data/ai/$(PARTICIPANT)

# --- Version Info ---------------------------------------------------
TAG_PREFIX ?= v

# Get the latest annotated tag (no commit suffix)
LATEST_TAG := $(shell git tag --list 'v[0-9]*' --sort=-creatordate 2>/dev/null | head -n1 || git describe --tags --abbrev=0 2>/dev/null || echo "v0.0.0")
# Back-compat raw tag variable used elsewhere
LATEST_TAG_RAW := $(LATEST_TAG)
# Extract numeric version (remove leading 'v')
LATEST_VERSION := $(subst $(TAG_PREFIX),,$(LATEST_TAG))
# Strip any git-describe metadata suffix from LATEST_VERSION (e.g., 1.2.3-20-gabcd)
LATEST_VERSION_CORE := $(shell $(PY) -c "import re; v='$(LATEST_VERSION)'; m=re.match(r'^(\d+\.\d+\.\d+)', v); print(m.group(1) if m else '0.0.0')")
# Get the short hash of the current commit
LATEST_COMMIT_ID := $(shell git rev-parse --short HEAD 2>/dev/null)
# Count commits since the latest tag
COMMITS_SINCE_LATEST_TAG := $(shell git rev-list --count $(LATEST_TAG)..HEAD 2>/dev/null)

# Auto-increment patch version if NEXT_VERSION not manually set
NEXT_VERSION ?= $(shell python -c "v='$(LATEST_VERSION_CORE)'; parts=v.split('.'); parts[-1]=str(int(parts[-1])+1); print('.'.join(parts))")
# Compose the next tag
NEXT_TAG := $(TAG_PREFIX)$(NEXT_VERSION)

# Backwards-compatible aliases used by existing scripts
VERSION ?= $(NEXT_VERSION)
TAG := $(NEXT_TAG)
RELEASE_TITLE ?= Tooling & Provenance Refactor

# Discover latest tag (prefer semver-like tags). Fallback to git describe or v0.0.0
LATEST_TAG_RAW  := $(shell git tag --list 'v[0-9]*' --sort=-creatordate 2>/dev/null | head -n1 || git describe --tags --abbrev=0 2>/dev/null || echo v0.0.0)
# LATEST_TAG is identical to LATEST_TAG_RAW but kept for backwards readability
LATEST_TAG := $(LATEST_TAG_RAW)
# Strip leading 'v' for the version number and remove any git-describe suffixes
LATEST_VERSION := $(shell $(PY) -c "import re,sys;v=sys.argv[1].lstrip('v');print(re.split(r'[-+]',v)[0])" "$(LATEST_TAG_RAW)")

# COMMITS_SINCE_LATEST_TAG: number of commits since latest tag
COMMITS_SINCE_LATEST_TAG := $(shell git rev-list --count $(LATEST_TAG_RAW)..HEAD 2>/dev/null || echo 0)
# LATEST_COMMIT_ID: short sha of HEAD
LATEST_COMMIT_ID := $(shell git rev-parse --short HEAD 2>/dev/null || echo unknown)

# NEXT_VERSION: bump patch of LATEST_VERSION (X.Y.Z -> X.Y.(Z+1)). If VERSION provided, use that.
NEXT_VERSION := $(if $(VERSION),$(VERSION),$(shell $(PY) -c "import sys;v=sys.argv[1];parts=v.split('.');parts+=['0']*(3-len(parts));p=parts[-1];parts[-1]=str(int(p)+1) if p.isdigit() else p+'1';print('.'.join(parts))" "$(LATEST_VERSION)"))

NEXT_TAG := $(TAG_PREFIX)$(NEXT_VERSION)
# If VERSION was provided explicitly, keep TAG consistent for backwards compat
TAG := $(if $(VERSION),$(if $(filter v%,$(VERSION)),$(VERSION),$(TAG_PREFIX)$(VERSION)),$(NEXT_TAG))


# Paths
BACKUP_DIR ?= C:\Users\Administrador\Apple\MobileSync\Backup\00008120-000E18E21144A01E
OUT_DIR    ?= decrypted_output

IOS_DIR    := ios_extract


# =========================
# Canonical path variables
# =========================

# Required runtime parameters (user-overridable)
PARTICIPANT   ?= P000001
SNAPSHOT_DATE ?= 2025-10-22

# Base roots (never include snapshot here)
DATA_BASE := ./data
RAW_DIR   := $(DATA_BASE)/raw/$(PARTICIPANT)
ETL_DIR   := $(DATA_BASE)/etl/$(PARTICIPANT)
AI_DIR    := $(DATA_BASE)/ai/$(PARTICIPANT)

# Snapshot-scoped roots (all ETL/AI writes happen under these)
ETL_SNAP_DIR := $(ETL_DIR)/snapshots/$(SNAPSHOT_DATE)
AI_SNAP_DIR  := $(AI_DIR)/snapshots/$(SNAPSHOT_DATE)

# ETL snapshot stages (write-only for stages)
EXTRACTED_DIR  := $(ETL_SNAP_DIR)/extracted
NORMALIZED_DIR := $(ETL_SNAP_DIR)/normalized
PROCESSED_DIR  := $(ETL_SNAP_DIR)/processed
JOINED_DIR     := $(ETL_SNAP_DIR)/joined

# AI snapshot outputs (model-ready, safe to publish subsets from here)
AI_INPUT_DIR := $(AI_SNAP_DIR)

# Provenance / reports (lightweight)
PROVENANCE_DIR := notebooks/eda_outputs/provenance

# =========================
# iOS / iTunes backup inputs (read-only)
# =========================
# Absolute or relative path to the *decrypted* iTunes backup root (never copied).
# Example external: C:/Users/Administrador/Apple/MobileSync/Backup/00008120-.../_decrypted
IOS_BACKUP_DIR ?= $(RAW_DIR)/ios/itunes_backup_$(SNAPSHOT_DATE)/_decrypted

# Commonly referenced files/locations inside the backup (resolved by scripts)
IOS_MANIFEST_DB     ?= $(IOS_BACKUP_DIR)/Manifest.db
IOS_DEVICEACTIVITY_GLOB ?= Library/DeviceActivity/**    # discovered via Manifest.db
IOS_KNOWLEDGEC_GLOB    ?= **/KnowledgeC*.db             # discovered via Manifest.db

# =========================
# Cross-stage defaults / flags
# =========================
DRY_RUN            ?= 0
QC_MODE            ?= flag            # flag|fail
NAN_THRESHOLD_PCT  ?= 5
HRV_METHOD         ?= both            # sdnn|rmssd|both
LOCK_TIMEOUT_SECS  ?= 900
FORCE_LOCK         ?= 0

# =========================
# Temporary back-compat aliases (deprecated)
# =========================
# NOTE: do not use these in new code; they resolve to the snapshot-scoped dirs above.
DATA_ETL_BASE  := $(ETL_DIR)         # legacy, alias to participant ETL root
DATA_AI_BASE   := $(AI_DIR)          # legacy, alias to participant AI root

# Legacy names mapped to snapshot-aware ones (soft-deprecated)
AI_SNAPSHOT_DIR := $(AI_SNAP_DIR)

# Per-target passthrough args (allow flag-style invocation)
CLEAN_ARGS ?=
ENV_ARGS ?=
REPORT_ARGS ?=
INTAKE_ARGS ?=
PROV_ARGS ?=
SOM_SCAN_FLAGS ?=


install-dev:
> $(PIP) install --upgrade -r requirements_dev.txt

install-all:
> $(PIP) install --upgrade -r requirements.txt

# --- iOS Extraction ------------------------------------------
decrypt:
> @IOS_BACKUP_DIR="$(IOS_BACKUP_DIR)" BACKUP_DIR="$(IOS_BACKUP_DIR)" EXTRACTED_DIR="$(EXTRACTED_DIR)" \
>  BACKUP_PASSWORD="$${BACKUP_PASSWORD:-}" \
>  $(PY) $(DEC_MANIFEST)

probe:
> @IOS_BACKUP_DIR="$(IOS_BACKUP_DIR)" BACKUP_DIR="$(IOS_BACKUP_DIR)" EXTRACTED_DIR="$(EXTRACTED_DIR)" BACKUP_PASSWORD="$${BACKUP_PASSWORD:-}" ; \
> $(PY) "$(PROBE)"

extract-plists:
> @IOS_BACKUP_DIR="$(IOS_BACKUP_DIR)" BACKUP_DIR="$(IOS_BACKUP_DIR)" EXTRACTED_DIR="$(EXTRACTED_DIR)" BACKUP_PASSWORD="$${BACKUP_PASSWORD:-}" ; \
> $(PY) "$(EXTRACT_PLISTS)"

plist-csv:
> @IOS_BACKUP_DIR="$(IOS_BACKUP_DIR)" BACKUP_DIR="$(IOS_BACKUP_DIR)" EXTRACTED_DIR="$(EXTRACTED_DIR)" BACKUP_PASSWORD="$${BACKUP_PASSWORD:-}" ; \
> $(PY) "$(PLISTS_TO_CSV)"

extract-knowledgec:
> @IOS_BACKUP_DIR="$(IOS_BACKUP_DIR)" BACKUP_DIR="$(IOS_BACKUP_DIR)" EXTRACTED_DIR="$(EXTRACTED_DIR)" BACKUP_PASSWORD="$${BACKUP_PASSWORD:-}" ; \
> $(PY) "$(EXTRACT_KNOWLEDGEC)"

>	@echo "  make init-data-layout       # create raw/extracted/normalized/processed/joined and ai/snapshots"
>	@echo "  make migrate-layout         # move legacy etl/.../exports -> data/raw/<pid>"
>	@echo "  make intake-zip ...         # ingest device/app zips into raw/<source> (optional minimal extract)"
>	@echo "  make clean-raw              # NO-OP unless flag --i-understand-raw is passed (safety)"
>	@echo "  make clean-extracted|normalized|processed|joined|ai-snapshot"
>	@echo "  make provenance             # run inventory & audit over normalized/processed/joined/ai snapshot"
>	@echo "  make lint-deprecated-exports"
>	@echo "  make lint-layout"
	@echo "  IOS_BACKUP_DIR: path to decrypted iTunes backup root (read-only)"
parse-knowledgec:
> @IOS_BACKUP_DIR="$(IOS_BACKUP_DIR)" BACKUP_DIR="$(IOS_BACKUP_DIR)" EXTRACTED_DIR="$(EXTRACTED_DIR)" BACKUP_PASSWORD="$${BACKUP_PASSWORD:-}" ; \
> if [ -f "$(PARSE_KNOWLEDGEC)" ]; then \
>   $(PY) "$(PARSE_KNOWLEDGEC)"; \
> else \
>   echo "parse_knowledgec_usage.py not present yet (will be added when schema is detected)."; \
> fi

.PHONY: help-layout
help-layout:
> 	@echo "Layout workflow targets and usage"

>	@echo "Makefile ETL runners:"
>	@echo "  make run-a6-apple PARTICIPANT=<PID> SNAPSHOT_DATE=<YYYY-MM-DD> [DRY_RUN=1]  # run A6 normalization"
>	@echo "  make run-a7-apple PARTICIPANT=<PID> SNAPSHOT_DATE=<YYYY-MM-DD> [DRY_RUN=1]  # run A7 daily aggregate & join"
>	@echo "  make run-a8-labels PARTICIPANT=<PID> SNAPSHOT_DATE=<YYYY-MM-DD> [DRY_RUN=1] # run A8 label integration"

probe-sqlite:
> @IOS_BACKUP_DIR="$(IOS_BACKUP_DIR)" BACKUP_DIR="$(IOS_BACKUP_DIR)" EXTRACTED_DIR="$(EXTRACTED_DIR)" BACKUP_PASSWORD="$${BACKUP_PASSWORD:-}" ; \
> $(PY) ios_extract/probe_sqlite_targets.py

extract-screentime-sqlite:
> @IOS_BACKUP_DIR="$(IOS_BACKUP_DIR)" BACKUP_DIR="$(IOS_BACKUP_DIR)" EXTRACTED_DIR="$(EXTRACTED_DIR)" BACKUP_PASSWORD="$${BACKUP_PASSWORD:-}" ; \
> $(PY) ios_extract/extract_screentime_sqlite.py

.PHONY: ios-backup-probe
ios-backup-probe:
> @echo "Probe iOS backup and write metadata to $(EXTRACTED_DIR)/ios/backup_paths.json"
> @IOS_BACKUP_DIR="$(IOS_BACKUP_DIR)" BACKUP_DIR="$(IOS_BACKUP_DIR)" BACKUP_PASSWORD="$${BACKUP_PASSWORD:-}" EXTRACTED_DIR="$(EXTRACTED_DIR)" PARTICIPANT="$(PARTICIPANT)" SNAPSHOT_DATE="$(SNAPSHOT_DATE)" \
> @$(VENV_PY) make_scripts/ios/ios_backup_probe.py --backup-root "$(IOS_BACKUP_DIR)" --manifest-db "$(IOS_MANIFEST_DB)" --out-dir "$(EXTRACTED_DIR)"

.PHONY: ios-manifest-probe
ios-manifest-probe:
> @echo "Probe Manifest.db and write manifest_probe outputs to $(EXTRACTED_DIR)/ios/"
> @IOS_MANIFEST_DB="$(IOS_MANIFEST_DB)" BACKUP_PASSWORD="$${BACKUP_PASSWORD:-}" EXTRACTED_DIR="$(EXTRACTED_DIR)" PARTICIPANT="$(PARTICIPANT)" SNAPSHOT_DATE="$(SNAPSHOT_DATE)" \
> $(VENV_PY) make_scripts/ios/manifest_probe.py --manifest-db "$(IOS_MANIFEST_DB)" --out-dir "$(EXTRACTED_DIR)"

.PHONY: ios-normalize-usage
ios-normalize-usage:
> @echo "Normalize iOS usage events into $(NORMALIZED_DIR)/ios/"
> @EXTRACTED_DIR="$(EXTRACTED_DIR)" NORMALIZED_DIR="$(NORMALIZED_DIR)" $(VENV_PY) make_scripts/ios/normalize_ios_usage.py --extracted-dir "$(EXTRACTED_DIR)" --normalized-dir "$(NORMALIZED_DIR)"

.PHONY: ios-daily-join
ios-daily-join:
> @echo "Aggregate iOS usage daily and join with AI features"
> @NORMALIZED_DIR="$(NORMALIZED_DIR)" PROCESSED_DIR="$(PROCESSED_DIR)" AI_INPUT_DIR="$(AI_INPUT_DIR)" JOINED_DIR="$(JOINED_DIR)" $(VENV_PY) make_scripts/ios/aggregate_join_ios_daily.py --normalized-dir "$(NORMALIZED_DIR)/ios" --ai-input "$(AI_INPUT_DIR)/features_daily.csv" --processed-dir "$(PROCESSED_DIR)" --joined-dir "$(JOINED_DIR)" --participant "$(PARTICIPANT)" --snapshot "$(SNAPSHOT_DATE)"

# Note: the `etl` alias (aliasing to `etl-full`) is defined later near the workflow help block.
# The full pipeline runner `etl-full` should be used for end-to-end runs.


# --- Zepp Export -------------------------------------------------
ZEPP_DIR := data_etl/P000001/zepp_export
ZEPP_ZIP ?= $(firstword $(wildcard $(ZEPP_DIR)/*.zip))
ZEPP_OUT ?= decrypted_output/zepp
ZEPP_ZIP_PASSWORD ?= $(ZEPP_ZIP_PASSWORD)

.PHONY: list-zepp parse-zepp unpack-zepp inspect-zepp zepp-parse-one zepp-aggregate

list-zepp:
> @test -n "$(ZEPP_ZIP)" || (echo "No .zip found in data_etl/$(PID)/zepp_export. Pass ZEPP_ZIP=... or put a zip there." && exit 2)
> @$(PY) -c "import os,zipfile; p=os.environ.get('ZEPP_ZIP','$(ZEPP_ZIP)'); z=zipfile.ZipFile(p); [print(n) for n in z.namelist()]"

# LEGADO: parse-zepp antigo (mantido p/ compat). Agora usa --outdir-root
# Ex.: make parse-zepp ZIP=data_etl/P000001/zepp_export/X.zip OUT=data_etl/P000001/zepp_processed PASS='senha'
parse-zepp:
> @echo "parse-zepp: parse Zepp ZIP/dir"

.PHONY: px8-lite
px8-lite:
>	@python make_scripts/etl_px8_lite.py \
>		--snapshot "$(ETL_SNAP_DIR)" \
>		--outdir "reports" \
>		--version-log "$(ETL_SNAP_DIR)/joined/version_log_enriched.csv" || exit $$?
> @echo "Running: PYTHONPATH=\"$$PWD\" $(PY) make_scripts/zepp/parse_zepp_make.py --input \"$(ZIP)\" --outdir-root \"$(OUT)\" --tz \"Europe/Dublin\" $$( [ -n "$(PASS)" ] && echo --password \"$(PASS)\" || true ) $$( [ "$(DRY_RUN)" = "1" ] && echo --dry-run || true )"
> @PYTHONPATH="$$PWD" $(PY) make_scripts/zepp/parse_zepp_make.py --input "$(ZIP)" --outdir-root "$(OUT)" --tz "Europe/Dublin" $$( [ -n "$(PASS)" ] && echo --password "$(PASS)" || true ) $$( [ "$(DRY_RUN)" = "1" ] && echo --dry-run || true )

# Uso:
# make unpack-zepp ZIP="data_etl/P000001/zepp_export/3088....zip" OUT="data_etl/P000001/zepp_raw_unpacked" PASS="sYhspDax"

# Parse a partir de diretório já extraído (contorna ZIP AES no Windows)
# Ex.: make zepp-parse-dir PID=P000001 DIR="data_etl/P000001/zepp_raw_unpacked"
.PHONY: zepp-parse-dir
zepp-parse-dir:
> @echo "zepp-parse-dir: parse zepp from dir"
> @echo "Running: PYTHONPATH=\"$$PWD\" $(PY) make_scripts/zepp/parse_zepp_make.py --input \"$(DIR)\" --outdir-root \"data_etl/$(PID)/zepp_processed\" --participant \"$(PID)\" --cutover \"$(CUTOVER)\" --tz_before \"$(TZ_BEFORE)\" --tz_after \"$(TZ_AFTER)\" $$( [ "$(DRY_RUN)" = "1" ] && echo --dry-run || true )"
> @PYTHONPATH="$$PWD" $(PY) make_scripts/zepp/parse_zepp_make.py --input "$(DIR)" --outdir-root "data_etl/$(PID)/zepp_processed" --participant "$(PID)" --cutover "$(CUTOVER)" --tz_before "$(TZ_BEFORE)" --tz_after "$(TZ_AFTER)" $$( [ "$(DRY_RUN)" = "1" ] && echo --dry-run || true )

.PHONY: zepp-zip-inventory
zepp-zip-inventory:
> @echo "Inventory Zepp ZIP and write deterministic metadata to $(EXTRACTED_DIR)/zepp"
> @ZIP="$(ZEPP_ZIP)" ZEPP_ZIP_PASSWORD="$(ZEPP_ZIP_PASSWORD)" $(VENV_PY) make_scripts/zepp/zepp_zip_inventory.py --zip-path "$(ZIP)" --out-dir "$(EXTRACTED_DIR)/zepp" --participant "$(PARTICIPANT)"

.PHONY: zepp-normalize
zepp-normalize:
> @echo "Normalize Zepp archive into $(NORMALIZED_DIR)/zepp/ (DRY_RUN honors env DRY_RUN)"
> @$(VENV_PY) make_scripts/zepp/zepp_normalize.py --zip-path "$(ZEPP_ZIP)" --filelist-tsv "$(EXTRACTED_DIR)/zepp/zepp_zip_filelist.tsv" --normalized-dir "$(NORMALIZED_DIR)/zepp" --participant "$(PARTICIPANT)" --snapshot "$(SNAPSHOT_DATE)" $$( [ "$(DRY_RUN)" = "1" ] && echo --dry-run || true )

.PHONY: zepp-daily-join
zepp-daily-join:
> @echo "Aggregate Zepp daily metrics and join with AI features"
> @NORMALIZED_DIR="$(NORMALIZED_DIR)" PROCESSED_DIR="$(PROCESSED_DIR)" AI_INPUT_DIR="$(AI_INPUT_DIR)" JOINED_DIR="$(JOINED_DIR)" $(VENV_PY) make_scripts/zepp/zepp_daily_aggregate.py --normalized-dir "$(NORMALIZED_DIR)/zepp" --ai-input "$(AI_INPUT_DIR)/features_daily.csv" --processed-dir "$(PROCESSED_DIR)" --joined-dir "$(JOINED_DIR)" --participant "$(PARTICIPANT)" --snapshot "$(SNAPSHOT_DATE)"

# Rebuild _latest a partir de todos os subdirs versionados (append-only)
# Ex.: make zepp-aggregate PID=P000001
zepp-aggregate:
> @echo "zepp-aggregate: rebuild latest from processed zepp for PID=$(PID)"
> @echo "Running: PYTHONPATH=\"$$PWD\" \"$(VENV_PY)\" etl_tools/zepp_rebuild_latest.py --root \"data_etl/$(PID)/zepp_processed\""
> @PYTHONPATH="$$PWD" "$(VENV_PY)" etl_tools/zepp_rebuild_latest.py --root "data_etl/$(PID)/zepp_processed"

# --- ETL Apple (um snapshot específico) -----------------------
# Ex.: make etl-one PID=P000001 SNAP=2025-09-29
etl-one:
> @test -n "$(PID)"  || (echo "Set PID=Pxxxxxx" && exit 2)
> @test -n "$(SNAP)" || (echo "Set SNAP=YYYY-MM-DD (ou YYYYMMDD)" && exit 2)
> @echo "etl-one: participant=$(PID) snapshot=$(SNAP)"
> @echo "Delegating export.xml check to make_scripts/etl_one_wrapper.sh"
> @make_scripts/etl_one_wrapper.sh "$(PID)" "$(SNAP)" "$(VENV_PY)" "$(if $(ETL_PIPELINE),$(ETL_PIPELINE),etl_pipeline.py)"

# Parse Apple Health export.xml into normalized per-metric CSVs
.PHONY: etl-apple-parse
etl-apple-parse:
> @test -n "$(PID)" || (echo "Set PID=Pxxxxxx" && exit 2)
> @echo "etl-apple-parse: participant=$(PID)" \
> && echo "Running: PYTHONPATH=\"$$PWD\" \"$(VENV_PY)\" etl/apple_inapp_parse.py --participant \"$(PID)\" --extracted-dir \"$(EXTRACTED_DIR)\" --normalized-dir \"$(NORMALIZED_DIR)\" $$( [ \"$(DRY_RUN)\" = \"1\" ] && echo --dry-run || true )" \
> && PYTHONPATH="$$PWD" "$(VENV_PY)" etl/apple_inapp_parse.py --participant "$(PID)" --extracted-dir "$(EXTRACTED_DIR)" --normalized-dir "$(NORMALIZED_DIR)" $$( [ "$(DRY_RUN)" = "1" ] && echo --dry-run || true )

.PHONY: etl-apple-qc
etl-apple-qc:
> @test -n "$(PID)" || (echo "Set PID=Pxxxxxx" && exit 2)
> @echo "etl-apple-qc: participant=$(PID)"
> @echo "Ensuring parse stage is up-to-date..."
> @$(MAKE) etl-apple-parse
> @echo "Running: PYTHONPATH=\"$$PWD\" \"$(VENV_PY)\" etl/apple_inapp_qc.py --participant \"$(PID)\" --normalized-dir \"$(NORMALIZED_DIR)\" --processed-dir \"$(PROCESSED_DIR)\" $$( [ \"$(DRY_RUN)\" = \"1\" ] && echo --dry-run || true )"
> @PYTHONPATH="$$PWD" "$(VENV_PY)" etl/apple_inapp_qc.py --participant "$(PID)" --normalized-dir "$(NORMALIZED_DIR)" --processed-dir "$(PROCESSED_DIR)" $$( [ "$(DRY_RUN)" = "1" ] && echo --dry-run || true )

.PHONY: etl-apple-daily
etl-apple-daily:
> @test -n "$(PID)" || (echo "Set PID=Pxxxxxx" && exit 2)
> @echo "etl-apple-daily: participant=$(PID)"
> @echo "Ensuring QC stage is up-to-date..."
> @$(MAKE) etl-apple-qc
> @echo "Running: PYTHONPATH=\"$$PWD\" \"$(VENV_PY)\" etl/apple_inapp_daily.py --participant \"$(PID)\" --normalized-dir \"$(NORMALIZED_DIR)\" --processed-dir \"$(PROCESSED_DIR)\" $$( [ \"$(DRY_RUN)\" = \"1\" ] && echo --dry-run || true )"
> @PYTHONPATH="$$PWD" "$(VENV_PY)" etl/apple_inapp_daily.py --participant "$(PID)" --normalized-dir "$(NORMALIZED_DIR)" --processed-dir "$(PROCESSED_DIR)" $$( [ "$(DRY_RUN)" = "1" ] && echo --dry-run || true )

.PHONY: etl-apple
etl-apple:
> @test -n "$(PID)" || (echo "Set PID=Pxxxxxx" && exit 2)
> @test -n "$(RUN_ID)" || (echo "Set RUN_ID (e.g. $(shell date -u +%Y%m%dT%H%M%SZ))" && exit 2)
> @echo "etl-apple: participant=$(PID) run=$(RUN_ID)"
> @echo "Running PII guard (non-blocking)..."
> @PYTHONPATH="$$PWD" "$(VENV_PY)" make_scripts/pii_guard.py --participant "$(PID)" || true
> @mkdir -p "$(ETL_DIR)/runs/$(RUN_ID)/logs"
> @echo "Running parse..."
> @PYTHONPATH="$$PWD" "$(VENV_PY)" etl/apple_inapp_parse.py --participant "$(PID)" --extracted-dir "$(EXTRACTED_DIR)" --normalized-dir "$(NORMALIZED_DIR)" --run-id "$(RUN_ID)" $$( [ "$(DRY_RUN)" = "1" ] && echo --dry-run || true ) 2>&1 | tee "$(ETL_DIR)/runs/$(RUN_ID)/logs/parse.log"
> @echo "Running qc..."
> @PYTHONPATH="$$PWD" "$(VENV_PY)" etl/apple_inapp_qc.py --participant "$(PID)" --normalized-dir "$(NORMALIZED_DIR)" --processed-dir "$(PROCESSED_DIR)" --run-id "$(RUN_ID)" $$( [ "$(DRY_RUN)" = "1" ] && echo --dry-run || true ) 2>&1 | tee "$(ETL_DIR)/runs/$(RUN_ID)/logs/qc.log"
> @echo "Running daily aggregation..."
> @PYTHONPATH="$$PWD" "$(VENV_PY)" etl/apple_inapp_daily.py --participant "$(PID)" --normalized-dir "$(NORMALIZED_DIR)" --processed-dir "$(PROCESSED_DIR)" --run-id "$(RUN_ID)" $$( [ "$(DRY_RUN)" = "1" ] && echo --dry-run || true ) 2>&1 | tee "$(ETL_DIR)/runs/$(RUN_ID)/logs/daily.log"
> @echo "Collecting run outputs and writing summary..."
> @PYTHONPATH="$$PWD" "$(VENV_PY)" make_scripts/apple/apple_etl_summary.py --participant "$(PID)" --run-id "$(RUN_ID)" $$( [ "$(DRY_RUN)" = "1" ] && echo --dry-run || true )
> @echo "Running provenance..."
> @$(MAKE) provenance PARTICIPANT=$(PID) RUN_ID=$(RUN_ID) || true
> @echo "ETL Apple In-App completed successfully for $(PID)."

.PHONY: idempotence-check
idempotence-check:
> @echo "Running idempotence-check for PID=$(PID) (full etl-apple chain)"
> @mkdir -p provenance
> @echo "Recording snapshot before..."
> @PYTHONPATH="$$PWD" "$(VENV_PY)" make_scripts/hash_snapshot.py --pid "$(PID)" --out provenance/hash_snapshot_before.json
> @echo "Running full etl-apple chain (parse->qc->daily)"
> @$(MAKE) etl-apple PID=$(PID) RUN_ID=checkrun DRY_RUN=0 || true
> @echo "Recording snapshot after..."
> @PYTHONPATH="$$PWD" "$(VENV_PY)" make_scripts/hash_snapshot.py --pid "$(PID)" --out provenance/hash_snapshot_after.json
> @PYTHONPATH="$$PWD" "$(VENV_PY)" make_scripts/compare_snapshots.py --before provenance/hash_snapshot_before.json --after provenance/hash_snapshot_after.json

.PHONY: atomicity-sim
atomicity-sim:
> @echo "Simulating partial write under $(NORMALIZED_DIR)/apple"
> @PYTHONPATH="$$PWD" "$(VENV_PY)" make_scripts/simulate_partial_write.py --dir "$(NORMALIZED_DIR)/apple"
> @echo "Now running etl-apple-parse to exercise cleanup/atomicity"
> @$(MAKE) etl-apple-parse
> @echo "Checking for leftover temp files..."
> @bash -c 'ls -1 "$(NORMALIZED_DIR)/apple" 2>/dev/null | grep "^\.tmp\." && echo "TEMP FILES LEFT" && exit 2 || echo "No leftover tmp files"'

# --- Comparação Zepp vs Apple + plots de uma vez --------------
# Ex.: make plots-all PID=P000001 SNAP=2025-09-29 POLICY=best_of_day
plots-all:
> @test -n "$(PID)"    || (echo "Set PID=Pxxxxxx" && exit 2)
> @test -n "$(SNAP)"   || (echo "Set SNAP=YYYY-MM-DD (ou YYYYMMDD)" && exit 2)
> @test -n "$(POLICY)" || (echo "Set POLICY={apple_first|zepp_first|best_of_day}" && exit 2)
> @echo "plots-all: compare+plot for PID=$(PID) SNAP=$(SNAP) POLICY=$(POLICY)"
> @echo "Running: PYTHONPATH=\"$$PWD\" \"$(VENV_PY)\" etl_tools/compare_zepp_apple.py --pid \"$(PID)\" --zepp-root \"data_etl/$(PID)/zepp_processed\" --apple-dir \"data_ai/$(PID)/snapshots/$(SNAP)\" --out-dir \"data_ai/$(PID)/snapshots/$(SNAP)/hybrid_join/$(POLICY)\" --sleep-policy \"$(POLICY)\""
> @PYTHONPATH="$$PWD" "$(VENV_PY)" etl_tools/compare_zepp_apple.py \
> 	--pid "$(PID)" \
> 	--zepp-root "data_etl/$(PID)/zepp_processed" \
> 	--apple-dir "data_ai/$(PID)/snapshots/$(SNAP)" \
> 	--out-dir "data_ai/$(PID)/snapshots/$(SNAP)/hybrid_join/$(POLICY)" \
> 	--sleep-policy "$(POLICY)"
> @echo "Running: PYTHONPATH=\"$$PWD\" \"$(VENV_PY)\" etl_tools/plot_sleep_compare.py --join \"data_ai/$(PID)/snapshots/$(SNAP)/hybrid_join/$(POLICY)/join_hybrid_daily.csv\" --outdir \"data_ai/$(PID)/snapshots/$(SNAP)/hybrid_join/$(POLICY)/plots\""
> @PYTHONPATH="$$PWD" "$(VENV_PY)" etl_tools/plot_sleep_compare.py \
> 	--join "data_ai/$(PID)/snapshots/$(SNAP)/hybrid_join/$(POLICY)/join_hybrid_daily.csv" \
> 	--outdir "data_ai/$(PID)/snapshots/$(SNAP)/hybrid_join/$(POLICY)/plots"

# --- Comparação + plots (modo "lite"): escreve/usa JOIN genérico sem subpasta da policy
# Uso:
#   make plots-all-lite PID=P000001 SNAP=2025-09-29            # usa POLICY=best_of_day por padrão
#   make plots-all-lite PID=P000001 SNAP=2025-09-29 POLICY=zepp_first
plots-all-lite:
> @test -n "$(PID)"  || (echo "Set PID=Pxxxxxx" && exit 2)
> @test -n "$(SNAP)" || (echo "Set SNAP=YYYY-MM-DD (ou YYYYMMDD)" && exit 2)
> @echo "plots-all-lite: compare+plot (lite) PID=$(PID) SNAP=$(SNAP)"
> @echo "Running: PYTHONPATH=\"$$PWD\" $(PY) make_scripts/plots_all_lite.py --pid \"$(PID)\" --snap \"$(SNAP)\" --policy \"$${POLICY:-best_of_day}\" $$( [ "$(DRY_RUN)" = "1" ] && echo --dry-run || true )"
> @PYTHONPATH="$$PWD" $(PY) make_scripts/plots_all_lite.py --pid "$(PID)" --snap "$(SNAP)" --policy "$${POLICY:-best_of_day}" $$( [ "$(DRY_RUN)" = "1" ] && echo --dry-run || true )

# --- Maintenance ---------------------------------------------
clean:
> @find . -name "__pycache__" -type d -prune -exec rm -rf {} + || true
> @find . -name "*.pyc" -delete || true
> @find . -name "*.log" -delete || true
> @echo "✔ cleaned caches/logs"

deepclean:
> @rm -rf $(OUT_DIR) $(IOS_DIR)/decrypted_output || true
> @echo "⚠ removed decrypted outputs (PII)."


.PHONY: diag-a8
diag-a8:
> @echo "Running A8 processed-daily diagnostics"
> @PYTHONPATH="$$PWD" $(PY) make_scripts/diag/a8_diag_processed.py --snapshot "$(ETL_SNAP_DIR)" --out "reports/diag/a8_diag_processed.md"


# Fallback builder when processed/*_daily.csv are empty.
.PHONY: fuse-from-normalized
fuse-from-normalized:
> @PYTHONPATH="$$PWD" $(PY) make_scripts/a8_fuse_from_normalized.py --snapshot "$(ETL_SNAP_DIR)" || exit $$?

promote-current:
> @rm -rf decrypted_output_old 2>/dev/null || true
> @test -n "$(SRC)" || (echo "Use: make promote-current SRC=decrypted_output_YYYYMMDD" && exit 1)
> @[ -d "$(SRC)" ] || (echo "SRC not found: $(SRC)" && exit 1)
> @[ -d decrypted_output ] && mv decrypted_output decrypted_output_old || true
> cp -r "$(SRC)" decrypted_output
> @echo "Promoted $(SRC) -> decrypted_output"

prune-decrypts:
> @ls -1d decrypted_output_* 2>/dev/null | sort -r | tail -n +4 | xargs -r rm -rf
> @echo "Pruned older decrypt folders (kept 3 newest)."


# ----------------------------------------------------------------------
# Makefile runners for A6/A7/A8 (M1)
# ----------------------------------------------------------------------
.PHONY: run-a6-apple
run-a6-apple:
> @test -n "$(PARTICIPANT)" || (echo "Set PARTICIPANT=<PID>" && exit 2)
> @test -n "$(SNAPSHOT_DATE)" || (echo "Set SNAPSHOT_DATE=YYYY-MM-DD" && exit 2)
> @echo "run-a6-apple: participant=$(PARTICIPANT) snapshot=$(SNAPSHOT_DATE) dry_run=$(DRY_RUN)"
> @PYTHONPATH="$$PWD" "$(VENV_PY)" make_scripts/apple/run_a6_apple.py --participant "$(PARTICIPANT)" --snapshot "$(SNAPSHOT_DATE)" $$( [ "$(DRY_RUN)" = "1" ] && echo --dry-run || true )

.PHONY: run-a7-apple
run-a7-apple:
> @test -n "$(PARTICIPANT)" || (echo "Set PARTICIPANT=<PID>" && exit 2)
> @test -n "$(SNAPSHOT_DATE)" || (echo "Set SNAPSHOT_DATE=YYYY-MM-DD" && exit 2)
> @echo "run-a7-apple: participant=$(PARTICIPANT) snapshot=$(SNAPSHOT_DATE) dry_run=$(DRY_RUN)"
> @PYTHONPATH="$$PWD" "$(VENV_PY)" make_scripts/apple/run_a7_apple.py --participant "$(PARTICIPANT)" --snapshot "$(SNAPSHOT_DATE)" $$( [ "$(DRY_RUN)" = "1" ] && echo --dry-run || true )

.PHONY: run-a8-labels
run-a8-labels:
> @test -n "$(PARTICIPANT)" || (echo "Set PARTICIPANT=<PID>" && exit 2)
> @test -n "$(SNAPSHOT_DATE)" || (echo "Set SNAPSHOT_DATE=YYYY-MM-DD" && exit 2)
> @echo "run-a8-labels: participant=$(PARTICIPANT) snapshot=$(SNAPSHOT_DATE) dry_run=$(DRY_RUN)"
> @PYTHONPATH="$$PWD" "$(VENV_PY)" make_scripts/apple/run_a8_labels.py --participant "$(PARTICIPANT)" --snapshot "$(SNAPSHOT_DATE)" $$( [ "$(DRY_RUN)" = "1" ] && echo --dry-run || true )

# --- Raw data helpers (consolidated layout) -----------------
.PHONY: init-raw move-raw clean-raw

init-raw:
> 	@echo "Initializing RAW_DIR: $(RAW_DIR)"
> 	@mkdir -p "$(RAW_DIR)/apple" "$(RAW_DIR)/zepp" || true
> 	@echo "Created $(RAW_DIR)/apple and $(RAW_DIR)/zepp"

# Move a raw ZIP into the RAW_DIR. Usage: make move-raw RAW_SRC=/path/to/apple_health_export_YYYY-MM-DD.zip
move-raw:
> 	@if [ -z "$(RAW_SRC)" ]; then echo "Usage: make move-raw RAW_SRC=/path/to/file.zip" && exit 2; fi; \
> 	mkdir -p "$(RAW_DIR)"; \
> 	cp "$(RAW_SRC)" "$(RAW_DIR)/"; \
> 	if [ "x$(BACKUP)" = "x1" ]; then echo "Backup kept: $(RAW_SRC)"; else rm -f "$(RAW_SRC)"; fi; \
> 	echo "Moved $(RAW_SRC) -> $(RAW_DIR)/"

# NOTE: raw cleaning is implemented via the canonical Python helper below
# (keeps a single authoritative implementation to avoid duplicate Make
# recipes and ensure consistent safety checks). See the `clean_data_make.py`
# helper for flags/confirmation behavior.


.PHONY: print-paths init-data init-data-layout test-clean-layout
print-paths:
>	@echo DATA_BASE=$(DATA_BASE)
>	@echo PARTICIPANT=$(PARTICIPANT)
>	@echo RAW_DIR=$(RAW_DIR)
>	@echo ETL_DIR=$(ETL_DIR)
>	@echo ETL_SNAP_DIR=$(ETL_SNAP_DIR)
>	@echo EXTRACTED_DIR=$(EXTRACTED_DIR)
>	@echo NORMALIZED_DIR=$(NORMALIZED_DIR)
>	@echo PROCESSED_DIR=$(PROCESSED_DIR)
>	@echo JOINED_DIR=$(JOINED_DIR)
>	@echo AI_DIR=$(AI_DIR)
>	@echo AI_SNAP_DIR=$(AI_SNAP_DIR)
>	@echo AI_INPUT_DIR=$(AI_INPUT_DIR)
>	@echo SNAPSHOT_DATE=$(SNAPSHOT_DATE)
>	@echo IOS_BACKUP_DIR=$(IOS_BACKUP_DIR)
>	@echo IOS_MANIFEST_DB=$(IOS_MANIFEST_DB)
>	@echo PY=$(PY)

# Existing init-data retained for compatibility with older workflows
init-data:
>	@mkdir -p "$(ETL_DIR)/exports/apple" "$(ETL_DIR)/exports/zepp" "$(ETL_DIR)/runs" "$(ETL_DIR)/logs" "$(AI_DIR)/snapshots"
>	@echo "Initialized data layout for $(PARTICIPANT)"

# New canonical initializer for the data taxonomy requested by UX
init-data-layout:
>	@mkdir -p "$(RAW_DIR)/apple" "$(RAW_DIR)/zepp" \
>		"$(EXTRACTED_DIR)" "$(NORMALIZED_DIR)" "$(PROCESSED_DIR)" "$(JOINED_DIR)" "$(AI_DIR)/snapshots"
>	@echo "Initialized new data taxonomy for $(PARTICIPANT)."

# Cleaning helpers split by data stage. These call the Python helper which
# implements safe defaults (dry-run, require explicit flag for raw).
clean-extracted:
>	@echo "Running clean for extracted (dry-run unless DRY_RUN=0)"
>	@$(VENV_PY) make_scripts/clean_data_make.py --stage extracted

clean-normalized:
>	@echo "Running clean for normalized (dry-run unless DRY_RUN=0)"
>	@$(VENV_PY) make_scripts/clean_data_make.py --stage normalized

clean-processed:
>	@echo "Running clean for processed (dry-run unless DRY_RUN=0)"
>	@$(VENV_PY) make_scripts/clean_data_make.py --stage processed

clean-joined:
>	@echo "Running clean for joined (dry-run unless DRY_RUN=0)"
>	@$(VENV_PY) make_scripts/clean_data_make.py --stage joined

clean-ai-snapshot:
>	@echo "Running clean for ai_snapshot (dry-run unless DRY_RUN=0)"
>	@$(VENV_PY) make_scripts/clean_data_make.py --stage ai_snapshot

# Dangerous: cleans raw files. Requires explicit acknowledgment via
# --i-understand-raw and should generally be avoided. Use with care.
clean-raw:
>	@echo "WARNING: This will delete raw files. Pass --i-understand-raw to confirm."
>	@$(VENV_PY) make_scripts/clean_data_make.py --stage raw --i-understand-raw

# Lint target to detect any remaining usage of the legacy `exports/` path.
.PHONY: lint-deprecated-exports
lint-deprecated-exports:
>	@$(VENV_PY) make_scripts/lint_deprecated_exports.py

# Sanity test target: small smoke test for layout and DRY_RUN
test-clean-layout:
>	@$(MAKE) print-paths
>	@$(MAKE) init-data
>	@echo "Checking exports dirs and counts..."
>	@ETL_DIR="$(ETL_DIR)" python make_scripts/check_layout.py || (echo 'layout check failed' && exit 2)
>	@DRY_RUN=1 NON_INTERACTIVE=1 $(MAKE) clean-data || (echo 'clean-data failed' && exit 2)
>	@echo "OK: layout sane"
.PHONY: test-etl-one-skip
test-etl-one-skip:
> @echo "Running test-etl-one-skip (creates temporary data_etl/<PID>/<SNAP> without export.xml)"
> @PYTHONPATH="$$PWD" $(PY) make_scripts/test_etl_one_skip.py || (echo "test-etl-one-skip: FAILED" && exit 2)
> @echo "test-etl-one-skip: completed"
.PHONY: venv clean-venv check-paths
venv:
> @echo "venv: creating .venv using $(PY)"
> @echo "Running: $(PY) make_scripts/venv_create.py --python $(PY) $(if $(DEV),--dev) $(if $(KAGGLE),--kaggle) $(if $(WIN),--win)"
> @$(PY) make_scripts/venv_create.py --python $(PY) $(if $(DEV),--dev) $(if $(KAGGLE),--kaggle) $(if $(WIN),--win) $(if $(NO_INSTALL),--no-install)

clean-venv:
>	@rm -rf .venv || true
>	@find . -name "__pycache__" -type d -prune -exec rm -rf {} + || true
>	@echo "Removed .venv and __pycache__ folders"

check-paths:
>	@echo DATA_ETL=$(ETL_DIR)
>	@echo DATA_AI=$(AI_DIR)
>	@echo PARTICIPANT=$(PARTICIPANT)

.PHONY: check-reqs pip-freeze
# Verify Python and key packages inside the virtualenv; exit non-zero on failure
check-reqs:
>	@bash -c 'if [ ! -x "$(VENV_PY)" ]; then echo ".venv python not found; falling back to $(PY_ABS)"; fi; echo "Using $(VENV_PY)"; echo "Python version:"; "$(VENV_PY)" -c "import sys; print(sys.version)"; echo "Checking imports: numpy, pandas, matplotlib"; "$(VENV_PY)" -c "import numpy" || { echo "FAILED numpy"; exit 2; }; "$(VENV_PY)" -c "import pandas" || { echo "FAILED pandas"; exit 2; }; "$(VENV_PY)" -c "import matplotlib" || { echo "FAILED matplotlib"; exit 2; }; echo "All imports OK"'

# Record pip freeze output using the venv python; write file to provenance and print path
pip-freeze:
>	@bash -c 'if [ ! -x "$(VENV_PY)" ]; then echo ".venv python not found; falling back to $(PY_ABS)"; fi; echo "Using $(VENV_PY)"; ts=$$(date -u +"%Y%m%dT%H%M%SZ"); mkdir -p provenance; fn=provenance/pip_freeze_$$ts.txt; echo $$fn; "$(VENV_PY)" -m pip freeze > $$fn'

.PHONY: print-venv venv-run
print-venv:
>	@echo VENV_PY=$(VENV_PY)
>	@if [ "$(VENV_PY)" = "$(PY_ABS)" ]; then echo "Warning: .venv python not found; falling back to $(PY_ABS)"; fi
>	@"$(VENV_PY)" --version || true

# Run arbitrary args via the venv python. Usage: make venv-run RUN_ARGS="-c 'import sys; print(sys.executable)'"
venv-run:
>	@if [ -z "$(RUN_ARGS)" ]; then echo 'Usage: make venv-run RUN_ARGS="..."'; exit 2; fi; "$(VENV_PY)" $(RUN_ARGS)


.PHONY: venv-shell print-shell venv-shell-bash venv-shell-cmd venv-shell-pwsh deactivate-note
venv-shell:
> @echo "Launching an interactive shell with .venv activated (this does NOT persist across Make recipes)."
> @echo "Detecting shell environment..."
> @echo "MSYSTEM=$${MSYSTEM:-}  ComSpec=$${ComSpec:-}  SHELL=$${SHELL:-}"
> @if [ -n "$${MSYSTEM:-}" ]; then \
>   echo "Git Bash detected; launching make_scripts/activate_venv.sh (bash)."; \
>   bash --noprofile --norc -i make_scripts/activate_venv.sh; \
> elif [ -n "$${PSModulePath:-}" ] || echo "$${ComSpec:-}" | grep -qi "powershell.exe"; then \
>   echo "PowerShell detected; activating via Activate.ps1."; \
>   powershell -NoLogo -NoExit -ExecutionPolicy Bypass -Command ". .venv\\Scripts\\Activate.ps1"; \
> else \
>   echo "Defaulting to cmd.exe; activating via activate.bat."; \
>   cmd.exe /K ".venv\\Scripts\\activate.bat"; \
> fi

print-shell:
> @echo "print-shell: MSYSTEM=$${MSYSTEM:-} ComSpec=$${ComSpec:-} PSModulePath=$${PSModulePath:-} SHELL=$${SHELL:-}"
> @if [ -n "$${MSYSTEM:-}" ]; then echo "Guess: git-bash"; elif echo "$${ComSpec:-}" | grep -qi "powershell.exe" || [ -n "$${PSModulePath:-}" ]; then echo "Guess: powershell"; else echo "Guess: cmd"; fi

venv-shell-bash:
> @if [ ! -f ".venv" ] && [ ! -d ".venv" ]; then echo "ERROR: .venv not found; run 'make venv' first." && exit 2; fi
> @if [ -z "$${MSYSTEM:-}" ]; then echo "ERROR: MSYS/MSYSTEM not detected; run from Git Bash/MSYS to use venv-shell-bash." && exit 2; fi
> @echo "Git Bash detected; launching activate_venv.sh"
> @bash --noprofile --norc -i make_scripts/activate_venv.sh

venv-shell-pwsh:
> @if [ ! -f ".venv" ] && [ ! -d ".venv" ]; then echo "ERROR: .venv not found; run 'make venv' first." && exit 2; fi
> @echo "Launching PowerShell with .venv Activate.ps1"
> @powershell -NoLogo -NoExit -ExecutionPolicy Bypass -Command ". .venv\\Scripts\\Activate.ps1"

venv-shell-cmd:
> @if [ ! -f ".venv" ] && [ ! -d ".venv" ]; then echo "ERROR: .venv not found; run 'make venv' first." && exit 2; fi
> @echo "Launching cmd.exe with .venv\\Scripts\\activate.bat"
> @cmd.exe /K ".venv\\Scripts\\activate.bat"

.PHONY: help-shell print-venv-status
help-shell:
> @echo "make venv-shell        # open shell with venv in the current shell (bash/cmd/pwsh)"
> @echo "make venv-shell-bash   # force Git Bash + venv"
> @echo "make venv-shell-pwsh   # force PowerShell + venv"
> @echo "make venv-shell-cmd    # force cmd.exe + venv"
> @echo "Layout: make help-layout (intake/migrate/clean/provenance targets)"

print-venv-status:
> @echo "Using VENV_PY=$(VENV_PY)";
> @"$(VENV_PY)" -c "import sys,sysconfig,site,os; print('VENV:', os.environ.get('VIRTUAL_ENV')); print('EXE:', sys.executable); print('SITE:', site.getsitepackages())"

deactivate-note:
> @echo "To deactivate the virtualenv: run 'deactivate' in POSIX shells. On Windows, close the cmd/powershell window or type 'exit'."

.PHONY: help-venv
help-venv:
>	@echo "make venv                 (base)"
>	@echo "make venv DEV=1           (dev)"
>	@echo "make venv KAGGLE=1       (base + Kaggle constraints)"
>	@echo "make venv WIN=1          (base + Windows constraints)"
>	@echo "make pip-freeze          (write provenance/pip_freeze_<UTC>.txt)"

.PHONY: print-version version-guard
print-version:
> 	@echo "LATEST_VERSION=$(LATEST_VERSION)"
> 	@echo "LATEST_TAG=$(LATEST_TAG_RAW)"
> 	@echo "COMMITS_SINCE_LATEST_TAG=$(COMMITS_SINCE_LATEST_TAG)"
> 	@echo "LATEST_COMMIT_ID=$(LATEST_COMMIT_ID)"
> 	@echo "NEXT_VERSION=$(NEXT_VERSION)"
> 	@echo "NEXT_TAG=$(NEXT_TAG)"

version-guard:
> 	@echo "Running version guard for TAG=$(TAG) (allow dirty: $${ALLOW_DIRTY:-false})"
> 	@$(VENV_PY) make_scripts/release/version_guard.py --tag "$(TAG)" $$( [ "$${ALLOW_DIRTY:-0}" = "1" ] && echo --allow-dirty || true )

.PHONY: help-release
help-release:
> 	@echo "make release-notes       # build docs/release_notes/release_notes_$(TAG).md using $(VENV_PY)"
> 	@echo "make version-guard       # run pre-release checks (clean tree, no existing tag)"
> 	@echo "make changelog           # update CHANGELOG.md from docs/release_notes/release_notes_$(TAG).md"
> 	@echo "make release-draft RELEASE_DRY_RUN=1  # dry-run the release flow (no side effects)"
> 	@echo "make release-draft        - Prepare draft: notes + changelog + assets (no tag push)"
> 	@echo "make release-assets       - Collect provenance/assets and build manifest"
> 	@echo "make provenance           - Generate provenance files (csv/md/pip_freeze)"
> 	@echo "make print-version        - Show LATEST/NEXT tag/version info"
> 	@echo "make publish-release      - (Final) Tag and create GitHub Release with assets"
> 	@echo "  Flags: RELEASE_DRY_RUN=1 (no changes), RELEASE_PUSH=1 (push tag + GH), RELEASE_DRAFT=1 (GH draft)"

# --- Publish config ---------------------------------------------------
NOTES_FILE := docs/release_notes/release_notes_$(NEXT_TAG).md
ASSET_DIR  := dist/assets/$(NEXT_TAG)
RELEASE_DRAFT ?= 0
RELEASE_PUSH  ?= 0

.PHONY: release-draft release-publish release-assets

<<<<<<< Updated upstream:Makefile
# Orchestrated release-draft (idempotent, single version-guard, auto-commit generated files)
release-draft:
> 	@echo "Preparing release draft for $(NEXT_TAG)"
> 	@# Run version guard once at start. Allow dirty only when ALLOW_DIRTY=1
> 	@echo "Running version guard for TAG=$(NEXT_TAG) (allow dirty: $${ALLOW_DIRTY:-false})"
> 	@$(VENV_PY) make_scripts/version_guard.py --tag "$(NEXT_TAG)" $$( [ "$${ALLOW_DIRTY:-0}" = "1" ] && echo --allow-dirty || true ) --remote-check --remote origin
> 	@# Render release notes and update changelog with explicit NEXT_VERSION/NEXT_TAG
> 	@$(MAKE) release-notes VERSION="$(NEXT_VERSION)" TAG="$(NEXT_TAG)"
> 	@$(MAKE) changelog-update VERSION="$(NEXT_VERSION)" TAG="$(NEXT_TAG)"
> 	@# Stage and auto-commit generated changelog + release notes if they changed
> 	@git add CHANGELOG.md docs/release_notes/release_notes_$(NEXT_TAG).md 2>/dev/null || true
> 	@if ! git diff --staged --quiet --exit-code; then \
> 	  git commit -m "release($(NEXT_TAG)): notes + changelog"; \
=======
release-draft: version-guard release-notes changelog-update release-assets
> 	@echo "Preparing release draft for $(TAG)"
> 	@# run version guard (do not swallow failures). Allow dirty only when ALLOW_DIRTY=1
> 	@$(VENV_PY) make_scripts/release/version_guard.py --tag "$(TAG)" $$( [ "$${ALLOW_DIRTY:-0}" = "1" ] && echo --allow-dirty || true ) --remote-check --remote origin
> 	@# After changelog, stage CHANGELOG and release notes and commit if changed
> 	@git add CHANGELOG.md docs/release_notes/release_notes_$(TAG).md 2>/dev/null || true
> 	@if ! git diff --cached --quiet --exit-code; then \
> 	  git commit -m "docs(release): update CHANGELOG for $(TAG)" || true; \
>>>>>>> Stashed changes:makefile
> 	else \
> 	  echo "No changelog/release-notes changes to commit"; \
> 	fi
> 	@# Collect release assets after committing generated docs
> 	@$(MAKE) release-assets RELEASE_DRY_RUN=$(RELEASE_DRY_RUN) TAG="$(NEXT_TAG)"
> 	@echo "Draft prepared for $(NEXT_TAG). No tag pushed."

release-assets:
> 	@echo "Collecting release assets into dist/assets/$(TAG)/"
> 	@mkdir -p "dist/assets/$(TAG)"
> 	@MISSING=0; \
> 	MISSING_LIST=""; \
> 	# mandatory provenance files
> 	if [ ! -f "provenance/etl_provenance_report.csv" ]; then MISSING=1; MISSING_LIST="$$MISSING_LIST provenance/etl_provenance_report.csv"; fi; \
> 	if [ ! -f "provenance/data_audit_summary.md" ]; then MISSING=1; MISSING_LIST="$$MISSING_LIST provenance/data_audit_summary.md"; fi; \
>	LF=$$(ls -1t provenance/pip_freeze_*.txt 2>/dev/null | head -n1 || true); \
>	if [ -z "$$LF" ]; then \
>	  echo "provenance: pip_freeze not found, attempting to generate via make provenance..."; \
>	  $(MAKE) provenance || true; \
>	  LF=$$(ls -1t provenance/pip_freeze_*.txt 2>/dev/null | head -n1 || true); \
>	fi; \
>	if [ -z "$$LF" ]; then \
>	  MISSING=1; \
>	  MISSING_LIST="$$MISSING_LIST provenance/pip_freeze_*.txt"; \
>	fi; \
> 	# copy mandatory provenance artifacts
> 	cp "provenance/etl_provenance_report.csv" "dist/assets/$(TAG)/"; \
> 	cp "provenance/data_audit_summary.md" "dist/assets/$(TAG)/"; \
> 	cp "$$LF" "dist/assets/$(TAG)/"; \
> 	COPIED=1; \
> 	# copy any intake logs for participant (optional)
> 	for f in $$(ls -1 data/etl/$(PARTICIPANT)/runs/*/logs/*.json 2>/dev/null || true); do cp "$$f" "dist/assets/$(TAG)/"; done; \
> 	if [ $$COPIED -eq 0 ]; then \
> 	  echo "No assets found to collect; dist/assets/$(TAG) will be empty."; \
> 	else \
> 	  echo "Building manifest for dist/assets/$(TAG)"; \
> 	  "$(VENV_PY)" make_scripts/utils/build_asset_manifest.py "dist/assets/$(TAG)" --out "dist/assets/$(TAG)/manifest.json"; \
> 	fi; \
> 	echo "Assets in dist/assets/$(TAG):"; ls -la "dist/assets/$(TAG)" || true


.PHONY: clean-assets
clean-assets:
> 	@echo "🧹 Cleaning old release-assets..."
> 	@if [ "x$${CLEAN_ASSETS_DRY_RUN:-0}" = "x1" ]; then \
> 		echo "DRY RUN: directories that would be removed:"; \
> 		find dist/assets -maxdepth 1 -type d ! -name 'v$(VERSION)' -print || true; \
> 		echo "(no deletion performed)"; \
> 	else \
> 		find dist/assets -maxdepth 1 -type d ! -name 'v$(VERSION)' -exec rm -rf {} +; \
> 		find dist/assets -type d -empty -delete 2>/dev/null || true; \
> 		ls -la dist/assets || true; \
> 	fi

release-publish: version-guard release-notes changelog-update release-assets
> 	@echo "Publishing release $(TAG) to origin/GitHub"
> 	@# run version guard (no allow-dirty for real publish)
> 	@$(VENV_PY) make_scripts/release/version_guard.py --tag "$(TAG)" --remote-check --remote origin
> 	@# ensure annotated tag exists locally and push it
> 	@if git rev-parse --verify "$(TAG)" >/dev/null 2>&1; then \
> 	  echo "Tag $(TAG) already exists locally"; \
> 	else \
> 	  git tag -a "$(TAG)" -m "$(RELEASE_TITLE)"; \
> 	fi; \
> 	git push origin "$(TAG)"
> 	@# sanitize title and release notes to avoid mojibake (e.g. â€” -> —)
> 	@mkdir -p "dist/assets/$(TAG)"
> 	@SANITIZED_NOTES="dist/assets/$(TAG)/release_notes_sanitized.md"; \
> 	$(VENV_PY) make_scripts/sanitize_release_text.py --infile "docs/release_notes/release_notes_$(TAG).md" --outfile "$$SANITIZED_NOTES"
> 	@CLEAN_TITLE=$$($(VENV_PY) make_scripts/sanitize_release_text.py --text "$(RELEASE_TITLE)"); \
> 	ASSETS=$$($(VENV_PY) make_scripts/utils/list_manifest_assets.py --manifest "dist/assets/$(TAG)/manifest.json") || true; \
> 	@echo "Creating GitHub release for $(TAG) with assets: $$ASSETS"; \
> 	gh release create "$(TAG)" $$ASSETS --title "$$CLEAN_TITLE" --notes-file "$$SANITIZED_NOTES"; \
> 	# create or update a 'latest' tag pointing at this tag and push it (force update)
> 	git tag -f latest "$(TAG)"; git push -f origin latest
> 	@echo "Published $(TAG) and updated 'latest' tag on origin"

suggest-version:
> 	@$(VENV_PY) make_scripts/suggested_version.py --tag "$(TAG)"
> 	@echo "make check-reqs          (verify numpy, pandas, matplotlib inside .venv)"
> 	@echo "Layout: make help-layout  (intake/migrate/clean/provenance targets)"

.PHONY: help-data
help-data:
>	@echo "Data targets — usage examples"
>	@echo ""
>	@echo "ENV-only style (export env vars when calling make):"
>	@echo "  DRY_RUN=1 NON_INTERACTIVE=1 make clean-data"
>	@echo "  make freeze-model-input PID=P000001 SNAP=2025-09-29 POLICY=best_of_day"
>	@echo "  make env-versions"
>	@echo "  make weekly-report PID=P000001 SNAP=2025-09-29 POLICY=best_of_day"
>	@echo ""
>	@echo "Flags passthrough style (use per-target *_ARGS to pass CLI flags):"
>	@echo "  make clean-data CLEAN_ARGS='--dry-run --non-interactive'"
>	@echo "  make freeze-model-input INTAKE_ARGS='--dry-run --non-interactive' PID=P000001 SNAP=..."
>	@echo "  make env-versions ENV_ARGS='--some-flag'"
>	@echo "  make weekly-report REPORT_ARGS='--verbose' PID=P000001 SNAP=..."
>	@echo "  make pip-freeze PROV_ARGS='--output provenance/pip.txt'"
>    @echo ""
>    @echo "Note: 'make etl-one PID=... SNAP=...' will skip (exit 0) if no export.xml exists for the given PID/SNAPSHOT."
>    @echo "Run 'make test-etl-one-skip' to verify this behavior (creates temporary data_etl; cleans up afterwards)."
>    @echo ""
>    @echo "Canonical path vars: PARTICIPANT=<PID> SNAPSHOT_DATE=<YYYY-MM-DD>"
>    @echo "  ETL_SNAP_DIR=./data/etl/<PID>/snapshots/<SNAPSHOT_DATE>"
>    @echo "  EXTRACTED_DIR=$(EXTRACTED_DIR)  (writes under snapshot)"
>    @echo "  AI_INPUT_DIR=$(AI_INPUT_DIR)"
>    @echo "IOS_BACKUP_DIR: path to decrypted iTunes backup root (read-only)"
>    @echo ""
>    @echo "Intake helper example:"
>    @echo "  make intake-zip INTAKE_ARGS='--source apple --zip-path data/raw/P000001/apple/apple_health_export_2025-10-22.zip --stage'"

.PHONY: help
help:
>	@echo "Practicum2 Makefile — high-level help"
>	@echo ""
>	@$(MAKE) help-venv
>	@echo ""
>	@$(MAKE) help-data

.PHONY: lint-make
lint-make:
>	@python make_scripts/lint_make.py

.PHONY: lint-layout
lint-layout:
>	@echo "Running layout linter (checks for legacy exports/ in make_scripts/ and stage writes)"
>	@$(VENV_PY) make_scripts/lint_layout.py

.PHONY: migrate-layout help-migrate
migrate-layout:
>	@echo "Running migrate-layout for PARTICIPANT=$(PARTICIPANT) (dry-run by setting MIGRATE_ARGS=--dry-run)"
>	@PARTICIPANT="$(PARTICIPANT)" ETL_DIR="$(DATA_BASE)/etl/$(PARTICIPANT)" \
>	RAW_DIR="$(DATA_BASE)/raw/$(PARTICIPANT)" \
>	$(VENV_PY) make_scripts/migrate_layout.py $(MIGRATE_ARGS)

help-migrate:
>	@echo "make migrate-layout MIGRATE_ARGS=\"--dry-run\"  # show what would move"
>	@echo "make migrate-layout                         # perform moves (careful)"
>	@echo "Audit JSON will be written to provenance/migrate_layout_<UTC_COMPACT>.json"

.PHONY: intake-zip
intake-zip:
>	@echo "Intake ZIP: copy to RAW_DIR and optionally stage minimal extracted files into EXTRACTED_DIR"
>	@RAW_DIR="$(RAW_DIR)" EXTRACTED_DIR="$(EXTRACTED_DIR)" PARTICIPANT="$(PARTICIPANT)" \
>	$(VENV_PY) make_scripts/intake_zip.py $(INTAKE_ARGS)

.PHONY: provenance
provenance:
> @echo "Running provenance audit for participant=$(PARTICIPANT)"
> @# Ensure pip freeze exists before running full provenance audit (idempotent)
> @$(VENV_PY) make_scripts/release/generate_pip_freeze.py --out-dir provenance || true
> @NORMALIZED_DIR="$(NORMALIZED_DIR)" PROCESSED_DIR="$(PROCESSED_DIR)" JOINED_DIR="$(JOINED_DIR)" AI_SNAPSHOT_DIR="$(AI_SNAPSHOT_DIR)" PYTHONPATH="$$PWD" "$(VENV_PY)" make_scripts/provenance_audit.py $(if $(PARTICIPANT),--participant $(PARTICIPANT),) $(if $(SNAPSHOT),--snapshot $(SNAPSHOT),) $(if $(DRY_RUN),--dry-run,)
> @echo "Summary:"
> @ls -1 provenance || true

.PHONY: provenance-snap
provenance-snap:
> @test -n "$(PID)" || (echo "Set PID=Pxxxxxx" && exit 2)
> @test -n "$(SNAPSHOT)" || (echo "Set SNAPSHOT=YYYY-MM-DD" && exit 2)
> @echo "Running participant-scoped provenance for PID=$(PID) snapshot=$(SNAPSHOT)"
> @PARTICIPANT="$(PID)" SNAPSHOT="$(SNAPSHOT)" $(MAKE) provenance

.PHONY: provenance-pip-freeze
provenance-pip-freeze:
> @echo "Ensuring provenance pip freeze exists (provenance/)"
> @$(VENV_PY) make_scripts/release/generate_pip_freeze.py --out-dir provenance || true

.PHONY: lint-args
lint-args:
> @echo "Checking for inline venv activation and heredoc tokens..."
> @matches=$$(grep -n -E "\.venv/Scripts/activate|\. \.venv/bin/activate" Makefile | grep -v "grep -n -E" || true); \
> if [ -n "$$matches" ]; then echo "ERROR: Found legacy inline venv activation in Makefile. Move logic to make_scripts/ and call $(VENV_PY) instead."; echo "Matches:"; echo "$$matches"; exit 2; fi
> @python make_scripts/lint_make.py || (echo "ERROR: heredoc tokens found by lint_make.py" && exit 2)
> @echo "lint-args: OK"

# Safe cleanup of snapshot outputs and provenance outputs for PID P000001
.PHONY: clean-data
clean-data:
> 	@echo "Cleaning under ETL_DIR=$(ETL_DIR) and AI_DIR=$(AI_DIR) (safe subset: extracted, normalized, processed, joined)"
>	@echo "Tip: set DRY_RUN=1 to preview what will be removed"
>	@echo "Running per-stage cleanup"
>	@ETL_DIR="$(ETL_DIR)" AI_DIR="$(AI_DIR)" SNAPSHOT_DATE="$(SNAPSHOT_DATE)" DRY_RUN="$(DRY_RUN)" NON_INTERACTIVE="$(NON_INTERACTIVE)" $(VENV_PY) make_scripts/clean_data_make.py --stage extracted $$( [ "$(PRUNE)" = "true" ] && echo --prune-extras || true ) $$( [ -n "$(PRUNE_LIST)" ] && echo --prune-list "$(PRUNE_LIST)" || true ) $(CLEAN_ARGS)
>	@ETL_DIR="$(ETL_DIR)" AI_DIR="$(AI_DIR)" SNAPSHOT_DATE="$(SNAPSHOT_DATE)" DRY_RUN="$(DRY_RUN)" NON_INTERACTIVE="$(NON_INTERACTIVE)" $(VENV_PY) make_scripts/clean_data_make.py --stage normalized $(CLEAN_ARGS)
>	@ETL_DIR="$(ETL_DIR)" AI_DIR="$(AI_DIR)" SNAPSHOT_DATE="$(SNAPSHOT_DATE)" DRY_RUN="$(DRY_RUN)" NON_INTERACTIVE="$(NON_INTERACTIVE)" $(VENV_PY) make_scripts/clean_data_make.py --stage processed $(CLEAN_ARGS)
>	@ETL_DIR="$(ETL_DIR)" AI_DIR="$(AI_DIR)" SNAPSHOT_DATE="$(SNAPSHOT_DATE)" DRY_RUN="$(DRY_RUN)" NON_INTERACTIVE="$(NON_INTERACTIVE)" $(VENV_PY) make_scripts/clean_data_make.py --stage joined $(CLEAN_ARGS)

.PHONY: env-versions
env-versions:
> 	@echo "env-versions: show environment python/tool versions"
>	@echo "Running: PYTHONPATH=\"$$PWD\" $(PY) make_scripts/env_versions.py $(ENV_ARGS)"
>	@PYTHONPATH="$$PWD" $(PY) make_scripts/env_versions.py $(ENV_ARGS)

.PHONY: zepp-apple-compare

# Ex.: make zepp-apple-compare PID=P000001 SNAP=2024-06-30 POLICY=best_of_day
zepp-apple-compare:
# converted to recipe prefix style
> @test -n "$(POLICY)" || (echo "Set POLICY={apple_first|zepp_first|best_of_day}" && exit 2)
> @echo "zepp-apple-compare: PID=$(PID) SNAP=$(SNAP) POLICY=$(POLICY)"
> @echo "Running: PYTHONPATH=\"$$PWD\" \"$(VENV_PY)\" etl_tools/compare_zepp_apple.py --pid \"$(PID)\" --zepp-root \"data_etl/$(PID)/zepp_processed\" --apple-dir \"data_ai/$(PID)/snapshots/$(SNAP)\" --out-dir \"data_ai/$(PID)/snapshots/$(SNAP)/hybrid_join\" --sleep-policy \"$(POLICY)\""
> @PYTHONPATH="$$PWD" "$(VENV_PY)" etl_tools/compare_zepp_apple.py \
> 	--pid "$(PID)" \
> 	--zepp-root "data_etl/$(PID)/zepp_processed" \
> 	--apple-dir "data_ai/$(PID)/snapshots/$(SNAP)" \
> 	--out-dir "data_ai/$(PID)/snapshots/$(SNAP)/hybrid_join" \
> 	--sleep-policy "$(POLICY)"

# Ex.: make freeze-model-input PID=P000001 SNAP=2024-06-30 POLICY=best_of_day
freeze-model-input:
> @echo "Freeze model input: participant=$(PID) snapshot=$(SNAP) policy=$(POLICY)"
> 	@echo "Running: PYTHONPATH=\"$$PWD\" $(PY) make_scripts/freeze_model_input.py --participant \"$(PID)\" --snapshot \"$(SNAP)\" --policy \"$(POLICY)\" $(INTAKE_ARGS)"
> 	@PYTHONPATH="$$PWD" $(PY) make_scripts/freeze_model_input.py --participant "$(PID)" --snapshot "$(SNAP)" --policy "$(POLICY)" $(INTAKE_ARGS)

# Ex.: make plot-sleep PID=P000001 SNAP=2024-06-30 POLICY=best_of_day
plot-sleep:
# converted to recipe prefix style
> @test -n "$(POLICY)" || (echo "Set POLICY={apple_first|zepp_first|best_of_day}" && exit 2)
> @echo "plot-sleep: PID=$(PID) SNAP=$(SNAP) POLICY=$(POLICY)"
> @echo "Running: PYTHONPATH=\"$$PWD\" \"$(VENV_PY)\" etl_tools/plot_sleep_compare.py --join \"data_ai/$(PID)/snapshots/$(SNAP)/hybrid_join/$(POLICY)/join_hybrid_daily.csv\" --outdir \"data_ai/$(PID)/snapshots/$(SNAP)/hybrid_join/$(POLICY)/plots\""
> @PYTHONPATH="$$PWD" "$(VENV_PY)" etl_tools/plot_sleep_compare.py \
> 	--join "data_ai/$(PID)/snapshots/$(SNAP)/hybrid_join/$(POLICY)/join_hybrid_daily.csv" \
> 	--outdir "data_ai/$(PID)/snapshots/$(SNAP)/hybrid_join/$(POLICY)/plots"
# ---------------- Release & Reports ----------------
# Release automation targets
.PHONY: release-notes
release-notes:
> @test -n "$(TAG)" || (echo "Set TAG=..." && exit 2)
> @echo "Rendering release notes for TAG=$(TAG) VERSION=$(VERSION)"
> @PRE_REL=reports/pre_release/changes_since_$(LATEST_TAG_RAW).md; \
> if [ ! -f "$$PRE_REL" ]; then \
>   echo "NEED:"; \
>   echo "- Missing pre-release changes summary at reports/pre_release/changes_since_$(LATEST_TAG_RAW).md"; \
>   echo "- Ask VADER to summarize changes since $(LATEST_TAG_RAW) and save there."; \
>   exit 2; \
> fi; \
> PYTHONPATH="$$PWD" LATEST_TAG="$(LATEST_TAG_RAW)" "$(VENV_PY)" make_scripts/release/render_release_from_templates.py --version "$(VERSION)" --tag "$(TAG)" --title "$(RELEASE_TITLE)" --branch "${BRANCH:-main}" --author "${AUTHOR:-Rodrigo Marques Teixeira}" --project "${PROJECT:-Practicum2 – N-of-1 ADHD + BD}" $$( [ "$(RELEASE_DRY_RUN)" = "1" ] && echo --dry-run || true )

.PHONY: changelog-update
changelog-update:
> @test -n "$(TAG)" || (echo "Set TAG=..." && exit 2)
> @echo "Updating CHANGELOG.md for VERSION=$(VERSION)"
> @OUTFILE=$$( [ "$(RELEASE_DRY_RUN)" = "1" ] && echo "CHANGELOG.dryrun.md" || echo "CHANGELOG.md" ); \
> PYTHONPATH="$$PWD" "$(VENV_PY)" make_scripts/release/update_changelog.py --tag "$(TAG)" --version "$(VERSION)" --outfile "$$OUTFILE" $$( [ "$(RELEASE_DRY_RUN)" = "1" ] && echo --allow-missing-notes || true )
# ---------------- Release & Reports ----------------
SNAP     ?= 2025-09-29
PID      ?= P000001
POLICY   ?= best_of_day
REL_TAG  ?= v0.1.0

.PHONY: weekly-report changelog release-pack release-all

weekly-report:
> @echo "Weekly report for participant=$(PID) snapshot=$(SNAP) policy=$(POLICY)"
> 	@echo "Running: PYTHONPATH=\"$$PWD\" $(PY) make_scripts/weekly_report.py --participant \"$(PID)\" --snapshot \"$(SNAP)\" --policy \"$(POLICY)\" $(REPORT_ARGS)"
> 	@PYTHONPATH="$$PWD" $(PY) make_scripts/weekly_report.py --participant "$(PID)" --snapshot "$(SNAP)" --policy "$(POLICY)" $(REPORT_ARGS)

changelog:
> @mkdir -p docs_build
> @git log --pretty=format:"- %h %s (%ad)" --date=short > docs_build/CHANGELOG.md
> @echo "✅ changelog → docs_build/CHANGELOG.md"

release-pack:
> @mkdir -p dist
> @echo "# Practicum2 – Release $(REL_TAG)" > docs_build/RELEASE_NOTES.md
> @echo "- PID: $(PID)" >> docs_build/RELEASE_NOTES.md
> @echo "- SNAP: $(SNAP)" >> docs_build/RELEASE_NOTES.md
> @echo "- POLICY: $(POLICY)" >> docs_build/RELEASE_NOTES.md
> @tar -czf dist/practicum2_$(REL_TAG).tgz \
>     --exclude='*.pyc' --exclude='__pycache__' \
>     Makefile requirements_lock.txt \
>     etl_pipeline.py etl_modules etl_tools \
>     docs_build/CHANGELOG.md docs_build/RELEASE_NOTES.md \
>     data_ai/$(PID)/snapshots/$(SNAP)/features_daily.csv \
>     data_ai/$(PID)/snapshots/$(SNAP)/version_log_enriched.csv \
>     data_ai/$(PID)/snapshots/$(SNAP)/hybrid_join/$(POLICY)/join_hybrid_daily.csv \
>     data_ai/$(PID)/snapshots/$(SNAP)/hybrid_join/$(POLICY)/plots
> @echo "📦 dist/practicum2_$(REL_TAG).tgz"

release-all: weekly-report changelog release-pack
> @echo "DEPRECATED: 'release-all' is a legacy alias. Use 'release-draft' and 'release-pack' explicitly."
> @echo "If you still want the old behavior, run: make release-draft && make release-pack"


tests:
> @echo "Running tests via $(VENV_PY)"
> @PYTHONPATH="$$PWD" "$(VENV_PY)" -m pytest -q


.PHONY: model model-notebook
model:
> @python modeling/baseline_train.py --participant $(PID) --snapsho$(SNAP) --use_agg auto

model-notebook:
> @python -m webbrowser -t 'notebooks/04_modeling_baseline.ipynb'


## Publish to GitHub: create annotated tag and GH release with assets
.PHONY: publish-release
publish-release: ## Create annotated tag and GitHub release with assets (guarded)
> 	@echo "Publishing $(NEXT_TAG) (draft=$(RELEASE_DRAFT), push=$(RELEASE_PUSH), dry=$(RELEASE_DRY_RUN))"
> 	@$(MAKE) print-version >/dev/null
> 	@test -f "$(NOTES_FILE)" || { echo "ERROR: missing release notes: $(NOTES_FILE)"; exit 2; }
> 	@test -d "$(ASSET_DIR)"  || { echo "ERROR: missing asset dir: $(ASSET_DIR)"; exit 2; }
> 	@if [ "$(RELEASE_DRY_RUN)" = "1" ]; then \
> 	   echo "[DRY-RUN] Would run version-guard for TAG=$(NEXT_TAG)"; \
> 	 else \
> 	   $(MAKE) version-guard TAG="$(NEXT_TAG)"; \
> 	 fi
> 	@if [ "$(RELEASE_DRY_RUN)" = "1" ]; then \
> 	   echo "[DRY-RUN] Would create annotated tag $(NEXT_TAG)"; \
> 	 else \
> 	   git tag -a "$(NEXT_TAG)" -m "Release $(NEXT_TAG)" || true; \
> 	 fi
> 	@if [ "$(RELEASE_PUSH)" = "1" ]; then \
> 	   if [ "$(RELEASE_DRY_RUN)" = "1" ]; then \
> 	     echo "[DRY-RUN] Would push tag $(NEXT_TAG) to origin"; \
> 	   else \
> 	     git push origin "$(NEXT_TAG)" || true; \
> 	   fi; \
> 	 else \
> 	   echo "Skipping tag push (RELEASE_PUSH=0)"; \
> 	 fi
> 	@if [ "$(RELEASE_DRY_RUN)" = "1" ]; then \
> 	   echo "[DRY-RUN] Would create GitHub release $(NEXT_TAG) with assets from $(ASSET_DIR)"; \
> 	 else \
> 	   if [ "$(RELEASE_DRAFT)" = "1" ]; then \
> 	     gh release create "$(NEXT_TAG)" "$(ASSET_DIR)"/* --draft --title "$(NEXT_TAG)" --notes-file "$(NOTES_FILE)" || true; \
> 	   else \
> 	     gh release create "$(NEXT_TAG)" "$(ASSET_DIR)"/* --title "$(NEXT_TAG)" --notes-file "$(NOTES_FILE)" || true; \
> 	   fi; \
> 	 fi
> 	@echo "publish-release finished (tag=$(NEXT_TAG), draft=$(RELEASE_DRAFT), pushed=$(RELEASE_PUSH), dry=$(RELEASE_DRY_RUN))"

# ----------------------------------------------------------------------
# Single-target: etl-extract
# Behavior: run participant-scoped etl_pipeline.py extract with auto-zip discovery
# Constraints: idempotent (handled by script), no access to other participants zips,
#              print short header, exit non-zero if raw participant folder or zips missing.
# Default variables (overridable by caller): PY, PARTICIPANT, SNAPSHOT, CUTOVER, TZ_BEFORE, TZ_AFTER
# This target is intentionally appended and does not modify other targets.

PY ?= python
PARTICIPANT ?= P000001
SNAPSHOT ?= 2025-09-29
CUTOVER ?= 2023-07-15
TZ_BEFORE ?= America/Sao_Paulo
TZ_AFTER ?= Europe/Dublin

.PHONY: etl-extract
etl-extract:
> @echo "=== etl-extract: participant=$(PARTICIPANT) snapshot=$(SNAPSHOT) PWD=$$(pwd) ==="
> @test -d "data/raw/$(PARTICIPANT)" || (echo "ERROR: data/raw/$(PARTICIPANT) not found. Place participant-scoped zips under data/raw/$(PARTICIPANT)/" && exit 2)
> @$(VENV_PY) tools/check_zips.py $(PARTICIPANT)
> $(VENV_PY) -m src.etl_pipeline extract \
>   --participant $(PARTICIPANT) --snapshot $(SNAPSHOT) \
>   --cutover $(CUTOVER) --tz_before $(TZ_BEFORE) --tz_after $(TZ_AFTER) \
>   --auto-zip

# ==== WORKFLOW TARGETS (ETL → AI → NB2) ============================================
# Defaults (override na CLI: make alvo VAR=valor)
PY            ?= python
PARTICIPANT   ?= P000001
SNAPSHOT      ?= 2025-09-29
CUTOVER       ?= 2023-07-15       # Brasil → Irlanda (ajuste a sua data real)
TZ_BEFORE     ?= America/Sao_Paulo
TZ_AFTER      ?= Europe/Dublin

AI_DIR           ?= data/ai/$(PARTICIPANT)/snapshots/$(SNAPSHOT)
# Script entrypoints moved to src/ for v4 layout
ETL_SCRIPT       ?= src/etl_pipeline.py
NB2_SCRIPT       ?= src/models_nb2.py
# New layout variables: local active datasets and kaggle export root
AI_LOCAL_ROOT    ?= ./data/ai/local
AI_KAGGLE_ROOT   ?= ./data/ai/kaggle
# FEATURES_LABELED defaults to the local active dataset location for backwards compat
FEATURES_LABELED ?= $(AI_LOCAL_ROOT)/$(PARTICIPANT)/features_daily_labeled.csv
ETL_JOINED_DIR   ?= data/etl/$(PARTICIPANT)/snapshots/$(SNAPSHOT)/joined
ETL_LABELED_SRC  ?= $(ETL_JOINED_DIR)/features_daily_labeled.csv

.PHONY: help-workflow etl etl-full etl-labels etl-aggregate nb2-dry-run nb2-run

help-workflow:
> echo ""
> echo "🧭 Workflow (snapshot=$(SNAPSHOT), participant=$(PARTICIPANT))"
> echo "  make etl-extract     → (já existente) extrai zips com --auto-zip"
> echo "  make etl-full        → normaliza + join + features (pipeline completo)"
> echo "  make etl-labels      → mescla labels no joined/features_daily.csv"
> echo "  make etl-aggregate   → promove para $(AI_DIR) e materializa features_daily_labeled.csv"
> echo "  make nb2-dry-run     → checa dataset e mostra comando NB2"
> echo "  make nb2-run         → executa NB2 apontando para $(FEATURES_LABELED)"
> echo ""
> echo "Variáveis úteis: PARTICIPANT SNAPSHOT CUTOVER TZ_BEFORE TZ_AFTER NB2_SCRIPT"
> echo ""

# Alias tradicional: 'make etl' chama o pipeline completo (etl-full)
etl: etl-full
> echo "⚙️  Alias: 'make etl' → 'make etl-full'"

# Pipeline complete (normalize + join + features). Respeita --auto-zip se implementado no script.
etl-full:
> echo ">>> ETL FULL > $(PARTICIPANT) @ $(SNAPSHOT)"
> $(PYTHON) -m src.etl_pipeline full \
>   --participant $(PARTICIPANT) \
>   --snapshot $(SNAPSHOT) \
>   --cutover $(CUTOVER) \
>   --tz_before $(TZ_BEFORE) \
>   --tz_after $(TZ_AFTER)
> echo "✅ ETL FULL concluído. Verifique: $(ETL_JOINED_DIR)"

# Apenas a fusão de labels (se o joined já existir)
etl-labels:
> echo "🧩 ETL LABELS → $(PARTICIPANT) @ $(SNAPSHOT)"
> $(PYTHON) -m src.etl_pipeline labels \
>   --participant $(PARTICIPANT) \
>   --snapshot $(SNAPSHOT)
> if [ ! -f "$(ETL_LABELED_SRC)" ]; then \
>   echo "❌ Esperado: $(ETL_LABELED_SRC) — rode 'make etl-full' antes."; exit 1; \
> fi
> echo "✅ Labels mesclados: $(ETL_LABELED_SRC)"

# Promove do ETL para data/ai (cópia determinística do labeled)
etl-aggregate:
> echo "📦 AGGREGATE → $(AI_DIR)"
> if [ ! -f "$(ETL_LABELED_SRC)" ]; then \
>   echo "❌ Não encontrei $(ETL_LABELED_SRC). Rode 'make etl-full' e 'make etl-labels' antes."; exit 1; \
> fi
> mkdir -p "$(AI_DIR)"
> cp -f "$(ETL_LABELED_SRC)" "$(FEATURES_LABELED)"
> if [ ! -f "$(FEATURES_LABELED)" ]; then \
>   echo "❌ Falha ao materializar $(FEATURES_LABELED)"; exit 1; \
> fi
> echo "✅ Dataset AI pronto: $(FEATURES_LABELED)"

# Mostra o comando NB2 e valida a existência do dataset (now using AI_LOCAL_ROOT)
nb2-dry-run:
> echo "NB2 DRY-RUN"
> if [ ! -f "$(FEATURES_LABELED)" ]; then \
>   echo "ERROR: Missing $(FEATURES_LABELED). Run 'make etl-aggregate' or etl-promote-slug first."; exit 1; \
> fi
> echo "OK: $(FEATURES_LABELED)"
> $(PY) $(NB2_SCRIPT) --features "$(FEATURES_LABELED)" --dry-run

# Executa o NB2 com o labeled promovido (default path now points to AI_LOCAL_ROOT)
nb2-run:
> echo "NB2 RUN"
> if [ ! -f "$(FEATURES_LABELED)" ]; then \
>   echo "ERROR: Missing $(FEATURES_LABELED). Run 'make etl-aggregate' or etl-promote-slug first."; exit 1; \
> fi
> $(PY) $(NB2_SCRIPT) --features "$(FEATURES_LABELED)"
> echo "NB2 finished; outputs -> notebooks/outputs/NB2/"

# ========= NB2 slug / packaging targets =========
.PHONY: nb2-run-slug etl-promote-slug kaggle-pack clean-legacy

# Run NB2 for a specific slug under AI_LOCAL_ROOT
nb2-run-slug: check-nb2-entry
> @test -n "$(SLUG)" || (echo "ERROR: SLUG must be provided. e.g. SLUG=p000001-s20250929-nb2v303-r1" && exit 1)
> @test -f "$(AI_LOCAL_ROOT)/$(SLUG)/features_daily_labeled.csv" || (echo "ERROR: features file not found for slug: $(AI_LOCAL_ROOT)/$(SLUG)/features_daily_labeled.csv" && exit 1)
> @echo "[NB2] Run slug=$(SLUG) -> $(AI_LOCAL_ROOT)/$(SLUG)/features_daily_labeled.csv"
> $(PY) "$(NB2_ENTRY)" --slug "$(SLUG)" --local-root "$(AI_LOCAL_ROOT)"

.PHONY: nb3-run-slug nb3-sweep

# Run NB3 (DL) for a specific slug under AI_LOCAL_ROOT
nb3-run-slug: check-nb2-entry
> @test -n "$(SLUG)" || (echo "ERROR: SLUG must be provided. e.g. SLUG=p000001-s20250929-nb2v303-r1" && exit 1)
> @test -f "$(AI_LOCAL_ROOT)/$(SLUG)/features_daily_labeled.csv" || (echo "ERROR: features file not found for slug: $(AI_LOCAL_ROOT)/$(SLUG)/features_daily_labeled.csv" && exit 1)
> @echo "[NB3] Run slug=$(SLUG) -> $(AI_LOCAL_ROOT)/$(SLUG)/features_daily_labeled.csv"
> $(PY) "$(NB3_ENTRY)" --slug "$(SLUG)" $(ARGS)

# Run NB3 sweep
nb3-sweep: check-nb2-entry
> @test -n "$(SLUG)" || (echo "ERROR: SLUG must be provided. e.g. SLUG=p000001-s20250929-nb2v303-r1" && exit 1)
> @echo "[NB3] Sweep slug=$(SLUG) -> $(AI_LOCAL_ROOT)/$(SLUG)/features_daily_labeled.csv"
> $(PY) "$(NB3_ENTRY)" --slug "$(SLUG)" --sweep $(ARGS)

# Run NB2 for a slug letting the script autodetect environment (no --local-root needed)
nb2-run-auto: check-nb2-entry
> @test -n "$(SLUG)" || (echo "ERROR: SLUG must be provided. e.g. SLUG=p000001-s20250929-nb2v303-r1" && exit 1)
> @test -d "$(AI_LOCAL_ROOT)/$(SLUG)" || (echo "ERROR: slug not found under $(AI_LOCAL_ROOT): $(SLUG). Run 'make etl-promote-slug' or check path." && exit 1)
> @echo "[NB2] Auto-run slug=$(SLUG) (script will autodetect local vs kaggle)"
> $(PY) "$(NB2_SCRIPT)" --slug "$(SLUG)"

# Promote an existing promoted snapshot into the slugged local datasets root
etl-promote-slug:
> @test -n "$(PARTICIPANT)" || (echo "Set PARTICIPANT=Pxxxxxx" && exit 2)
> @test -n "$(SNAPSHOT)" || (echo "Set SNAPSHOT=YYYY-MM-DD" && exit 2)
> @test -n "$(SLUG)" || (echo "Set SLUG=p000001-s20250929-nb2v303-r1" && exit 1)
> if [ ! -f "$(ETL_LABELED_SRC)" ]; then echo "ERROR: expected labeled source at $(ETL_LABELED_SRC). Run 'make etl-full' and 'make etl-labels' first."; exit 1; fi
> mkdir -p "$(AI_LOCAL_ROOT)/$(SLUG)"
> cp -f "$(ETL_LABELED_SRC)" "$(AI_LOCAL_ROOT)/$(SLUG)/features_daily_labeled.csv"
> if [ -f "$(ETL_JOINED_DIR)/version_log_enriched.csv" ]; then cp -f "$(ETL_JOINED_DIR)/version_log_enriched.csv" "$(AI_LOCAL_ROOT)/$(SLUG)/version_log_enriched.csv"; fi
> echo "✅ Promoted snapshot -> $(AI_LOCAL_ROOT)/$(SLUG)/features_daily_labeled.csv"

# Create a kaggle-ready zip from a slug folder with a small provenance file
kaggle-pack:
> @test -n "$(SLUG)" || (echo "ERROR: SLUG must be provided. e.g. SLUG=p000001-s20250929-nb2v303-r1" && exit 1)
> @test -d "$(AI_LOCAL_ROOT)/$(SLUG)" || (echo "ERROR: slug folder missing: $(AI_LOCAL_ROOT)/$(SLUG)" && exit 1)
> mkdir -p "$(AI_KAGGLE_ROOT)"
> $(PY) -c "import json, pathlib; p=pathlib.Path('$(AI_LOCAL_ROOT)')/ '$(SLUG)'; (p/'provenance.json').write_text(json.dumps({'slug':'$(SLUG)'})); (p/'README.md').write_text('Kaggle package for $(SLUG)\n')"
> $(PY) -c "import shutil; shutil.make_archive('$(AI_KAGGLE_ROOT)/$(SLUG)','zip','$(AI_LOCAL_ROOT)/$(SLUG)')"
> echo "✅ Wrote kaggle package -> $(AI_KAGGLE_ROOT)/$(SLUG).zip"

# Safely handle legacy data_ai folder: rename to data_ai_legacy_YYYYMMDD or delete if FORCE=1
clean-legacy:
> if [ -d "data_ai" ]; then \
>   if [ "$(FORCE)" = "1" ]; then echo "Deleting legacy data_ai (FORCE=1)"; rm -rf data_ai; else bak="data_ai_legacy_$$(date +%Y%m%d)"; echo "Renaming data_ai -> $$bak"; mv data_ai "$$bak"; fi; \
> else echo "No legacy data_ai dir found"; fi

.PHONY: nb2-batch-dry-run nb2-batch check-nb2-entry check-local-root

# ========= NB2 batch (discovery-based) =========
# Vars:
#   LOCAL_DATASETS_ROOT: where slugged datasets live locally (default ./data/ai/datasets)
#   LIMIT: max number of datasets to process in batch (default 1)
# Usage:
#   make nb2-batch-dry-run
#   make nb2-batch-dry-run LOCAL_DATASETS_ROOT=./data/ai/local LIMIT=2
#   make nb2-batch
#   make nb2-batch LOCAL_DATASETS_ROOT=./data/ai/local LIMIT=5
AI_LOCAL_ROOT ?= ./data/ai/local
LOCAL_DATASETS_ROOT ?= $(AI_LOCAL_ROOT)
# Entrypoints now reference src/ modules (v4 layout)
NB2_ENTRY ?= src/models_nb2.py
NB3_ENTRY ?= src/models_nb3.py
LIMIT ?= 1


check-nb2-entry:
> @test -f "$(NB2_ENTRY)" || (echo "ERROR: NB2 entry not found at $(NB2_ENTRY). Set NB2_ENTRY or fix path." && exit 1)

check-local-root:
> @test -d "$(LOCAL_DATASETS_ROOT)" || (echo "ERROR: LOCAL_DATASETS_ROOT not found: $(LOCAL_DATASETS_ROOT). Create a slug dir like p000001-s20250929-nb2v303-r1/" && exit 1)

nb2-batch-dry-run: check-nb2-entry check-local-root
> @echo "[NB2] Batch DRY-RUN from $(LOCAL_DATASETS_ROOT) (limit=$(LIMIT))"
> $(PY) "$(NB2_ENTRY)" --batch --local-root "$(LOCAL_DATASETS_ROOT)" --limit $(LIMIT) --dry-run

nb2-batch: check-nb2-entry check-local-root
> @echo "[NB2] Batch RUN from $(LOCAL_DATASETS_ROOT) (limit=$(LIMIT))"
> $(PY) "$(NB2_ENTRY)" --batch --local-root "$(LOCAL_DATASETS_ROOT)" --limit $(LIMIT)

.PHONY: selftest-extract
selftest-extract:
> @echo "PWD=$$(pwd)"
> @echo "Scanning: data/raw/$(PARTICIPANT)/apple and zepp"
> @ls -lh data/raw/$(PARTICIPANT)/apple 2>/dev/null || true
> @ls -lh data/raw/$(PARTICIPANT)/zepp  2>/dev/null || true

SOM_SCAN_FLAGS ?=

.PHONY: som-scan
som-scan:
> @test -n "$(PARTICIPANT)" || (echo "Set PARTICIPANT=Pxxxxxx" && exit 2)
> @test -n "$(SNAPSHOT_DATE)" || (echo "Set SNAPSHOT_DATE=YYYY-MM-DD" && exit 2)
> @echo "Running som-scan for $(PARTICIPANT) snapshot=$(SNAPSHOT_DATE)"
> @PYTHONPATH="$$PWD" $(VENV_PY) -m src.etl_pipeline som-scan \
>   --participant "$(PARTICIPANT)" \
>   --snapshot "$(SNAPSHOT_DATE)" \
>   --cutover "$(CUTOVER)" \
>   --tz_before "$(TZ_BEFORE)" \
>   --tz_after "$(TZ_AFTER)" \
>   $(SOM_SCAN_FLAGS)

.PHONY: help-som-scan
help-som-scan:
> @echo "Examples:"
> @echo "  make som-scan PARTICIPANT=P000001 SNAPSHOT_DATE=2025-09-29 SOM_SCAN_FLAGS='--trace'"
> @echo "  make som-scan PARTICIPANT=P000001 SNAPSHOT_DATE=2025-09-29 SOM_SCAN_FLAGS='--write-normalized'"
