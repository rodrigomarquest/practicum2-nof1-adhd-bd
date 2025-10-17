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
SHELL := /usr/bin/env bash
PY    ?= python
PIP   ?= $(PY) -m pip

# Paths
BACKUP_DIR ?= C:\Users\Administrador\Apple\MobileSync\Backup\00008120-000E18E21144A01E
OUT_DIR    ?= decrypted_output

IOS_DIR    := ios_extract
ETL_DIR    := etl

# Scripts
DEC_MANIFEST         := $(IOS_DIR)/decrypt_manifest.py
PROBE                := $(IOS_DIR)/quick_post_backup_probe.py
EXTRACT_PLISTS       := $(IOS_DIR)/smart_extract_plists.py
PLISTS_TO_CSV        := $(IOS_DIR)/plist_to_usage_csv.py
EXTRACT_KNOWLEDGEC   := $(IOS_DIR)/extract_knowledgec.py
PARSE_KNOWLEDGEC     := $(IOS_DIR)/parse_knowledgec_usage.py
ETL_PIPELINE         := $(ETL_DIR)/etl_pipeline.py

# Flags
CUTOVER ?= 2023-04-10
TZ_BEFORE ?= America/Sao_Paulo
TZ_AFTER  ?= Europe/Dublin

.PHONY: help venv install decrypt probe extract-plists plist-csv extract-knowledgec parse-knowledgec etl clean deepclean

help:
	@echo ""
	@echo "Targets:"
	@echo "  venv                - create/refresh virtualenv (.venv)"
	@echo "  install             - install Python deps (incl. iphone-backup-decrypt)"
	@echo "  decrypt             - decrypt Manifest and validate SQLite"
	@echo "  probe               - list candidate files w/ blobs present"
	@echo "  extract-plists      - extract DeviceActivity & ScreenTimeAgent plists"
	@echo "  plist-csv           - parse plists -> usage_daily_from_plists.csv"
	@echo "  extract-knowledgec  - extract CoreDuet/KnowledgeC.db if present"
	@echo "  parse-knowledgec    - parse KnowledgeC.db -> usage_daily_from_knowledgec.csv"
	@echo "  etl                 - run ETL end-to-end with timezone cutover"
	@echo "  clean               - remove caches and logs"
	@echo "  deepclean           - remove decrypted_output (PII!)"
	@echo ""

venv:
	@test -d .venv || python -m venv .venv
	@echo "→ activate: source .venv/Scripts/activate  # (Git Bash/Windows)"

install:
	$(PIP) install --upgrade pip setuptools wheel
	$(PIP) install --upgrade -r $(ETL_DIR)/requirements.txt || true
	$(PIP) install --upgrade iphone-backup-decrypt==0.9.0 pycryptodome

decrypt:
	@BACKUP_DIR="$(BACKUP_DIR)" $(PY) $(DEC_MANIFEST)

probe:
	@$(PY) $(PROBE)

extract-plists:
	@$(PY) $(EXTRACT_PLISTS)

plist-csv:
	@$(PY) $(PLISTS_TO_CSV)

extract-knowledgec:
	@$(PY) $(EXTRACT_KNOWLEDGEC)

parse-knowledgec:
	@if [ -f "$(PARSE_KNOWLEDGEC)" ]; then \
	  $(PY) $(PARSE_KNOWLEDGEC); \
	else \
	  echo "parse_knowledgec_usage.py not present yet (will be added when schema is detected)."; \
	fi

etl:
	@$(PY) $(ETL_PIPELINE) --cutover $(CUTOVER) --tz_before $(TZ_BEFORE) --tz_after $(TZ_AFTER)

clean:
	@find . -name "__pycache__" -type d -prune -exec rm -rf {} + || true
	@find . -name "*.pyc" -delete || true
	@find . -name "*.log" -delete || true
	@echo "✔ cleaned caches/logs"

deepclean:
	@rm -rf $(OUT_DIR) $(IOS_DIR)/decrypted_output || true
	@echo "⚠ removed decrypted outputs (PII)."
