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
PIP   ?= $(PY) -m pip
PID ?= P000001
TZ  ?= Europe/Dublin


# Paths
BACKUP_DIR ?= C:\Users\Administrador\Apple\MobileSync\Backup\00008120-000E18E21144A01E
OUT_DIR    ?= decrypted_output

IOS_DIR    := ios_extract
ETL_DIR    := etl


install-dev:
> $(PIP) install --upgrade -r requirements_dev.txt

install-all:
> $(PIP) install --upgrade -r requirements.txt

# --- iOS Extraction ------------------------------------------
decrypt:
> @BACKUP_DIR="$(BACKUP_DIR)" OUT_DIR="$(OUT_DIR)" \
>  BACKUP_PASSWORD="$${BACKUP_PASSWORD:-}" \
>  $(PY) $(DEC_MANIFEST)

probe:
> @BACKUP_DIR="$(BACKUP_DIR)" OUT_DIR="$(OUT_DIR)" BACKUP_PASSWORD="$${BACKUP_PASSWORD:-}" ; \
> $(PY) "$(PROBE)"

extract-plists:
> @BACKUP_DIR="$(BACKUP_DIR)" OUT_DIR="$(OUT_DIR)" BACKUP_PASSWORD="$${BACKUP_PASSWORD:-}" ; \
> $(PY) "$(EXTRACT_PLISTS)"

plist-csv:
> @BACKUP_DIR="$(BACKUP_DIR)" OUT_DIR="$(OUT_DIR)" BACKUP_PASSWORD="$${BACKUP_PASSWORD:-}" ; \
> $(PY) "$(PLISTS_TO_CSV)"

extract-knowledgec:
> @BACKUP_DIR="$(BACKUP_DIR)" OUT_DIR="$(OUT_DIR)" BACKUP_PASSWORD="$${BACKUP_PASSWORD:-}" ; \
> $(PY) "$(EXTRACT_KNOWLEDGEC)"

parse-knowledgec:
> @BACKUP_DIR="$(BACKUP_DIR)" OUT_DIR="$(OUT_DIR)" BACKUP_PASSWORD="$${BACKUP_PASSWORD:-}" ; \
> if [ -f "$(PARSE_KNOWLEDGEC)" ]; then \
>   $(PY) "$(PARSE_KNOWLEDGEC)"; \
> else \
>   echo "parse_knowledgec_usage.py not present yet (will be added when schema is detected)."; \
> fi

probe-sqlite:
> @BACKUP_DIR="$(BACKUP_DIR)" OUT_DIR="$(OUT_DIR)" BACKUP_PASSWORD="$${BACKUP_PASSWORD:-}" ; \
> $(PY) ios_extract/probe_sqlite_targets.py

extract-screentime-sqlite:
> @BACKUP_DIR="$(BACKUP_DIR)" OUT_DIR="$(OUT_DIR)" BACKUP_PASSWORD="$${BACKUP_PASSWORD:-}" ; \
> $(PY) ios_extract/extract_screentime_sqlite.py

etl:
> @BACKUP_DIR="$(BACKUP_DIR)" OUT_DIR="$(OUT_DIR)" \
>  BACKUP_PASSWORD="$${BACKUP_PASSWORD:-}" \
>  $(PY) $(ETL_PIPELINE) --cutover $(CUTOVER) --tz_before $(TZ_BEFORE) --tz_after $(TZ_AFTER)


# --- Zepp Export -------------------------------------------------
ZEPP_DIR := data_etl/P000001/zepp_export
ZEPP_ZIP ?= $(firstword $(wildcard $(ZEPP_DIR)/*.zip))
ZEPP_OUT ?= decrypted_output/zepp

.PHONY: list-zepp parse-zepp unpack-zepp inspect-zepp zepp-parse-one zepp-aggregate

list-zepp:
> @test -n "$(ZEPP_ZIP)" || (echo "No .zip found in data_etl/$(PID)/zepp_export. Pass ZEPP_ZIP=... or put a zip there." && exit 2)
> @$(PY) -c "import os,zipfile; p=os.environ.get('ZEPP_ZIP','$(ZEPP_ZIP)'); z=zipfile.ZipFile(p); [print(n) for n in z.namelist()]"

# LEGADO: parse-zepp antigo (mantido p/ compat). Agora usa --outdir-root
# Ex.: make parse-zepp ZIP=data_etl/P000001/zepp_export/X.zip OUT=data_etl/P000001/zepp_processed PASS='senha'
parse-zepp:
> . .venv/Scripts/activate && \
> export PYTHONPATH="$$PWD" && \
> python etl_modules/parse_zepp_export.py \
>     --input "$(ZIP)" \
>     --outdir-root "$(OUT)" \
>     --tz "Europe/Dublin" \
>     $$([ -n "$(PASS)" ] && echo --password \"$(PASS)\" || true)

# Uso:
# make unpack-zepp ZIP="data_etl/P000001/zepp_export/3088....zip" OUT="data_etl/P000001/zepp_raw_unpacked" PASS="sYhspDax"

# Parse a partir de diretório já extraído (contorna ZIP AES no Windows)
# Ex.: make zepp-parse-dir PID=P000001 DIR="data_etl/P000001/zepp_raw_unpacked"
.PHONY: zepp-parse-dir
zepp-parse-dir:
> . .venv/Scripts/activate && \
> export PYTHONPATH="$$PWD" && \
> python -m etl_modules.parse_zepp_export \
>     --input "$(DIR)" \
>     --outdir-root "data_etl/$(PID)/zepp_processed" \
>     --participant "$(PID)" \
>     --cutover "$(CUTOVER)" \
>     --tz_before "$(TZ_BEFORE)" \
>     --tz_after  "$(TZ_AFTER)"

# Rebuild _latest a partir de todos os subdirs versionados (append-only)
# Ex.: make zepp-aggregate PID=P000001
zepp-aggregate:
> . .venv/Scripts/activate && \
> export PYTHONPATH="$$PWD" && \
> python etl_tools/zepp_rebuild_latest.py \
>     --root "data_etl/$(PID)/zepp_processed"

# --- ETL Apple (um snapshot específico) -----------------------
# Ex.: make etl-one PID=P000001 SNAP=2025-09-29
etl-one:
> @test -n "$(PID)"  || (echo "Set PID=Pxxxxxx" && exit 2)
> @test -n "$(SNAP)" || (echo "Set SNAP=YYYY-MM-DD (ou YYYYMMDD)" && exit 2)
> @. .venv/Scripts/activate && \
> export PYTHONPATH="$$PWD" && \
> $(PY) $(ETL_PIPELINE) \
>     --participant "$(PID)" \
>     --snapshot "$(SNAP)" \
>     --cutover "$(CUTOVER)" \
>     --tz_before "$(TZ_BEFORE)" \
>     --tz_after "$(TZ_AFTER)"

# --- Comparação Zepp vs Apple + plots de uma vez --------------
# Ex.: make plots-all PID=P000001 SNAP=2025-09-29 POLICY=best_of_day
plots-all:
> @test -n "$(PID)"    || (echo "Set PID=Pxxxxxx" && exit 2)
> @test -n "$(SNAP)"   || (echo "Set SNAP=YYYY-MM-DD (ou YYYYMMDD)" && exit 2)
> @test -n "$(POLICY)" || (echo "Set POLICY={apple_first|zepp_first|best_of_day}" && exit 2)
> @. .venv/Scripts/activate && \
> export PYTHONPATH="$$PWD" && \
> $(PY) etl_tools/compare_zepp_apple.py \
>     --pid "$(PID)" \
>     --zepp-root "data_etl/$(PID)/zepp_processed" \
>     --apple-dir "data_ai/$(PID)/snapshots/$(SNAP)" \
>     --out-dir "data_ai/$(PID)/snapshots/$(SNAP)/hybrid_join/$(POLICY)" \
>     --sleep-policy "$(POLICY)"
> @. .venv/Scripts/activate && \
> export PYTHONPATH="$$PWD" && \
> $(PY) etl_tools/plot_sleep_compare.py \
>     --join "data_ai/$(PID)/snapshots/$(SNAP)/hybrid_join/$(POLICY)/join_hybrid_daily.csv" \
>     --outdir "data_ai/$(PID)/snapshots/$(SNAP)/hybrid_join/$(POLICY)/plots"

# --- Comparação + plots (modo "lite"): escreve/usa JOIN genérico sem subpasta da policy
# Uso:
#   make plots-all-lite PID=P000001 SNAP=2025-09-29            # usa POLICY=best_of_day por padrão
#   make plots-all-lite PID=P000001 SNAP=2025-09-29 POLICY=zepp_first
plots-all-lite:
> @test -n "$(PID)"  || (echo "Set PID=Pxxxxxx" && exit 2)
> @test -n "$(SNAP)" || (echo "Set SNAP=YYYY-MM-DD (ou YYYYMMDD)" && exit 2)
> @. .venv/Scripts/activate && \
> export PYTHONPATH="$$PWD" && \
> policy="$${POLICY:-best_of_day}" && \
> $(PY) etl_tools/compare_zepp_apple.py \
>     --pid "$(PID)" \
>     --zepp-root "data_etl/$(PID)/zepp_processed" \
>     --apple-dir "data_ai/$(PID)/snapshots/$(SNAP)" \
>     --out-dir "data_ai/$(PID)/snapshots/$(SNAP)/hybrid_join" \
>     --sleep-policy "$$policy" && \
> $(PY) etl_tools/plot_sleep_compare.py \
>     --join "data_ai/$(PID)/snapshots/$(SNAP)/hybrid_join/join_hybrid_daily.csv" \
>     --outdir "data_ai/$(PID)/snapshots/$(SNAP)/hybrid_join/plots"

# --- Maintenance ---------------------------------------------
clean:
> @find . -name "__pycache__" -type d -prune -exec rm -rf {} + || true
> @find . -name "*.pyc" -delete || true
> @find . -name "*.log" -delete || true
> @echo "✔ cleaned caches/logs"

deepclean:
> @rm -rf $(OUT_DIR) $(IOS_DIR)/decrypted_output || true
> @echo "⚠ removed decrypted outputs (PII)."

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

.PHONY: zepp-apple-compare

# Ex.: make zepp-apple-compare PID=P000001 SNAP=2024-06-30 POLICY=best_of_day
zepp-apple-compare:
# converted to recipe prefix style
> @test -n "$(POLICY)" || (echo "Set POLICY={apple_first|zepp_first|best_of_day}" && exit 2)
> @. .venv/Scripts/activate && \
> export PYTHONPATH="$$PWD" && \
> python etl_tools/compare_zepp_apple.py \
>     --pid "$(PID)" \
>     --zepp-root "data_etl/$(PID)/zepp_processed" \
>     --apple-dir "data_ai/$(PID)/snapshots/$(SNAP)" \
>     --out-dir "data_ai/$(PID)/snapshots/$(SNAP)/hybrid_join" \
>     --sleep-policy "$(POLICY)"

# Ex.: make freeze-model-input PID=P000001 SNAP=2024-06-30 POLICY=best_of_day
freeze-model-input:
> @PYTHONPATH="$$PWD" $(PY) make_scripts/freeze_model_input.py --participant "$(PID)" --snapshot "$(SNAP)" --policy "$(POLICY)"

# Ex.: make plot-sleep PID=P000001 SNAP=2024-06-30 POLICY=best_of_day
plot-sleep:
# converted to recipe prefix style
> @test -n "$(POLICY)" || (echo "Set POLICY={apple_first|zepp_first|best_of_day}" && exit 2)
> @. .venv/Scripts/activate && \
> export PYTHONPATH="$$PWD" && \
> python etl_tools/plot_sleep_compare.py \
>     --join "data_ai/$(PID)/snapshots/$(SNAP)/hybrid_join/$(POLICY)/join_hybrid_daily.csv" \
>     --outdir "data_ai/$(PID)/snapshots/$(SNAP)/hybrid_join/$(POLICY)/plots"

# ---------------- Release & Reports ----------------
SNAP     ?= 2025-09-29
PID      ?= P000001
POLICY   ?= best_of_day
REL_TAG  ?= v0.1.0

.PHONY: weekly-report changelog release-pack release-all

weekly-report:
> @PYTHONPATH="$$PWD" $(PY) make_scripts/weekly_report.py --participant "$(PID)" --snapshot "$(SNAP)" --policy "$(POLICY)"

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
> @echo "✔ release bundle pronto. Sugestão:"
> @echo "  git add -A && git commit -m 'release $(REL_TAG)' && git tag $(REL_TAG)"

tests:
> @. .venv/Scripts/activate || true
> @PYTHONPATH="$$PWD" pytest -q


.PHONY: model model-notebook
model:
> @python modeling/baseline_train.py --participant $(PID) --snapshot $(SNAP) --use_agg auto

model-notebook:
> @python -m webbrowser -t 'notebooks/04_modeling_baseline.ipynb'
