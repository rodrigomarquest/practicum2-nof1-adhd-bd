# ARG/PARAM POLICY
# - CLI flags and ENV are both supported. Precedence: CLI flags > ENV vars > Make defaults > Script defaults.
# - Core ENV: ETL_DIR, AI_DIR, PARTICIPANT, SNAPSHOT_DATE, DRY_RUN, NON_INTERACTIVE.
# - Each target provides a *_ARGS pass-through (e.g., CLEAN_ARGS) for CLI flags.
# - Never rely on venv activation in recipes; always call $(VENV_PY).
# - Never use heredoc; multi-line logic lives in make_scripts/*.py (see lint targets).

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
NON_INTERACTIVE ?= 0
PIP   ?= $(PY) -m pip
# Resolve venv python interpreter path (prefer Windows Scripts, then POSIX bin). Falls back to $(PY)
VENV_PY ?= $(shell if [ -x .venv/Scripts/python.exe ]; then echo $(abspath .venv/Scripts/python.exe); elif [ -x .venv/Scripts/python ]; then echo $(abspath .venv/Scripts/python); elif [ -x .venv/bin/python ]; then echo $(abspath .venv/bin/python); else echo $(PY_ABS); fi)
PID ?= P000001
TZ  ?= Europe/Dublin


# Paths
BACKUP_DIR ?= C:\Users\Administrador\Apple\MobileSync\Backup\00008120-000E18E21144A01E
OUT_DIR    ?= decrypted_output

IOS_DIR    := ios_extract
ETL_DIR    := etl

# Canonical data path variables (user-overridable where appropriate)
# New taxonomy: raw -> extracted -> normalized -> processed -> joined -> ai_input
DATA_BASE       := ./data
PARTICIPANT    ?= P000001
RAW_DIR         := $(DATA_BASE)/raw/$(PARTICIPANT)
ETL_DIR         := $(DATA_BASE)/etl/$(PARTICIPANT)
EXTRACTED_DIR   := $(ETL_DIR)/extracted
NORMALIZED_DIR  := $(ETL_DIR)/normalized
PROCESSED_DIR   := $(ETL_DIR)/processed
JOINED_DIR      := $(ETL_DIR)/joined
AI_DIR          := $(DATA_BASE)/ai/$(PARTICIPANT)
SNAPSHOT_DATE  ?= 2025-09-29
AI_SNAPSHOT_DIR := $(AI_DIR)/snapshots/$(SNAPSHOT_DATE)

# Backwards-compatible aliases (kept for older targets/settings)
DATA_ETL_BASE := $(DATA_BASE)/etl
DATA_AI_BASE  := $(DATA_BASE)/ai

# Per-target passthrough args (allow flag-style invocation)
CLEAN_ARGS ?=
ENV_ARGS ?=
REPORT_ARGS ?=
INTAKE_ARGS ?=
PROV_ARGS ?=


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
.PHONY: help-layout
help-layout:
>	@echo "Layout workflow targets and usage"
>	@echo "  make init-data-layout       # create raw/extracted/normalized/processed/joined and ai/snapshots"
>	@echo "  make migrate-layout         # move legacy etl/.../exports -> data/raw/<pid>"
>	@echo "  make intake-zip ...         # ingest device/app zips into raw/<source> (optional minimal extract)"
>	@echo "  make clean-raw              # NO-OP unless flag --i-understand-raw is passed (safety)"
>	@echo "  make clean-extracted|normalized|processed|joined|ai-snapshot"
>	@echo "  make provenance             # run inventory & audit over normalized/processed/joined/ai snapshot"
>	@echo "  make lint-deprecated-exports"
>	@echo "  make lint-layout"
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
> @echo "parse-zepp: parse Zepp ZIP/dir"
> @echo "Running: PYTHONPATH=\"$$PWD\" $(PY) make_scripts/parse_zepp_make.py --input \"$(ZIP)\" --outdir-root \"$(OUT)\" --tz \"Europe/Dublin\" $$( [ -n "$(PASS)" ] && echo --password \"$(PASS)\" || true ) $$( [ "$(DRY_RUN)" = "1" ] && echo --dry-run || true )"
> @PYTHONPATH="$$PWD" $(PY) make_scripts/parse_zepp_make.py --input "$(ZIP)" --outdir-root "$(OUT)" --tz "Europe/Dublin" $$( [ -n "$(PASS)" ] && echo --password "$(PASS)" || true ) $$( [ "$(DRY_RUN)" = "1" ] && echo --dry-run || true )

# Uso:
# make unpack-zepp ZIP="data_etl/P000001/zepp_export/3088....zip" OUT="data_etl/P000001/zepp_raw_unpacked" PASS="sYhspDax"

# Parse a partir de diretório já extraído (contorna ZIP AES no Windows)
# Ex.: make zepp-parse-dir PID=P000001 DIR="data_etl/P000001/zepp_raw_unpacked"
.PHONY: zepp-parse-dir
zepp-parse-dir:
> @echo "zepp-parse-dir: parse zepp from dir"
> @echo "Running: PYTHONPATH=\"$$PWD\" $(PY) make_scripts/parse_zepp_make.py --input \"$(DIR)\" --outdir-root \"data_etl/$(PID)/zepp_processed\" --participant \"$(PID)\" --cutover \"$(CUTOVER)\" --tz_before \"$(TZ_BEFORE)\" --tz_after \"$(TZ_AFTER)\" $$( [ "$(DRY_RUN)" = "1" ] && echo --dry-run || true )"
> @PYTHONPATH="$$PWD" $(PY) make_scripts/parse_zepp_make.py --input "$(DIR)" --outdir-root "data_etl/$(PID)/zepp_processed" --participant "$(PID)" --cutover "$(CUTOVER)" --tz_before "$(TZ_BEFORE)" --tz_after "$(TZ_AFTER)" $$( [ "$(DRY_RUN)" = "1" ] && echo --dry-run || true )

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

# --- Comparação Zepp vs Apple + plots de uma vez --------------
# Ex.: make plots-all PID=P000001 SNAP=2025-09-29 POLICY=best_of_day
plots-all:
> @test -n "$(PID)"    || (echo "Set PID=Pxxxxxx" && exit 2)
> @test -n "$(SNAP)"   || (echo "Set SNAP=YYYY-MM-DD (ou YYYYMMDD)" && exit 2)
> @test -n "$(POLICY)" || (echo "Set POLICY={apple_first|zepp_first|best_of_day}" && exit 2)
> @echo "plots-all: compare+plot for PID=$(PID) SNAP=$(SNAP) POLICY=$(POLICY)"
> @echo "Running: PYTHONPATH=\"$$PWD\" \"$(VENV_PY)\" etl_tools/compare_zepp_apple.py --pid \"$(PID)\" --zepp-root \"data_etl/$(PID)/zepp_processed\" --apple-dir \"data_ai/$(PID)/snapshots/$(SNAP)\" --out-dir \"data_ai/$(PID)/snapshots/$(SNAP)/hybrid_join/$(POLICY)\" --sleep-policy \"$(POLICY)\""
> @PYTHONPATH="$$PWD" "$(VENV_PY)" etl_tools/compare_zepp_apple.py \
	--pid "$(PID)" \
	--zepp-root "data_etl/$(PID)/zepp_processed" \
	--apple-dir "data_ai/$(PID)/snapshots/$(SNAP)" \
	--out-dir "data_ai/$(PID)/snapshots/$(SNAP)/hybrid_join/$(POLICY)" \
	--sleep-policy "$(POLICY)"
> @echo "Running: PYTHONPATH=\"$$PWD\" \"$(VENV_PY)\" etl_tools/plot_sleep_compare.py --join \"data_ai/$(PID)/snapshots/$(SNAP)/hybrid_join/$(POLICY)/join_hybrid_daily.csv\" --outdir \"data_ai/$(PID)/snapshots/$(SNAP)/hybrid_join/$(POLICY)/plots\""
> @PYTHONPATH="$$PWD" "$(VENV_PY)" etl_tools/plot_sleep_compare.py \
	--join "data_ai/$(PID)/snapshots/$(SNAP)/hybrid_join/$(POLICY)/join_hybrid_daily.csv" \
	--outdir "data_ai/$(PID)/snapshots/$(SNAP)/hybrid_join/$(POLICY)/plots"

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


.PHONY: print-paths init-data init-data-layout test-clean-layout
print-paths:
>	@echo DATA_BASE=$(DATA_BASE)
>	@echo PARTICIPANT=$(PARTICIPANT)
>	@echo RAW_DIR=$(RAW_DIR)
>	@echo ETL_DIR=$(ETL_DIR)
>	@echo EXTRACTED_DIR=$(EXTRACTED_DIR)
>	@echo NORMALIZED_DIR=$(NORMALIZED_DIR)
>	@echo PROCESSED_DIR=$(PROCESSED_DIR)
>	@echo JOINED_DIR=$(JOINED_DIR)
>	@echo AI_DIR=$(AI_DIR)
>	@echo AI_SNAPSHOT_DIR=$(AI_SNAPSHOT_DIR)
>	@echo SNAPSHOT_DATE=$(SNAPSHOT_DATE)
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
>	@echo "make check-reqs          (verify numpy, pandas, matplotlib inside .venv)"
>	@echo "Layout: make help-layout  (intake/migrate/clean/provenance targets)"

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
>	@echo "Running provenance audit for participant=$(PARTICIPANT)"
>	@NORMALIZED_DIR="$(NORMALIZED_DIR)" PROCESSED_DIR="$(PROCESSED_DIR)" JOINED_DIR="$(JOINED_DIR)" AI_SNAPSHOT_DIR="$(AI_SNAPSHOT_DIR)" $(VENV_PY) make_scripts/provenance_audit.py $(if $(DRY_RUN),--dry-run,)
>	@echo "Summary:"
>	@cat provenance/etl_provenance_checks.csv || true

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
	--pid "$(PID)" \
	--zepp-root "data_etl/$(PID)/zepp_processed" \
	--apple-dir "data_ai/$(PID)/snapshots/$(SNAP)" \
	--out-dir "data_ai/$(PID)/snapshots/$(SNAP)/hybrid_join" \
	--sleep-policy "$(POLICY)"

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
	--join "data_ai/$(PID)/snapshots/$(SNAP)/hybrid_join/$(POLICY)/join_hybrid_daily.csv" \
	--outdir "data_ai/$(PID)/snapshots/$(SNAP)/hybrid_join/$(POLICY)/plots"

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
> @echo "✔ release bundle pronto. Sugestão:"
> @echo "  git add -A && git commit -m 'release $(REL_TAG)' && git tag $(REL_TAG)"

tests:
tests:
> @echo "Running tests via $(VENV_PY)"
> @PYTHONPATH="$$PWD" "$(VENV_PY)" -m pytest -q


.PHONY: model model-notebook
model:
> @python modeling/baseline_train.py --participant $(PID) --snapshot $(SNAP) --use_agg auto

model-notebook:
> @python -m webbrowser -t 'notebooks/04_modeling_baseline.ipynb'



