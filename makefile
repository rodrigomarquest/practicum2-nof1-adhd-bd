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
PID ?= P000001
TZ  ?= Europe/Dublin


# Paths
BACKUP_DIR ?= C:\Users\Administrador\Apple\MobileSync\Backup\00008120-000E18E21144A01E
OUT_DIR    ?= decrypted_output

IOS_DIR    := ios_extract
ETL_DIR    := etl

# Scripts
DEC_MANIFEST         := $(IOS_DIR)/decrypt_manifest.py
PROBE                := $(IOS_DIR)/quick_post_backup_probe.py
EXTRACT_PLISTS       := $(IOS_DIR)/extract_plist_screentime.py
PLISTS_TO_CSV        := $(IOS_DIR)/plist_to_usage.py
EXTRACT_KNOWLEDGEC   := $(IOS_DIR)/extract_knowledgec.py
PARSE_KNOWLEDGEC     := $(IOS_DIR)/parse_knowledgec_usage.py
ETL_PIPELINE         := etl_pipeline.py

# Flags
CUTOVER ?= 2023-04-10
TZ_BEFORE ?= America/Sao_Paulo
TZ_AFTER  ?= Europe/Dublin

.PHONY: help venv install decrypt probe extract-plists plist-csv extract-knowledgec parse-knowledgec etl clean deepclean promote-current prune-decrypts list-zepp parse-zepp unpack-zepp inspect-zepp zepp-parse-one zepp-aggregate zepp-apple-compare plot-sleep etl-one plots-all plots-all-lite

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
	@echo "  promote-current     - promote a dated decrypted_output_* to canonical decrypted_output"
	@echo "  prune-decrypts      - keep only 3 newest decrypted_output_* folders"
	@echo ""

venv:
	@test -d .venv || python -m venv .venv
	@echo "→ activate: source .venv/Scripts/activate  # (Git Bash/Windows)"

install:
	$(PIP) install --upgrade pip setuptools wheel
	$(PIP) install --upgrade -r requirements_etl.txt

install-ios:
	$(PIP) install --upgrade -r requirements_ios.txt

install-ai:
	$(PIP) install --upgrade -r requirements_ai_kaggle.txt

install-dev:
	$(PIP) install --upgrade -r requirements_dev.txt

install-all:
	$(PIP) install --upgrade -r requirements.txt

# --- iOS Extraction ------------------------------------------
decrypt:
	@BACKUP_DIR="$(BACKUP_DIR)" OUT_DIR="$(OUT_DIR)" \
	 BACKUP_PASSWORD="$${BACKUP_PASSWORD:-}" \
	 $(PY) $(DEC_MANIFEST)

probe:
	@BACKUP_DIR="$(BACKUP_DIR)" OUT_DIR="$(OUT_DIR)" BACKUP_PASSWORD="$${BACKUP_PASSWORD:-}" ; \
	$(PY) "$(PROBE)"

extract-plists:
	@BACKUP_DIR="$(BACKUP_DIR)" OUT_DIR="$(OUT_DIR)" BACKUP_PASSWORD="$${BACKUP_PASSWORD:-}" ; \
	$(PY) "$(EXTRACT_PLISTS)"

plist-csv:
	@BACKUP_DIR="$(BACKUP_DIR)" OUT_DIR="$(OUT_DIR)" BACKUP_PASSWORD="$${BACKUP_PASSWORD:-}" ; \
	$(PY) "$(PLISTS_TO_CSV)"

extract-knowledgec:
	@BACKUP_DIR="$(BACKUP_DIR)" OUT_DIR="$(OUT_DIR)" BACKUP_PASSWORD="$${BACKUP_PASSWORD:-}" ; \
	$(PY) "$(EXTRACT_KNOWLEDGEC)"

parse-knowledgec:
	@BACKUP_DIR="$(BACKUP_DIR)" OUT_DIR="$(OUT_DIR)" BACKUP_PASSWORD="$${BACKUP_PASSWORD:-}" ; \
	if [ -f "$(PARSE_KNOWLEDGEC)" ]; then \
	  $(PY) "$(PARSE_KNOWLEDGEC)"; \
	else \
	  echo "parse_knowledgec_usage.py not present yet (will be added when schema is detected)."; \
	fi

probe-sqlite:
	@BACKUP_DIR="$(BACKUP_DIR)" OUT_DIR="$(OUT_DIR)" BACKUP_PASSWORD="$${BACKUP_PASSWORD:-}" ; \
	$(PY) ios_extract/probe_sqlite_targets.py

extract-screentime-sqlite:
	@BACKUP_DIR="$(BACKUP_DIR)" OUT_DIR="$(OUT_DIR)" BACKUP_PASSWORD="$${BACKUP_PASSWORD:-}" ; \
	$(PY) ios_extract/extract_screentime_sqlite.py

etl:
	@BACKUP_DIR="$(BACKUP_DIR)" OUT_DIR="$(OUT_DIR)" \
	 BACKUP_PASSWORD="$${BACKUP_PASSWORD:-}" \
	 $(PY) $(ETL_PIPELINE) --cutover $(CUTOVER) --tz_before $(TZ_BEFORE) --tz_after $(TZ_AFTER)


# --- Zepp Export -------------------------------------------------
ZEPP_DIR := data_etl/P000001/zepp_export
ZEPP_ZIP ?= $(firstword $(wildcard $(ZEPP_DIR)/*.zip))
ZEPP_OUT ?= decrypted_output/zepp

.PHONY: list-zepp parse-zepp unpack-zepp inspect-zepp zepp-parse-one zepp-aggregate

list-zepp:
	@test -n "$(ZEPP_ZIP)" || (echo "No .zip found in data_etl/$(PID)/zepp_export. Pass ZEPP_ZIP=... or put a zip there." && exit 2)
	@$(PY) -c "import os,zipfile; p=os.environ.get('ZEPP_ZIP','$(ZEPP_ZIP)'); z=zipfile.ZipFile(p); [print(n) for n in z.namelist()]"

# LEGADO: parse-zepp antigo (mantido p/ compat). Agora usa --outdir-root
# Ex.: make parse-zepp ZIP=data_etl/P000001/zepp_export/X.zip OUT=data_etl/P000001/zepp_processed PASS='senha'
parse-zepp:
	. .venv/Scripts/activate && \
	export PYTHONPATH="$$PWD" && \
	python etl_modules/parse_zepp_export.py \
		--input "$(ZIP)" \
		--outdir-root "$(OUT)" \
		--tz "Europe/Dublin" \
		$$([ -n "$(PASS)" ] && echo --password \"$(PASS)\" || true)

# Uso:
# make unpack-zepp ZIP="data_etl/P000001/zepp_export/3088....zip" OUT="data_etl/P000001/zepp_raw_unpacked" PASS="sYhspDax"
# Flags opcionais:
#   ONLYCSV=1            -> --only-csv
#   INCLUDE="HEARTRATE,HEALTH_DATA,SLEEP,BODY"
#   LISTONLY=1           -> --list-only
#   NOOVERWRITE=1        -> --no-overwrite
ifdef PASS
PASS_ARG=--password "$(PASS)"
endif
ifdef ONLYCSV
ONLYCSV_FLAG=--only-csv
endif
ifdef INCLUDE
INCLUDE_FLAG=--include "$(INCLUDE)"
endif
ifdef LISTONLY
LISTONLY_FLAG=--list-only
endif
ifdef NOOVERWRITE
NOOVERWRITE_FLAG=--no-overwrite
endif

unpack-zepp:
	. .venv/Scripts/activate && \
	export PYTHONPATH="$$PWD" && \
	python etl_tools/unpack_encrypted_zip.py \
		--zip "$(ZIP)" \
		--out "$(OUT)" \
		$(PASS_ARG) \
		$(ONLYCSV_FLAG) \
		$(INCLUDE_FLAG) \
		$(LISTONLY_FLAG) \
		$(NOOVERWRITE_FLAG)

# Relatório Excel dos CSVs extraídos (para abrir no Excel)
inspect-zepp:
	. .venv/Scripts/activate && \
	export PYTHONPATH="$$PWD" && \
	python etl_tools/inspect_csv_dir.py \
		--dir "$(DIR)" \
		--out "$(OUT)" \
		$$([ -n "$(PATTERN)" ] && echo --pattern \"$(PATTERN)\" || true) \
		$$([ -n "$(SAMPLE)" ] && echo --sample \"$(SAMPLE)\" || true)

# Parse 1 ZIP Zepp em diretório versionado por export_id e atualiza _latest
# Ex.: make zepp-parse-one PID=P000001 ZIP=".../1760774574887.zip" PASS="RTuNFjPK" TZ="Europe/Dublin"
zepp-parse-one:
	. .venv/Scripts/activate && \
	export PYTHONPATH="$$PWD" && \
	python -m etl_modules.parse_zepp_export \
		--input "$(ZIP)" \
		--outdir-root "data_etl/$(PID)/zepp_processed" \
		--participant "$(PID)" \
		--tz "$(TZ)" \
		$$([ -n "$(PASS)" ] && echo --password \"$(PASS)\" || true)

# Parse a partir de diretório já extraído (contorna ZIP AES no Windows)
# Ex.: make zepp-parse-dir PID=P000001 DIR="data_etl/P000001/zepp_raw_unpacked"
.PHONY: zepp-parse-dir
zepp-parse-dir:
	. .venv/Scripts/activate && \
	export PYTHONPATH="$$PWD" && \
	python -m etl_modules.parse_zepp_export \
		--input "$(DIR)" \
		--outdir-root "data_etl/$(PID)/zepp_processed" \
		--participant "$(PID)" \
		--cutover "$(CUTOVER)" \
		--tz_before "$(TZ_BEFORE)" \
		--tz_after  "$(TZ_AFTER)"

# Rebuild _latest a partir de todos os subdirs versionados (append-only)
# Ex.: make zepp-aggregate PID=P000001
zepp-aggregate:
	. .venv/Scripts/activate && \
	export PYTHONPATH="$$PWD" && \
	python etl_tools/zepp_rebuild_latest.py \
		--root "data_etl/$(PID)/zepp_processed"

# --- ETL Apple (um snapshot específico) -----------------------
# Ex.: make etl-one PID=P000001 SNAP=2025-09-29
etl-one:
	@test -n "$(PID)"  || (echo "Set PID=Pxxxxxx" && exit 2)
	@test -n "$(SNAP)" || (echo "Set SNAP=YYYY-MM-DD (ou YYYYMMDD)" && exit 2)
	. .venv/Scripts/activate && \
	export PYTHONPATH="$$PWD" && \
	$(PY) $(ETL_PIPELINE) \
		--participant "$(PID)" \
		--snapshot "$(SNAP)" \
		--cutover "$(CUTOVER)" \
		--tz_before "$(TZ_BEFORE)" \
		--tz_after "$(TZ_AFTER)"

# --- Comparação Zepp vs Apple + plots de uma vez --------------
# Ex.: make plots-all PID=P000001 SNAP=2025-09-29 POLICY=best_of_day
plots-all:
	@test -n "$(PID)"    || (echo "Set PID=Pxxxxxx" && exit 2)
	@test -n "$(SNAP)"   || (echo "Set SNAP=YYYY-MM-DD (ou YYYYMMDD)" && exit 2)
	@test -n "$(POLICY)" || (echo "Set POLICY={apple_first|zepp_first|best_of_day}" && exit 2)
	. .venv/Scripts/activate && \
	export PYTHONPATH="$$PWD" && \
	$(PY) etl_tools/compare_zepp_apple.py \
		--pid "$(PID)" \
		--zepp-root "data_etl/$(PID)/zepp_processed" \
		--apple-dir "data_ai/$(PID)/snapshots/$(SNAP)" \
		--out-dir "data_ai/$(PID)/snapshots/$(SNAP)/hybrid_join/$(POLICY)" \
		--sleep-policy "$(POLICY)"
	. .venv/Scripts/activate && \
	export PYTHONPATH="$$PWD" && \
	$(PY) etl_tools/plot_sleep_compare.py \
		--join "data_ai/$(PID)/snapshots/$(SNAP)/hybrid_join/$(POLICY)/join_hybrid_daily.csv" \
		--outdir "data_ai/$(PID)/snapshots/$(SNAP)/hybrid_join/$(POLICY)/plots"

# --- Comparação + plots (modo "lite"): escreve/usa JOIN genérico sem subpasta da policy
# Uso:
#   make plots-all-lite PID=P000001 SNAP=2025-09-29            # usa POLICY=best_of_day por padrão
#   make plots-all-lite PID=P000001 SNAP=2025-09-29 POLICY=zepp_first
plots-all-lite:
	@test -n "$(PID)"  || (echo "Set PID=Pxxxxxx" && exit 2)
	@test -n "$(SNAP)" || (echo "Set SNAP=YYYY-MM-DD (ou YYYYMMDD)" && exit 2)
	. .venv/Scripts/activate && \
	export PYTHONPATH="$$PWD" && \
	policy="$${POLICY:-best_of_day}" && \
	$(PY) etl_tools/compare_zepp_apple.py \
		--pid "$(PID)" \
		--zepp-root "data_etl/$(PID)/zepp_processed" \
		--apple-dir "data_ai/$(PID)/snapshots/$(SNAP)" \
		--out-dir "data_ai/$(PID)/snapshots/$(SNAP)/hybrid_join" \
		--sleep-policy "$$policy" && \
	$(PY) etl_tools/plot_sleep_compare.py \
		--join "data_ai/$(PID)/snapshots/$(SNAP)/hybrid_join/join_hybrid_daily.csv" \
		--outdir "data_ai/$(PID)/snapshots/$(SNAP)/hybrid_join/plots"

# --- Maintenance ---------------------------------------------
clean:
	@find . -name "__pycache__" -type d -prune -exec rm -rf {} + || true
	@find . -name "*.pyc" -delete || true
	@find . -name "*.log" -delete || true
	@echo "✔ cleaned caches/logs"

deepclean:
	@rm -rf $(OUT_DIR) $(IOS_DIR)/decrypted_output || true
	@echo "⚠ removed decrypted outputs (PII)."

promote-current:
	@rm -rf decrypted_output_old 2>/dev/null || true
	@test -n "$(SRC)" || (echo "Use: make promote-current SRC=decrypted_output_YYYYMMDD" && exit 1)
	@[ -d "$(SRC)" ] || (echo "SRC not found: $(SRC)" && exit 1)
	@[ -d decrypted_output ] && mv decrypted_output decrypted_output_old || true
	cp -r "$(SRC)" decrypted_output
	@echo "Promoted $(SRC) -> decrypted_output"

prune-decrypts:
	@ls -1d decrypted_output_* 2>/dev/null | sort -r | tail -n +4 | xargs -r rm -rf
	@echo "Pruned older decrypt folders (kept 3 newest)."

.PHONY: zepp-apple-compare

# Ex.: make zepp-apple-compare PID=P000001 SNAP=2024-06-30 POLICY=best_of_day
zepp-apple-compare:
	@test -n "$(POLICY)" || (echo "Set POLICY={apple_first|zepp_first|best_of_day}" && exit 2)
	. .venv/Scripts/activate && \
	export PYTHONPATH="$$PWD" && \
	python etl_tools/compare_zepp_apple.py \
		--pid "$(PID)" \
		--zepp-root "data_etl/$(PID)/zepp_processed" \
		--apple-dir "data_ai/$(PID)/snapshots/$(SNAP)" \
		--out-dir "data_ai/$(PID)/snapshots/$(SNAP)/hybrid_join" \
		--sleep-policy "$(POLICY)"

# Ex.: make freeze-model-input PID=P000001 SNAP=2024-06-30 POLICY=best_of_day
freeze-model-input:
	@test -n "$(POLICY)" || (echo "Set POLICY=..." && exit 2)
	. .venv/Scripts/activate && \
	export PYTHONPATH="$$PWD" && \
	python - <<'PY'
	from pathlib import Path
	import json, pandas as pd, hashlib, sys, os
	pid   = "$(PID)"; snap = "$(SNAP)"; policy = "$(POLICY)"
	base  = Path(f"data_ai/{pid}/snapshots/{snap}")
	join  = base / "hybrid_join" / policy / "join_hybrid_daily.csv"
	feat  = base / "features_daily.csv"   # gerado pelo ETL Apple
	outd  = base / "model_input" / policy
	if not join.exists(): sys.exit("join_hybrid_daily.csv missing for this POLICY")
	if not feat.exists(): sys.exit("features_daily.csv missing; run ETL Apple first")
	outd.mkdir(parents=True, exist_ok=True)
	# lê
	j = pd.read_csv(join, dtype={"policy":"string"}, parse_dates=["date"])
	f = pd.read_csv(feat, parse_dates=["date"])
	# injeta a coluna híbrida sem perder apple/zepp originais
	cols = [c for c in f.columns if c != "sleep_minutes_hybrid"]
	df = f[cols].merge(j[["date","sleep_minutes_hybrid","sleep_source","policy"]],
					on="date", how="left")
	# escreve
	df.to_csv(outd / "features_daily.csv", index=False)
	meta = {
	"pid": pid, "snapshot": snap, "policy": policy,
	"created_at": pd.Timestamp.utcnow().isoformat()
	}
	(json.dumps(meta, indent=2)).encode()
	(outd / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
	(outd / "lock.ok").write_text("frozen\n")
	print(f"✅ model_input frozen → {outd}")
	PY

# Ex.: make plot-sleep PID=P000001 SNAP=2024-06-30 POLICY=best_of_day
plot-sleep:
	@test -n "$(POLICY)" || (echo "Set POLICY={apple_first|zepp_first|best_of_day}" && exit 2)
	. .venv/Scripts/activate && \
	export PYTHONPATH="$$PWD" && \
	python etl_tools/plot_sleep_compare.py \
		--join "data_ai/$(PID)/snapshots/$(SNAP)/hybrid_join/$(POLICY)/join_hybrid_daily.csv" \
		--outdir "data_ai/$(PID)/snapshots/$(SNAP)/hybrid_join/$(POLICY)/plots"

# ---------------- Release & Reports ----------------
SNAP     ?= 2025-09-29
PID      ?= P000001
POLICY   ?= best_of_day
REL_TAG  ?= v0.1.0

.PHONY: weekly-report changelog release-pack release-all

weekly-report:
	. .venv/Scripts/activate && \
	python - <<'PY'
import pandas as pd, pathlib as P, json, sys
pid   = "$(PID)"
snap  = "$(SNAP)"
base  = P.Path(f"data_ai/{pid}/snapshots/{snap}")
feat  = base/"features_daily.csv"
join  = base/f"hybrid_join/$(POLICY)/join_hybrid_daily.csv"
qc    = base/"etl_qc_summary.csv"
outd  = P.Path("docs_build"); outd.mkdir(parents=True, exist_ok=True)
md    = outd/f"weekly_report_{pid}_{snap}.md"

def span(p):
    if not p.exists(): return ("–","–",0)
    df = pd.read_csv(p, parse_dates=["date"])
    return (str(df["date"].min().date()), str(df["date"].max().date()), len(df))

fmin,fmax,fn = span(feat)
jmin,jmax,jn = span(join)

qcrows = []
if qc.exists():
    q = pd.read_csv(qc)
    qcrows = [f"- {r['metric']}: {int(r.get('value', r.get('days_present',0)))}" for _,r in q.iterrows()]

md.write_text(f"""# Weekly Report — {pid} / {snap}

## Snapshot
- Features: **{fn}** linhas · período **{fmin} → {fmax}**
- Join (policy=`$(POLICY)`): **{jn}** linhas · período **{jmin} → {jmax}**

## Diretores/artefatos
- Features: `{feat.as_posix()}`
- Join: `{join.as_posix()}`
- Plots: `{(base/f'hybrid_join/$(POLICY)/plots').as_posix()}`

## QC (ETL Apple)
{chr(10).join(qcrows) if qcrows else "- (sem métricas adicionais)"}

## Observações
- Outliers de sono no Zepp (> 16h) aparecem como colunas verticais nos scatters.
- Política recomendada: **best_of_day** (para notebooks iniciais).

""", encoding="utf-8")
print(f"✅ weekly report → {md}")
PY

changelog:
	@mkdir -p docs_build
	@git log --pretty=format:"- %h %s (%ad)" --date=short > docs_build/CHANGELOG.md
	@echo "✅ changelog → docs_build/CHANGELOG.md"

release-pack:
	@mkdir -p dist
	@echo "# Practicum2 – Release $(REL_TAG)" > docs_build/RELEASE_NOTES.md
	@echo "- PID: $(PID)" >> docs_build/RELEASE_NOTES.md
	@echo "- SNAP: $(SNAP)" >> docs_build/RELEASE_NOTES.md
	@echo "- POLICY: $(POLICY)" >> docs_build/RELEASE_NOTES.md
	@tar -czf dist/practicum2_$(REL_TAG).tgz \
		--exclude='*.pyc' --exclude='__pycache__' \
		Makefile requirements_lock.txt \
		etl_pipeline.py etl_modules etl_tools \
		docs_build/CHANGELOG.md docs_build/RELEASE_NOTES.md \
		data_ai/$(PID)/snapshots/$(SNAP)/features_daily.csv \
		data_ai/$(PID)/snapshots/$(SNAP)/version_log_enriched.csv \
		data_ai/$(PID)/snapshots/$(SNAP)/hybrid_join/$(POLICY)/join_hybrid_daily.csv \
		data_ai/$(PID)/snapshots/$(SNAP)/hybrid_join/$(POLICY)/plots
	@echo "📦 dist/practicum2_$(REL_TAG).tgz"

release-all: weekly-report changelog release-pack
	@echo "✔ release bundle pronto. Sugestão:"
	@echo "  git add -A && git commit -m 'release $(REL_TAG)' && git tag $(REL_TAG)"
