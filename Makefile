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
> $(PYTHON) -m src.etl_pipeline

# Adjust FEATURES_PATH/PID/SNAPSHOT via env or config/settings.yaml as needed.
labels:
> echo ">>> labels: running src.make_labels"
> $(PYTHON) -m src.make_labels --rules config/label_rules.yaml --in data/etl/FEATURES_PATH/features_daily.csv --out data/etl/FEATURES_PATH/features_daily_labeled.csv

qc:
> echo ">>> qc: running src.eda"
> $(PYTHON) -m src.eda

# Minimal pack-kaggle: bundles features + version log. Edit PID/SNAPSHOT if needed.
# Reads from data/etl/P000001/snapshots/2025-09-29/joined by default.
PID ?= P000001
SNAPSHOT ?= 2025-09-29
PACK_OUT := dist/assets

pack-kaggle:
> echo ">>> pack-kaggle: packaging Kaggle dataset snapshot for $(PID) $(SNAPSHOT)"
> $(PYTHON) - <<'PY'
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
