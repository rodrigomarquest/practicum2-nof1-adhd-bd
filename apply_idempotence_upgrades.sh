#!/usr/bin/env bash
set -euo pipefail

TS="$(date +%Y%m%d%H%M%S)"

backup() {
  local f="$1"
  if [ -f "$f" ]; then
    cp -f "$f" "$f.bak.$TS"
    echo "BACKUP: $f -> $f.bak.$TS"
  fi
}

# Files we intend to patch (back up them before running any script)
TARGETS=(
  "etl_pipeline.py"
  "etl_modules/cardiovascular/cardio_features.py"
  "etl_modules/cardiovascular/zepp/loader.py"
)

for f in "${TARGETS[@]}"; do
  backup "$f" || true
done

SCRIPTS=(
  "make_scripts/patch_etl_pipeline_helpers.py"
  "make_scripts/patch_cardio_features.py"
  "make_scripts/patch_zepp_loader.py"
)

for s in "${SCRIPTS[@]}"; do
  if [ -f "$s" ]; then
    echo "RUN: python $s"
    if ! python "$s"; then
      echo "ERROR: script failed: $s"
      exit 1
    fi
  else
    echo "SKIP: $s not found"
  fi
done

echo "✅ All patches applied."
