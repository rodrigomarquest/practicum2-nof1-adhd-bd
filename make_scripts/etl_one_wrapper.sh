#!/usr/bin/env bash
# Wrapper used by Makefile etl-one target.
# Usage: etl_one_wrapper.sh <PID> <SNAP> [<VENV_PY> [<ETL_PIPELINE>]]
set -euo pipefail
PID="$1"
SNAP="$2"
VENV_PY="${3:-python}"
ETL_PIPELINE="${4:-etl_pipeline.py}"

# Allow environment overrides for consolidated layout
RAW_DIR_ENV="${RAW_DIR:-data/raw}"
ETL_DIR_ENV="${ETL_DIR:-data/etl}"

# normalize snap forms

# normalize snap forms
snap_hyphen="$SNAP"
snap_nodash="${snap_hyphen//-/}"

def_candidates=(
  "${ETL_DIR_ENV}/${PID}/snapshots/${snap_hyphen}/export.xml"
  "${ETL_DIR_ENV}/${PID}/snapshots/${snap_nodash}/export.xml"
  "${ETL_DIR_ENV}/${PID}/${snap_hyphen}/export.xml"
  "${ETL_DIR_ENV}/${PID}/${snap_nodash}/export.xml"
  # refactored/extracted layout under ETL_DIR with participant subfolder
  "${ETL_DIR_ENV}/${PID}/extracted/apple/export.xml"
  "${ETL_DIR_ENV}/${PID}/extracted/export.xml"
  # consolidated layout under RAW_DIR
  "${RAW_DIR_ENV}/${PID}/apple/export.zip"
  "${RAW_DIR_ENV}/${PID}/apple/export.xml"
  # legacy fallback
  "data/etl/${PID}/extracted/apple/export.xml"
  "data/etl/${PID}/extracted/export.xml"
)

found=""
for p in "${def_candidates[@]}"; do
  if [ -f "$p" ]; then
    found="$p"
    break
  fi
done

if [ -z "$found" ]; then
  echo "ℹ️ export.xml not found for: ${PID} ${SNAP} — skipping etl-one (no error)"
  exit 0
fi

# If present, run the extract subcommand
echo "Running: PYTHONPATH=\"$PWD\" \"${VENV_PY}\" ${ETL_PIPELINE} extract --participant \"${PID}\" --snapshot \"${SNAP}\" --cutover \"${CUTOVER:-}\" --tz_before \"${TZ_BEFORE:-}\" --tz_after \"${TZ_AFTER:-}\""
PYTHONPATH="$PWD" "${VENV_PY}" ${ETL_PIPELINE} extract --participant "${PID}" --snapshot "${SNAP}" --cutover "${CUTOVER:-}" --tz_before "${TZ_BEFORE:-}" --tz_after "${TZ_AFTER:-}"
