#!/usr/bin/env bash
set -e
ZDIR="data_etl/P000001/zepp_processed/2025-09-28"
python etl_pipeline.py cardio \
  --participant P000001 \
  --snapshot 2025-09-29 \
  --zepp_dir "$ZDIR"
