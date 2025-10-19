#!/usr/bin/env bash
set -e
python etl_pipeline.py extract \
  --participant P000001 \
  --snapshot 2025-09-29 \
  --cutover 2024-03-11 \
  --tz_before America/Sao_Paulo \
  --tz_after Europe/Dublin
