#!/usr/bin/env bash
set -euo pipefail

# 1) Backup e rename do pipeline atual → etl_pipeline_legacy.py (se ainda não fizemos)
if [ -f "etl_pipeline.py" ] && [ ! -f "etl_pipeline_legacy.py" ]; then
  cp -f etl_pipeline.py "etl_pipeline_legacy.py"
  echo "BACKUP: etl_pipeline.py -> etl_pipeline_legacy.py"
fi

mkdir -p etl_modules
python make_scripts/create_apple_raw_to_per_metric.py

python make_scripts/install_etl_pipeline.py

echo "✅ etl_pipeline.py (v2) instalado com subcomandos."
echo "   - apple-per-metric  |  - cardio  |  - legacy-apple"
