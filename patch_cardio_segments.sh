#!/usr/bin/env bash
set -euo pipefail

FILE="etl_modules/cardiovascular/cardio_etl.py"
cp -f "$FILE" "$FILE.bak.$(date +%Y%m%d%H%M%S)"

python make_scripts/patch_cardio_segments.py

echo "Done."
