#!/usr/bin/env bash
set -euo pipefail

# Regrava os 4 arquivos do domínio cardiovascular com conteúdo correto.
# Faz backup .bak.TIMESTAMP se existir.

ts="$(date +%Y%m%d%H%M%S)"
backup() { [ -f "$1" ] && cp -f "$1" "$1.bak.$ts" && echo "BACKUP: $1 -> $1.bak.$ts" || true; }

python make_scripts/create_cardio_join.py

python make_scripts/create_cardio_apple_loader.py

python make_scripts/create_cardio_zepp_loader.py

python make_scripts/create_cardio_features.py

echo "✅ Cardio files fixed."

echo "✅ Cardio files fixed."
