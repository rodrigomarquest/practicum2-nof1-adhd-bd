#!/usr/bin/env bash
# ======================================================
# install_cardio_min.sh
# Instala implementação funcional mínima do ETL Cardiovascular
# (Apple + Zepp loaders, _join e features). Faz backup se o
# arquivo já existir: cria *.bak.YYYYmmddHHMMSS
# ======================================================
set -euo pipefail

#!/usr/bin/env bash
# ======================================================
# install_cardio_min.sh
# Instala implementação funcional mínima do ETL Cardiovascular
# (Apple + Zepp loaders, _join e features). Faz backup se o
# arquivo já existir: cria *.bak.YYYYmmddHHMMSS
# ======================================================
set -euo pipefail

ts="$(date +%Y%m%d%H%M%S)"

backup_and_write() {
  local file="$1"; shift
  local dir; dir="$(dirname "$file")"
  mkdir -p "$dir"
  if [ -f "$file" ]; then
    cp -f "$file" "$file.bak.$ts"
    echo "BACKUP: $file -> $file.bak.$ts"
  fi
  # This helper expects content on stdin when used. It's kept for
  # backward compatibility but is not used by the minimal installer below.
  cat > "$file"
  echo "WROTE:  $file"
}

# -----------------------------
# apple/loader.py
# -----------------------------
python make_scripts/apple/create_cardio_apple_loader.py

# -----------------------------
# zepp/loader.py
# -----------------------------
python make_scripts/zepp/create_cardio_zepp_loader.py

# -----------------------------
# _join/join.py
# -----------------------------
python make_scripts/create_cardio_join.py

# -----------------------------
# cardio_features.py
# -----------------------------
python make_scripts/create_cardio_features.py

echo "✅ Cardiovascular minimal implementation installed."
echo "→ Próximo: rode o stage cardio no seu snapshot:"
echo "   python etl_pipeline.py --stage cardio --input data_ai/P000001/snapshots/2025-09-29 --out data_ai/P000001/snapshots/2025-09-29"
