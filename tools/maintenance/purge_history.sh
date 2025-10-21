#!/usr/bin/env bash
set -euo pipefail

echo "⚠️  Isto vai reescrever o histórico local (force-push será necessário)."
read -p "Continuar? (yes/N): " ans
[[ "${ans:-N}" == "yes" ]] || { echo "Abortado."; exit 1; }

# Branch de segurança
SAFE="safety-filter-$(date +%Y%m%d_%H%M)"
git branch "$SAFE" || true
echo "📦 Branch de segurança criado/sobreposto: $SAFE"

# Verifica git-filter-repo
if ! command -v git-filter-repo >/dev/null 2>&1; then
  echo "❌ git-filter-repo não encontrado. Instale:  pip install git-filter-repo"
  exit 1
fi

# Purga caminhos/padrões de dados e artefatos
git filter-repo --force \
  --path-glob 'data_ai/**' \
  --path-glob 'data_etl/**' \
  --path-glob 'decrypted_output/**' \
  --path-glob 'notebooks/**/eda_outputs/**' \
  --path-glob 'notebooks/**/data_*/**' \
  --path-glob '__pycache__/**' \
  --path-glob '.ipynb_checkpoints/**' \
  --invert-paths

echo
echo "✅ Histórico reescrito localmente."
echo "   Publique com:"
echo "     git push --force --all"
echo "     git push --force --tags"
echo "   Avise colaboradores para RE-CLONAR o repositório."