#!/usr/bin/env bash
set -euo pipefail

# 1) branch de segurança
git branch safety-cleanup-$(date +%Y%m%d_%H%M) || true

# 2) garantir .gitignore atualizado (assuma que você já substituiu o arquivo)
git add .gitignore
git commit -m "chore(gitignore): tighten ignores for data_ai, data_etl, notebooks outputs" || true

# 3) parar de trackear artefatos já versionados
git rm -r --cached --ignore-unmatch \
  data_ai \
  data_etl \
  decrypted_output \
  ios_extract \
  notebooks/**/eda_outputs \
  notebooks/**/data_* \
  notebooks/**/figures \
  __pycache__ \
  .ipynb_checkpoints

# 4) placeholders para manter estrutura
mkdir -p data_ai data_etl notebooks/eda_outputs
echo "# ignored output" > data_ai/.keep
echo "# ignored output" > data_etl/.keep
echo "# ignored output" > notebooks/eda_outputs/.keep
git add data_ai/.keep data_etl/.keep notebooks/eda_outputs/.keep || true

# 5) commit de limpeza
git add -A
git commit -m "chore(cleanup): untrack generated data & notebook outputs per .gitignore"

echo
echo "✅ Limpeza rápida feita (sem reescrever histórico)."
echo "   Revise 'git status' e 'git log -1', depois:"
echo "   git push origin main"
