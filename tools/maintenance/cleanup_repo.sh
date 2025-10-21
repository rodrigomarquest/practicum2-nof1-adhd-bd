#!/usr/bin/env bash
set -euo pipefail

# Limpa arquivos gerados do índice (respeitando .gitignore)
echo "🧹 Untracking de diretórios e artefatos gerados…"

# Garante que os diretórios grandes estão ignorados
if ! grep -q '^data_ai/' .gitignore; then
  echo "data_ai/" >> .gitignore
fi
if ! grep -q '^data_etl/' .gitignore; then
  echo "data_etl/" >> .gitignore
fi
if ! grep -q '^notebooks/eda_outputs/' .gitignore; then
  echo "notebooks/eda_outputs/" >> .gitignore
fi

# Desindexa gerados
git rm -r --cached --ignore-unmatch data_ai || true
git rm -r --cached --ignore-unmatch data_etl || true
git rm -r --cached --ignore-unmatch notebooks/eda_outputs || true
git rm -r --cached --ignore-unmatch .ipynb_checkpoints || true
git rm -r --cached --ignore-unmatch '**/__pycache__' || true

# Opcional: desindexar tipos pesados remanescentes
git rm -r --cached --ignore-unmatch '*.csv' '*.parquet' '*.zip' '*.tar.gz' '*.npz' '*.npy' || true

git add .gitignore
git commit -m "chore(cleanup): untrack generated data & notebook outputs per .gitignore"

echo "✅ Limpeza concluída no índice. Se quiser purgar histórico, rode tools/maintenance/purge_history.sh."
