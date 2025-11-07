#!/usr/bin/env bash
set -euo pipefail

echo "🧹 Starting repository cleanup..."

# 1. Remove legacy code
echo "📁 Removing legacy code..."
find src/ -type f \( -name "*backup*" -o -name "*old*" -o -name "*_v[0-9]*" \) -delete
find src/ -type d -name "legacy" -exec rm -rf {} + 2>/dev/null || true

# 2. Clean notebooks outputs
echo "📓 Cleaning notebook outputs..."
find notebooks/ -name "*.ipynb" -exec jupyter nbconvert --clear-output --inplace {} \; 2>/dev/null || true
rm -rf notebooks/outputs/* 2>/dev/null || true

# 3. Remove build artifacts
echo "🔨 Removing build artifacts..."
make clean-provenance
rm -rf dist/assets/* dist/changelog/* 2>/dev/null || true

# 4. Clean Python cache
echo "🐍 Cleaning Python cache..."
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find . -type f -name "*.pyc" -delete
find . -type f -name "*.pyo" -delete
rm -rf .pytest_cache/ .coverage htmlcov/ 2>/dev/null || true

# 5. Clean IDE artifacts
echo "💻 Cleaning IDE artifacts..."
rm -rf .idea/ 2>/dev/null || true
find . -name ".DS_Store" -delete 2>/dev/null || true
find . -name "Thumbs.db" -delete 2>/dev/null || true

# 6. Verify .gitignore
echo "🔒 Verifying sensitive paths are ignored..."
git check-ignore data/raw data/etl data/ai .env || {
    echo "⚠️  WARNING: Sensitive paths not in .gitignore!"
    echo "Run: git rm -r --cached data/raw data/etl data/ai .env"
}

# 7. List large files (potential data leaks)
echo "📊 Checking for large files..."
git ls-files | xargs du -sh 2>/dev/null | sort -rh | head -20

echo "✅ Cleanup complete!"
echo ""
echo "📋 Next steps:"
echo "  1. Review git status"
echo "  2. Commit cleanup: git commit -am 'chore: repository cleanup'"
echo "  3. Verify no sensitive data: git log --all -- data/"