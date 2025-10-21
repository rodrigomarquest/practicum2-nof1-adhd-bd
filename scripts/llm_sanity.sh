#!/usr/bin/env bash
set -euo pipefail

OLLAMA_HOST="${1:-http://localhost:11434}"
CHAT_MODEL="${2:-llama3.1:8b}"
PYTHON_BIN="${3:-python}"

echo "🔍 Verifying local Ollama connectivity at ${OLLAMA_HOST} with model ${CHAT_MODEL}..."

# Ping server
if ! curl -s "${OLLAMA_HOST}/api/tags" >/dev/null; then
  echo "❌ Ollama server not reachable at ${OLLAMA_HOST}"
  exit 1
fi

# Simple chat generate
RESP="$(curl -s -X POST "${OLLAMA_HOST}/api/generate" \
  -H "Content-Type: application/json" \
  -d "{ \"model\": \"${CHAT_MODEL}\", \"prompt\": \"say OK\", \"stream\": false }")"

# Print small verdict using Python (avoids jq dependency)
echo "$RESP" | "$PYTHON_BIN" - <<'PY'
import sys, json
raw = sys.stdin.read()
try:
    j = json.loads(raw)
    text = j.get("response","") or raw[:120]
except Exception:
    text = raw[:120]
print("✅ Ollama connected OK" if "OK" in text else f"⚠️ Response: {text[:120]}")
PY
