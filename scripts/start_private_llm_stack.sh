#!/usr/bin/env bash
# Запуск стека на домашнем сервере (macOS/Linux):
# 1) LM Studio Server (если установлен lms)
# 2) Приватный HTTP API (python -m llm_service)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

if command -v lms >/dev/null 2>&1; then
  if ! curl -sf "${LLM_AGENT_LOCAL_BASE_URL:-http://127.0.0.1:1234/v1}/models" >/dev/null 2>&1; then
    echo "Запуск LM Studio Server (lms server start)..."
    lms server start &
    sleep 3
  fi
else
  echo "lms не найден — убедитесь, что inference доступен по LLM_AGENT_LOCAL_BASE_URL"
fi

export LLM_AGENT_BACKEND="${LLM_AGENT_BACKEND:-local}"
echo "Запуск приватного API на ${LLM_SERVICE_HOST:-0.0.0.0}:${LLM_SERVICE_PORT:-8080}"
exec python -m llm_service
