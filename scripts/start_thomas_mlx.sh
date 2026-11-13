#!/bin/zsh
set -euo pipefail

ROOT="/Users/local.local/Projects/TradingSystem"
VENV="$ROOT/.venv-thomas"
MODEL="${THOMAS_MLX_MODEL:-mlx-community/Qwen3-0.6B-4bit}"

cd "$ROOT"
exec "$VENV/bin/mlx_lm.server" \
  --model "$MODEL" \
  --host 127.0.0.1 \
  --port 8080 \
  --chat-template-args '{"enable_thinking":false}' \
  --log-level WARNING
