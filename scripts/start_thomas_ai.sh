#!/bin/zsh
set -euo pipefail

ROOT="/Users/local.local/Projects/TradingSystem"
export PATH="/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin"
export NVIDIA_KEYCHAIN_SERVICE="${NVIDIA_KEYCHAIN_SERVICE:-NVIDIA_API_KEY}"
export NVIDIA_KEYCHAIN_ACCOUNT="${NVIDIA_KEYCHAIN_ACCOUNT:-local.local}"
export NVIDIA_MODEL="${NVIDIA_MODEL:-nvidia/nemotron-3.5-lightning-30b-a3b}"
export NVIDIA_FAST_MODEL="${NVIDIA_FAST_MODEL:-nvidia/nemotron-3.5-bolt-e4b}"
export NVIDIA_HARD_MODEL="${NVIDIA_HARD_MODEL:-deepseek-ai/deepseek-v4.1-flash}"
export NVIDIA_FALLBACK_MODEL="${NVIDIA_FALLBACK_MODEL:-nvidia/nemotron-3.5-bolt-e4b}"
export NVIDIA_REQUEST_TIMEOUT="${NVIDIA_REQUEST_TIMEOUT:-45}"
export THOMAS_TOOL_TIMEOUT="${THOMAS_TOOL_TIMEOUT:-120}"
export AI_PROVIDER="${AI_PROVIDER:-nvidia}"
export AI_LOCAL_MAX_DIFFICULTY="${AI_LOCAL_MAX_DIFFICULTY:-3}"
export MLX_MODEL="${MLX_MODEL:-mlx-community/Qwen3-0.6B-4bit}"
export THOMAS_ACCESS_MODE="full"
export THOMAS_REPO_ROOT="$ROOT"

cd "$ROOT/thomas-ai"
exec "$ROOT/thomas-ai/target/release/thomas-ai" serve
