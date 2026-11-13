# Local AI Harbor

This project now includes a small Harbor-like gateway for the trading system.
It gives local MLX inference one stable OpenAI-compatible endpoint and supports
NVIDIA's hosted NIM API as an explicit provider or guarded fallback.

## 1. Install MLX-LM

On an Apple-silicon Mac:

```sh
python3 -m pip install mlx-lm
```

Start a local model. Replace the model with any compatible MLX/Hugging Face
model you want to use:

```sh
python3 -m mlx_lm.server \
  --model mlx-community/Qwen3-8B-4bit \
  --port 8080
```

The MLX server exposes an OpenAI-compatible API on `http://127.0.0.1:8080/v1`.

## 2. Start the gateway

In another terminal:

```sh
python3 local_ai_harbor.py
```

The gateway listens on `http://127.0.0.1:8788` and automatically routes by
request difficulty: ordinary requests use MLX, while harder requests use the
stronger NVIDIA model. If a selected backend is unavailable, it fails over to
the other configured backend.

## 3. Test local inference

```sh
curl http://127.0.0.1:8788/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"local","messages":[{"role":"user","content":"Summarize the trading system in one sentence."}]}'
```

## NVIDIA free API provider

NVIDIA's hosted API is OpenAI-compatible at
`https://integrate.api.nvidia.com/v1`. The existing `nvidia_api.py` client can
read `NVIDIA_API_KEY` from the environment or the macOS Keychain. NVIDIA's
API Catalog marks eligible hosted models as free endpoints; the gateway keeps
the model configurable through `NVIDIA_MODEL` and does not hard-code a model
that may later leave the free tier.

Use NVIDIA explicitly for one request:

```sh
curl http://127.0.0.1:8788/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -H 'X-AI-Provider: nvidia' \
  -d '{"model":"deepseek-ai/deepseek-v4.1-flash","messages":[{"role":"user","content":"Explain this backtest result."}]}'
```

Or run the gateway with NVIDIA as the default:

```sh
python3 local_ai_harbor.py --provider nvidia
```

Automatic difficulty routing is enabled with:

```sh
AI_PROVIDER=auto AI_LOCAL_MAX_DIFFICULTY=3 python3 local_ai_harbor.py
```

Scores at or below `AI_LOCAL_MAX_DIFFICULTY` stay local. Higher scores go to
NVIDIA. Set `AI_LOCAL_MAX_DIFFICULTY=1` to use NVIDIA for nearly all but trivial
prompts. Requests routed to NVIDIA leave the Mac, so use `--provider local`
when data must remain strictly on-device.

## Current scope

This first version supports non-streaming chat completions, health checks, and
model discovery. It intentionally does not place orders or change any Moomoo
strategy. The next useful layer is a trading-aware assistant endpoint that
reads research results and logs, then exposes explanations and diagnostics to
the existing dashboard.
