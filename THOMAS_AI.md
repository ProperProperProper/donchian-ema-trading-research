# Thomas AI

Thomas AI is the native Rust control plane for this TradingSystem's private AI
stack. It is optimized for Apple Silicon and keeps Python only where Apple's
MLX-LM runtime is the practical integration boundary.

## Build

```sh
cd thomas-ai
cargo build --release --offline
```

## Run

Start MLX-LM on the host:

```sh
./target/release/thomas-ai mlx-start --model mlx-community/Qwen3-8B-4bit
```

Then start Thomas AI in another terminal:

```sh
./target/release/thomas-ai serve
open http://127.0.0.1:8788
```

The gateway routes by difficulty and current in-flight load. NVIDIA is the
direct UI default, with Nemotron Lightning as the default free endpoint,
Nemotron Bolt for easy work, and DeepSeek V4.1 Flash for difficult work.
Set `NVIDIA_API_KEY` or store a secret in
the macOS Keychain under service `NVIDIA_API_KEY` and account `local.local`.
Configure a different NVIDIA free
endpoint model with `NVIDIA_MODEL`.

## CLI

```sh
./target/release/thomas-ai status
./target/release/thomas-ai models
./target/release/thomas-ai chat "summarize the latest backtest"
./target/release/thomas-ai chat --difficulty 8 "debug this strategy architecture"
./target/release/thomas-ai cluster status
./target/release/thomas-ai exec -- git status --short --branch
./target/release/thomas-ai exec -- cargo test --offline
```

The NVIDIA planner uses the configured free NVIDIA pool. NVIDIA requests have
bounded timeouts, endpoint health checks, capped tool output, and retry through
the configured free fallback model when the selected model fails. Direct local
commands use `thomas-ai exec` and do not wait for an NVIDIA response.

## Apple Silicon cluster

`EXO_BASE_URL` identifies an EXO-compatible cluster endpoint. RDMA support is
designed around MLX JACCL over Thunderbolt 5 on compatible Macs and macOS
26.2+. Thomas AI reports this cluster path in the UI and status API; the next
cluster milestone is automated JACCL setup and measured link verification.

Thomas AI never places trades. It can explain research and logs, but trading
execution remains in the existing separately controlled Moomoo components.
