# Thomas AI services

Thomas AI is kept running by two per-user macOS launchd agents:

- `com.thomas-ai.mlx` — MLX-LM local server on `127.0.0.1:8080`.
- `com.thomas-ai.gateway` — Rust Thomas AI gateway on `127.0.0.1:8788`.

Both start at login and restart after an unexpected exit. Install or reload
them with:

```sh
./scripts/install_thomas_services.sh
```

Check them:

```sh
launchctl list | rg 'com.thomas-ai'
curl http://127.0.0.1:8080/v1/models
curl http://127.0.0.1:8788/health
open http://127.0.0.1:8788
```

Logs are written to `logs/thomas-mlx.log`, `logs/thomas-mlx.err`,
`logs/thomas-ai.log`, and `logs/thomas-ai.err`.

The default MLX model is `mlx-community/Qwen3-0.6B-4bit`. Set
`THOMAS_MLX_MODEL` in `scripts/start_thomas_mlx.sh` or the launch agent
environment when a larger model is appropriate. EXO/RDMA remains conditional
on a configured multi-Mac cluster and is not started as a fake single-node
service.
