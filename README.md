# Moomoo Paper Trading Bot & Walk-Forward Quant Dashboard

Open-source Moomoo OpenD paper-trading system for US.APP with a local SQLite
market-data cache, 10-minute Donchian breakout and EMA trend strategy,
walk-forward-only parameter optimization, risk APIs, audit logs, and a browser
dashboard. Designed for research and paper competitions on macOS Apple Silicon.

**Keywords:** Moomoo paper trading bot, Moomoo OpenD API, quantitative trading,
algorithmic trading, APP stock, Donchian breakout, EMA strategy, walk-forward
optimization, SQLite market data, paper trading dashboard, Apple Silicon, Rust.

> Research and paper trading only. This project does not provide financial
> advice and does not guarantee performance. Live trading is intentionally not
> supported by the dashboard.

Research and backtesting code for a long-only intraday Donchian breakout strategy with an EMA trend filter.

## Highlights

- Moomoo OpenD candle ingestion with SQLite persistence.
- Strict walk-forward testing: parameters are selected only on prior data and
  scored only on the next unseen fold.
- Shared strategy, data, risk, audit, and execution interfaces in
  `quant_api.py`.
- Automatic Moomoo `TrdEnv.SIMULATE` paper orders with account-mode checks.
- Local dashboard for charts, trades, optimizer progress, risk decisions, and
  execution status.
- Apple-Silicon-oriented Rust control-plane work in `thomas-ai/`.

## 2 bps baseline

The current benchmark uses completed 10-minute candles and enters at the next candle open:

| Setting | Value |
|---|---:|
| Donchian lookback | 45 candles |
| Fast EMA | 8 |
| Slow EMA | 89 |
| Take profit | 6.38% |
| Stop loss | 3.95% |
| Slippage | 2 bps per side |
| Position | Long only, one position |
| Session handling | Flat before the close |

Saved US.APP results for the benchmark: 78 trades, 59.0% win rate, 44.01% sum of trade returns, 51.53% compounded return, and 11.10% maximum drawdown.

The 2 bps figure is an execution assumption, not a guarantee. Results depend on the data vendor, fills, commissions, liquidity, and date range.

## Files

- `BBI_5m_Backtest/donchian_ema_10m_baseline_backtest.py` — 10-minute baseline backtester.
- `BBI_5m_Backtest/donchian_ema_10m_research.py` — cost sensitivity, holdout, and walk-forward research harness.
- `BBI_5m_Backtest/bbi_bull_switch_backtest.py` — BBI bull-switch comparison backtester.
- `Moomoo_Quant_Donchian_EMA_10m.py` — preserved 6.38% TP Quant baseline.
- `Moomoo_Quant_Donchian_EMA_10m_8pctTP.py` — promoted 8% TP Quant candidate.
- `Moomoo_Quant_Donchian_EMA_10m_GreenYellowRed.py` — exit-only Green/Yellow/Red candidate.
- `EXIT_OPTIMIZATION_RESULTS.md` — frozen-entry exit optimization results.
- `BBI_5m_Backtest/US_APP_donchian_ema_10m_optimized.csv` — saved baseline trades.
- `BBI_5m_Backtest/US_APP_donchian_ema_research.json` — latest robustness report.
- `BBI_5m_Backtest/bhp_profit_lock_daily_backtest.py` — daily ASX BHP backtester for the exact profit-lock/yellow-exit strategy, including optimization.
- `BBI_5m_Backtest/local_quant_tester.py` — local Quant-style engine and self-contained HTML report with price/trade, equity, drawdown, monthly-return, and trade tables.
- `BBI_5m_Backtest/local_quant_portal.py` — local browser portal for selecting ASX 200 symbols, rerunning the optimizer, and viewing charts/stats.

Moomoo OpenD must be running on `127.0.0.1:11111` to fetch fresh candles. Do not commit API keys or credentials; use the macOS Keychain or environment variables locally.

## NVIDIA API models

`nvidia_api.py` provides a dependency-free chat client for NVIDIA's
OpenAI-compatible endpoint (`https://integrate.api.nvidia.com/v1`). By default it
reads the secret from the macOS Keychain service `NVIDIA_API_KEY` and uses
`deepseek-ai/deepseek-v4.1-flash`:

```sh
python3 nvidia_api.py "Summarize the current research results"
```

If the existing Keychain entry uses another service or account, set only the
metadata needed to locate it; the secret itself is never printed:

```sh
NVIDIA_KEYCHAIN_SERVICE="your-service" \\
NVIDIA_KEYCHAIN_ACCOUNT="your-account" \\
python3 nvidia_api.py "Summarize the current research results"
```

You can select another model with `--model` or `NVIDIA_MODEL`. The client is
separate from live trading and does not place orders.

## Thomas AI

The native Apple-Silicon AI control plane is in `thomas-ai/`. It uses Rust for
the gateway, routing, UI serving, CLI, NVIDIA Keychain access, and process
orchestration, with MLX-LM retained as the Python runtime boundary. It routes
ordinary prompts to local MLX and difficult prompts to the configured NVIDIA
API model. See [THOMAS_AI.md](THOMAS_AI.md).

## Local AI Harbor gateway

`local_ai_harbor.py` provides one OpenAI-compatible endpoint for local MLX
inference with optional NVIDIA NIM routing. See `LOCAL_AI_HARBOR.md` for setup.
Local MLX is the default; NVIDIA cloud fallback must be explicitly enabled.

## Research note

The original 2 bps baseline remains the profit benchmark. The research harness is for validation and should not automatically replace the baseline with a lower-return parameter set.

The baseline is immutable. Its locked Git tag is `baseline-2bps-6.38tp`; all exit experiments, including green/yellow/red state models, must use separate files and candidate versions.

The promoted 8% TP candidate tested at +53.86% compounded return with approximately -11.10% maximum drawdown over the full test period. It is preserved separately because its Jan–June split was weaker than the 6.38% baseline.

This is research software, not financial advice or a promise of future performance.

## Moomoo paper-competition bot

`competition_bot.py` runs the frozen 10-minute Donchian/EMA baseline against
Moomoo US paper data. It uses `TrdEnv.SIMULATE` only. The default mode is
dry-run; `--live-paper` is required before it will submit simulated orders.

Start Moomoo OpenD and log in, then identify the US paper account index:

```sh
python3 competition_bot.py --list-accounts
```

Run a signal-only check:

```sh
python3 competition_bot.py AAPL NVDA --acc-index 0
```

After selecting the competition account and confirming the symbols:

```sh
python3 competition_bot.py AAPL NVDA --acc-index 0 --live-paper
```

The bot writes `competition_bot.log` and `competition_bot_state.json` beside
the script. It never unlocks or submits to a live account, but verify that the
selected account row is `SIMULATE` before using `--live-paper`.

## Moomoo Quant Dashboard

`quant_dashboard.py` combines the tester, optimizer, charts, trade list, and
automatic paper execution controls in one local dashboard:

```sh
python3 quant_dashboard.py
open http://127.0.0.1:8780
```

The dashboard fetches `US.APP` candles directly from Moomoo OpenD with
`request_history_kline`, caches them in `quant_dashboard.sqlite3`, and uses
those cached candles for the tester. It stores each run, optimizer result,
trade, and paper execution event in SQLite. The dashboard reports return,
ending equity, maximum drawdown, trades, win rate, profit factor, average
trade, Sharpe approximation, equity curve, monthly returns, optimizer
rankings, and the complete trade list. The paper panel verifies the selected
OpenD account is `SIMULATE` before starting automatic order placement.

For an ASX BHP test, export daily candles from Moomoo Quant and run:

```sh
python3 BBI_5m_Backtest/bhp_profit_lock_daily_backtest.py \\
  --csv /path/to/BHP_daily.csv --optimize
```

The script does not alter Moomoo files or place orders.

To generate the local Quant-style report:

```sh
python3 BBI_5m_Backtest/local_quant_tester.py \\
  --csv /path/to/CDA_daily.csv --symbol CDA --optimize
open local_quant_report/report.html
```

For an interactive portal:

```sh
python3 BBI_5m_Backtest/local_quant_portal.py
open http://127.0.0.1:8765
```
