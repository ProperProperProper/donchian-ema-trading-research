#!/usr/bin/env python3
"""Daily-candle comparison using the frozen Donchian/EMA entry logic."""
from __future__ import annotations

import argparse
import json
from datetime import time
from pathlib import Path

import numpy as np
import pandas as pd

from bbi_bull_switch_backtest import fetch_candles
from donchian_ema_10m_baseline_backtest import add_features, run
from donchian_ema_10m_exit_optimizer import metric


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--start", default="2023-01-01")
    parser.add_argument("--end", default="2026-11-30")
    parser.add_argument("--output", default="daily_backtest_results.json")
    args = parser.parse_args()
    candles = fetch_candles(args.symbol, args.start, args.end, 1440, False)
    features = add_features(candles, 45, 8, 89)
    # Reuse the live-style next-bar engine while representing each daily bar
    # as a completed regular-session bar; no intraday close rule is applied.
    features.index = features.index.normalize() + pd.Timedelta(hours=10)
    trades, signals, _ = run(features, 0.08, 0.0395, 2.0, daily=True)
    rows = [{"return_pct": t.return_pct, "reason": t.reason,
             "entry_time": t.entry_time, "exit_time": t.exit_time} for t in trades]
    result = {"symbol": args.symbol, "interval": "1D", "bars": len(features),
              "signals": len(signals), "metrics": metric(rows),
              "entry": {"lookback": 45, "fast_ema": 8, "slow_ema": 89},
              "exit": {"take_profit": 0.08, "initial_stop": 0.0395}}
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
