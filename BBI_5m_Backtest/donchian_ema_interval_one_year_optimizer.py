#!/usr/bin/env python3
"""Optimize all core parameters separately for each candle interval."""
from __future__ import annotations

import argparse
import json
import random
import warnings
from pathlib import Path

import pandas as pd

from bbi_bull_switch_backtest import fetch_candles
from donchian_ema_10m_baseline_backtest import add_features, run
from donchian_ema_10m_exit_optimizer import metric

warnings.filterwarnings("ignore", category=FutureWarning)


def evaluate(features, params, daily, start):
    x = features.copy()
    if daily:
        x.index = x.index.normalize() + pd.Timedelta(hours=10)
        x = x.loc[x.index >= start]
    trades, _, _ = run(x, params["tp"], params["stop"], 2.0, daily=daily)
    if not daily:
        trades = [t for t in trades if pd.Timestamp(t.entry_time) >= start]
    return metric([{"return_pct": t.return_pct} for t in trades])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--end", default="2026-11-12")
    parser.add_argument("--days", type=int, default=365)
    parser.add_argument("--samples", type=int, default=250)
    parser.add_argument("--seed", type=int, default=20261112)
    parser.add_argument("--output", default="interval_one_year_results.json")
    args = parser.parse_args()
    end = pd.Timestamp(args.end)
    start = end - pd.Timedelta(days=args.days)
    rng = random.Random(args.seed)
    all_results = []

    for interval in (5, 10, 15, 30, 1440):
        daily = interval == 1440
        candles = fetch_candles(args.symbol, "2025-01-01", args.end, interval, False)
        feature_cache = {}
        results = []
        for _ in range(args.samples):
            params = {
                "lookback": rng.randint(5, 120),
                "fast": rng.randint(2, 30),
                "slow": rng.randint(10, 200),
                "tp": round(rng.uniform(0.02, 0.50), 4),
                "stop": round(rng.uniform(0.01, 0.25), 4),
            }
            if params["fast"] >= params["slow"]:
                continue
            key = (params["lookback"], params["fast"], params["slow"])
            # Cache feature frames because each interval search reuses many
            # entry combinations while only TP/SL changes.
            if key not in feature_cache:
                feature_cache[key] = add_features(candles, *key)
            score = evaluate(feature_cache[key], params, daily, start)
            results.append({"interval_min": interval, "params": params, "metrics": score})
        results.sort(key=lambda x: (x["metrics"]["compounded_pct"],
                                    x["metrics"]["max_drawdown_pct"]), reverse=True)
        all_results.extend(results)
        print(json.dumps({"interval_min": interval, "top_all": results[:3],
                          "top_10plus_trades": [r for r in results if r["metrics"]["trades"] >= 10][:3]}))

    Path(args.output).write_text(json.dumps({"window_start": str(start.date()),
                                              "window_end": str(end.date()),
                                              "results": all_results}, indent=2) + "\n")


if __name__ == "__main__":
    main()
