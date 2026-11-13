#!/usr/bin/env python3
"""Randomized all-parameter daily optimizer for one recent year."""
from __future__ import annotations

import argparse
import json
import random
import warnings
from pathlib import Path

import pandas as pd

from bbi_bull_switch_backtest import fetch_candles
from donchian_ema_10m_baseline_backtest import add_features, run
from donchian_ema_10m_exit_optimizer import ExitParams, metric, run_exits

warnings.filterwarnings("ignore", category=FutureWarning)


def score_fixed(features, p):
    t, _, _ = run(features, p["tp"], p["stop"], 2.0, daily=True)
    return metric([{"return_pct": z.return_pct} for z in t])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--end", default="2026-11-12")
    parser.add_argument("--days", type=int, default=365)
    parser.add_argument("--samples", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=20261112)
    parser.add_argument("--output", default="daily_one_year_all_optimizer.json")
    args = parser.parse_args()
    end = pd.Timestamp(args.end)
    start = end - pd.Timedelta(days=args.days)
    candles = fetch_candles(args.symbol, "2023-01-01", args.end, 1440, False)
    rng = random.Random(args.seed)
    results = []
    feature_cache = {}

    for _ in range(args.samples):
        lookback = rng.randint(5, 120)
        fast = rng.randint(2, 30)
        slow = rng.randint(max(fast + 1, 10), 200)
        key = (lookback, fast, slow)
        if key not in feature_cache:
            x = add_features(candles, lookback, fast, slow)
            x.index = x.index.normalize() + pd.Timedelta(hours=10)
            feature_cache[key] = x.loc[x.index >= start]
        features = feature_cache[key]

        if rng.random() < 0.55:
            p = {"family": "fixed", "lookback": lookback, "fast": fast, "slow": slow,
                 "tp": round(rng.uniform(0.02, 0.50), 4),
                 "stop": round(rng.uniform(0.01, 0.25), 4)}
            metrics = score_fixed(features, p)
        else:
            p = {"family": "atr_trailing", "lookback": lookback, "fast": fast, "slow": slow,
                 "atr_period": rng.randint(5, 40), "atr_multiple": round(rng.uniform(2, 15), 2),
                 "red_mode": rng.choice(["none", "ema_slow"]),
                 "time_cap_bars": rng.choice([0, 20, 40, 60, 80, 120]),
                 "time_cap_loss_only": True}
            ep = ExitParams(tp=1.0, initial_stop=0.99,
                            atr_period=p["atr_period"], atr_multiple=p["atr_multiple"],
                            red_mode=p["red_mode"], time_cap_bars=p["time_cap_bars"],
                            time_cap_loss_only=True)
            metrics = metric(run_exits(features, ep, daily=True))
        results.append({"params": p, "metrics": metrics})

    results.sort(key=lambda r: (r["metrics"]["compounded_pct"],
                                r["metrics"]["max_drawdown_pct"]), reverse=True)
    eligible = [r for r in results if r["metrics"]["trades"] >= 3]
    Path(args.output).write_text(json.dumps({
        "window_start": str(start.date()), "window_end": str(end.date()),
        "samples": args.samples, "results": results,
    }, indent=2) + "\n")
    print(json.dumps({"window": [str(start.date()), str(end.date())],
                      "top_all": results[:10], "top_3plus_trades": eligible[:10]}, indent=2))


if __name__ == "__main__":
    main()
