#!/usr/bin/env python3
"""Daily Donchian/EMA parameter search with a simple recent holdout."""
from __future__ import annotations

import argparse
import itertools
import json
from dataclasses import asdict
from datetime import time
from pathlib import Path

import numpy as np
import pandas as pd

from bbi_bull_switch_backtest import fetch_candles
from donchian_ema_10m_baseline_backtest import add_features, run
from donchian_ema_10m_exit_optimizer import metric


def evaluate(candles: pd.DataFrame, params: dict) -> dict:
    features = add_features(candles, params["lookback"], params["fast"], params["slow"])
    features.index = features.index.normalize() + pd.Timedelta(hours=10)
    trades, _, _ = run(features, params["tp"], params["stop"], 2.0, daily=True)
    rows = [{"return_pct": t.return_pct} for t in trades]
    return metric(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--start", default="2023-01-01")
    parser.add_argument("--end", default="2026-11-30")
    parser.add_argument("--output", default="daily_optimizer_results.json")
    args = parser.parse_args()
    candles = fetch_candles(args.symbol, args.start, args.end, 1440, False)
    split = candles.index.min() + (candles.index.max() - candles.index.min()) * 0.8
    train = candles.loc[candles.index < split]
    holdout = candles.loc[candles.index >= split]
    values = {
        "lookback": [20, 30, 45, 60, 90],
        "fast": [5, 8, 13],
        "slow": [40, 60, 89, 120, 150],
        "tp": [0.06, 0.08, 0.10, 0.12],
        "stop": [0.03, 0.0395, 0.05],
    }
    results = []
    for item in itertools.product(*values.values()):
        params = dict(zip(values, item))
        if params["fast"] >= params["slow"]:
            continue
        train_score = evaluate(train, params)
        if train_score["trades"] < 8:
            continue
        holdout_score = evaluate(holdout, params)
        full_score = evaluate(candles, params)
        results.append({"params": params, "full": full_score,
                        "train": train_score, "holdout": holdout_score})
    results.sort(key=lambda x: (x["full"]["compounded_pct"], x["full"]["max_drawdown_pct"]), reverse=True)
    Path(args.output).write_text(json.dumps(results, indent=2) + "\n")
    robust = [r for r in results if r["full"]["compounded_pct"] >= 51 and r["holdout"]["compounded_pct"] > 0]
    robust.sort(key=lambda x: (x["full"]["max_drawdown_pct"], x["full"]["compounded_pct"]), reverse=True)
    print(json.dumps({"top_full": results[:5], "lowest_drawdown_over_51": robust[:5],
                      "tested": len(results), "split": str(split.date())}, indent=2))


if __name__ == "__main__":
    main()
