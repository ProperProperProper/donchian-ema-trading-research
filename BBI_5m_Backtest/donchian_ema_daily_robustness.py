#!/usr/bin/env python3
"""Walk-forward robustness test for daily Donchian/EMA parameters."""
from __future__ import annotations

import argparse
import json
import random
from datetime import timedelta
from pathlib import Path

import pandas as pd

from bbi_bull_switch_backtest import fetch_candles
from donchian_ema_10m_baseline_backtest import add_features, run
from donchian_ema_10m_exit_optimizer import metric


def evaluate(features: pd.DataFrame, params: dict, start=None, end=None) -> dict:
    x = features
    if start is not None:
        x = x.loc[x.index >= start]
    if end is not None:
        x = x.loc[x.index < end]
    trades, _, _ = run(x, params["tp"], params["stop"], 2.0, daily=True)
    return metric([{"return_pct": t.return_pct} for t in trades])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--start", default="2022-01-01")
    parser.add_argument("--end", default="2026-11-30")
    parser.add_argument("--output", default="daily_robustness_results.json")
    parser.add_argument("--samples", type=int, default=300)
    parser.add_argument("--seed", type=int, default=20261112)
    args = parser.parse_args()
    candles = fetch_candles(args.symbol, args.start, args.end, 1440, False)
    date_index = candles.index.normalize() + pd.Timedelta(hours=10)
    candles.index = date_index
    end_date = candles.index.max()

    # Randomized, reproducible sampling avoids a rectangular grid's tendency
    # to over-test arbitrary round values while preserving sane ranges.
    rng = random.Random(args.seed)
    candidates = []
    seen = set()
    while len(candidates) < args.samples:
        params = {
            "lookback": rng.randint(35, 90),
            "fast": rng.randint(5, 16),
            "slow": rng.randint(45, 140),
            "tp": round(rng.uniform(0.06, 0.12), 4),
            "stop": round(rng.uniform(0.03, 0.06), 4),
        }
        key = tuple(params.values())
        if params["fast"] >= params["slow"] or key in seen:
            continue
        seen.add(key)
        candidates.append(params)
    folds = []
    cursor = candles.index.min() + timedelta(days=540)
    while cursor + timedelta(days=180) <= end_date:
        folds.append((cursor, cursor + timedelta(days=180)))
        cursor += timedelta(days=180)

    results = []
    for params in candidates:
        features = add_features(candles, params["lookback"], params["fast"], params["slow"])
        fold_scores = []
        for fold_start, fold_end in folds:
            score = evaluate(features, params, fold_start, fold_end)
            fold_scores.append({"start": str(fold_start.date()),
                                "end": str(fold_end.date()), "metrics": score})
        valid = [f["metrics"] for f in fold_scores if f["metrics"]["trades"] >= 3]
        if len(valid) < 3:
            continue
        full = evaluate(features, params)
        results.append({"params": params, "full": full, "folds": fold_scores,
                        "validation_median_pct": round(float(pd.Series(
                            [f["compounded_pct"] for f in valid]).median()), 3),
                        "validation_worst_dd_pct": round(float(min(
                            f["max_drawdown_pct"] for f in valid)), 3),
                        "validation_folds": len(valid)})

    # Stable validation return first, then less-negative worst drawdown, then
    # full-sample return. This deliberately rejects a single spectacular run.
    results.sort(key=lambda r: (r["validation_median_pct"],
                                r["validation_worst_dd_pct"],
                                r["full"]["compounded_pct"]), reverse=True)
    Path(args.output).write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps({"folds": len(folds), "tested": len(results),
                      "top_stable_candidates": results[:5]}, indent=2))


if __name__ == "__main__":
    main()
