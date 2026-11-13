#!/usr/bin/env python3
"""Robustness research for the 10-minute Donchian/EMA strategy.

This deliberately keeps the production backtest unchanged and adds:
  * rolling walk-forward parameter selection;
  * slippage/cost sensitivity;
  * risk-adjusted metrics and a recent contiguous holdout.

All signals still use completed candles and entries occur on the next open.
"""

from __future__ import annotations

import argparse
import itertools
import json
import warnings
from dataclasses import asdict
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from donchian_ema_10m_baseline_backtest import add_features, fetch_candles, run

warnings.filterwarnings("ignore", category=FutureWarning)


BASELINE = {
    "lookback": 45,
    "fast": 8,
    "slow": 89,
    "tp": 0.0638,
    "stop": 0.0395,
}


def metrics(trades) -> dict:
    returns = np.array([trade.return_pct / 100.0 for trade in trades], dtype=float)
    if not len(returns):
        return {"trades": 0, "win_rate_pct": 0.0, "sum_return_pct": 0.0,
                "compounded_pct": 0.0, "max_drawdown_pct": 0.0,
                "profit_factor": 0.0, "avg_trade_pct": 0.0}
    equity = np.cumprod(1.0 + returns)
    drawdown = equity / np.maximum.accumulate(equity) - 1.0
    gains = returns[returns > 0].sum()
    losses = -returns[returns < 0].sum()
    return {
        "trades": int(len(returns)),
        "win_rate_pct": round(float((returns > 0).mean() * 100), 3),
        "sum_return_pct": round(float(returns.sum() * 100), 3),
        "compounded_pct": round(float((equity[-1] - 1) * 100), 3),
        "max_drawdown_pct": round(float(drawdown.min() * 100), 3),
        "profit_factor": round(float(gains / losses), 3) if losses else float("inf"),
        "avg_trade_pct": round(float(returns.mean() * 100), 4),
    }


def evaluate(candles: pd.DataFrame, params: dict, slippage_bps: float):
    features = add_features(candles, params["lookback"], params["fast"], params["slow"])
    trades, _, _ = run(features, params["tp"], params["stop"], slippage_bps)
    return trades, metrics(trades)


def parameter_grid() -> list[dict]:
    # Focused +/-10% neighbourhood prevents a brute-force search from selecting noise.
    values = {
        "lookback": [40, 50],
        "fast": [7, 9],
        "slow": [80, 98],
        "tp": [0.060, 0.068],
        "stop": [0.037, 0.042],
    }
    grid = [BASELINE]
    grid.extend(dict(zip(values, item)) for item in itertools.product(*values.values()))
    return [params for params in grid if params["fast"] < params["slow"]]


def walk_forward(candles: pd.DataFrame, train_days: int, test_days: int,
                 slippage_bps: float, step_days: int = 20) -> list[dict]:
    start = candles.index.min().normalize()
    end = candles.index.max().normalize()
    rows = []
    cursor = start + timedelta(days=train_days)
    grid = parameter_grid()
    while cursor + timedelta(days=test_days) <= end:
        train_start = cursor - timedelta(days=train_days)
        test_end = cursor + timedelta(days=test_days)
        train = candles.loc[(candles.index >= train_start) & (candles.index < cursor)]
        test = candles.loc[(candles.index >= cursor) & (candles.index < test_end)]
        if train.empty or test.empty:
            cursor += timedelta(days=step_days)
            continue
        candidates = []
        for params in grid:
            _, score = evaluate(train, params, slippage_bps)
            if score["trades"] >= 5:
                candidates.append((score["compounded_pct"], params, score))
        if not candidates:
            cursor += timedelta(days=test_days)
            continue
        _, best_params, train_score = max(candidates, key=lambda item: item[0])
        _, test_score = evaluate(test, best_params, slippage_bps)
        rows.append({"train_start": train_start.date().isoformat(),
                     "train_end": (cursor - timedelta(days=1)).date().isoformat(),
                     "test_start": cursor.date().isoformat(),
                     "test_end": (test_end - timedelta(days=1)).date().isoformat(),
                     "params": best_params, "train": train_score, "test": test_score})
        cursor += timedelta(days=step_days)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Robustness research for Donchian/EMA")
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--start", default="2026-01-01")
    parser.add_argument("--end", default="2026-11-30")
    parser.add_argument("--interval", type=int, choices=[5, 10, 15, 30], default=10)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    candles = fetch_candles(args.symbol, args.start, args.end, args.interval, False)
    candles = candles.loc[(candles.index.time >= pd.Timestamp("09:30").time()) &
                          (candles.index.time <= pd.Timestamp("16:00").time())]

    baseline = {}
    for cost in (0.0, 2.0, 4.0, 6.0, 8.0, 10.0):
        _, baseline[str(cost)] = evaluate(candles, BASELINE, cost)

    span = candles.index.max() - candles.index.min()
    holdout_start = candles.index.min() + span * 0.8
    _, holdout = evaluate(candles.loc[candles.index >= holdout_start], BASELINE, 2.0)
    result = {
        "symbol": args.symbol, "interval_minutes": args.interval,
        "baseline": BASELINE, "cost_sensitivity": baseline,
        "recent_holdout_start": holdout_start.date().isoformat(),
        "recent_holdout": holdout,
        "walk_forward": walk_forward(candles, 60, 20, 2.0, 20),
    }
    output = Path(args.output or f"{args.symbol.replace('.', '_')}_donchian_ema_10m_research.json")
    output.write_text(json.dumps(result, indent=2, default=str) + "\n")
    print(json.dumps(result, indent=2, default=str))
    print(f"Research report: {output}")


if __name__ == "__main__":
    main()
