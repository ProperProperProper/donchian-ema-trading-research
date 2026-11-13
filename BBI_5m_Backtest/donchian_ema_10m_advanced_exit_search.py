#!/usr/bin/env python3
"""Broad exit-only Pareto search for the frozen 10-minute entries."""
from __future__ import annotations

import argparse
import json
import random
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from bbi_bull_switch_backtest import fetch_candles
from donchian_ema_10m_baseline_backtest import add_features
from donchian_ema_10m_exit_optimizer import metric

warnings.filterwarnings("ignore", category=FutureWarning)


def simulate(x: pd.DataFrame, p: dict, slippage_bps: float = 2.0) -> list[dict]:
    rows = list(x.iterrows())
    position = None
    trades = []
    for i, (bar_time, bar) in enumerate(rows[:-1]):
        next_time, next_bar = rows[i + 1]
        if position is not None:
            entry = position["entry"]
            position["bars"] += 1
            stop = entry * (1 - p["initial_stop"])
            if position["be_armed"]:
                stop = max(stop, entry * (1 + p["be_offset"]))
            if position["lock_level"]:
                stop = max(stop, entry * (1 + position["lock_level"]))
            if p["trail_activation"] and position["peak"] >= entry * (1 + p["trail_activation"]):
                stop = max(stop, position["peak"] * (1 - p["trail_distance"]))

            target = entry * (1 + p["tp"])
            time_exit = p["time_cap"] and position["bars"] >= p["time_cap"] and float(bar["close"]) < entry
            red = position["red_count"] >= p["red_confirm"]
            yellow = (p["yellow_exit"] and float(bar["close"]) < float(bar["ema_fast"])
                      and float(bar["close"]) >= float(bar["ema_slow"])
                      and float(bar["close"]) >= entry * (1 + p["yellow_profit"]))
            if time_exit:
                raw, reason, exit_time = float(bar["close"]), "TIME", bar_time
            elif float(bar["open"]) >= target:
                raw, reason, exit_time = float(bar["open"]), "TP_GAP", bar_time
            elif float(bar["low"]) <= stop:
                raw, reason, exit_time = stop, "STOP", bar_time
            elif float(bar["high"]) >= target:
                raw, reason, exit_time = target, "TP", bar_time
            elif red or yellow:
                raw, reason, exit_time = float(next_bar["open"]), ("RED" if red else "YELLOW"), next_time
            else:
                raw = reason = exit_time = None
            if raw is not None:
                exit_price = raw * (1 - slippage_bps / 10000)
                trades.append({"return_pct": (exit_price / entry - 1) * 100, "reason": reason,
                               "entry_time": position["entry_time"].isoformat(),
                               "exit_time": exit_time.isoformat()})
                position = None
                continue

            position["peak"] = max(position["peak"], float(bar["high"]))
            if not position["be_armed"] and p["be_trigger"] and float(bar["high"]) >= entry * (1 + p["be_trigger"]):
                position["be_armed"] = True
            if p["lock_trigger"] and position["peak"] >= entry * (1 + p["lock_trigger"]):
                position["lock_level"] = max(position["lock_level"], p["lock_profit"])
            position["red_count"] = position["red_count"] + 1 if float(bar["close"]) < float(bar["ema_slow"]) else 0
            continue

        if (bar_time.time().hour >= 15 and bar_time.time().minute >= 45) or \
                next_time.date() != bar_time.date() or next_time.time().hour >= 15 or not bool(bar["breakout"]):
            continue
        position = {"entry_time": next_time,
                    "entry": float(next_bar["open"]) * (1 + slippage_bps / 10000),
                    "peak": float(next_bar["open"]), "bars": 0, "be_armed": False,
                    "lock_level": 0.0, "red_count": 0}
    return trades


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--start", default="2026-01-01")
    parser.add_argument("--end", default="2026-11-12")
    parser.add_argument("--samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20261112)
    parser.add_argument("--output", default="advanced_exit_results.json")
    args = parser.parse_args()
    candles = fetch_candles(args.symbol, "2025-01-01", args.end, 10, False)
    x = add_features(candles, 45, 8, 89)
    x = x.loc[x.index >= pd.Timestamp(args.start)]
    rng = random.Random(args.seed)
    results = []
    for _ in range(args.samples):
        p = {
            "tp": round(rng.uniform(.06, .12), 4),
            "initial_stop": round(rng.uniform(.03, .06), 4),
            "be_trigger": rng.choice([0, .02, .03, .04]),
            "be_offset": rng.choice([0, .002, .005]),
            "lock_trigger": rng.choice([0, .04, .05, .06, .08]),
            "lock_profit": rng.choice([0, .005, .01, .02, .03]),
            "trail_activation": rng.choice([0, .04, .06, .08]),
            "trail_distance": rng.choice([0, .015, .02, .03, .04]),
            "red_confirm": rng.choice([1, 2, 3]),
            "yellow_exit": rng.choice([False, True]),
            "yellow_profit": rng.choice([.0, .01, .02]),
            "time_cap": rng.choice([0, 20, 40, 60]),
        }
        if (p["trail_activation"] == 0) != (p["trail_distance"] == 0):
            continue
        if not p["yellow_exit"]:
            p["yellow_profit"] = 0.0
        score = metric(simulate(x, p))
        results.append({"params": p, "metrics": score})
    results.sort(key=lambda r: (r["metrics"]["compounded_pct"], r["metrics"]["max_drawdown_pct"]), reverse=True)
    eligible = [r for r in results if r["metrics"]["compounded_pct"] >= 51 and r["metrics"]["trades"] >= 50]
    eligible.sort(key=lambda r: (r["metrics"]["max_drawdown_pct"], r["metrics"]["compounded_pct"]), reverse=True)
    Path(args.output).write_text(json.dumps({"results": results, "best_over_51": eligible[:20]}, indent=2) + "\n")
    print(json.dumps({"top_profit": results[:10], "best_drawdown_over_51": eligible[:10]}, indent=2))


if __name__ == "__main__":
    main()
