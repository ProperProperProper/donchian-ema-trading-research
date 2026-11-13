#!/usr/bin/env python3
"""Exit-only optimizer for the immutable 10-minute Donchian/EMA entries.

Every candidate uses the same completed-candle entry signal as the immutable
baseline. Exit state is discovered after the next-bar fill and updated only
from bars that have completed since that fill.
"""
from __future__ import annotations

import itertools
import json
from dataclasses import asdict, dataclass
from datetime import time
from pathlib import Path

import numpy as np
import pandas as pd

from bbi_bull_switch_backtest import fetch_candles
from donchian_ema_10m_baseline_backtest import add_features


@dataclass(frozen=True)
class ExitParams:
    tp: float = 0.08
    initial_stop: float = 0.0395
    be_trigger: float = 0.03
    be_offset: float = 0.0
    trail_activation: float = 0.0
    trail_distance: float = 0.0
    atr_period: int = 14
    atr_multiple: float = 0.0
    yellow_mode: str = "off"  # off, profit, close_fast
    yellow_profit: float = 0.0
    red_mode: str = "ema_slow"  # ema_slow, ema_slow_atr
    red_atr_buffer: float = 0.0
    time_cap_bars: int = 0
    adverse_stop: float = 0.0
    adverse_bars: int = 0
    time_cap_loss_only: bool = False


def atr(x: pd.DataFrame, period: int) -> pd.Series:
    previous = x["close"].shift(1)
    true_range = pd.concat(
        [x["high"] - x["low"], (x["high"] - previous).abs(),
         (x["low"] - previous).abs()], axis=1).max(axis=1)
    return true_range.rolling(period).mean()


def metric(trades: list[dict]) -> dict:
    values = np.array([t["return_pct"] / 100 for t in trades], dtype=float)
    if not len(values):
        return {"trades": 0, "win_rate_pct": 0, "compounded_pct": 0,
                "max_drawdown_pct": 0, "sum_return_pct": 0}
    equity = np.cumprod(1 + values)
    dd = equity / np.maximum.accumulate(equity) - 1
    return {
        "trades": int(len(values)),
        "win_rate_pct": round(float((values > 0).mean() * 100), 3),
        "compounded_pct": round(float((equity[-1] - 1) * 100), 3),
        "max_drawdown_pct": round(float(dd.min() * 100), 3),
        "sum_return_pct": round(float(values.sum() * 100), 3),
    }


def run_exits(x: pd.DataFrame, p: ExitParams, slippage_bps: float = 2.0,
              daily: bool = False) -> list[dict]:
    y = x.copy()
    y["atr"] = atr(y, p.atr_period)
    rows = list(y.iterrows())
    position = None
    trades = []
    for i, (bar_time, bar) in enumerate(rows[:-1]):
        next_time, next_bar = rows[i + 1]
        if position is not None:
            entry = position["entry"]
            position["bars"] += 1
            # Stops/trailing levels are calculated from information available
            # before this bar. A completed bar can arm a new level only for
            # the following bar; this avoids same-bar high/low lookahead.
            stop = entry * (1 - p.initial_stop)
            if position["be_armed"]:
                stop = max(stop, entry * (1 + p.be_offset))
            if p.trail_activation and position["peak"] >= entry * (1 + p.trail_activation):
                stop = max(stop, position["peak"] * (1 - p.trail_distance))
            if p.atr_multiple and pd.notna(bar["atr"]):
                stop = max(stop, position["peak"] - p.atr_multiple * float(bar["atr"]))

            target = entry * (1 + p.tp)
            adverse = (p.adverse_stop > 0 and position["bars"] <= p.adverse_bars
                       and float(bar["low"]) <= entry * (1 - p.adverse_stop))
            timed = (p.time_cap_bars > 0 and position["bars"] >= p.time_cap_bars
                     and (not p.time_cap_loss_only or float(bar["close"]) < entry))
            if adverse:
                raw_exit, reason, exit_time = stop, "ADVERSE", bar_time
            elif timed:
                raw_exit, reason, exit_time = float(bar["close"]), "TIME_CAP", bar_time
            elif float(bar["open"]) >= target:
                raw_exit, reason, exit_time = float(bar["open"]), "TP_GAP", bar_time
            elif float(bar["low"]) <= stop:
                raw_exit, reason, exit_time = stop, "STOP", bar_time
            elif float(bar["high"]) >= target:
                raw_exit, reason, exit_time = target, "TP", bar_time
            else:
                red_level = float(bar["ema_slow"])
                if p.red_mode == "ema_slow_atr" and pd.notna(bar["atr"]):
                    red_level -= p.red_atr_buffer * float(bar["atr"])
                red = (p.red_mode != "none" and float(bar["close"]) < red_level)
                yellow = (float(bar["close"]) < float(bar["ema_fast"])
                          and float(bar["ema_fast"]) > float(bar["ema_slow"])
                          and float(bar["close"]) >= float(bar["ema_slow"])
                          and (p.yellow_mode == "close_fast" or
                               (p.yellow_mode == "profit" and
                                float(bar["close"]) >= entry * (1 + p.yellow_profit))))
                if red or yellow:
                    raw_exit, reason, exit_time = float(next_bar["open"]), ("RED" if red else "YELLOW"), next_time
                elif bar_time.time() >= time(15, 45):
                    raw_exit, reason, exit_time = float(bar["close"]), "END_OF_SESSION", bar_time
                else:
                    continue
            # Only after the bar has been processed do we discover the next
            # bar's dynamic state from the completed bar.
            position["peak"] = max(position["peak"], float(bar["high"]))
            if not position["be_armed"] and float(bar["high"]) >= entry * (1 + p.be_trigger):
                position["be_armed"] = True
            exit_price = raw_exit * (1 - slippage_bps / 10000)
            trades.append({"entry_time": position["entry_time"].isoformat(),
                           "exit_time": exit_time.isoformat(),
                           "return_pct": (exit_price / entry - 1) * 100,
                           "reason": reason})
            position = None
            continue

        if (not daily and (bar_time.time() >= time(15, 45)
                or next_time.date() != bar_time.date()
                or next_time.time() >= time(15, 45))) or not bool(bar["breakout"]):
            continue
        position = {"entry_time": next_time, "entry": float(next_bar["open"]) * (1 + slippage_bps / 10000),
                    "peak": float(next_bar["open"]), "be_armed": False, "bars": 0}
    return trades


def grid() -> list[ExitParams]:
    # Small neighbourhood around the promoted candidate; dynamic choices are
    # intentionally sparse to reduce selection noise.
    values = {
        "tp": [0.08, 0.085, 0.09],
        "initial_stop": [0.0395, 0.0425],
        "be_trigger": [0.03, 0.035],
        "be_offset": [0.0, 0.001],
        "trail_activation": [0.0, 0.04],
        "trail_distance": [0.0, 0.025],
        "atr_multiple": [0.0, 2.5],
        "yellow_mode": ["off", "profit", "close_fast"],
        "yellow_profit": [0.0, 0.01],
        "red_mode": ["ema_slow", "ema_slow_atr"],
        "red_atr_buffer": [0.0, 0.15],
    }
    # Keep only meaningful combinations; exhaustive combinations would be noisy.
    candidates = []
    for tp, stop, be, offset, activation, distance, atr_mult, yellow, yprofit, red, rbuf in itertools.product(*values.values()):
        if yellow == "off" and yprofit != 0.0:
            continue
        if yellow == "profit" and yprofit == 0.0:
            continue
        if activation == 0.0 and distance != 0.0:
            continue
        if activation and distance == 0.0:
            continue
        if atr_mult == 0.0 and rbuf != 0.0:
            continue
        if atr_mult and activation:
            continue
        if red == "ema_slow" and rbuf != 0.0:
            continue
        candidates.append(ExitParams(tp, stop, be, offset, activation, distance, 14, atr_mult, yellow, yprofit, red, rbuf))
    return candidates


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--start", default="2026-01-01")
    parser.add_argument("--end", default="2026-11-30")
    parser.add_argument("--output", default="exit_optimizer_results.json")
    args = parser.parse_args()
    candles = fetch_candles(args.symbol, args.start, args.end, 10, False)
    candles = candles.loc[(candles.index.time >= time(9, 30)) & (candles.index.time <= time(16, 0))]
    features = add_features(candles, 45, 8, 89)
    results = []
    for params in grid():
        trades = run_exits(features, params)
        score = metric(trades)
        if score["trades"] >= 55:
            results.append({"params": asdict(params), "metrics": score})
    results.sort(key=lambda r: (r["metrics"]["compounded_pct"], r["metrics"]["max_drawdown_pct"]), reverse=True)
    Path(args.output).write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results[:20], indent=2))
    print(f"Candidates evaluated: {len(results)}; report: {args.output}")


if __name__ == "__main__":
    main()
