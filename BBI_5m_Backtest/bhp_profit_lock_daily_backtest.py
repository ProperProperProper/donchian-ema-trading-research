#!/usr/bin/env python3
"""Daily backtester for the Donchian/EMA profit-lock/yellow-exit strategy.

The model follows the Moomoo strategy in
``Moomoo_Quant_Donchian_EMA_10m_ProfitLockYellow.py`` while making execution
explicit for research:

* indicators use completed daily candles only;
* a signal executes at the next candle's open;
* exits execute at the next candle's open;
* one long position is allowed;
* stop state is based on the filled entry price and highest completed close.

Use a CSV exported from Moomoo Quant for ASX BHP. The CSV must contain a date
column plus open, high, low, and close columns (case-insensitive). Yahoo-style
CSV files are also accepted.
"""

from __future__ import annotations

import argparse
import itertools
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULTS = {
    "lookback": 45,
    "fast": 8,
    "slow": 89,
    "target": 0.1069,
    "initial_stop": 0.0354,
    "break_even_trigger": 0.02,
    "profit_lock_trigger": 0.08,
    "profit_lock": 0.01,
    "trail_trigger": 0.06,
    "trail_pct": 0.02,
    "yellow_exit": True,
}


@dataclass
class Trade:
    signal_time: str
    entry_time: str
    entry_price: float
    exit_time: str
    exit_price: float
    return_pct: float
    reason: str


def load_csv(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    names = {str(column).strip().lower(): column for column in frame.columns}
    date_column = next(
        (names[name] for name in ("date", "datetime", "time", "time_key") if name in names),
        None,
    )
    if date_column is None:
        raise ValueError("CSV needs one of: date, datetime, time, time_key")
    required = {name: names.get(name) for name in ("open", "high", "low", "close")}
    missing = [name for name, column in required.items() if column is None]
    if missing:
        raise ValueError(f"CSV is missing columns: {', '.join(missing)}")
    result = frame.rename(columns={date_column: "date", **{v: k for k, v in required.items()}})
    result["date"] = pd.to_datetime(result["date"], errors="coerce")
    for column in ("open", "high", "low", "close"):
        result[column] = pd.to_numeric(result[column], errors="coerce")
    result = result.dropna(subset=["date", "open", "high", "low", "close"])
    return result.sort_values("date").drop_duplicates("date").set_index("date")


def add_signals(candles: pd.DataFrame, params: dict) -> pd.DataFrame:
    result = candles.copy()
    result["ema_fast"] = result["close"].ewm(span=params["fast"], adjust=False).mean()
    result["ema_slow"] = result["close"].ewm(span=params["slow"], adjust=False).mean()
    result["prior_high"] = result["high"].shift(1).rolling(params["lookback"]).max()
    result["long_signal"] = (
        (result["close"] > result["prior_high"])
        & (result["ema_fast"] > result["ema_slow"])
        & (result["close"] > result["ema_slow"])
    )
    result["yellow_exit"] = (
        (result["close"] < result["ema_fast"])
        & (result["close"] >= result["ema_slow"])
        & (result["ema_fast"] > result["ema_slow"])
    )
    result["red_exit"] = result["close"] < result["ema_slow"]
    return result.dropna(subset=["prior_high"])


def backtest(candles: pd.DataFrame, params: dict, slippage_bps: float = 2.0) -> list[Trade]:
    rows = list(add_signals(candles, params).iterrows())
    trades: list[Trade] = []
    position: dict | None = None
    slippage = slippage_bps / 10000.0

    for index, (bar_time, bar) in enumerate(rows[:-1]):
        next_time, next_bar = rows[index + 1]
        if position is not None:
            close = float(bar["close"])
            position["peak"] = max(position["peak"], close)
            if close >= position["entry"] * (1.0 + params["break_even_trigger"]):
                position["break_even_armed"] = True
            if position["peak"] >= position["entry"] * (1.0 + params["profit_lock_trigger"]):
                position["profit_lock_armed"] = True

            stop = position["entry"] * (1.0 - params["initial_stop"])
            if position["break_even_armed"]:
                stop = max(stop, position["entry"] * 1.002)
            if position["profit_lock_armed"]:
                stop = max(stop, position["entry"] * (1.0 + params["profit_lock"]))
            if position["peak"] >= position["entry"] * (1.0 + params["trail_trigger"]):
                stop = max(stop, position["peak"] * (1.0 - params["trail_pct"]))

            if close >= position["entry"] * (1.0 + params["target"]):
                reason = "TARGET"
            elif close <= stop:
                reason = "STOP"
            elif params["yellow_exit"] and bool(bar["yellow_exit"]):
                reason = "YELLOW"
            elif bool(bar["red_exit"]):
                reason = "RED"
            else:
                reason = None

            if reason:
                exit_price = float(next_bar["open"]) * (1.0 - slippage)
                trades.append(
                    Trade(
                        signal_time=position["signal_time"].isoformat(),
                        entry_time=position["entry_time"].isoformat(),
                        entry_price=position["entry"],
                        exit_time=next_time.isoformat(),
                        exit_price=exit_price,
                        return_pct=(exit_price / position["entry"] - 1.0) * 100.0,
                        reason=reason,
                    )
                )
                position = None
                continue

        if position is None and bool(bar["long_signal"]):
            entry_price = float(next_bar["open"]) * (1.0 + slippage)
            position = {
                "signal_time": bar_time,
                "entry_time": next_time,
                "entry": entry_price,
                "peak": entry_price,
                "break_even_armed": False,
                "profit_lock_armed": False,
            }

    if position is not None:
        last_time, last_bar = rows[-1]
        exit_price = float(last_bar["close"]) * (1.0 - slippage)
        trades.append(
            Trade(
                signal_time=position["signal_time"].isoformat(),
                entry_time=position["entry_time"].isoformat(),
                entry_price=position["entry"],
                exit_time=last_time.isoformat(),
                exit_price=exit_price,
                return_pct=(exit_price / position["entry"] - 1.0) * 100.0,
                reason="MARK_TO_MARKET",
            )
        )
    return trades


def metrics(trades: list[Trade]) -> dict:
    returns = np.array([trade.return_pct / 100.0 for trade in trades], dtype=float)
    if not len(returns):
        return {"trades": 0, "win_rate_pct": 0.0, "sum_return_pct": 0.0,
                "compounded_pct": 0.0, "max_drawdown_pct": 0.0, "profit_factor": 0.0}
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
    }


def optimize(candles: pd.DataFrame, slippage_bps: float) -> list[dict]:
    grid = {
        "lookback": [20, 30, 45, 60],
        "fast": [5, 8, 13],
        "slow": [30, 50, 89],
        "target": [0.08, 0.1069, 0.15],
        "initial_stop": [0.03, 0.0354, 0.05],
        "yellow_exit": [True, False],
    }
    results = []
    for values in itertools.product(*(grid[name] for name in grid)):
        params = dict(zip(grid, values))
        if params["fast"] >= params["slow"]:
            continue
        trades = backtest(candles, {**DEFAULTS, **params}, slippage_bps)
        score = metrics(trades)
        if score["trades"] >= 5:
            results.append({"params": params, "metrics": score})
    return sorted(results, key=lambda row: row["metrics"]["compounded_pct"], reverse=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Daily BHP/ASX profit-lock backtester")
    parser.add_argument("--csv", required=True, help="Moomoo Quant daily candle CSV")
    parser.add_argument("--optimize", action="store_true")
    parser.add_argument("--slippage-bps", type=float, default=2.0)
    parser.add_argument("--output", default="bhp_profit_lock_daily_trades.csv")
    args = parser.parse_args()
    candles = load_csv(args.csv)
    trades = backtest(candles, DEFAULTS, args.slippage_bps)
    print(json.dumps({"data_start": str(candles.index.min()), "data_end": str(candles.index.max()),
                      "params": DEFAULTS, "metrics": metrics(trades)}, indent=2))
    pd.DataFrame([asdict(trade) for trade in trades]).to_csv(args.output, index=False)
    print(f"Trades CSV: {args.output}")
    if args.optimize:
        results = optimize(candles, args.slippage_bps)
        print("Top optimization results:")
        print(json.dumps(results[:10], indent=2))


if __name__ == "__main__":
    main()
