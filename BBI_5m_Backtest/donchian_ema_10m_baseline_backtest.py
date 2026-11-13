#!/usr/bin/env python3
"""Live-style Donchian/EMA bull-market strategy backtest.

This is the faster bull detector found while comparing alternatives to BBI:
  * 10-minute completed candles by default.
  * Long only.
  * Entry when the completed candle breaks the prior 20-candle high and
    EMA(21) is above EMA(55), with price above EMA(55).
  * Entry at the next candle open.
  * 6.38% take-profit, 3.95% stop-loss, 2 bps slippage per side.
  * Exit on trend failure or the completed 15:45 candle.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import asdict, dataclass
from datetime import time

import numpy as np
import pandas as pd

from bbi_bull_switch_backtest import fetch_candles


@dataclass
class Trade:
    signal_time: str
    entry_time: str
    entry_price: float
    exit_time: str
    exit_price: float
    return_pct: float
    reason: str


def add_features(candles: pd.DataFrame, lookback: int, fast: int, slow: int) -> pd.DataFrame:
    x = candles.copy()
    x["ema_fast"] = x["close"].ewm(span=fast, adjust=False).mean()
    x["ema_slow"] = x["close"].ewm(span=slow, adjust=False).mean()
    x["prior_high"] = x["high"].shift(1).rolling(lookback).max()
    x["breakout"] = (
        (x["close"] > x["prior_high"])
        & (x["ema_fast"] > x["ema_slow"])
        & (x["close"] > x["ema_slow"])
    ).fillna(False)

    # BBI comparison signal.
    for period in (3, 6, 12, 24):
        x[f"sma_{period}"] = x["close"].rolling(period).mean()
    x["bbi"] = x[["sma_3", "sma_6", "sma_12", "sma_24"]].mean(axis=1)
    bbi_state = ((x["close"] > x["bbi"]) & (x["bbi"] > x["bbi"].shift(1))).fillna(False)
    x["bbi_switch"] = (bbi_state & ~bbi_state.shift(1).fillna(False)).fillna(False)
    return x.dropna(subset=["prior_high", "bbi"])


def run(candles, tp: float, stop: float, slippage_bps: float, daily: bool = False):
    rows = list(candles.iterrows())
    position = None
    trades = []
    candidate_times = []
    bbi_times = []

    for i, (bar_time, bar) in enumerate(rows[:-1]):
        next_time, next_bar = rows[i + 1]
        if bool(bar["breakout"]):
            candidate_times.append(bar_time)
        if bool(bar["bbi_switch"]):
            bbi_times.append(bar_time)

        if position is not None:
            entry = position["entry"]
            target = entry * (1.0 + tp)
            stop_price = entry * (1.0 - stop)
            if float(bar["open"]) >= target:
                exit_price, reason = float(bar["open"]), "TP_GAP"
            elif float(bar["low"]) <= stop_price:
                exit_price, reason = stop_price, "STOP"
            elif float(bar["high"]) >= target:
                exit_price, reason = target, "TP"
            elif bar_time.time() >= time(15, 45):
                exit_price, reason = float(bar["close"]), "END_OF_SESSION"
            elif float(bar["close"]) < float(bar["ema_slow"]) and float(bar["ema_fast"]) < float(bar["ema_slow"]):
                exit_price, reason = float(next_bar["open"]), "TREND_EXIT"
                exit_time = next_time
                trades.append(_trade(position, exit_time, exit_price, reason, slippage_bps))
                position = None
                continue
            else:
                continue

            trades.append(_trade(position, bar_time, exit_price, reason, slippage_bps))
            position = None
            continue

        if (not daily and (bar_time.time() >= time(15, 45)
            or next_time.date() != bar_time.date()
            or next_time.time() >= time(15, 45))) or not bool(bar["breakout"]):
            continue

        position = {
            "signal_time": bar_time,
            "entry_time": next_time,
            "entry": float(next_bar["open"]) * (1.0 + slippage_bps / 10000.0),
        }

    return trades, candidate_times, bbi_times


def _trade(position, exit_time, raw_exit, reason, slippage_bps):
    exit_price = raw_exit * (1.0 - slippage_bps / 10000.0)
    entry = position["entry"]
    return Trade(
        signal_time=position["signal_time"].isoformat(),
        entry_time=position["entry_time"].isoformat(),
        entry_price=entry,
        exit_time=exit_time.isoformat(),
        exit_price=exit_price,
        return_pct=(exit_price / entry - 1.0) * 100.0,
        reason=reason,
    )


def main():
    parser = argparse.ArgumentParser(description="Donchian/EMA bull-market backtest")
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--start", default="2026-01-01")
    parser.add_argument("--end", default="2026-11-30")
    # Frozen from the July-November randomized training search.
    parser.add_argument("--interval", type=int, choices=[5, 10, 15, 30], default=10)
    parser.add_argument("--lookback", type=int, default=45)
    parser.add_argument("--fast-ema", type=int, default=8)
    parser.add_argument("--slow-ema", type=int, default=89)
    parser.add_argument("--tp", type=float, default=0.0638)
    parser.add_argument("--stop", type=float, default=0.0395)
    parser.add_argument("--slippage-bps", type=float, default=2.0)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    candles = fetch_candles(args.symbol, args.start, args.end, args.interval, False)
    candles = candles.loc[(candles.index.time >= time(9, 30)) & (candles.index.time <= time(16, 0))]
    candles = add_features(candles, args.lookback, args.fast_ema, args.slow_ema)
    trades, candidate_times, bbi_times = run(candles, args.tp, args.stop, args.slippage_bps)

    returns = np.array([t.return_pct / 100.0 for t in trades])
    compounded = np.prod(1.0 + returns) - 1.0 if len(returns) else 0.0
    wins = int((returns > 0).sum())
    leads = []
    for candidate in candidate_times:
        later_bbi = [t for t in bbi_times if t.date() == candidate.date() and t > candidate]
        if later_bbi:
            leads.append((later_bbi[0] - candidate).total_seconds() / 60.0)

    print(f"Symbol: {args.symbol}")
    print(f"Interval: {args.interval} minutes")
    print(f"Trades: {len(trades)}; wins: {wins}; win rate: {(wins / len(trades) * 100) if trades else 0:.2f}%")
    print(f"Sum of returns: {returns.sum() * 100:.2f}%")
    print(f"Compounded return: {compounded * 100:.2f}%")
    print(f"Breakout signals: {len(candidate_times)}")
    print(f"Median lead versus later BBI switch: {np.median(leads) if leads else 0:.1f} minutes")

    output = args.output or os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        f"{args.symbol.replace('.', '_')}_donchian_ema_{args.interval}m_trades.csv",
    )
    pd.DataFrame([asdict(t) for t in trades]).to_csv(output, index=False)
    print(f"Trades CSV: {output}")


if __name__ == "__main__":
    main()
