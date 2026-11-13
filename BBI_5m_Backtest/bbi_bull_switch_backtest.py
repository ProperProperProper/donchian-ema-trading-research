#!/usr/bin/env python3
"""Live-style BBI bull-switch backtest using configurable Moomoo candles.

The simulation deliberately avoids lookahead:
  * Regular-session signals use completed candles only.
  * A regular-session signal enters at the next regular candle open.
  * Optional premarket data may authorize an entry at the first regular open.
  * One long position only; never short.
  * TP defaults to 1%.
  * Bear exits require two completed bear candles by default.
  * Positions are closed on the completed 15:55 candle to avoid overnight gaps.

Moomoo OpenD must be running at 127.0.0.1:11111.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import asdict, dataclass
from datetime import time
from typing import List, Optional

import pandas as pd
from moomoo import AuType, KLType, KL_FIELD, OpenQuoteContext, RET_OK


DEFAULT_START = "2026-01-01"
DEFAULT_END = "2026-11-30"
REGULAR_OPEN = time(9, 30)
REGULAR_CLOSE = time(16, 0)


@dataclass
class Trade:
    symbol: str
    signal_time: str
    entry_time: str
    entry_price: float
    exit_time: str
    exit_price: float
    return_pct: float
    pnl_per_share: float
    exit_reason: str


def fetch_candles(symbol: str, start: str, end: str, interval: int, include_extended: bool) -> pd.DataFrame:
    """Fetch all Moomoo candles, handling pagination."""
    interval_types = {
        1: KLType.K_1M,
        3: KLType.K_3M,
        5: KLType.K_5M,
        10: KLType.K_10M,
        15: KLType.K_15M,
        30: KLType.K_30M,
        1440: KLType.K_DAY,
    }
    if interval not in interval_types:
        raise ValueError("interval must be one of: 1, 3, 5, 10, 15, 30, 1440")
    quote_ctx = OpenQuoteContext(host="127.0.0.1", port=11111)
    frames: List[pd.DataFrame] = []
    page_req_key = None
    try:
        while True:
            ret, data, page_req_key = quote_ctx.request_history_kline(
                code=symbol,
                start=start,
                end=end,
                ktype=interval_types[interval],
                autype=AuType.QFQ,
                fields=[KL_FIELD.ALL],
                max_count=1000,
                page_req_key=page_req_key,
                extended_time=include_extended,
            )
            if ret != RET_OK:
                raise RuntimeError(f"Moomoo request_history_kline failed: {data}")
            if data is not None and not data.empty:
                frames.append(data.copy())
            if page_req_key is None:
                break
    finally:
        quote_ctx.close()

    if not frames:
        raise RuntimeError("Moomoo returned no candles for the requested range.")
    candles = pd.concat(frames, ignore_index=True)
    candles["time_key"] = pd.to_datetime(candles["time_key"])
    candles = candles.sort_values("time_key").drop_duplicates("time_key").set_index("time_key")
    required = ["open", "high", "low", "close"]
    missing = [name for name in required if name not in candles.columns]
    if missing:
        raise RuntimeError(f"Moomoo candle response is missing columns: {missing}")
    return candles.dropna(subset=required)


def add_bbi_columns(candles: pd.DataFrame) -> pd.DataFrame:
    result = candles.copy()
    for period in (3, 6, 12, 24):
        result[f"sma_{period}"] = result["close"].rolling(period).mean()
    result["bbi"] = result[["sma_3", "sma_6", "sma_12", "sma_24"]].mean(axis=1)

    # Fast switch: enter on the first completed candle where price is above a
    # rising BBI after the prior candle was not in that bullish state. This
    # avoids repeated re-entry signals caused by a single noisy BBI tick.
    result["bull_state"] = (
        (result["close"] > result["bbi"])
        & (result["bbi"] > result["bbi"].shift(1))
    ).fillna(False)
    result["bull_switch"] = (
        result["bull_state"] & ~result["bull_state"].shift(1).fillna(False)
    ).fillna(False)
    result["bear_bar"] = (
        (result["close"] < result["bbi"])
        & (result["bbi"] < result["bbi"].shift(1))
    ).fillna(False)
    return result.dropna(subset=["bbi"])


def regular_rows(candles: pd.DataFrame) -> pd.DataFrame:
    times = candles.index.time
    mask = [(REGULAR_OPEN <= value <= REGULAR_CLOSE) for value in times]
    return candles.loc[mask]


def premarket_bias(candles: pd.DataFrame):
    """Return last completed premarket BBI state and value for each date."""
    result = {}
    for date, group in candles.groupby(candles.index.date):
        pre = group.loc[group.index.time < REGULAR_OPEN]
        if pre.empty:
            continue
        last = pre.iloc[-1]
        bullish = bool(last["bull_state"])
        result[date] = {"bullish": bullish, "bbi": float(last["bbi"]), "time": pre.index[-1]}
    return result


def adjusted_buy(raw_price: float, slippage_bps: float) -> float:
    return raw_price * (1.0 + slippage_bps / 10000.0)


def adjusted_sell(raw_price: float, slippage_bps: float) -> float:
    return raw_price * (1.0 - slippage_bps / 10000.0)


def run_backtest(
    candles: pd.DataFrame,
    symbol: str,
    tp_pct: float = 0.01,
    bear_bars: int = 2,
    use_premarket: bool = True,
    flat_at_close: bool = True,
    slippage_bps: float = 2.0,
    flat_time: time = time(15, 55),
):
    """Run one-position, live-style simulation."""
    regular = regular_rows(candles)
    premarket = premarket_bias(candles) if use_premarket else {}
    bear_confirmed = regular["bear_bar"].rolling(bear_bars).sum().eq(bear_bars)
    rows = list(regular.iterrows())
    trades: List[Trade] = []
    position: Optional[dict] = None
    last_date = None

    for i, (bar_time, bar) in enumerate(rows):
        next_item = rows[i + 1] if i + 1 < len(rows) else None
        next_time, next_bar = next_item if next_item else (None, None)
        first_regular_bar = bar_time.date() != last_date
        last_date = bar_time.date()

        if position is not None:
            entry_price = position["entry_price"]
            target = entry_price * (1.0 + tp_pct)

            # TP is active during the current completed candle.
            if float(bar["open"]) >= target:
                exit_price = adjusted_sell(float(bar["open"]), slippage_bps)
                reason = "TP_GAP"
            elif float(bar["high"]) >= target:
                exit_price = adjusted_sell(target, slippage_bps)
                reason = "TP"
            elif flat_at_close and bar_time.time() >= flat_time:
                exit_price = adjusted_sell(float(bar["close"]), slippage_bps)
                reason = "END_OF_SESSION"
            elif bool(bear_confirmed.iloc[i]) and next_bar is not None:
                exit_price = adjusted_sell(float(next_bar["open"]), slippage_bps)
                reason = "BEAR_EXIT"
            else:
                continue

            trades.append(
                Trade(
                    symbol=symbol,
                    signal_time=position["signal_time"].isoformat(),
                    entry_time=position["entry_time"].isoformat(),
                    entry_price=entry_price,
                    exit_time=(bar_time if reason.startswith("TP") or reason == "END_OF_SESSION" else next_time).isoformat(),
                    exit_price=exit_price,
                    return_pct=(exit_price / entry_price - 1.0) * 100.0,
                    pnl_per_share=exit_price - entry_price,
                    exit_reason=reason,
                )
            )
            position = None
            continue

        if next_item is None:
            continue
        if flat_at_close and bar_time.time() >= flat_time:
            continue

        premarket_signal = False
        if first_regular_bar and use_premarket:
            bias = premarket.get(bar_time.date())
            premarket_signal = bool(bias and bias["bullish"] and float(bar["open"]) > bias["bbi"])

        if premarket_signal:
            entry_time, entry_bar = bar_time, bar
            signal_time = premarket[bar_time.date()]["time"]
        elif bool(bar["bull_switch"]):
            entry_time, entry_bar = next_time, next_bar
            signal_time = bar_time
            if entry_time.date() != bar_time.date() or entry_time.time() >= flat_time:
                continue
        else:
            continue

        entry_price = adjusted_buy(float(entry_bar["open"]), slippage_bps)
        target = entry_price * (1.0 + tp_pct)
        position = {"signal_time": signal_time, "entry_time": entry_time, "entry_price": entry_price}

        # The TP order is active immediately after the entry fill.
        if float(entry_bar["open"]) >= target:
            exit_price = adjusted_sell(float(entry_bar["open"]), slippage_bps)
            reason = "TP_GAP"
        elif float(entry_bar["high"]) >= target:
            exit_price = adjusted_sell(target, slippage_bps)
            reason = "TP"
        else:
            continue

        trades.append(
            Trade(
                symbol=symbol,
                signal_time=signal_time.isoformat(),
                entry_time=entry_time.isoformat(),
                entry_price=entry_price,
                exit_time=entry_time.isoformat(),
                exit_price=exit_price,
                return_pct=(exit_price / entry_price - 1.0) * 100.0,
                pnl_per_share=exit_price - entry_price,
                exit_reason=reason,
            )
        )
        position = None

    return trades, position


def print_report(candles, trades, open_position, symbol, args):
    completed = len(trades)
    wins = sum(t.return_pct > 0 for t in trades)
    compounded = 1.0
    for trade in trades:
        compounded *= 1.0 + trade.return_pct / 100.0
    print(f"Symbol: {symbol}")
    print(f"Interval: {args.interval} minutes")
    print(f"Data: {candles.index.min()} to {candles.index.max()}")
    print(f"Candles: {len(candles)}")
    print(f"TP: {args.tp:.2%}; bear confirmation: {args.bear_bars} bars")
    print(f"Premarket open filter: {args.use_premarket}; flat at 15:55: {not args.no_flat_at_close}")
    print(f"Slippage: {args.slippage_bps:.1f} bps per side")
    print(f"Completed trades: {completed}")
    print(f"Winning trades: {wins}")
    print(f"Win rate: {(wins / completed * 100.0) if completed else 0.0:.2f}%")
    print(f"Sum of trade returns: {sum(t.return_pct for t in trades):.2f}%")
    print(f"Compounded return: {(compounded - 1.0) * 100.0:.2f}%")
    if open_position:
        print(f"Open position: entry={open_position['entry_price']:.4f} at {open_position['entry_time']}")
    else:
        print("Open position: none")


def main():
    parser = argparse.ArgumentParser(description="Live-style BBI interval Moomoo backtest")
    parser.add_argument("--symbol", required=True, help="Moomoo symbol, e.g. US.APP")
    parser.add_argument("--start", default=DEFAULT_START)
    parser.add_argument("--end", default=DEFAULT_END)
    parser.add_argument("--interval", type=int, choices=[1, 3, 5, 10, 15, 30], default=15, help="Candle interval in minutes; default 15")
    parser.add_argument("--tp", type=float, default=0.02, help="Take-profit fraction; default 0.02")
    parser.add_argument("--bear-bars", type=int, default=3, help="Completed bear bars required before exit")
    parser.add_argument("--use-premarket", action="store_true", help="Allow a premarket-confirmed entry at the first regular open")
    parser.add_argument("--no-flat-at-close", action="store_true", help="Allow overnight positions")
    parser.add_argument("--slippage-bps", type=float, default=2.0, help="Slippage per side in basis points")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    if args.bear_bars < 1 or args.tp <= 0:
        parser.error("--bear-bars must be >= 1 and --tp must be positive")

    candles = add_bbi_columns(fetch_candles(args.symbol, args.start, args.end, args.interval, args.use_premarket))
    flat_time = time(15, 45) if args.interval >= 15 else time(15, 55)
    trades, open_position = run_backtest(
        candles,
        args.symbol,
        tp_pct=args.tp,
        bear_bars=args.bear_bars,
        use_premarket=args.use_premarket,
        flat_at_close=not args.no_flat_at_close,
        slippage_bps=args.slippage_bps,
        flat_time=flat_time,
    )
    print_report(candles, trades, open_position, args.symbol, args)
    output = args.output or os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        f"{args.symbol.replace('.', '_')}_bbi_{args.interval}m_trades.csv",
    )
    pd.DataFrame([asdict(trade) for trade in trades]).to_csv(output, index=False)
    print(f"Trades CSV: {output}")


if __name__ == "__main__":
    main()
