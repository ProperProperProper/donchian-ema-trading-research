#!/usr/bin/env python3
"""Moomoo paper-competition bot for the 10-minute Donchian/EMA baseline.

The bot is paper-only by construction: every order uses TrdEnv.SIMULATE.
Order submission additionally requires --live-paper. Without that flag the
bot prints signals and intended orders but never submits anything.

Strategy:
  - 45 completed 10-minute bars for the Donchian breakout
  - EMA 8 / EMA 89 trend filter
  - long-only, one position per symbol
  - 6.38% take profit or 3.95% stop loss
  - 95% of available cash per entry

Moomoo OpenD must be running on 127.0.0.1:11111 and logged in. This script
does not contain credentials or attempt to unlock a live account.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import signal
import time
from datetime import time as dt_time
from pathlib import Path
from typing import Any

import pandas as pd
from moomoo import (
    KLType,
    OpenQuoteContext,
    OpenSecTradeContext,
    OrderType,
    RET_OK,
    TrdEnv,
    TrdMarket,
    TrdSide,
)

from quant_storage import save_candles, save_execution_event
from quant_api import audit_api, risk_api, strategy_api


POSITION_FRACTION = 0.95
POLL_SECONDS = 30
MIN_BARS = 89 + 45 + 5
STATE_PATH = Path(__file__).with_name("competition_bot_state.json")
LOG_PATH = Path(__file__).with_name("competition_bot.log")


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(LOG_PATH), logging.StreamHandler()],
    )


def scalar(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
        return result if math.isfinite(result) else default
    except (TypeError, ValueError):
        return default


def row_value(frame: pd.DataFrame, row: int, names: tuple[str, ...], default: float = 0.0) -> float:
    for name in names:
        if name in frame.columns:
            return scalar(frame.iloc[row][name], default)
    return default


def code(symbol: str) -> str:
    symbol = symbol.upper().strip()
    return symbol if "." in symbol else f"US.{symbol}"


class CompetitionBot:
    def __init__(self, symbols: list[str], acc_index: int, live_paper: bool, interval: int, acc_id: int | None = None) -> None:
        self.symbols = [code(s) for s in symbols]
        self.acc_index = acc_index
        self.acc_id = acc_id
        self.live_paper = live_paper
        self.interval = interval
        self.running = True
        self.last_error: str | None = None
        self.peak_price = 0.0
        self.break_even_armed = False
        self.profit_lock_armed = False
        self.state = self._load_state()
        self.quote = OpenQuoteContext(host="127.0.0.1", port=11111)
        self.trade = OpenSecTradeContext(
            filter_trdmarket=TrdMarket.US,
            host="127.0.0.1",
            port=11111,
        )

    def _load_state(self) -> dict[str, Any]:
        if STATE_PATH.exists():
            state = json.loads(STATE_PATH.read_text())
        else:
            state = {"last_bar": {}, "entries": {}}
        state.setdefault("pending", {})
        return state

    def _save_state(self) -> None:
        STATE_PATH.write_text(json.dumps(self.state, indent=2) + "\n")

    def close(self) -> None:
        self.quote.close()
        self.trade.close()

    def stop(self, *_: Any) -> None:
        self.running = False

    def account_funds(self) -> float:
        kwargs = {"trd_env": TrdEnv.SIMULATE, "acc_id": self.acc_id} if self.acc_id else {"trd_env": TrdEnv.SIMULATE, "acc_index": self.acc_index}
        ret, data = self.trade.accinfo_query(**kwargs)
        if ret != RET_OK:
            raise RuntimeError(f"account query failed: {data}")
        for field in ("available_funds", "cash", "power", "total_assets"):
            if field in data.columns:
                return scalar(data.iloc[0][field])
        raise RuntimeError(f"no usable cash field in account response: {list(data.columns)}")

    def position(self, symbol: str) -> tuple[float, float]:
        kwargs = {"trd_env": TrdEnv.SIMULATE, "acc_id": self.acc_id} if self.acc_id else {"trd_env": TrdEnv.SIMULATE, "acc_index": self.acc_index}
        ret, data = self.trade.position_list_query(**kwargs)
        if ret != RET_OK:
            raise RuntimeError(f"position query failed: {data}")
        if data.empty or "code" not in data.columns:
            return 0.0, 0.0
        rows = data[data["code"].astype(str) == symbol]
        if rows.empty:
            return 0.0, 0.0
        row = rows.iloc[0]
        qty = scalar(row.get("qty", row.get("position", 0)))
        avg = scalar(row.get("cost_price", row.get("average_cost", 0)))
        return qty, avg

    def market_price(self, symbol: str) -> float:
        ret, data = self.quote.get_market_snapshot([symbol])
        if ret != RET_OK or data.empty:
            raise RuntimeError(f"snapshot failed for {symbol}: {data}")
        return row_value(data, 0, ("last_price", "cur_price"))

    def candles(self, symbol: str) -> pd.DataFrame:
        ret, data, _ = self.quote.request_history_kline(
            symbol,
            ktype=KLType.K_10M,
            max_count=MIN_BARS + 20,
            extended_time=False,
        )
        if ret != RET_OK:
            raise RuntimeError(f"candle query failed for {symbol}: {data}")
        save_candles(data.set_index("time_key"), symbol, 10)
        return data

    def submit(self, symbol: str, side: Any, qty: int, reason: str) -> None:
        logging.info("%s %s qty=%d mode=%s", reason, symbol, qty, "LIVE_PAPER" if self.live_paper else "DRY_RUN")
        save_execution_event("INTENDED_ORDER", symbol, {"side": str(side), "qty": qty, "reason": reason, "paper": True})
        if not self.live_paper:
            return
        order_kwargs = {"trd_env": TrdEnv.SIMULATE, "acc_id": self.acc_id} if self.acc_id else {"trd_env": TrdEnv.SIMULATE, "acc_index": self.acc_index}
        ret, data = self.trade.place_order(
            price=0,
            qty=qty,
            code=symbol,
            trd_side=side,
            order_type=OrderType.MARKET,
            **order_kwargs,
        )
        if ret != RET_OK:
            save_execution_event("ORDER_ERROR", symbol, {"side": str(side), "qty": qty, "reason": reason, "error": str(data), "paper": True})
            raise RuntimeError(f"paper order failed for {symbol}: {data}")
        logging.info("order accepted: %s", data.to_dict("records"))
        save_execution_event("ORDER_ACCEPTED", symbol, {"side": str(side), "qty": qty, "reason": reason, "response": data.to_dict("records"), "paper": True})

    def tick(self) -> None:
        for symbol in self.symbols:
            try:
                signal_data = strategy_api.signal(self.candles(symbol))
                if self.state["last_bar"].get(symbol) == signal_data["bar"]:
                    continue
                qty, avg = self.position(symbol)
                price = self.market_price(symbol)
                logging.info("%s bar=%s price=%.4f position=%.4f signal=%s trend_exit=%s", symbol, signal_data["bar"], price, qty, signal_data["long_signal"], signal_data["trend_exit"])

                pending = self.state["pending"].get(symbol)
                if pending:
                    filled = (pending == "BUY" and qty > 0) or (pending == "SELL" and qty <= 0)
                    if filled:
                        self.state["pending"].pop(symbol, None)
                    else:
                        self.state["last_bar"][symbol] = signal_data["bar"]
                        self._save_state()
                        continue

                if qty > 0:
                    take_profit = price >= avg * 1.0274
                    stop_loss = price <= avg * (1 - 0.1324)
                    session_exit = bool(signal_data["session_exit"])
                    if take_profit or stop_loss or session_exit or signal_data["trend_exit"]:
                        self.submit(symbol, TrdSide.SELL, int(qty), "EXIT")
                        self.state["pending"][symbol] = "SELL"
                        self.peak_price = 0.0
                        self.break_even_armed = False
                        self.profit_lock_armed = False
                    self.state["last_bar"][symbol] = signal_data["bar"]
                    self._save_state()
                    continue

                if signal_data["long_signal"] and signal_data["entry_allowed"]:
                    cash = self.account_funds()
                    decision = risk_api.entry_decision(cash, price, POSITION_FRACTION)
                    audit_api.event("RISK_DECISION", symbol, {**decision.__dict__, "signal": signal_data})
                    if decision.allowed:
                        self.submit(symbol, TrdSide.BUY, decision.quantity, "ENTRY")
                        self.state["pending"][symbol] = "BUY"
                        self.peak_price = price
                        self.break_even_armed = False
                        self.profit_lock_armed = False
                    self.state["last_bar"][symbol] = signal_data["bar"]
                    self._save_state()
                    continue
                # Mark the bar only after all queries and any order submission
                # have completed. A failed cycle can therefore be retried.
                self.state["last_bar"][symbol] = signal_data["bar"]
                self._save_state()
            except Exception as exc:
                self.last_error = repr(exc)
                logging.exception("symbol cycle failed: %s", symbol)
                save_execution_event("CYCLE_ERROR", symbol, {"paper": True, "error": repr(exc)})

    def run(self) -> None:
        logging.info("starting %s symbols=%s account_index=%d", "LIVE_PAPER" if self.live_paper else "DRY_RUN", self.symbols, self.acc_index)
        while self.running:
            self.tick()
            for _ in range(self.interval):
                if not self.running:
                    break
                time.sleep(1)
        logging.info("stopped")


def list_accounts() -> int:
    ctx = OpenSecTradeContext(filter_trdmarket=TrdMarket.US, host="127.0.0.1", port=11111)
    try:
        ret, data = ctx.get_acc_list()
        if ret != RET_OK:
            print(data)
            return 1
        print(data.to_string(index=False))
        return 0
    finally:
        ctx.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("symbols", nargs="*", help="US symbols, e.g. AAPL NVDA; defaults to config only when supplied")
    parser.add_argument("--acc-index", type=int, default=0, help="SIMULATE US account index from --list-accounts")
    parser.add_argument("--interval", type=int, default=POLL_SECONDS)
    parser.add_argument("--live-paper", action="store_true", help="Actually submit TrdEnv.SIMULATE orders")
    parser.add_argument("--list-accounts", action="store_true")
    args = parser.parse_args()
    setup_logging()
    if args.list_accounts:
        return list_accounts()
    if not args.symbols:
        parser.error("provide at least one symbol, or use --list-accounts")
    bot = CompetitionBot(args.symbols, args.acc_index, args.live_paper, args.interval)
    signal.signal(signal.SIGINT, bot.stop)
    signal.signal(signal.SIGTERM, bot.stop)
    try:
        bot.run()
    finally:
        bot.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
