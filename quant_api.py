"""Stable local service interfaces shared by the dashboard, bot and research.

These are deliberately small Python APIs first. The dashboard exposes selected
methods as HTTP endpoints, while the bot can call the same objects directly.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

import pandas as pd

from quant_storage import load_candles, save_execution_event
from strategy_57pct import BASELINE, strategy_signal, walk_forward_optimize, walk_forward_replay, stats


class StrategyAPI:
    name = "DONCHIAN_EMA_10M_WALK_FORWARD"
    version = "2026-11-13.1"

    def signal(self, candles: pd.DataFrame, params: dict[str, Any] | None = None) -> dict[str, Any]:
        return strategy_signal(candles, {**BASELINE, **(params or {})})

    def walk_forward(self, candles: pd.DataFrame, capital: float, allocation: float,
                     slippage_bps: float, commission: float, trials: int = 12000,
                     progress=None) -> dict[str, Any]:
        folds = walk_forward_optimize(
            candles, BASELINE, capital, allocation, slippage_bps, commission,
            progress=progress, trials=trials,
        )
        _, trades, data = walk_forward_replay(
            candles, folds, capital, allocation, slippage_bps, commission,
        )
        return {"stats": stats(trades, data, capital), "folds": folds,
                "trades": trades, "data": data}


@dataclass(frozen=True)
class RiskDecision:
    allowed: bool
    quantity: int
    reason: str
    risk_fraction: float


class RiskAPI:
    """Pure, auditable risk calculations; it does not submit orders."""

    def entry_decision(self, cash: float, price: float, allocation: float = .95,
                       max_position_value: float | None = None,
                       drawdown: float = 0.0, max_drawdown: float = .20) -> RiskDecision:
        if price <= 0 or cash <= 0:
            return RiskDecision(False, 0, "invalid_cash_or_price", 0.0)
        if drawdown <= -abs(max_drawdown):
            return RiskDecision(False, 0, "drawdown_circuit_breaker", 0.0)
        value = cash * max(0.0, min(1.0, allocation))
        if max_position_value is not None:
            value = min(value, max(0.0, max_position_value))
        qty = int(value // price)
        if qty <= 0:
            return RiskDecision(False, 0, "insufficient_buying_power", 0.0)
        return RiskDecision(True, qty, "approved", value / cash)


class AuditAPI:
    def event(self, event_type: str, symbol: str | None, payload: Any) -> None:
        save_execution_event(event_type, symbol, payload)


class DataAPI:
    def candles(self, symbol: str, interval: int, start: str, end: str) -> pd.DataFrame:
        return load_candles(symbol, interval, start, end)


strategy_api = StrategyAPI()
risk_api = RiskAPI()
audit_api = AuditAPI()
data_api = DataAPI()


def risk_json(**kwargs: Any) -> dict[str, Any]:
    return asdict(risk_api.entry_decision(**kwargs))
