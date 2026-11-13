"""Exact local implementation of the GitHub 10-minute winner.

Source model: BBI_5m_Backtest/donchian_ema_10m_baseline_backtest.py
Winner: lookback 33, EMA 17/26, TP 2.74%, stop 13.24%.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, asdict
from datetime import time

import numpy as np
import pandas as pd


BASELINE = {"lookback": 33, "fast": 17, "slow": 26, "tp": 0.0274, "stop": 0.1324}
ALLOCATION = 0.95
SESSION_END = time(15, 45)


@dataclass
class Trade:
    signal_time: str
    entry_time: str
    entry_price: float
    qty: int
    exit_time: str
    exit_price: float
    gross_pnl: float
    costs: float
    net_pnl: float
    return_pct: float
    reason: str


def add_features(candles: pd.DataFrame, params: dict = BASELINE) -> pd.DataFrame:
    x = candles.copy().sort_index()
    close = x["close"].astype(float)
    x["ema_fast"] = close.ewm(span=int(params["fast"]), adjust=False).mean()
    x["ema_slow"] = close.ewm(span=int(params["slow"]), adjust=False).mean()
    x["prior_high"] = x["high"].shift(1).rolling(int(params["lookback"])).max()
    x["breakout"] = ((close > x["prior_high"]) & (x["ema_fast"] > x["ema_slow"]) & (close > x["ema_slow"])).fillna(False)
    x["trend_exit"] = ((close < x["ema_slow"]) & (x["ema_fast"] < x["ema_slow"])).fillna(False)
    # Causal downtrend gate: every input is known at the completed signal bar.
    # It requires agreement between slow-EMA drift, log-price regression drift,
    # and distance from the recent peak, so a normal pullback does not get
    # blocked by one noisy candle.
    slope_window = int(params.get("regime_window", 18))
    slope_lag = int(params.get("regime_lag", 12))
    log_close = np.log(close.clip(lower=1e-12))
    x["slow_drift"] = x["ema_slow"].pct_change(slope_lag)
    x["regime_ema"] = close.ewm(span=int(params.get("regime_ema", 89)), adjust=False).mean()
    x["regime_drift"] = x["regime_ema"].pct_change(slope_lag)
    x["log_slope"] = log_close.rolling(slope_window).apply(
        lambda v: float(np.polyfit(np.arange(len(v)), v, 1)[0]), raw=True
    )
    x["recent_peak"] = close.shift(1).rolling(slope_window * 2).max()
    x["peak_distance"] = close / x["recent_peak"] - 1.0
    x["downtrend_blocked"] = (
        (x["slow_drift"] <= -0.002)
        & (x["log_slope"] <= -0.001)
        & (x["peak_distance"] <= -0.015)
        | ((x["regime_drift"] <= -0.001) & (close < x["regime_ema"]))
    ).fillna(False)
    x.attrs["fast"] = int(params["fast"]); x.attrs["slow"] = int(params["slow"])
    return x.dropna(subset=["prior_high"])


def strategy_signal(frame: pd.DataFrame, params: dict = BASELINE) -> dict:
    data = frame.copy()
    if "time_key" in data.columns:
        data["time_key"] = pd.to_datetime(data["time_key"])
        data = data.set_index("time_key")
    x = add_features(data, params)
    if len(x) < 2:
        raise ValueError("not enough completed candles")
    bar = x.iloc[-2]
    bar_time = pd.Timestamp(x.index[-2]).time()
    return {"bar": str(x.index[-2]), "close": float(bar.close), "ema_fast": float(bar.ema_fast), "ema_slow": float(bar.ema_slow), "raw_long_signal": bool(bar.breakout), "downtrend_blocked": bool(bar.downtrend_blocked), "long_signal": bool(bar.breakout) and not bool(bar.downtrend_blocked), "trend_exit": bool(bar.trend_exit), "entry_allowed": bar_time < time(15, 40), "session_exit": bar_time >= time(15, 40)}


def run(candles: pd.DataFrame, params: dict = BASELINE, capital: float = 100000.0, allocation: float = ALLOCATION, slippage_bps: float = 2.0, commission: float = 0.0, evaluation_start=None):
    x = add_features(candles, params)
    rows = list(x.iterrows()); cash = float(capital); position = None; trades = []; equity_rows = []; slip = slippage_bps / 10000.0
    for i, (stamp, bar) in enumerate(rows[:-1]):
        if evaluation_start is not None and stamp < pd.Timestamp(evaluation_start): continue
        next_stamp, next_bar = rows[i + 1]
        mark = cash + (position["qty"] * float(bar.close) if position else 0.0)
        equity_rows.append({"date": stamp, "equity": mark})
        if position:
            entry = position["entry"]; target = entry * (1 + params["tp"]); stop = entry * (1 - params["stop"])
            reason = None; exit_raw = None; exit_time = stamp
            if float(bar.open) >= target: exit_raw, reason = float(bar.open), "TP_GAP"
            elif float(bar.low) <= stop: exit_raw, reason = stop, "STOP"
            elif float(bar.high) >= target: exit_raw, reason = target, "TP"
            elif stamp.time() >= SESSION_END: exit_raw, reason = float(bar.close), "END_OF_SESSION"
            elif bool(bar.trend_exit): exit_raw, reason, exit_time = float(next_bar.open), "TREND_EXIT", next_stamp
            if reason:
                exit_price = exit_raw * (1 - slip); gross = (exit_price - entry) * position["qty"]; costs = commission * 2; net = gross - costs; cash += position["qty"] * exit_price - costs
                trades.append(Trade(position["signal_time"].isoformat(), position["entry_time"].isoformat(), entry, position["qty"], exit_time.isoformat(), exit_price, gross, costs, net, net / (entry * position["qty"]) * 100, reason)); position = None; continue
        allowed = stamp.time() < SESSION_END and next_stamp.date() == stamp.date() and next_stamp.time() < SESSION_END
        gate_on = bool(params.get("downtrend_filter", True))
        if position is None and allowed and bool(bar.breakout) and (not gate_on or not bool(bar.downtrend_blocked)):
            entry = float(next_bar.open) * (1 + slip); qty = math.floor(cash * allocation / entry)
            if qty > 0: cash -= qty * entry + commission; position = {"signal_time": stamp, "entry_time": next_stamp, "entry": entry, "qty": qty}
    if position and rows:
        stamp, bar = rows[-1]; exit_price = float(bar.close) * (1 - slip); gross = (exit_price - position["entry"]) * position["qty"]; costs = commission * 2; net = gross - costs; cash += position["qty"] * exit_price - costs
        trades.append(Trade(position["signal_time"].isoformat(), position["entry_time"].isoformat(), position["entry"], position["qty"], stamp.isoformat(), exit_price, gross, costs, net, net / (position["entry"] * position["qty"]) * 100, "MARK_TO_MARKET"))
        position = None
    equity = pd.DataFrame(equity_rows).set_index("date") if equity_rows else pd.DataFrame(columns=["equity"])
    if position is None and rows and (equity.empty or equity.index[-1] != rows[-1][0]):
        equity.loc[rows[-1][0], "equity"] = cash
    if not equity.empty: equity["drawdown"] = equity.equity / equity.equity.cummax() - 1
    return x, trades, {"equity": equity, "ending_cash": cash}


def _stats(trades, equity, capital):
    returns = np.array([t.net_pnl / (t.entry_price * t.qty) for t in trades]) if trades else np.array([])
    compounded = np.cumprod(1 + returns) if len(returns) else np.array([1.0])
    ending = capital * float(compounded[-1])
    gains = returns[returns > 0].sum() if len(returns) else 0; losses = -returns[returns < 0].sum() if len(returns) else 0
    dd = compounded / np.maximum.accumulate(compounded) - 1
    daily = equity["equity"].resample("1D").last().pct_change(fill_method=None).dropna() if not equity.empty else pd.Series(dtype=float)
    return {
        "starting_capital": round(float(capital), 2),
        "ending_equity": round(ending, 2),
        "total_return_pct": round((ending / capital - 1) * 100, 3),
        "max_drawdown_pct": round(float(dd.min() * 100), 3) if len(dd) else 0,
        "trades": len(trades), "wins": int((returns > 0).sum()),
        "win_rate_pct": round(float((returns > 0).mean() * 100), 2) if len(returns) else 0,
        "profit_factor": round(float(gains / losses), 3) if losses else float("inf"),
        "average_trade_pct": round(float(returns.mean() * 100), 3) if len(returns) else 0,
        "sharpe_approx": round(float(daily.mean() / daily.std() * np.sqrt(252)), 3) if len(daily) > 1 and daily.std() else 0,
    }


def stats(trades, data, capital):
    """Dashboard-compatible metrics using the GitHub winner's compounding rule."""
    return _stats(trades, data["equity"] if isinstance(data, dict) else data, capital)


def optimize(candles, base=BASELINE, capital=100000.0, allocation=ALLOCATION, slippage_bps=2.0, commission=0.0, progress=None, trials=12000, seed=20261112):
    """Same random ranges as the GitHub interval optimizer."""
    rng = random.Random(seed); rows = []
    for done in range(1, int(trials) + 1):
        p = {"lookback": rng.randint(5, 120), "fast": rng.randint(2, 30), "slow": rng.randint(10, 200), "tp": round(rng.uniform(.02, .50), 4), "stop": round(rng.uniform(.01, .25), 4)}
        if p["fast"] >= p["slow"]: continue
        _, trades, data = run(candles, p, capital, allocation, slippage_bps, commission)
        stats = _stats(trades, data["equity"], capital)
        if stats["trades"] >= 3: rows.append({"params": p, "stats": stats})
        if progress and (done % 100 == 0 or done == trials): progress(done, int(trials), f"GitHub 57% random trial {done:,}/{int(trials):,}")
    return sorted(rows, key=lambda r: r["stats"]["total_return_pct"], reverse=True)[:25]


def walk_forward_optimize(candles, base=BASELINE, capital=100000.0, allocation=ALLOCATION, slippage_bps=2.0, commission=0.0, progress=None, trials=12000, seed=20261112, folds=3):
    """Select parameters on each past window and score only the next unseen window."""
    frame = candles.sort_index(); n = len(frame); per_fold = max(1, int(trials) // folds); results = []
    for fold in range(folds):
        train_end = int(n * (0.50 + fold * (0.50 / folds)))
        test_end = int(n * (0.50 + (fold + 1) * (0.50 / folds)))
        if test_end > n: continue
        train = frame.iloc[:train_end]
        local_progress = (lambda done, total, msg, f=fold: progress(f * per_fold + done, trials, f"Walk-forward fold {f + 1}/{folds} · {msg}") if progress else None)
        candidates = optimize(train, base, capital, allocation, slippage_bps, commission, local_progress, per_fold, seed + fold)
        if not candidates: continue
        p = candidates[0]["params"]; test_start = frame.index[train_end]
        _, trades, data = run(frame.iloc[:test_end], p, capital, allocation, slippage_bps, commission, evaluation_start=test_start)
        result = _stats(trades, data["equity"], capital); result.update({"fold": fold + 1, "test_start": str(test_start), "test_end": str(frame.index[test_end - 1])})
        results.append({"params": p, "stats": result})
    if progress: progress(trials, trials, f"Walk-forward complete · {len(results)} unseen folds")
    return results


def walk_forward_replay(candles, fold_results, capital=100000.0, allocation=ALLOCATION,
                        slippage_bps=2.0, commission=0.0):
    """Replay only the already-selected unseen folds, in chronological order."""
    frame = candles.sort_index(); all_trades = []; equity_parts = []; fold_capital = float(capital)
    for row in fold_results:
        p = row["params"]
        start = pd.Timestamp(row["stats"]["test_start"])
        end = pd.Timestamp(row["stats"]["test_end"])
        test_frame = frame.loc[:end]
        x, trades, data = run(test_frame, p, fold_capital, allocation, slippage_bps, commission, evaluation_start=start)
        all_trades.extend(trades)
        if not data["equity"].empty:
            equity_parts.append(data["equity"])
        fold_returns = np.array([t.net_pnl / (t.entry_price * t.qty) for t in trades])
        if len(fold_returns): fold_capital *= float(np.prod(1 + fold_returns))
    equity = pd.concat(equity_parts).sort_index() if equity_parts else pd.DataFrame(columns=["equity", "drawdown"])
    if not equity.empty:
        equity = equity[~equity.index.duplicated(keep="last")]
        equity["drawdown"] = equity.equity / equity.equity.cummax() - 1
    return add_features(frame, BASELINE), all_trades, {"equity": equity, "ending_cash": fold_capital}
