"""Shared implementation of the frozen Moomoo Quant 10-minute baseline."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd

try:
    from numba import njit, prange
except ImportError:  # The reference engine still works without Numba.
    njit = None
    prange = range


BASELINE = {"lookback": 45, "fast": 8, "slow": 89, "regime_slow_slope": 18, "regime_fast_slope": 3, "regime_confirm": 3, "tp": 0.1069, "stop": 0.0354, "break_even_trigger": 0.02, "break_even_lock": 0.002, "profit_lock_trigger": 0.08, "profit_lock": 0.01, "trail_trigger": 0.06, "trail_pct": 0.02}
ALLOCATION = 0.95


@dataclass
class BaselineTrade:
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


if njit is not None:
    @njit(parallel=True, cache=True)
    def _random_search_kernel(opens, highs, closes, params, capital, allocation, slip, commission):
        n_trials = params.shape[0]
        n = closes.shape[0]
        metrics = np.zeros((n_trials, 5), dtype=np.float64)
        for trial in prange(n_trials):
            lookback = int(params[trial, 0]); fast = int(params[trial, 1]); slow = int(params[trial, 2])
            regime_slow = int(params[trial, 3]); regime_fast = int(params[trial, 4]); regime_confirm = int(params[trial, 5])
            tp = params[trial, 6]; stop_pct = params[trial, 7]; be_trigger = params[trial, 8]
            be_lock = params[trial, 9]; lock_trigger = params[trial, 10]; profit_lock = params[trial, 11]
            trail_trigger = params[trial, 12]; trail_pct = params[trial, 13]
            cash = capital; qty = 0; entry = 0.0; peak = 0.0; be = False; lock = False
            equity_peak = capital; min_drawdown = 0.0; trades = 0; wins = 0; gains = 0.0; losses = 0.0
            ema_fast = 0.0; ema_slow = 0.0; prev_fast = 2.0 / (fast + 1.0); prev_slow = 2.0 / (slow + 1.0)
            fast_hist = np.empty(n, dtype=np.float64); slow_hist = np.empty(n, dtype=np.float64)
            for i in range(n):
                if i == 0:
                    ema_fast = closes[i]; ema_slow = closes[i]
                else:
                    ema_fast = closes[i] * prev_fast + ema_fast * (1.0 - prev_fast)
                    ema_slow = closes[i] * prev_slow + ema_slow * (1.0 - prev_slow)
                fast_hist[i] = ema_fast; slow_hist[i] = ema_slow
                mark = cash + (qty * closes[i] if qty > 0 else 0.0)
                if mark > equity_peak: equity_peak = mark
                dd = mark / equity_peak - 1.0
                if dd < min_drawdown: min_drawdown = dd
                if i < lookback: continue
                prior_high = highs[i - lookback]
                for j in range(i - lookback + 1, i):
                    if highs[j] > prior_high: prior_high = highs[j]
                regime_up = i >= regime_slow + regime_confirm - 1 and i >= regime_fast + regime_confirm - 1
                if regime_up:
                    for k in range(regime_confirm):
                        j = i - k
                        if not (slow_hist[j] > slow_hist[j - regime_slow] and fast_hist[j] > fast_hist[j - regime_fast] and closes[j] > slow_hist[j]):
                            regime_up = False
                            break
                long_signal = closes[i] > prior_high and ema_fast > ema_slow and closes[i] > ema_slow and regime_up
                if qty > 0:
                    if closes[i] > peak: peak = closes[i]
                    if entry > peak: peak = entry
                    if closes[i] >= entry * (1.0 + be_trigger): be = True
                    if peak >= entry * (1.0 + lock_trigger): lock = True
                    stop = entry * (1.0 - stop_pct)
                    if be and entry * (1.0 + be_lock) > stop: stop = entry * (1.0 + be_lock)
                    if lock and entry * (1.0 + profit_lock) > stop: stop = entry * (1.0 + profit_lock)
                    if peak >= entry * (1.0 + trail_trigger) and peak * (1.0 - trail_pct) > stop: stop = peak * (1.0 - trail_pct)
                    yellow = closes[i] < ema_fast and closes[i] >= ema_slow and ema_fast > ema_slow
                    red = closes[i] < ema_slow
                    reason = closes[i] >= entry * (1.0 + tp) or closes[i] <= stop or yellow or red
                    if reason and i < n - 1:
                        exit_price = opens[i + 1] * (1.0 - slip)
                        net = (exit_price - entry) * qty - commission * 2.0
                        cash += qty * exit_price - commission * 2.0
                        trades += 1
                        if net > 0: wins += 1; gains += net / (entry * qty)
                        else: losses -= net / (entry * qty)
                        qty = 0; entry = 0.0; peak = 0.0; be = False; lock = False
                        continue
                if qty == 0 and long_signal and i < n - 1:
                    entry = opens[i + 1] * (1.0 + slip)
                    qty = int(cash * allocation / entry)
                    if qty > 0: cash -= qty * entry + commission; peak = entry
            if qty > 0:
                exit_price = closes[n - 1] * (1.0 - slip)
                net = (exit_price - entry) * qty - commission * 2.0
                cash += qty * exit_price - commission * 2.0
                trades += 1
                if net > 0: wins += 1; gains += net / (entry * qty)
                else: losses -= net / (entry * qty)
            metrics[trial, 0] = (cash / capital - 1.0) * 100.0
            metrics[trial, 1] = min_drawdown * 100.0
            metrics[trial, 2] = trades
            metrics[trial, 3] = (wins / trades * 100.0) if trades else 0.0
            metrics[trial, 4] = gains / losses if losses > 0 else 0.0
        return metrics


def add_features(candles: pd.DataFrame, params: dict) -> pd.DataFrame:
    x = candles.copy().sort_index()
    close = x["close"].astype(float)
    x["ema_fast"] = close.ewm(span=int(params["fast"]), adjust=False).mean()
    x["ema_slow"] = close.ewm(span=int(params["slow"]), adjust=False).mean()
    x["prior_high"] = x["high"].shift(1).rolling(int(params["lookback"])).max()
    raw_regime_up = (x["ema_slow"] > x["ema_slow"].shift(int(params.get("regime_slow_slope", 18)))) & (x["ema_fast"] > x["ema_fast"].shift(int(params.get("regime_fast_slope", 3)))) & (close > x["ema_slow"])
    confirm = int(params.get("regime_confirm", 6))
    regime_up = raw_regime_up.astype(int).rolling(confirm, min_periods=confirm).min().fillna(0).astype(bool)
    x["regime_up"] = regime_up
    x["long_signal"] = (close > x["prior_high"]) & (x["ema_fast"] > x["ema_slow"]) & (close > x["ema_slow"]) & regime_up
    x["yellow_exit"] = (close < x["ema_fast"]) & (close >= x["ema_slow"]) & (x["ema_fast"] > x["ema_slow"])
    x["red_exit"] = close < x["ema_slow"]
    return x.dropna(subset=["prior_high", "ema_fast", "ema_slow"])


def strategy_signal(frame: pd.DataFrame, params: dict = BASELINE) -> dict:
    """The live bot's newest completed-bar signal."""
    data = frame.copy()
    if "time_key" in data.columns:
        data["time_key"] = pd.to_datetime(data["time_key"])
        data = data.set_index("time_key")
    x = add_features(data, params)
    if len(x) < 2:
        raise ValueError("not enough completed candles")
    bar = x.iloc[-2]
    stamp = x.index[-2]
    return {"bar": str(stamp), "close": float(bar.close), "ema_fast": float(bar.ema_fast), "ema_slow": float(bar.ema_slow), "regime_up": bool(bar.regime_up), "long_signal": bool(bar.long_signal), "yellow_exit": bool(bar.yellow_exit), "red_exit": bool(bar.red_exit)}


def run_baseline(candles: pd.DataFrame, params: dict = BASELINE, capital: float = 100000.0, allocation: float = ALLOCATION, slippage_bps: float = 2.0, commission: float = 0.0, evaluation_start=None):
    x = add_features(candles, params)
    rows = list(x.iterrows())
    cash = float(capital)
    position = None
    trades = []
    equity_rows = []
    slip = slippage_bps / 10000.0

    for i, (stamp, bar) in enumerate(rows):
        if evaluation_start is not None and stamp < pd.Timestamp(evaluation_start):
            continue
        mark = cash + (position["qty"] * float(bar.close) if position else 0.0)
        equity_rows.append({"date": stamp, "equity": mark, "cash": cash, "position": position["qty"] if position else 0})
        if i >= len(rows) - 1:
            continue
        next_stamp, next_bar = rows[i + 1]
        if position:
            entry = position["entry"]
            position["peak"] = max(position["peak"], float(bar.close), entry)
            position["be"] |= float(bar.close) >= entry * (1 + params["break_even_trigger"])
            position["lock"] |= position["peak"] >= entry * (1 + params["profit_lock_trigger"])
            stop = entry * (1 - params["stop"])
            if position["be"]: stop = max(stop, entry * (1 + params["break_even_lock"]))
            if position["lock"]: stop = max(stop, entry * (1 + params["profit_lock"]))
            if position["peak"] >= entry * (1 + params["trail_trigger"]): stop = max(stop, position["peak"] * (1 - params["trail_pct"]))
            reason = None
            if float(bar.close) >= entry * (1 + params["tp"]):
                reason = "TAKE_PROFIT"
            elif float(bar.close) <= stop:
                reason = "STOP_LOSS"
            elif bool(bar.yellow_exit):
                reason = "YELLOW_EXIT"
            elif bool(bar.red_exit):
                reason = "RED_EXIT"
            if reason:
                exit_price = float(next_bar.open) * (1 - slip)
                gross = (exit_price - entry) * position["qty"]
                costs = commission * 2
                net = gross - costs
                cash += position["qty"] * exit_price - costs
                trades.append(BaselineTrade(position["signal_time"].isoformat(), position["entry_time"].isoformat(), entry, position["qty"], next_stamp.isoformat(), exit_price, gross, costs, net, net / (entry * position["qty"]) * 100, reason))
                position = None
                continue
        if position is None and bool(bar.long_signal):
            entry = float(next_bar.open) * (1 + slip)
            qty = math.floor(cash * allocation / entry)
            if qty > 0:
                cash -= qty * entry + commission
                position = {"signal_time": stamp, "entry_time": next_stamp, "entry": entry, "qty": qty, "peak": entry, "be": False, "lock": False}

    if position and rows:
        stamp, bar = rows[-1]
        exit_price = float(bar.close) * (1 - slip)
        gross = (exit_price - position["entry"]) * position["qty"]
        costs = commission * 2
        net = gross - costs
        cash += position["qty"] * exit_price - costs
        trades.append(BaselineTrade(position["signal_time"].isoformat(), position["entry_time"].isoformat(), position["entry"], position["qty"], stamp.isoformat(), exit_price, gross, costs, net, net / (position["entry"] * position["qty"]) * 100, "MARK_TO_MARKET"))

    equity = pd.DataFrame(equity_rows).set_index("date")
    if not equity.empty:
        equity["drawdown"] = equity.equity / equity.equity.cummax() - 1
    return x, trades, {"equity": equity, "ending_cash": cash}


def _run_stats(trades, equity, capital):
    returns = np.array([t.net_pnl / (t.entry_price * t.qty) for t in trades]) if trades else np.array([])
    gains = returns[returns > 0].sum() if len(returns) else 0.0
    losses = -returns[returns < 0].sum() if len(returns) else 0.0
    ending = float(equity.equity.iloc[-1]) if not equity.empty else float(capital)
    return {"total_return_pct": round((ending / capital - 1) * 100, 3), "max_drawdown_pct": round(float(equity.drawdown.min() * 100), 3) if not equity.empty else 0.0, "trades": len(trades), "win_rate_pct": round(float((returns > 0).mean() * 100), 2) if len(returns) else 0.0, "profit_factor": round(float(gains / losses), 3) if losses else 0.0}


def walk_forward_optimize(candles, base=BASELINE, capital=100000.0, allocation=ALLOCATION, slippage_bps=2.0, commission=0.0, progress=None, trials=12_000, seed=20261113, folds=3):
    """Random-search training folds, then score selected parameters OOS only."""
    frame = candles.sort_index()
    n = len(frame)
    fold_trials = max(1, int(trials) // int(folds))
    results = []
    for fold in range(int(folds)):
        train_end = int(n * (0.50 + fold * 0.12))
        test_end = int(n * (0.66 + fold * 0.12))
        if test_end >= n or train_end <= 500:
            continue
        train = frame.iloc[:train_end]
        test_start = frame.index[train_end]
        selected = optimize_baseline(train, base, capital, allocation, slippage_bps, commission,
            progress=(lambda done, total, message, f=fold: progress(f * fold_trials + done, int(trials), f"Walk-forward fold {f + 1}/{folds} · {message}") if progress else None),
            trials=fold_trials, seed=seed + fold)
        if not selected:
            continue
        best = selected[0]["params"]
        _, oos_trades, oos_data = run_baseline(frame.iloc[:test_end], best, capital, allocation, slippage_bps, commission, evaluation_start=test_start)
        oos_stats = _run_stats(oos_trades, oos_data["equity"], capital)
        oos_stats["fold"] = fold + 1
        oos_stats["training_end"] = str(train.index[-1])
        oos_stats["test_start"] = str(test_start)
        oos_stats["test_end"] = str(frame.index[test_end - 1])
        results.append({"params": best, "stats": oos_stats})
    if progress:
        progress(int(trials), int(trials), f"Walk-forward complete · {len(results)} unseen test folds")
    return sorted(results, key=lambda row: row["stats"]["total_return_pct"], reverse=True)


def optimize_baseline(candles, base=BASELINE, capital=100000.0, allocation=ALLOCATION, slippage_bps=2.0, commission=0.0, progress=None, trials=12_000, seed=20261113):
    """Run a reproducible random search and return its top 25 configurations.

    The random ranges cover the entry, regime, and profit-lock parameters. The
    production baseline remains the starting configuration; optimisation does
    not change it or the live bot automatically.
    """
    trials = int(trials)
    rng = random.Random(seed)
    generated = []
    for _ in range(trials):
        params = {**base,
            "lookback": rng.randint(20, 100),
            "fast": rng.randint(3, 20),
            "slow": rng.randint(40, 200),
            "regime_slow_slope": rng.randint(6, 36),
            "regime_fast_slope": rng.randint(2, 9),
            "regime_confirm": rng.randint(2, 9),
            "tp": round(rng.uniform(.04, .15), 5),
            "stop": round(rng.uniform(.02, .06), 5),
            "break_even_trigger": round(rng.uniform(.01, .04), 5),
            "break_even_lock": round(rng.uniform(.001, .01), 5),
            "profit_lock_trigger": round(rng.uniform(.05, .12), 5),
            "profit_lock": round(rng.uniform(.005, .03), 5),
            "trail_trigger": round(rng.uniform(.04, .10), 5),
            "trail_pct": round(rng.uniform(.01, .04), 5),
        }
        generated.append(params)
    if njit is not None:
        matrix = np.array([[p[k] for k in ("lookback", "fast", "slow", "regime_slow_slope", "regime_fast_slope", "regime_confirm", "tp", "stop", "break_even_trigger", "break_even_lock", "profit_lock_trigger", "profit_lock", "trail_trigger", "trail_pct")] for p in generated], dtype=np.float64)
        arrays = (candles.sort_index()["open"].to_numpy(dtype=np.float64), candles.sort_index()["high"].to_numpy(dtype=np.float64), candles.sort_index()["close"].to_numpy(dtype=np.float64))
        metric_parts = []
        chunk_size = 500
        for start in range(0, trials, chunk_size):
            stop = min(start + chunk_size, trials)
            metric_parts.append(_random_search_kernel(*arrays, matrix[start:stop], capital, allocation, slippage_bps / 10000.0, commission))
            if progress:
                progress(stop, trials, f"Native Apple Silicon optimisation {stop:,}/{trials:,}")
        metrics = np.vstack(metric_parts)
        # Prefer robust equity curves over a high-return/high-drawdown outlier.
        # Drawdown is negative, so this penalises deep losses while retaining
        # return as the primary objective.
        risk_adjusted_score = metrics[:, 0] + 0.75 * metrics[:, 1]
        order = np.argsort(risk_adjusted_score)[::-1]
        top = []
        for done, idx in enumerate(order, 1):
            if metrics[idx, 2] < 3: continue
            p = generated[int(idx)]; m = metrics[idx]
            top.append({"params": p, "stats": {"total_return_pct": round(float(m[0]), 3), "max_drawdown_pct": round(float(m[1]), 3), "risk_adjusted_score": round(float(risk_adjusted_score[idx]), 3), "trades": int(m[2]), "win_rate_pct": round(float(m[3]), 2), "profit_factor": round(float(m[4]), 3)}})
            if len(top) >= 25: break
        if progress: progress(trials, trials, f"Random optimisation complete · {trials:,}/{trials:,}")
        return top
    return []
