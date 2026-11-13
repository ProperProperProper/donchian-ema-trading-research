#!/usr/bin/env python3
"""Local Moomoo-Quant-style backtester and HTML report generator.

The engine is deliberately conservative: indicators are calculated on the
completed bar, orders fill on the next bar's open, and a position cannot be
re-entered until the prior exit is complete. It implements the current
Donchian/EMA profit-lock/yellow-exit strategy and produces a self-contained
HTML report with charts and detailed trade statistics.
"""

from __future__ import annotations

import argparse
import base64
import io
import itertools
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle


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
    qty: int
    exit_time: str
    exit_price: float
    gross_pnl: float
    costs: float
    net_pnl: float
    return_pct: float
    reason: str


def load_candles(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    frame = pd.read_parquet(path) if path.suffix.lower() in {".parquet", ".pq"} else pd.read_csv(path)
    lower = {str(c).strip().lower(): c for c in frame.columns}
    date_col = next((lower[x] for x in ("date", "datetime", "time", "time_key") if x in lower), None)
    if date_col is None:
        raise ValueError("Input needs date, datetime, time, or time_key column")
    rename = {date_col: "date"}
    for name in ("open", "high", "low", "close", "volume"):
        if name in lower:
            rename[lower[name]] = name
    frame = frame.rename(columns=rename)
    missing = [x for x in ("open", "high", "low", "close") if x not in frame]
    if missing:
        raise ValueError(f"Input is missing OHLC columns: {', '.join(missing)}")
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    for name in ("open", "high", "low", "close", "volume"):
        if name in frame:
            frame[name] = pd.to_numeric(frame[name], errors="coerce")
    return frame.dropna(subset=["date", "open", "high", "low", "close"]).sort_values("date").drop_duplicates("date").set_index("date")


def add_signals(candles: pd.DataFrame, p: dict) -> pd.DataFrame:
    x = candles.copy()
    x["ema_fast"] = x.close.ewm(span=p["fast"], adjust=False).mean()
    x["ema_slow"] = x.close.ewm(span=p["slow"], adjust=False).mean()
    x["prior_high"] = x.high.shift(1).rolling(p["lookback"]).max()
    x["long_signal"] = (x.close > x.prior_high) & (x.ema_fast > x.ema_slow) & (x.close > x.ema_slow)
    x["yellow_exit"] = (x.close < x.ema_fast) & (x.close >= x.ema_slow) & (x.ema_fast > x.ema_slow)
    x["red_exit"] = x.close < x.ema_slow
    return x.dropna(subset=["prior_high"])


def run(candles: pd.DataFrame, p: dict, capital: float, allocation: float,
        slippage_bps: float, commission: float) -> tuple[pd.DataFrame, list[Trade], dict]:
    x = add_signals(candles, p)
    rows = list(x.iterrows())
    cash, position, trades, equity_rows = capital, None, [], []
    slip = slippage_bps / 10000.0
    for i, (stamp, bar) in enumerate(rows):
        mark = cash + (position["qty"] * float(bar.close) if position else 0.0)
        equity_rows.append({"date": stamp, "equity": mark, "cash": cash, "position": position["qty"] if position else 0})
        if i >= len(rows) - 1:
            continue
        next_stamp, next_bar = rows[i + 1]
        if position:
            price = float(bar.close)
            position["peak"] = max(position["peak"], price)
            position["be"] |= price >= position["entry"] * (1 + p["break_even_trigger"])
            position["lock"] |= position["peak"] >= position["entry"] * (1 + p["profit_lock_trigger"])
            stop = position["entry"] * (1 - p["initial_stop"])
            if position["be"]: stop = max(stop, position["entry"] * 1.002)
            if position["lock"]: stop = max(stop, position["entry"] * (1 + p["profit_lock"]))
            if position["peak"] >= position["entry"] * (1 + p["trail_trigger"]):
                stop = max(stop, position["peak"] * (1 - p["trail_pct"]))
            reason = None
            if price >= position["entry"] * (1 + p["target"]): reason = "TARGET"
            elif price <= stop: reason = "STOP"
            elif p["yellow_exit"] and bool(bar.yellow_exit): reason = "YELLOW"
            elif bool(bar.red_exit): reason = "RED"
            if reason:
                exit_price = float(next_bar.open) * (1 - slip)
                gross = (exit_price - position["entry"]) * position["qty"]
                costs = commission * 2
                net = gross - costs
                cash += position["qty"] * exit_price - costs
                trades.append(Trade(position["signal_time"].isoformat(), position["entry_time"].isoformat(), position["entry"], position["qty"], next_stamp.isoformat(), exit_price, gross, costs, net, net / (position["entry"] * position["qty"]) * 100, reason))
                position = None
                continue
        if position is None and bool(bar.long_signal):
            entry = float(next_bar.open) * (1 + slip)
            qty = math.floor(cash * allocation / entry)
            if qty > 0:
                cost = qty * entry + commission
                cash -= cost
                position = {"signal_time": stamp, "entry_time": next_stamp, "entry": entry, "qty": qty, "peak": entry, "be": False, "lock": False}
    if position:
        stamp, bar = rows[-1]
        exit_price = float(bar.close) * (1 - slip)
        gross = (exit_price - position["entry"]) * position["qty"]
        costs = commission * 2
        net = gross - costs
        cash += position["qty"] * exit_price - commission
        trades.append(Trade(position["signal_time"].isoformat(), position["entry_time"].isoformat(), position["entry"], position["qty"], stamp.isoformat(), exit_price, gross, costs, net, net / (position["entry"] * position["qty"]) * 100, "MARK_TO_MARKET"))
    equity = pd.DataFrame(equity_rows).set_index("date")
    if not equity.empty:
        equity["equity"] = equity.equity.ffill()
        equity["drawdown"] = equity.equity / equity.equity.cummax() - 1
    return x, trades, {"equity": equity, "ending_cash": cash}


def stats(trades: list[Trade], equity: pd.DataFrame, capital: float) -> dict:
    returns = np.array([t.net_pnl / (t.entry_price * t.qty) for t in trades]) if trades else np.array([])
    eq = equity.equity if not equity.empty else pd.Series([capital])
    daily = eq.resample("1D").last().pct_change(fill_method=None).dropna() if len(eq.index) > 1 else pd.Series(dtype=float)
    gains = returns[returns > 0].sum() if len(returns) else 0
    losses = -returns[returns < 0].sum() if len(returns) else 0
    return {
        "starting_capital": round(capital, 2), "ending_equity": round(float(eq.iloc[-1]), 2),
        "total_return_pct": round(float((eq.iloc[-1] / capital - 1) * 100), 3),
        "max_drawdown_pct": round(float(equity.drawdown.min() * 100), 3) if not equity.empty else 0,
        "trades": len(trades), "wins": int((returns > 0).sum()),
        "win_rate_pct": round(float((returns > 0).mean() * 100), 2) if len(returns) else 0,
        "profit_factor": round(float(gains / losses), 3) if losses else float("inf"),
        "average_trade_pct": round(float(returns.mean() * 100), 3) if len(returns) else 0,
        "sharpe_approx": round(float(daily.mean() / daily.std() * np.sqrt(252)), 3) if len(daily) > 1 and daily.std() else 0,
    }


def _png(fig) -> str:
    buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=140, bbox_inches="tight"); plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def charts(x: pd.DataFrame, trades: list[Trade], equity: pd.DataFrame) -> dict[str, str]:
    # A clean full-period view: close/indicators, blocked down-regimes, and
    # drawdown. This stays self-contained in the dashboard as a PNG.
    plt.rcParams.update({"font.size": 9, "axes.titlesize": 13, "axes.labelsize": 9})
    fig, (ax, ax2, ax3) = plt.subplots(3, 1, figsize=(16, 10), sharex=True,
        gridspec_kw={"height_ratios": [4, .55, 1.25], "hspace": .06})
    ax.plot(x.index, x.close, label="APP close", color="#172b4d", linewidth=1.05)
    fast_label = f"EMA {x.attrs.get('fast', '')}".strip()
    slow_label = f"EMA {x.attrs.get('slow', '')}".strip()
    ax.plot(x.index, x.ema_fast, label=fast_label, color="#168aad", linewidth=1.0)
    ax.plot(x.index, x.ema_slow, label=slow_label, color="#c4457b", linewidth=1.0)
    ax.plot(x.index, x.prior_high, label="Donchian prior high", color="#e09f3e", linewidth=.85, alpha=.9)
    if "downtrend_blocked" in x:
        down = x.downtrend_blocked.fillna(False).astype(bool)
        ax.fill_between(x.index, x.close.min(), x.close.max(), where=down, color="#e63946", alpha=.055, label="Math downtrend gate")
        ax2.fill_between(x.index, 0, 1, where=down, color="#e63946", alpha=.72, step="mid")
    elif "regime_up" in x:
        down = ~x.regime_up.fillna(False).astype(bool)
        ax.fill_between(x.index, x.close.min(), x.close.max(), where=down, color="#e63946", alpha=.055, label="Down-regime blocked")
        ax2.fill_between(x.index, 0, 1, where=down, color="#e63946", alpha=.72, step="mid")
    for t in trades:
        ax.scatter(pd.Timestamp(t.entry_time), t.entry_price, marker="^", color="#078a62", edgecolors="white", linewidths=.5, s=48, zorder=5)
        ax.scatter(pd.Timestamp(t.exit_time), t.exit_price, marker="v", color="#d62839", edgecolors="white", linewidths=.5, s=42, zorder=5)
    ax.set_title("US.APP · 10-minute Donchian / EMA strategy")
    ax.set_ylabel("Price"); ax.legend(loc="upper left", ncol=4, framealpha=.92)
    ax.grid(axis="y", alpha=.18); ax.spines[["top", "right"]].set_visible(False)
    ax2.set_ylim(0, 1); ax2.set_yticks([]); ax2.set_ylabel("Regime", rotation=0, labelpad=28, va="center")
    ax2.spines[["top", "right", "left"]].set_visible(False)
    ax3.plot(equity.index, equity.drawdown * 100, color="#d62839", linewidth=1.15)
    ax3.fill_between(equity.index, equity.drawdown * 100, 0, color="#d62839", alpha=.14)
    ax3.set_ylabel("Drawdown %"); ax3.grid(axis="y", alpha=.18); ax3.spines[["top", "right"]].set_visible(False)
    price = _png(fig)
    fig, ax = plt.subplots(figsize=(16, 4.5)); ax.plot(equity.index, equity.equity, color="#1864ab", linewidth=1.3); ax.fill_between(equity.index, equity.equity, equity.equity.min(), color="#1864ab", alpha=.08); ax.set_title("Equity curve"); ax.set_ylabel("Account equity"); ax.grid(axis="y", alpha=.18); ax.spines[["top", "right"]].set_visible(False); equity_img = _png(fig)
    monthly = equity.equity.resample("ME").last().pct_change().dropna() * 100
    fig, ax = plt.subplots(figsize=(16, 3.7)); ax.bar(monthly.index, monthly, width=20, color=np.where(monthly >= 0, "#20a464", "#d9534f")); ax.axhline(0, color="#52606d", linewidth=.7); ax.set_title("Monthly returns"); ax.set_ylabel("Return %"); ax.grid(axis="y", alpha=.18); ax.spines[["top", "right"]].set_visible(False); monthly_img = _png(fig)
    return {"price": price, "equity": equity_img, "monthly": monthly_img}


def report(path: Path, symbol: str, p: dict, x: pd.DataFrame, trades: list[Trade], run_data: dict, capital: float, optimization: list[dict] | None) -> None:
    summary = stats(trades, run_data["equity"], capital)
    imgs = charts(x, trades, run_data["equity"])
    trade_rows = "".join(f"<tr>{''.join(f'<td>{getattr(t, field)}</td>' for field in ('entry_time','entry_price','qty','exit_time','exit_price','net_pnl','return_pct','reason'))}</tr>" for t in trades)
    opt_rows = "" if not optimization else "".join(f"<tr><td>{i+1}</td><td>{r['metrics']['total_return_pct']}</td><td>{r['metrics']['max_drawdown_pct']}</td><td>{r['metrics']['trades']}</td><td><code>{json.dumps(r['params'])}</code></td></tr>" for i,r in enumerate(optimization[:10]))
    cards = "".join(f"<div class='card'><span>{k.replace('_',' ')}</span><strong>{v}</strong></div>" for k,v in summary.items())
    html = f"""<!doctype html><meta charset='utf-8'><title>{symbol} Local Quant Report</title><style>body{{font-family:-apple-system,BlinkMacSystemFont,Segoe UI,sans-serif;background:#f5f7fa;color:#17202a;max-width:1500px;margin:24px auto;padding:0 20px}}h1,h2{{margin-bottom:8px}}.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px}}.card{{background:white;border:1px solid #dfe4ea;border-radius:8px;padding:12px}}.card span{{display:block;color:#667085;text-transform:capitalize;font-size:12px}}.card strong{{font-size:20px}}.panel{{background:white;border:1px solid #dfe4ea;border-radius:8px;padding:15px;margin:16px 0;overflow:auto}}img{{max-width:100%}}table{{border-collapse:collapse;width:100%;font-size:13px}}td,th{{border-bottom:1px solid #e5e7eb;padding:7px;text-align:right}}td:first-child,th:first-child{{text-align:left}}code{{font-size:11px}}</style><h1>{symbol} Local Quant Backtest</h1><p>Completed-bar signals; next-bar-open fills; one long position. Parameters: <code>{json.dumps(p)}</code></p><div class='cards'>{cards}</div><div class='panel'><h2>Price and trade map</h2><img src='data:image/png;base64,{imgs['price']}'></div><div class='panel'><h2>Equity curve</h2><img src='data:image/png;base64,{imgs['equity']}'></div><div class='panel'><h2>Monthly returns</h2><img src='data:image/png;base64,{imgs['monthly']}'></div><div class='panel'><h2>Trades</h2><table><tr><th>Entry</th><th>Entry price</th><th>Qty</th><th>Exit</th><th>Exit price</th><th>Net P/L</th><th>Return %</th><th>Reason</th></tr>{trade_rows}</table></div><div class='panel'><h2>Optimization</h2><table><tr><th>Rank</th><th>Return %</th><th>Max DD %</th><th>Trades</th><th>Parameters</th></tr>{opt_rows}</table></div>"""
    path.write_text(html)


def main() -> None:
    ap = argparse.ArgumentParser(description="Moomoo-Quant-style local backtester")
    ap.add_argument("--csv", required=True); ap.add_argument("--symbol", default="ASX")
    ap.add_argument("--output-dir", default="local_quant_report"); ap.add_argument("--optimize", action="store_true")
    ap.add_argument("--capital", type=float, default=100000); ap.add_argument("--allocation", type=float, default=.95)
    ap.add_argument("--slippage-bps", type=float, default=2); ap.add_argument("--commission", type=float, default=0)
    for key, value in DEFAULTS.items():
        if isinstance(value, bool): ap.add_argument(f"--{key.replace('_','-')}", action="store_true", default=value)
        elif isinstance(value, int): ap.add_argument(f"--{key.replace('_','-')}", type=int, default=value)
        else: ap.add_argument(f"--{key.replace('_','-')}", type=float, default=value)
    args = ap.parse_args(); p={key:getattr(args,key) for key in DEFAULTS}; candles=load_candles(args.csv)
    x,trades,data=run(candles,p,args.capital,args.allocation,args.slippage_bps,args.commission); optimization=None
    if args.optimize:
        grid={'lookback':[20,30,45,60],'fast':[5,8,13],'slow':[30,50,89],'target':[.08,.1069,.15],'initial_stop':[.03,.0354,.05],'break_even_trigger':[.02,.03],'profit_lock_trigger':[.06,.08],'trail_trigger':[.04,.06],'trail_pct':[.02,.03],'yellow_exit':[True,False]}; optimization=[]
        for values in itertools.product(*(grid[k] for k in grid)):
            q={**p,**dict(zip(grid,values))}
            if q['fast'] >= q['slow']: continue
            _,ts,rd=run(candles,q,args.capital,args.allocation,args.slippage_bps,args.commission); optimization.append({'params':q,'metrics':stats(ts,rd['equity'],args.capital)})
        optimization.sort(key=lambda z:z['metrics']['total_return_pct'],reverse=True)
    out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True); report(out/'report.html',args.symbol,p,x,trades,data,args.capital,optimization)
    pd.DataFrame([asdict(t) for t in trades]).to_csv(out/'trades.csv',index=False); print(json.dumps(stats(trades,data['equity'],args.capital),indent=2)); print(f"Report: {out/'report.html'}")


if __name__ == '__main__': main()
