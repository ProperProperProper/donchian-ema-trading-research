#!/usr/bin/env python3
"""Local browser portal for the Quant-style backtester.

Start with ``python3 BBI_5m_Backtest/local_quant_portal.py`` and open the URL
printed in the terminal. The server is local-only and never places orders.
"""

from __future__ import annotations

import base64
import io
import json
import threading
from dataclasses import asdict
from datetime import date, timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf

from local_quant_tester import DEFAULTS, run, stats


UNIVERSE_URL = "https://raw.githubusercontent.com/siifmelbourne/ASX200_Screener/main/asx200.csv"
COMPARISON_SYMBOLS = ["APP"]  # US.APP; kept separate from the ASX 200 universe.
_universe_cache: list[str] | None = None
_cache_lock = threading.Lock()


def universe() -> list[str]:
    global _universe_cache
    with _cache_lock:
        if _universe_cache is None:
            try:
                import requests
                text = requests.get(UNIVERSE_URL, timeout=20).text
                values = pd.read_csv(io.StringIO(text))["NAME"].dropna().astype(str).str.strip().str.upper().tolist()
                _universe_cache = sorted(x for x in values if x and x != "NAME")
                _universe_cache = COMPARISON_SYMBOLS + _universe_cache
            except Exception:
                _universe_cache = ["AMP", "CBA", "CSL", "CDA", "BHP", "WDS", "RIO", "NAB", "WBC", "ANZ", "MQG", "WOW", "TLS", "FMG", "GMG"]
        return _universe_cache


def candles(symbols: list[str], start: str, end: str) -> dict[str, pd.DataFrame]:
    tickers = [s if s in COMPARISON_SYMBOLS else f"{s}.AX" for s in symbols]
    data = yf.download(tickers, start=start, end=end, interval="1d", auto_adjust=False, progress=False, threads=True, group_by="ticker", timeout=30)
    result = {}
    for symbol, ticker in zip(symbols, tickers):
        try:
            frame = data[ticker].copy() if isinstance(data.columns, pd.MultiIndex) else data.copy()
            frame.columns = [str(c).lower() for c in frame.columns]
            frame = frame[["open", "high", "low", "close"]].dropna()
            frame.index = pd.to_datetime(frame.index)
            result[symbol] = frame[~frame.index.duplicated()].sort_index()
        except Exception:
            continue
    return result


def plot_image(frame: pd.DataFrame, trades) -> str:
    x = frame.copy(); x["ema_fast"] = x.close.ewm(span=8, adjust=False).mean(); x["ema_slow"] = x.close.ewm(span=89, adjust=False).mean()
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(14, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    for stamp, row in x.iterrows():
        color = "#28a36a" if row.close >= row.open else "#e05252"
        ax.vlines(stamp, row.low, row.high, color=color, linewidth=.7)
        ax.bar(stamp, row.close-row.open, bottom=row.open, width=.65, color=color, alpha=.8)
    ax.plot(x.index, x.ema_fast, color="#2f80ed", label="EMA 8")
    ax.plot(x.index, x.ema_slow, color="#d63384", label="EMA 89")
    for trade in trades:
        ax.scatter(pd.Timestamp(trade.entry_time), trade.entry_price, marker="^", color="#087f5b", s=60)
        ax.scatter(pd.Timestamp(trade.exit_time), trade.exit_price, marker="v", color="#c92a2a", s=50)
    ax.legend(); ax.grid(alpha=.2); ax.set_title("Price and trades")
    equity = [100000]
    for trade in trades: equity.append(equity[-1] * (1 + trade.return_pct / 100))
    ax2.plot(range(len(equity)), equity, color="#1864ab"); ax2.set_title("Trade equity (indexed)"); ax2.grid(alpha=.2)
    buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=120, bbox_inches="tight"); plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def params_from(payload: dict) -> dict:
    p = DEFAULTS.copy()
    for key in p:
        if key in payload:
            p[key] = bool(payload[key]) if isinstance(p[key], bool) else type(p[key])(payload[key])
    return p


def optimize(frame: pd.DataFrame, base: dict, capital: float, allocation: float, slip: float, commission: float) -> list[dict]:
    grid = {"lookback": [20, 30, 45, 60], "fast": [5, 8, 13], "slow": [30, 50, 89], "target": [.08, .1069, .15], "initial_stop": [.03, .0354, .05], "break_even_trigger": [.02, .03], "profit_lock_trigger": [.06, .08], "trail_trigger": [.04, .06], "trail_pct": [.02, .03], "yellow_exit": [True, False]}
    rows = []
    for values in __import__("itertools").product(*(grid[k] for k in grid)):
        p = {**base, **dict(zip(grid, values))}
        if p["fast"] >= p["slow"]: continue
        _, trades, data = run(frame, p, capital, allocation, slip, commission)
        result = stats(trades, data["equity"], capital)
        if result["trades"] >= 3: rows.append({"params": p, "stats": result})
    return sorted(rows, key=lambda row: row["stats"]["total_return_pct"], reverse=True)[:10]


INDEX_HTML = r'''<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Local Quant Lab</title>
<style>
:root{--bg:#09111f;--panel:#101c2e;--line:#22334d;--text:#e8eef7;--muted:#8da0ba;--accent:#6ee7b7;--blue:#77a8ff}*{box-sizing:border-box}body{margin:0;background:radial-gradient(circle at 85% -20%,#17345b 0,#09111f 42%);color:var(--text);font:14px/1.45 -apple-system,BlinkMacSystemFont,Segoe UI,sans-serif}main{max-width:1500px;margin:auto;padding:28px}.eyebrow{color:var(--accent);letter-spacing:.16em;text-transform:uppercase;font-size:11px;font-weight:700}h1{font-size:32px;line-height:1.1;margin:8px 0}h2{font-size:17px;margin:0 0 14px}.sub{color:var(--muted);max-width:760px}.layout{display:grid;grid-template-columns:330px 1fr;gap:18px;margin-top:24px}.panel{background:rgba(16,28,46,.92);border:1px solid var(--line);border-radius:14px;padding:17px;box-shadow:0 18px 50px #0002}.controls{position:sticky;top:16px;height:max-content}.field{margin:12px 0}.field label{display:block;color:var(--muted);font-size:12px;margin-bottom:5px}.field input,.field select{width:100%;background:#0b1627;color:var(--text);border:1px solid var(--line);border-radius:7px;padding:9px}.buttons{display:flex;gap:8px;flex-wrap:wrap}.btn{border:0;border-radius:7px;padding:9px 12px;background:#1e3352;color:var(--text);cursor:pointer}.btn.primary{background:#1f9d71;color:#04120d;font-weight:700}.stockbox{max-height:310px;overflow:auto;border:1px solid var(--line);border-radius:7px;padding:7px}.stock{display:inline-block;width:49%;padding:4px;color:var(--text)}.stock input{width:auto}.status{color:var(--muted);min-height:22px;margin-top:10px}.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(135px,1fr));gap:10px}.card{background:#0b1627;border:1px solid var(--line);border-radius:9px;padding:12px}.card span{display:block;color:var(--muted);font-size:11px;text-transform:uppercase}.card strong{font-size:20px}.chart{width:100%;border-radius:8px;margin-top:14px;background:#fff}.tablewrap{overflow:auto}table{width:100%;border-collapse:collapse;font-size:12px}th,td{text-align:right;padding:8px;border-bottom:1px solid var(--line);white-space:nowrap}th:first-child,td:first-child{text-align:left}th{color:var(--muted);font-weight:500}.positive{color:var(--accent)}.negative{color:#ff8e8e}.small{color:var(--muted);font-size:12px}@media(max-width:900px){.layout{grid-template-columns:1fr}.controls{position:static}}
</style></head><body><main><div class="eyebrow">Private local research tool</div><h1>Local Quant Lab</h1><p class="sub">Screen ASX 200 stocks with the Donchian / EMA profit-lock strategy, rerun the optimizer, and inspect trades without touching Moomoo or placing orders.</p><div class="layout"><aside class="panel controls"><h2>Test controls</h2><div class="field"><label>Start date</label><input id="start" type="date"></div><div class="field"><label>End date</label><input id="end" type="date"></div><div class="field"><label>Starting capital (AUD)</label><input id="capital" type="number" value="100000"></div><div class="field"><label>Allocation per entry</label><input id="allocation" type="number" step=".01" value=".95"></div><div class="field"><label>Slippage (bps)</label><input id="slip" type="number" step=".1" value="2"></div><div class="field"><label>Stocks</label><div class="buttons"><button class="btn" onclick="selectAll(true)">All 200</button><button class="btn" onclick="selectAll(false)">Clear</button></div><input id="search" placeholder="Filter tickers" oninput="filterStocks()"><div id="stocks" class="stockbox"></div></div><div class="field"><label><input id="optimize" type="checkbox" checked> Optimize settings</label></div><button class="btn primary" onclick="runTest()">Run backtest</button><div id="status" class="status"></div><p class="small">Data: Yahoo daily OHLC. For official execution validation, repeat the selected result in Moomoo Quant Replay.</p></aside><section><div id="summary" class="panel"><h2>Ready</h2><p class="sub">Choose stocks and run the test.</p></div><div id="chartPanel" class="panel" style="display:none"><h2 id="chartTitle">Selected stock</h2><img id="chart" class="chart"></div><div id="results" class="panel" style="display:none"><h2>Screen results</h2><div class="tablewrap"><table><thead><tr><th>Symbol</th><th>Return</th><th>Max DD</th><th>Trades</th><th>Win rate</th><th>Profit factor</th><th>Settings</th></tr></thead><tbody id="rows"></tbody></table></div></div><div id="tradesPanel" class="panel" style="display:none"><h2>Selected trades</h2><div class="tablewrap"><table><thead><tr><th>Entry</th><th>Price</th><th>Qty</th><th>Exit</th><th>Price</th><th>Net P/L</th><th>Return</th><th>Reason</th></tr></thead><tbody id="trades"></tbody></table></div></div></section></div></main><script>
let symbols=[];const $=id=>document.getElementById(id);function fmt(x){return Number(x).toFixed(2)}
function setDates(){let e=new Date(),s=new Date();s.setFullYear(e.getFullYear()-1);$('end').value=e.toISOString().slice(0,10);$('start').value=s.toISOString().slice(0,10)}
async function load(){setDates();symbols=await (await fetch('/api/universe')).json();renderStocks()}function renderStocks(){let q=$('search').value.toUpperCase();$('stocks').innerHTML=symbols.filter(s=>s.includes(q)).map(s=>`<label class="stock"><input type="checkbox" value="${s}" checked> ${s}</label>`).join('')}function filterStocks(){renderStocks()}function selectAll(v){document.querySelectorAll('#stocks input').forEach(x=>x.checked=v)}
async function runTest(){let selected=[...document.querySelectorAll('#stocks input:checked')].map(x=>x.value);if(!selected.length)return alert('Select at least one stock');$('status').textContent=`Running ${selected.length} stock${selected.length>1?'s':''}…`;$('summary').innerHTML='<h2>Running</h2><p class="sub">Downloading daily data and evaluating completed-bar signals.</p>';let body={symbols:selected,start:$('start').value,end:$('end').value,capital:+$('capital').value,allocation:+$('allocation').value,slippage_bps:+$('slip').value,optimize:$('optimize').checked};let res=await fetch('/api/backtest',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});let data=await res.json();if(!res.ok){$('status').textContent=data.error;return}$('status').textContent=`Complete: ${data.results.length} symbols`;let top=data.results[0];let ref=top.reference?`<p class="small">Saved Moomoo daily reference: ${top.reference.metrics.compounded_pct}% compounded, ${top.reference.metrics.trades} trade(s). This may use different strategy settings/time window.</p>`:'';$('summary').innerHTML=`<h2>Best result: ${top.symbol}</h2><div class="cards">${[['Return',top.stats.total_return_pct+'%'],['Max drawdown',top.stats.max_drawdown_pct+'%'],['Trades',top.stats.trades],['Win rate',top.stats.win_rate_pct+'%'],['Profit factor',top.stats.profit_factor]].map(x=>`<div class="card"><span>${x[0]}</span><strong class="${String(x[1]).startsWith('-')?'negative':'positive'}">${x[1]}</strong></div>`).join('')}</div>${ref}<p class="small">${data.note}</p>`;$('results').style.display='block';$('rows').innerHTML=data.results.map((r,i)=>`<tr onclick="selectResult(${i})"><td><b>${r.symbol}</b></td><td class="${r.stats.total_return_pct>=0?'positive':'negative'}">${r.stats.total_return_pct}%</td><td class="negative">${r.stats.max_drawdown_pct}%</td><td>${r.stats.trades}</td><td>${r.stats.win_rate_pct}%</td><td>${r.stats.profit_factor}</td><td class="small">${r.params?JSON.stringify(r.params):''}</td></tr>`).join('');window.results=data.results;selectResult(0)}
function selectResult(i){let r=window.results[i];$('chartPanel').style.display='block';$('chartTitle').textContent=r.symbol+' price and trades';$('chart').src='data:image/png;base64,'+r.chart;$('tradesPanel').style.display='block';$('trades').innerHTML=r.trades.map(t=>`<tr><td>${t.entry_time.slice(0,10)}</td><td>${fmt(t.entry_price)}</td><td>${t.qty}</td><td>${t.exit_time.slice(0,10)}</td><td>${fmt(t.exit_price)}</td><td class="${t.net_pnl>=0?'positive':'negative'}">${fmt(t.net_pnl)}</td><td>${fmt(t.return_pct)}%</td><td>${t.reason}</td></tr>`).join('')}
load();</script></body></html>'''


class Handler(BaseHTTPRequestHandler):
    def _send(self, body: bytes, content_type: str = "application/json", status: int = 200):
        self.send_response(status); self.send_header("Content-Type", content_type); self.send_header("Cache-Control", "no-store"); self.end_headers(); self.wfile.write(body)

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/": self._send(INDEX_HTML.encode(), "text/html; charset=utf-8")
        elif path == "/api/universe": self._send(json.dumps(universe()).encode())
        else: self._send(b'{"error":"not found"}', status=404)

    def do_POST(self):
        if urlparse(self.path).path != "/api/backtest": self._send(b'{"error":"not found"}', status=404); return
        try:
            payload = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))))
            symbols = [s for s in payload.get("symbols", []) if s in universe()]
            if not symbols: raise ValueError("No valid ASX 200 symbols selected")
            frames = candles(symbols, payload["start"], payload["end"])
            if not frames: raise ValueError("No daily data returned")
            capital, allocation, slip = float(payload.get("capital", 100000)), float(payload.get("allocation", .95)), float(payload.get("slippage_bps", 2))
            results = []
            for symbol, frame in frames.items():
                p = DEFAULTS.copy(); _, trades, data = run(frame, p, capital, allocation, slip, 0); best_p = p
                result = stats(trades, data["equity"], capital)
                if payload.get("optimize"):
                    options = optimize(frame, p, capital, allocation, slip, 0)
                    if options: best_p, result = options[0]["params"], options[0]["stats"]; _, trades, data = run(frame, best_p, capital, allocation, slip, 0)
                reference = None
                if symbol == "APP":
                    reference_path = Path(__file__).with_name("US_APP_donchian_ema_daily_2026_results.json")
                    if reference_path.exists():
                        saved = json.loads(reference_path.read_text())
                        reference = {"source": "saved Moomoo daily result", "metrics": saved.get("metrics", {}), "window": "2026"}
                results.append({"symbol": symbol, "stats": result, "params": best_p, "reference": reference, "trades": [asdict(t) for t in trades], "chart": plot_image(frame, trades)})
            results.sort(key=lambda x: x["stats"]["total_return_pct"], reverse=True)
            note = "Optimization is in-sample for the selected date range; validate finalists in Moomoo Quant Replay."
            self._send(json.dumps({"results": results, "note": note}, default=str).encode())
        except Exception as exc:
            self._send(json.dumps({"error": str(exc)}).encode(), status=400)


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Run the local Quant portal")
    ap.add_argument("--host", default="127.0.0.1"); ap.add_argument("--port", type=int, default=8765); args = ap.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Local Quant portal: http://{args.host}:{args.port}")
    print("Press Ctrl-C to stop. No orders are placed.")
    try: server.serve_forever()
    except KeyboardInterrupt: pass
    finally: server.server_close()


if __name__ == "__main__": main()
