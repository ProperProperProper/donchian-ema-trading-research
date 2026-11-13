#!/usr/bin/env python3
"""Local ASX paper trader with Moomoo AU fee simulation.

This is deliberately a paper ledger. It never connects to an order API and
never submits, cancels, or modifies a broker order. Quotes are obtained from
Yahoo Finance and should be treated as delayed/research data.
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import asdict, dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import yfinance as yf


STATE_PATH = Path(__file__).with_name("paper_trader_state.json")
FEE_RULES = {
    "commission_rate": 0.0001,
    "commission_min": 1.0,
    "platform_rate": 0.0002,
    "platform_min": 2.0,
    "currency": "AUD",
    "source": "Moomoo AU ASX fee schedule; GST inclusive",
}


@dataclass
class Fill:
    timestamp: str
    symbol: str
    side: str
    qty: int
    price: float
    notional: float
    commission: float
    platform_fee: float
    total_fees: float
    cash_after: float


def fees(notional: float) -> dict[str, float]:
    commission = max(FEE_RULES["commission_min"], notional * FEE_RULES["commission_rate"])
    platform = max(FEE_RULES["platform_min"], notional * FEE_RULES["platform_rate"])
    return {"commission": round(commission, 2), "platform_fee": round(platform, 2), "total_fees": round(commission + platform, 2)}


class Ledger:
    def __init__(self, path: Path, cash: float = 5000.0):
        self.path = path
        self.state = {"cash": cash, "starting_cash": cash, "positions": {}, "fills": [], "created": time.strftime("%Y-%m-%dT%H:%M:%S")}
        if path.exists():
            self.state = json.loads(path.read_text())

    def save(self):
        self.path.write_text(json.dumps(self.state, indent=2) + "\n")

    def execute(self, symbol: str, side: str, qty: int, price: float) -> Fill:
        symbol, side, qty = symbol.upper().strip(), side.upper(), int(qty)
        if side not in {"BUY", "SELL"} or qty <= 0 or price <= 0:
            raise ValueError("Use a positive quantity and BUY or SELL")
        notional = qty * price
        cost = fees(notional)
        positions = self.state["positions"]
        held = int(positions.get(symbol, {}).get("qty", 0))
        if side == "BUY":
            required = notional + cost["total_fees"]
            if required > self.state["cash"]:
                raise ValueError(f"Insufficient paper cash: need A${required:,.2f}")
            old_cost = float(positions.get(symbol, {}).get("cost", 0))
            new_qty = held + qty
            positions[symbol] = {"qty": new_qty, "cost": old_cost + notional + cost["total_fees"], "average_price": (old_cost + notional + cost["total_fees"]) / new_qty}
            self.state["cash"] -= required
        else:
            if qty > held:
                raise ValueError(f"Cannot sell {qty}; only {held} paper shares held")
            self.state["cash"] += notional - cost["total_fees"]
            if qty == held:
                positions.pop(symbol)
            else:
                positions[symbol]["qty"] = held - qty
                positions[symbol]["cost"] *= (held - qty) / held
                positions[symbol]["average_price"] = positions[symbol]["cost"] / (held - qty)
        fill = Fill(time.strftime("%Y-%m-%dT%H:%M:%S"), symbol, side, qty, round(price, 4), round(notional, 2), cost["commission"], cost["platform_fee"], cost["total_fees"], round(self.state["cash"], 2))
        self.state["fills"].append(asdict(fill)); self.save(); return fill

    def reset(self, cash: float):
        self.state = {"cash": cash, "starting_cash": cash, "positions": {}, "fills": [], "created": time.strftime("%Y-%m-%dT%H:%M:%S")}; self.save()


def quote(symbol: str) -> dict:
    ticker = f"{symbol.upper().strip()}.AX"
    frame = yf.download(ticker, period="5d", interval="1m", auto_adjust=False, progress=False, threads=False)
    if frame.empty:
        raise ValueError(f"No quote returned for ASX.{symbol}")
    close = frame["Close"]
    price = float(close.iloc[-1].iloc[0] if hasattr(close.iloc[-1], "iloc") else close.iloc[-1])
    stamp = frame.index[-1].isoformat()
    return {"symbol": symbol.upper(), "price": round(price, 4), "timestamp": stamp, "source": "Yahoo Finance (delayed/research data)"}


HTML = r'''<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>ASX Paper Trader</title><style>body{font:14px -apple-system,BlinkMacSystemFont,Segoe UI,sans-serif;background:#09111f;color:#eaf0f8;margin:0}main{max-width:1100px;margin:auto;padding:28px}.grid{display:grid;grid-template-columns:340px 1fr;gap:16px}.panel{background:#101c2e;border:1px solid #263b59;border-radius:12px;padding:16px;margin-bottom:16px}h1{margin:0 0 5px}.muted{color:#91a4bd}.field{margin:12px 0}.field label{display:block;color:#91a4bd;font-size:12px;margin-bottom:4px}input,select{width:100%;padding:9px;border:1px solid #263b59;border-radius:6px;background:#0b1627;color:#fff}.button{padding:9px 13px;border:0;border-radius:6px;background:#22a06b;color:#04140d;font-weight:700;cursor:pointer}.danger{background:#9c3d50;color:#fff}.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px}.card{padding:12px;background:#0b1627;border-radius:8px}.card span{display:block;color:#91a4bd;font-size:11px;text-transform:uppercase}.card strong{font-size:20px}table{width:100%;border-collapse:collapse}td,th{text-align:right;padding:8px;border-bottom:1px solid #263b59}td:first-child,th:first-child{text-align:left}.positive{color:#70e1b0}.negative{color:#ff9292}@media(max-width:800px){.grid{grid-template-columns:1fr}}</style><main><h1>ASX Paper Trader</h1><p class="muted">Local-only simulated trading. No broker connection and no live orders.</p><div class="grid"><section><div class="panel"><h2>Manual paper order</h2><div class="field"><label>ASX code</label><input id="symbol" value="BHP"></div><button class="button" onclick="getQuote()">Refresh quote</button><p id="quote" class="muted">No quote loaded.</p><div class="field"><label>Side</label><select id="side"><option>BUY</option><option>SELL</option></select></div><div class="field"><label>Quantity</label><input id="qty" type="number" min="1" value="10"></div><button class="button" onclick="order()">Place paper order</button><p id="message" class="muted"></p></div><div class="panel"><h2>Fee model</h2><p>Commission: 0.01%, minimum A$1/order<br>Platform fee: 0.02%, minimum A$2/order<br>Total minimum: A$3/order, GST inclusive.</p><button class="button danger" onclick="resetLedger()">Reset paper account</button></div></section><section><div id="summary" class="panel"></div><div class="panel"><h2>Positions</h2><table><thead><tr><th>Symbol</th><th>Qty</th><th>Avg cost</th><th>Last</th><th>Market value</th><th>Unrealised P/L</th></tr></thead><tbody id="positions"></tbody></table></div><div class="panel"><h2>Fills</h2><table><thead><tr><th>Time</th><th>Symbol</th><th>Side</th><th>Qty</th><th>Price</th><th>Fees</th><th>Cash after</th></tr></thead><tbody id="fills"></tbody></table></div></section></div></main><script>
async function api(url,opt){let r=await fetch(url,opt),j=await r.json();if(!r.ok)throw Error(j.error);return j}function money(x){return 'A$'+Number(x).toFixed(2)}async function refresh(){try{let s=await api('/api/state');let value=0,pnl=0;let rows=[];for(let [sym,p] of Object.entries(s.positions)){let q=await api('/api/quote?symbol='+sym);let mv=q.price*p.qty;value+=mv;pnl+=mv-p.cost;rows.push(`<tr><td>${sym}</td><td>${p.qty}</td><td>${money(p.average_price)}</td><td>${money(q.price)}</td><td>${money(mv)}</td><td class="${mv-p.cost>=0?'positive':'negative'}">${money(mv-p.cost)}</td></tr>`)}$('summary').innerHTML=`<div class="cards"><div class="card"><span>Cash</span><strong>${money(s.cash)}</strong></div><div class="card"><span>Positions</span><strong>${money(value)}</strong></div><div class="card"><span>Total equity</span><strong>${money(s.cash+value)}</strong></div><div class="card"><span>Net P/L</span><strong class="${s.cash+value-s.starting_cash>=0?'positive':'negative'}">${money(s.cash+value-s.starting_cash)}</strong></div></div>`;$('positions').innerHTML=rows.join('')||'<tr><td colspan="6">No paper positions</td></tr>';$('fills').innerHTML=s.fills.slice().reverse().map(f=>`<tr><td>${f.timestamp}</td><td>${f.symbol}</td><td>${f.side}</td><td>${f.qty}</td><td>${money(f.price)}</td><td>${money(f.total_fees)}</td><td>${money(f.cash_after)}</td></tr>`).join('')||'<tr><td colspan="7">No fills</td></tr>'}catch(e){$('message').textContent=e.message}}async function getQuote(){try{let q=await api('/api/quote?symbol='+$('symbol').value);$('quote').textContent=`${q.symbol}: ${money(q.price)} · ${q.timestamp} · ${q.source}`}catch(e){$('message').textContent=e.message}}async function order(){try{let q=await api('/api/quote?symbol='+$('symbol').value);let f=await api('/api/order',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({symbol:$('symbol').value,side:$('side').value,qty:+$('qty').value,price:q.price})});$('message').textContent=`Filled ${f.side} ${f.qty} ${f.symbol} at ${money(f.price)}; fees ${money(f.total_fees)}`;refresh()}catch(e){$('message').textContent=e.message}}async function resetLedger(){if(confirm('Reset the local paper account?')){await api('/api/reset',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({cash:5000})});refresh()}}refresh();setInterval(refresh,30000)</script>'''


ledger = Ledger(STATE_PATH)


class Handler(BaseHTTPRequestHandler):
    def send_json(self, value, status=200):
        data = json.dumps(value).encode(); self.send_response(status); self.send_header("Content-Type", "application/json"); self.send_header("Content-Length", str(len(data))); self.end_headers(); self.wfile.write(data)

    def do_GET(self):
        parsed = urlparse(self.path)
        try:
            if parsed.path == "/": self.send_response(200); self.send_header("Content-Type", "text/html; charset=utf-8"); self.end_headers(); self.wfile.write(HTML.encode())
            elif parsed.path == "/api/state": self.send_json(ledger.state)
            elif parsed.path == "/api/fees": self.send_json(FEE_RULES)
            elif parsed.path == "/api/quote": self.send_json(quote(parse_qs(parsed.query).get("symbol", [""])[0]))
            else: self.send_json({"error": "not found"}, 404)
        except Exception as exc: self.send_json({"error": str(exc)}, 400)

    def do_POST(self):
        try:
            payload = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))))
            if self.path == "/api/order": self.send_json(asdict(ledger.execute(payload["symbol"], payload["side"], payload["qty"], float(payload["price"]))))
            elif self.path == "/api/reset": ledger.reset(float(payload.get("cash", 5000))); self.send_json({"ok": True})
            else: self.send_json({"error": "not found"}, 404)
        except Exception as exc: self.send_json({"error": str(exc)}, 400)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Local ASX paper trader")
    parser.add_argument("--host", default="127.0.0.1"); parser.add_argument("--port", type=int, default=8770); args = parser.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), Handler); print(f"ASX paper trader: http://{args.host}:{args.port}")
    print(f"State file: {STATE_PATH} (local only; no broker orders)")
    try: server.serve_forever()
    except KeyboardInterrupt: pass
    finally: server.server_close()


if __name__ == "__main__": main()
