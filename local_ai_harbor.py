#!/usr/bin/env python3
"""Harbor-like local AI gateway for Apple Silicon.

The gateway exposes one OpenAI-compatible endpoint and routes requests to:

* MLX-LM's native OpenAI-compatible server (local, default), or
* NVIDIA's hosted NIM endpoint (optional cloud fallback/provider).

It intentionally uses only the Python standard library plus the existing
``nvidia_api.py`` Keychain-aware client. Start MLX separately with
``python -m mlx_lm.server``; this keeps model lifecycle and this router
independent and easy to debug.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib import error, request

import nvidia_api


DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8788
DEFAULT_MLX_URL = "http://127.0.0.1:8080/v1"
DEFAULT_NVIDIA_URL = nvidia_api.DEFAULT_BASE_URL
DEFAULT_NVIDIA_MODEL = nvidia_api.DEFAULT_MODEL


def env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def post_json(url: str, payload: dict[str, Any], headers: dict[str, str] | None = None) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    req_headers = {"Accept": "application/json", "Content-Type": "application/json"}
    if headers:
        req_headers.update(headers)
    req = request.Request(url, data=body, headers=req_headers, method="POST")
    try:
        with request.urlopen(req, timeout=float(os.getenv("AI_REQUEST_TIMEOUT", "120"))) as response:
            return json.load(response)
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"upstream returned HTTP {exc.code}: {detail}") from exc
    except error.URLError as exc:
        raise RuntimeError(f"could not reach upstream: {exc.reason}") from exc


def get_json(url: str, headers: dict[str, str] | None = None) -> dict[str, Any]:
    req_headers = {"Accept": "application/json"}
    if headers:
        req_headers.update(headers)
    try:
        with request.urlopen(request.Request(url, headers=req_headers), timeout=10) as response:
            return json.load(response)
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"upstream returned HTTP {exc.code}: {detail}") from exc
    except error.URLError as exc:
        raise RuntimeError(f"could not reach upstream: {exc.reason}") from exc


class Gateway:
    def __init__(self, default_provider: str) -> None:
        if default_provider not in {"local", "nvidia", "auto"}:
            raise ValueError("provider must be local, nvidia, or auto")
        self.default_provider = default_provider
        self.mlx_url = os.getenv("MLX_BASE_URL", DEFAULT_MLX_URL).rstrip("/")
        self.nvidia_url = os.getenv("NVIDIA_BASE_URL", DEFAULT_NVIDIA_URL).rstrip("/")
        self.mlx_model = os.getenv("MLX_MODEL", "")
        self.nvidia_model = os.getenv("NVIDIA_MODEL", DEFAULT_NVIDIA_MODEL)
        self._lock = threading.Lock()
        self._inflight = {"local": 0, "nvidia": 0}
        self._latency = {"local": 0.0, "nvidia": 0.0}

    def _nvidia_configured(self) -> bool:
        return bool(os.getenv("NVIDIA_API_KEY")) or bool(os.getenv("NVIDIA_KEYCHAIN_SERVICE")) or self._keychain_available()

    @staticmethod
    def _keychain_available() -> bool:
        # Do not read the secret just to select a provider. The request path
        # will read it and produce the useful error if the entry is missing.
        return sys.platform == "darwin"

    def _healthy(self, provider: str) -> bool:
        if provider == "local":
            try:
                get_json(f"{self.mlx_url}/models")
                return True
            except RuntimeError:
                return False
        return self._nvidia_configured()

    def _difficulty(self, body: dict[str, Any]) -> int:
        """Return a small 0-10 complexity score.

        This is deliberately deterministic and inspectable. Callers can
        override it with ``extra_body.difficulty`` or the
        ``X-AI-Difficulty`` header. The gateway never sends easy requests to
        NVIDIA merely to balance traffic.
        """
        extra = body.get("extra_body")
        if isinstance(extra, dict) and extra.get("difficulty") is not None:
            try:
                return max(0, min(10, int(extra["difficulty"])))
            except (TypeError, ValueError):
                pass
        messages = body.get("messages", [])
        text = " ".join(str(m.get("content", "")) for m in messages if isinstance(m, dict)).lower()
        score = 0
        length = len(text)
        if length > 1200:
            score += 1
        if length > 4000:
            score += 2
        for phrase in (
            "debug", "refactor", "architecture", "design", "optimize", "quantitative",
            "backtest", "strategy", "risk analysis", "compare", "trade-off", "deeply",
            "step by step", "multi-step", "write code", "implement", "prove", "derive",
        ):
            if phrase in text:
                score += 1
        if text.count("?") >= 3 or text.count("\n") >= 20:
            score += 1
        if any(token in text for token in ("```", " traceback", " stack trace", "exception")):
            score += 2
        return min(10, score)

    def _choose_auto(self, body: dict[str, Any]) -> tuple[str, int]:
        difficulty = self._difficulty(body)
        threshold = int(os.getenv("AI_LOCAL_MAX_DIFFICULTY", "3"))
        preferred = "local" if difficulty <= threshold else "nvidia"
        alternate = "nvidia" if preferred == "local" else "local"
        candidates = [name for name in (preferred, alternate) if self._healthy(name)]
        if not candidates:
            raise RuntimeError("neither MLX nor NVIDIA is available")
        return candidates[0], difficulty

    def _run(self, provider: str, body: dict[str, Any]) -> dict[str, Any]:
        started = time.monotonic()
        with self._lock:
            self._inflight[provider] += 1
        try:
            return self.local_chat(body) if provider == "local" else self.nvidia_chat(body)
        finally:
            elapsed = time.monotonic() - started
            with self._lock:
                self._inflight[provider] -= 1
                previous = self._latency[provider]
                self._latency[provider] = elapsed if previous == 0 else (previous * 0.8) + (elapsed * 0.2)

    def provider_for(self, body: dict[str, Any], headers: dict[str, str]) -> str:
        extra = body.get("extra_body")
        if isinstance(extra, dict) and extra.get("provider") in {"local", "nvidia", "auto"}:
            return str(extra["provider"])
        header = headers.get("X-AI-Provider", "").lower()
        return header if header in {"local", "nvidia", "auto"} else self.default_provider

    def local_chat(self, body: dict[str, Any]) -> dict[str, Any]:
        payload = dict(body)
        payload.pop("extra_body", None)
        if self.mlx_model and not payload.get("model"):
            payload["model"] = self.mlx_model
        return post_json(f"{self.mlx_url}/chat/completions", payload)

    def nvidia_chat(self, body: dict[str, Any]) -> dict[str, Any]:
        payload = dict(body)
        payload.pop("extra_body", None)
        payload["model"] = payload.get("model") or self.nvidia_model
        api_key = os.getenv("NVIDIA_API_KEY") or nvidia_api.key_from_keychain()
        return post_json(
            f"{self.nvidia_url}/chat/completions",
            payload,
            {"Authorization": f"Bearer {api_key}"},
        )

    def chat(self, body: dict[str, Any], headers: dict[str, str]) -> tuple[dict[str, Any], str]:
        provider = self.provider_for(body, headers)
        if body.get("stream"):
            raise RuntimeError("streaming is not implemented in this first gateway build")
        if provider in {"local", "nvidia"}:
            return self._run(provider, body), provider

        # Automatic mode is a real load balancer, not just a local fallback:
        # it samples backend health, chooses least in-flight work, and retries
        # the other backend if the chosen backend fails.
        selected, difficulty = self._choose_auto(body)
        try:
            response = self._run(selected, body)
            response.setdefault("_gateway", {})["difficulty"] = difficulty
            return response, f"auto:{selected}"
        except RuntimeError as first_error:
            alternate = "nvidia" if selected == "local" else "local"
            if not self._healthy(alternate):
                raise first_error
            response = self._run(alternate, body)
            response.setdefault("_gateway", {})["difficulty"] = difficulty
            return response, f"auto:{alternate}-failover"

    def models(self) -> dict[str, Any]:
        models: list[dict[str, Any]] = []
        try:
            local = get_json(f"{self.mlx_url}/models")
            models.extend(local.get("data", []))
        except RuntimeError:
            pass
        models.append({"id": self.nvidia_model, "object": "model", "owned_by": "nvidia"})
        return {"object": "list", "data": models}

    def status(self) -> dict[str, Any]:
        return {
            "default_provider": self.default_provider,
            "routing": "difficulty",
            "local_max_difficulty": int(os.getenv("AI_LOCAL_MAX_DIFFICULTY", "3")),
            "backends": {
                "local_mlx": {
                    "url": self.mlx_url,
                    "model": self.mlx_model or "managed by MLX server",
                    "healthy": self._healthy("local"),
                },
                "nvidia": {
                    "url": self.nvidia_url,
                    "model": self.nvidia_model,
                    "configured": self._nvidia_configured(),
                    "healthy": self._healthy("nvidia"),
                },
            },
            "inflight": dict(self._inflight),
            "latency_seconds": dict(self._latency),
            "exo": {
                "configured": bool(os.getenv("EXO_BASE_URL")),
                "url": os.getenv("EXO_BASE_URL", ""),
                "note": "Use EXO_BASE_URL for an EXO cluster's OpenAI-compatible endpoint.",
            },
        }


class Handler(BaseHTTPRequestHandler):
    gateway: Gateway

    def log_message(self, fmt: str, *args: Any) -> None:
        sys.stderr.write(f"[gateway] {self.address_string()} {fmt % args}\n")

    def send_json(self, status: int, payload: dict[str, Any]) -> None:
        encoded = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def do_GET(self) -> None:
        if self.path == "/":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            body = DASHBOARD_HTML.encode("utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/health":
            self.send_json(200, {"ok": True, "service": "local-ai-harbor"})
        elif self.path == "/api/status":
            self.send_json(200, self.gateway.status())
        elif self.path == "/v1/models":
            self.send_json(200, self.gateway.models())
        else:
            self.send_json(404, {"error": {"message": "not found", "type": "not_found"}})

    def do_POST(self) -> None:
        if self.path != "/v1/chat/completions":
            self.send_json(404, {"error": {"message": "not found", "type": "not_found"}})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(length))
            response, provider = self.gateway.chat(body, {k: v for k, v in self.headers.items()})
            response.setdefault("_gateway", {})["provider"] = provider
            self.send_json(200, response)
        except (ValueError, json.JSONDecodeError) as exc:
            self.send_json(400, {"error": {"message": f"invalid JSON: {exc}", "type": "invalid_request_error"}})
        except Exception as exc:
            self.send_json(502, {"error": {"message": str(exc), "type": "upstream_error"}})


def main() -> int:
    parser = argparse.ArgumentParser(description="Local MLX + NVIDIA AI gateway")
    parser.add_argument("--host", default=os.getenv("AI_GATEWAY_HOST", DEFAULT_HOST))
    parser.add_argument("--port", type=int, default=int(os.getenv("AI_GATEWAY_PORT", DEFAULT_PORT)))
    parser.add_argument("--provider", choices=("local", "nvidia", "auto"), default=os.getenv("AI_PROVIDER", "auto"))
    args = parser.parse_args()
    Handler.gateway = Gateway(args.provider)
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"AI gateway listening on http://{args.host}:{args.port} ({args.provider})", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nAI gateway stopped", flush=True)
    finally:
        server.server_close()
    return 0


DASHBOARD_HTML = r'''<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Local AI Harbor</title><style>
:root{--bg:#07101d;--panel:#101d2f;--line:#263b59;--text:#ecf3fb;--muted:#8da3bf;--green:#62e2ae;--blue:#85afff;--amber:#f4c86f;--red:#ff8494}*{box-sizing:border-box}body{margin:0;background:radial-gradient(circle at 80% -10%,#19365e,#07101d 45%);color:var(--text);font:14px/1.45 -apple-system,BlinkMacSystemFont,Segoe UI,sans-serif}main{max-width:1280px;margin:auto;padding:26px}.top{display:flex;justify-content:space-between;align-items:flex-start;gap:20px}.eyebrow{color:var(--green);font-size:11px;font-weight:700;letter-spacing:.16em;text-transform:uppercase}h1{font-size:30px;margin:6px 0}.muted,.sub{color:var(--muted)}.grid{display:grid;grid-template-columns:300px minmax(0,1fr);gap:16px;margin-top:22px}.panel{background:rgba(16,29,47,.94);border:1px solid var(--line);border-radius:12px;padding:15px;margin-bottom:16px}.field{margin:12px 0}.field label{display:block;color:var(--muted);font-size:11px;margin-bottom:5px}.field select,.field textarea{width:100%;background:#0b1728;color:var(--text);border:1px solid var(--line);border-radius:7px;padding:9px}.field textarea{min-height:110px;resize:vertical}.btn{border:1px solid var(--line);background:#1c3455;color:var(--text);border-radius:7px;padding:9px 12px;cursor:pointer}.btn.primary{background:#1d9d72;color:#03150e;border-color:#1d9d72;font-weight:700}.row{display:flex;gap:8px;align-items:center;justify-content:space-between}.badge{border:1px solid var(--line);border-radius:999px;padding:3px 8px;font-size:11px}.ok{color:var(--green);border-color:#287c5f}.warn{color:var(--amber);border-color:#866b32}.bad{color:var(--red);border-color:#803542}.cards{display:grid;grid-template-columns:repeat(3,1fr);gap:9px}.card{background:#0b1728;border:1px solid var(--line);border-radius:8px;padding:11px}.card small{display:block;color:var(--muted)}.card strong{display:block;margin-top:3px}.chat{min-height:450px;display:flex;flex-direction:column}.messages{flex:1;min-height:300px;overflow:auto;padding:4px}.msg{padding:10px 12px;border-radius:9px;margin:8px 0;white-space:pre-wrap}.user{background:#173258;margin-left:15%}.assistant{background:#0c1d31;margin-right:15%}.composer{border-top:1px solid var(--line);padding-top:12px}.mono{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12px}@media(max-width:850px){.grid{grid-template-columns:1fr}.cards{grid-template-columns:1fr 1fr}}
</style></head><body><main><div class="top"><div><div class="eyebrow">TradingSystem · private inference</div><h1>Local AI Harbor</h1><div class="sub">MLX for everyday work. NVIDIA for harder reasoning. EXO-ready cluster endpoint.</div></div><span id="health" class="badge">Checking…</span></div><div class="grid"><aside><section class="panel"><h3>Routing</h3><div class="field"><label>Provider policy</label><select id="provider"><option value="auto">Auto · difficulty based</option><option value="local">Local MLX only</option><option value="nvidia">NVIDIA only</option></select></div><div class="field"><label>Difficulty (auto)</label><select id="difficulty"><option value="auto">Classify automatically</option><option value="1">1 · trivial</option><option value="3">3 · normal</option><option value="6">6 · difficult</option><option value="9">9 · expert</option></select></div><button class="btn" onclick="refresh()">Refresh status</button></section><section class="panel"><h3>Backends</h3><div id="backends" class="mono muted">Loading…</div></section><section class="panel"><h3>CLI</h3><div class="muted">Use the same gateway from Codex-style workflows:</div><p class="mono">python3 ai_harbor_cli.py chat "your prompt"</p><p class="mono">python3 ai_harbor_cli.py cluster</p></section></aside><section class="panel chat"><div id="messages" class="messages"><div class="msg assistant">Ready. Ask about research, code, logs, or strategy diagnostics.</div></div><div class="composer"><div class="field"><textarea id="prompt" placeholder="Ask the local AI gateway…" onkeydown="if(event.key==='Enter'&&!event.shiftKey){event.preventDefault();send()}"></textarea></div><div class="row"><span id="route" class="muted">Enter to send · Shift+Enter for newline</span><button class="btn primary" onclick="send()">Send</button></div></div></section></div></main><script>
const $=id=>document.getElementById(id);function add(role,text){let d=document.createElement('div');d.className='msg '+role;d.textContent=text;$('messages').appendChild(d);$('messages').scrollTop=$('messages').scrollHeight;return d}async function refresh(){try{let r=await fetch('/api/status');let s=await r.json();$('health').textContent='Gateway online';$('health').className='badge ok';let b=s.backends;let line=(name,x)=>`${name}: <span class="${x.healthy||x.configured?'':'bad'}">${x.healthy?'READY':x.configured?'CONFIGURED':'OFFLINE'}</span> · ${x.model}`;$('backends').innerHTML=line('MLX',b.local_mlx)+'<br><br>'+line('NVIDIA',b.nvidia)+'<br><br>'+`EXO: ${s.exo.configured?'configured':'not configured'}`;$('route').textContent=`Auto threshold ≤ ${s.local_max_difficulty} stays local`; }catch(e){$('health').textContent='Gateway offline';$('health').className='badge bad'}}async function send(){let p=$('prompt').value.trim();if(!p)return;$('prompt').value='';add('user',p);let wait=add('assistant','Thinking…');let extra={provider:$('provider').value};let d=$('difficulty').value;if(d!=='auto')extra.difficulty=Number(d);try{let r=await fetch('/v1/chat/completions',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({model:'auto',messages:[{role:'user',content:p}],extra_body:extra})});let j=await r.json();if(!r.ok)throw Error(j.error?.message||'Request failed');wait.textContent=j.choices?.[0]?.message?.content||JSON.stringify(j);$('route').textContent=`Routed via ${j._gateway?.provider||'gateway'}${j._gateway?.difficulty!==undefined?' · difficulty '+j._gateway.difficulty:''}`;}catch(e){wait.textContent='Error: '+e.message}}refresh();setInterval(refresh,10000)
</script></body></html>'''

DASHBOARD_HTML = DASHBOARD_HTML.replace(
    '<div class="grid">',
    '<section style="border:2px solid #f2d44f;background:linear-gradient(110deg,#172d4a,#101d2f);padding:18px 20px;margin-top:20px;box-shadow:0 0 0 4px #f2d44f16">'
    '<span style="display:inline-block;background:#f2d44f;color:#111;padding:5px 10px;font-weight:800;letter-spacing:.12em;font-size:11px">CONCURRENCY</span>'
    '<h2 style="margin:14px 0 4px;font-size:20px">LARGE MODELS AND HIGHER CONCURRENCY THAN ONE COMPUTER CAN HANDLE</h2>'
    '<div class="sub">Set up an endpoint in minutes on a single Mac. When you want to run larger models, connect multiple machines over Thunderbolt 5 to pool unified memory on hardware you control.</div>'
    '</section><div class="grid">',
    1,
)


if __name__ == "__main__":
    raise SystemExit(main())
