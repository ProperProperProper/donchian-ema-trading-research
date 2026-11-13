#!/usr/bin/env python3
"""Codex-style command line client for the local AI Harbor gateway."""

from __future__ import annotations

import argparse
import json
import os
import sys
from urllib import request


BASE = os.getenv("AI_GATEWAY_URL", "http://127.0.0.1:8788").rstrip("/")


def get(path: str) -> dict:
    with request.urlopen(request.Request(BASE + path), timeout=10) as response:
        return json.load(response)


def post(path: str, payload: dict) -> dict:
    body = json.dumps(payload).encode("utf-8")
    req = request.Request(
        BASE + path,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with request.urlopen(req, timeout=180) as response:
        return json.load(response)


def main() -> int:
    parser = argparse.ArgumentParser(prog="ai_harbor", description="MLX/EXO/NVIDIA local AI gateway client")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("status", help="show backend and cluster status")
    sub.add_parser("models", help="list models visible to the gateway")
    sub.add_parser("cluster", help="show EXO/RDMA cluster configuration")
    chat = sub.add_parser("chat", help="send a prompt through difficulty-based routing")
    chat.add_argument("prompt", nargs="+", help="prompt text")
    chat.add_argument("--provider", choices=("auto", "local", "nvidia"), default="auto")
    chat.add_argument("--difficulty", type=int, choices=range(11), help="override difficulty from 0 to 10")
    args = parser.parse_args()
    try:
        if args.command == "status":
            print(json.dumps(get("/api/status"), indent=2))
        elif args.command == "models":
            print(json.dumps(get("/v1/models"), indent=2))
        elif args.command == "cluster":
            status = get("/api/status")
            print(json.dumps(status.get("exo", {}), indent=2))
            print("RDMA: configure MLX JACCL over Thunderbolt on the cluster hosts.")
        else:
            extra = {"provider": args.provider}
            if args.difficulty is not None:
                extra["difficulty"] = args.difficulty
            result = post("/v1/chat/completions", {"model": "auto", "messages": [{"role": "user", "content": " ".join(args.prompt)}], "extra_body": extra})
            print(result.get("choices", [{}])[0].get("message", {}).get("content", json.dumps(result)))
            print(f"\n[route: {result.get('_gateway', {}).get('provider', 'unknown')}]", file=sys.stderr)
    except Exception as exc:
        print(f"ai_harbor: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
