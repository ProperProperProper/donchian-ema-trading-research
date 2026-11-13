#!/usr/bin/env python3
"""Small NVIDIA API client using a macOS Keychain-stored API key.

The client is intentionally independent of the trading strategy files. It can
be imported by research tools without adding an SDK dependency.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from typing import Any
from urllib import error, request


DEFAULT_BASE_URL = "https://integrate.api.nvidia.com/v1"
DEFAULT_MODEL = "deepseek-ai/deepseek-v4.1-flash"
DEFAULT_KEYCHAIN_SERVICE = "NVIDIA_API_KEY"


def key_from_keychain() -> str:
    """Read the secret without writing it to stdout or exposing it in logs."""
    service = os.getenv("NVIDIA_KEYCHAIN_SERVICE", DEFAULT_KEYCHAIN_SERVICE)
    account = os.getenv("NVIDIA_KEYCHAIN_ACCOUNT")
    command = ["security", "find-generic-password", "-s", service, "-w"]
    if account:
        command[4:4] = ["-a", account]
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise RuntimeError("macOS security command is unavailable") from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"No keychain secret found for service {service!r}. "
            "Set NVIDIA_KEYCHAIN_SERVICE (and optionally NVIDIA_KEYCHAIN_ACCOUNT)."
        ) from exc
    secret = result.stdout.strip()
    if not secret:
        raise RuntimeError(f"Keychain entry {service!r} is empty")
    return secret


def chat(
    messages: list[dict[str, str]],
    *,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.2,
    max_tokens: int = 1024,
) -> dict[str, Any]:
    """Create one NVIDIA-hosted chat completion."""
    api_key = os.getenv("NVIDIA_API_KEY") or key_from_keychain()
    base_url = os.getenv("NVIDIA_BASE_URL", DEFAULT_BASE_URL).rstrip("/")
    payload = json.dumps(
        {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }
    ).encode("utf-8")
    req = request.Request(
        f"{base_url}/chat/completions",
        data=payload,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Accept": "application/json",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        timeout = float(os.getenv("NVIDIA_REQUEST_TIMEOUT", "180"))
        with request.urlopen(req, timeout=timeout) as response:
            return json.load(response)
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"NVIDIA API returned HTTP {exc.code}: {detail}") from exc
    except error.URLError as exc:
        raise RuntimeError(f"Could not reach NVIDIA API: {exc.reason}") from exc


def main() -> int:
    parser = argparse.ArgumentParser(description="Call an NVIDIA API model")
    parser.add_argument("prompt", help="User prompt")
    parser.add_argument("--model", default=os.getenv("NVIDIA_MODEL", DEFAULT_MODEL))
    parser.add_argument("--max-tokens", type=int, default=1024)
    args = parser.parse_args()
    result = chat(
        [{"role": "user", "content": args.prompt}],
        model=args.model,
        max_tokens=args.max_tokens,
    )
    print(result["choices"][0]["message"]["content"])
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)
