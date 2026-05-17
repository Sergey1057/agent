#!/usr/bin/env python3
"""
Smoke-тест приватного LLM-сервиса: health, чат, rate limit, параллельные запросы.

  python scripts/smoke_llm_service.py --base-url http://127.0.0.1:8080
  LLM_SERVICE_API_KEY=secret python scripts/smoke_llm_service.py
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import sys
import urllib.error
import urllib.request


def _request(
    method: str,
    url: str,
    *,
    api_key: str = "",
    body: dict | None = None,
    timeout: float = 120.0,
) -> tuple[int, dict | str]:
    data = None
    headers = {"Accept": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    if body is not None:
        headers["Content-Type"] = "application/json"
        data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode()
            code = resp.status
    except urllib.error.HTTPError as e:
        code = e.code
        raw = e.read().decode()
    try:
        parsed: dict | str = json.loads(raw)
    except json.JSONDecodeError:
        parsed = raw
    return code, parsed


def main() -> int:
    p = argparse.ArgumentParser(description="Smoke-тест LLM HTTP-сервиса")
    p.add_argument("--base-url", default="http://127.0.0.1:8080")
    p.add_argument("--api-key", default=os.environ.get("LLM_SERVICE_API_KEY", ""))
    p.add_argument("--parallel", type=int, default=4)
    p.add_argument("--spam", type=int, default=35, help="Запросов для проверки rate limit")
    args = p.parse_args()
    base = args.base_url.rstrip("/")
    key = args.api_key

    print("1) GET /health")
    code, data = _request("GET", f"{base}/health", api_key=key)
    print(f"   HTTP {code}: {data}")
    if code != 200:
        return 1

    print("2) POST /api/chat")
    code, data = _request(
        "POST",
        f"{base}/api/chat",
        api_key=key,
        body={"message": "Ответь одним словом: ок"},
    )
    print(f"   HTTP {code}")
    if code != 200 or not isinstance(data, dict):
        print(f"   FAIL: {data}")
        return 1
    print(f"   session_id={data.get('session_id')} reply={str(data.get('reply', ''))[:80]}")

    sid = data.get("session_id")
    if sid:
        print("3) POST /api/chat (та же сессия)")
        code2, data2 = _request(
            "POST",
            f"{base}/api/chat",
            api_key=key,
            body={"message": "Повтори предыдущий ответ кратко", "session_id": sid},
        )
        print(f"   HTTP {code2}: reply={str(data2.get('reply', ''))[:80] if isinstance(data2, dict) else data2}")

    print(f"4) Параллельно {args.parallel} запросов /api/chat")
    def one_chat(i: int) -> tuple[int, str]:
        c, d = _request(
            "POST",
            f"{base}/api/chat",
            api_key=key,
            body={"message": f"Скажи число {i}"},
            timeout=180.0,
        )
        err = ""
        if isinstance(d, dict) and "error" in d:
            err = str(d["error"])
        return c, err

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.parallel) as ex:
        results = list(ex.map(one_chat, range(args.parallel)))
    ok = sum(1 for c, _ in results if c == 200)
    rate_limited = sum(1 for c, _ in results if c == 429)
    print(f"   OK={ok} 429={rate_limited} other={args.parallel - ok - rate_limited}")

    print(f"5) Rate limit: {args.spam} быстрых GET /api/ping")
    codes: list[int] = []
    for _ in range(args.spam):
        c, _ = _request("GET", f"{base}/api/ping", api_key=key, timeout=5.0)
        codes.append(c)
    n429 = sum(1 for c in codes if c == 429)
    print(f"   429 count={n429} (ожидается >0 при LLM_SERVICE_RATE_LIMIT_RPM < {args.spam})")

    print("6) POST /v1/chat/completions (слишком длинный контекст)")
    big_messages = [
        {"role": "user", "content": "x" * 20_000}
        for _ in range(60)
    ]
    code, data = _request(
        "POST",
        f"{base}/v1/chat/completions",
        api_key=key,
        body={"messages": big_messages},
    )
    print(f"   HTTP {code} (ожидается 400): {data}")

    print("\nГотово.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
