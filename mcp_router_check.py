#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from agent import LLMAgent


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Проверка policy-роутера MCP")
    parser.add_argument("--request", required=True, help="Запрос для авто-роутера")
    parser.add_argument(
        "--expect-route",
        default="",
        help="Ожидаемое значение selected_route (необязательно)",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    agent = LLMAgent()
    payload = agent.route_mcp_request(args.request)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if payload.get("status") != "ok":
        raise SystemExit(1)
    if args.expect_route and payload.get("selected_route") != args.expect_route:
        raise SystemExit(1)
    print("\nMCP router: OK")


if __name__ == "__main__":
    main()
