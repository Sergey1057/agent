#!/usr/bin/env python3
"""
Проверка длинного MCP orchestration flow между несколькими серверами.

Проверяет:
1) корректный выбор инструментов с разных серверов;
2) корректность порядка вызовов;
3) успешность каждого шага флоу.
"""

from __future__ import annotations

import argparse
import json

from agent import LLMAgent


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Проверка multi-server MCP flow")
    parser.add_argument("--owner", required=True, help="Owner GitHub репозитория")
    parser.add_argument("--repo", required=True, help="Имя GitHub репозитория")
    parser.add_argument("--query", required=True, help="Запрос для pipeline шага")
    parser.add_argument(
        "--file-path",
        default="memory/pipeline_result.txt",
        help="Куда сохранить результат pipeline saveToFile",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    agent = LLMAgent()
    payload = agent.run_multi_server_mcp_flow(
        owner=args.owner,
        repo=args.repo,
        query=args.query,
        file_path=args.file_path,
    )

    print(json.dumps(payload, ensure_ascii=False, indent=2))

    if payload.get("status") != "ok":
        raise SystemExit(1)
    verification = payload.get("verification")
    if not isinstance(verification, dict):
        raise SystemExit(1)
    if not bool(verification.get("order_ok")):
        raise SystemExit(1)
    if not bool(verification.get("routes_ok")):
        raise SystemExit(1)

    steps = payload.get("steps")
    if not isinstance(steps, list) or not steps:
        raise SystemExit(1)
    if not all(bool(step.get("ok")) for step in steps if isinstance(step, dict)):
        raise SystemExit(1)

    print("\nMulti-server MCP flow: OK")


if __name__ == "__main__":
    main()
