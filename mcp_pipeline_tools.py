#!/usr/bin/env python3
"""
Проверка MCP-пайплайна:
1) list_tools и проверка наличия search/summorize/saveToFile/run_tools_pipeline
2) вызов run_tools_pipeline
3) валидация автоцепочки и передачи данных между шагами
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Проверка MCP pipeline инструментов")
    parser.add_argument("--query", required=True, help="Поисковый запрос для pipeline")
    parser.add_argument(
        "--file-path",
        default="memory/pipeline_result.txt",
        help="Куда сохранить результат saveToFile",
    )
    return parser.parse_args()


async def _run() -> int:
    args = _parse_args()
    server_script = str((Path(__file__).resolve().parent / "scheduler_mcp_server.py").resolve())
    server_params = StdioServerParameters(
        command="python3",
        args=[server_script],
        env=dict(os.environ),
    )

    async with stdio_client(server_params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            tools_result = await session.list_tools()
            tool_names = {t.name for t in tools_result.tools}
            required = {"search", "summorize", "saveToFile", "run_tools_pipeline"}
            missing = sorted(required - tool_names)
            if missing:
                print(f"ERROR: missing tools: {missing}")
                return 1

            call_result = await session.call_tool(
                "run_tools_pipeline",
                {
                    "query": args.query,
                    "file_path": args.file_path,
                    "limit": 5,
                },
            )

    payload: dict[str, object] | None = None
    for item in getattr(call_result, "content", []):
        text = getattr(item, "text", "")
        if not isinstance(text, str) or not text.strip():
            continue
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            print("ERROR: invalid JSON returned by MCP tool")
            return 1
        if isinstance(parsed, dict):
            payload = parsed
            break

    if not payload:
        print("ERROR: empty MCP payload")
        return 1

    status_ok = payload.get("status") == "ok"
    auto_ok = bool(payload.get("auto_executed"))
    transmission_ok = bool(payload.get("transmission_ok"))
    if not (status_ok and auto_ok and transmission_ok):
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        print("ERROR: pipeline validation failed")
        return 1

    print("MCP pipeline: OK")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def main() -> None:
    raise SystemExit(asyncio.run(_run()))


if __name__ == "__main__":
    main()
