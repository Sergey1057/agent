#!/usr/bin/env python3
"""
Минимальный пример локального MCP-вызова:
1) запускает локальный сервер github_mcp_server.py по stdio
2) показывает зарегистрированные tools/list
3) вызывает github_get_repo(owner, repo)
4) печатает результат инструмента
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Локальный вызов GitHub MCP tool")
    parser.add_argument("--owner", required=True, help="Owner репозитория")
    parser.add_argument("--repo", required=True, help="Имя репозитория")
    parser.add_argument(
        "--include-readme",
        action="store_true",
        help="Запросить readme_preview в результате",
    )
    return parser.parse_args()


async def _run() -> int:
    args = _parse_args()
    load_dotenv()
    env = dict(os.environ)
    server_script = str((Path(__file__).resolve().parent / "github_mcp_server.py").resolve())

    server_params = StdioServerParameters(
        command="python3",
        args=[server_script],
        env=env,
    )

    async with stdio_client(server_params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            result = await session.list_tools()
            call_result = await session.call_tool(
                "github_get_repo",
                {
                    "owner": args.owner,
                    "repo": args.repo,
                    "include_readme": args.include_readme,
                },
            )

    tools = result.tools
    print("MCP connection: OK")
    print(f"Tools returned: {len(tools)}")
    for tool in tools:
        print(f"- {tool.name}: {tool.description or 'no description'}")
    print("\nTool call result:")
    for item in getattr(call_result, "content", []):
        text = getattr(item, "text", "")
        if text:
            try:
                data = json.loads(text)
                print(json.dumps(data, ensure_ascii=False, indent=2))
            except json.JSONDecodeError:
                print(text)

    return 0


def main() -> None:
    raise SystemExit(asyncio.run(_run()))


if __name__ == "__main__":
    main()
