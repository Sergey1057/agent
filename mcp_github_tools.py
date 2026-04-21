#!/usr/bin/env python3
"""
Минимальный пример:
1) подключается к GitHub MCP Server по stdio
2) запрашивает tools/list
3) печатает список доступных инструментов
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys

from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Подключиться к GitHub MCP Server и вывести tools/list"
    )
    parser.add_argument(
        "--token-env",
        default="GITHUB_PERSONAL_ACCESS_TOKEN",
        help="Имя переменной окружения с GitHub PAT",
    )
    parser.add_argument(
        "--server-command",
        default="npx",
        help="Команда для запуска GitHub MCP Server",
    )
    parser.add_argument(
        "--server-args",
        nargs="*",
        default=["-y", "@modelcontextprotocol/server-github"],
        help="Аргументы для server-command",
    )
    return parser.parse_args()


async def _run() -> int:
    args = _parse_args()
    load_dotenv()

    token = os.getenv(args.token_env)
    if not token:
        print(
            f"Предупреждение: {args.token_env} не задан, запускаю сервер без токена.",
            file=sys.stderr,
        )

    env = dict(os.environ)
    if token:
        env[args.token_env] = token

    server_params = StdioServerParameters(
        command=args.server_command,
        args=args.server_args,
        env=env,
    )

    async with stdio_client(server_params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            result = await session.list_tools()

    tools = result.tools
    print("MCP connection: OK")
    print(f"Tools returned: {len(tools)}")
    for tool in tools:
        print(f"- {tool.name}: {tool.description or 'no description'}")

    return 0


def main() -> None:
    raise SystemExit(asyncio.run(_run()))


if __name__ == "__main__":
    main()
