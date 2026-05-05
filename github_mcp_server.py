#!/usr/bin/env python3
"""
Локальный MCP-сервер с инструментом GitHub API.
Инструмент:
- github_get_repo(owner, repo, include_readme=False)
"""

from __future__ import annotations

import json
import ssl
import urllib.error
import urllib.request
from typing import Any

import certifi
from mcp.server.fastmcp import FastMCP


mcp = FastMCP("github-api-server")


def _http_get_json(url: str, headers: dict[str, str] | None = None) -> dict[str, Any]:
    req = urllib.request.Request(url, headers=headers or {}, method="GET")
    ctx = ssl.create_default_context(cafile=certifi.where())
    with urllib.request.urlopen(req, timeout=30.0, context=ctx) as resp:
        raw = resp.read().decode("utf-8")
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise RuntimeError("Ожидался JSON-объект от GitHub API.")
    return data


@mcp.tool(
    description=(
        "Возвращает метаданные репозитория GitHub по owner/repo: "
        "название, описание, звезды, язык, URL и опционально README."
    )
)
def github_get_repo(owner: str, repo: str, include_readme: bool = False) -> dict[str, Any]:
    """
    Вход:
    - owner: владелец репозитория (например, "python")
    - repo: имя репозитория (например, "cpython")
    - include_readme: если True, дополнительно загружается README в формате raw

    Выход:
    - JSON-объект с полями репозитория и status="ok" либо status="error"
    """
    owner = (owner or "").strip()
    repo = (repo or "").strip()
    if not owner or not repo:
        return {
            "status": "error",
            "error": "Параметры owner и repo обязательны.",
        }

    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "llm-agent-mcp-server",
    }
    repo_url = f"https://api.github.com/repos/{owner}/{repo}"
    try:
        payload = _http_get_json(repo_url, headers=headers)
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        return {
            "status": "error",
            "error": f"GitHub API HTTP {e.code}",
            "details": body[:1000],
        }
    except (OSError, json.JSONDecodeError, RuntimeError) as e:
        return {"status": "error", "error": str(e)}

    result: dict[str, Any] = {
        "status": "ok",
        "name": payload.get("name"),
        "full_name": payload.get("full_name"),
        "description": payload.get("description"),
        "stargazers_count": payload.get("stargazers_count"),
        "forks_count": payload.get("forks_count"),
        "open_issues_count": payload.get("open_issues_count"),
        "language": payload.get("language"),
        "html_url": payload.get("html_url"),
    }

    if include_readme:
        readme_url = f"https://api.github.com/repos/{owner}/{repo}/readme"
        headers_readme = dict(headers)
        headers_readme["Accept"] = "application/vnd.github.raw+json"
        try:
            req = urllib.request.Request(readme_url, headers=headers_readme, method="GET")
            ctx = ssl.create_default_context(cafile=certifi.where())
            with urllib.request.urlopen(req, timeout=30.0, context=ctx) as resp:
                readme = resp.read().decode("utf-8", errors="replace")
            result["readme_preview"] = readme[:2000]
        except urllib.error.HTTPError as e:
            result["readme_preview_error"] = f"HTTP {e.code}"
        except OSError as e:
            result["readme_preview_error"] = str(e)

    return result


if __name__ == "__main__":
    mcp.run()
