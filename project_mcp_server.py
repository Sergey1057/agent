#!/usr/bin/env python3
"""
MCP-сервер контекста локального проекта (git и файлы).

Инструменты:
- project_git_branch(repo_path) — текущая ветка
- project_git_list_files(repo_path, subpath="", max_files=80) — список файлов (git ls-files или обход)
- project_git_diff(repo_path, staged=False, max_lines=200) — diff рабочей копии
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("project-context-server")

_TEXT_SUFFIXES = frozenset(
    {
        ".kt",
        ".java",
        ".md",
        ".txt",
        ".json",
        ".yaml",
        ".yml",
        ".xml",
        ".gradle",
        ".kts",
        ".properties",
    }
)


def _resolve_repo(path: str) -> Path:
    p = Path((path or "").strip() or ".").expanduser()
    try:
        return p.resolve()
    except OSError:
        return p.absolute()


def _run_git(args: list[str], cwd: Path, *, timeout: float = 30.0) -> dict[str, Any]:
    try:
        proc = subprocess.run(
            ["git", *args],
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except FileNotFoundError:
        return {"status": "error", "error": "git не найден в PATH"}
    except subprocess.TimeoutExpired:
        return {"status": "error", "error": "git: превышено время ожидания"}
    except OSError as e:
        return {"status": "error", "error": str(e)}

    return {
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }


def _is_git_repo(cwd: Path) -> bool:
    r = _run_git(["rev-parse", "--is-inside-work-tree"], cwd)
    return r.get("returncode") == 0 and (r.get("stdout") or "").strip() == "true"


@mcp.tool(
    description="Возвращает текущую git-ветку и краткий статус репозитория по пути проекта."
)
def project_git_branch(repo_path: str) -> dict[str, Any]:
    cwd = _resolve_repo(repo_path)
    if not cwd.is_dir():
        return {"status": "error", "error": f"Каталог не найден: {cwd}"}
    if not _is_git_repo(cwd):
        return {
            "status": "ok",
            "repo_path": str(cwd),
            "is_git": False,
            "branch": None,
            "note": "Не git-репозиторий",
        }

    branch_r = _run_git(["rev-parse", "--abbrev-ref", "HEAD"], cwd)
    if branch_r.get("returncode") != 0:
        return {
            "status": "error",
            "error": (branch_r.get("stderr") or "не удалось получить ветку").strip(),
        }

    short_r = _run_git(["status", "-sb"], cwd)
    return {
        "status": "ok",
        "repo_path": str(cwd),
        "is_git": True,
        "branch": (branch_r.get("stdout") or "").strip(),
        "status_short": (short_r.get("stdout") or "").strip()[:500],
    }


@mcp.tool(
    description=(
        "Список файлов проекта: git ls-files при наличии .git, иначе рекурсивный обход "
        "(только текстовые расширения)."
    )
)
def project_git_list_files(
    repo_path: str,
    subpath: str = "",
    max_files: int = 80,
) -> dict[str, Any]:
    cwd = _resolve_repo(repo_path)
    if not cwd.is_dir():
        return {"status": "error", "error": f"Каталог не найден: {cwd}"}

    limit = max(1, min(int(max_files), 500))
    base = (cwd / subpath.strip().lstrip("/")) if subpath.strip() else cwd
    if not base.is_dir():
        base = cwd

    files: list[str] = []
    if _is_git_repo(cwd):
        rel = str(base.relative_to(cwd)) if base != cwd else ""
        args = ["ls-files"]
        if rel and rel != ".":
            args.extend(["--", rel])
        r = _run_git(args, cwd)
        if r.get("returncode") == 0:
            for line in (r.get("stdout") or "").splitlines():
                line = line.strip()
                if line:
                    files.append(line)
        else:
            return {
                "status": "error",
                "error": (r.get("stderr") or "git ls-files failed").strip(),
            }
    else:
        for p in sorted(base.rglob("*")):
            if not p.is_file():
                continue
            if p.suffix.lower() not in _TEXT_SUFFIXES and p.name not in (
                "README",
                "REDME.md",
            ):
                continue
            try:
                files.append(str(p.relative_to(cwd)))
            except ValueError:
                files.append(str(p))

    truncated = len(files) > limit
    if truncated:
        files = files[:limit]

    return {
        "status": "ok",
        "repo_path": str(cwd),
        "subpath": subpath.strip() or ".",
        "file_count": len(files),
        "truncated": truncated,
        "files": files,
    }


@mcp.tool(description="Git diff рабочей копии (или --cached при staged=True).")
def project_git_diff(
    repo_path: str,
    staged: bool = False,
    max_lines: int = 200,
) -> dict[str, Any]:
    cwd = _resolve_repo(repo_path)
    if not cwd.is_dir():
        return {"status": "error", "error": f"Каталог не найден: {cwd}"}
    if not _is_git_repo(cwd):
        return {
            "status": "ok",
            "repo_path": str(cwd),
            "is_git": False,
            "diff": "",
            "note": "Не git-репозиторий — diff недоступен",
        }

    args = ["diff", "--no-color"]
    if staged:
        args.append("--cached")
    r = _run_git(args, cwd)
    if r.get("returncode") not in (0, 1):
        return {
            "status": "error",
            "error": (r.get("stderr") or "git diff failed").strip(),
        }

    diff_text = (r.get("stdout") or "").strip()
    limit = max(20, min(int(max_lines), 2000))
    lines = diff_text.splitlines()
    truncated = len(lines) > limit
    if truncated:
        lines = lines[:limit]
        diff_text = "\n".join(lines) + f"\n… (обрезано до {limit} строк)"

    return {
        "status": "ok",
        "repo_path": str(cwd),
        "staged": bool(staged),
        "truncated": truncated,
        "diff": diff_text,
    }


if __name__ == "__main__":
    mcp.run()
