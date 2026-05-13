"""
Сравнение качества retrieval: legacy (без порога/гибрида/rewrite) vs enhanced.

  python -m document_index.rag_compare --index memory/index_out/index_structure.json \\
      --questions "установить мобильное приложение" "MCP сервер"

Переменные окружения (как у агента): LLM_AGENT_RAG_RETRIEVE_K, LLM_AGENT_RAG_MIN_SIM,
LLM_AGENT_RAG_QUERY_REWRITE, LLM_AGENT_RAG_HYBRID, LLM_AGENT_RAG_LEGACY, …
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from document_index.rag import compare_rag_modes

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None


def _default_questions() -> list[str]:
    return [
        "Как установить мобильное приложение по туру?",
        "Что такое MCP и как подключить сервер?",
        "расскажи про память агента пожалуйста",
    ]


def main() -> int:
    if load_dotenv:
        load_dotenv()

    p = argparse.ArgumentParser(
        description="A/B: RAG без улучшений vs с фильтром, гибридным реранком и query rewrite.",
    )
    p.add_argument("--index", type=Path, required=True, help="Путь к index_*.json")
    p.add_argument(
        "--questions-file",
        type=Path,
        default=None,
        help="Файл: один вопрос на строку (пустые и # — пропуск)",
    )
    p.add_argument(
        "--questions",
        nargs="*",
        default=[],
        help="Вопросы в argv (если пусто и нет --questions-file — встроенный набор)",
    )
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument(
        "--dummy-embeddings",
        action="store_true",
        help="Принудительно dummy-режим (лексика), даже если в индексе не hash",
    )
    args = p.parse_args()
    idx = args.index.expanduser().resolve()
    if not idx.is_file():
        print(f"Индекс не найден: {idx}", file=sys.stderr)
        return 1

    qs: list[str] = []
    if args.questions_file is not None:
        fp = args.questions_file.expanduser().resolve()
        if not fp.is_file():
            print(f"Файл вопросов не найден: {fp}", file=sys.stderr)
            return 1
        for line in fp.read_text(encoding="utf-8").splitlines():
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            qs.append(s)
    qs.extend(q.strip() for q in args.questions if q.strip())
    if not qs:
        qs = _default_questions()

    report: list[dict[str, object]] = []
    for q in qs:
        report.append(
            compare_rag_modes(
                q,
                idx,
                top_k=args.top_k,
                dummy_embeddings=True if args.dummy_embeddings else None,
            )
        )

    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
