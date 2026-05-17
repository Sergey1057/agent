"""
Ассистент разработчика: RAG по README/docs проекта + MCP (git branch, файлы, diff).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

from document_index.corpus import collect_corpus_paths, build_index_for_corpus
from document_index.rag import (
    RagRetrievalConfig,
    format_rag_hit_lines,
    resolve_rag_index_path,
    retrieve_for_rag,
)

if TYPE_CHECKING:
    from agent import LLMAgent

_AGENT_DIR = Path(__file__).resolve().parent
_DEFAULT_PROJECT_ROOT = Path(
    "/Users/sergei/Documents/ai-course/AiAdvent1"
)
_DEFAULT_INDEX_REL = Path("memory/index_out/project_docs_index.json")
_DEV_PROMPT_REL = Path("memory/prompts/dev_assistant_rag.txt")


def default_project_root() -> Path:
    raw = (os.environ.get("LLM_AGENT_PROJECT_ROOT") or "").strip()
    if raw:
        return Path(raw).expanduser().resolve()
    return _DEFAULT_PROJECT_ROOT.resolve()


def default_project_index_path() -> Path:
    env = (os.environ.get("LLM_AGENT_PROJECT_RAG_INDEX") or "").strip()
    if env:
        return resolve_rag_index_path(env)
    for base in (_AGENT_DIR, Path.cwd()):
        cand = (base / _DEFAULT_INDEX_REL).resolve()
        if cand.is_file():
            return cand
    return (_AGENT_DIR / _DEFAULT_INDEX_REL).resolve()


def dev_assistant_prompt_template() -> str:
    p = _AGENT_DIR / _DEV_PROMPT_REL
    if p.is_file():
        return p.read_text(encoding="utf-8")
    return (
        "Ответь по документации проекта.\n\n{{GIT_CONTEXT}}\n\n{{EXCERPTS}}\n\n"
        "Вопрос: {{QUESTION}}"
    )


def build_project_docs_index(
    project_root: Path | None = None,
    *,
    out_path: Path | None = None,
    dummy_embeddings: bool = False,
) -> dict[str, Any]:
    """Индексирует README + docs/ (+ схемы) в один JSON для RAG."""
    root = (project_root or default_project_root()).resolve()
    paths = collect_corpus_paths(root)
    if not paths:
        raise FileNotFoundError(
            f"Документация не найдена в {root}. Ожидаются README* и каталог docs/."
        )
    out = out_path or default_project_index_path()
    return build_index_for_corpus(
        paths,
        strategy="structure",
        out_path=out,
        project_root=root,
        dummy_embeddings=dummy_embeddings,
    )


def _format_git_context(
    branch_payload: dict[str, Any],
    files_payload: dict[str, Any] | None,
    diff_payload: dict[str, Any] | None,
) -> str:
    lines = ["## Контекст репозитория (MCP)"]
    if branch_payload.get("status") != "ok":
        lines.append(f"Git: ошибка — {branch_payload.get('error', branch_payload)}")
    elif not branch_payload.get("is_git", True):
        lines.append(f"Путь: {branch_payload.get('repo_path')} (не git)")
    else:
        lines.append(f"Путь: {branch_payload.get('repo_path')}")
        lines.append(f"Ветка: {branch_payload.get('branch')}")
        st = branch_payload.get("status_short")
        if st:
            lines.append(f"Статус: {st}")

    if files_payload and files_payload.get("status") == "ok":
        fl = files_payload.get("files") or []
        lines.append(f"\nФайлы ({files_payload.get('file_count', len(fl))}):")
        for f in fl[:40]:
            lines.append(f"  - {f}")
        if files_payload.get("truncated"):
            lines.append("  … список обрезан")

    if diff_payload and diff_payload.get("status") == "ok":
        diff = (diff_payload.get("diff") or "").strip()
        if diff:
            lines.append("\nDiff (фрагмент):")
            lines.append(diff[:3000])

    return "\n".join(lines)


def _build_dev_user_message(
    question: str,
    *,
    git_context: str,
    rag_prompt: str,
    excerpts_block: str,
) -> str:
    tpl = rag_prompt
    if "{{GIT_CONTEXT}}" in tpl:
        body = tpl.replace("{{GIT_CONTEXT}}", git_context.strip())
    else:
        body = git_context.strip() + "\n\n" + tpl
    body = body.replace("{{EXCERPTS}}", excerpts_block.strip())
    body = body.replace("{{QUESTION}}", question.strip())
    return body.strip()


def answer_project_help(
    agent: LLMAgent,
    question: str,
    *,
    project_root: Path | None = None,
    index_path: Path | None = None,
    include_file_list: bool = True,
    include_diff: bool = False,
    top_k: int = 5,
) -> str:
    """
    Ответ на вопрос о проекте: MCP git + RAG по индексу документации.
    """
    root = (project_root or default_project_root()).resolve()
    idx = index_path or default_project_index_path()
    if not idx.is_file():
        return (
            f"Индекс документации не найден: {idx}\n"
            "Соберите индекс:\n"
            "  python -m document_index --project-root "
            f"{root} --build-project-index --dummy-embeddings\n"
            "или: python -m dev_assistant --build-index"
        )

    repo_s = str(root)
    branch_payload = agent.fetch_project_git_branch_via_mcp(repo_s)
    files_payload: dict[str, Any] | None = None
    diff_payload: dict[str, Any] | None = None
    if include_file_list:
        files_payload = agent.fetch_project_list_files_via_mcp(repo_s, max_files=60)
    if include_diff:
        diff_payload = agent.fetch_project_git_diff_via_mcp(repo_s, max_lines=120)

    git_ctx = _format_git_context(branch_payload, files_payload, diff_payload)

    cfg = RagRetrievalConfig.from_env(top_k)
    rag_out = retrieve_for_rag(question, idx, top_k=top_k, config=cfg)

    excerpts_parts: list[str] = []
    if rag_out.hits:
        excerpts_parts.append("## Выдержки из документации (RAG)")
        for h in rag_out.hits:
            meta = h.get("metadata") if isinstance(h.get("metadata"), dict) else {}
            src = meta.get("source") or meta.get("title") or "документ"
            section = meta.get("section") or ""
            header = f"### [{Path(str(src)).name}]"
            if section:
                header += f" — {section}"
            excerpts_parts.append(header)
            excerpts_parts.append((h.get("text") or "").strip())
        excerpts_parts.append("")
        excerpts_parts.extend(format_rag_hit_lines(rag_out.hits))
    elif not rag_out.context_sufficient:
        excerpts_parts.append(
            f"(Релевантные фрагменты не найдены: {rag_out.weak_reason or 'слабый retrieval'})"
        )

    user_msg = _build_dev_user_message(
        question,
        git_context=git_ctx,
        rag_prompt=dev_assistant_prompt_template(),
        excerpts_block="\n".join(excerpts_parts),
    )

    result = agent.run(user_msg, rag=False)
    footer_parts: list[str] = []
    if branch_payload.get("branch"):
        footer_parts.append(f"[git: {branch_payload.get('branch')}]")
    if rag_out.hits:
        sources = sorted(
            {
                str(h.get("metadata", {}).get("source", ""))
                for h in rag_out.hits
                if h.get("metadata", {}).get("source")
            }
        )
        if sources:
            footer_parts.append("[RAG: " + ", ".join(Path(s).name for s in sources[:5]) + "]")
    if footer_parts:
        return result.text.rstrip() + "\n\n" + " ".join(footer_parts)
    return result.text


def main() -> int:
    import argparse

    p = argparse.ArgumentParser(description="Ассистент разработчика: сборка индекса документации.")
    p.add_argument(
        "--project-root",
        type=Path,
        default=None,
        help="Корень проекта (иначе LLM_AGENT_PROJECT_ROOT)",
    )
    p.add_argument("--build-index", action="store_true", help="Построить RAG-индекс")
    p.add_argument("--out", type=Path, default=None, help="Путь к JSON-индексу")
    p.add_argument("--dummy-embeddings", action="store_true")
    p.add_argument("--list-sources", action="store_true", help="Только список файлов корпуса")
    args = p.parse_args()
    root = (args.project_root or default_project_root()).resolve()

    if args.list_sources:
        for path in collect_corpus_paths(root):
            print(path)
        return 0

    if args.build_index:
        meta = build_project_docs_index(
            root,
            out_path=args.out,
            dummy_embeddings=args.dummy_embeddings,
        )
        print(json.dumps(
            {
                "index": str(args.out or default_project_index_path()),
                "sources": meta.get("corpus_sources"),
                "chunks": meta.get("chunk_stats", {}).get("chunk_count"),
            },
            ensure_ascii=False,
            indent=2,
        ))
        return 0

    p.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
