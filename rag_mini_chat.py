"""
Мини-чат: история в отдельном файле, каждый ход — RAG, всегда печатаются источники retrieval,
память задачи (цель / уточнения / термины) в JSON + детерминированные маркеры в репликах пользователя.
"""

from __future__ import annotations

import os
from copy import deepcopy
from pathlib import Path
from typing import Any

from agent import LLMAgent, RunResult
from context_strategies import ContextStrategyKind
from document_index.rag import parse_rag_grounding_sections
from task_memory import DialogTaskMemory, parse_task_memory_json_from_reply

_AGENT_DIR = Path(__file__).resolve().parent
_DEFAULT_INDEX = _AGENT_DIR / "memory" / "index_out" / "index_structure.json"


class RagMiniChatAgent(LLMAgent):
    """Подмешивает блок памяти задачи в начало первого system-сообщения."""

    def __init__(
        self,
        *,
        task_memory_path: Path | str,
        **kwargs: Any,
    ) -> None:
        self._dialog_task_memory_path = Path(task_memory_path).expanduser().resolve()
        self.dialog_task_memory = DialogTaskMemory.load(self._dialog_task_memory_path)
        super().__init__(**kwargs)

    def _build_messages_for_api(self) -> list[dict[str, str]]:
        msgs = super()._build_messages_for_api()
        block = self.dialog_task_memory.format_system_block().strip()
        if not block:
            return msgs
        if msgs and str(msgs[0].get("role")) == "system":
            first = dict(msgs[0])
            prev = str(first.get("content") or "").strip()
            first["content"] = block + ("\n\n" + prev if prev else "")
            msgs = [first, *msgs[1:]]
            return msgs
        return [{"role": "system", "content": block}, *msgs]


def _default_task_memory_path() -> Path:
    return (_AGENT_DIR / "memory" / "rag_mini_task_memory.json").resolve()


def _default_history_path() -> Path:
    return (_AGENT_DIR / "memory" / "rag_mini_chat_history.json").resolve()


def print_mini_chat_result(result: RunResult) -> None:
    print(result.text)
    if result.stats is not None:
        line = result.stats.format_line()
        if line:
            print(line)
    print("\n--- RAG: чанки (retrieval), всегда ---")
    if result.rag_source_lines:
        for ln in result.rag_source_lines:
            print(ln)
    else:
        print("(нет попаданий retrieval для этого хода)")
    sec = parse_rag_grounding_sections(result.text)
    print("\n--- RAG: раздел «### Источники» в ответе модели ---")
    src = (sec.get("sources") or "").strip()
    print(src if src else "(пусто — проверьте формат ответа)")
    if result.rag:
        r = result.rag
        g = r.get("grounding") or {}
        suf = r.get("context_sufficient")
        print(
            "\n[RAG] "
            f"контекст_достаточен={suf} | "
            f"разделы: ответ={g.get('has_answer_section')} "
            f"источники={g.get('has_sources_section')} цитаты={g.get('has_quotes_section')} | "
            f"цитаты_дословно_в_чанках={g.get('quotes_verbatim_in_chunks')}"
        )


def _path_or_default(val: Path | str | None, default: Path) -> Path:
    if val is None:
        return default
    s = str(val).strip()
    if not s:
        return default
    return Path(s).expanduser().resolve()


def run_rag_mini_chat_interactive(
    *,
    rag_index: Path | str | None,
    rag_top_k: int | None,
    task_memory_path: Path | str | None,
    history_path: Path | str | None,
) -> None:
    idx_raw = rag_index
    if idx_raw is None or str(idx_raw).strip() == "":
        idx = _DEFAULT_INDEX
    else:
        idx = Path(str(idx_raw).strip()).expanduser()
    if not idx.is_file():
        print(f"Индекс RAG не найден: {idx}")
        print("Укажите --rag-index или положите индекс в memory/index_out/index_structure.json")
        return

    hist = _path_or_default(history_path, _default_history_path())
    task_p = _path_or_default(task_memory_path, _default_task_memory_path())

    os.environ["LLM_AGENT_HISTORY_FILE"] = str(hist)
    os.environ.setdefault("LLM_AGENT_CONTEXT_STRATEGY", "sliding_window")

    agent = RagMiniChatAgent(
        context_strategy=ContextStrategyKind.SLIDING_WINDOW,
        rag_enabled=True,
        rag_index_path=idx,
        rag_top_k=rag_top_k,
        task_memory_path=task_p,
    )
    agent.set_rag(True, index_path=idx)

    print(
        "Мини-чат RAG + память задачи. Пустая строка — выход.\n"
        f"История: {hist}\n"
        f"Память задачи: {task_p}\n"
        f"{agent.rag_status_line()}\n"
        "Маркеры в сообщении пользователя:\n"
        "  ЦЕЛЬ: …  |  УТОЧНЕНИЕ: …  |  ТЕРМИН: ключ = значение\n"
        "Команды: /taskmem show | /taskmem clear\n"
    )
    while True:
        try:
            user = input("Вы: ").rstrip("\n\r")
        except EOFError:
            break
        if user == "":
            break
        ulow = user.strip().lower()
        if ulow in ("/taskmem show", "/taskmem", "/memory-task"):
            m = agent.dialog_task_memory
            print(
                "Память задачи:\n"
                f"  goal: {m.goal or '(пусто)'}\n"
                f"  clarifications: {m.clarifications}\n"
                f"  terms: {m.terms}\n"
            )
            continue
        if ulow in ("/taskmem clear", "/taskmem reset"):
            agent.dialog_task_memory = DialogTaskMemory()
            agent.dialog_task_memory.persist(task_p)
            print("Память задачи очищена.\n")
            continue

        agent.dialog_task_memory.apply_user_line_patterns(user)
        agent.dialog_task_memory.persist(task_p)

        result = agent.run(user, rag=True)
        patch = parse_task_memory_json_from_reply(result.text)
        if patch:
            agent.dialog_task_memory.merge_assistant_patch(patch)
            agent.dialog_task_memory.persist(task_p)

        print_mini_chat_result(result)
        print()


def dry_run_turn_messages(
    *,
    history_file: Path,
    task_file: Path,
    index_path: Path,
    user_messages: list[str],
    assistant_replies: list[str],
) -> tuple[list[str], DialogTaskMemory]:
    """
    Без API: после каждой user-реплики фиксирует текст первого system (как у модели до ответа).
    Возвращает копию итоговой памяти задачи с агента.
    """
    os.environ["LLM_AGENT_HISTORY_FILE"] = str(history_file.resolve())
    agent = RagMiniChatAgent(
        context_strategy=ContextStrategyKind.SLIDING_WINDOW,
        rag_enabled=True,
        rag_index_path=index_path,
        rag_top_k=3,
        task_memory_path=task_file,
    )
    agent.set_rag(True, index_path=index_path)

    systems: list[str] = []
    for i, user in enumerate(user_messages):
        agent.dialog_task_memory.apply_user_line_patterns(user)
        agent._state.messages.append({"role": "user", "content": user})
        api = agent._build_messages_for_api()
        if api and api[0].get("role") == "system":
            systems.append(str(api[0].get("content") or ""))
        if i < len(assistant_replies):
            rep = assistant_replies[i]
            agent._state.messages.append({"role": "assistant", "content": rep})
            p = parse_task_memory_json_from_reply(rep)
            if p:
                agent.dialog_task_memory.merge_assistant_patch(p)
    return systems, deepcopy(agent.dialog_task_memory)
