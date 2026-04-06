"""
Стратегии управления контекстом для LLMAgent: sliding window, sticky facts, branching.
Формат файла истории: version 3 (см. load_unified_state / save_unified_state).
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable


def _read_recent_message_window() -> int:
    """
    Сколько последних реплик (user/assistant) хранить в файле и отдавать в API
    для sliding_window и sticky_facts.

    По умолчанию 6 — это три полных обмена «запрос–ответ» (3 user + 3 assistant).
    Раньше было 3 реплики: после второго хода обрезка давала в JSON только
    «одну пару» в нормальном виде (assistant, user, assistant).

    Переопределение: LLM_AGENT_RECENT_MESSAGES=3 — ровно три реплики в массиве.
    """
    raw = (os.environ.get("LLM_AGENT_RECENT_MESSAGES") or "").strip()
    if not raw:
        return 6
    try:
        n = int(raw)
        return max(1, min(n, 500))
    except ValueError:
        return 6


# Значение читается при импорте модуля (как и раньше константа).
RECENT_MESSAGE_WINDOW = _read_recent_message_window()

FACTS_SYSTEM_PROMPT = (
    "Ты извлекаешь и обновляешь структурированные факты из диалога. "
    "Ключи — короткие идентификаторы (цель, ограничения, предпочтения, решения, договоренности, клуб, язык). "
    "Значения — краткие строки. "
    "Ответь строго одним JSON-объектом: {\"ключ\": \"значение\", ...}. "
    "Все значения — строки в двойных кавычках. Без текста до или после JSON, без markdown, без комментариев."
)


class ContextStrategyKind(str, Enum):
    SLIDING_WINDOW = "sliding_window"
    STICKY_FACTS = "sticky_facts"
    BRANCHING = "branching"


def _normalize_chat_messages(raw: Any) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    if not isinstance(raw, list):
        return out
    for m in raw:
        if not isinstance(m, dict):
            continue
        role = m.get("role")
        content = m.get("content")
        if role not in ("user", "assistant", "system"):
            continue
        if not isinstance(content, str):
            content = "" if content is None else str(content)
        out.append({"role": str(role), "content": content})
    return out


def _trim_tail(messages: list[dict[str, str]], max_count: int) -> None:
    if len(messages) > max_count:
        del messages[: len(messages) - max_count]


def trim_messages(messages: list[dict[str, str]], max_count: int) -> None:
    """Обрезает список сообщений с начала, оставляя не больше max_count последних."""
    _trim_tail(messages, max_count)


def _facts_dict_from_parsed(data: Any) -> dict[str, str]:
    """Превращает распарсенный JSON в плоский dict[str, str]."""
    out: dict[str, str] = {}
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                out.update(_facts_dict_from_parsed(item))
        return out
    if not isinstance(data, dict):
        return out
    for k, v in data.items():
        if k is None:
            continue
        key = str(k).strip()
        if not key:
            continue
        if v is None:
            out[key] = ""
        elif isinstance(v, (dict, list)):
            out[key] = json.dumps(v, ensure_ascii=False)
        else:
            out[key] = str(v).strip()
    return out


def _try_json_loads_candidates(candidates: list[str]) -> dict[str, str]:
    for cand in candidates:
        if not cand:
            continue
        try:
            data = json.loads(cand)
        except json.JSONDecodeError:
            continue
        flat = _facts_dict_from_parsed(data)
        if flat:
            return flat
    return {}


def _parse_facts_json(text: str) -> dict[str, str]:
    """
    Достаёт JSON из ответа модели: блок ```json```, целая строка, срез {…}, затем поиск {...} в тексте.
    """
    s = (text or "").strip()
    if not s:
        return {}

    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", s, re.IGNORECASE)
    if fence:
        inner = fence.group(1).strip()
        got = _try_json_loads_candidates([inner])
        if got:
            return got

    candidates = [s]
    if "{" in s:
        start, end = s.find("{"), s.rfind("}")
        if start >= 0 and end > start:
            candidates.append(s[start : end + 1])
    got = _try_json_loads_candidates(candidates)
    if got:
        return got

    decoder = json.JSONDecoder()
    for i, ch in enumerate(s):
        if ch not in "{[":
            continue
        try:
            obj, _end = decoder.raw_decode(s, i)
        except json.JSONDecodeError:
            continue
        flat = _facts_dict_from_parsed(obj)
        if flat:
            return flat
    return {}


@dataclass
class UnifiedChatState:
    """Состояние чата v3: стратегия + поля для каждого режима."""

    strategy: ContextStrategyKind = ContextStrategyKind.SLIDING_WINDOW
    # sliding + sticky: линейная лента (не больше RECENT_MESSAGE_WINDOW после сохранения)
    messages: list[dict[str, str]] = field(default_factory=list)
    facts: dict[str, str] = field(default_factory=dict)
    # branching: до split — только messages; после split — prefix + ветки
    branching_split: bool = False
    branch_prefix: list[dict[str, str]] = field(default_factory=list)
    branches: dict[str, list[dict[str, str]]] = field(default_factory=dict)
    active_branch: str = "branch_a"

    BRANCH_IDS: tuple[str, str] = ("branch_a", "branch_b")


def load_unified_state(path: Path) -> UnifiedChatState:
    if not path.is_file():
        return UnifiedChatState()
    try:
        text = path.read_text(encoding="utf-8-sig")
        data = json.loads(text)
    except (OSError, json.JSONDecodeError):
        return UnifiedChatState()

    # v2: { version, summary, messages }
    if isinstance(data, dict) and data.get("version") == 2:
        raw_list = data.get("messages")
        msgs = _normalize_chat_messages(raw_list)
        _trim_tail(msgs, RECENT_MESSAGE_WINDOW)
        return UnifiedChatState(
            strategy=ContextStrategyKind.SLIDING_WINDOW,
            messages=msgs,
        )

    # v3
    if isinstance(data, dict) and data.get("version") == 3:
        strat_raw = (data.get("context_strategy") or "sliding_window").strip()
        try:
            strategy = ContextStrategyKind(strat_raw)
        except ValueError:
            strategy = ContextStrategyKind.SLIDING_WINDOW

        facts_raw = data.get("facts")
        facts: dict[str, str] = {}
        if isinstance(facts_raw, dict):
            for k, v in facts_raw.items():
                if k is None:
                    continue
                facts[str(k)] = "" if v is None else str(v)

        msgs = _normalize_chat_messages(data.get("messages"))
        bdata = data.get("branching")
        branching_split = False
        branch_prefix: list[dict[str, str]] = []
        branches: dict[str, list[dict[str, str]]] = {}
        active_branch = "branch_a"

        if isinstance(bdata, dict):
            branching_split = bool(bdata.get("split", False))
            branch_prefix = _normalize_chat_messages(bdata.get("prefix"))
            br = bdata.get("branches")
            if isinstance(br, dict):
                for bid, arr in br.items():
                    if bid is None:
                        continue
                    branches[str(bid)] = _normalize_chat_messages(arr)
            ab = bdata.get("active_branch")
            if isinstance(ab, str) and ab.strip():
                active_branch = ab.strip()

        st = UnifiedChatState(
            strategy=strategy,
            messages=msgs,
            facts=facts,
            branching_split=branching_split,
            branch_prefix=branch_prefix,
            branches=branches,
            active_branch=active_branch,
        )
        if st.strategy == ContextStrategyKind.SLIDING_WINDOW:
            _trim_tail(st.messages, RECENT_MESSAGE_WINDOW)
        elif st.strategy == ContextStrategyKind.STICKY_FACTS:
            _trim_tail(st.messages, RECENT_MESSAGE_WINDOW)
        return st

    # legacy: только список сообщений
    if isinstance(data, list):
        msgs = _normalize_chat_messages(data)
        _trim_tail(msgs, RECENT_MESSAGE_WINDOW)
        return UnifiedChatState(messages=msgs)

    return UnifiedChatState()


def save_unified_state(path: Path, state: UnifiedChatState) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    branching_obj: dict[str, Any] | None = None
    if state.strategy == ContextStrategyKind.BRANCHING and state.branching_split:
        branching_obj = {
            "split": True,
            "prefix": list(state.branch_prefix),
            "branches": {k: list(v) for k, v in state.branches.items()},
            "active_branch": state.active_branch,
        }

    to_save = state.messages
    if state.strategy == ContextStrategyKind.SLIDING_WINDOW:
        to_save = list(state.messages)[-RECENT_MESSAGE_WINDOW:]
    elif state.strategy == ContextStrategyKind.STICKY_FACTS:
        to_save = list(state.messages)[-RECENT_MESSAGE_WINDOW:]

    facts_out: dict[str, str] = (
        dict(state.facts) if state.strategy == ContextStrategyKind.STICKY_FACTS else {}
    )
    payload: dict[str, Any] = {
        "version": 3,
        "context_strategy": state.strategy.value,
        "messages": to_save,
        "facts": facts_out,
    }
    if branching_obj is not None:
        payload["branching"] = branching_obj

    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    tmp.replace(path)


def build_sliding_api_messages(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    return list(messages[-RECENT_MESSAGE_WINDOW:])


def build_sticky_facts_api_messages(
    facts: dict[str, str], messages: list[dict[str, str]]
) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    if facts:
        lines = [f"- {k}: {v}" for k, v in sorted(facts.items()) if k]
        block = (
            "Известные факты и договорённости (ключ–значение):\n"
            + "\n".join(lines)
        )
        out.append({"role": "system", "content": block})
    out.extend(messages[-RECENT_MESSAGE_WINDOW:])
    return out


def merge_adjacent_same_role(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    """
    Склеивает подряд идущие сообщения с одной ролью (частый случай на стыке
    checkpoint и ветки: последняя реплика prefix — user, первая в ветке — снова user).
    Без этого многие chat API плохо учитывают вторую реплику.
    """
    if not messages:
        return []
    out: list[dict[str, str]] = []
    cur = {
        "role": str(messages[0].get("role", "user")),
        "content": str(messages[0].get("content", "")),
    }
    for m in messages[1:]:
        role = str(m.get("role", "user"))
        content = str(m.get("content", ""))
        if role == cur["role"]:
            cur["content"] = (cur["content"] + "\n\n" + content).strip()
        else:
            out.append(cur)
            cur = {"role": role, "content": content}
    out.append(cur)
    return out


def build_branching_api_messages(state: UnifiedChatState) -> list[dict[str, str]]:
    if not state.branching_split:
        return list(state.messages)
    active = state.active_branch or "branch_a"
    branch_msgs = state.branches.get(active, [])
    combined = list(state.branch_prefix) + list(branch_msgs)
    merged = merge_adjacent_same_role(combined)
    system = {
        "role": "system",
        "content": (
            f"Ты продолжаешь ветку диалога «{active}». "
            "Ниже сначала общий checkpoint (история до разветвления), "
            "затем только сообщения этой ветки. Учитывай весь показанный контекст при ответе."
        ),
    }
    return [system] + merged


def update_facts_with_llm(
    complete_fn: Callable[[list[dict[str, str]]], str],
    current_facts: dict[str, str],
    recent_messages: list[dict[str, str]],
    new_user_message: str,
) -> dict[str, str]:
    """Вызывает LLM для обновления facts после сообщения пользователя."""
    payload = {
        "текущие_факты": current_facts,
        "последние_реплики": [
            {"role": m.get("role"), "content": m.get("content", "")}
            for m in recent_messages[-RECENT_MESSAGE_WINDOW:]
        ],
        "новое_сообщение_пользователя": new_user_message,
    }
    user_content = (
        "Обнови факты с учётом нового сообщения. "
        "Сохрани релевантные старые факты, добавь новые, убери противоречащие. "
        "Верни только JSON.\n\n"
        + json.dumps(payload, ensure_ascii=False)
    )
    msgs = [
        {"role": "system", "content": FACTS_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
    raw = complete_fn(msgs)
    merged = _parse_facts_json(raw)
    if not merged:
        # Модель часто возвращает пояснение без JSON — не затираем уже накопленные facts
        return dict(current_facts)
    out = dict(current_facts)
    out.update(merged)
    return out


def split_into_two_branches(state: UnifiedChatState) -> tuple[bool, str]:
    """
    Checkpoint: текущая линейная лента становится prefix; две пустые ветки.
    Возвращает (ok, сообщение об ошибке).
    """
    if state.strategy != ContextStrategyKind.BRANCHING:
        return False, "Ветвление доступно только при стратегии branching."
    if state.branching_split:
        return False, "Ветки уже созданы. Сбросьте историю или начните новый чат."
    state.branch_prefix = list(state.messages)
    state.branches = {state.BRANCH_IDS[0]: [], state.BRANCH_IDS[1]: []}
    state.active_branch = state.BRANCH_IDS[0]
    state.branching_split = True
    state.messages = []
    return True, ""


def switch_branch(state: UnifiedChatState, branch_id: str) -> tuple[bool, str]:
    bid = (branch_id or "").strip()
    if state.strategy != ContextStrategyKind.BRANCHING:
        return False, "Переключение веток только в режиме branching."
    if not state.branching_split:
        return False, "Сначала создайте ветки (checkpoint / split)."
    if bid not in state.branches:
        known = ", ".join(sorted(state.branches.keys()))
        return False, f"Неизвестная ветка: {bid}. Доступны: {known}"
    state.active_branch = bid
    return True, ""


def list_branch_ids(state: UnifiedChatState) -> list[str]:
    if not state.branching_split:
        return []
    return list(state.branches.keys())


def flatten_messages_for_export(state: UnifiedChatState) -> list[dict[str, str]]:
    """Линейный вид истории для отображения / обратной совместимости."""
    if state.strategy != ContextStrategyKind.BRANCHING or not state.branching_split:
        return list(state.messages)
    return list(state.branch_prefix) + list(
        state.branches.get(state.active_branch, [])
    )
