"""
Память задачи диалога: цель, уточнения пользователя, зафиксированные термины/ограничения.
Хранится в JSON отдельно от FSM task_state.json (этапы planning/execution).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_GOAL_RE = re.compile(r"(?im)^\s*цель\s*[:：]\s*(.+)\s*$")
_CLAR_RE = re.compile(r"(?im)^\s*уточнение\s*[:：]\s*(.+)\s*$")
_TERM_EQ_RE = re.compile(r"(?im)^\s*термин\s*[:：]\s*(\S+?)\s*=\s*(.+)\s*$")
_TERM_COLON_RE = re.compile(r"(?im)^\s*термин\s*[:：]\s*(\S+?)\s*[:：]\s*(.+)\s*$")

_TASK_JSON_HEAD = "### Память задачи (JSON)"


@dataclass
class DialogTaskMemory:
    """Состояние «о чём этот диалог» для подмешивания в system."""

    version: int = 1
    goal: str = ""
    clarifications: list[str] = field(default_factory=list)
    terms: dict[str, str] = field(default_factory=dict)

    @staticmethod
    def load(path: Path) -> DialogTaskMemory:
        if not path.is_file():
            return DialogTaskMemory()
        try:
            raw = json.loads(path.read_text(encoding="utf-8-sig"))
        except (OSError, json.JSONDecodeError):
            return DialogTaskMemory()
        if not isinstance(raw, dict):
            return DialogTaskMemory()
        goal = str(raw.get("goal") or "").strip()
        clar: list[str] = []
        c = raw.get("clarifications")
        if isinstance(c, list):
            for x in c:
                if isinstance(x, str) and x.strip():
                    clar.append(x.strip())
        terms: dict[str, str] = {}
        t = raw.get("terms")
        if isinstance(t, dict):
            for k, v in t.items():
                if isinstance(k, str) and isinstance(v, str) and k.strip() and v.strip():
                    terms[k.strip()] = v.strip()
        return DialogTaskMemory(goal=goal, clarifications=clar, terms=terms)

    def persist(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": self.version,
            "type": "dialog_task_memory",
            "goal": self.goal,
            "clarifications": list(self.clarifications),
            "terms": dict(self.terms),
        }
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        tmp.replace(path)

    def apply_user_line_patterns(self, user_line: str) -> None:
        """Детерминированные маркеры в сообщении пользователя (удобно для сценариев и скриптов)."""
        for line in (user_line or "").splitlines():
            s = line.strip()
            if not s:
                continue
            m = _GOAL_RE.match(s)
            if m:
                self.goal = m.group(1).strip()
                continue
            m = _CLAR_RE.match(s)
            if m:
                self.clarifications.append(m.group(1).strip())
                continue
            m = _TERM_EQ_RE.match(s) or _TERM_COLON_RE.match(s)
            if m:
                self.terms[m.group(1).strip()] = m.group(2).strip()

    def merge_assistant_patch(self, patch: dict[str, Any]) -> None:
        """Слияние JSON из ответа модели (частичное обновление)."""
        g = patch.get("goal")
        if isinstance(g, str) and g.strip():
            self.goal = g.strip()
        cl = patch.get("clarifications")
        if isinstance(cl, list):
            for x in cl:
                if isinstance(x, str) and x.strip():
                    self.clarifications.append(x.strip())
        te = patch.get("terms")
        if isinstance(te, dict):
            for k, v in te.items():
                if isinstance(k, str) and isinstance(v, str) and k.strip() and v.strip():
                    self.terms[k.strip()] = v.strip()

    def format_system_block(self) -> str:
        """Текст для system: текущая память + правило JSON в конце ответа."""
        clar_lines = "\n".join(f"  - {c}" for c in self.clarifications) or "  (пока нет)"
        term_lines = (
            "\n".join(f"  - {k}: {v}" for k, v in sorted(self.terms.items()))
            or "  (пока нет)"
        )
        goal_s = self.goal.strip() or "(цель ещё не сформулирована явно — выведи из диалога)"
        return (
            "ПАМЯТЬ ЗАДАЧИ (держи в голове на каждом ходе; не противоречь зафиксированному):\n"
            f"- Цель диалога: {goal_s}\n"
            "- Уже уточнено пользователем:\n"
            f"{clar_lines}\n"
            "- Зафиксированные термины и ограничения:\n"
            f"{term_lines}\n"
            "\n"
            "В конце **каждого** своего ответа (после разделов ### Ответ / ### Источники / ### Цитаты) "
            "добавь блок обновления памяти задачи **ровно** в таком виде:\n"
            f"{_TASK_JSON_HEAD}\n"
            "{{ \"goal\": \"…\", \"clarifications\": [\"новое уточнение с этого хода\"], "
            '"terms": { "аббревиатура": "значение" } }\n'
            "\n"
            "Правила JSON-блока:\n"
            "- goal: одна строка — текущая цель; если не менялась, повтори прежнюю формулировку из блока выше.\n"
            "- clarifications: только **новые** фразы с этого хода (можно []). Не дублируй весь список.\n"
            "- terms: только новые или изменённые пары; объединение с уже зафиксированным делается снаружи.\n"
            "- Если обновлений нет: \"clarifications\": [], \"terms\": {{}}.\n"
        )


def _first_json_object(s: str) -> dict[str, Any] | None:
    i = s.find("{")
    if i < 0:
        return None
    depth = 0
    for j in range(i, len(s)):
        ch = s[j]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                chunk = s[i : j + 1]
                try:
                    out = json.loads(chunk)
                except json.JSONDecodeError:
                    return None
                return out if isinstance(out, dict) else None
    return None


def parse_task_memory_json_from_reply(reply: str) -> dict[str, Any] | None:
    """Достаёт объект памяти из ответа ассистента."""
    t = reply or ""
    idx = t.rfind(_TASK_JSON_HEAD)
    if idx >= 0:
        tail = t[idx + len(_TASK_JSON_HEAD) :].strip()
        return _first_json_object(tail)
    fence = "```json"
    j = t.rfind(fence)
    if j >= 0:
        rest = t[j + len(fence) :]
        end = rest.find("```")
        if end >= 0:
            return _first_json_object(rest[:end])
    return _first_json_object(t)


def snapshot_dict(m: DialogTaskMemory) -> dict[str, Any]:
    return {
        "goal": m.goal,
        "clarifications": list(m.clarifications),
        "terms": dict(m.terms),
    }
