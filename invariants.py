"""
Инварианты проекта: архитектура, принятые решения, стек, бизнес-правила.
Хранятся отдельно от истории диалога (см. файл в каталоге памяти).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_AGENT_DIR = Path(__file__).resolve().parent
INVARIANTS_FILENAME = "agent_invariants.json"

# Ключи в JSON и в CLI
SECTION_KEYS: tuple[str, ...] = (
    "architecture",
    "technical_decisions",
    "stack",
    "business_rules",
)

_SECTION_ALIASES: dict[str, str] = {
    "architecture": "architecture",
    "arch": "architecture",
    "архитектура": "architecture",
    "technical_decisions": "technical_decisions",
    "technical": "technical_decisions",
    "decisions": "technical_decisions",
    "tech": "technical_decisions",
    "решения": "technical_decisions",
    "stack": "stack",
    "стек": "stack",
    "business_rules": "business_rules",
    "business": "business_rules",
    "бизнес": "business_rules",
}


def canonical_section(name: str) -> str | None:
    k = (name or "").strip().lower().replace("-", "_")
    if not k:
        return None
    return _SECTION_ALIASES.get(k, k if k in SECTION_KEYS else None)


def section_title(key: str) -> str:
    return {
        "architecture": "Архитектура (зафиксированная)",
        "technical_decisions": "Принятые технические решения",
        "stack": "Ограничения по стеку",
        "business_rules": "Бизнес-правила",
    }.get(key, key)


@dataclass
class ProjectInvariants:
    """Тексты по разделам; пустые строки не попадают в системное сообщение."""

    architecture: str = ""
    technical_decisions: str = ""
    stack: str = ""
    business_rules: str = ""
    extra: dict[str, str] = field(default_factory=dict)


def _invariants_from_dict(data: Any) -> ProjectInvariants:
    if not isinstance(data, dict):
        return ProjectInvariants()
    extra_raw = data.get("extra")
    extra: dict[str, str] = {}
    if isinstance(extra_raw, dict):
        for k, v in extra_raw.items():
            key = str(k).strip() if k is not None else ""
            if not key:
                continue
            extra[key] = "" if v is None else str(v).strip()

    def s(key: str) -> str:
        v = data.get(key)
        return "" if v is None else str(v).strip()

    return ProjectInvariants(
        architecture=s("architecture"),
        technical_decisions=s("technical_decisions"),
        stack=s("stack"),
        business_rules=s("business_rules"),
        extra=extra,
    )


def _invariants_to_dict(inv: ProjectInvariants) -> dict[str, Any]:
    return {
        "architecture": inv.architecture,
        "technical_decisions": inv.technical_decisions,
        "stack": inv.stack,
        "business_rules": inv.business_rules,
        "extra": dict(sorted(inv.extra.items())),
    }


class InvariantsStore:
    """Инварианты на диске; одна запись на проект (каталог памяти)."""

    def __init__(self, memory_dir: Path | None = None) -> None:
        self._dir = memory_dir or self._resolve_memory_dir()
        self._path = self._dir / INVARIANTS_FILENAME
        self._data = ProjectInvariants()
        self._load_into()

    @staticmethod
    def _resolve_memory_dir() -> Path:
        raw = (os.environ.get("LLM_AGENT_MEMORY_DIR") or "").strip()
        if raw:
            return Path(raw).expanduser().resolve()
        return (_AGENT_DIR / "memory").resolve()

    def _read_json(self) -> Any:
        if not self._path.is_file():
            return None
        try:
            return json.loads(self._path.read_text(encoding="utf-8-sig"))
        except (OSError, json.JSONDecodeError):
            return None

    def _write_json(self, payload: dict[str, Any]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_name(self._path.name + ".tmp")
        tmp.write_text(
            json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
            encoding="utf-8",
        )
        tmp.replace(self._path)

    def _load_into(self) -> None:
        raw = self._read_json()
        if not isinstance(raw, dict):
            self._data = ProjectInvariants()
            return
        if raw.get("type") == "agent_invariants" or "architecture" in raw:
            self._data = _invariants_from_dict(raw)
            return
        self._data = ProjectInvariants()

    def persist(self) -> None:
        payload: dict[str, Any] = {
            "version": 1,
            "type": "agent_invariants",
            **_invariants_to_dict(self._data),
        }
        self._write_json(payload)

    @property
    def data(self) -> ProjectInvariants:
        return self._data

    def set_section(self, section: str, value: str) -> tuple[bool, str]:
        raw = (section or "").strip()
        if raw.lower().startswith("extra."):
            sub = raw[6:].strip()
            if not sub:
                return False, "Укажите ключ: extra.<имя>"
            v = (value or "").strip()
            if v:
                self._data.extra[sub] = v
            else:
                self._data.extra.pop(sub, None)
            self.persist()
            return True, ""

        key = canonical_section(raw)
        if not key:
            allowed = "architecture | technical_decisions | stack | business_rules | extra.<имя>"
            return False, f"Неизвестный раздел. Допустимо: {allowed}"
        v = (value or "").strip()
        setattr(self._data, key, v)
        self.persist()
        return True, ""

    def clear_section(self, section: str) -> tuple[bool, str]:
        """section: имя раздела или 'all' для всех стандартных + extra."""
        raw = (section or "").strip()
        s = raw.lower()
        if s in ("all", "*", "все"):
            self._data = ProjectInvariants()
            self.persist()
            return True, ""
        if s.startswith("extra."):
            sub = raw[6:].strip()
            if not sub:
                return False, "Укажите ключ: extra.<имя>"
            self._data.extra.pop(sub, None)
            self.persist()
            return True, ""

        key = canonical_section(raw)
        if not key:
            return False, "Укажите раздел или all."
        setattr(self._data, key, "")
        self.persist()
        return True, ""

    def format_lines(self) -> str:
        d = self._data
        lines: list[str] = [
            f"Файл инвариантов: {self._path}",
            "(хранится отдельно от истории диалога)",
            "",
        ]
        for k in SECTION_KEYS:
            title = section_title(k)
            val = getattr(d, k, "") or ""
            lines.append(f"{title}:")
            lines.append(f"  {val if val else '(пусто)'}")
            lines.append("")
        if d.extra:
            lines.append("Дополнительные разделы (extra):")
            for ek, ev in sorted(d.extra.items()):
                lines.append(f"  {ek}: {ev}")
        else:
            lines.append("Дополнительные разделы (extra): (пусто)")
        lines.append("")
        lines.append("Редактирование: /invariants set <раздел> <текст>")
        return "\n".join(lines).rstrip()

    def build_system_messages(self) -> list[dict[str, str]]:
        """Системный блок с инвариантами; пустой набор — пустой список."""
        d = self._data
        blocks: list[str] = []
        for k in SECTION_KEYS:
            val = (getattr(d, k, "") or "").strip()
            if val:
                block = f"### {section_title(k)}\n{val}"
                if k == "stack":
                    block += (
                        "\n\n[Код и инструменты] Если здесь указан язык программирования, рантайм, "
                        "SDK или стек — любые примеры кода и команды в ответах должны строго "
                        "соответствовать этому. Не подставляй «по умолчанию» Kotlin, Swift, Go, "
                        "Python, JavaScript и т.п., если инвариант требует иное (например только Java). "
                        "Перед выдачей кода проверь соответствие разделу «Ограничения по стеку»."
                    )
                blocks.append(block)
        for ek, ev in sorted(d.extra.items()):
            ev = (ev or "").strip()
            if ev:
                blocks.append(f"### {ek}\n{ev}")
        if not blocks:
            return []

        intro = (
            "НЕИЗМЕНЯЕМЫЕ ИНВАРИАНТЫ ПРОЕКТА (хранятся вне диалога; не подменяются "
            "фактами из переписки и не ослабляются без явного изменения файла инвариантов).\n\n"
            "Ты обязан:\n"
            "1) Перед ответом и в ходе рассуждения явно учитывать каждый непустой блок ниже.\n"
            "2) В рассуждении кратко связывать выводы с релевантными инвариантами (что именно ограничивает решение).\n"
            "3) Не предлагать архитектуру, стек, технические решения или бизнес-логику, противоречащие инвариантам.\n"
            "4) Если запрос пользователя требует нарушения инварианта — откажись от такого предложения, объясни "
            "конфликт и предложи допустимую альтернативу в рамках инвариантов; не выдумывай изменение инвариантов сам.\n"
            "5) Примеры кода: язык и экосистема определяются инвариантами (часто раздел «Ограничения по стеку»). "
            "Не выбирай язык по привычке модели; при несоответствии перепиши ответ.\n"
        )
        body = "\n\n".join(blocks)
        content = intro + "\n" + body
        return [{"role": "system", "content": content}]
