"""
Модель памяти ассистента с тремя типами:
- краткосрочная (short_term): текущий диалог и временные заметки
- рабочая (working): данные текущей задачи (ключ-значение)
- долговременная (long_term): профиль, решения, знания

Типы памяти хранятся раздельно в разных файлах JSON.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_AGENT_DIR = Path(__file__).resolve().parent

SHORT_TERM_FILE = "short_term_memory.json"
WORKING_FILE = "working_memory.json"
LONG_TERM_FILE = "long_term_memory.json"
SHORT_TERM_NOTE_TTL_TURNS = 5


@dataclass
class ShortTermMemory:
    dialog: list[dict[str, str]] = field(default_factory=list)
    notes: dict[str, str] = field(default_factory=dict)
    note_ttl: dict[str, int] = field(default_factory=dict)


@dataclass
class WorkingMemory:
    data: dict[str, str] = field(default_factory=dict)


@dataclass
class LongTermMemory:
    profile: dict[str, str] = field(default_factory=dict)
    decisions: dict[str, str] = field(default_factory=dict)
    knowledge: dict[str, str] = field(default_factory=dict)


class AssistantMemoryStore:
    def __init__(self, memory_dir: Path | None = None) -> None:
        self._dir = memory_dir or self._resolve_memory_dir()
        self._short_path = self._dir / SHORT_TERM_FILE
        self._working_path = self._dir / WORKING_FILE
        self._long_path = self._dir / LONG_TERM_FILE

        self.short_term = self._load_short_term()
        self.working = self._load_working()
        self.long_term = self._load_long_term()

    @staticmethod
    def _resolve_memory_dir() -> Path:
        raw = (os.environ.get("LLM_AGENT_MEMORY_DIR") or "").strip()
        if raw:
            return Path(raw).expanduser().resolve()
        return (_AGENT_DIR / "memory").resolve()

    @staticmethod
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
            out.append({"role": str(role), "content": "" if content is None else str(content)})
        return out

    @staticmethod
    def _normalize_str_map(raw: Any) -> dict[str, str]:
        out: dict[str, str] = {}
        if not isinstance(raw, dict):
            return out
        for k, v in raw.items():
            key = str(k).strip() if k is not None else ""
            if not key:
                continue
            out[key] = "" if v is None else str(v).strip()
        return out

    def _read_json(self, path: Path) -> Any:
        if not path.is_file():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8-sig"))
        except (OSError, json.JSONDecodeError):
            return None

    def _write_json(self, path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(
            json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
            encoding="utf-8",
        )
        tmp.replace(path)

    def _load_short_term(self) -> ShortTermMemory:
        data = self._read_json(self._short_path)
        if not isinstance(data, dict):
            return ShortTermMemory()
        notes = self._normalize_str_map(data.get("notes"))
        raw_ttl = data.get("note_ttl")
        ttl: dict[str, int] = {}
        if isinstance(raw_ttl, dict):
            for k, v in raw_ttl.items():
                key = str(k).strip() if k is not None else ""
                if not key:
                    continue
                try:
                    ttl[key] = max(0, int(v))
                except (TypeError, ValueError):
                    ttl[key] = SHORT_TERM_NOTE_TTL_TURNS
        # Обратная совместимость: старые заметки без TTL получают дефолтный срок жизни.
        for key in notes.keys():
            ttl.setdefault(key, SHORT_TERM_NOTE_TTL_TURNS)
        return ShortTermMemory(
            dialog=self._normalize_chat_messages(data.get("dialog")),
            notes=notes,
            note_ttl=ttl,
        )

    def _load_working(self) -> WorkingMemory:
        data = self._read_json(self._working_path)
        if not isinstance(data, dict):
            return WorkingMemory()
        return WorkingMemory(data=self._normalize_str_map(data.get("data")))

    def _load_long_term(self) -> LongTermMemory:
        data = self._read_json(self._long_path)
        if not isinstance(data, dict):
            return LongTermMemory()
        return LongTermMemory(
            profile=self._normalize_str_map(data.get("profile")),
            decisions=self._normalize_str_map(data.get("decisions")),
            knowledge=self._normalize_str_map(data.get("knowledge")),
        )

    def persist_all(self) -> None:
        self._write_json(
            self._short_path,
            {
                "version": 1,
                "type": "short_term",
                "dialog": self.short_term.dialog,
                "notes": self.short_term.notes,
                "note_ttl": self.short_term.note_ttl,
            },
        )
        self._write_json(
            self._working_path,
            {
                "version": 1,
                "type": "working",
                "data": self.working.data,
            },
        )
        self._write_json(
            self._long_path,
            {
                "version": 1,
                "type": "long_term",
                "profile": self.long_term.profile,
                "decisions": self.long_term.decisions,
                "knowledge": self.long_term.knowledge,
            },
        )

    def update_short_term_dialog(
        self, messages: list[dict[str, str]], max_messages: int = 20, *, decay_notes: bool = True
    ) -> None:
        normalized = self._normalize_chat_messages(messages)
        if len(normalized) > max_messages:
            normalized = normalized[-max_messages:]
        self.short_term.dialog = normalized
        if decay_notes:
            self._decay_short_notes()
        self.persist_all()

    def _decay_short_notes(self) -> None:
        """Уменьшает TTL краткосрочных заметок на 1 ход и удаляет просроченные."""
        to_delete: list[str] = []
        for key in list(self.short_term.notes.keys()):
            ttl = int(self.short_term.note_ttl.get(key, SHORT_TERM_NOTE_TTL_TURNS))
            ttl -= 1
            if ttl <= 0:
                to_delete.append(key)
            else:
                self.short_term.note_ttl[key] = ttl
        for key in to_delete:
            self.short_term.notes.pop(key, None)
            self.short_term.note_ttl.pop(key, None)

    def put_short_note(self, key: str, value: str) -> tuple[bool, str]:
        k = (key or "").strip()
        if not k:
            return False, "Пустой ключ short_term note."
        self.short_term.notes[k] = (value or "").strip()
        self.short_term.note_ttl[k] = SHORT_TERM_NOTE_TTL_TURNS
        self.persist_all()
        return True, ""

    def put_working(self, key: str, value: str) -> tuple[bool, str]:
        k = (key or "").strip()
        if not k:
            return False, "Пустой ключ working memory."
        self.working.data[k] = (value or "").strip()
        self.persist_all()
        return True, ""

    def put_long_term(self, section: str, key: str, value: str) -> tuple[bool, str]:
        sec = (section or "").strip().lower()
        if sec not in ("profile", "decisions", "knowledge"):
            return (
                False,
                "Секция long_term должна быть одной из: profile, decisions, knowledge.",
            )
        k = (key or "").strip()
        if not k:
            return False, "Пустой ключ long_term memory."
        target = getattr(self.long_term, sec)
        target[k] = (value or "").strip()
        self.persist_all()
        return True, ""

    def format_summary(self) -> str:
        def _fmt_map(title: str, m: dict[str, str]) -> list[str]:
            if not m:
                return [f"{title}: (пусто)"]
            lines = [f"{title}:"]
            for k, v in sorted(m.items()):
                lines.append(f"  - {k}: {v}")
            return lines

        lines: list[str] = []
        lines.append(f"short_term.dialog: {len(self.short_term.dialog)} сообщений")
        lines.extend(_fmt_map("short_term.notes", self.short_term.notes))
        lines.extend(_fmt_map("working", self.working.data))
        lines.extend(_fmt_map("long_term.profile", self.long_term.profile))
        lines.extend(_fmt_map("long_term.decisions", self.long_term.decisions))
        lines.extend(_fmt_map("long_term.knowledge", self.long_term.knowledge))
        return "\n".join(lines)

    def build_memory_system_messages(self) -> list[dict[str, str]]:
        blocks: list[str] = []

        if self.short_term.notes:
            lines = [f"- {k}: {v}" for k, v in sorted(self.short_term.notes.items())]
            blocks.append(
                "Краткосрочная память (актуальные заметки текущего диалога):\n"
                + "\n".join(lines)
            )

        if self.working.data:
            lines = [f"- {k}: {v}" for k, v in sorted(self.working.data.items())]
            blocks.append("Рабочая память (текущая задача):\n" + "\n".join(lines))

        lt_lines: list[str] = []
        for sec, data in (
            ("profile", self.long_term.profile),
            ("decisions", self.long_term.decisions),
            ("knowledge", self.long_term.knowledge),
        ):
            if not data:
                continue
            lt_lines.append(f"{sec}:")
            lt_lines.extend([f"- {k}: {v}" for k, v in sorted(data.items())])
        if lt_lines:
            blocks.append("Долговременная память:\n" + "\n".join(lt_lines))

        if not blocks:
            return []
        guidance = (
            "Используй эти блоки памяти как источник фактов. "
            "Если ответ уже есть в памяти, опирайся на него в первую очередь и не говори, "
            "что у тебя нет доступа к данным."
        )
        return [{"role": "system", "content": guidance + "\n\n" + "\n\n".join(blocks)}]
