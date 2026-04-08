"""
Персонализация поверх памяти: профиль пользователя и предпочтения (стиль, формат, ограничения).
Хранится отдельно от long_term.profile — это явные инструкции к ответу, а не факты из диалога.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_AGENT_DIR = Path(__file__).resolve().parent
USER_PROFILE_FILENAME = "user_profile.json"

_PROFILE_SPLIT_RE = re.compile(
    r"\s+\b(display_name|style|format|constraints)\s+",
    re.IGNORECASE,
)


def _canonical_profile_field(name: str) -> str | None:
    """Имя поля из CLI → канонический ключ для set_field / внутреннего применения."""
    key = (name or "").strip().lower().replace("-", "_")
    allowed = {
        "display_name": "display_name",
        "name": "display_name",
        "style": "style",
        "format": "format",
        "response_format": "format",
        "constraints": "constraints",
    }
    return allowed.get(key)


def parse_profile_set_rest(first_field: str, rest: str) -> tuple[dict[str, str], str | None]:
    """
    Разбирает хвост команды `/profile set <поле> <текст>`.

    Если в тексте встречаются отдельные слова display_name / style / format / constraints,
    строка делится на несколько пар поле→значение. Иначе весь текст — значение первого поля.

    Возвращает (словарь {каноническое_имя: значение}, ошибка_или_None).
    """
    first_raw = (first_field or "").strip()
    if not first_raw:
        return {}, "Укажите поле профиля."
    if first_raw.lower().startswith("extra."):
        return {first_raw: (rest or "").strip()}, None

    first_canon = _canonical_profile_field(first_raw)
    if not first_canon:
        return {}, f"Неизвестное поле: {first_raw}"

    r = rest or ""
    if not r.strip():
        return {first_canon: ""}, None

    parts = _PROFILE_SPLIT_RE.split(r)
    out: dict[str, str] = {first_canon: parts[0].strip()}

    i = 1
    while i + 1 < len(parts):
        key_raw = parts[i]
        val = parts[i + 1].strip()
        key_canon = _canonical_profile_field(key_raw)
        if key_canon:
            out[key_canon] = val
        i += 2

    return out, None


@dataclass
class UserProfile:
    """Поля профиля; пустые строки не попадают в системное сообщение."""

    display_name: str = ""
    style: str = ""
    response_format: str = ""
    constraints: str = ""
    extra: dict[str, str] = field(default_factory=dict)


def _profile_from_dict(data: Any) -> UserProfile:
    if not isinstance(data, dict):
        return UserProfile()
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

    return UserProfile(
        display_name=s("display_name"),
        style=s("style"),
        response_format=s("format"),
        constraints=s("constraints"),
        extra=extra,
    )


def _profile_to_dict(p: UserProfile) -> dict[str, Any]:
    return {
        "display_name": p.display_name,
        "style": p.style,
        "format": p.response_format,
        "constraints": p.constraints,
        "extra": dict(sorted(p.extra.items())),
    }


def _format_wants_json(p: UserProfile) -> bool:
    blob = f"{p.response_format} {p.constraints} {' '.join(p.extra.values())}".lower()
    return "json" in blob


class UserProfileStore:
    """Несколько именованных профилей; активный используется в запросах к API."""

    def __init__(self, memory_dir: Path | None = None) -> None:
        self._dir = memory_dir or self._resolve_memory_dir()
        self._path = self._dir / USER_PROFILE_FILENAME
        self._profiles: dict[str, UserProfile] = {}
        self._active_id: str = "default"
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
        data = self._read_json()
        if not isinstance(data, dict):
            self._profiles = {"default": UserProfile()}
            self._active_id = "default"
            return

        if data.get("version") == 2 and isinstance(data.get("profiles"), dict):
            raw_profiles = data["profiles"]
            self._profiles = {}
            for pid, pd in raw_profiles.items():
                key = str(pid).strip()
                if not key:
                    continue
                self._profiles[key] = _profile_from_dict(pd)
            if not self._profiles:
                self._profiles = {"default": UserProfile()}
            active = str(data.get("active") or "").strip()
            if active not in self._profiles:
                self._active_id = next(iter(self._profiles))
            else:
                self._active_id = active
            return

        # legacy v1: один плоский профиль
        self._profiles = {"default": _profile_from_dict(data)}
        self._active_id = "default"

    @property
    def profile(self) -> UserProfile:
        return self._profiles[self._active_id]

    @property
    def active_profile_id(self) -> str:
        return self._active_id

    def persist(self) -> None:
        payload: dict[str, Any] = {
            "version": 2,
            "type": "user_profile",
            "active": self._active_id,
            "profiles": {
                pid: _profile_to_dict(p) for pid, p in sorted(self._profiles.items())
            },
        }
        self._write_json(payload)

    def list_profile_ids(self) -> list[str]:
        return sorted(self._profiles.keys())

    def activate(self, profile_id: str) -> tuple[bool, str]:
        pid = (profile_id or "").strip()
        if not pid:
            return False, "Укажите имя профиля."
        if pid not in self._profiles:
            known = ", ".join(self.list_profile_ids())
            return False, f"Нет профиля «{pid}». Доступны: {known}"
        self._active_id = pid
        self.persist()
        return True, ""

    def create_profile(
        self,
        profile_id: str,
        *,
        template: UserProfile | None = None,
    ) -> tuple[bool, str]:
        pid = (profile_id or "").strip()
        if not pid:
            return False, "Укажите имя нового профиля."
        if pid in self._profiles:
            return False, f"Профиль «{pid}» уже есть."
        self._profiles[pid] = UserProfile(
            display_name=template.display_name if template else "",
            style=template.style if template else "",
            response_format=template.response_format if template else "",
            constraints=template.constraints if template else "",
            extra=dict(template.extra) if template else {},
        )
        self._active_id = pid
        self.persist()
        return True, ""

    def duplicate_profile(self, profile_id: str) -> tuple[bool, str]:
        """Копия текущего активного профиля под новым именем; новый профиль становится активным."""
        return self.create_profile(profile_id, template=self.profile)

    def delete_profile(self, profile_id: str) -> tuple[bool, str]:
        pid = (profile_id or "").strip()
        if not pid:
            return False, "Укажите имя профиля."
        if pid not in self._profiles:
            return False, f"Нет профиля «{pid}»."
        if len(self._profiles) <= 1:
            return False, "Нельзя удалить единственный профиль."
        del self._profiles[pid]
        if self._active_id == pid:
            self._active_id = next(iter(sorted(self._profiles.keys())))
        self.persist()
        return True, ""

    def _apply_field(self, name: str, value: str) -> tuple[bool, str]:
        """Записывает одно поле в профиль без сохранения на диск."""
        raw = (name or "").strip()
        if not raw:
            return False, "Укажите поле профиля."
        v = (value or "").strip()

        if raw.lower().startswith("extra."):
            sub = raw[6:].strip()
            if not sub:
                return False, "Для extra укажите ключ: extra.my_key"
            self.profile.extra[sub] = v
            if not v:
                self.profile.extra.pop(sub, None)
            return True, ""

        key = raw.lower().replace("-", "_")
        allowed = {
            "display_name": "display_name",
            "name": "display_name",
            "style": "style",
            "format": "response_format",
            "response_format": "response_format",
            "constraints": "constraints",
        }
        attr = allowed.get(key)
        if not attr:
            return (
                False,
                "Неизвестное поле. Допустимо: display_name, style, format, constraints, extra.<ключ>.",
            )
        setattr(self.profile, attr, v)
        return True, ""

    def set_field(self, name: str, value: str) -> tuple[bool, str]:
        """
        name: display_name | style | format | constraints | extra.<ключ>
        """
        ok, err = self._apply_field(name, value)
        if ok:
            self.persist()
        return ok, err

    def set_fields(self, assignments: dict[str, str]) -> tuple[bool, str]:
        """Несколько полей за один раз; одна запись на диск."""
        for name, value in assignments.items():
            ok, err = self._apply_field(name, value)
            if not ok:
                return False, err
        self.persist()
        return True, ""

    def format_list_lines(self) -> str:
        lines = ["Сохранённые профили (* — активен):"]
        for pid in self.list_profile_ids():
            if pid == self._active_id:
                lines.append(f"  * {pid}")
            else:
                lines.append(f"    {pid}")
        lines.append("Переключение: /profile use <имя>  |  новый пустой: /profile new <имя>")
        lines.append("Копия активного: /profile copy <имя>  |  удалить: /profile delete <имя>")
        return "\n".join(lines)

    def format_lines(self) -> str:
        p = self.profile
        lines: list[str] = [
            f"Активный профиль: «{self._active_id}»",
            f"Все профили: {', '.join(self.list_profile_ids())}",
            "(переключение: /profile use <имя>)",
            "",
        ]
        lines.append(f"Содержимое «{self._active_id}»:")
        if p.display_name:
            lines.append(f"  display_name: {p.display_name}")
        else:
            lines.append("  display_name: (не задано)")
        lines.append(f"  style: {p.style or '(не задано)'}")
        lines.append(f"  format: {p.response_format or '(не задано)'}")
        lines.append(f"  constraints: {p.constraints or '(не задано)'}")
        if p.extra:
            lines.append("  extra:")
            for k, v in sorted(p.extra.items()):
                lines.append(f"    {k}: {v}")
        else:
            lines.append("  extra: (пусто)")
        return "\n".join(lines)

    def build_system_messages(self) -> list[dict[str, str]]:
        """Одно системное сообщение с предпочтениями; пустой профиль — пустой список."""
        p = self.profile
        if not (
            p.display_name.strip()
            or p.style.strip()
            or p.response_format.strip()
            or p.constraints.strip()
            or p.extra
        ):
            return []

        parts: list[str] = []

        parts.append(
            f"Активный сценарий профиля (не смешивай с другими): «{self._active_id}»."
        )
        if p.display_name:
            parts.append(f"Обращайся к пользователю с учётом имени/роли: {p.display_name}.")
        if p.style:
            parts.append(f"Стиль общения: {p.style}")
        if p.response_format:
            parts.append(f"Формат ответов: {p.response_format}")
        if p.constraints:
            parts.append(f"Ограничения и табу: {p.constraints}")
        for k, v in sorted(p.extra.items()):
            if v:
                parts.append(f"{k}: {v}")

        if not _format_wants_json(p):
            parts.append(
                "Отвечай обычным связным текстом. Не дублируй ответ в JSON и не добавляй "
                "блоки ``` с разметкой/структурированными данными, если пользователь явно не "
                "просит машиночитаемый формат (JSON, код, таблицу для экспорта)."
            )

        intro = (
            "Профиль пользователя и предпочтения (применяй при каждом ответе только к этому сценарию; "
            "не противоречь им без веской причины):"
        )
        body = "\n".join(f"- {line}" for line in parts)
        return [{"role": "system", "content": intro + "\n" + body}]
