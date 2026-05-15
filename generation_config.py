"""
Параметры генерации LLM: temperature, max_tokens, окно истории диалога (context window).
Читаются из env, CLI и команды /gen в чате.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from typing import Any


def _parse_optional_float(raw: str | None) -> float | None:
    s = (raw or "").strip()
    if not s:
        return None
    v = float(s)
    if v < 0.0 or v > 2.0:
        raise ValueError("temperature должна быть в диапазоне 0.0–2.0")
    return v


def _parse_optional_int(
    raw: str | None,
    *,
    name: str,
    min_v: int,
    max_v: int,
) -> int | None:
    s = (raw or "").strip()
    if not s:
        return None
    v = int(s)
    if v < min_v or v > max_v:
        raise ValueError(f"{name} должно быть от {min_v} до {max_v}")
    return v


def _parse_recent_window(raw: str | None, *, default: int) -> int:
    s = (raw or "").strip()
    if not s:
        return default
    v = int(s)
    if v < 1 or v > 500:
        raise ValueError("context window (число реплик) должно быть от 1 до 500")
    return v


@dataclass
class GenerationConfig:
    """Настройки запроса к chat/completions и размера истории в промпте."""

    temperature: float | None = None
    max_tokens: int | None = None
    recent_message_window: int = 6

    @classmethod
    def from_env(
        cls,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        recent_message_window: int | None = None,
    ) -> GenerationConfig:
        env_temp = _parse_optional_float(os.environ.get("LLM_AGENT_TEMPERATURE"))
        env_max = _parse_optional_int(
            os.environ.get("LLM_AGENT_MAX_TOKENS"),
            name="max_tokens",
            min_v=1,
            max_v=128_000,
        )
        default_recent = _parse_recent_window(
            os.environ.get("LLM_AGENT_RECENT_MESSAGES"),
            default=6,
        )
        env_recent = (
            _parse_recent_window(
                os.environ.get("LLM_AGENT_CONTEXT_WINDOW"),
                default=default_recent,
            )
            if (os.environ.get("LLM_AGENT_CONTEXT_WINDOW") or "").strip()
            else default_recent
        )
        rw = (
            int(recent_message_window)
            if recent_message_window is not None
            else env_recent
        )
        if rw < 1 or rw > 500:
            raise ValueError("context window должно быть от 1 до 500")
        return cls(
            temperature=temperature if temperature is not None else env_temp,
            max_tokens=max_tokens if max_tokens is not None else env_max,
            recent_message_window=rw,
        )

    def apply_to_payload(self, payload: dict[str, Any]) -> None:
        if self.temperature is not None:
            payload["temperature"] = self.temperature
        if self.max_tokens is not None:
            payload["max_tokens"] = self.max_tokens

    def format_status_line(self) -> str:
        t = (
            f"{self.temperature:g}"
            if self.temperature is not None
            else "(сервер по умолчанию)"
        )
        m = (
            str(self.max_tokens)
            if self.max_tokens is not None
            else "(сервер по умолчанию)"
        )
        return (
            f"Генерация: temperature={t} | max_tokens={m} | "
            f"context_window={self.recent_message_window} реплик"
        )

    def format_help_block(self) -> str:
        return (
            "  /gen — текущие параметры\n"
            "  /gen temperature <0.0–2.0> | /gen temp <…>\n"
            "  /gen max-tokens <N> | /gen max_tokens <N>\n"
            "  /gen context-window <N> | /gen context <N> | /gen recent <N> — реплик в истории (1–500)\n"
            "  /gen reset — сброс к значениям при старте (env / флаги CLI)\n"
            "Переменные: LLM_AGENT_TEMPERATURE, LLM_AGENT_MAX_TOKENS, "
            "LLM_AGENT_CONTEXT_WINDOW (или LLM_AGENT_RECENT_MESSAGES)\n"
            "CLI: --temperature, --max-tokens, --context-window"
        )


def resolve_message_window(message_window: int | None, *, fallback: int) -> int:
    if message_window is None:
        return fallback
    return max(1, min(int(message_window), 500))


def handle_gen_slash_command(
    agent: Any,
    arg: str,
) -> str:
    """Команда /gen для интерактивного чата."""
    a = (arg or "").strip()
    if not a or a.lower() in ("show", "status"):
        return agent.generation_status_line()

    parts = a.split(maxsplit=1)
    sub = parts[0].lower().replace("_", "-")
    rest = parts[1].strip() if len(parts) > 1 else ""

    if sub == "reset":
        agent.reset_generation_config()
        return f"Сброшено. {agent.generation_status_line()}"

    if sub in ("temperature", "temp"):
        if not rest:
            return "Формат: /gen temperature <0.0–2.0>"
        try:
            agent.set_temperature(float(rest))
        except ValueError as e:
            return str(e)
        return agent.generation_status_line()

    if sub in ("max-tokens", "maxtokens"):
        if not rest:
            return "Формат: /gen max-tokens <N>"
        try:
            agent.set_max_tokens(int(rest))
        except ValueError as e:
            return str(e)
        return agent.generation_status_line()

    if sub in ("context-window", "context", "recent", "window"):
        if not rest:
            return "Формат: /gen context-window <N>  (число последних реплик user/assistant)"
        try:
            agent.set_context_window(int(rest))
        except ValueError as e:
            return str(e)
        return agent.generation_status_line()

    return (
        "Неизвестная подкоманда /gen.\n"
        + agent.generation_config.format_help_block()
    )
