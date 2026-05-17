"""
Конфигурация приватного HTTP-сервиса локальной LLM.
Переменные окружения — см. .env.example в корне репозитория.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _env_int(name: str, default: int, *, min_v: int, max_v: int) -> int:
    raw = (os.environ.get(name) or "").strip()
    if not raw:
        return default
    v = int(raw)
    if v < min_v or v > max_v:
        raise ValueError(f"{name} должно быть от {min_v} до {max_v}, получено {v}")
    return v


def _env_float(name: str, default: float, *, min_v: float, max_v: float) -> float:
    raw = (os.environ.get(name) or "").strip()
    if not raw:
        return default
    v = float(raw)
    if v < min_v or v > max_v:
        raise ValueError(f"{name} должно быть от {min_v} до {max_v}, получено {v}")
    return v


def _env_bool(name: str, default: bool) -> bool:
    raw = (os.environ.get(name) or "").strip().lower()
    if not raw:
        return default
    return raw in ("1", "true", "yes", "on")


@dataclass(frozen=True)
class ServiceConfig:
    host: str = "0.0.0.0"
    port: int = 8080
    api_key: str = ""
    rate_limit_rpm: int = 30
    max_concurrent: int = 2
    max_context_messages: int = 50
    max_message_chars: int = 16_000
    session_ttl_sec: int = 3600
    max_sessions: int = 100
    history_dir: Path | None = None
    require_api_key: bool = False

    @classmethod
    def from_env(cls) -> ServiceConfig:
        api_key = (os.environ.get("LLM_SERVICE_API_KEY") or "").strip()
        require = _env_bool("LLM_SERVICE_REQUIRE_API_KEY", default=bool(api_key))
        hist_raw = (os.environ.get("LLM_SERVICE_HISTORY_DIR") or "").strip()
        history_dir = Path(hist_raw).expanduser().resolve() if hist_raw else None
        return cls(
            host=(os.environ.get("LLM_SERVICE_HOST") or "0.0.0.0").strip(),
            port=_env_int("LLM_SERVICE_PORT", 8080, min_v=1, max_v=65535),
            api_key=api_key,
            rate_limit_rpm=_env_int(
                "LLM_SERVICE_RATE_LIMIT_RPM", 30, min_v=1, max_v=10_000
            ),
            max_concurrent=_env_int(
                "LLM_SERVICE_MAX_CONCURRENT", 2, min_v=1, max_v=64
            ),
            max_context_messages=_env_int(
                "LLM_SERVICE_MAX_CONTEXT_MESSAGES", 50, min_v=1, max_v=500
            ),
            max_message_chars=_env_int(
                "LLM_SERVICE_MAX_MESSAGE_CHARS", 16_000, min_v=256, max_v=500_000
            ),
            session_ttl_sec=_env_int(
                "LLM_SERVICE_SESSION_TTL_SEC", 3600, min_v=60, max_v=86_400
            ),
            max_sessions=_env_int(
                "LLM_SERVICE_MAX_SESSIONS", 100, min_v=1, max_v=10_000
            ),
            history_dir=history_dir,
            require_api_key=require,
        )
