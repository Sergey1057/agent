"""
LLM-агент: инкапсулирует запрос к API и разбор ответа.
Интерфейс (CLI, web) только вызывает агента, не строит HTTP вручную.
"""

from __future__ import annotations

import json
import os
import ssl
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import certifi

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None  # type: ignore[misc, assignment]

_AGENT_DIR = Path(__file__).resolve().parent


def _read_env_file_as_text(path: Path) -> str:
    """UTF-8 / UTF-16 (иногда так сохраняет Блокнот) и пустой файл."""
    try:
        raw = path.read_bytes()
    except OSError:
        return ""
    if not raw:
        return ""
    if raw.startswith((b"\xff\xfe", b"\xfe\xff")):
        return raw.decode("utf-16")
    return raw.decode("utf-8-sig")


def _parse_simple_env_file(path: Path) -> dict[str, str]:
    """Парсинг .env: export KEY=, BOM, UTF-16."""
    if not path.is_file():
        return {}
    text = _read_env_file_as_text(path)
    out: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if line.lower().startswith("export "):
            line = line[7:].lstrip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        k, _, v = line.partition("=")
        k, v = k.strip(), v.strip()
        if len(v) >= 2 and v[0] == v[-1] and v[0] in "\"'":
            v = v[1:-1]
        if k:
            out[k] = v
    return out


def _env_file_hints(primary: Path) -> str:
    """Подсказки, если .env есть, а ключа нет."""
    hints: list[str] = []
    if not primary.is_file():
        return ""
    try:
        size = primary.stat().st_size
    except OSError:
        return ""
    if size == 0:
        hints.append(
            "Файл .env на диске пустой (0 байт) — сохраните вкладку в редакторе (Cmd+S), затем снова запустите cli."
        )
    parsed = _parse_simple_env_file(primary)
    if size > 0 and "GROQ_API_KEY" not in parsed and "OPENAI_API_KEY" not in parsed:
        hints.append(
            "В .env не найдена строка GROQ_API_KEY=... Проверьте имя переменной, без пробелов до/после =."
        )
    if hints:
        return "\n  " + "\n  ".join(hints)
    return ""


def _dotenv_paths() -> list[Path]:
    """Рядом с agent.py и (если отличается) в текущей рабочей директории."""
    paths: list[Path] = []
    seen: set[Path] = set()
    for p in (_AGENT_DIR / ".env", Path.cwd() / ".env"):
        try:
            r = p.resolve()
        except OSError:
            r = p
        if r not in seen:
            seen.add(r)
            paths.append(p)
    return paths


def _load_project_dotenv() -> None:
    # Пустая строка в окружении не должна перекрывать .env (поведение python-dotenv по умолчанию).
    for var in ("GROQ_API_KEY", "OPENAI_API_KEY"):
        if not (os.environ.get(var) or "").strip():
            os.environ.pop(var, None)
    for path in _dotenv_paths():
        if not path.is_file():
            continue
        if load_dotenv is not None:
            load_dotenv(path, encoding="utf-8-sig")
        for k, v in _parse_simple_env_file(path).items():
            if v and not (os.environ.get(k) or "").strip():
                os.environ[k] = v


# Как в AiAdvent1: Groq OpenAI-совместимый endpoint и та же модель по умолчанию.
GROQ_CHAT_COMPLETIONS_BASE = "https://api.groq.com/openai/v1"
DEFAULT_GROQ_MODEL = "llama-3.3-70b-versatile"

# Groq за Cloudflare: дефолтный User-Agent urllib (Python-urllib/…) часто даёт 403 + error 1010.
_DEFAULT_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
)


def _ssl_context_for_url(url: str) -> ssl.SSLContext | None:
    """На macOS у многих сборок Python нет своих CA; certifi даёт актуальный bundle."""
    if not url.startswith("https:"):
        return None
    return ssl.create_default_context(cafile=certifi.where())


@dataclass
class AgentConfig:
    """Настройки Groq (или другого OpenAI-совместимого API при смене base_url)."""

    base_url: str = GROQ_CHAT_COMPLETIONS_BASE
    api_key: str | None = None
    model: str | None = DEFAULT_GROQ_MODEL
    timeout_sec: float = 120.0

    @classmethod
    def from_env(cls) -> AgentConfig:
        _load_project_dotenv()
        base = os.environ.get("OPENAI_BASE_URL", GROQ_CHAT_COMPLETIONS_BASE).rstrip("/")
        if not base.endswith("/v1"):
            base = base.rstrip("/") + "/v1"
        # Ключ: .env (локально) или export GROQ_API_KEY=... / как в Android local.properties
        key = (os.environ.get("GROQ_API_KEY") or os.environ.get("OPENAI_API_KEY") or "").strip()
        model = (
            os.environ.get("GROQ_MODEL")
            or os.environ.get("OPENAI_MODEL")
            or DEFAULT_GROQ_MODEL
        ).strip()
        return cls(
            base_url=base,
            api_key=key or None,
            model=model or DEFAULT_GROQ_MODEL,
        )


class LLMAgent:
    """
    Сущность «агент»: принимает пользовательский запрос, вызывает LLM, возвращает текст ответа.
    """

    def __init__(self, config: AgentConfig | None = None) -> None:
        self._config = config or AgentConfig.from_env()
        self._resolved_model: str | None = self._config.model

    def _chat_completions_url(self) -> str:
        return f"{self._config.base_url}/chat/completions"

    def _effective_api_key(self) -> str:
        """Ключ из явного AgentConfig или заново из окружения/.env (после старта CLI)."""
        if self._config.api_key:
            return self._config.api_key.strip()
        _load_project_dotenv()
        return (
            os.environ.get("GROQ_API_KEY") or os.environ.get("OPENAI_API_KEY") or ""
        ).strip()

    def _headers(self) -> dict[str, str]:
        h = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": os.environ.get("HTTP_USER_AGENT", _DEFAULT_UA),
        }
        key = self._effective_api_key()
        if key:
            h["Authorization"] = f"Bearer {key}"
        return h

    def _ensure_model(self) -> str:
        if self._resolved_model:
            return self._resolved_model
        url = f"{self._config.base_url}/models"
        req = urllib.request.Request(url, headers=self._headers(), method="GET")
        ctx = _ssl_context_for_url(url)
        with urllib.request.urlopen(req, timeout=30, context=ctx) as resp:
            data: dict[str, Any] = json.loads(resp.read().decode())
        models = data.get("data") or []
        if not models:
            raise RuntimeError(
                "Список моделей пуст. Укажите GROQ_MODEL или OPENAI_MODEL в окружении."
            )
        self._resolved_model = str(models[0]["id"])
        return self._resolved_model

    def run(self, user_message: str) -> str:
        """
        Основной вход агента: один пользовательский запрос → текст ответа модели.
        """
        text = (user_message or "").strip()
        if not text:
            return ""

        if not self._effective_api_key():
            primary = _AGENT_DIR / ".env"
            extra = _env_file_hints(primary)
            size_s = "—"
            if primary.is_file():
                try:
                    size_s = str(primary.stat().st_size)
                except OSError:
                    pass
            return (
                "Добавьте GROQ_API_KEY: файл .env рядом с agent.py (см. .env.example) "
                "или export GROQ_API_KEY=...\n"
                f"  Ожидаемый путь: {primary} (есть на диске: {'да' if primary.is_file() else 'нет'})\n"
                f"  Размер файла: {size_s} байт\n"
                "  Ключи: https://console.groq.com/keys"
                f"{extra}"
            )

        model = self._ensure_model()
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": text}],
        }
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            self._chat_completions_url(),
            data=body,
            headers=self._headers(),
            method="POST",
        )
        try:
            ctx = _ssl_context_for_url(self._chat_completions_url())
            with urllib.request.urlopen(
                req, timeout=self._config.timeout_sec, context=ctx
            ) as resp:
                raw = resp.read().decode("utf-8")
        except urllib.error.HTTPError as e:
            err_body = e.read().decode("utf-8", errors="replace")
            detail = err_body
            try:
                err_json: dict[str, Any] = json.loads(err_body)
                detail = str(
                    (err_json.get("error") or {}).get("message") or err_body
                )
            except (json.JSONDecodeError, TypeError, AttributeError):
                pass
            return f"Ошибка HTTP {e.code}: {detail[:2000]}"
        except OSError as e:
            return f"Сетевая ошибка: {e}"

        try:
            response_json: dict[str, Any] = json.loads(raw)
            choice = (response_json.get("choices") or [{}])[0]
            msg = choice.get("message") or {}
            content = msg.get("content")
            if isinstance(content, str):
                return content.strip()
            if content is not None:
                return str(content).strip()
            return raw[:2000]
        except (json.JSONDecodeError, KeyError, IndexError, TypeError) as e:
            return f"Не удалось разобрать ответ API: {e}\nФрагмент: {raw[:500]}"
