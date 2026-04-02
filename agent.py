"""
LLM-агент: инкапсулирует запрос к GigaChat API и разбор ответа.
Интерфейс (CLI, web) только вызывает агента, не строит HTTP вручную.

Документация: https://developers.sber.ru/docs/ru/gigachat/quickstart/ind-using-api
"""

from __future__ import annotations

import json
import os
import ssl
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import certifi

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None  # type: ignore[misc, assignment]

_AGENT_DIR = Path(__file__).resolve().parent

DEFAULT_HISTORY_FILENAME = "chat_history.json"
DEFAULT_HISTORY_MAX_MESSAGES = 200

# GigaChat: OAuth и REST (см. документацию Сбера)
GIGACHAT_OAUTH_URL = "https://ngw.devices.sberbank.ru:9443/api/v2/oauth"
GIGACHAT_API_V1_BASE = "https://gigachat.devices.sberbank.ru/api/v1"
DEFAULT_GIGACHAT_MODEL = "GigaChat"
DEFAULT_GIGACHAT_SCOPE = "GIGACHAT_API_PERS"

# Дефолтный User-Agent для HTTP-клиента
_DEFAULT_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
)


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
    if size > 0 and "GIGACHAT_API_KEY" not in parsed:
        hints.append(
            "В .env не найдена строка GIGACHAT_API_KEY=... Проверьте имя переменной, без пробелов до/после =."
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
    # Пустая строка в окружении не должна перекрывать .env
    for var in ("GIGACHAT_API_KEY",):
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


def _env_flag_false(name: str, default: str = "1") -> bool:
    """True, если переменная явно выключена (0 / false / no / off)."""
    v = (os.environ.get(name) or default).strip().lower()
    return v in ("0", "false", "no", "off")


def _ssl_verify_disabled() -> bool:
    """Явное отключение проверки сертификата (только для отладки / изолированных сетей)."""
    _load_project_dotenv()
    v = (os.environ.get("GIGACHAT_SSL_VERIFY") or "").strip().lower()
    return v in ("0", "false", "no", "off")


def _ssl_ca_bundle_path() -> str | None:
    """Путь к PEM с доверенными корнями (цепочка Сбера/Минцифры, корпоративный CA и т.д.)."""
    _load_project_dotenv()
    for key in ("GIGACHAT_CA_BUNDLE", "SSL_CERT_FILE", "REQUESTS_CA_BUNDLE"):
        raw = (os.environ.get(key) or "").strip()
        if not raw:
            continue
        p = Path(raw).expanduser()
        if p.is_file():
            return str(p.resolve())
    return None


def _ssl_context_for_url(url: str) -> ssl.SSLContext | None:
    """
    HTTPS: по умолчанию truststore (системное хранилище ОС), иначе certifi.
    GIGACHAT_CA_BUNDLE — свой PEM; GIGACHAT_SSL_VERIFY=0 — без проверки (небезопасно).
    """
    if not url.startswith("https:"):
        return None
    _load_project_dotenv()
    if _ssl_verify_disabled():
        return ssl._create_unverified_context()
    ca = _ssl_ca_bundle_path()
    if ca:
        return ssl.create_default_context(cafile=ca)
    if not _env_flag_false("GIGACHAT_USE_TRUSTSTORE", default="1"):
        try:
            import truststore

            return truststore.SSLContext()
        except ImportError:
            pass
    return ssl.create_default_context(cafile=certifi.where())


def _history_path_resolved() -> Path:
    """Путь к JSON с историей: LLM_AGENT_HISTORY_FILE или chat_history.json рядом с agent.py."""
    _load_project_dotenv()
    raw = (os.environ.get("LLM_AGENT_HISTORY_FILE") or "").strip()
    if raw:
        return Path(raw).expanduser().resolve()
    return (_AGENT_DIR / DEFAULT_HISTORY_FILENAME).resolve()


def _history_max_from_env() -> int:
    _load_project_dotenv()
    raw = (os.environ.get("LLM_AGENT_HISTORY_MAX_MESSAGES") or "").strip()
    if not raw:
        return DEFAULT_HISTORY_MAX_MESSAGES
    try:
        n = int(raw)
        return max(2, min(n, 10_000))
    except ValueError:
        return DEFAULT_HISTORY_MAX_MESSAGES


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


def load_chat_history_file(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    try:
        text = path.read_text(encoding="utf-8-sig")
        data = json.loads(text)
    except (OSError, json.JSONDecodeError):
        return []
    if isinstance(data, dict):
        raw_list = data.get("messages")
    elif isinstance(data, list):
        raw_list = data
    else:
        return []
    return _normalize_chat_messages(raw_list)


def save_chat_history_file(path: Path, messages: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"version": 1, "messages": messages}
    tmp = path.with_name(path.name + ".tmp")
    data = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    tmp.write_text(data, encoding="utf-8")
    tmp.replace(path)


def clear_history_file() -> None:
    """Удаляет файл истории (путь как у LLMAgent)."""
    p = _history_path_resolved()
    try:
        if p.is_file():
            p.unlink()
    except OSError:
        pass


def _ssl_troubleshooting_hint(exc: BaseException) -> str:
    s = f"{exc!s} {getattr(exc, 'reason', '')!s}".lower()
    if "certificate_verify_failed" not in s and "certificate verify failed" not in s:
        return ""
    return (
        "\n  SSL: укажите в .env один из вариантов:\n"
        "  — GIGACHAT_USE_TRUSTSTORE=1 (по умолчанию) и pip install truststore — доверие как в браузере;\n"
        "  — GIGACHAT_CA_BUNDLE=/путь/к/ca-bundle.pem — свой PEM (корни НУЦ/Минцифры, корп. CA);\n"
        "  — GIGACHAT_SSL_VERIFY=0 — только для проверки, без проверки сертификата (небезопасно)."
    )


# Кэш access token: (токен, expires_at_ms по ответу OAuth)
_gigachat_token_cache: tuple[str, int] | None = None


def _fetch_gigachat_access_token(
    authorization_key: str,
    oauth_url: str,
    scope: str,
    timeout_sec: float,
) -> tuple[str, int]:
    """
    POST /api/v2/oauth: Basic + RqUID (uuid4), scope в теле form-urlencoded.
    Возвращает (access_token, expires_at_ms).
    """
    body = urllib.parse.urlencode({"scope": scope}).encode("utf-8")
    headers = {
        "Content-Type": "application/x-www-form-urlencoded",
        "Accept": "application/json",
        "RqUID": str(uuid.uuid4()),
        "Authorization": f"Basic {authorization_key.strip()}",
        "User-Agent": os.environ.get("HTTP_USER_AGENT", _DEFAULT_UA),
    }
    req = urllib.request.Request(oauth_url, data=body, headers=headers, method="POST")
    ctx = _ssl_context_for_url(oauth_url)
    with urllib.request.urlopen(req, timeout=timeout_sec, context=ctx) as resp:
        raw = resp.read().decode("utf-8")
    data: dict[str, Any] = json.loads(raw)
    token = data.get("access_token")
    if not isinstance(token, str) or not token:
        raise RuntimeError(f"В ответе OAuth нет access_token: {raw[:500]}")
    expires_at = data.get("expires_at")
    now_ms = int(time.time() * 1000)
    if isinstance(expires_at, (int, float)):
        expires_ms = int(expires_at)
    else:
        # Документация: токен ~30 мин; если поля нет — запас 25 мин
        expires_ms = now_ms + 25 * 60 * 1000
    return token, expires_ms


def _get_gigachat_access_token(
    authorization_key: str,
    oauth_url: str,
    scope: str,
    timeout_sec: float,
) -> str:
    """Access token с кэшем (обновление до истечения с запасом ~60 с)."""
    global _gigachat_token_cache
    now_ms = int(time.time() * 1000)
    if _gigachat_token_cache is not None:
        tok, exp_ms = _gigachat_token_cache
        if now_ms < exp_ms - 60_000:
            return tok
    tok, exp_ms = _fetch_gigachat_access_token(
        authorization_key, oauth_url, scope, timeout_sec
    )
    _gigachat_token_cache = (tok, exp_ms)
    return tok


@dataclass
class AgentConfig:
    """Настройки GigaChat API."""

    oauth_url: str = GIGACHAT_OAUTH_URL
    api_base: str = GIGACHAT_API_V1_BASE
    authorization_key: str | None = None
    scope: str = DEFAULT_GIGACHAT_SCOPE
    model: str | None = DEFAULT_GIGACHAT_MODEL
    timeout_sec: float = 120.0

    @classmethod
    def from_env(cls) -> AgentConfig:
        _load_project_dotenv()
        key = (os.environ.get("GIGACHAT_API_KEY") or "").strip()
        model = (os.environ.get("GIGACHAT_MODEL") or DEFAULT_GIGACHAT_MODEL).strip()
        scope = (os.environ.get("GIGACHAT_SCOPE") or DEFAULT_GIGACHAT_SCOPE).strip()
        oauth = (os.environ.get("GIGACHAT_OAUTH_URL") or GIGACHAT_OAUTH_URL).strip()
        base = (os.environ.get("GIGACHAT_API_BASE") or GIGACHAT_API_V1_BASE).rstrip("/")
        return cls(
            oauth_url=oauth,
            api_base=base,
            authorization_key=key or None,
            scope=scope or DEFAULT_GIGACHAT_SCOPE,
            model=model or DEFAULT_GIGACHAT_MODEL,
        )


class LLMAgent:
    """
    Сущность «агент»: принимает пользовательский запрос, вызывает LLM, возвращает текст ответа.
    История диалога хранится в JSON и подгружается при старте (см. load_chat_history_file).
    """

    def __init__(self, config: AgentConfig | None = None) -> None:
        self._config = config or AgentConfig.from_env()
        self._resolved_model: str | None = self._config.model
        self._history_path = _history_path_resolved()
        self._history_max_messages = _history_max_from_env()
        self._messages: list[dict[str, str]] = load_chat_history_file(self._history_path)

    def _trim_messages(self) -> None:
        if len(self._messages) <= self._history_max_messages:
            return
        del self._messages[: len(self._messages) - self._history_max_messages]

    def _persist_history(self) -> None:
        try:
            save_chat_history_file(self._history_path, self._messages)
        except OSError:
            pass

    def _chat_completions_url(self) -> str:
        return f"{self._config.api_base}/chat/completions"

    def _models_url(self) -> str:
        return f"{self._config.api_base}/models"

    def _effective_authorization_key(self) -> str:
        if self._config.authorization_key:
            return self._config.authorization_key.strip()
        _load_project_dotenv()
        return (os.environ.get("GIGACHAT_API_KEY") or "").strip()

    def _bearer_headers(self, access_token: str) -> dict[str, str]:
        return {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": f"Bearer {access_token}",
            "User-Agent": os.environ.get("HTTP_USER_AGENT", _DEFAULT_UA),
        }

    def _ensure_model(self, access_token: str) -> str:
        if self._resolved_model:
            return self._resolved_model
        url = self._models_url()
        req = urllib.request.Request(
            url, headers=self._bearer_headers(access_token), method="GET"
        )
        ctx = _ssl_context_for_url(url)
        with urllib.request.urlopen(req, timeout=30, context=ctx) as resp:
            data: dict[str, Any] = json.loads(resp.read().decode())
        models = data.get("data") or []
        if not models:
            raise RuntimeError(
                "Список моделей пуст. Укажите GIGACHAT_MODEL в окружении."
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

        auth_key = self._effective_authorization_key()
        if not auth_key:
            primary = _AGENT_DIR / ".env"
            extra = _env_file_hints(primary)
            size_s = "—"
            if primary.is_file():
                try:
                    size_s = str(primary.stat().st_size)
                except OSError:
                    pass
            return (
                "Добавьте GIGACHAT_API_KEY: файл .env рядом с agent.py "
                "или export GIGACHAT_API_KEY=...\n"
                f"  Ожидаемый путь: {primary} (есть на диске: {'да' if primary.is_file() else 'нет'})\n"
                f"  Размер файла: {size_s} байт\n"
                "  Ключ авторизации: личный кабинет Studio → проект GigaChat API → Настройки API → Получить ключ\n"
                "  Документация: https://developers.sber.ru/docs/ru/gigachat/quickstart/ind-using-api"
                f"{extra}"
            )

        try:
            access_token = _get_gigachat_access_token(
                auth_key,
                self._config.oauth_url,
                self._config.scope,
                self._config.timeout_sec,
            )
        except OSError as e:
            msg = f"Ошибка при получении токена OAuth: {e}"
            return msg + _ssl_troubleshooting_hint(e)
        except urllib.error.HTTPError as e:
            err_body = e.read().decode("utf-8", errors="replace")
            return f"Ошибка OAuth HTTP {e.code}: {err_body[:2000]}"
        except (json.JSONDecodeError, RuntimeError, KeyError, TypeError) as e:
            return f"Ошибка разбора ответа OAuth: {e}"

        try:
            model = self._ensure_model(access_token)
        except OSError as e:
            msg = f"Сетевая ошибка при запросе списка моделей: {e}"
            return msg + _ssl_troubleshooting_hint(e)
        except urllib.error.HTTPError as e:
            err_body = e.read().decode("utf-8", errors="replace")
            return f"Ошибка HTTP {e.code} (models): {err_body[:2000]}"
        except RuntimeError as e:
            return str(e)

        self._messages.append({"role": "user", "content": text})
        self._trim_messages()

        payload = {
            "model": model,
            "messages": list(self._messages),
            "stream": False,
        }
        body = json.dumps(payload).encode("utf-8")
        url = self._chat_completions_url()
        req = urllib.request.Request(
            url,
            data=body,
            headers=self._bearer_headers(access_token),
            method="POST",
        )
        try:
            ctx = _ssl_context_for_url(url)
            with urllib.request.urlopen(
                req, timeout=self._config.timeout_sec, context=ctx
            ) as resp:
                raw = resp.read().decode("utf-8")
        except urllib.error.HTTPError as e:
            if self._messages and self._messages[-1].get("role") == "user":
                self._messages.pop()
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
            if self._messages and self._messages[-1].get("role") == "user":
                self._messages.pop()
            msg = f"Сетевая ошибка: {e}"
            return msg + _ssl_troubleshooting_hint(e)

        try:
            response_json: dict[str, Any] = json.loads(raw)
            choice = (response_json.get("choices") or [{}])[0]
            msg = choice.get("message") or {}
            content = msg.get("content")
            if isinstance(content, str):
                reply = content.strip()
            elif content is not None:
                reply = str(content).strip()
            else:
                reply = raw[:2000]
            self._messages.append({"role": "assistant", "content": reply})
            self._trim_messages()
            self._persist_history()
            return reply
        except (json.JSONDecodeError, KeyError, IndexError, TypeError) as e:
            if self._messages and self._messages[-1].get("role") == "user":
                self._messages.pop()
            return f"Не удалось разобрать ответ API: {e}\nФрагмент: {raw[:500]}"
