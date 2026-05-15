"""
LLM-агент: инкапсулирует запрос к GigaChat API и разбор ответа.
Интерфейс (CLI, web) только вызывает агента, не строит HTTP вручную.

Документация: https://developers.sber.ru/docs/ru/gigachat/quickstart/ind-using-api
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import ssl
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import certifi

from invariants import InvariantsStore
from memory_store import AssistantMemoryStore
from user_profile import UserProfileStore
from context_strategies import (
    RECENT_MESSAGE_WINDOW,
    ContextStrategyKind,
    UnifiedChatState,
    build_branching_api_messages,
    build_sliding_api_messages,
    build_sticky_facts_api_messages,
    flatten_messages_for_export,
    load_unified_state,
    save_unified_state,
    split_into_two_branches,
    switch_branch,
    trim_messages,
    update_facts_with_llm,
)
from document_index.rag import (
    RagRetrievalConfig,
    augment_user_message_with_rag,
    default_rag_index_path,
    format_rag_hit_lines,
    resolve_rag_index_path,
    validate_rag_grounding_reply,
)

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None  # type: ignore[misc, assignment]

_AGENT_DIR = Path(__file__).resolve().parent
_MCP_SERVER_SCRIPTS: dict[str, Path] = {
    "github": (_AGENT_DIR / "github_mcp_server.py").resolve(),
    "scheduler": (_AGENT_DIR / "scheduler_mcp_server.py").resolve(),
}
_MCP_TOOL_ROUTES: dict[str, str] = {
    "github_get_repo": "github",
    "search": "scheduler",
    "summorize": "scheduler",
    "saveToFile": "scheduler",
    "run_tools_pipeline": "scheduler",
    "schedule_upsert_task": "scheduler",
    "schedule_list_tasks": "scheduler",
    "schedule_run_due": "scheduler",
    "schedule_get_summary": "scheduler",
    "schedule_get_human_summary": "scheduler",
}

DEFAULT_HISTORY_FILENAME = "chat_history.json"
DEFAULT_HISTORY_MAX_MESSAGES = 200
TASK_STATE_FILENAME = "task_state.json"

# GigaChat: OAuth и REST (см. документацию Сбера)
GIGACHAT_OAUTH_URL = "https://ngw.devices.sberbank.ru:9443/api/v2/oauth"
GIGACHAT_API_V1_BASE = "https://gigachat.devices.sberbank.ru/api/v1"
DEFAULT_GIGACHAT_MODEL = "GigaChat"
DEFAULT_GIGACHAT_SCOPE = "GIGACHAT_API_PERS"

# LM Studio / OpenAI-compatible локальный сервер (lms server start)
DEFAULT_LOCAL_API_BASE = "http://127.0.0.1:1234/v1"
DEFAULT_LOCAL_API_KEY = "lm-studio"

BackendKind = Literal["cloud", "local"]


def _normalize_backend(raw: str) -> BackendKind:
    s = (raw or "cloud").strip().lower()
    if s in ("local", "lmstudio", "lm_studio", "lm-studio"):
        return "local"
    if s in ("cloud", "gigachat", "giga", "sber"):
        return "cloud"
    raise ValueError(f"Неизвестный backend: {raw!r} (ожидается cloud или local)")

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
def load_chat_history_file(path: Path) -> list[dict[str, str]]:
    """Список сообщений для отображения (ветка branching — активная ветка)."""
    st = load_unified_state(path)
    return flatten_messages_for_export(st)


def load_chat_history_state(path: Path) -> tuple[str, list[dict[str, str]]]:
    """(summary, messages): summary всегда пуст для v3; v2 summary не поднимается в API."""
    st = load_unified_state(path)
    return "", flatten_messages_for_export(st)


def save_chat_history_file(
    path: Path, messages: list[dict[str, str]], summary: str = ""
) -> None:
    """Совместимость: сохраняет как v3 sliding с последними репликами (summary игнорируется)."""
    st = UnifiedChatState(
        strategy=ContextStrategyKind.SLIDING_WINDOW,
        messages=messages[-RECENT_MESSAGE_WINDOW:],
    )
    save_unified_state(path, st)


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
    """Настройки API: cloud — GigaChat; local — LM Studio (OpenAI-compatible)."""

    backend: BackendKind = "cloud"
    oauth_url: str = GIGACHAT_OAUTH_URL
    api_base: str = GIGACHAT_API_V1_BASE
    authorization_key: str | None = None
    scope: str = DEFAULT_GIGACHAT_SCOPE
    model: str | None = DEFAULT_GIGACHAT_MODEL
    timeout_sec: float = 120.0

    @classmethod
    def from_env(
        cls,
        *,
        backend: str | None = None,
        local_model: str | None = None,
        local_url: str | None = None,
    ) -> AgentConfig:
        _load_project_dotenv()
        bk = _normalize_backend(
            backend or os.environ.get("LLM_AGENT_BACKEND") or "cloud"
        )
        if bk == "local":
            base = (
                (local_url or "").strip()
                or (os.environ.get("LLM_AGENT_LOCAL_BASE_URL") or "").strip()
                or (os.environ.get("LM_STUDIO_BASE_URL") or "").strip()
                or DEFAULT_LOCAL_API_BASE
            ).rstrip("/")
            model_raw = (
                (local_model or "").strip()
                or (os.environ.get("LLM_AGENT_LOCAL_MODEL") or "").strip()
                or (os.environ.get("LM_STUDIO_MODEL") or "").strip()
            )
            api_key = (
                (os.environ.get("LLM_AGENT_LOCAL_API_KEY") or "").strip()
                or (os.environ.get("LM_API_TOKEN") or "").strip()
                or DEFAULT_LOCAL_API_KEY
            )
            return cls(
                backend="local",
                oauth_url=GIGACHAT_OAUTH_URL,
                api_base=base,
                authorization_key=api_key,
                scope=DEFAULT_GIGACHAT_SCOPE,
                model=model_raw or None,
            )
        key = (os.environ.get("GIGACHAT_API_KEY") or "").strip()
        model = (os.environ.get("GIGACHAT_MODEL") or DEFAULT_GIGACHAT_MODEL).strip()
        scope = (os.environ.get("GIGACHAT_SCOPE") or DEFAULT_GIGACHAT_SCOPE).strip()
        oauth = (os.environ.get("GIGACHAT_OAUTH_URL") or GIGACHAT_OAUTH_URL).strip()
        base = (os.environ.get("GIGACHAT_API_BASE") or GIGACHAT_API_V1_BASE).rstrip("/")
        return cls(
            backend="cloud",
            oauth_url=oauth,
            api_base=base,
            authorization_key=key or None,
            scope=scope or DEFAULT_GIGACHAT_SCOPE,
            model=model or DEFAULT_GIGACHAT_MODEL,
        )


def _parse_tokens_count_response(data: Any) -> int | None:
    """Разбор ответа POST /api/v1/tokens/count (dict, список по строкам input или число)."""
    if isinstance(data, (int, float)):
        return int(data)
    if isinstance(data, list):
        total = 0
        found = False
        for item in data:
            n = _parse_tokens_count_response(item)
            if n is not None:
                total += n
                found = True
        return total if found else None
    if not isinstance(data, dict):
        return None
    u = data.get("usage")
    if isinstance(u, dict):
        for key in ("total_tokens", "prompt_tokens", "tokens"):
            v = u.get(key)
            if isinstance(v, (int, float)):
                return int(v)
    for key in ("total_tokens", "tokens", "token"):
        v = data.get(key)
        if isinstance(v, (int, float)):
            return int(v)
    inner = data.get("data")
    if isinstance(inner, list):
        return _parse_tokens_count_response(inner)
    return None


def _parse_usage_from_completion(response_json: dict[str, Any]) -> dict[str, int | None]:
    u = response_json.get("usage")
    if not isinstance(u, dict):
        return {
            "prompt_tokens": None,
            "completion_tokens": None,
            "total_tokens": None,
            "precached_prompt_tokens": None,
        }
    out: dict[str, int | None] = {}
    for key in ("prompt_tokens", "completion_tokens", "total_tokens", "precached_prompt_tokens"):
        v = u.get(key)
        out[key] = int(v) if isinstance(v, (int, float)) else None
    return out


@dataclass
class TokenStats:
    """Токены: текущая реплика пользователя, весь диалог на входе (подсчёт), ответ; данные usage из ответа генерации."""

    user_turn_tokens: int | None
    dialog_input_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None
    precached_prompt_tokens: int | None
    prompt_tokens_usage: int | None

    def format_line(self) -> str:
        parts: list[str] = []
        if self.user_turn_tokens is not None:
            parts.append(f"текущая реплика: {self.user_turn_tokens}")
        if self.dialog_input_tokens is not None:
            parts.append(f"весь диалог (вход): {self.dialog_input_tokens}")
        if self.completion_tokens is not None:
            parts.append(f"ответ: {self.completion_tokens}")
        if self.prompt_tokens_usage is not None:
            parts.append(f"prompt (usage): {self.prompt_tokens_usage}")
        if self.precached_prompt_tokens is not None and self.precached_prompt_tokens > 0:
            parts.append(f"из кэша prompt: {self.precached_prompt_tokens}")
        if self.total_tokens is not None:
            parts.append(f"всего к тарификации: {self.total_tokens}")
        if not parts:
            return ""
        return "[Токены] " + " | ".join(parts)


@dataclass
class RunResult:
    text: str
    stats: TokenStats | None = None
    """Метаданные последнего RAG-хода (если RAG был включён для этого run)."""
    rag: dict[str, Any] | None = None
    """Строки retrieval (чанки), всегда печатаются мини-чатом рядом с ответом."""
    rag_source_lines: list[str] | None = None


TaskStage = Literal["planning", "plan_approved", "execution", "validation", "done"]
_TASK_STAGES: tuple[TaskStage, ...] = (
    "planning",
    "plan_approved",
    "execution",
    "validation",
    "done",
)
_TASK_ALLOWED_TRANSITIONS: dict[TaskStage, tuple[TaskStage, ...]] = {
    "planning": ("plan_approved",),
    "plan_approved": ("execution",),
    "execution": ("validation",),
    "validation": ("done",),
    "done": (),
}
_TASK_DEFAULT_EXPECTED_ACTION: dict[TaskStage, str] = {
    "planning": "Сформировать и согласовать план.",
    "plan_approved": "План утверждён. Перейти к реализации, когда пользователь подтвердит старт.",
    "execution": "Реализовать согласованный план без пропуска валидации.",
    "validation": "Проверить результат (тесты/проверки) перед финалом.",
    "done": "Финализировать результат и закрыть задачу.",
}
_TASK_STAGE_GUIDANCE: dict[TaskStage, str] = {
    "planning": (
        "Только анализ и планирование. Нельзя переходить к реализации до состояния "
        "plan_approved."
    ),
    "plan_approved": (
        "План зафиксирован. Подготовка к реализации допустима, но сама реализация "
        "начинается только в execution."
    ),
    "execution": "Только реализация. Финализировать результат до валидации нельзя.",
    "validation": "Только валидация результата. Переход в done возможен после проверки.",
    "done": "Только финальный ответ и закрытие задачи.",
}


@dataclass
class TaskState:
    stage: TaskStage = "planning"
    current_step: str = ""
    expected_action: str = ""
    paused: bool = False


class TaskStateMachine:
    def _normalize_stage(self, stage: str) -> TaskStage | None:
        val = (stage or "").strip().lower()
        aliases = {
            "approved_plan": "plan_approved",
            "approved": "plan_approved",
            "plan-approved": "plan_approved",
        }
        val = aliases.get(val, val)
        if val in _TASK_STAGES:
            return val  # type: ignore[return-value]
        return None

    """Формализованное состояние задачи: этап, шаг, ожидаемое действие."""

    def __init__(self, storage_path: Path) -> None:
        self._path = storage_path
        self.state = TaskState()
        self._load()

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

    def _load(self) -> None:
        data = self._read_json()
        if not isinstance(data, dict):
            self.state = TaskState()
            return
        stage = self._normalize_stage(str(data.get("stage") or "planning")) or "planning"
        self.state = TaskState(
            stage=stage,
            current_step=str(data.get("current_step") or "").strip(),
            expected_action=str(data.get("expected_action") or "").strip(),
            paused=bool(data.get("paused", False)),
        )

    def persist(self) -> None:
        self._write_json(
            {
                "version": 1,
                "type": "task_state",
                "stage": self.state.stage,
                "current_step": self.state.current_step,
                "expected_action": self.state.expected_action,
                "paused": self.state.paused,
            }
        )

    def _next_stage(self) -> TaskStage | None:
        allowed = _TASK_ALLOWED_TRANSITIONS[self.state.stage]
        if not allowed:
            return None
        return allowed[0]

    def set_stage(self, stage: str) -> tuple[bool, str]:
        val = self._normalize_stage(stage)
        if val is None:
            return (
                False,
                "Этап должен быть одним из: planning, plan_approved, execution, validation, done.",
            )
        current = self.state.stage
        if val == current:
            return True, ""
        allowed = _TASK_ALLOWED_TRANSITIONS[current]
        if val not in allowed:
            next_s = ", ".join(allowed) if allowed else "(нет — задача завершена)"
            return (
                False,
                f"Недопустимый переход: {current} -> {val}. "
                f"Разрешён следующий переход: {next_s}.",
            )
        self.state.stage = val
        if not self.state.expected_action.strip():
            self.state.expected_action = _TASK_DEFAULT_EXPECTED_ACTION.get(val, "")
        self.persist()
        return True, ""

    def set_step(self, step: str) -> tuple[bool, str]:
        self.state.current_step = (step or "").strip()
        self.persist()
        return True, ""

    def set_expected_action(self, action: str) -> tuple[bool, str]:
        self.state.expected_action = (action or "").strip()
        self.persist()
        return True, ""

    def advance(self) -> tuple[bool, str]:
        next_stage = self._next_stage()
        if next_stage is None:
            return False, "Задача уже в состоянии done."
        return self.set_stage(next_stage)

    def pause(self) -> tuple[bool, str]:
        if self.state.paused:
            return False, "Задача уже на паузе."
        self.state.paused = True
        self.persist()
        return True, ""

    def resume(self) -> tuple[bool, str]:
        if not self.state.paused:
            return False, "Задача не на паузе."
        self.state.paused = False
        self.persist()
        return True, ""

    def reset(self, *, step: str = "", expected_action: str = "") -> None:
        self.state = TaskState(
            stage="planning",
            current_step=(step or "").strip(),
            expected_action=(
                (expected_action or "").strip()
                or _TASK_DEFAULT_EXPECTED_ACTION["planning"]
            ),
            paused=False,
        )
        self.persist()

    def format_lines(self) -> str:
        status = "paused" if self.state.paused else "active"
        allowed = _TASK_ALLOWED_TRANSITIONS[self.state.stage]
        next_s = ", ".join(allowed) if allowed else "(нет — финальное состояние)"
        lines = [
            "Состояние задачи (FSM):",
            f"  этап: {self.state.stage}",
            f"  текущий шаг: {self.state.current_step or '(не задан)'}",
            f"  ожидаемое действие: {self.state.expected_action or '(не задано)'}",
            f"  статус: {status}",
            f"  разрешённый следующий этап: {next_s}",
        ]
        return "\n".join(lines)

    def build_system_message(self) -> list[dict[str, str]]:
        status = "paused" if self.state.paused else "active"
        allowed = _TASK_ALLOWED_TRANSITIONS[self.state.stage]
        next_s = ", ".join(allowed) if allowed else "(нет)"
        content = (
            "Формализованное состояние задачи (используй как конечный автомат):\n"
            f"- этап: {self.state.stage}\n"
            f"- текущий шаг: {self.state.current_step or '(не задан)'}\n"
            f"- ожидаемое действие: {self.state.expected_action or '(не задано)'}\n"
            f"- статус: {status}\n"
            f"- разрешённый следующий этап: {next_s}\n"
            "- Допустимые этапы: planning -> plan_approved -> execution -> validation -> done.\n"
            "- Запрещено перепрыгивать этапы или выполнять действия следующего этапа заранее.\n"
            f"- Правило текущего этапа: {_TASK_STAGE_GUIDANCE[self.state.stage]}\n"
            "- Если пользователь просит действие не из текущего этапа, откажись и попроси "
            "перевести задачу в корректный этап через /task next или /task stage.\n"
            "- Если состояние resumed после паузы, продолжай с текущего шага без повторного "
            "пересказа прошлых объяснений."
        )
        return [{"role": "system", "content": content}]


def _replace_last_user_message_content(
    messages: list[dict[str, str]], new_content: str
) -> list[dict[str, str]]:
    """Копия списка сообщений с заменой текста последней реплики role=user (для RAG)."""
    out: list[dict[str, str]] = [dict(m) for m in messages]
    for i in range(len(out) - 1, -1, -1):
        if str(out[i].get("role")) == "user":
            out[i] = {"role": "user", "content": new_content}
            break
    return out


def _merge_leading_system_messages(
    messages: list[dict[str, str]],
) -> list[dict[str, str]]:
    """
    GigaChat отклоняет несколько подряд system: «system message must be the first message».
    Склеиваем все ведущие system в одно, затем идут остальные роли.
    """
    if not messages:
        return []
    parts: list[str] = []
    i = 0
    while i < len(messages) and messages[i].get("role") == "system":
        parts.append(str(messages[i].get("content", "")))
        i += 1
    out: list[dict[str, str]] = []
    merged = "\n\n".join(p.strip() for p in parts if p.strip())
    if merged:
        out.append({"role": "system", "content": merged})
    out.extend(messages[i:])
    return out


class LLMAgent:
    """
    Сущность «агент»: принимает пользовательский запрос, вызывает LLM, возвращает RunResult.
    Контекст задаётся стратегией (см. context_strategies): sliding_window и sticky_facts
    хранят последние N реплик (по умолчанию 6 = три пары user/assistant; N задаётся
    LLM_AGENT_RECENT_MESSAGES), branching — checkpoint и две независимые ветки.
    Память разделена на три слоя (см. memory_store): short_term, working, long_term.
    Персонализация (см. user_profile): несколько именованных профилей, активный задаёт
    предпочтения стиля/формата/ограничений и подмешивается в запрос перед блоками памяти.
    Инварианты проекта (см. invariants): архитектура, решения, стек, бизнес-правила —
    хранятся вне диалога и подмешиваются в системный контекст; агент обязан их соблюдать.
    working/long_term попадают в запрос как системные блоки; записи в них делаются явно
    через save_memory_entry (CLI: /memory put ...).
    Режим RAG (run(..., rag=True) или set_rag): эмбеддинг вопроса, top-k чанков из JSON-индекса,
    объединение с вопросом в одно пользовательское сообщение к API; в истории хранится исходный вопрос.
    """

    def __init__(
        self,
        config: AgentConfig | None = None,
        *,
        backend: str | None = None,
        local_model: str | None = None,
        local_url: str | None = None,
        context_strategy: ContextStrategyKind | str | None = None,
        rag_enabled: bool | None = None,
        rag_index_path: Path | str | None = None,
        rag_top_k: int | None = None,
        rag_retrieval_config: RagRetrievalConfig | None = None,
    ) -> None:
        if config is None:
            config = AgentConfig.from_env(
                backend=backend,
                local_model=local_model,
                local_url=local_url,
            )
        self._config = config
        self._resolved_model: str | None = self._config.model
        self._history_path = _history_path_resolved()
        self._history_max_messages = _history_max_from_env()
        self._state = load_unified_state(self._history_path)
        if context_strategy is not None:
            self._state.strategy = self._coerce_strategy(context_strategy)
        else:
            env_s = (os.environ.get("LLM_AGENT_CONTEXT_STRATEGY") or "").strip()
            if env_s:
                try:
                    self._state.strategy = self._coerce_strategy(env_s)
                except ValueError:
                    pass
        self._memory = AssistantMemoryStore()
        self._invariants = InvariantsStore(memory_dir=self._memory._dir)
        self._user_profile = UserProfileStore()
        self._task_state = TaskStateMachine(
            self._memory._dir / TASK_STATE_FILENAME
        )
        if rag_enabled is None:
            self._rag_enabled = (os.environ.get("LLM_AGENT_RAG") or "").strip().lower() in (
                "1",
                "true",
                "yes",
                "on",
            )
        else:
            self._rag_enabled = bool(rag_enabled)
        rp = ""
        if rag_index_path is not None:
            rp = str(Path(rag_index_path).expanduser()).strip()
        else:
            rp = (os.environ.get("LLM_AGENT_RAG_INDEX") or "").strip()
        if rp:
            try:
                self._rag_index_path = resolve_rag_index_path(rp)
            except FileNotFoundError:
                self._rag_index_path = Path(rp).expanduser().resolve()
        else:
            self._rag_index_path = default_rag_index_path()
        tk_env = (os.environ.get("LLM_AGENT_RAG_TOP_K") or "").strip()
        self._rag_top_k = max(
            1,
            int(rag_top_k)
            if rag_top_k is not None
            else (int(tk_env) if tk_env.isdigit() else 5),
        )
        self._rag_retrieval_config = rag_retrieval_config
        self._sync_short_term_memory(decay_notes=False)

    @staticmethod
    def _coerce_strategy(val: ContextStrategyKind | str) -> ContextStrategyKind:
        if isinstance(val, ContextStrategyKind):
            return val
        return ContextStrategyKind(str(val).strip())

    def set_context_strategy(self, strategy: ContextStrategyKind | str) -> None:
        """Переключает стратегию для текущей сессии (состояние в памяти — из файла)."""
        self._state.strategy = self._coerce_strategy(strategy)

    @property
    def context_strategy(self) -> ContextStrategyKind:
        return self._state.strategy

    def split_dialog_branches(self) -> tuple[bool, str]:
        """Checkpoint: создаёт две ветки (branch_a, branch_b) от текущего конца диалога."""
        ok, err = split_into_two_branches(self._state)
        if ok:
            self._persist_history()
        return (ok, err)

    def switch_dialog_branch(self, branch_id: str) -> tuple[bool, str]:
        ok, err = switch_branch(self._state, branch_id)
        if ok:
            self._persist_history()
        return (ok, err)

    def branching_status_line(self) -> str:
        if self._state.strategy != ContextStrategyKind.BRANCHING:
            return f"Стратегия: {self._state.strategy.value}"
        if not self._state.branching_split:
            return (
                f"Стратегия: branching (до checkpoint; реплик: {len(self._state.messages)})"
            )
        bids = ", ".join(sorted(self._state.branches.keys()))
        return (
            f"Стратегия: branching | prefix: {len(self._state.branch_prefix)} | "
            f"ветки: {bids} | активная: {self._state.active_branch}"
        )

    @property
    def rag_enabled(self) -> bool:
        return self._rag_enabled

    def set_rag(self, enabled: bool, *, index_path: Path | str | None = None) -> None:
        self._rag_enabled = bool(enabled)
        if index_path is not None:
            p = str(Path(index_path).expanduser()).strip()
            if not p:
                self._rag_index_path = None
            else:
                try:
                    self._rag_index_path = resolve_rag_index_path(p)
                except FileNotFoundError:
                    self._rag_index_path = Path(p).resolve()
        elif enabled and self._rag_index_path is None:
            self._rag_index_path = default_rag_index_path()

    def set_rag_top_k(self, k: int) -> None:
        self._rag_top_k = max(1, int(k))

    @property
    def backend(self) -> BackendKind:
        return self._config.backend

    def set_backend(
        self,
        backend: str,
        *,
        local_model: str | None = None,
        local_url: str | None = None,
    ) -> None:
        """Переключить cloud (GigaChat) / local (LM Studio) в текущей сессии."""
        bk = _normalize_backend(backend)
        if bk == "local":
            keep_model = (
                (local_model or "").strip()
                or (
                    self._config.model
                    if self._config.backend == "local"
                    else ""
                )
            )
            keep_url = (
                (local_url or "").strip()
                or (
                    self._config.api_base
                    if self._config.backend == "local"
                    else ""
                )
            )
            self._config = AgentConfig.from_env(
                backend="local",
                local_model=keep_model or None,
                local_url=keep_url or None,
            )
        else:
            self._config = AgentConfig.from_env(backend="cloud")
        self._resolved_model = self._config.model
        self._clear_dialog_on_backend_switch()

    def _clear_dialog_on_backend_switch(self) -> None:
        """Сброс диалога при смене backend — иначе FSM/отказы из прошлой сессии ломают local."""
        self._state.messages = []
        self._state.facts = {}
        self._state.branching_split = False
        self._state.branch_prefix = []
        self._state.branches = {}
        self._state.active_branch = "branch_a"
        self._persist_history()

    def _local_backend_system_messages(self) -> list[dict[str, str]]:
        model_s = self._resolved_model or self._config.model or "local"
        return [
            {
                "role": "system",
                "content": (
                    f"Ты локальная языковая модель «{model_s}», запущенная через LM Studio на этом компьютере. "
                    "Это не GigaChat и не облачный ChatGPT. "
                    "Отвечай по существу на вопросы пользователя на том языке, на котором он пишет. "
                    "Не отказывайся от обычных информационных вопросов."
                ),
            }
        ]

    def backend_status_line(self) -> str:
        if self._config.backend == "local":
            model_s = self._resolved_model or self._config.model or "(авто из /v1/models)"
            return (
                f"Backend: local (LM Studio) | {self._config.api_base} | модель: {model_s}"
            )
        model_s = self._config.model or DEFAULT_GIGACHAT_MODEL
        return f"Backend: cloud (GigaChat) | модель: {model_s}"

    def rag_status_line(self) -> str:
        if not self._rag_enabled:
            return "RAG: выкл"
        p = self._rag_index_path
        ps = str(p) if p else "(путь не задан)"
        if self._rag_retrieval_config is not None:
            c = self._rag_retrieval_config
            return (
                f"RAG: вкл | явный конфиг | final_k={c.final_top_k} | retrieve_k={c.retrieve_k} | "
                f"min_sim={c.min_similarity} | hybrid={c.hybrid_rerank} | rewrite={c.query_rewrite} | индекс: {ps}"
            )
        c = RagRetrievalConfig.from_env(self._rag_top_k)
        return (
            f"RAG: вкл | final_k={c.final_top_k} | retrieve_k={c.retrieve_k} | "
            f"min_sim={c.min_similarity} | hybrid={c.hybrid_rerank} | rewrite={c.query_rewrite} | индекс: {ps}"
        )

    def merge_fact(self, key: str, value: str) -> tuple[bool, str]:
        """Ручное добавление или обновление пары ключ–значение в facts (только sticky_facts)."""
        if self._state.strategy != ContextStrategyKind.STICKY_FACTS:
            return (
                False,
                "facts используются только в режиме sticky_facts. Введите: /strategy sticky_facts",
            )
        k = (key or "").strip()
        if not k:
            return False, "Укажите непустой ключ."
        self._state.facts[k] = (value or "").strip()
        self._persist_history()
        return True, ""

    def format_facts_lines(self) -> str:
        if not self._state.facts:
            return "(facts пусто)"
        lines = [f"  {k}: {v}" for k, v in sorted(self._state.facts.items())]
        return "\n".join(lines)

    def save_memory_entry(
        self,
        memory_type: str,
        key: str,
        value: str,
        *,
        long_term_section: str = "",
    ) -> tuple[bool, str]:
        """
        Явное сохранение в выбранный тип памяти:
        - short_term: временные заметки в текущем диалоге
        - working: данные текущей задачи
        - long_term: профиль/решения/знания (нужна секция)
        """
        mt = (memory_type or "").strip().lower()
        if mt in ("short", "short_term"):
            return self._memory.put_short_note(key, value)
        if mt in ("working", "working_memory"):
            return self._memory.put_working(key, value)
        if mt in ("long", "long_term"):
            return self._memory.put_long_term(long_term_section, key, value)
        return (
            False,
            "Неизвестный тип памяти. Используйте: short_term, working, long_term.",
        )

    def format_memory_lines(self) -> str:
        return self._memory.format_summary()

    def format_user_profile_lines(self) -> str:
        return self._user_profile.format_lines()

    def format_user_profile_list_lines(self) -> str:
        return self._user_profile.format_list_lines()

    def format_task_state_lines(self) -> str:
        return self._task_state.format_lines()

    def format_invariants_lines(self) -> str:
        return self._invariants.format_lines()

    def set_invariant_section(self, section: str, value: str) -> tuple[bool, str]:
        return self._invariants.set_section(section, value)

    def clear_invariant_section(self, section: str) -> tuple[bool, str]:
        return self._invariants.clear_section(section)

    def task_set_stage(self, stage: str) -> tuple[bool, str]:
        return self._task_state.set_stage(stage)

    def task_set_step(self, step: str) -> tuple[bool, str]:
        return self._task_state.set_step(step)

    def task_set_expected_action(self, action: str) -> tuple[bool, str]:
        return self._task_state.set_expected_action(action)

    def task_advance(self) -> tuple[bool, str]:
        return self._task_state.advance()

    def task_pause(self) -> tuple[bool, str]:
        return self._task_state.pause()

    def task_resume(self) -> tuple[bool, str]:
        return self._task_state.resume()

    def task_reset(self, *, step: str = "", expected_action: str = "") -> None:
        self._task_state.reset(step=step, expected_action=expected_action)

    def set_user_profile_field(self, name: str, value: str) -> tuple[bool, str]:
        return self._user_profile.set_field(name, value)

    def set_user_profile_fields(self, assignments: dict[str, str]) -> tuple[bool, str]:
        return self._user_profile.set_fields(assignments)

    def activate_user_profile(self, profile_id: str) -> tuple[bool, str]:
        return self._user_profile.activate(profile_id)

    def create_user_profile(
        self, profile_id: str, *, copy_from_active: bool
    ) -> tuple[bool, str]:
        template = self._user_profile.profile if copy_from_active else None
        return self._user_profile.create_profile(profile_id, template=template)

    def duplicate_user_profile(self, profile_id: str) -> tuple[bool, str]:
        return self._user_profile.duplicate_profile(profile_id)

    def delete_user_profile(self, profile_id: str) -> tuple[bool, str]:
        return self._user_profile.delete_profile(profile_id)

    def propose_memory_entries(self, user_message: str) -> list[dict[str, str]]:
        """
        Простая эвристика маршрутизации памяти.
        Ничего не сохраняет — только предлагает кандидаты.
        """
        text = (user_message or "").strip()
        low = text.lower()
        out: list[dict[str, str]] = []

        def add(mem_type: str, key: str, value: str, section: str = "") -> None:
            out.append(
                {
                    "type": mem_type,
                    "section": section,
                    "key": key.strip(),
                    "value": value.strip(),
                }
            )

        m = re.search(r"(?:моя|my)\s+цель\s*[:=-]\s*(.+)", text, flags=re.IGNORECASE)
        if m:
            add("working", "goal", m.group(1))
        elif low.startswith("цель ") or low.startswith("goal "):
            add("working", "goal", text.split(" ", 1)[1] if " " in text else text)

        m = re.search(
            r"(?:ограничени[ея]|constraint[s]?)\s*[:=-]\s*(.+)",
            text,
            flags=re.IGNORECASE,
        )
        if m:
            add("working", "constraints", m.group(1))

        m = re.search(
            r"(?:дедлайн|deadline)\s*[:=-]?\s*(.+)", text, flags=re.IGNORECASE
        )
        if m:
            add("working", "deadline", m.group(1))

        m = re.search(r"я\s+болею\s+за\s+(.+)", text, flags=re.IGNORECASE)
        if m:
            add("long_term", "favorite_team", m.group(1), section="profile")

        m = re.search(
            r"(?:предпочитаю|люблю|мой язык)\s+(.+)", text, flags=re.IGNORECASE
        )
        if m:
            add("long_term", "preference", m.group(1), section="profile")

        m = re.search(
            r"(?:решили|договорились)\s*[:,-]?\s*(.+)", text, flags=re.IGNORECASE
        )
        if m:
            add("long_term", "latest_decision", m.group(1), section="decisions")

        uniq: list[dict[str, str]] = []
        seen: set[tuple[str, str, str, str]] = set()
        for item in out:
            signature = (
                item["type"],
                item["section"],
                item["key"],
                item["value"],
            )
            if signature in seen:
                continue
            seen.add(signature)
            uniq.append(item)
        return uniq

    def apply_memory_proposals(
        self, proposals: list[dict[str, str]]
    ) -> tuple[int, list[str]]:
        """Сохраняет предложенные записи и возвращает (кол-во, ошибки)."""
        saved = 0
        errors: list[str] = []
        for item in proposals:
            mem_type = item.get("type", "")
            section = item.get("section", "")
            key = item.get("key", "")
            value = item.get("value", "")
            ok, err = self.save_memory_entry(
                mem_type, key, value, long_term_section=section
            )
            if ok:
                saved += 1
            else:
                errors.append(err)
        return saved, errors

    def fetch_github_repo_via_mcp(
        self, owner: str, repo: str, *, include_readme: bool = False
    ) -> dict[str, Any]:
        """
        Вызывает локальный MCP-сервер GitHub и возвращает результат инструмента.
        """
        return self._call_mcp_tool(
            "github_get_repo",
            {
                "owner": owner,
                "repo": repo,
                "include_readme": include_readme,
            },
        )

    def _call_scheduler_tool_via_mcp(
        self, tool_name: str, arguments: dict[str, Any]
    ) -> dict[str, Any]:
        return self._call_mcp_tool(tool_name, arguments, server_name="scheduler")

    def _call_mcp_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        *,
        server_name: str = "",
    ) -> dict[str, Any]:
        route = (server_name or "").strip().lower() or _MCP_TOOL_ROUTES.get(tool_name, "")
        if route not in _MCP_SERVER_SCRIPTS:
            return {
                "status": "error",
                "error": (
                    f"Не найден MCP-сервер для инструмента '{tool_name}'. "
                    f"Известные серверы: {', '.join(sorted(_MCP_SERVER_SCRIPTS.keys()))}"
                ),
            }

        async def _call() -> dict[str, Any]:
            from mcp import ClientSession, StdioServerParameters
            from mcp.client.stdio import stdio_client

            server_script = str(_MCP_SERVER_SCRIPTS[route])
            server_params = StdioServerParameters(
                command="python3",
                args=[server_script],
                env=dict(os.environ),
            )
            async with stdio_client(server_params) as (read_stream, write_stream):
                async with ClientSession(read_stream, write_stream) as session:
                    await session.initialize()
                    tool_result = await session.call_tool(tool_name, arguments)
            for item in getattr(tool_result, "content", []):
                text = getattr(item, "text", "")
                if isinstance(text, str) and text.strip():
                    try:
                        parsed = json.loads(text)
                        if isinstance(parsed, dict):
                            return parsed
                    except json.JSONDecodeError:
                        return {"status": "ok", "raw": text}
            return {
                "status": "error",
                "error": f"Пустой ответ MCP инструмента '{tool_name}' от сервера '{route}'.",
            }

        try:
            return asyncio.run(_call())
        except RuntimeError:
            loop = asyncio.new_event_loop()
            try:
                return loop.run_until_complete(_call())
            finally:
                loop.close()
        except Exception as e:
            return {
                "status": "error",
                "error": (
                    f"MCP вызов не выполнен: tool={tool_name}, server={route}, error={e}"
                ),
            }

    def run_multi_server_mcp_flow(
        self,
        *,
        owner: str,
        repo: str,
        query: str,
        file_path: str = "",
    ) -> dict[str, Any]:
        """
        Длинный orchestration-flow через несколько MCP-серверов:
        1) github_get_repo (github server)
        2) run_tools_pipeline (scheduler server)
        3) schedule_upsert_task reminder (scheduler server)
        4) schedule_list_tasks (scheduler server)
        """
        steps: list[dict[str, Any]] = []
        call_order = 0

        def _step(tool: str, args: dict[str, Any]) -> dict[str, Any]:
            nonlocal call_order
            call_order += 1
            route = _MCP_TOOL_ROUTES.get(tool, "unknown")
            result = self._call_mcp_tool(tool, args)
            row = {
                "order": call_order,
                "tool": tool,
                "server": route,
                "status": result.get("status"),
                "ok": result.get("status") == "ok",
                "result": result,
            }
            steps.append(row)
            return result

        github_data = _step(
            "github_get_repo",
            {"owner": owner, "repo": repo, "include_readme": False},
        )
        if github_data.get("status") != "ok":
            return {"status": "error", "failed_step": "github_get_repo", "steps": steps}

        repo_name = str(github_data.get("full_name") or f"{owner}/{repo}")
        stars = int(github_data.get("stargazers_count") or 0)
        pipeline_query = f"{query}. repo={repo_name}; stars={stars}"
        pipeline_data = _step(
            "run_tools_pipeline",
            {"query": pipeline_query, "file_path": file_path, "limit": 5},
        )
        if pipeline_data.get("status") != "ok":
            return {"status": "error", "failed_step": "run_tools_pipeline", "steps": steps}

        reminder_message = f"Проверить pipeline для {repo_name}"
        upsert_data = _step(
            "schedule_upsert_task",
            {
                "name": f"mcp-flow-{owner}-{repo}",
                "kind": "reminder",
                "payload": {"message": reminder_message},
                "delay_seconds": 1,
                "interval_seconds": 0,
                "active": True,
                "max_runs": 1,
            },
        )
        if upsert_data.get("status") != "ok":
            return {"status": "error", "failed_step": "schedule_upsert_task", "steps": steps}

        list_data = _step("schedule_list_tasks", {"include_inactive": True})
        if list_data.get("status") != "ok":
            return {"status": "error", "failed_step": "schedule_list_tasks", "steps": steps}

        expected_tools = [
            "github_get_repo",
            "run_tools_pipeline",
            "schedule_upsert_task",
            "schedule_list_tasks",
        ]
        expected_servers = ["github", "scheduler", "scheduler", "scheduler"]
        actual_tools = [str(x.get("tool") or "") for x in steps]
        actual_servers = [str(x.get("server") or "") for x in steps]
        order_ok = actual_tools == expected_tools
        routes_ok = actual_servers == expected_servers

        return {
            "status": "ok",
            "flow": "multi_server_orchestration",
            "steps": steps,
            "verification": {
                "order_ok": order_ok,
                "routes_ok": routes_ok,
                "expected_tools": expected_tools,
                "actual_tools": actual_tools,
                "expected_servers": expected_servers,
                "actual_servers": actual_servers,
            },
        }

    @staticmethod
    def _extract_repo_ref(text: str) -> tuple[str, str] | None:
        m = re.search(r"\b([A-Za-z0-9_.-]+)/([A-Za-z0-9_.-]+)\b", text or "")
        if not m:
            return None
        return m.group(1), m.group(2)

    def route_mcp_request(self, request: str) -> dict[str, Any]:
        """
        Policy-роутер MCP:
        - GitHub metadata: github_get_repo
        - Pipeline: run_tools_pipeline
        - Schedule: schedule_* инструменты
        - Смешанный сценарий: multi-server flow
        """
        text = (request or "").strip()
        if not text:
            return {"status": "error", "error": "Пустой запрос для MCP-роутера."}

        low = text.lower()
        repo_ref = self._extract_repo_ref(text)
        has_repo = repo_ref is not None
        has_schedule = any(
            token in low
            for token in (
                "schedule",
                "распис",
                "напомин",
                "reminder",
                "summary",
                "сводк",
                "collector",
            )
        )
        has_pipeline = any(
            token in low
            for token in ("pipeline", "пайп", "search", "summorize", "save")
        )
        has_github = any(
            token in low for token in ("github", "repo", "repository", "репозитор")
        )
        wants_run_due = any(token in low for token in ("run due", "run_due", "выполни due"))
        wants_list = any(token in low for token in (" list", "список", "show tasks"))
        wants_summary = any(token in low for token in ("summary", "сводк", "report"))

        if has_repo and has_pipeline:
            owner, repo = repo_ref or ("", "")
            result = self.run_multi_server_mcp_flow(
                owner=owner,
                repo=repo,
                query=text,
                file_path="memory/pipeline_from_cli.txt",
            )
            return {
                "status": result.get("status", "error"),
                "selected_route": "multi_server_flow",
                "selected_server": "github+scheduler",
                "selected_tool": "github_get_repo + run_tools_pipeline + schedule_*",
                "result": result,
            }

        if has_repo and (has_github or not has_schedule):
            owner, repo = repo_ref or ("", "")
            result = self._call_mcp_tool(
                "github_get_repo",
                {"owner": owner, "repo": repo, "include_readme": False},
            )
            return {
                "status": result.get("status", "error"),
                "selected_route": "github_repo_metadata",
                "selected_server": "github",
                "selected_tool": "github_get_repo",
                "result": result,
            }

        if has_schedule and wants_run_due:
            result = self._call_mcp_tool("schedule_run_due", {"limit": 20})
            return {
                "status": result.get("status", "error"),
                "selected_route": "scheduler_run_due",
                "selected_server": "scheduler",
                "selected_tool": "schedule_run_due",
                "result": result,
            }

        if has_schedule and wants_list:
            result = self._call_mcp_tool("schedule_list_tasks", {"include_inactive": True})
            return {
                "status": result.get("status", "error"),
                "selected_route": "scheduler_list",
                "selected_server": "scheduler",
                "selected_tool": "schedule_list_tasks",
                "result": result,
            }

        if has_schedule and wants_summary:
            result = self._call_mcp_tool("schedule_get_human_summary", {"hours": 24})
            return {
                "status": result.get("status", "error"),
                "selected_route": "scheduler_summary",
                "selected_server": "scheduler",
                "selected_tool": "schedule_get_human_summary",
                "result": result,
            }

        if has_pipeline:
            result = self._call_mcp_tool(
                "run_tools_pipeline",
                {
                    "query": text,
                    "file_path": "memory/pipeline_from_cli.txt",
                    "limit": 5,
                },
            )
            return {
                "status": result.get("status", "error"),
                "selected_route": "pipeline",
                "selected_server": "scheduler",
                "selected_tool": "run_tools_pipeline",
                "result": result,
            }

        result = self._call_mcp_tool("schedule_get_summary", {"hours": 24})
        return {
            "status": result.get("status", "error"),
            "selected_route": "fallback_summary",
            "selected_server": "scheduler",
            "selected_tool": "schedule_get_summary",
            "result": result,
            "note": "Запрос не распознан явно; применён fallback summary.",
        }

    def scheduler_upsert_task_via_mcp(
        self,
        name: str,
        kind: str,
        *,
        payload: dict[str, Any] | None = None,
        delay_seconds: int = 0,
        interval_seconds: int = 0,
        active: bool = True,
        max_runs: int = 0,
    ) -> dict[str, Any]:
        return self._call_scheduler_tool_via_mcp(
            "schedule_upsert_task",
            {
                "name": name,
                "kind": kind,
                "payload": payload or {},
                "delay_seconds": int(delay_seconds),
                "interval_seconds": int(interval_seconds),
                "active": bool(active),
                "max_runs": int(max_runs),
            },
        )

    def scheduler_list_tasks_via_mcp(self, *, include_inactive: bool = False) -> dict[str, Any]:
        return self._call_scheduler_tool_via_mcp(
            "schedule_list_tasks",
            {"include_inactive": include_inactive},
        )

    def scheduler_run_due_via_mcp(self, *, limit: int = 20) -> dict[str, Any]:
        return self._call_scheduler_tool_via_mcp("schedule_run_due", {"limit": int(limit)})

    def scheduler_summary_via_mcp(self, *, hours: int = 24) -> dict[str, Any]:
        return self._call_scheduler_tool_via_mcp(
            "schedule_get_summary",
            {"hours": int(hours)},
        )

    def scheduler_human_summary_via_mcp(self, *, hours: int = 24) -> dict[str, Any]:
        return self._call_scheduler_tool_via_mcp(
            "schedule_get_human_summary",
            {"hours": int(hours)},
        )

    def scheduler_run_tools_pipeline_via_mcp(
        self, *, query: str, file_path: str = "", limit: int = 5
    ) -> dict[str, Any]:
        return self._call_scheduler_tool_via_mcp(
            "run_tools_pipeline",
            {
                "query": str(query or "").strip(),
                "file_path": str(file_path or "").strip(),
                "limit": int(limit),
            },
        )

    def _sync_short_term_memory(self, *, decay_notes: bool = True) -> None:
        """Краткосрочная память: синхронизируется с текущим диалогом."""
        dialog = flatten_messages_for_export(self._state)
        try:
            self._memory.update_short_term_dialog(dialog, decay_notes=decay_notes)
        except OSError:
            pass

    def _persist_history(self) -> None:
        try:
            save_unified_state(self._history_path, self._state)
        except OSError:
            pass
        self._sync_short_term_memory()

    def _rollback_last_user(self) -> None:
        if (
            self._state.strategy == ContextStrategyKind.BRANCHING
            and self._state.branching_split
        ):
            branch = self._state.branches.get(self._state.active_branch)
            if branch and branch[-1].get("role") == "user":
                branch.pop()
            return
        if self._state.messages and self._state.messages[-1].get("role") == "user":
            self._state.messages.pop()

    def _trim_branching_linear(self) -> None:
        if self._state.strategy != ContextStrategyKind.BRANCHING:
            return
        if self._state.branching_split:
            return
        while len(self._state.messages) > self._history_max_messages:
            self._state.messages.pop(0)

    def _trim_branching_after_split(self) -> None:
        """
        Обрезка только активной ветки. Общий checkpoint (branch_prefix) не трогаем —
        иначе длинная сессия в другой ветке «съедала» бы историю до разветвления
        и ломала бы контекст при возврате на первую ветку.
        """
        if self._state.strategy != ContextStrategyKind.BRANCHING:
            return
        if not self._state.branching_split:
            return
        active = self._state.active_branch
        branch = self._state.branches.setdefault(active, [])
        while len(branch) > self._history_max_messages:
            branch.pop(0)

    def _build_messages_for_api(self) -> list[dict[str, str]]:
        out: list[dict[str, str]] = []
        if self._config.backend == "local":
            # FSM задачи требует отказов вне этапа planning — ломает обычный чат у local-моделей.
            out.extend(self._local_backend_system_messages())
        else:
            out.extend(self._task_state.build_system_message())
        out.extend(self._invariants.build_system_messages())
        out.extend(self._user_profile.build_system_messages())
        out.extend(self._memory.build_memory_system_messages())
        s = self._state.strategy
        if s == ContextStrategyKind.SLIDING_WINDOW:
            out.extend(build_sliding_api_messages(self._state.messages))
            return _merge_leading_system_messages(out)
        if s == ContextStrategyKind.STICKY_FACTS:
            out.extend(
                build_sticky_facts_api_messages(self._state.facts, self._state.messages)
            )
            return _merge_leading_system_messages(out)
        out.extend(build_branching_api_messages(self._state))
        return _merge_leading_system_messages(out)

    def _complete_chat(
        self,
        access_token: str,
        model: str,
        messages: list[dict[str, str]],
    ) -> tuple[str, dict[str, Any]]:
        payload: dict[str, Any] = {
            "model": model,
            "messages": list(messages),
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
        ctx = _ssl_context_for_url(url)
        with urllib.request.urlopen(
            req, timeout=self._config.timeout_sec, context=ctx
        ) as resp:
            raw = resp.read().decode("utf-8")
        response_json: dict[str, Any] = json.loads(raw)
        choice = (response_json.get("choices") or [{}])[0]
        msg = choice.get("message") or {}
        content = msg.get("content")
        if isinstance(content, str):
            reply = content.strip()
        elif content is not None:
            reply = str(content).strip()
        else:
            reply = ""
        if not reply:
            reasoning = msg.get("reasoning_content")
            if isinstance(reasoning, str) and reasoning.strip():
                reply = reasoning.strip()
        if not reply:
            reply = raw[:2000].strip()
        return reply, response_json

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
            if self._config.backend == "local":
                raise RuntimeError(
                    "Список моделей пуст. Запустите LM Studio Server (lms server start) "
                    "и укажите LLM_AGENT_LOCAL_MODEL или --local-model."
                )
            raise RuntimeError(
                "Список моделей пуст. Укажите GIGACHAT_MODEL в окружении."
            )
        self._resolved_model = str(models[0]["id"])
        return self._resolved_model

    def _tokens_count(
        self,
        access_token: str,
        model: str,
        input_strings: list[str],
    ) -> int | None:
        """POST /api/v1/tokens/count — оценка токенов по списку строк input."""
        if not input_strings:
            return None
        url = f"{self._config.api_base}/tokens/count"
        payload: dict[str, Any] = {"model": model, "input": input_strings}
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=body,
            headers=self._bearer_headers(access_token),
            method="POST",
        )
        try:
            ctx = _ssl_context_for_url(url)
            with urllib.request.urlopen(
                req, timeout=min(60.0, self._config.timeout_sec), context=ctx
            ) as resp:
                raw = resp.read().decode("utf-8")
            data: Any = json.loads(raw)
        except (OSError, urllib.error.HTTPError, json.JSONDecodeError):
            return None
        return _parse_tokens_count_response(data)

    def _resolve_access_token(self) -> tuple[str | None, RunResult | None]:
        if self._config.backend == "local":
            return (
                self._effective_authorization_key() or DEFAULT_LOCAL_API_KEY,
                None,
            )
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
                None,
                RunResult(
                    text=(
                        "Добавьте GIGACHAT_API_KEY: файл .env рядом с agent.py "
                        "или export GIGACHAT_API_KEY=...\n"
                        f"  Ожидаемый путь: {primary} (есть на диске: {'да' if primary.is_file() else 'нет'})\n"
                        f"  Размер файла: {size_s} байт\n"
                        "  Ключ авторизации: личный кабинет Studio → проект GigaChat API → Настройки API → Получить ключ\n"
                        "  Документация: https://developers.sber.ru/docs/ru/gigachat/quickstart/ind-using-api"
                        f"{extra}"
                    )
                ),
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
            return None, RunResult(text=msg + _ssl_troubleshooting_hint(e))
        except urllib.error.HTTPError as e:
            err_body = e.read().decode("utf-8", errors="replace")
            return None, RunResult(text=f"Ошибка OAuth HTTP {e.code}: {err_body[:2000]}")
        except (json.JSONDecodeError, RuntimeError, KeyError, TypeError) as e:
            return None, RunResult(text=f"Ошибка разбора ответа OAuth: {e}")
        return access_token, None

    def run(self, user_message: str, *, rag: bool | None = None) -> RunResult:
        """
        Основной вход агента: один пользовательский запрос → текст ответа модели и статистика токенов.
        rag=None — использовать режим агента (set_rag / LLM_AGENT_RAG); True/False — принудительно
        с RAG или без для этого вызова.
        """
        text = (user_message or "").strip()
        if not text:
            return RunResult(text="")

        access_token, auth_err = self._resolve_access_token()
        if auth_err is not None:
            return auth_err
        assert access_token is not None

        try:
            model = self._ensure_model(access_token)
        except OSError as e:
            msg = f"Сетевая ошибка при запросе списка моделей: {e}"
            return RunResult(text=msg + _ssl_troubleshooting_hint(e))
        except urllib.error.HTTPError as e:
            err_body = e.read().decode("utf-8", errors="replace")
            return RunResult(text=f"Ошибка HTTP {e.code} (models): {err_body[:2000]}")
        except RuntimeError as e:
            return RunResult(text=str(e))

        facts_backup = dict(self._state.facts)

        if (
            self._state.strategy == ContextStrategyKind.BRANCHING
            and self._state.branching_split
        ):
            self._state.branches.setdefault(self._state.active_branch, []).append(
                {"role": "user", "content": text}
            )
        else:
            self._state.messages.append({"role": "user", "content": text})

        if self._state.strategy in (
            ContextStrategyKind.SLIDING_WINDOW,
            ContextStrategyKind.STICKY_FACTS,
        ):
            trim_messages(self._state.messages, RECENT_MESSAGE_WINDOW)

        if self._state.strategy == ContextStrategyKind.STICKY_FACTS:
            try:

                def _complete_facts(msgs: list[dict[str, str]]) -> str:
                    r, _ = self._complete_chat(access_token, model, msgs)
                    return r

                self._state.facts = update_facts_with_llm(
                    _complete_facts,
                    self._state.facts,
                    self._state.messages,
                    text,
                )
            except (
                OSError,
                urllib.error.HTTPError,
                RuntimeError,
                KeyError,
                TypeError,
                json.JSONDecodeError,
            ):
                self._rollback_last_user()
                self._state.facts = facts_backup
                return RunResult(
                    text="Не удалось обновить блок facts (ошибка сети или API)."
                )

        self._trim_branching_linear()

        use_rag = self._rag_enabled if rag is None else bool(rag)
        api_user_content: str | None = None
        rag_bundle: dict[str, Any] | None = None
        rag_aug_for_check = None
        if use_rag:
            idx = self._rag_index_path
            if idx is None:
                self._rollback_last_user()
                self._state.facts = facts_backup
                return RunResult(
                    text=(
                        "RAG включён, но путь к индексу не задан. Задайте LLM_AGENT_RAG_INDEX, "
                        "аргумент rag_index_path при создании агента, --rag-index в CLI или /rag on <путь>."
                    )
                )
            try:
                rag_aug = augment_user_message_with_rag(
                    text,
                    idx,
                    top_k=self._rag_top_k,
                    config=self._rag_retrieval_config,
                )
                api_user_content = rag_aug.prompt
                _cids: list[str] = []
                for _h in rag_aug.hits:
                    _md = _h.get("metadata")
                    _cids.append(
                        str(_md.get("chunk_id") or "")
                        if isinstance(_md, dict)
                        else ""
                    )
                rag_aug_for_check = rag_aug
                rag_bundle = {
                    "context_sufficient": rag_aug.context_sufficient,
                    "weak_reason": rag_aug.weak_reason,
                    "best_retrieval_score": rag_aug.best_score,
                    "chunk_ids": _cids,
                }
            except (OSError, ValueError, RuntimeError, json.JSONDecodeError, TypeError) as e:
                self._rollback_last_user()
                self._state.facts = facts_backup
                return RunResult(text=f"Ошибка RAG: {e}")

        turn_for_token_count = api_user_content if api_user_content is not None else text
        user_turn_tokens = self._tokens_count(access_token, model, [turn_for_token_count])
        api_messages = self._build_messages_for_api()
        if api_user_content is not None:
            api_messages = _replace_last_user_message_content(
                api_messages, api_user_content
            )
        dialog_input_tokens = self._tokens_count(
            access_token, model, [m["content"] for m in api_messages]
        )

        try:
            reply, response_json = self._complete_chat(access_token, model, api_messages)
        except urllib.error.HTTPError as e:
            self._rollback_last_user()
            self._state.facts = facts_backup
            err_body = e.read().decode("utf-8", errors="replace")
            detail = err_body
            try:
                err_json: dict[str, Any] = json.loads(err_body)
                detail = str(
                    (err_json.get("error") or {}).get("message") or err_body
                )
            except (json.JSONDecodeError, TypeError, AttributeError):
                pass
            return RunResult(text=f"Ошибка HTTP {e.code}: {detail[:2000]}")
        except OSError as e:
            self._rollback_last_user()
            self._state.facts = facts_backup
            msg = f"Сетевая ошибка: {e}"
            return RunResult(text=msg + _ssl_troubleshooting_hint(e))
        except (json.JSONDecodeError, KeyError, IndexError, TypeError) as e:
            self._rollback_last_user()
            self._state.facts = facts_backup
            return RunResult(text=f"Не удалось разобрать ответ API: {e}")

        usage = _parse_usage_from_completion(response_json)
        stats = TokenStats(
            user_turn_tokens=user_turn_tokens,
            dialog_input_tokens=dialog_input_tokens,
            completion_tokens=usage["completion_tokens"],
            total_tokens=usage["total_tokens"],
            precached_prompt_tokens=usage["precached_prompt_tokens"],
            prompt_tokens_usage=usage["prompt_tokens"],
        )
        if (
            self._state.strategy == ContextStrategyKind.BRANCHING
            and self._state.branching_split
        ):
            self._state.branches.setdefault(self._state.active_branch, []).append(
                {"role": "assistant", "content": reply}
            )
        else:
            self._state.messages.append({"role": "assistant", "content": reply})

        if self._state.strategy in (
            ContextStrategyKind.SLIDING_WINDOW,
            ContextStrategyKind.STICKY_FACTS,
        ):
            trim_messages(self._state.messages, RECENT_MESSAGE_WINDOW)

        self._trim_branching_after_split()
        self._persist_history()
        rag_out: dict[str, Any] | None = None
        rag_lines: list[str] | None = None
        if rag_bundle is not None and rag_aug_for_check is not None:
            chk = validate_rag_grounding_reply(
                reply,
                rag_aug_for_check.hits,
                context_sufficient=rag_aug_for_check.context_sufficient,
            )
            rag_out = {**rag_bundle, "grounding": asdict(chk)}
            rag_lines = format_rag_hit_lines(rag_aug_for_check.hits)
        return RunResult(text=reply, stats=stats, rag=rag_out, rag_source_lines=rag_lines)
