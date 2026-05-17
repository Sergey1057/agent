"""
Приватный HTTP-сервис чата поверх локальной LLM (LM Studio / Ollama / OpenAI-compatible).

Запуск:
  LLM_AGENT_BACKEND=local python -m llm_service

Проверка:
  curl http://127.0.0.1:8080/health
  curl -H "Authorization: Bearer $LLM_SERVICE_API_KEY" \\
    -H "Content-Type: application/json" \\
    -d '{"message":"Привет"}' http://127.0.0.1:8080/api/chat
"""

from __future__ import annotations

import asyncio
import os
import time
from contextlib import asynccontextmanager
from functools import partial
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from agent import LLMAgent, RunResult
from llm_service.config import ServiceConfig
from llm_service.rate_limit import ConcurrencyGate, RateLimitExceeded, RateLimiter
from llm_service.sessions import SessionStore

# Загрузка .env до создания агента
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)
    session_id: str | None = None
    rag: bool | None = None


class ChatResponse(BaseModel):
    reply: str
    session_id: str
    stats: dict[str, Any] | None = None
    rag: dict[str, Any] | None = None


class OpenAIMessage(BaseModel):
    role: str
    content: str


class ChatCompletionsRequest(BaseModel):
    model: str | None = None
    messages: list[OpenAIMessage]
    temperature: float | None = None
    max_tokens: int | None = None
    stream: bool = False


def _client_key(request: Request, cfg: ServiceConfig) -> str:
    if cfg.api_key:
        auth = request.headers.get("Authorization", "")
        if auth.startswith("Bearer "):
            return auth[7:].strip() or "anon"
    forwarded = request.headers.get("X-Forwarded-For", "")
    if forwarded:
        return forwarded.split(",")[0].strip()
    if request.client:
        return request.client.host
    return "unknown"


def _check_api_key(request: Request, cfg: ServiceConfig) -> None:
    if not cfg.require_api_key:
        return
    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer ") or auth[7:].strip() != cfg.api_key:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


def _validate_messages(
    messages: list[dict[str, str]], cfg: ServiceConfig
) -> None:
    if len(messages) > cfg.max_context_messages:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Too many messages: {len(messages)} > "
                f"max {cfg.max_context_messages}"
            ),
        )
    for m in messages:
        content = str(m.get("content", ""))
        if len(content) > cfg.max_message_chars:
            raise HTTPException(
                status_code=400,
                detail=f"Message exceeds max length {cfg.max_message_chars}",
            )


def _run_result_to_stats(result: RunResult) -> dict[str, Any] | None:
    if result.stats is None:
        return None
    s = result.stats
    return {
        "user_turn_tokens": s.user_turn_tokens,
        "dialog_input_tokens": s.dialog_input_tokens,
        "completion_tokens": s.completion_tokens,
        "total_tokens": s.total_tokens,
    }


def create_app(
    service_config: ServiceConfig | None = None,
    agent_factory: Any | None = None,
) -> FastAPI:
    cfg = service_config or ServiceConfig.from_env()
    rate_limiter = RateLimiter(cfg.rate_limit_rpm, window_sec=60.0)
    concurrency = ConcurrencyGate(cfg.max_concurrent)

    def _default_agent_factory() -> LLMAgent:
        os.environ.setdefault("LLM_AGENT_BACKEND", "local")
        return LLMAgent()

    factory = agent_factory or _default_agent_factory
    sessions = SessionStore(
        ttl_sec=cfg.session_ttl_sec,
        max_sessions=cfg.max_sessions,
        history_dir=cfg.history_dir,
        agent_factory=factory,
    )

    # Один shared-агент для stateless /v1/chat/completions
    stateless_agent: LLMAgent | None = None

    def _get_stateless_agent() -> LLMAgent:
        nonlocal stateless_agent
        if stateless_agent is None:
            stateless_agent = factory()
        return stateless_agent

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        yield

    app = FastAPI(
        title="Private Local LLM Service",
        version="1.0.0",
        lifespan=lifespan,
    )

    def get_config() -> ServiceConfig:
        return cfg

    async def _run_blocking(func: Any, *args: Any, **kwargs: Any) -> Any:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, partial(func, *args, **kwargs))

    async def enforce_limits(request: Request) -> None:
        _check_api_key(request, cfg)
        key = _client_key(request, cfg)
        try:
            await rate_limiter.check(key)
        except RateLimitExceeded as e:
            raise HTTPException(
                status_code=429,
                detail="Rate limit exceeded",
                headers={"Retry-After": str(int(e.retry_after_sec) + 1)},
            ) from e

    @app.get("/health")
    async def health() -> dict[str, Any]:
        # Без API key — для docker healthcheck; rate limit не применяем
        agent = _get_stateless_agent()
        return {
            "status": "ok",
            "backend": agent.backend,
            "backend_line": agent.backend_status_line(),
            "generation": agent.generation_status_line(),
            "limits": {
                "rate_limit_rpm": cfg.rate_limit_rpm,
                "max_concurrent": cfg.max_concurrent,
                "max_context_messages": cfg.max_context_messages,
                "max_message_chars": cfg.max_message_chars,
            },
            "sessions": sessions.stats(),
        }

    @app.get("/api/ping")
    async def api_ping(request: Request) -> dict[str, str]:
        """Проверка API key и rate limit без вызова LLM."""
        await enforce_limits(request)
        return {"status": "pong"}

    @app.post("/api/chat", response_model=ChatResponse)
    async def api_chat(
        body: ChatRequest,
        request: Request,
        _cfg: ServiceConfig = Depends(get_config),
    ) -> ChatResponse:
        await enforce_limits(request)
        msg = body.message.strip()
        if len(msg) > cfg.max_message_chars:
            raise HTTPException(
                status_code=400,
                detail=f"Message exceeds max length {cfg.max_message_chars}",
            )

        sid = (body.session_id or "").strip()
        if not sid:
            sid = sessions.create()
        agent = sessions.get(sid)
        if agent is None:
            sid = sessions.create()
            agent = sessions.get(sid)
        assert agent is not None

        async with concurrency:
            result = await _run_blocking(agent.run, msg, rag=body.rag)

        return ChatResponse(
            reply=result.text,
            session_id=sid,
            stats=_run_result_to_stats(result),
            rag=result.rag,
        )

    @app.delete("/api/sessions/{session_id}")
    async def delete_session(session_id: str, request: Request) -> dict[str, bool]:
        await enforce_limits(request)
        ok = sessions.delete(session_id)
        if not ok:
            raise HTTPException(status_code=404, detail="Session not found")
        return {"deleted": True}

    @app.get("/v1/models")
    async def list_models(request: Request) -> dict[str, Any]:
        await enforce_limits(request)
        agent = _get_stateless_agent()
        try:
            return await _run_blocking(agent.fetch_models_json)
        except (OSError, RuntimeError) as e:
            raise HTTPException(
                status_code=503, detail=f"Cannot reach LLM backend: {e}"
            ) from e

    @app.post("/v1/chat/completions")
    async def chat_completions(
        body: ChatCompletionsRequest,
        request: Request,
    ) -> dict[str, Any]:
        await enforce_limits(request)
        if body.stream:
            raise HTTPException(
                status_code=400, detail="stream=true is not supported"
            )

        messages = [{"role": m.role, "content": m.content} for m in body.messages]
        _validate_messages(messages, cfg)

        agent = _get_stateless_agent()
        try:
            async with concurrency:
                reply, response_json = await _run_blocking(
                    agent.complete_messages_stateless,
                    messages,
                    model=body.model,
                    temperature=body.temperature,
                    max_tokens=body.max_tokens,
                )
        except RuntimeError as e:
            raise HTTPException(status_code=503, detail=str(e)) from e
        except OSError as e:
            raise HTTPException(status_code=502, detail=str(e)) from e

        model = str(
            body.model or response_json.get("model") or "unknown"
        )
        usage = response_json.get("usage") or {}
        return {
            "id": f"chatcmpl-{int(time.time())}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": reply},
                    "finish_reason": "stop",
                }
            ],
            "usage": usage,
        }

    @app.exception_handler(HTTPException)
    async def http_exception_handler(
        _request: Request, exc: HTTPException
    ) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": {"message": exc.detail, "type": "http_error"}},
            headers=getattr(exc, "headers", None) or {},
        )

    return app


app = create_app()
