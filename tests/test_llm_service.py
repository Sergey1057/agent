"""Тесты HTTP-сервиса: rate limit, контекст, сессии (без реальной LLM)."""

from __future__ import annotations

import unittest

from fastapi.testclient import TestClient

from agent import RunResult
from llm_service.app import create_app
from llm_service.config import ServiceConfig


class _FakeAgent:
    backend = "local"

    def backend_status_line(self) -> str:
        return "Backend: local (test)"

    def generation_status_line(self) -> str:
        return "Генерация: test"

    def run(self, user_message: str, *, rag: bool | None = None) -> RunResult:
        return RunResult(text=f"echo:{user_message[:40]}")

    def fetch_models_json(self) -> dict:
        return {"data": [{"id": "test-model"}]}

    def complete_messages_stateless(
        self,
        messages: list[dict[str, str]],
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> tuple[str, dict]:
        last = messages[-1]["content"] if messages else ""
        return f"stateless:{last[:20]}", {"model": model or "test-model", "usage": {}}


def _make_client(**overrides: object) -> TestClient:
    base = {
        "host": "127.0.0.1",
        "port": 8080,
        "api_key": "test-key",
        "require_api_key": True,
        "rate_limit_rpm": 5,
        "max_concurrent": 2,
        "max_context_messages": 10,
        "max_message_chars": 1000,
        "session_ttl_sec": 3600,
        "max_sessions": 10,
        "history_dir": None,
    }
    base.update(overrides)
    cfg = ServiceConfig(**base)  # type: ignore[arg-type]
    app = create_app(
        service_config=cfg,
        agent_factory=lambda: _FakeAgent(),  # type: ignore[return-value]
    )
    return TestClient(app)


class TestLlmService(unittest.TestCase):
    def test_health_is_public(self) -> None:
        client = _make_client()
        r = client.get("/health")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["status"], "ok")

    def test_chat_requires_key(self) -> None:
        client = _make_client()
        r = client.post("/api/chat", json={"message": "hi"})
        self.assertEqual(r.status_code, 401)

    def test_chat_creates_session(self) -> None:
        client = _make_client()
        r = client.post(
            "/api/chat",
            json={"message": "привет"},
            headers={"Authorization": "Bearer test-key"},
        )
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertIn("session_id", body)
        self.assertTrue(body["reply"].startswith("echo:"))

    def test_rate_limit_returns_429(self) -> None:
        client = _make_client(rate_limit_rpm=3)
        headers = {"Authorization": "Bearer test-key"}
        codes = [client.get("/api/ping", headers=headers).status_code for _ in range(6)]
        self.assertIn(429, codes)

    def test_max_context_messages(self) -> None:
        client = _make_client(max_context_messages=3)
        messages = [{"role": "user", "content": f"m{i}"} for i in range(5)]
        r = client.post(
            "/v1/chat/completions",
            json={"messages": messages},
            headers={"Authorization": "Bearer test-key"},
        )
        self.assertEqual(r.status_code, 400)

    def test_max_message_chars(self) -> None:
        client = _make_client(max_message_chars=50)
        r = client.post(
            "/api/chat",
            json={"message": "x" * 100},
            headers={"Authorization": "Bearer test-key"},
        )
        self.assertEqual(r.status_code, 400)

    def test_sequential_chat_requests(self) -> None:
        client = _make_client(max_concurrent=2, rate_limit_rpm=100)
        headers = {"Authorization": "Bearer test-key"}
        codes = [
            client.post(
                "/api/chat", json={"message": f"n{i}"}, headers=headers
            ).status_code
            for i in range(4)
        ]
        self.assertEqual(codes, [200, 200, 200, 200])


if __name__ == "__main__":
    unittest.main()
