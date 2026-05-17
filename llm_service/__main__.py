"""python -m llm_service — запуск uvicorn."""

from __future__ import annotations

import argparse

from llm_service.config import ServiceConfig


def main() -> None:
    p = argparse.ArgumentParser(description="Приватный HTTP-сервис локальной LLM")
    p.add_argument("--host", default=None, help="Переопределить LLM_SERVICE_HOST")
    p.add_argument("--port", type=int, default=None, help="Переопределить LLM_SERVICE_PORT")
    args = p.parse_args()

    cfg = ServiceConfig.from_env()
    host = args.host or cfg.host
    port = args.port or cfg.port

    import uvicorn

    uvicorn.run(
        "llm_service.app:app",
        host=host,
        port=port,
        log_level="info",
    )


if __name__ == "__main__":
    main()
