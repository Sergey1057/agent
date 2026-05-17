"""
Простой rate limiter (скользящее окно) и семафор параллельных запросов к LLM.
"""

from __future__ import annotations

import asyncio
import time
from collections import defaultdict, deque
from typing import Deque


class RateLimitExceeded(Exception):
    def __init__(self, retry_after_sec: float) -> None:
        self.retry_after_sec = retry_after_sec
        super().__init__(f"rate limit exceeded, retry after {retry_after_sec:.1f}s")


class RateLimiter:
    """Лимит запросов на ключ (IP или api-key) за скользящее окно."""

    def __init__(self, max_requests: int, window_sec: float = 60.0) -> None:
        self._max = max(1, int(max_requests))
        self._window = max(1.0, float(window_sec))
        self._events: dict[str, Deque[float]] = defaultdict(deque)
        self._lock = asyncio.Lock()

    async def check(self, key: str) -> None:
        now = time.monotonic()
        async with self._lock:
            q = self._events[key]
            cutoff = now - self._window
            while q and q[0] < cutoff:
                q.popleft()
            if len(q) >= self._max:
                retry = self._window - (now - q[0]) if q else self._window
                raise RateLimitExceeded(max(0.5, retry))
            q.append(now)


class ConcurrencyGate:
    """Ограничение одновременных вызовов LLM (стабильность под нагрузкой)."""

    def __init__(self, max_concurrent: int) -> None:
        self._sem = asyncio.Semaphore(max(1, int(max_concurrent)))

    async def __aenter__(self) -> ConcurrencyGate:
        await self._sem.acquire()
        return self

    async def __aexit__(self, *args: object) -> None:
        self._sem.release()
