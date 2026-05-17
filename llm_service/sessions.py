"""
In-memory сессии чата (для POST /api/chat). История на диск опционально через LLM_AGENT_HISTORY_FILE.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import Any

from agent import LLMAgent


@dataclass
class SessionEntry:
    agent: LLMAgent
    created_at: float = field(default_factory=time.time)
    last_used: float = field(default_factory=time.time)


class SessionStore:
    def __init__(
        self,
        *,
        ttl_sec: int,
        max_sessions: int,
        history_dir: Path | None,
        agent_factory: Any,
    ) -> None:
        self._ttl = max(60, int(ttl_sec))
        self._max = max(1, int(max_sessions))
        self._history_dir = history_dir
        self._agent_factory = agent_factory
        self._sessions: dict[str, SessionEntry] = {}
        self._lock = Lock()

    def _history_path(self, session_id: str) -> Path | None:
        if self._history_dir is None:
            return None
        self._history_dir.mkdir(parents=True, exist_ok=True)
        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in session_id)
        return self._history_dir / f"session_{safe}.json"

    def _new_agent(self, session_id: str) -> LLMAgent:
        import os

        hist = self._history_path(session_id)
        if hist is not None:
            os.environ["LLM_AGENT_HISTORY_FILE"] = str(hist)
        return self._agent_factory()

    def create(self) -> str:
        sid = uuid.uuid4().hex
        with self._lock:
            self._evict_expired_locked()
            if len(self._sessions) >= self._max:
                oldest = min(
                    self._sessions.items(), key=lambda x: x[1].last_used
                )[0]
                del self._sessions[oldest]
            self._sessions[sid] = SessionEntry(agent=self._new_agent(sid))
        return sid

    def get(self, session_id: str) -> LLMAgent | None:
        with self._lock:
            self._evict_expired_locked()
            entry = self._sessions.get(session_id)
            if entry is None:
                return None
            entry.last_used = time.time()
            return entry.agent

    def delete(self, session_id: str) -> bool:
        with self._lock:
            return self._sessions.pop(session_id, None) is not None

    def _evict_expired_locked(self) -> None:
        now = time.time()
        expired = [
            sid
            for sid, e in self._sessions.items()
            if now - e.last_used > self._ttl
        ]
        for sid in expired:
            del self._sessions[sid]

    def stats(self) -> dict[str, int]:
        with self._lock:
            self._evict_expired_locked()
            return {"active_sessions": len(self._sessions)}
