#!/usr/bin/env python3
"""
MCP-сервер планировщика:
- отложенные и периодические задачи
- хранение состояния и выполнений в JSON
- агрегированная сводка по результатам
"""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("scheduler-server")

_STATE_PATH = (Path(__file__).resolve().parent / "memory" / "scheduler_state.json").resolve()
_DEFAULT_TTL_SECONDS = 3600


def _read_state() -> dict[str, Any]:
    if not _STATE_PATH.is_file():
        return {"version": 1, "tasks": [], "events": []}
    try:
        data = json.loads(_STATE_PATH.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError):
        return {"version": 1, "tasks": [], "events": []}
    if not isinstance(data, dict):
        return {"version": 1, "tasks": [], "events": []}
    tasks = data.get("tasks")
    events = data.get("events")
    return {
        "version": 1,
        "tasks": tasks if isinstance(tasks, list) else [],
        "events": events if isinstance(events, list) else [],
    }


def _write_state(state: dict[str, Any]) -> None:
    _STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = _STATE_PATH.with_name(_STATE_PATH.name + ".tmp")
    tmp.write_text(
        json.dumps(state, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    tmp.replace(_STATE_PATH)


def _coerce_positive_int(value: Any, default: int) -> int:
    try:
        n = int(value)
    except (TypeError, ValueError):
        return default
    return n if n > 0 else default


def _now() -> int:
    return int(time.time())


def _execute_task(task: dict[str, Any], now_ts: int) -> dict[str, Any]:
    kind = str(task.get("kind", "data_collection"))
    payload = task.get("payload")
    if not isinstance(payload, dict):
        payload = {}

    result: dict[str, Any] = {
        "task_id": task.get("id"),
        "task_name": task.get("name"),
        "kind": kind,
        "executed_at": now_ts,
    }
    if kind == "reminder":
        result["message"] = str(payload.get("message") or "Напоминание")
    elif kind == "summary":
        result["message"] = str(payload.get("message") or "Регулярный summary")
    else:
        source = str(payload.get("source") or "unknown")
        synthetic_value = len(source) + (now_ts % 100)
        result["source"] = source
        result["value"] = synthetic_value
        result["message"] = f"Собраны данные из {source}"
    return result


@mcp.tool(description="Создать или обновить задачу расписания.")
def schedule_upsert_task(
    name: str,
    kind: str,
    payload: dict[str, Any] | None = None,
    delay_seconds: int = 0,
    interval_seconds: int = 0,
    active: bool = True,
    max_runs: int = 0,
) -> dict[str, Any]:
    state = _read_state()
    now_ts = _now()
    kind_val = (kind or "").strip().lower()
    if kind_val not in ("reminder", "data_collection", "summary"):
        return {"status": "error", "error": "kind: reminder | data_collection | summary"}
    if not (name or "").strip():
        return {"status": "error", "error": "name обязателен"}

    task_name = name.strip()
    delay = max(0, int(delay_seconds or 0))
    interval = max(0, int(interval_seconds or 0))
    runs_limit = max(0, int(max_runs or 0))
    next_run_at = now_ts + delay
    body = payload if isinstance(payload, dict) else {}

    tasks: list[dict[str, Any]] = state["tasks"]
    existing = next((t for t in tasks if t.get("name") == task_name), None)
    if existing is None:
        existing = {
            "id": str(uuid.uuid4()),
            "name": task_name,
            "kind": kind_val,
            "payload": body,
            "interval_seconds": interval,
            "next_run_at": next_run_at,
            "active": bool(active),
            "run_count": 0,
            "max_runs": runs_limit,
            "created_at": now_ts,
            "updated_at": now_ts,
        }
        tasks.append(existing)
    else:
        existing["kind"] = kind_val
        existing["payload"] = body
        existing["interval_seconds"] = interval
        existing["next_run_at"] = next_run_at
        existing["active"] = bool(active)
        existing["max_runs"] = runs_limit
        existing["updated_at"] = now_ts

    _write_state(state)
    return {"status": "ok", "task": existing}


@mcp.tool(description="Список задач расписания.")
def schedule_list_tasks(include_inactive: bool = False) -> dict[str, Any]:
    state = _read_state()
    tasks = state["tasks"]
    if not include_inactive:
        tasks = [t for t in tasks if bool(t.get("active", True))]
    tasks_sorted = sorted(tasks, key=lambda x: int(x.get("next_run_at", 0)))
    return {"status": "ok", "count": len(tasks_sorted), "tasks": tasks_sorted}


@mcp.tool(description="Запустить все due-задачи и вернуть агрегированный результат.")
def schedule_run_due(limit: int = 20) -> dict[str, Any]:
    state = _read_state()
    now_ts = _now()
    max_items = _coerce_positive_int(limit, 20)
    tasks: list[dict[str, Any]] = state["tasks"]
    events: list[dict[str, Any]] = state["events"]

    due = [
        t
        for t in tasks
        if bool(t.get("active", True)) and int(t.get("next_run_at", 0)) <= now_ts
    ]
    due = sorted(due, key=lambda x: int(x.get("next_run_at", 0)))[:max_items]

    executed_results: list[dict[str, Any]] = []
    by_kind: dict[str, int] = {"reminder": 0, "data_collection": 0, "summary": 0}
    for task in due:
        res = _execute_task(task, now_ts)
        executed_results.append(res)
        kind = str(task.get("kind") or "data_collection")
        by_kind[kind] = by_kind.get(kind, 0) + 1

        task["run_count"] = int(task.get("run_count", 0)) + 1
        task["last_run_at"] = now_ts

        interval = int(task.get("interval_seconds", 0))
        max_runs = int(task.get("max_runs", 0))
        if interval > 0 and (max_runs == 0 or task["run_count"] < max_runs):
            task["next_run_at"] = now_ts + interval
        else:
            task["active"] = False

        events.append(
            {
                "id": str(uuid.uuid4()),
                "task_id": task.get("id"),
                "task_name": task.get("name"),
                "kind": kind,
                "executed_at": now_ts,
                "result": res,
            }
        )

    ttl = now_ts - _DEFAULT_TTL_SECONDS * 24
    state["events"] = [e for e in events if int(e.get("executed_at", 0)) >= ttl]
    _write_state(state)

    return {
        "status": "ok",
        "executed_count": len(executed_results),
        "aggregated": {
            "executed_total": len(executed_results),
            "by_kind": by_kind,
        },
        "items": executed_results,
    }


@mcp.tool(description="Получить агрегированный summary за последние N часов.")
def schedule_get_summary(hours: int = 24) -> dict[str, Any]:
    state = _read_state()
    now_ts = _now()
    window = max(1, int(hours or 24)) * 3600
    threshold = now_ts - window
    events = [e for e in state["events"] if int(e.get("executed_at", 0)) >= threshold]

    by_kind: dict[str, int] = {}
    by_task: dict[str, int] = {}
    for item in events:
        kind = str(item.get("kind") or "unknown")
        task = str(item.get("task_name") or "unknown")
        by_kind[kind] = by_kind.get(kind, 0) + 1
        by_task[task] = by_task.get(task, 0) + 1

    top_tasks = sorted(by_task.items(), key=lambda x: (-x[1], x[0]))[:10]
    latest = sorted(events, key=lambda x: int(x.get("executed_at", 0)), reverse=True)[:10]
    return {
        "status": "ok",
        "hours": int(hours or 24),
        "events_count": len(events),
        "by_kind": by_kind,
        "top_tasks": [{"task_name": k, "runs": v} for k, v in top_tasks],
        "latest_items": latest,
    }


@mcp.tool(description="Получить текстовую сводку за последние N часов.")
def schedule_get_human_summary(hours: int = 24) -> dict[str, Any]:
    summary = schedule_get_summary(hours=hours)
    if summary.get("status") != "ok":
        return summary

    hours_val = int(summary.get("hours", hours))
    events_count = int(summary.get("events_count", 0))
    by_kind = summary.get("by_kind")
    if not isinstance(by_kind, dict):
        by_kind = {}
    top_tasks = summary.get("top_tasks")
    if not isinstance(top_tasks, list):
        top_tasks = []

    reminder_count = int(by_kind.get("reminder", 0) or 0)
    collector_count = int(by_kind.get("data_collection", 0) or 0)
    summary_count = int(by_kind.get("summary", 0) or 0)

    lines = [
        f"Сводка за последние {hours_val}ч:",
        f"- Всего выполнений: {events_count}",
        (
            "- По типам: reminder={0}, data_collection={1}, summary={2}".format(
                reminder_count,
                collector_count,
                summary_count,
            )
        ),
    ]
    if top_tasks:
        top_line = ", ".join(
            f"{str(x.get('task_name', 'unknown'))} ({int(x.get('runs', 0) or 0)})"
            for x in top_tasks[:5]
        )
        lines.append(f"- Топ задач: {top_line}")
    else:
        lines.append("- Топ задач: нет данных.")

    latest = summary.get("latest_items")
    if isinstance(latest, list) and latest:
        sample = latest[0]
        lines.append(
            "Последнее событие: "
            f"{sample.get('task_name', 'unknown')} [{sample.get('kind', 'unknown')}]."
        )

    return {
        "status": "ok",
        "hours": hours_val,
        "summary_text": "\n".join(lines),
        "aggregated": summary,
    }


if __name__ == "__main__":
    mcp.run()
