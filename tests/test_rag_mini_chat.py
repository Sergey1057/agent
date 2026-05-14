"""
Два длинных сценария по 10 сообщений: память задачи (цель/уточнения/термины) не теряется,
RAG даёт строки источников для релевантных вопросов.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from document_index.rag import format_rag_hit_lines, retrieve_for_rag
from rag_mini_chat import dry_run_turn_messages
from task_memory import (
    DialogTaskMemory,
    parse_task_memory_json_from_reply,
    snapshot_dict,
)

_REPO = Path(__file__).resolve().parents[1]
_INDEX = _REPO / "memory" / "index_out" / "index_structure.json"


def _synthetic_assistant(
    *,
    answer_line: str,
    source_line: str,
    quote_line: str,
    goal: str,
    new_clar: str | None = None,
    new_terms: dict[str, str] | None = None,
) -> str:
    clar: list[str] = [new_clar] if new_clar else []
    terms = dict(new_terms or {})
    mem = json.dumps(
        {"goal": goal, "clarifications": clar, "terms": terms},
        ensure_ascii=False,
    )
    return (
        "### Ответ\n"
        f"{answer_line}\n"
        "### Источники\n"
        f"{source_line}\n"
        "### Цитаты\n"
        f"{quote_line}\n"
        "### Память задачи (JSON)\n"
        f"{mem}\n"
    )


class TestRagMiniChatScenarios(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if not _INDEX.is_file():
            raise unittest.SkipTest(f"Нет индекса: {_INDEX}")

    def test_parse_task_memory_json(self) -> None:
        body = _synthetic_assistant(
            answer_line="ok",
            source_line="- x | chunk_id: a",
            quote_line="> ok",
            goal="g1",
            new_clar="c1",
            new_terms={"SVO": "Шереметьево"},
        )
        p = parse_task_memory_json_from_reply(body)
        self.assertIsNotNone(p)
        assert p is not None
        self.assertEqual(p.get("goal"), "g1")
        self.assertEqual(p.get("clarifications"), ["c1"])
        self.assertEqual(p.get("terms"), {"SVO": "Шереметьево"})

    def test_rag_hit_lines_for_documented_questions(self) -> None:
        """На релевантных вопросах retrieval возвращает чанки — строки источников не пустые."""
        questions = [
            "Какое мобильное приложение рекомендуют туристам?",
            "Что такое SVO DME VKO в авиабилете?",
            "Где взять онлайн-табло аэропортов по ссылкам из памятки?",
        ]
        for q in questions:
            with self.subTest(q=q):
                out = retrieve_for_rag(q, _INDEX, top_k=3)
                self.assertTrue(out.context_sufficient, msg=q)
                lines = format_rag_hit_lines(out.hits)
                self.assertTrue(lines)
                self.assertTrue(all(l.startswith("- ") for l in lines))

    def _run_ten_turn_dry(
        self,
        users: list[str],
        assistant_goal: str,
    ) -> tuple[list[str], DialogTaskMemory]:
        self.assertEqual(len(users), 10)
        assistants: list[str] = []
        for i, u in enumerate(users):
            assistants.append(
                _synthetic_assistant(
                    answer_line=f"Кратко по шагу {i + 1}.",
                    source_line="- note.txt | раздел: тест | chunk_id: dummy",
                    quote_line="> тестовая строка цитаты",
                    goal=assistant_goal,
                    new_clar=f"шаг_{i + 1}",
                    new_terms=None,
                )
            )
        with tempfile.TemporaryDirectory() as td:
            tdir = Path(td)
            hist = tdir / "h.json"
            task = tdir / "task.json"
            systems, mem = dry_run_turn_messages(
                history_file=hist,
                task_file=task,
                index_path=_INDEX,
                user_messages=users,
                assistant_replies=assistants,
            )
        return systems, mem

    def test_scenario_a_tourist_memo_ten_messages(self) -> None:
        users = [
            "ЦЕЛЬ: консультация по туристической памятке только из RAG-документа",
            "Какое мобильное приложение рекомендуют туристам?",
            "УТОЧНЕНИЕ: отвечай кратко, без лишних советов вне документа",
            "Что такое SVO DME VKO в авиабилете?",
            "ТЕРМИН: SVO = код аэропорта Шереметьево (для фиксации в памяти задачи)",
            "За сколько часов нужно прибыть в аэропорт на международный рейс?",
            "УТОЧНЕНИЕ: интересуют только правила из памятки, не общие нормы IATA",
            "Где взять онлайн-табло аэропортов по ссылкам из памятки?",
            "Кто такой Coral Travel в тексте памятки?",
            "Напомни цель и последнее уточнение одной строкой.",
        ]
        systems, mem = self._run_ten_turn_dry(
            users,
            "консультация по туристической памятке только из RAG-документа",
        )
        self.assertEqual(len(systems), 10)
        for i, sys_txt in enumerate(systems):
            with self.subTest(turn=i):
                self.assertIn("ПАМЯТЬ ЗАДАЧИ", sys_txt)
                self.assertIn("туристической памятке", sys_txt.lower())
        snap = snapshot_dict(mem)
        self.assertIn("SVO", snap["terms"])
        self.assertGreaterEqual(len(snap["clarifications"]), 8)

    def test_scenario_b_abbreviations_and_constraints_ten_messages(self) -> None:
        users = [
            "ЦЕЛЬ: разобрать аббревиатуры и ограничения из памятки для нового сотрудника",
            "Что означают буквы SVO DME VKO в маршрутной квитанции?",
            "УТОЧНЕНИЕ: сотрудник не летает, ему нужны только расшифровки из текста",
            "ТЕРМИН: DME = Домодедово",
            "ТЕРМИН: VKO = Внуково",
            "Есть ли в памятке рекомендация по мобильному приложению?",
            "УТОЧНЕНИЕ: не предлагай сторонние приложения — только если есть в документе",
            "Шереметьево SVO регистрация — что сказано в памятке?",
            "УТОЧНЕНИЕ: все ответы со списком источников и цитатами",
            "Суммируй зафиксированные термины и цель.",
        ]
        systems, mem = self._run_ten_turn_dry(
            users,
            "разобрать аббревиатуры и ограничения из памятки для нового сотрудника",
        )
        self.assertEqual(len(systems), 10)
        last = systems[-1]
        self.assertIn("аббревиатур", last.lower())
        self.assertIn("DME", mem.terms)
        self.assertIn("VKO", mem.terms)
        self.assertIn("нового сотрудника", last)
