"""
Проверка RAG: обязательные источники/цитаты в промпте, режим «не знаю» при слабом контексте,
валидация структуры ответа (без вызова LLM).
"""

from __future__ import annotations

import unittest
from pathlib import Path

from document_index.rag import (
    build_rag_grounded_user_message,
    retrieve_for_rag,
    validate_rag_grounding_reply,
)

_REPO = Path(__file__).resolve().parents[1]
_INDEX = _REPO / "memory" / "index_out" / "index_structure.json"


class TestRagGrounding(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if not _INDEX.is_file():
            raise unittest.SkipTest(f"Нет индекса для теста: {_INDEX}")

    def test_ten_questions_prompt_and_gate(self) -> None:
        questions: list[tuple[str, str]] = [
            ("Какое мобильное приложение рекомендуют туристам?", "on_doc"),
            ("Что такое SVO DME VKO в авиабилете?", "on_doc"),
            ("За сколько часов нужно прибыть в аэропорт на международный рейс?", "on_doc"),
            ("Где взять онлайн-табло аэропортов по ссылкам из памятки?", "on_doc"),
            ("Кто такой Coral Travel в тексте памятки?", "on_doc"),
            ("Квантовая гравитация и уравнения Эйнштейна в памятке туриста", "off_doc"),
            ("asdfghjkl zxcvbnm qwertyuiop несуществующий запрос", "off_doc"),
            ("Напиши рецепт борща из документа", "off_doc"),
            ("Какой API ключ нужен для OpenAI GPT-5 в этом PDF", "off_doc"),
            ("Шереметьево SVO регистрация", "on_doc"),
        ]
        self.assertEqual(len(questions), 10)

        for q, kind in questions:
            with self.subTest(q=q, kind=kind):
                out = retrieve_for_rag(q, _INDEX, top_k=3)
                prompt = build_rag_grounded_user_message(q, out)
                self.assertIn("### Ответ", prompt)
                self.assertIn("### Источники", prompt)
                self.assertIn("### Цитаты", prompt)

                if out.context_sufficient:
                    self.assertIn("chunk_id:", prompt)
                    self.assertIn("--- Выдержка", prompt)
                    self.assertNotIn("намеренно НЕ приводятся", prompt)
                else:
                    self.assertIn("намеренно НЕ приводятся", prompt)
                    self.assertIn("не знаешь", prompt.lower())

                if kind == "off_doc":
                    self.assertFalse(
                        out.context_sufficient,
                        msg="Ожидался слабый контекст для off_doc вопроса",
                    )
                if kind == "on_doc":
                    self.assertTrue(
                        out.context_sufficient,
                        msg="Ожидался достаточный контекст для on_doc вопроса",
                    )

    def test_validate_strong_synthetic_reply(self) -> None:
        out = retrieve_for_rag(
            "SVO DME VKO аббревиатуры аэропортов", _INDEX, top_k=2
        )
        self.assertTrue(out.context_sufficient)
        h0 = out.hits[0]
        meta = h0.get("metadata") if isinstance(h0.get("metadata"), dict) else {}
        cid = str(meta.get("chunk_id") or "").strip()
        snippet = (h0.get("text") or "")[:200].strip()
        self.assertTrue(cid and snippet)
        reply = f"""### Ответ
В маршрутной квитанции аэропорты Москвы кодируются аббревиатурами SVO, DME, VKO.

### Источники
- note.txt | раздел: Авиаперелет. | chunk_id: {cid}

### Цитаты
> [{cid}] {snippet}
"""
        chk = validate_rag_grounding_reply(
            reply, out.hits, context_sufficient=True
        )
        self.assertTrue(chk.has_answer_section)
        self.assertTrue(chk.has_sources_section)
        self.assertTrue(chk.has_quotes_section)
        self.assertTrue(chk.quotes_nonempty)
        self.assertTrue(chk.quotes_verbatim_in_chunks)

    def test_validate_weak_synthetic_reply(self) -> None:
        reply = """### Ответ
По этим документам я не знаю надёжного ответа; уточните, пожалуйста, о каком разделе памятки речь.

### Источники
- нет надёжных источников — релевантность ниже порога

### Цитаты
нет — цитировать нечего (контекст недостаточен)
"""
        chk = validate_rag_grounding_reply(
            reply, [], context_sufficient=False
        )
        self.assertTrue(chk.has_answer_section)
        self.assertTrue(chk.has_sources_section)
        self.assertTrue(chk.has_quotes_section)
        self.assertIn("weak_context", chk.notes)


if __name__ == "__main__":
    unittest.main()
