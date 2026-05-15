"""Шаблоны промптов: рендер RAG и команда /prompt."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from agent import LLMAgent
from document_index.rag import (
    RagRetrievalOutcome,
    build_rag_grounded_user_message,
    retrieve_for_rag,
)
from prompt_templates import (
    DEFAULT_RAG_USER_TEMPLATE,
    render_local_system_message,
    render_rag_user_prompt,
)

_REPO = Path(__file__).resolve().parents[1]
_INDEX = _REPO / "memory" / "index_out" / "index_structure.json"
_EXAMPLE_RAG = _REPO / "memory" / "prompts" / "rag_grounded.txt"


class TestPromptTemplates(unittest.TestCase):
    def test_default_template_has_sections(self) -> None:
        if not _INDEX.is_file():
            self.skipTest("нет индекса")
        out = retrieve_for_rag("SVO DME VKO", _INDEX, top_k=2)
        prompt = build_rag_grounded_user_message("SVO?", out)
        self.assertIn("### Ответ", prompt)
        self.assertIn("SVO?", prompt)

    def test_custom_template_placeholders(self) -> None:
        outcome = RagRetrievalOutcome(
            hits=[],
            context_sufficient=False,
            best_score=0.0,
            weak_reason="test",
        )
        tpl = "CASE: {{CONTEXT_RULES}}\nQ: {{QUESTION}}\nE: {{EXCERPTS}}"
        text = render_rag_user_prompt("мой вопрос", outcome, template=tpl)
        self.assertIn("мой вопрос", text)
        self.assertIn("не знаешь", text.lower())
        self.assertIn("CASE:", text)

    def test_load_rag_file_via_agent(self) -> None:
        if not _EXAMPLE_RAG.is_file():
            self.skipTest("нет примера шаблона")
        agent = LLMAgent(rag_prompt_file=_EXAMPLE_RAG)
        self.assertEqual(agent.prompt_overrides.rag_template_path, _EXAMPLE_RAG.resolve())
        self.assertIn("Coral Travel", agent.prompt_overrides.rag_template_text or "")

    def test_prompt_slash_load(self) -> None:
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as f:
            f.write("Только вопрос: {{QUESTION}}\n{{CONTEXT_RULES}}")
            f.flush()
            path = f.name
        try:
            agent = LLMAgent()
            from prompt_templates import handle_prompt_slash_command

            handle_prompt_slash_command(agent, f"rag load {path}")
            self.assertIn("Только вопрос", agent.prompt_overrides.rag_template_text or "")
            handle_prompt_slash_command(agent, "rag reset")
            self.assertIsNone(agent.prompt_overrides.rag_template_text)
        finally:
            Path(path).unlink(missing_ok=True)

    def test_local_system_model_placeholder(self) -> None:
        msg = render_local_system_message("test-model", "Модель: {{MODEL}}")
        self.assertIn("test-model", msg)
        self.assertNotIn("{{MODEL}}", msg)

    def test_default_rag_template_constant(self) -> None:
        self.assertIn("{{QUESTION}}", DEFAULT_RAG_USER_TEMPLATE)


if __name__ == "__main__":
    unittest.main()
