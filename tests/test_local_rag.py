"""Локальный RAG: индекс по умолчанию, retrieval без облака."""

from __future__ import annotations

import os
import unittest
from pathlib import Path
from unittest import mock

from document_index.rag import (
    DEFAULT_RAG_INDEX_REL,
    default_rag_index_path,
    retrieve_for_rag,
)

_REPO = Path(__file__).resolve().parents[1]
_INDEX = _REPO / "memory" / "index_out" / "index_structure.json"


class TestLocalRag(unittest.TestCase):
    def test_default_index_path_constant(self) -> None:
        self.assertEqual(
            DEFAULT_RAG_INDEX_REL,
            Path("memory/index_out/index_structure.json"),
        )

    def test_default_rag_index_path_resolves(self) -> None:
        if not _INDEX.is_file():
            self.skipTest(f"Нет индекса: {_INDEX}")
        p = default_rag_index_path()
        self.assertIsNotNone(p)
        assert p is not None
        self.assertTrue(p.is_file())
        self.assertEqual(p.name, "index_structure.json")

    def test_retrieve_uses_lexical_for_hash_index_without_openai(self) -> None:
        if not _INDEX.is_file():
            self.skipTest(f"Нет индекса: {_INDEX}")
        with mock.patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test-fake"}, clear=False):
            out = retrieve_for_rag(
                "мобильное приложение Coral Travel установить",
                _INDEX,
                top_k=3,
            )
        self.assertTrue(out.hits)
        self.assertTrue(out.context_sufficient)

    def test_rag_local_env_skips_openai_embed(self) -> None:
        if not _INDEX.is_file():
            self.skipTest(f"Нет индекса: {_INDEX}")
        with mock.patch.dict(
            os.environ,
            {"OPENAI_API_KEY": "sk-test", "LLM_AGENT_RAG_LOCAL": "1"},
            clear=False,
        ):
            with mock.patch(
                "document_index.rag.embed_texts_openai",
                side_effect=AssertionError("OpenAI не должен вызываться"),
            ):
                out = retrieve_for_rag("SVO DME VKO аэропорт", _INDEX, top_k=2)
        self.assertTrue(out.hits)


if __name__ == "__main__":
    unittest.main()
