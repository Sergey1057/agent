"""Ассистент разработчика: корпус документации, MCP project, /help."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from dev_assistant import (
    build_project_docs_index,
    default_project_index_path,
    default_project_root,
)
from document_index.corpus import collect_corpus_paths

_REPO = Path(__file__).resolve().parents[1]
_AIADVENT = Path("/Users/sergei/Documents/ai-course/AiAdvent1")


class TestCorpusCollect(unittest.TestCase):
    def test_collect_aiadvent_has_readme_and_docs(self) -> None:
        if not _AIADVENT.is_dir():
            self.skipTest(f"Нет проекта: {_AIADVENT}")
        paths = collect_corpus_paths(_AIADVENT)
        names = {p.name for p in paths}
        self.assertIn("REDME.md", names)
        self.assertTrue(any(p.parent.name == "docs" for p in paths))

    def test_build_index_dummy(self) -> None:
        if not _AIADVENT.is_dir():
            self.skipTest(f"Нет проекта: {_AIADVENT}")
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "project_index.json"
            meta = build_project_docs_index(_AIADVENT, out_path=out, dummy_embeddings=True)
            self.assertTrue(out.is_file())
            data = json.loads(out.read_text(encoding="utf-8"))
            self.assertGreater(len(data.get("chunks") or []), 0)
            self.assertEqual(meta.get("strategy"), "structure")


class TestProjectMcp(unittest.TestCase):
    def test_git_branch_on_llm_agent_repo(self) -> None:
        from project_mcp_server import project_git_branch

        r = project_git_branch(str(_REPO))
        self.assertEqual(r.get("status"), "ok")
        self.assertTrue(r.get("is_git"))
        self.assertTrue((r.get("branch") or "").strip())

    def test_list_files(self) -> None:
        from project_mcp_server import project_git_list_files

        r = project_git_list_files(str(_REPO), subpath="document_index", max_files=10)
        self.assertEqual(r.get("status"), "ok")
        self.assertGreater(r.get("file_count", 0), 0)


class TestHelpSlash(unittest.TestCase):
    def test_help_with_question_calls_dev_assistant(self) -> None:
        from cli import _handle_slash_command

        agent = mock.MagicMock()
        with mock.patch(
            "dev_assistant.answer_project_help",
            return_value="ответ про структуру",
        ) as m:
            out = _handle_slash_command(agent, "/help структура проекта")
        self.assertEqual(out, "ответ про структуру")
        m.assert_called_once()

    def test_help_without_arg_returns_commands(self) -> None:
        from cli import _handle_slash_command

        agent = mock.MagicMock()
        out = _handle_slash_command(agent, "/help")
        self.assertIn("/help <вопрос", out or "")


class TestDefaults(unittest.TestCase):
    def test_default_project_root_env(self) -> None:
        with mock.patch.dict(os.environ, {"LLM_AGENT_PROJECT_ROOT": "/tmp/proj"}):
            self.assertEqual(str(default_project_root()), "/private/tmp/proj")

    def test_default_index_under_memory(self) -> None:
        p = default_project_index_path()
        self.assertIn("project_docs_index.json", p.name)


if __name__ == "__main__":
    unittest.main()
