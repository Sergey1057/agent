"""Параметры генерации: парсинг env и команда /gen."""

from __future__ import annotations

import os
import unittest
from unittest import mock

from agent import LLMAgent
from generation_config import GenerationConfig, handle_gen_slash_command


class TestGenerationConfig(unittest.TestCase):
    def test_from_env_defaults(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=True):
            cfg = GenerationConfig.from_env()
        self.assertIsNone(cfg.temperature)
        self.assertIsNone(cfg.max_tokens)
        self.assertEqual(cfg.recent_message_window, 6)

    def test_from_env_and_cli_override(self) -> None:
        with mock.patch.dict(
            os.environ,
            {
                "LLM_AGENT_TEMPERATURE": "0.5",
                "LLM_AGENT_MAX_TOKENS": "256",
                "LLM_AGENT_CONTEXT_WINDOW": "4",
            },
            clear=False,
        ):
            cfg = GenerationConfig.from_env(
                temperature=0.2,
                max_tokens=512,
                recent_message_window=8,
            )
        self.assertEqual(cfg.temperature, 0.2)
        self.assertEqual(cfg.max_tokens, 512)
        self.assertEqual(cfg.recent_message_window, 8)

    def test_apply_to_payload(self) -> None:
        payload: dict = {"model": "x", "messages": []}
        GenerationConfig(
            temperature=0.1, max_tokens=100, recent_message_window=3
        ).apply_to_payload(payload)
        self.assertEqual(payload["temperature"], 0.1)
        self.assertEqual(payload["max_tokens"], 100)

    def test_gen_slash_commands(self) -> None:
        agent = LLMAgent(
            temperature=0.3,
            max_tokens=200,
            context_window=5,
        )
        self.assertIn("temperature=0.3", handle_gen_slash_command(agent, ""))
        out = handle_gen_slash_command(agent, "temperature 0.15")
        self.assertIn("0.15", out)
        self.assertEqual(agent.generation_config.temperature, 0.15)
        handle_gen_slash_command(agent, "context-window 3")
        self.assertEqual(agent.generation_config.recent_message_window, 3)
        handle_gen_slash_command(agent, "reset")
        self.assertEqual(agent.generation_config.temperature, 0.3)


if __name__ == "__main__":
    unittest.main()
