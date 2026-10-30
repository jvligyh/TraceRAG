from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tracerag.providers.deepseek import DeepSeekLLMClient, extract_chat_content  # noqa: E402


class DeepSeekLLMClientTests(unittest.TestCase):
    def test_default_model_matches_provider_documentation(self) -> None:
        client = DeepSeekLLMClient("test-key")
        self.assertEqual("deepseek-flash", client.model)

    def test_extract_chat_content(self) -> None:
        response = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="  grounded answer  "))]
        )
        self.assertEqual("grounded answer", extract_chat_content(response))

    def test_empty_key_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            DeepSeekLLMClient("   ")

    def test_reasoning_is_not_exposed_as_final_answer(self) -> None:
        response=SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(
            content=None,reasoning_content="internal reasoning"))])
        with self.assertRaises(RuntimeError):
            extract_chat_content(response)


if __name__ == "__main__":
    unittest.main()
