from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tracerag import Document, TraceRAGPipeline  # noqa: E402
from tracerag.providers import MockLLMClient  # noqa: E402


class RecordingClient:
    def __init__(self) -> None:
        self.prompt = ""

    def complete(self, prompt: str) -> str:
        self.prompt = prompt
        return "中文回答"


class TraceRAGPipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.pipeline = TraceRAGPipeline(MockLLMClient())
        self.documents = [
            Document(document_id="doc-a", title="Evaluation", text="Measure retrieval before generation."),
            Document(document_id="doc-b", title="Tracing", text="Keep source ids with every answer."),
        ]

    def test_answer_is_deterministic(self) -> None:
        first = self.pipeline.answer("How should the system be evaluated?", self.documents)
        second = self.pipeline.answer("How should the system be evaluated?", self.documents)
        self.assertEqual(first, second)

    def test_answer_propagates_unique_citations(self) -> None:
        documents = [self.documents[0], self.documents[0], self.documents[1]]
        response = self.pipeline.answer("What should be traced?", documents)
        self.assertEqual(("doc-a", "doc-b"), response.citations)

    def test_empty_question_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            self.pipeline.answer("   ", self.documents)

    def test_empty_document_list_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            self.pipeline.answer("A valid question", [])

    def test_answer_language_is_independent_from_question_language(self) -> None:
        client = RecordingClient()
        pipeline = TraceRAGPipeline(client)

        response = pipeline.answer(
            "Why evaluate retrieval separately?",
            self.documents,
            answer_language="Chinese",
        )

        self.assertEqual("中文回答", response.answer)
        self.assertIn("Write the final answer in Chinese.", client.prompt)

    def test_empty_answer_language_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            self.pipeline.answer("A valid question", self.documents, answer_language="   ")

    def test_mock_client_rejects_empty_prompt(self) -> None:
        client = MockLLMClient()
        with self.assertRaises(ValueError):
            client.complete("   ")

if __name__ == "__main__":
    unittest.main()
