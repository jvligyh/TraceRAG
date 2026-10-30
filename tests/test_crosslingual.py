from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tracerag import Chunk, CrossLingualBM25Retriever, LLMQueryTranslator, QwenQueryTranslator  # noqa: E402
from tracerag.translation import _load_pretrained  # noqa: E402


class StaticTranslator:
    def __init__(self, *, english: str = "unused", chinese: str = "未使用") -> None:
        self.english = english
        self.chinese = chinese
        self.english_calls = 0
        self.chinese_calls = 0

    def translate_to_english(self, query: str) -> str:
        self.english_calls += 1
        return self.english

    def translate_to_chinese(self, query: str) -> str:
        self.chinese_calls += 1
        return self.chinese


class RecordingLLMClient:
    def __init__(self, response: str) -> None:
        self.response = response
        self.prompts: list[str] = []

    def complete(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.response


class RecordingModelFactory:
    calls: list[tuple[str, bool]] = []
    fail_local = False
    local_error: type[Exception] = OSError

    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        *,
        revision: str = "",
        local_files_only: bool = False,
        **kwargs: object,
    ) -> str:
        cls.calls.append((model_name, local_files_only))
        if local_files_only and cls.fail_local:
            raise cls.local_error("not cached or incomplete")
        return f"{model_name}@{revision}"


class LLMQueryTranslatorTests(unittest.TestCase):
    def test_returns_clean_translation_and_caches_same_query(self) -> None:
        client = RecordingLLMClient('"Why evaluate retrieval and generation separately?"')
        translator = LLMQueryTranslator(client)

        first = translator.translate_to_english("为什么要分别评估检索和生成？")
        second = translator.translate_to_english("为什么要分别评估检索和生成？")
        client.response = "为什么要分别评估检索和生成？"
        chinese = translator.translate_to_chinese("Why evaluate retrieval and generation separately?")

        self.assertEqual("Why evaluate retrieval and generation separately?", first)
        self.assertEqual(first, second)
        self.assertEqual("为什么要分别评估检索和生成？", chinese)
        self.assertEqual(2, len(client.prompts))
        self.assertIn("QUERY_JSON", client.prompts[0])
        self.assertIn("Chinese", client.prompts[1])

    def test_rejects_empty_query_and_translation(self) -> None:
        translator = LLMQueryTranslator(RecordingLLMClient("   "))
        with self.assertRaises(ValueError):
            translator.translate_to_english("   ")
        with self.assertRaises(RuntimeError):
            translator.translate_to_english("有效问题")


class LocalModelLoadingTests(unittest.TestCase):
    def setUp(self) -> None:
        RecordingModelFactory.calls = []
        RecordingModelFactory.fail_local = False
        RecordingModelFactory.local_error = OSError

    def test_uses_local_cache_without_network_fallback(self) -> None:
        with patch("tracerag.translation._resolve_local_snapshot", return_value="cached-model"):
            loaded = _load_pretrained(RecordingModelFactory, "model", "revision")

        self.assertEqual("cached-model@", loaded)
        self.assertEqual([("cached-model", True)], RecordingModelFactory.calls)

    def test_downloads_only_when_local_cache_is_missing(self) -> None:
        RecordingModelFactory.fail_local = True

        with patch("tracerag.translation._resolve_local_snapshot", return_value=None):
            loaded = _load_pretrained(RecordingModelFactory, "model", "revision")

        self.assertEqual("model@revision", loaded)
        self.assertEqual([("model", False)], RecordingModelFactory.calls)

    def test_downloads_when_local_cache_is_incomplete(self) -> None:
        RecordingModelFactory.fail_local = True
        RecordingModelFactory.local_error = AttributeError

        with patch("tracerag.translation._resolve_local_snapshot", return_value="incomplete-model"):
            loaded = _load_pretrained(RecordingModelFactory, "model", "revision")

        self.assertEqual("model@revision", loaded)
        self.assertEqual(
            [("incomplete-model", True), ("model", False)],
            RecordingModelFactory.calls,
        )


class QwenQueryTranslatorTests(unittest.TestCase):
    def test_rejects_invalid_configuration_before_loading_model(self) -> None:
        with self.assertRaises(ValueError):
            QwenQueryTranslator(model_name="   ")
        with self.assertRaises(ValueError):
            QwenQueryTranslator(revision="   ")
        with self.assertRaises(ValueError):
            QwenQueryTranslator(max_new_tokens=0)


class CrossLingualBM25RetrieverTests(unittest.TestCase):
    def setUp(self) -> None:
        self.chunks = (
            Chunk(
                "chunk-english",
                "doc-en",
                "Retrieval and generation should be evaluated separately.",
                0,
                57,
                "english.md",
            ),
            Chunk("chunk-chinese", "doc-zh", "杭州今天是晴朗天气。", 0, 11, "chinese.md"),
        )

    def test_chinese_query_retrieves_english_document_through_translation(self) -> None:
        translator = StaticTranslator(english="why evaluate retrieval and generation separately")
        retriever = CrossLingualBM25Retriever(self.chunks, translator)

        results = retriever.retrieve("为什么要分别评估检索和生成？", top_k=1)

        self.assertEqual("chunk-english", results[0].chunk.chunk_id)
        self.assertEqual(1, translator.english_calls)
        self.assertEqual(0, translator.chinese_calls)

    def test_same_chunk_is_returned_once_when_multiple_lexical_variants_match(self) -> None:
        chunks = (
            Chunk("chunk-rag", "doc-a", "检索系统负责寻找相关证据。", 0, 13, "a.md"),
            Chunk("chunk-weather", "doc-b", "杭州今天是晴朗天气。", 0, 11, "b.md"),
        )
        retriever = CrossLingualBM25Retriever(chunks, StaticTranslator(english="retrieval evidence"))

        results = retriever.retrieve("检索证据", top_k=5)

        chunk_ids = tuple(result.chunk.chunk_id for result in results)
        self.assertEqual(("chunk-rag",), chunk_ids)

    def test_english_query_retrieves_chinese_document_through_translation(self) -> None:
        translator = StaticTranslator(chinese="杭州 晴朗 天气")
        retriever = CrossLingualBM25Retriever(self.chunks, translator)

        results = retriever.retrieve("What is the weather in Hangzhou?", top_k=1)

        self.assertEqual("chunk-chinese", results[0].chunk.chunk_id)
        self.assertEqual(0, translator.english_calls)
        self.assertEqual(1, translator.chinese_calls)

    def test_invalid_parameters_are_rejected(self) -> None:
        translator = StaticTranslator(english="translation")
        with self.assertRaises(ValueError):
            CrossLingualBM25Retriever(self.chunks, translator, candidate_k=0)
        with self.assertRaises(ValueError):
            CrossLingualBM25Retriever(self.chunks, translator, fusion_k=0)

        retriever = CrossLingualBM25Retriever(self.chunks, translator)
        with self.assertRaises(ValueError):
            retriever.retrieve("   ")
        with self.assertRaises(ValueError):
            retriever.retrieve("query", top_k=0)


if __name__ == "__main__":
    unittest.main()
