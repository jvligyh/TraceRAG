from __future__ import annotations

import math
import sys
import unittest
from collections.abc import Sequence
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tracerag import Chunk, DenseRetriever, cosine_similarity  # noqa: E402
from tracerag.embeddings import Embedding  # noqa: E402


class FakeEmbeddingModel:
    """只为单元测试返回预先写好的向量，不冒充真实语义模型。"""

    def __init__(self, passages: dict[str, Embedding], queries: dict[str, Embedding]) -> None:
        self._passages = passages
        self._queries = queries

    def embed_passages(self, texts: Sequence[str]) -> tuple[Embedding, ...]:
        return tuple(self._passages[text] for text in texts)

    def embed_query(self, text: str) -> Embedding:
        return self._queries[text]


class TooFewEmbeddingsModel:
    """故意少返回一个文档向量，用于验证数量校验。"""

    def embed_passages(self, texts: Sequence[str]) -> tuple[Embedding, ...]:
        return tuple((1.0, 0.0) for _ in texts[:-1])

    def embed_query(self, text: str) -> Embedding:
        return (1.0, 0.0)


class CosineSimilarityTests(unittest.TestCase):
    def test_orthogonal_vectors_have_zero_similarity(self) -> None:
        self.assertEqual(0.0, cosine_similarity((1.0, 0.0), (0.0, 1.0)))

    def test_same_direction_ignores_vector_length(self) -> None:
        self.assertAlmostEqual(1.0, cosine_similarity((1.0, 0.0), (10.0, 0.0)))

    def test_known_angle(self) -> None:
        self.assertAlmostEqual(1 / math.sqrt(2), cosine_similarity((1.0, 0.0), (1.0, 1.0)))

    def test_dimension_mismatch_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            cosine_similarity((1.0, 0.0), (1.0,))


class DenseRetrieverTests(unittest.TestCase):
    def setUp(self) -> None:
        self.chunks = (
            Chunk("chunk-retrieval", "doc-a", "检索负责找到相关证据。", 0, 11, "a.md"),
            Chunk("chunk-citation", "doc-b", "引用用于追踪答案来源。", 0, 11, "b.md"),
            Chunk("chunk-generation", "doc-c", "生成模型根据证据回答。", 0, 11, "c.md"),
        )
        self.model = FakeEmbeddingModel(
            passages={
                "检索负责找到相关证据。": (1.0, 0.0),
                "引用用于追踪答案来源。": (0.0, 1.0),
                "生成模型根据证据回答。": (0.8, 0.2),
            },
            queries={"怎样找到证据？": (1.0, 0.0)},
        )

    def test_retrieve_orders_chunks_by_cosine_similarity(self) -> None:
        results = DenseRetriever(self.chunks, self.model).retrieve("怎样找到证据？", top_k=2)

        self.assertEqual(("chunk-retrieval", "chunk-generation"), tuple(item.chunk.chunk_id for item in results))
        self.assertEqual((1, 2), tuple(item.rank for item in results))
        self.assertGreater(results[0].score, results[1].score)
        self.assertEqual("a.md", results[0].chunk.source)

    def test_top_k_larger_than_index_returns_every_chunk(self) -> None:
        results = DenseRetriever(self.chunks, self.model).retrieve("怎样找到证据？", top_k=10)
        self.assertEqual(3, len(results))

    def test_empty_query_is_rejected(self) -> None:
        retriever = DenseRetriever(self.chunks, self.model)
        with self.assertRaises(ValueError):
            retriever.retrieve("   ")

    def test_non_positive_top_k_is_rejected(self) -> None:
        retriever = DenseRetriever(self.chunks, self.model)
        with self.assertRaises(ValueError):
            retriever.retrieve("怎样找到证据？", top_k=0)

    def test_embedding_count_must_match_chunk_count(self) -> None:
        with self.assertRaises(ValueError):
            DenseRetriever(self.chunks, TooFewEmbeddingsModel())

    def test_equal_scores_are_ordered_by_chunk_id(self) -> None:
        chunks = (
            Chunk(
                chunk_id="chunk-b",
                document_id="doc-b",
                text="第二段文本",
                start_char=0,
                end_char=5,
                source="b.md",
            ),
            Chunk(
                chunk_id="chunk-a",
                document_id="doc-a",
                text="第一段文本",
                start_char=0,
                end_char=5,
                source="a.md",
            ),
        )
        model = FakeEmbeddingModel(
            passages={
                "第二段文本": (1.0, 0.0),
                "第一段文本": (1.0, 0.0),
            },
            queries={
                "测试问题": (1.0, 0.0),
            },
        )

        results = DenseRetriever(
            chunks,
            model,
        ).retrieve(
            "测试问题",
            top_k=2,
        )

        self.assertEqual(
            ("chunk-a", "chunk-b"),
            tuple(
                result.chunk.chunk_id
                for result in results
            ),
        )

if __name__ == "__main__":
    unittest.main()
