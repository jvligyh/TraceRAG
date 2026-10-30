from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tracerag import (  # noqa: E402
    Chunk,
    ReciprocalRankFusionRetriever,
    RetrievedChunk,
    reciprocal_rank_fusion,
)


def make_chunk(chunk_id: str, text: str | None = None) -> Chunk:
    content = text or chunk_id
    return Chunk(chunk_id, f"doc-{chunk_id}", content, 0, len(content), f"{chunk_id}.md")


def result(chunk: Chunk, rank: int, score: float = 1.0) -> RetrievedChunk:
    return RetrievedChunk(chunk=chunk, score=score, rank=rank)


class StaticRetriever:
    def __init__(self, results: tuple[RetrievedChunk, ...]) -> None:
        self.results = results
        self.calls: list[tuple[str, int]] = []

    def retrieve(self, query: str, top_k: int = 5) -> tuple[RetrievedChunk, ...]:
        self.calls.append((query, top_k))
        return self.results[:top_k]


class ReciprocalRankFusionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.chunk_a = make_chunk("chunk-a")
        self.chunk_b = make_chunk("chunk-b")
        self.chunk_c = make_chunk("chunk-c")

    def test_chunk_supported_by_two_rankings_moves_to_first(self) -> None:
        rankings = (
            (result(self.chunk_a, 1), result(self.chunk_b, 2)),
            (result(self.chunk_b, 1), result(self.chunk_c, 2)),
        )

        fused = reciprocal_rank_fusion(rankings, top_k=3, fusion_k=1)

        self.assertEqual(
            ("chunk-b", "chunk-a", "chunk-c"),
            tuple(item.chunk.chunk_id for item in fused),
        )
        self.assertAlmostEqual(1 / 3 + 1 / 2, fused[0].score)

    def test_same_chunk_is_returned_once(self) -> None:
        rankings = (
            (result(self.chunk_a, 1),),
            (result(self.chunk_a, 2),),
        )

        fused = reciprocal_rank_fusion(rankings, top_k=5)

        self.assertEqual(1, len(fused))
        self.assertEqual("chunk-a", fused[0].chunk.chunk_id)

    def test_raw_retrieval_scores_do_not_affect_rrf(self) -> None:
        ranking = (
            result(self.chunk_a, rank=1, score=-999.0),
            result(self.chunk_b, rank=2, score=999.0),
        )

        fused = reciprocal_rank_fusion((ranking,), top_k=2, fusion_k=1)

        self.assertEqual(
            ("chunk-a", "chunk-b"),
            tuple(item.chunk.chunk_id for item in fused),
        )

    def test_equal_fusion_scores_use_best_rank_then_chunk_id(self) -> None:
        rankings = (
            (result(self.chunk_b, 1), result(self.chunk_c, 2)),
            (result(self.chunk_a, 1),),
        )

        fused = reciprocal_rank_fusion(rankings, top_k=3, fusion_k=1)

        self.assertEqual(
            ("chunk-a", "chunk-b", "chunk-c"),
            tuple(item.chunk.chunk_id for item in fused),
        )

    def test_duplicate_chunk_inside_one_ranking_is_rejected(self) -> None:
        ranking = ((result(self.chunk_a, 1), result(self.chunk_a, 2)),)
        with self.assertRaises(ValueError):
            reciprocal_rank_fusion(ranking)

    def test_conflicting_data_for_same_chunk_id_is_rejected(self) -> None:
        conflicting = make_chunk("chunk-a", text="different")
        rankings = ((result(self.chunk_a, 1),), (result(conflicting, 1),))
        with self.assertRaises(ValueError):
            reciprocal_rank_fusion(rankings)

    def test_empty_individual_rankings_return_no_results(self) -> None:
        self.assertEqual((), reciprocal_rank_fusion(((), ())))

    def test_invalid_arguments_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            reciprocal_rank_fusion(())
        with self.assertRaises(ValueError):
            reciprocal_rank_fusion(((),), top_k=0)
        with self.assertRaises(ValueError):
            reciprocal_rank_fusion(((),), fusion_k=0)


class ReciprocalRankFusionRetrieverTests(unittest.TestCase):
    def test_retriever_requests_candidate_depth_and_forwards_query(self) -> None:
        chunk_a = make_chunk("chunk-a")
        chunk_b = make_chunk("chunk-b")
        first = StaticRetriever((result(chunk_a, 1),))
        second = StaticRetriever((result(chunk_b, 1),))
        retriever = ReciprocalRankFusionRetriever((first, second), candidate_k=10)

        fused = retriever.retrieve("检索问题", top_k=3)

        self.assertEqual([("检索问题", 10)], first.calls)
        self.assertEqual([("检索问题", 10)], second.calls)
        self.assertEqual(2, len(fused))

    def test_invalid_configuration_and_query_are_rejected(self) -> None:
        empty = StaticRetriever(())
        with self.assertRaises(ValueError):
            ReciprocalRankFusionRetriever((empty,))
        with self.assertRaises(ValueError):
            ReciprocalRankFusionRetriever((empty, empty), candidate_k=0)
        with self.assertRaises(ValueError):
            ReciprocalRankFusionRetriever((empty, empty), fusion_k=0)

        retriever = ReciprocalRankFusionRetriever((empty, empty))
        with self.assertRaises(ValueError):
            retriever.retrieve("   ")
        with self.assertRaises(ValueError):
            retriever.retrieve("query", top_k=0)


if __name__ == "__main__":
    unittest.main()
