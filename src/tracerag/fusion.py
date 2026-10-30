from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from .retrieval import Retriever
from .schemas import Chunk, RetrievedChunk


class ReciprocalRankFusionRetriever:
    """调用多个检索器取得候选榜，再使用 RRF 生成统一排名。"""

    def __init__(
        self,
        retrievers: Sequence[Retriever],
        *,
        candidate_k: int = 20,
        fusion_k: int = 60,
    ) -> None:
        if len(retrievers) < 2:
            raise ValueError("at least two retrievers are required")
        if candidate_k < 1:
            raise ValueError("candidate_k must be positive")
        if fusion_k < 1:
            raise ValueError("fusion_k must be positive")

        self._retrievers = tuple(retrievers)
        self.candidate_k = candidate_k
        self.fusion_k = fusion_k

    def retrieve(self, query: str, top_k: int = 5) -> tuple[RetrievedChunk, ...]:
        query = query.strip()
        if not query:
            raise ValueError("query must not be empty")
        if top_k < 1:
            raise ValueError("top_k must be positive")

        depth = max(top_k, self.candidate_k)
        rankings = tuple(retriever.retrieve(query, top_k=depth) for retriever in self._retrievers)
        return reciprocal_rank_fusion(rankings, top_k=top_k, fusion_k=self.fusion_k)


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[RetrievedChunk]],
    *,
    top_k: int = 5,
    fusion_k: int = 60,
) -> tuple[RetrievedChunk, ...]:
    """只使用名次融合多个检索结果，并按 chunk id 去重。"""

    if not rankings:
        raise ValueError("at least one ranking is required")
    if top_k < 1:
        raise ValueError("top_k must be positive")
    if fusion_k < 1:
        raise ValueError("fusion_k must be positive")

    candidates: dict[str, _FusionCandidate] = {}
    for ranking in rankings:
        seen_chunk_ids: set[str] = set()
        for result in ranking:
            chunk_id = result.chunk.chunk_id
            if chunk_id in seen_chunk_ids:
                raise ValueError("one ranking must not contain duplicate chunk ids")
            seen_chunk_ids.add(chunk_id)

            candidate = candidates.get(chunk_id)
            if candidate is None:
                candidate = _FusionCandidate(chunk=result.chunk, best_rank=result.rank)
                candidates[chunk_id] = candidate
            elif candidate.chunk != result.chunk:
                raise ValueError("the same chunk id must refer to identical chunk data")

            candidate.score += 1.0 / (fusion_k + result.rank)
            candidate.best_rank = min(candidate.best_rank, result.rank)

    ordered = sorted(
        candidates.values(),
        key=lambda candidate: (-candidate.score, candidate.best_rank, candidate.chunk.chunk_id),
    )
    return tuple(
        RetrievedChunk(chunk=candidate.chunk, score=candidate.score, rank=rank)
        for rank, candidate in enumerate(ordered[:top_k], start=1)
    )


@dataclass(slots=True)
class _FusionCandidate:
    chunk: Chunk
    best_rank: int
    score: float = 0.0
