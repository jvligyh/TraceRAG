from __future__ import annotations

from collections.abc import Sequence

from .bm25 import BM25Retriever, tokenize_char_ngrams
from .schemas import Chunk, RetrievedChunk
from .translation import QueryTranslator


class CrossLingualBM25Retriever:
    """用查询翻译扩展 BM25，并将词法变体压成一张公平排名。"""

    def __init__(
        self,
        chunks: Sequence[Chunk],
        translator: QueryTranslator,
        *,
        candidate_k: int = 20,
        fusion_k: int = 60,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        if candidate_k < 1:
            raise ValueError("candidate_k must be positive")
        if fusion_k < 1:
            raise ValueError("fusion_k must be positive")

        self._translator = translator
        self._candidate_k = candidate_k
        self._fusion_k = fusion_k
        self._word_retriever = BM25Retriever(chunks, k1=k1, b=b)
        self._char_retriever = BM25Retriever(chunks, tokenizer=tokenize_char_ngrams, k1=k1, b=b)

    def retrieve(self, query: str, top_k: int = 5) -> tuple[RetrievedChunk, ...]:
        query = query.strip()
        if not query:
            raise ValueError("query must not be empty")
        if top_k < 1:
            raise ValueError("top_k must be positive")

        depth = max(top_k, self._candidate_k)
        rankings = [
            self._word_retriever.retrieve(query, top_k=depth),
            self._char_retriever.retrieve(query, top_k=depth),
        ]

        translated_query = (
            self._translator.translate_to_english(query)
            if _contains_han(query)
            else self._translator.translate_to_chinese(query)
        ).strip()
        if not translated_query:
            raise RuntimeError("query translator returned empty text")
        if translated_query.casefold() != query.casefold():
            rankings.extend(
                (
                    self._word_retriever.retrieve(translated_query, top_k=depth),
                    self._char_retriever.retrieve(translated_query, top_k=depth),
                )
            )

        return _fuse_by_best_rank(rankings, top_k=top_k, fusion_k=self._fusion_k)


def _fuse_by_best_rank(
    rankings: Sequence[Sequence[RetrievedChunk]],
    *,
    top_k: int,
    fusion_k: int,
) -> tuple[RetrievedChunk, ...]:
    """每个 Chunk 只采用最佳词法名次，避免相关通道重复投票。"""

    candidates: dict[str, tuple[float, Chunk]] = {}
    for ranking in rankings:
        for result in ranking:
            score = 1.0 / (fusion_k + result.rank)
            previous = candidates.get(result.chunk.chunk_id)
            if previous is None or score > previous[0]:
                candidates[result.chunk.chunk_id] = (score, result.chunk)

    ordered = sorted(candidates.values(), key=lambda item: (-item[0], item[1].chunk_id))
    return tuple(
        RetrievedChunk(chunk=chunk, score=score, rank=rank)
        for rank, (score, chunk) in enumerate(ordered[:top_k], start=1)
    )


def _contains_han(text: str) -> bool:
    return any("\u3400" <= character <= "\u4dbf" or "\u4e00" <= character <= "\u9fff" for character in text)
