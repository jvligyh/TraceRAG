from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Protocol

from .embeddings import Embedding, EmbeddingModel
from .schemas import Chunk, RetrievedChunk


class Retriever(Protocol):
    """检索器的统一接口。"""

    def retrieve(self, query: str, top_k: int = 5) -> tuple[RetrievedChunk, ...]:
        """返回按照相关性从高到低排列的文本块。"""


class DenseRetriever:
    """在内存中使用余弦相似度执行穷举稠密检索。"""

    def __init__(self, chunks: Sequence[Chunk], embedding_model: EmbeddingModel) -> None:
        if not chunks:
            raise ValueError("at least one chunk is required")

        self._chunks = tuple(chunks)
        self._embedding_model = embedding_model
        self._embeddings = embedding_model.embed_passages(tuple(chunk.text for chunk in self._chunks))
        self._dimension = _validate_embedding_batch(self._embeddings, len(self._chunks))

    def retrieve(self, query: str, top_k: int = 5) -> tuple[RetrievedChunk, ...]:
        query = query.strip()
        if not query:
            raise ValueError("query must not be empty")
        if top_k < 1:
            raise ValueError("top_k must be positive")

        query_embedding = self._embedding_model.embed_query(query)
        _validate_embedding(query_embedding, expected_dimension=self._dimension)

        scored_chunks = [
            (cosine_similarity(query_embedding, embedding), chunk)
            for chunk, embedding in zip(self._chunks, self._embeddings, strict=True)
        ]
        scored_chunks.sort(key=lambda item: (-item[0], item[1].chunk_id))

        return tuple(
            RetrievedChunk(chunk=chunk, score=score, rank=rank)
            for rank, (score, chunk) in enumerate(scored_chunks[:top_k], start=1)
        )


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    """计算两个等长非零向量的余弦相似度。"""

    _validate_embedding(left)
    _validate_embedding(right, expected_dimension=len(left))
    dot_product = math.fsum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(math.fsum(value * value for value in left))
    right_norm = math.sqrt(math.fsum(value * value for value in right))
    return dot_product / (left_norm * right_norm)


def _validate_embedding_batch(embeddings: Sequence[Embedding], expected_count: int) -> int:
    if len(embeddings) != expected_count:
        raise ValueError("embedding count must match chunk count")
    if not embeddings:
        raise ValueError("embedding batch must not be empty")

    dimension = len(embeddings[0])
    for embedding in embeddings:
        _validate_embedding(embedding, expected_dimension=dimension)
    return dimension


def _validate_embedding(vector: Sequence[float], expected_dimension: int | None = None) -> None:
    if not vector:
        raise ValueError("embedding vector must not be empty")
    if expected_dimension is not None and len(vector) != expected_dimension:
        raise ValueError("embedding vectors must have the same dimension")
    if not all(math.isfinite(value) for value in vector):
        raise ValueError("embedding vector values must be finite")
    if not any(value != 0.0 for value in vector):
        raise ValueError("embedding vector must not be zero")
