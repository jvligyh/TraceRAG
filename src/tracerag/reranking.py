from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any, Protocol

from .schemas import RetrievedChunk
from .transformer_embeddings import prepare_snapshot


class Reranker(Protocol):
    def rerank(self, query: str, chunks: Sequence[RetrievedChunk], top_k: int | None = None) -> tuple[RetrievedChunk, ...]: ...


class NoOpReranker:
    def rerank(self, query: str, chunks: Sequence[RetrievedChunk], top_k: int | None = None) -> tuple[RetrievedChunk, ...]:
        _validate(query, top_k)
        return tuple(chunks if top_k is None else chunks[:top_k])


class CrossEncoderReranker:
    """对 query/passage 成对编码，用相关性 logit 重排 RRF 候选。"""

    DEFAULT_MODEL = "BAAI/bge-reranker-v2-m3"
    DEFAULT_REVISION = "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"

    def __init__(
        self, model_name: str = DEFAULT_MODEL, *, revision: str = DEFAULT_REVISION,
        device: str = "auto", batch_size: int = 4, max_length: int = 512,
    ) -> None:
        if not model_name.strip() or not revision.strip():
            raise ValueError("model and revision must not be empty")
        if batch_size < 1 or max_length < 1:
            raise ValueError("batch_size and max_length must be positive")
        if device not in {"auto", "cpu", "cuda"}:
            raise ValueError("device must be auto, cpu or cuda")
        self.model_name, self.revision, self.device = model_name, revision, device
        self.batch_size, self.max_length = batch_size, max_length
        self._model: Any = None

    def _load(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        device = "cuda" if self.device == "auto" and torch.cuda.is_available() else self.device
        if device == "auto":
            device = "cpu"
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        self.device, self._torch = device, torch
        snapshot = prepare_snapshot(self.model_name, self.revision)
        self._tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
        dtype = torch.float16 if device == "cuda" else torch.float32
        model = AutoModelForSequenceClassification.from_pretrained(snapshot, local_files_only=True, dtype=dtype)
        model.to(device).eval()
        self._model = model

    def _score(self, query: str, texts: Sequence[str]) -> list[float]:
        self._load()
        scores: list[float] = []
        for start in range(0, len(texts), self.batch_size):
            pairs = [[query, text] for text in texts[start : start + self.batch_size]]
            encoded = self._tokenizer(pairs, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt")
            encoded = {name: value.to(self.device) for name, value in encoded.items()}
            with self._torch.inference_mode():
                logits = self._model(**encoded, return_dict=True).logits.reshape(-1).float()
            scores.extend(float(x) for x in logits.cpu().tolist())
        return scores

    def rerank(self, query: str, chunks: Sequence[RetrievedChunk], top_k: int | None = None) -> tuple[RetrievedChunk, ...]:
        _validate(query, top_k)
        if not chunks:
            return ()
        if len({result.chunk.chunk_id for result in chunks}) != len(chunks):
            raise ValueError("reranker candidates must have unique chunk ids")
        scores = self._score(query.strip(), [result.chunk.text for result in chunks])
        if len(scores) != len(chunks) or any(not math.isfinite(score) for score in scores):
            raise RuntimeError("reranker returned invalid scores")
        ordered = sorted(zip(scores, chunks, strict=True), key=lambda item: (-item[0], item[1].chunk.chunk_id))
        if top_k is not None:
            ordered = ordered[:top_k]
        return tuple(RetrievedChunk(chunk=result.chunk, score=score, rank=rank) for rank, (score, result) in enumerate(ordered, 1))


def _validate(query: str, top_k: int | None) -> None:
    if not query.strip():
        raise ValueError("query must not be empty")
    if top_k is not None and top_k < 1:
        raise ValueError("top_k must be positive")
