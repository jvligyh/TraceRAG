from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class Document:
    """从一个来源文件中解析出的可检索文档。"""

    document_id: str
    title: str
    text: str
    source: str = ""
    page_number: int | None = None
    extraction_method: str = "native"
    ocr_confidence: float | None = None

    def __post_init__(self) -> None:
        if not self.document_id.strip():
            raise ValueError("document_id must not be empty")
        if not self.title.strip():
            raise ValueError("title must not be empty")
        if not self.text.strip():
            raise ValueError("text must not be empty")
        if self.page_number is not None and self.page_number < 1:
            raise ValueError("page_number must be positive")
        _validate_extraction(self.extraction_method,self.ocr_confidence)


@dataclass(frozen=True, slots=True)
class Chunk:
    """带有稳定 ID 和来源位置的文本块。"""

    chunk_id: str
    document_id: str
    text: str
    start_char: int
    end_char: int
    source: str = ""
    page_number: int | None = None
    extraction_method: str = "native"
    ocr_confidence: float | None = None

    def __post_init__(self) -> None:
        if not self.chunk_id.strip():
            raise ValueError("chunk_id must not be empty")
        if not self.document_id.strip():
            raise ValueError("document_id must not be empty")
        if not self.text.strip():
            raise ValueError("text must not be empty")
        if self.start_char < 0 or self.end_char <= self.start_char:
            raise ValueError("chunk character range is invalid")
        if self.page_number is not None and self.page_number < 1:
            raise ValueError("page_number must be positive")
        _validate_extraction(self.extraction_method,self.ocr_confidence)


@dataclass(frozen=True, slots=True)
class RetrievedChunk:
    """一次检索命中的文本块、相似度分数和名次。"""

    chunk: Chunk
    score: float
    rank: int

    def __post_init__(self) -> None:
        if not math.isfinite(self.score):
            raise ValueError("retrieval score must be finite")
        if self.rank < 1:
            raise ValueError("retrieval rank must be positive")


@dataclass(frozen=True, slots=True)
class QueryResponse:
    """An answer plus the source ids supplied to the model."""

    answer: str
    citations: tuple[str, ...] = field(default_factory=tuple)


def _validate_extraction(method: str, confidence: float | None) -> None:
    if method not in {"native","ocr"}:
        raise ValueError("unknown extraction method")
    if confidence is not None and (not math.isfinite(confidence) or not 0<=confidence<=1):
        raise ValueError("OCR confidence must be finite and between 0 and 1")
