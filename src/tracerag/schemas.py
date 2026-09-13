from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class Document:
    """A minimal source document used by the Day 1 pipeline."""

    document_id: str
    title: str
    text: str

    def __post_init__(self) -> None:
        if not self.document_id.strip():
            raise ValueError("document_id must not be empty")
        if not self.title.strip():
            raise ValueError("title must not be empty")
        if not self.text.strip():
            raise ValueError("text must not be empty")


@dataclass(frozen=True, slots=True)
class QueryResponse:
    """An answer plus the source ids supplied to the model."""

    answer: str
    citations: tuple[str, ...] = field(default_factory=tuple)

