from __future__ import annotations

from collections.abc import Sequence

from .providers.base import LLMClient
from .schemas import Document, QueryResponse


class TraceRAGPipeline:
    """Day 1 pipeline with explicit context and citation propagation."""

    def __init__(self, client: LLMClient) -> None:
        self._client = client

    def answer(self, question: str, documents: Sequence[Document]) -> QueryResponse:
        question = question.strip()
        if not question:
            raise ValueError("question must not be empty")
        if not documents:
            raise ValueError("at least one document is required")

        prompt = self._build_prompt(question, documents)
        answer = self._client.complete(prompt)
        citations = tuple(dict.fromkeys(document.document_id for document in documents))
        return QueryResponse(answer=answer, citations=citations)

    @staticmethod
    def _build_prompt(question: str, documents: Sequence[Document]) -> str:
        context_blocks = [
            f"[SOURCE {document.document_id}] {document.title}\n{document.text}"
            for document in documents
        ]
        context = "\n\n".join(context_blocks)
        return (
            "Answer the question only from the supplied sources. "
            "If the sources are insufficient, say so.\n\n"
            f"SOURCES:\n{context}\n\nQUESTION:\n{question}\n"
        )

