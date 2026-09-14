from __future__ import annotations

import hashlib

from .schemas import Chunk, Document


class TextChunker:
    """优先在段落或空白处切分的确定性文本切块器。"""

    def __init__(self, max_chars: int = 800, overlap_chars: int = 120) -> None:
        if max_chars < 1:
            raise ValueError("max_chars must be positive")
        if overlap_chars < 0:
            raise ValueError("overlap_chars must not be negative")
        if overlap_chars >= max_chars:
            raise ValueError("overlap_chars must be smaller than max_chars")
        self.max_chars = max_chars
        self.overlap_chars = overlap_chars

    def chunk(self, document: Document) -> tuple[Chunk, ...]:
        text = document.text
        chunks: list[Chunk] = []
        start = 0

        while start < len(text):
            end = self._choose_end(text, start)
            content_start = start
            content_end = end

            while content_start < content_end and text[content_start].isspace():
                content_start += 1
            while content_end > content_start and text[content_end - 1].isspace():
                content_end -= 1

            if content_start < content_end:
                content = text[content_start:content_end]
                chunks.append(
                    Chunk(
                        chunk_id=self._chunk_id(document.document_id, content_start, content_end, content),
                        document_id=document.document_id,
                        text=content,
                        start_char=content_start,
                        end_char=content_end,
                        source=document.source,
                        page_number=document.page_number,
                    )
                )

            if end >= len(text):
                break
            start = max(end - self.overlap_chars, start + 1)

        return tuple(chunks)

    def _choose_end(self, text: str, start: int) -> int:
        hard_end = min(start + self.max_chars, len(text))
        if hard_end >= len(text):
            return len(text)

        preferred_start = start + self.max_chars // 2
        for separator in ("\n\n", "\n", ". ", " "):
            boundary = text.rfind(separator, preferred_start, hard_end)
            if boundary != -1:
                return boundary + len(separator)
        return hard_end

    @staticmethod
    def _chunk_id(document_id: str, start: int, end: int, text: str) -> str:
        identity = f"{document_id}:{start}:{end}:{text}".encode("utf-8")
        return f"chunk-{hashlib.sha256(identity).hexdigest()[:20]}"
