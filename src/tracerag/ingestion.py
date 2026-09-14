from __future__ import annotations

import hashlib
import re
import unicodedata
from pathlib import Path
from typing import Protocol

from pypdf import PdfReader

from .schemas import Document


class DocumentParser(Protocol):
    """文档解析器的统一接口。"""

    def parse(self, path: Path) -> tuple[Document, ...]:
        """解析一个文件，返回一个或多个文档。"""


class MarkdownParser:
    """将一个 Markdown 文件解析为一个文档。"""

    def parse(self, path: Path) -> tuple[Document, ...]:
        path = _require_file(path)
        raw = path.read_bytes()
        text = normalize_text(raw.decode("utf-8-sig"))
        title = _markdown_title(text) or path.stem
        document_id = f"md-{_short_hash(raw)}"
        return (
            Document(
                document_id=document_id,
                title=title,
                text=text,
                source=str(path.resolve()),
            ),
        )


class PDFParser:
    """按页提取 PDF 文本，保留可用于引用的页码。"""

    def parse(self, path: Path) -> tuple[Document, ...]:
        path = _require_file(path)
        file_hash = _short_hash(path.read_bytes())
        reader = PdfReader(path)
        documents: list[Document] = []

        for page_number, page in enumerate(reader.pages, start=1):
            text = normalize_text(page.extract_text() or "")
            if not text:
                continue
            documents.append(
                Document(
                    document_id=f"pdf-{file_hash}-p{page_number:04d}",
                    title=f"{path.stem}（第 {page_number} 页）",
                    text=text,
                    source=str(path.resolve()),
                    page_number=page_number,
                )
            )

        if not documents:
            raise ValueError(f"PDF does not contain extractable text: {path}")
        return tuple(documents)


def parse_path(path: str | Path) -> tuple[Document, ...]:
    """根据文件后缀选择解析器。"""

    resolved = Path(path)
    suffix = resolved.suffix.lower()
    if suffix in {".md", ".markdown"}:
        return MarkdownParser().parse(resolved)
    if suffix == ".pdf":
        return PDFParser().parse(resolved)
    raise ValueError(f"unsupported document type: {suffix or '<none>'}")


def normalize_text(text: str) -> str:
    """统一 Unicode、换行符和过多空行，保留代码缩进。"""

    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _require_file(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"document file not found: {path}")
    return path


def _markdown_title(text: str) -> str | None:
    for line in text.splitlines():
        match = re.match(r"^\s*#\s+(.+?)\s*$", line)
        if match:
            return match.group(1)
    return None


def _short_hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:16]
