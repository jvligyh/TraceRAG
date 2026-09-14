from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tracerag import Document, MarkdownParser, PDFParser, TextChunker, parse_path  # noqa: E402
from tracerag.ingestion import normalize_text  # noqa: E402


class IngestionTests(unittest.TestCase):
    def test_normalize_text_preserves_indentation_and_collapses_blank_lines(self) -> None:
        text = "Ａ\r\n    code()   \r\n\r\n\r\nnext"
        self.assertEqual("A\n    code()\n\nnext", normalize_text(text))

    def test_markdown_parser_extracts_title_and_stable_id(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "guide.md"
            path.write_text("# 检索指南\n\n使用稳定的 chunk id。", encoding="utf-8")
            first = MarkdownParser().parse(path)[0]
            second = MarkdownParser().parse(path)[0]

        self.assertEqual("检索指南", first.title)
        self.assertEqual(first.document_id, second.document_id)
        self.assertTrue(first.document_id.startswith("md-"))

    def test_markdown_content_change_changes_document_id(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "guide.md"
            path.write_text("# Guide\nfirst", encoding="utf-8")
            first_id = MarkdownParser().parse(path)[0].document_id
            path.write_text("# Guide\nsecond", encoding="utf-8")
            second_id = MarkdownParser().parse(path)[0].document_id

        self.assertNotEqual(first_id, second_id)

    def test_pdf_parser_preserves_page_numbers_and_skips_empty_pages(self) -> None:
        pages = [
            SimpleNamespace(extract_text=lambda: "first page"),
            SimpleNamespace(extract_text=lambda: "   "),
            SimpleNamespace(extract_text=lambda: "third page"),
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "paper.pdf"
            path.write_bytes(b"stable-pdf-bytes")
            with patch("tracerag.ingestion.PdfReader", return_value=SimpleNamespace(pages=pages)):
                documents = PDFParser().parse(path)

        self.assertEqual((1, 3), tuple(document.page_number for document in documents))
        self.assertTrue(documents[0].document_id.endswith("-p0001"))
        self.assertTrue(documents[1].document_id.endswith("-p0003"))

    def test_parse_path_rejects_unsupported_extension(self) -> None:
        with self.assertRaises(ValueError):
            parse_path("notes.txt")


class TextChunkerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.document = Document(
            document_id="doc-stable",
            title="Chunking",
            text=("first paragraph has useful context. " * 4) + "\n\n" + ("second paragraph continues. " * 4),
            source="demo.md",
        )

    def test_chunk_ids_are_stable(self) -> None:
        chunker = TextChunker(max_chars=100, overlap_chars=20)
        self.assertEqual(chunker.chunk(self.document), chunker.chunk(self.document))

    def test_chunks_respect_size_and_keep_source(self) -> None:
        chunks = TextChunker(max_chars=100, overlap_chars=20).chunk(self.document)

        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(len(chunk.text) <= 100 for chunk in chunks))
        self.assertTrue(all(chunk.source == "demo.md" for chunk in chunks))
        self.assertTrue(all(chunk.chunk_id.startswith("chunk-") for chunk in chunks))

    def test_invalid_chunk_configuration_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            TextChunker(max_chars=100, overlap_chars=100)


if __name__ == "__main__":
    unittest.main()
