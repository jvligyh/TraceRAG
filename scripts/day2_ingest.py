from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tracerag import TextChunker, parse_path  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="解析 Markdown/PDF 并生成稳定 chunk id。")
    parser.add_argument("paths", nargs="+", type=Path, help="要接入的 .md/.markdown/.pdf 文件")
    parser.add_argument("--max-chars", type=int, default=800, help="每个 chunk 的最大字符数")
    parser.add_argument("--overlap-chars", type=int, default=120, help="相邻 chunk 的目标重叠字符数")
    return parser


def main() -> None:
    logging.getLogger("pypdf").setLevel(logging.ERROR)
    args = build_parser().parse_args()
    chunker = TextChunker(max_chars=args.max_chars, overlap_chars=args.overlap_chars)
    summaries: list[dict[str, object]] = []

    for path in args.paths:
        documents = parse_path(path)
        chunks = tuple(chunk for document in documents for chunk in chunker.chunk(document))
        summaries.append(
            {
                "path": str(path),
                "documents": len(documents),
                "chunks": len(chunks),
                "document_ids": [document.document_id for document in documents],
                "chunk_ids": [chunk.chunk_id for chunk in chunks],
                "pages": [document.page_number for document in documents if document.page_number is not None],
            }
        )

    print(json.dumps(summaries, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
