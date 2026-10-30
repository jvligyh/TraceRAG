from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from time import perf_counter


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tracerag import BM25Retriever, TextChunker, parse_path, tokenize_char_ngrams  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="使用双语分词和 BM25 执行词法检索。")
    parser.add_argument("paths", nargs="+", type=Path, help="要建立索引的 Markdown/PDF 文件")
    parser.add_argument("--query", required=True, help="要检索的问题")
    parser.add_argument("--top-k", type=int, default=3, help="返回结果数量")
    parser.add_argument("--max-chars", type=int, default=180, help="每个 chunk 的最大字符数")
    parser.add_argument("--overlap-chars", type=int, default=30, help="相邻 chunk 的目标重叠字符数")
    parser.add_argument("--k1", type=float, default=1.5, help="BM25 词频饱和参数")
    parser.add_argument("--b", type=float, default=0.75, help="BM25 文档长度归一化参数")
    parser.add_argument(
        "--analyzer",
        choices=("word", "char-ngram"),
        default="word",
        help="中文分析方式：jieba 词级或字符 2/3-gram",
    )
    return parser


def main() -> None:
    logging.getLogger("pypdf").setLevel(logging.ERROR)
    args = build_parser().parse_args()

    chunker = TextChunker(max_chars=args.max_chars, overlap_chars=args.overlap_chars)
    documents = tuple(document for path in args.paths for document in parse_path(path))
    chunks = tuple(chunk for document in documents for chunk in chunker.chunk(document))

    index_started = perf_counter()
    tokenizer = tokenize_char_ngrams if args.analyzer == "char-ngram" else None
    retriever = BM25Retriever(chunks, tokenizer=tokenizer, k1=args.k1, b=args.b)
    index_seconds = perf_counter() - index_started

    search_started = perf_counter()
    results = retriever.retrieve(args.query, top_k=args.top_k)
    search_seconds = perf_counter() - search_started

    output = {
        "query": args.query,
        "retriever": f"bm25-{args.analyzer}",
        "k1": retriever.k1,
        "b": retriever.b,
        "documents": len(documents),
        "chunks": len(chunks),
        "index_seconds": round(index_seconds, 6),
        "search_seconds": round(search_seconds, 6),
        "results": [
            {
                "rank": result.rank,
                "score": round(result.score, 6),
                "chunk_id": result.chunk.chunk_id,
                "document_id": result.chunk.document_id,
                "source": result.chunk.source,
                "page_number": result.chunk.page_number,
                "text": result.chunk.text,
            }
            for result in results
        ],
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
