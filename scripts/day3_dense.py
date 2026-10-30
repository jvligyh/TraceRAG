from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from time import perf_counter


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tracerag import DenseRetriever, FastEmbedEmbeddingModel, TextChunker, parse_path  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="使用本地多语言 Embedding 执行稠密检索。")
    parser.add_argument("paths", nargs="+", type=Path, help="要建立索引的 Markdown/PDF 文件")
    parser.add_argument("--query", required=True, help="要检索的问题")
    parser.add_argument("--top-k", type=int, default=3, help="返回结果数量")
    parser.add_argument("--max-chars", type=int, default=400, help="每个 chunk 的最大字符数")
    parser.add_argument("--overlap-chars", type=int, default=80, help="相邻 chunk 的目标重叠字符数")
    parser.add_argument("--model", default=FastEmbedEmbeddingModel.DEFAULT_MODEL, help="FastEmbed 模型名称")
    return parser


def main() -> None:
    logging.getLogger("pypdf").setLevel(logging.ERROR)
    args = build_parser().parse_args()

    chunker = TextChunker(max_chars=args.max_chars, overlap_chars=args.overlap_chars)
    documents = tuple(document for path in args.paths for document in parse_path(path))
    chunks = tuple(chunk for document in documents for chunk in chunker.chunk(document))

    index_started = perf_counter()
    embedding_model = FastEmbedEmbeddingModel(model_name=args.model)
    retriever = DenseRetriever(chunks, embedding_model)
    index_seconds = perf_counter() - index_started

    search_started = perf_counter()
    results = retriever.retrieve(args.query, top_k=args.top_k)
    search_seconds = perf_counter() - search_started

    output = {
        "query": args.query,
        "model": embedding_model.model_name,
        "documents": len(documents),
        "chunks": len(chunks),
        "index_seconds": round(index_seconds, 4),
        "search_seconds": round(search_seconds, 4),
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
