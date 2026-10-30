from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from time import perf_counter

from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
load_dotenv(ROOT / ".env")

from tracerag import (  # noqa: E402
    CrossLingualBM25Retriever,
    LLMQueryTranslator,
    TextChunker,
    parse_path,
)
from tracerag.providers import DeepSeekLLMClient  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="双向翻译中英文查询并执行跨语言 BM25 检索。")
    parser.add_argument("paths", nargs="+", type=Path, help="要建立索引的 Markdown/PDF 文件")
    parser.add_argument("--query", required=True, help="中文或英文检索问题")
    parser.add_argument("--top-k", type=int, default=3, help="返回结果数量")
    parser.add_argument("--candidate-k", type=int, default=20, help="每个词法变体召回的候选数量")
    parser.add_argument("--max-chars", type=int, default=180, help="每个 chunk 的最大字符数")
    parser.add_argument("--overlap-chars", type=int, default=30, help="相邻 chunk 的目标重叠字符数")
    return parser


def main() -> None:
    logging.getLogger("pypdf").setLevel(logging.ERROR)
    args = build_parser().parse_args()

    api_key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("DEEPSEEK_API_KEY is required in .env")

    client = DeepSeekLLMClient(
        api_key,
        api_base=os.environ.get("DEEPSEEK_API_BASE", "https://api.deepseek.com"),
        model=os.environ.get("DEEPSEEK_MODEL", "deepseek-flash"),
    )
    translator = LLMQueryTranslator(client)

    chunker = TextChunker(max_chars=args.max_chars, overlap_chars=args.overlap_chars)
    documents = tuple(document for path in args.paths for document in parse_path(path))
    chunks = tuple(chunk for document in documents for chunk in chunker.chunk(document))

    index_started = perf_counter()
    retriever = CrossLingualBM25Retriever(chunks, translator, candidate_k=args.candidate_k)
    index_seconds = perf_counter() - index_started

    has_chinese = any("\u3400" <= char <= "\u4dbf" or "\u4e00" <= char <= "\u9fff" for char in args.query)
    translation_started = perf_counter()
    translated_query = (
        translator.translate_to_english(args.query)
        if has_chinese
        else translator.translate_to_chinese(args.query)
    )
    translation_seconds = perf_counter() - translation_started

    search_started = perf_counter()
    results = retriever.retrieve(args.query, top_k=args.top_k)
    search_seconds = perf_counter() - search_started

    output = {
        "query": args.query,
        "translated_query": translated_query,
        "retriever": "cross-lingual-bm25-family",
        "lexical_variants": ["original-word", "original-char-ngram", "translated-word", "translated-char-ngram"],
        "family_fusion": "best reciprocal rank",
        "translation_model": client.model,
        "documents": len(documents),
        "chunks": len(chunks),
        "index_seconds": round(index_seconds, 6),
        "translation_seconds": round(translation_seconds, 6),
        "search_seconds_after_translation": round(search_seconds, 6),
        "results": [
            {
                "rank": result.rank,
                "family_score": round(result.score, 8),
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
