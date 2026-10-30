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
    DenseRetriever,
    FastEmbedEmbeddingModel,
    LLMQueryTranslator,
    MarianQueryTranslator,
    QueryTranslator,
    QwenQueryTranslator,
    TextChunker,
    parse_path,
    reciprocal_rank_fusion,
)
from tracerag.providers import DeepSeekLLMClient  # noqa: E402


class IdentityTranslator:
    """关闭查询翻译时使用；返回原查询，因此不会新增查询变体。"""

    def translate_to_english(self, query: str) -> str:
        return query

    def translate_to_chinese(self, query: str) -> str:
        return query


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="融合 BM25 家族和 Dense 排名的 Day 5 RRF 演示。")
    parser.add_argument("paths", nargs="+", type=Path, help="要建立索引的 Markdown/PDF 文件")
    parser.add_argument("--query", required=True, help="要检索的问题")
    parser.add_argument("--top-k", type=int, default=3, help="最终返回结果数量")
    parser.add_argument("--candidate-k", type=int, default=20, help="每个检索器提供的候选数量")
    parser.add_argument("--fusion-k", type=int, default=60, help="RRF 排名平滑常数")
    parser.add_argument("--max-chars", type=int, default=180, help="每个 chunk 的最大字符数")
    parser.add_argument("--overlap-chars", type=int, default=30, help="相邻 chunk 的目标重叠字符数")
    parser.add_argument("--model", default=FastEmbedEmbeddingModel.DEFAULT_MODEL, help="FastEmbed 模型名称")
    parser.add_argument(
        "--translator",
        choices=("qwen", "marian", "deepseek", "none"),
        default="qwen",
        help="中英双向查询翻译方式；默认使用本地 Qwen3-0.6B",
    )
    parser.add_argument("--translation-device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser


def main() -> None:
    logging.getLogger("pypdf").setLevel(logging.ERROR)
    args = build_parser().parse_args()
    if args.top_k < 1 or args.candidate_k < 1 or args.fusion_k < 1:
        raise SystemExit("top-k, candidate-k and fusion-k must be positive")

    chunker = TextChunker(max_chars=args.max_chars, overlap_chars=args.overlap_chars)
    documents = tuple(document for path in args.paths for document in parse_path(path))
    chunks = tuple(chunk for document in documents for chunk in chunker.chunk(document))

    translator_load_started = perf_counter()
    translator, translation_model = _build_translator(args)
    translator_load_seconds = perf_counter() - translator_load_started
    translation_started = perf_counter()
    translated_query = (
        _translate_to_other_language(translator, args.query)
        if translation_model is not None
        else None
    )
    translation_seconds = perf_counter() - translation_started

    index_started = perf_counter()
    lexical_retriever = CrossLingualBM25Retriever(
        chunks,
        translator,
        candidate_k=args.candidate_k,
    )
    embedding_model = FastEmbedEmbeddingModel(model_name=args.model)
    dense_retriever = DenseRetriever(chunks, embedding_model)
    index_seconds = perf_counter() - index_started

    depth = max(args.top_k, args.candidate_k)
    lexical_started = perf_counter()
    lexical_results = lexical_retriever.retrieve(args.query, top_k=depth)
    lexical_seconds = perf_counter() - lexical_started

    dense_started = perf_counter()
    dense_results = dense_retriever.retrieve(args.query, top_k=depth)
    dense_seconds = perf_counter() - dense_started

    fusion_started = perf_counter()
    fused_results = reciprocal_rank_fusion(
        (lexical_results, dense_results),
        top_k=args.top_k,
        fusion_k=args.fusion_k,
    )
    fusion_seconds = perf_counter() - fusion_started

    lexical_ranks = {result.chunk.chunk_id: result.rank for result in lexical_results}
    dense_ranks = {result.chunk.chunk_id: result.rank for result in dense_results}
    output = {
        "query": args.query,
        "translated_query": translated_query,
        "translation_model": translation_model,
        "embedding_model": embedding_model.model_name,
        "documents": len(documents),
        "chunks": len(chunks),
        "candidate_k": args.candidate_k,
        "fusion_k": args.fusion_k,
        "index_seconds": round(index_seconds, 6),
        "translator_load_seconds": round(translator_load_seconds, 6),
        "translation_seconds": round(translation_seconds, 6),
        "lexical_search_seconds": round(lexical_seconds, 6),
        "dense_search_seconds": round(dense_seconds, 6),
        "fusion_seconds": round(fusion_seconds, 6),
        "results": [
            {
                "rank": result.rank,
                "rrf_score": round(result.score, 8),
                "bm25_family_rank": lexical_ranks.get(result.chunk.chunk_id),
                "dense_rank": dense_ranks.get(result.chunk.chunk_id),
                "chunk_id": result.chunk.chunk_id,
                "document_id": result.chunk.document_id,
                "source": result.chunk.source,
                "page_number": result.chunk.page_number,
                "text": result.chunk.text,
            }
            for result in fused_results
        ],
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))


def _build_translator(
    args: argparse.Namespace,
) -> tuple[IdentityTranslator | LLMQueryTranslator | MarianQueryTranslator | QwenQueryTranslator, str | None]:
    if args.translator == "none":
        return IdentityTranslator(), None

    if args.translator == "qwen":
        translator = QwenQueryTranslator(device=args.translation_device)
        model = f"{translator.model_name}@{translator.revision} ({translator.device})"
        return translator, model

    if args.translator == "marian":
        translator = MarianQueryTranslator(device=args.translation_device)
        model = f"{translator.model_name}@{translator.revision} ({translator.device})"
        return translator, model

    api_key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("DEEPSEEK_API_KEY is required when --translator deepseek is used")

    client = DeepSeekLLMClient(
        api_key,
        api_base=os.environ.get("DEEPSEEK_API_BASE", "https://api.deepseek.com"),
        model=os.environ.get("DEEPSEEK_MODEL", "deepseek-flash"),
    )
    return LLMQueryTranslator(client), client.model


def _contains_han(text: str) -> bool:
    return any("\u3400" <= char <= "\u4dbf" or "\u4e00" <= char <= "\u9fff" for char in text)


def _translate_to_other_language(translator: QueryTranslator, query: str) -> str:
    if _contains_han(query):
        return translator.translate_to_english(query)
    return translator.translate_to_chinese(query)


if __name__ == "__main__":
    main()
