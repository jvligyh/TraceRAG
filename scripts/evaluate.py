"""在固定语料/切块/问题上运行检索消融；不会把 AI 草拟集标成已人工核验。"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tracerag.bm25 import BM25Retriever, tokenize_char_ngrams
from tracerag.chunking import TextChunker
from tracerag.crosslingual import CrossLingualBM25Retriever
from tracerag.embeddings import FastEmbedEmbeddingModel
from tracerag.evaluation import EvaluationCase, summarize
from tracerag.fusion import ReciprocalRankFusionRetriever
from tracerag.ingestion import parse_path
from tracerag.retrieval import DenseRetriever


class IdentityTranslator:
    def translate_to_english(self, query: str) -> str:
        return query

    def translate_to_chinese(self, query: str) -> str:
        return query


def read_dataset(corpus: Path, questions: Path, max_chars: int, overlap_chars: int):
    files = sorted(corpus.glob("*.md"))
    if not files:
        raise ValueError("corpus is empty; run scripts/fetch_corpus.py first")
    documents = tuple(document for path in files for document in parse_path(path))
    by_source = {path.stem: parse_path(path)[0] for path in files}
    raw_cases = [json.loads(line) for line in questions.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len({row["id"] for row in raw_cases}) != len(raw_cases):
        raise ValueError("duplicate question id")
    cases = []
    for row in raw_cases:
        relevant = tuple(by_source[name].document_id for name in row["relevant_source_ids"])
        if tuple(row.get("relevant_document_ids", relevant)) != relevant:
            raise ValueError(f"corpus changed since gold labels were bound for {row['id']}; re-review labels")
        if relevant:
            evidence_source = row.get("evidence_source", row["relevant_source_ids"][0])
            if row["evidence"] not in by_source[evidence_source].text:
                raise ValueError(f"evidence quote not present for {row['id']}")
        cases.append(EvaluationCase(row["id"], row["query"], relevant, tuple(row["tags"]), row.get("human_reviewed", False)))
    chunker = TextChunker(max_chars=max_chars, overlap_chars=overlap_chars)
    chunks = tuple(chunk for document in documents for chunk in chunker.chunk(document))
    corpus_hash = hashlib.sha256("\n".join(f"{path.name}:{hashlib.sha256(path.read_bytes()).hexdigest()}" for path in files).encode()).hexdigest()
    return documents, chunks, cases, corpus_hash


def run(args) -> dict:
    documents, chunks, cases, corpus_hash = read_dataset(args.corpus, args.questions, args.max_chars, args.overlap_chars)
    if args.limit:
        cases = cases[:args.limit]
    models = {}
    records = []

    def dense(name: str):
        if name not in models:
            if name == "minilm":
                model = FastEmbedEmbeddingModel()
            else:
                from tracerag.transformer_embeddings import TransformerEmbeddingModel
                model = TransformerEmbeddingModel(device=args.device)
            models[name] = DenseRetriever(chunks, model)
        return models[name]

    def lexical(mode: str):
        key = "translator-" + mode
        if key not in models:
            if mode == "qwen":
                from tracerag.translation import QwenQueryTranslator
                models[key] = QwenQueryTranslator(device=args.device)
            elif mode == "marian":
                from tracerag.translation import MarianQueryTranslator
                class OneWayBaseline:
                    def __init__(self):
                        self.model = MarianQueryTranslator(device=args.device)
                    def translate_to_english(self, query):
                        return self.model.translate_to_english(query)
                    def translate_to_chinese(self, query):
                        return query  # Marian zh-en has no English-to-Chinese capability.
                models[key] = OneWayBaseline()
            elif mode == "deepseek":
                import os
                from dotenv import load_dotenv
                from tracerag.providers import DeepSeekLLMClient
                from tracerag.translation import LLMQueryTranslator
                load_dotenv(ROOT / ".env")
                models[key] = LLMQueryTranslator(DeepSeekLLMClient(os.environ.get("DEEPSEEK_API_KEY", ""), model=os.environ.get("DEEPSEEK_MODEL", "deepseek-flash")))
            else:
                models[key] = IdentityTranslator()
        # 同一模型可常驻，但每个策略在计时前清掉查询翻译结果缓存，避免前一组
        # 把同一批40题预热成“零翻译耗时”，从而破坏延迟对比。
        for method_name in ("translate_to_english", "translate_to_chinese"):
            method = getattr(models[key], method_name)
            if hasattr(method, "cache_clear"):
                method.cache_clear()
        return CrossLingualBM25Retriever(chunks, models[key], candidate_k=args.candidate_k)

    for strategy in args.strategies.split(","):
        strategy = strategy.strip()
        started = time.perf_counter()
        try:
            if strategy == "bm25":
                retriever = BM25Retriever(chunks)
            elif strategy == "char":
                retriever = BM25Retriever(chunks, tokenizer=tokenize_char_ngrams)
            elif strategy == "dense":
                retriever = dense("minilm")
            elif strategy == "dense-qwen":
                retriever = dense("qwen")
            elif strategy in {"lexical-none", "lexical-qwen", "lexical-deepseek", "lexical-marian"}:
                retriever = lexical(strategy.removeprefix("lexical-"))
            elif strategy in {"hybrid-none", "hybrid-qwen", "hybrid-marian", "hybrid-deepseek", "hybrid-qwen-embedding", "hybrid-rerank"}:
                translation_mode = "qwen" if strategy in {"hybrid-qwen", "hybrid-qwen-embedding", "hybrid-rerank"} else strategy.removeprefix("hybrid-")
                retriever = ReciprocalRankFusionRetriever((lexical(translation_mode), dense("qwen" if strategy == "hybrid-qwen-embedding" else "minilm")), candidate_k=args.candidate_k)
                if strategy == "hybrid-rerank":
                    from tracerag.reranking import CrossEncoderReranker
                    base = retriever
                    reranker = CrossEncoderReranker(device=args.device)
                    class RerankedRetriever:
                        def retrieve(self, query, top_k=5):
                            candidates = base.retrieve(query, top_k=args.rerank_depth)
                            return reranker.rerank(query, candidates, top_k=top_k)
                    retriever = RerankedRetriever()
            else:
                raise ValueError(f"unsupported strategy: {strategy}")
            index_seconds = time.perf_counter() - started
            warmup_started = time.perf_counter()
            retriever.retrieve("Warmup: how does evidence retrieval work?", top_k=5)
            warmup_seconds = time.perf_counter() - warmup_started
            details = []
            rankings = []
            timings = []
            for case in cases:
                query_started = time.perf_counter()
                retrieved = retriever.retrieve(case.query, top_k=args.candidate_k)
                duration = time.perf_counter() - query_started
                ranked_ids = tuple(dict.fromkeys(item.chunk.document_id for item in retrieved))[:5]
                rankings.append(ranked_ids)
                timings.append(duration)
                details.append({"id": case.case_id, "query": case.query, "tags": case.tags, "relevant_document_ids": case.relevant_document_ids, "retrieved_document_ids": ranked_ids, "retrieved_chunk_ids": [item.chunk.chunk_id for item in retrieved[:5]], "seconds": duration, "miss_at_3": bool(case.relevant_document_ids) and not bool(set(ranked_ids[:3]) & set(case.relevant_document_ids))})
            record = {"strategy": strategy, "status": "completed", "index_or_load_seconds": index_seconds, "warmup_seconds": warmup_seconds, "metrics": summarize(cases, rankings, timings), "queries": details}
            records.append(record)
            print(json.dumps({"strategy": strategy, "status": "completed", "metrics": record["metrics"]}, ensure_ascii=True), flush=True)
        except Exception as exc:
            # API credentials, authorization headers and arbitrary provider error strings are never logged.
            records.append({"strategy": strategy, "status": "failed", "error_type": type(exc).__name__, "elapsed_seconds": time.perf_counter() - started})
            print(f"{strategy}: failed ({type(exc).__name__})", flush=True)
    versions = {}
    for package in ("fastembed", "jieba", "torch", "transformers", "onnxruntime"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    hardware = {}
    try:
        import torch
        hardware = {"cuda_available": torch.cuda.is_available(), "cuda_build": torch.version.cuda, "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
    except ImportError:
        pass
    model_hash = None
    try:
        from huggingface_hub import snapshot_download
        snapshot = Path(snapshot_download("qdrant/paraphrase-multilingual-MiniLM-L12-v2-onnx-Q", revision=FastEmbedEmbeddingModel.DEFAULT_REVISION, local_files_only=True))
        model_file = snapshot / "model_optimized.onnx"
        if model_file.is_file():
            with model_file.open("rb") as stream:
                model_hash = hashlib.file_digest(stream, "sha256").hexdigest()
    except OSError:
        pass
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset_status": "AI-drafted evidence-checked seed benchmark; requires independent human relevance review; no production-quality claim",
        "metric_scope": "document-level rankings deduplicate chunk hits; recall denominator includes all labeled relevant documents; MRR truncated at 5",
        "corpus": {"files": len(documents), "chunks": len(chunks), "sha256": corpus_hash, "contains": "25 downloaded public abstracts/documentation excerpts + 3 explicitly AI-authored Chinese reading notes"},
        "questions_sha256": hashlib.sha256(args.questions.read_bytes()).hexdigest(),
        "questions": len(cases),
        "settings": {"max_chars": args.max_chars, "overlap_chars": args.overlap_chars, "candidate_k": args.candidate_k, "device_requested": args.device, "fusion_k": 60, "rerank_depth": args.rerank_depth, "translation_cache": "cleared at each strategy boundary"},
        "environment": {"python": platform.python_version(), "platform": platform.platform(), "packages": versions, "hardware": hardware},
        "models": {"minilm": {"name": FastEmbedEmbeddingModel.DEFAULT_MODEL, "repo": "qdrant/paraphrase-multilingual-MiniLM-L12-v2-onnx-Q", "pooling": "mean", "revision": FastEmbedEmbeddingModel.DEFAULT_REVISION, "onnx_sha256": model_hash}, "translation": {"name": "Qwen/Qwen3-0.6B", "revision": "c1899de289a04d12100db370d81485cdf75e47ca"}, "qwen_embedding": {"name": "Qwen/Qwen3-Embedding-0.6B", "revision": "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3", "query_instruction": "Given a search query, retrieve relevant passages that answer the query", "pooling": "last valid token + L2 normalization"}, "reranker": {"name": "BAAI/bge-reranker-v2-m3", "revision": "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"}, "marian_note": "zh-en only; English queries keep original text without translation in the Marian baseline"},
        "results": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=ROOT / "data/corpus")
    parser.add_argument("--questions", type=Path, default=ROOT / "data/evaluation/questions.jsonl")
    parser.add_argument("--strategies", default="bm25,char,dense,hybrid-none")
    parser.add_argument("--max-chars", type=int, default=800)
    parser.add_argument("--overlap-chars", type=int, default=120)
    parser.add_argument("--candidate-k", type=int, default=50)
    parser.add_argument("--rerank-depth", type=int, default=20)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output", type=Path, default=ROOT / "docs/evidence/final-evaluation.json")
    parser.add_argument("--append", action="store_true", help="在同语料/问题/配置的已有报告中追加或更新策略")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if args.candidate_k < 5:
        parser.error("--candidate-k must be at least 5 for Recall@5")
    if args.rerank_depth < 5:
        parser.error("--rerank-depth must be at least 5")
    report = run(args)
    if args.append and args.output.exists():
        previous = json.loads(args.output.read_text(encoding="utf-8"))
        if (previous["corpus"]["sha256"], previous["questions_sha256"], previous["settings"]) != (report["corpus"]["sha256"], report["questions_sha256"], report["settings"]):
            raise ValueError("cannot merge results from different corpus/question/configuration")
        if previous["environment"]["packages"] != report["environment"]["packages"]:
            raise ValueError("cannot merge results from different dependency versions")
        records = {row["strategy"]: row for row in previous["results"]}
        records.update({row["strategy"]: row for row in report["results"]})
        report["results"] = list(records.values())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    rows = ["# 检索消融实测", "", "AI 起草、证据字符串核对过的种子评测集，尚未独立人工复核；这些分数不代表真实用户或长篇论文效果。", "", f"语料 {report['corpus']['files']} 篇，{report['corpus']['chunks']} 个 Chunk，问题 {report['questions']} 条。固定 corpus SHA256：`{report['corpus']['sha256']}`。", "", "| 策略 | 状态 | Recall@1 | Recall@3 | Recall@5 | MRR@5 | P50秒 | P95秒 |", "|---|---|---:|---:|---:|---:|---:|---:|"]
    for item in report["results"]:
        if item["status"] != "completed":
            rows.append(f"| {item['strategy']} | failed:{item['error_type']} | — | — | — | — | — | — |")
            continue
        m = item["metrics"]
        rows.append(f"| {item['strategy']} | completed | {m['recall@1']:.3f} | {m['recall@3']:.3f} | {m['recall@5']:.3f} | {m['mrr@5']:.3f} | {m['query_seconds_p50']:.4f} | {m['query_seconds_p95']:.4f} |")
    rows += ["", "MRR 只看前5个去重文档。无答案题单独报告空检索比例，不计入 Recall 分母；Dense 始终有最近邻，不能把返回文档当作可回答的证据。", "", "完整 JSON 包含每题相关ID/返回ID/失败清单、分组结果和模型环境。引用ID有效率只校验ID是否合法，不替代事实支持度与答案正确性。", "", "尚未执行的模型组合没有分数；更大模型不预设优于基线。"]
    rows += ["", "## Recall@3 未命中案例", "", "| 策略 | 题目ID | 问题 |", "|---|---|---|"]
    for item in report["results"]:
        if item["status"] == "completed":
            for query in item["queries"]:
                if query["miss_at_3"]:
                    rows.append(f"| {item['strategy']} | {query['id']} | {query['query'].replace('|', ' ')} |")
    args.output.with_suffix(".md").write_text("\n".join(rows) + "\n", encoding="utf-8")
    if any(row["status"] != "completed" for row in report["results"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
