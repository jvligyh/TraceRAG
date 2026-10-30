"""固定样例的真实模型冒烟与预热延迟；不等同于完整质量评测。"""
from __future__ import annotations

import argparse
import gc
import json
import statistics
import sys
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tracerag.reranking import CrossEncoderReranker  # noqa: E402
from tracerag.schemas import Chunk, RetrievedChunk  # noqa: E402
from tracerag.transformer_embeddings import TransformerEmbeddingModel  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("embedding", "reranker", "all"), default="all")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("repeats must be positive")
    import torch

    def sync():
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    query = "How are retrieval rankings fused?"
    texts = ["Pandas are bears native to China.", "RRF combines rankings from multiple retrievers."]
    report = {"torch": torch.__version__, "cuda": torch.version.cuda, "repeats": args.repeats, "kind": "smoke-and-latency-not-quality-benchmark", "results": []}
    for kind in ("embedding", "reranker"):
        if args.model not in {"all", kind}:
            continue
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        started = perf_counter()
        if kind == "embedding":
            model = TransformerEmbeddingModel(device=args.device)
            passages = model.embed_passages(texts)
            def run():
                vector = model.embed_query(query)
                return [sum(a*b for a,b in zip(vector, passage, strict=True)) for passage in passages]
        else:
            model = CrossEncoderReranker(device=args.device)
            candidates = tuple(RetrievedChunk(Chunk(str(i), "sample", text, 0, len(text)), 0.0, i + 1) for i, text in enumerate(texts))
            def run():
                return [(row.chunk.chunk_id, row.score) for row in model.rerank(query, candidates)]
        scores = run()
        sync()
        cold_seconds = perf_counter() - started
        durations = []
        for _ in range(args.repeats):
            started = perf_counter()
            scores = run()
            sync()
            durations.append(perf_counter() - started)
        report["results"].append({
            "kind": kind, "model": model.model_name, "revision": model.revision, "device": model.device,
            "query_instruction": getattr(model, "instruction", None),
            "query_template": "Instruct: {instruction}\\nQuery:{query}" if kind == "embedding" else None,
            "pooling": "last-nonpadding-token; left-padding" if kind == "embedding" else "classification-logit",
            "normalized": kind == "embedding", "max_length": model.max_length,
            "official_model_card": f"https://huggingface.co/{model.model_name}",
            "scores": scores, "cold_seconds": cold_seconds,
            "warm_seconds": durations, "p50_seconds": statistics.median(durations),
            "p95_seconds": sorted(durations)[min(len(durations)-1, int(.95*len(durations)))],
            "peak_cuda_allocated_mb": torch.cuda.max_memory_allocated()/1024**2 if model.device == "cuda" else None,
        })
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    output = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
