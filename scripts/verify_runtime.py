"""真实端到端冒烟和常驻缓存性能样本；只有 --live 才访问DeepSeek。"""
from __future__ import annotations

import argparse
import json
import tempfile
from dataclasses import replace
from pathlib import Path

from tracerag.config import Settings
from tracerag.service import RAGService


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--live",action="store_true")
    parser.add_argument("--agent",action="store_true")
    parser.add_argument("--output",type=Path)
    args=parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="tracerag-verify-") as tmp:
        settings=replace(Settings.from_env(),data_dir=Path(tmp),provider="deepseek" if args.live else "extractive")
        service=RAGService(settings)
        source=Path(__file__).resolve().parents[1]/"data/demo/rag_evaluation.md"
        service.ingest_bytes(source.name,source.read_bytes())
        first=service.query("为什么要分别评估检索和生成？",top_k=2)
        warm=service.query("为什么要分别评估检索和生成？",top_k=2,generate=False)
        report={"live":args.live,"first_query":first,"warm_search":warm,
                "cache_note":"Warm query reuses model, index and translation cache; generation disabled."}
        if args.agent:
            report["agent"]=service.agent("为什么要分别评估检索和生成？",max_steps=4)
        payload=json.dumps(report,ensure_ascii=False,indent=2)
        if args.output:
            args.output.parent.mkdir(parents=True,exist_ok=True)
            args.output.write_text(payload,encoding="utf-8")
        print(payload)


if __name__=="__main__":
    main()
