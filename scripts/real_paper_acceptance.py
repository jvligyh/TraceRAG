"""对本机运行服务做完整PDF、单文档范围、多轮追问与引用页码验收。"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import httpx
from pypdf import PdfReader

from tracerag.ingestion import normalize_text


ROOT=Path(__file__).resolve().parents[1]
path=ROOT/"results/real-paper/rag-original.pdf"
reader=PdfReader(path)
raw=path.read_bytes()
report={"source_url":"https://arxiv.org/pdf/2005.11401v4","sha256":hashlib.sha256(raw).hexdigest(),
        "pdf_pages":len(reader.pages),"responses":[]}
with httpx.Client(base_url="http://127.0.0.1:8000",timeout=240) as client:
    uploaded=client.post("/documents",files={"file":("RAG原始论文.pdf",raw,"application/pdf")})
    uploaded.raise_for_status()
    asset=uploaded.json()["document_id"]
    report["upload"]=uploaded.json()
    original=client.get(f"/documents/{asset}/file")
    assert original.content==raw and original.headers["content-type"].startswith("application/pdf")
    history=[]
    questions=["这篇论文提出的 RAG-Sequence 和 RAG-Token 有什么区别？", "其中后者可以在每个输出 token 使用不同的文档吗？",
               "请给出论文实验训练耗电量的确切数值，单位 kWh。"]
    for question in questions:
        result=client.post("/query",json={"question":question,"top_k":5,"document_ids":[asset],
                                          "history":history,"answer_language":"Chinese"})
        result.raise_for_status()
        response=result.json()
        assert all(e["asset_id"]==asset for e in response["results"])
        for citation in response["citations"]:
            page=citation["page_number"]
            assert 1<=page<=len(reader.pages)
            text=normalize_text(reader.pages[page-1].extract_text() or "")
            assert text[citation["start_char"]:citation["end_char"]]==citation["text"]
            assert citation["source_url"]==f"/documents/{asset}/file#page={page}"
        report["responses"].append(response)
        history.extend([{"role":"user","content":question},{"role":"assistant","content":response["answer"][:4000]}])
        print(json.dumps({"status":response["status"],"query":response["retrieval_query"],
                          "pages":[c["page_number"] for c in response["citations"]]},ensure_ascii=True),flush=True)
    assert report["responses"][0]["status"]=="answered"
    assert report["responses"][1]["status"]=="answered"
    assert report["responses"][1]["history_used"]
    assert report["responses"][2]["status"]=="insufficient_evidence"
    empty=client.post("/query",json={"question":"RAG","document_ids":[],"generate":False}).json()
    assert empty["results"]==[]
    report["empty_scope_keeps_empty"]=True
output=ROOT/"docs/evidence/real-paper-acceptance.json"
output.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
print(output)
