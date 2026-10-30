"""对运行中的本地服务做回答/拒答验收，记录待人工审核答案，不自动冒称事实正确率。"""
from __future__ import annotations

import argparse
import json
import statistics
from datetime import datetime,timezone
from pathlib import Path

import httpx

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url",default="http://127.0.0.1:8000")
    parser.add_argument("--limit",type=int)
    parser.add_argument("--output",type=Path,default=ROOT/"docs/evidence/answer-evaluation.json")
    args=parser.parse_args()
    cases=[json.loads(line) for line in (ROOT/"data/evaluation/questions.jsonl").read_text(encoding="utf-8").splitlines() if line]
    if args.limit:
        cases=cases[:args.limit]
    rows=[]
    with httpx.Client(base_url=args.url,timeout=180) as client:
        state=client.get("/health").json()
        for case in cases:
            response=client.post("/query",json={"question":case["query"],"top_k":5,"answer_language":"Chinese"})
            data=response.json()
            evidence={e["chunk_id"] for e in data.get("results",[])}
            citations=data.get("citations",[])
            valid=all(c["chunk_id"] in evidence for c in citations)
            rows.append({"id":case["id"],"query":case["query"],"expected_answerable":bool(case["relevant_document_ids"]),
                "reference_document_ids":case["relevant_document_ids"],"http_status":response.status_code,
                "citation_ids_valid":valid,"human_answer_review":"pending","response":data})
            print(case["id"],response.status_code,data.get("status"),flush=True)
    unknown=[r for r in rows if not r["expected_answerable"]]
    answered=[r for r in rows if r["response"].get("status")=="answered"]
    report={"created_at":datetime.now(timezone.utc).isoformat(),"service":state,
        "scope":"AI-drafted seed cases; citation-ID validation and abstention only, not factual correctness",
        "questions":len(rows),"answered":len(answered),
        "http_failures":sum(r["http_status"]!=200 for r in rows),
        "citation_id_valid_rate":sum(r["citation_ids_valid"] for r in answered)/len(answered) if answered else None,
        "unanswerable_cases":len(unknown),
        "unanswerable_abstention_rate":sum(r["response"].get("status")=="insufficient_evidence" for r in unknown)/len(unknown) if unknown else None,
        "results":rows}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps({k:v for k,v in report.items() if k!="results"},ensure_ascii=True))
    if report["http_failures"]:
        raise SystemExit(1)


if __name__=="__main__":
    main()
