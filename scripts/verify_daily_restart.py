"""在重新启动服务后核对完整PDF原文件和限定范围检索。"""
import hashlib
import json
from pathlib import Path

import httpx

root=Path(__file__).resolve().parents[1]
previous=json.loads((root/"docs/evidence/real-paper-acceptance.json").read_text(encoding="utf-8"))
asset=previous["upload"]["document_id"]
with httpx.Client(base_url="http://127.0.0.1:8000",timeout=180) as client:
    health=client.get("/health").json()
    documents=client.get("/documents").json()["documents"]
    document=next(d for d in documents if d["document_id"]==asset)
    assert document["original_available"]
    original=client.get(f"/documents/{asset}/file")
    assert hashlib.sha256(original.content).hexdigest()==previous["sha256"]
    result=client.post("/query",json={"question":"RAG-Token","document_ids":[asset],"generate":False})
    result.raise_for_status()
    data=result.json()
    assert data["results"] and all(r["asset_id"]==asset for r in data["results"])
    report={"status":"passed","health_before_query":health,"document":document,
            "original_sha256_matches":True,"selected_scope_kept":True,"search_response":data}
output=root/"docs/evidence/daily-restart-validation.json"
output.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
print(json.dumps({"status":report["status"],"original_sha256_matches":True,"selected_scope_kept":True,
                  "models_loaded_before_query":health["models_loaded"]}))
