"""真实图片/纯扫描PDF上传 -> 本地OCR -> DeepSeek回答 -> 来源核验。"""
import hashlib
import json
from pathlib import Path
from time import perf_counter

import httpx

root=Path(__file__).resolve().parents[1]
fixtures=root/"results/ocr-fixtures"
report={"fixture_kind":"deterministic synthetic printed lab records, not a handwriting/scene benchmark","cases":[]}
with httpx.Client(base_url="http://127.0.0.1:8000",timeout=180) as client:
    for path,name,question,expected,expected_page in (
        (fixtures/"chinese-english.png","OCR中英实验记录.png","这张实验记录中的光电转换效率是多少？","23.5",None),
        (fixtures/"scan-with-blank.pdf","OCR扫描实验记录.pdf","这份扫描记录中的催化剂温度是多少？","42",3),
    ):
        started=perf_counter()
        uploaded=client.post("/documents",files={"file":(name,path.read_bytes())})
        uploaded.raise_for_status()
        item=uploaded.json()
        upload_seconds=perf_counter()-started
        assert item["ocr_used"] and item["ocr_pages"]==(1 if expected_page is None else 2)
        if expected_page:
            assert any("第2页" in w for w in item["warnings"])
        asset=item["document_id"]
        response=client.post("/query",json={"question":question,"document_ids":[asset],"top_k":3})
        response.raise_for_status()
        answer=response.json()
        assert answer["status"]=="answered" and expected in answer["answer"]
        assert answer["citations"] and all(c["asset_id"]==asset and c["extraction_method"]=="ocr" for c in answer["citations"])
        if expected_page:
            assert any(c["page_number"]==expected_page and c["source_url"].endswith(f"#page={expected_page}") for c in answer["citations"])
        original=client.get(f"/documents/{asset}/file")
        assert hashlib.sha256(original.content).digest()==hashlib.sha256(path.read_bytes()).digest()
        if not expected_page:
            assert original.headers["content-type"].startswith("image/png")
        case={"filename":name,"upload":item,"ocr_upload_seconds":upload_seconds,
              "response":answer,"original_bytes_match":True}
        report["cases"].append(case)
        print(json.dumps({"file":name,"ocr_pages":item["ocr_pages"],"status":answer["status"],
                          "upload_seconds":round(upload_seconds,3)},ensure_ascii=True),flush=True)
    report["health"]=client.get("/health").json()
output=root/"docs/evidence/ocr-live-validation.json"
output.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
print(output)
