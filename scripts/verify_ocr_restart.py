"""重启服务后核对OCR文本、页码、原图持久化，以及空白图的真实错误提示。"""
import hashlib
import json
from io import BytesIO
from pathlib import Path

import httpx
from PIL import Image


root = Path(__file__).resolve().parents[1]
prior = json.loads((root / "docs/evidence/ocr-live-validation.json").read_text(encoding="utf-8"))
report = {"restart_cases": []}
with httpx.Client(base_url="http://127.0.0.1:8000", timeout=180) as client:
    health = client.get("/health")
    health.raise_for_status()
    assert health.json()["version"] == "0.4.0"
    assert not health.json()["ocr_loaded"], "Run this verification after restarting the service"
    report["health_after_restart"] = health.json()
    for old, expected, page in zip(prior["cases"], ("23.5", "42"), (None, 3), strict=True):
        asset = old["upload"]["document_id"]
        result = client.post("/query", json={"question": old["response"]["question"],
            "document_ids": [asset], "generate": False, "top_k": 3})
        result.raise_for_status()
        evidence = result.json()["results"]
        assert result.json()["status"] == "search_only"
        assert evidence and all(c["asset_id"] == asset and c["extraction_method"] == "ocr" for c in evidence)
        assert any(expected in c["text"] and c["page_number"] == page for c in evidence)
        original = client.get(f"/documents/{asset}/file")
        original.raise_for_status()
        fixture = "chinese-english.png" if page is None else "scan-with-blank.pdf"
        assert hashlib.sha256(original.content).digest() == hashlib.sha256((root / "results/ocr-fixtures" / fixture).read_bytes()).digest()
        report["restart_cases"].append({"document_id": asset, "metadata_and_original_preserved": True,
            "retrieved_pages": [c["page_number"] for c in evidence]})
    assert not client.get("/health").json()["ocr_loaded"], "Querying saved text must not rerun OCR"
    before = client.get("/documents").json()
    buffer = BytesIO()
    with Image.new("RGB", (400, 300), "white") as image:
        image.save(buffer, format="PNG")
    blank = client.post("/documents", files={"file": ("空白图验收.png", buffer.getvalue())})
    assert blank.status_code == 400 and blank.json()["code"] == "ocr_empty"
    assert "清晰截图" in blank.json()["detail"]
    assert client.get("/documents").json() == before
    report["blank_image_error"] = {"status": blank.status_code, **blank.json(), "not_persisted": True}
output = root / "docs/evidence/ocr-restart-validation.json"
output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps({"status": "passed", "restart_cases": len(report["restart_cases"]), "blank_image": 400}))
