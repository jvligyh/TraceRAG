from __future__ import annotations
import json
import platform
import importlib.metadata
from pathlib import Path
from datetime import datetime,timezone

packages = ("tracerag","fastembed","jieba","openai","pypdf","fastapi","starlette","uvicorn",
            "python-multipart","httpx","torch","transformers","onnxruntime","huggingface-hub",
            "rapidocr","pypdfium2","Pillow","opencv-python-headless")
report={"recorded_at":datetime.now(timezone.utc).isoformat(),"python":platform.python_version(),
        "platform":platform.platform(),"packages":{p:importlib.metadata.version(p) for p in packages}}
try:
    import torch
    report["cuda"]={"available":torch.cuda.is_available(),"build":torch.version.cuda,
                    "gpu":torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
except ImportError:
    report["cuda"]={"available":False}
root=Path(__file__).resolve().parents[1]
from tracerag.ocr import RapidOCREngine
report["ocr"]=RapidOCREngine(cache_dir=root/"data/models/ocr").metadata()
path=root/"docs/evidence/environment.json"
path.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
print(path)
