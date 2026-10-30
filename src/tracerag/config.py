from __future__ import annotations

import os
import math
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv


# Editable installs use the repository root. Wheel installs keep state in a
# stable user directory unless TRACERAG_HOME explicitly selects another home.
_SOURCE_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = _SOURCE_ROOT if (_SOURCE_ROOT / "pyproject.toml").is_file() else Path.home() / ".tracerag"


@dataclass(frozen=True)
class Settings:
    """单进程本地服务配置。密钥不参与 repr、状态接口或日志。"""

    data_dir: Path = PROJECT_ROOT / "data/runtime"
    retrieval: str = "hybrid"
    translator: str = "qwen"
    embedding: str = "minilm"
    provider: str = "extractive"
    reranker: str = "none"
    device: str = "auto"
    chunk_chars: int = 800
    overlap_chars: int = 120
    candidate_k: int = 20
    max_upload_bytes: int = 10 * 1024 * 1024
    max_context_chars: int = 16000
    ocr_enabled: bool = True
    ocr_max_pages: int = 30
    ocr_max_image_pixels: int = 20_000_000
    ocr_min_confidence: float = 0.5
    timeout_seconds: float = 60.0
    api_retries: int = 2
    deepseek_model: str = "deepseek-flash"
    deepseek_base: str = "https://api.deepseek.com"
    api_key: str = field(default="", repr=False)

    def __post_init__(self) -> None:
        for value, allowed in (
            (self.retrieval, {"bm25", "dense", "hybrid"}),
            (self.translator, {"none", "qwen", "deepseek", "marian"}),
            (self.embedding, {"minilm", "qwen"}),
            (self.provider, {"extractive", "deepseek"}),
            (self.reranker, {"none", "bge"}),
            (self.device, {"auto", "cpu", "cuda"}),
        ):
            if value not in allowed:
                raise ValueError(f"invalid configuration choice: {value}")
        if not 0 <= self.overlap_chars < self.chunk_chars:
            raise ValueError("overlap must be smaller than chunk size")
        if min(self.candidate_k, self.max_upload_bytes, self.max_context_chars, self.timeout_seconds) <= 0:
            raise ValueError("limits must be positive")
        if not math.isfinite(self.timeout_seconds):
            raise ValueError("timeout must be finite")
        if not 0 <= self.api_retries <= 5:
            raise ValueError("api_retries must be between 0 and 5")
        if not 1<=self.ocr_max_pages<=200 or not 100_000<=self.ocr_max_image_pixels<=50_000_000:
            raise ValueError("OCR page/pixel limits are out of range")
        if not math.isfinite(self.ocr_min_confidence) or not 0<=self.ocr_min_confidence<=1:
            raise ValueError("OCR confidence threshold must be between 0 and 1")

    @classmethod
    def from_env(cls) -> Settings:
        root = Path(os.environ.get("TRACERAG_HOME", str(PROJECT_ROOT))).expanduser().resolve()
        # Never walk upward from the caller's working directory to find .env.
        load_dotenv(root / ".env", override=False)
        configured_data = Path(os.getenv("TRACERAG_DATA_DIR", "data/runtime")).expanduser()
        data_dir = configured_data if configured_data.is_absolute() else root / configured_data
        key = os.getenv("DEEPSEEK_API_KEY", "").strip()
        return cls(
            data_dir=data_dir.resolve(),
            retrieval=os.getenv("TRACERAG_RETRIEVAL", "hybrid"),
            translator=os.getenv("TRACERAG_TRANSLATOR", "qwen"),
            embedding=os.getenv("TRACERAG_EMBEDDING", "minilm"),
            provider=os.getenv("TRACERAG_PROVIDER", "deepseek" if key else "extractive"),
            reranker=os.getenv("TRACERAG_RERANKER", "none"),
            device=os.getenv("TRACERAG_DEVICE", "auto"),
            chunk_chars=int(os.getenv("TRACERAG_CHUNK_CHARS", "800")),
            overlap_chars=int(os.getenv("TRACERAG_OVERLAP_CHARS", "120")),
            candidate_k=int(os.getenv("TRACERAG_CANDIDATE_K", "20")),
            max_upload_bytes=int(os.getenv("TRACERAG_MAX_UPLOAD_BYTES", str(10 * 1024 * 1024))),
            max_context_chars=int(os.getenv("TRACERAG_MAX_CONTEXT_CHARS", "16000")),
            ocr_enabled=_env_bool("TRACERAG_OCR_ENABLED",True),
            ocr_max_pages=int(os.getenv("TRACERAG_OCR_MAX_PAGES","30")),
            ocr_max_image_pixels=int(os.getenv("TRACERAG_OCR_MAX_IMAGE_PIXELS","20000000")),
            ocr_min_confidence=float(os.getenv("TRACERAG_OCR_MIN_CONFIDENCE","0.5")),
            timeout_seconds=float(os.getenv("TRACERAG_TIMEOUT_SECONDS", "60")),
            api_retries=int(os.getenv("TRACERAG_API_RETRIES", "2")),
            deepseek_model=os.getenv("DEEPSEEK_MODEL", "deepseek-flash"),
            deepseek_base=os.getenv("DEEPSEEK_API_BASE", "https://api.deepseek.com"),
            api_key=key,
        )


def _env_bool(name: str, default: bool) -> bool:
    value=os.getenv(name,str(default)).strip().lower()
    if value in {"true","1","yes","on"}:
        return True
    if value in {"false","0","no","off"}:
        return False
    raise ValueError(f"{name} must be true or false")
