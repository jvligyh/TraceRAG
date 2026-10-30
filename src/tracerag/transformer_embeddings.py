from __future__ import annotations

from collections.abc import Sequence
import json
from pathlib import Path
from typing import Any

from .embeddings import Embedding


class TransformerEmbeddingModel:
    """Qwen3 Embedding：query instruction、最后一个有效 token 池化、L2 归一化。"""

    DEFAULT_MODEL = "Qwen/Qwen3-Embedding-0.6B"
    DEFAULT_REVISION = "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"
    DEFAULT_INSTRUCTION = "Given a search query, retrieve relevant passages that answer the query"

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        *,
        revision: str = DEFAULT_REVISION,
        device: str = "auto",
        max_length: int = 1024,
        batch_size: int = 8,
        instruction: str = DEFAULT_INSTRUCTION,
    ) -> None:
        if not model_name.strip() or not revision.strip() or not instruction.strip():
            raise ValueError("model, revision and instruction must not be empty")
        if max_length < 1 or batch_size < 1:
            raise ValueError("max_length and batch_size must be positive")
        if device not in {"auto", "cpu", "cuda"}:
            raise ValueError("device must be auto, cpu or cuda")
        self.model_name, self.revision = model_name, revision
        self.device = device
        self.max_length, self.batch_size = max_length, batch_size
        self.instruction = instruction
        self._model: Any = None
        self._tokenizer: Any = None

    def _load(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModel, AutoTokenizer

        device = "cuda" if self.device == "auto" and torch.cuda.is_available() else self.device
        if device == "auto":
            device = "cpu"
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        self.device = device
        self._torch = torch
        snapshot = prepare_snapshot(self.model_name, self.revision)
        self._tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True, padding_side="left")
        dtype = torch.float16 if device == "cuda" else torch.float32
        model = AutoModel.from_pretrained(snapshot, local_files_only=True, dtype=dtype)
        model.to(device).eval()
        self._model = model

    def embed_passages(self, texts: Sequence[str]) -> tuple[Embedding, ...]:
        if any(not text.strip() for text in texts):
            raise ValueError("embedding passages must not be empty")
        return self._encode(texts) if texts else ()

    def embed_query(self, text: str) -> Embedding:
        if not text.strip():
            raise ValueError("embedding query must not be empty")
        return self._encode((f"Instruct: {self.instruction}\nQuery:{text.strip()}",))[0]

    def _encode(self, texts: Sequence[str]) -> tuple[Embedding, ...]:
        self._load()
        vectors: list[Embedding] = []
        for start in range(0, len(texts), self.batch_size):
            encoded = self._tokenizer(
                list(texts[start : start + self.batch_size]), padding=True,
                truncation=True, max_length=self.max_length, return_tensors="pt",
            )
            encoded = {name: value.to(self.device) for name, value in encoded.items()}
            with self._torch.inference_mode():
                hidden = self._model(**encoded).last_hidden_state
                # Tokenizer 固定左 padding，所以最后一列总是最后一个有效 token。
                pooled = hidden[:, -1].float()
                normalized = self._torch.nn.functional.normalize(pooled, p=2, dim=1)
            vectors.extend(tuple(float(x) for x in row) for row in normalized.cpu().tolist())
        return tuple(vectors)


def prepare_snapshot(model_name: str, revision: str) -> str:
    """仅下载推理文件；不把只有 tokenizer 的半成品缓存当成完整模型。"""
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import LocalEntryNotFoundError

    try:
        local = snapshot_download(model_name, revision=revision, local_files_only=True)
        path = Path(local)
        if (path / "config.json").is_file() and _complete_safetensors(path / "model.safetensors"):
            return local
    except LocalEntryNotFoundError:
        pass
    return snapshot_download(
        model_name, revision=revision,
        allow_patterns=["config.json", "model.safetensors", "tokenizer*", "special_tokens_map.json", "*.txt", "vocab.json", "sentencepiece.bpe.model"],
        max_workers=2,
    )


def _complete_safetensors(path: Path) -> bool:
    """快速检查文件长度，避免把中断下载留下的部分权重交给模型。"""
    try:
        with path.open("rb") as handle:
            header_size = int.from_bytes(handle.read(8), "little")
            if not 2 <= header_size <= 16 * 1024 * 1024:
                return False
            header = json.loads(handle.read(header_size))
        end = max(item["data_offsets"][1] for key, item in header.items() if key != "__metadata__")
        return path.stat().st_size == 8 + header_size + end
    except (OSError, ValueError, TypeError, KeyError):
        return False
