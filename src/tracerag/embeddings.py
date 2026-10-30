from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Protocol
from pathlib import Path
import warnings


Embedding = tuple[float, ...]


class EmbeddingModel(Protocol):
    """将查询和文档片段转换为稠密向量的统一接口。"""

    def embed_passages(self, texts: Sequence[str]) -> tuple[Embedding, ...]:
        """批量生成文档片段向量。"""

    def embed_query(self, text: str) -> Embedding:
        """生成一个查询向量。"""


class FastEmbedEmbeddingModel:
    """使用 FastEmbed 在本地运行中英文 Embedding 模型。"""

    DEFAULT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    DEFAULT_REVISION = "faf4aa4225822f3bc6376869cb1164e8e3feedd0"

    def __init__(self, model_name: str = DEFAULT_MODEL) -> None:
        from fastembed import TextEmbedding

        model_name = model_name.strip()
        if not model_name:
            raise ValueError("model_name must not be empty")
        self.model_name = model_name
        kwargs = {}
        if model_name == self.DEFAULT_MODEL:
            from huggingface_hub import snapshot_download
            from huggingface_hub.errors import LocalEntryNotFoundError
            self.revision = self.DEFAULT_REVISION
            repo = "qdrant/paraphrase-multilingual-MiniLM-L12-v2-onnx-Q"
            try:
                path = snapshot_download(repo,revision=self.revision,local_files_only=True)
                if not (Path(path)/"model_optimized.onnx").is_file():
                    raise LocalEntryNotFoundError("incomplete snapshot")
            except LocalEntryNotFoundError:
                path = snapshot_download(repo,revision=self.revision,
                    allow_patterns=["*.json","model_optimized.onnx","*.model","*.txt"])
            kwargs["specific_model_path"] = path
        # fastembed 0.8 已固定采用 mean pooling；此处明确接受，而不是旧CLS行为。
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore",message="The model .* now uses mean pooling.*",category=UserWarning)
            self._model = TextEmbedding(model_name=model_name,**kwargs)

    def embed_passages(self, texts: Sequence[str]) -> tuple[Embedding, ...]:
        if not texts:
            return ()
        return _to_embeddings(self._model.passage_embed(list(texts)))

    def embed_query(self, text: str) -> Embedding:
        text = text.strip()
        if not text:
            raise ValueError("embedding query must not be empty")
        embeddings = _to_embeddings(self._model.query_embed([text]))
        if len(embeddings) != 1:
            raise ValueError("embedding model must return exactly one query vector")
        return embeddings[0]


def _to_embeddings(vectors: Iterable[Iterable[float]]) -> tuple[Embedding, ...]:
    return tuple(tuple(float(value) for value in vector) for vector in vectors)
