from __future__ import annotations

import json
from functools import lru_cache
from typing import Any, Protocol

from .providers.base import LLMClient


class QueryTranslator(Protocol):
    """在中英文之间扩展检索查询的最小接口。"""

    def translate_to_english(self, query: str) -> str:
        """返回只包含译文的英文查询。"""

    def translate_to_chinese(self, query: str) -> str:
        """返回只包含译文的中文查询。"""


class LLMQueryTranslator:
    """使用供应商无关的 LLMClient 翻译并缓存查询。"""

    def __init__(self, client: LLMClient) -> None:
        self._client = client

    @lru_cache(maxsize=256)
    def translate_to_english(self, query: str) -> str:
        return self._translate(query, target_language="English")

    @lru_cache(maxsize=256)
    def translate_to_chinese(self, query: str) -> str:
        return self._translate(query, target_language="Chinese")

    def _translate(self, query: str, *, target_language: str) -> str:
        query = query.strip()
        if not query:
            raise ValueError("query must not be empty")

        prompt = (
            f"Translate the following search query into concise {target_language} for lexical document retrieval. "
            "Preserve model names, abbreviations, formulas, and technical terminology. "
            "Return only the translated query, without labels, quotation marks, explanations, or Markdown.\n"
            f"QUERY_JSON: {json.dumps(query, ensure_ascii=False)}"
        )
        translation = _clean_translation(self._client.complete(prompt))
        if not translation:
            raise RuntimeError("query translator returned empty text")
        return translation


class MarianQueryTranslator:
    """使用本地 MarianMT 模型执行确定性中译英。"""

    DEFAULT_MODEL = "Helsinki-NLP/opus-mt-zh-en"
    DEFAULT_REVISION = "cf109095479db38d6df799875e34039d4938aaa6"

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        *,
        revision: str = DEFAULT_REVISION,
        device: str = "auto",
        max_input_tokens: int = 256,
        max_new_tokens: int = 128,
        num_beams: int = 4,
    ) -> None:
        model_name = model_name.strip()
        revision = revision.strip()
        if not model_name:
            raise ValueError("model_name must not be empty")
        if not revision:
            raise ValueError("revision must not be empty")
        if max_input_tokens < 1 or max_new_tokens < 1:
            raise ValueError("token limits must be positive")
        if num_beams < 1:
            raise ValueError("num_beams must be positive")

        try:
            import torch
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                'local translation dependencies are missing; run: python -m pip install -e ".[local-translation]"'
            ) from exc

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device not in {"cpu", "cuda"}:
            raise ValueError("device must be 'auto', 'cpu', or 'cuda'")
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available to PyTorch")

        self.model_name = model_name
        self.revision = revision
        self.device = device
        self.max_input_tokens = max_input_tokens
        self.max_new_tokens = max_new_tokens
        self.num_beams = num_beams
        self._torch = torch
        self._tokenizer = _load_pretrained(AutoTokenizer, model_name, revision)
        self._model = _load_pretrained(AutoModelForSeq2SeqLM, model_name, revision)
        self._model.to(device)
        self._model.eval()

    @lru_cache(maxsize=256)
    def translate_to_english(self, query: str) -> str:
        query = query.strip()
        if not query:
            raise ValueError("query must not be empty")

        encoded = self._tokenizer(
            query,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_input_tokens,
        )
        encoded = {name: tensor.to(self.device) for name, tensor in encoded.items()}
        with self._torch.inference_mode():
            generated = self._model.generate(
                **encoded,
                max_new_tokens=self.max_new_tokens,
                num_beams=self.num_beams,
                do_sample=False,
                renormalize_logits=True,
            )
        translation = self._tokenizer.batch_decode(generated, skip_special_tokens=True)[0].strip()
        if not translation:
            raise RuntimeError("local query translator returned empty text")
        return translation

    def translate_to_chinese(self, query: str) -> str:
        raise RuntimeError(
            "Helsinki-NLP/opus-mt-zh-en only supports Chinese-to-English; "
            "use --translator qwen or --translator deepseek for English-to-Chinese"
        )


class QwenQueryTranslator:
    """使用本地 Qwen3 小模型翻译技术检索查询。"""

    DEFAULT_MODEL = "Qwen/Qwen3-0.6B"
    DEFAULT_REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        *,
        revision: str = DEFAULT_REVISION,
        device: str = "auto",
        max_input_tokens: int = 256,
        max_new_tokens: int = 96,
    ) -> None:
        model_name = model_name.strip()
        revision = revision.strip()
        if not model_name:
            raise ValueError("model_name must not be empty")
        if not revision:
            raise ValueError("revision must not be empty")
        if max_input_tokens < 1 or max_new_tokens < 1:
            raise ValueError("token limits must be positive")

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                'local translation dependencies are missing; run: python -m pip install -e ".[local-translation]"'
            ) from exc

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device not in {"cpu", "cuda"}:
            raise ValueError("device must be 'auto', 'cpu', or 'cuda'")
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available to PyTorch")

        self.model_name = model_name
        self.revision = revision
        self.device = device
        self.max_input_tokens = max_input_tokens
        self.max_new_tokens = max_new_tokens
        self._torch = torch
        self._tokenizer = _load_pretrained(AutoTokenizer, model_name, revision)
        model_dtype = torch.float32 if device == "cpu" else "auto"
        self._model = _load_pretrained(AutoModelForCausalLM, model_name, revision, dtype=model_dtype)
        self._model.to(device)
        self._model.eval()

    @lru_cache(maxsize=256)
    def translate_to_english(self, query: str) -> str:
        return self._translate(query, target_language="English")

    @lru_cache(maxsize=256)
    def translate_to_chinese(self, query: str) -> str:
        return self._translate(query, target_language="Chinese")

    def _translate(self, query: str, *, target_language: str) -> str:
        query = query.strip()
        if not query:
            raise ValueError("query must not be empty")

        messages = [
            {
                "role": "system",
                "content": (
                    f"You translate academic search queries into concise {target_language} retrieval queries. "
                    "Preserve model names, abbreviations, formulas, and technical terminology exactly when possible. "
                    f"Return only the {target_language} translation, with no explanation or quotation marks."
                ),
            },
            {"role": "user", "content": query},
        ]
        encoded = self._tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            enable_thinking=False,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_input_tokens,
        )
        encoded = {name: tensor.to(self.device) for name, tensor in encoded.items()}
        input_length = encoded["input_ids"].shape[1]
        with self._torch.inference_mode():
            generated = self._model.generate(
                **encoded,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
                pad_token_id=self._tokenizer.eos_token_id,
            )
        translation = self._tokenizer.decode(
            generated[0][input_length:],
            skip_special_tokens=True,
        )
        translation = _clean_translation(translation)
        if not translation:
            raise RuntimeError("Qwen query translator returned empty text")
        return translation


def _clean_translation(text: str) -> str:
    text = text.strip()
    if text.startswith("```") and text.endswith("```"):
        lines = text.splitlines()
        if len(lines) >= 3:
            text = "\n".join(lines[1:-1]).strip()

    for prefix in ("translation:", "english:", "英文：", "英文:"):
        if text.lower().startswith(prefix.lower()):
            text = text[len(prefix) :].strip()
            break

    quote_pairs = (("\"", "\""), ("'", "'"), ("“", "”"), ("‘", "’"))
    for left, right in quote_pairs:
        if len(text) >= 2 and text.startswith(left) and text.endswith(right):
            return text[len(left) : -len(right)].strip()
    return text


def _load_pretrained(factory: Any, model_name: str, revision: str, **kwargs: Any) -> Any:
    """优先读取本地缓存；首次缺失时才访问 Hugging Face 下载。"""

    local_path = _resolve_local_snapshot(model_name, revision)
    if local_path is not None:
        try:
            # 传本地目录而不是仓库名，避免部分 Transformers 版本即使设置了
            # local_files_only=True 仍调用 Hugging Face API 检查模型类型。
            return factory.from_pretrained(local_path, local_files_only=True, **kwargs)
        except (OSError, AttributeError):
            # 缓存可能只有配置而没有完整权重，此时继续走联网下载分支。
            pass
    return factory.from_pretrained(model_name, revision=revision, **kwargs)


def _resolve_local_snapshot(model_name: str, revision: str) -> str | None:
    try:
        from huggingface_hub import snapshot_download
        from huggingface_hub.errors import LocalEntryNotFoundError
    except ImportError:
        return None

    try:
        return snapshot_download(model_name, revision=revision, local_files_only=True)
    except LocalEntryNotFoundError:
        return None
