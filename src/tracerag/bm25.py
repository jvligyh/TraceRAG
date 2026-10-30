from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Callable, Sequence
from functools import lru_cache

import jieba

from .schemas import Chunk, RetrievedChunk


Tokenizer = Callable[[str], Sequence[str]]
_TOKEN_PATTERN = re.compile(r"[a-z0-9]+(?:[._#@-][a-z0-9]+)*\+*|[\u3400-\u4dbf\u4e00-\u9fff]+", re.IGNORECASE)
_RAG_DOMAIN_TERMS = (
    "大语言模型",
    "倒排索引",
    "多路召回",
    "混合检索",
    "检索增强生成",
    "交叉编码器",
    "文档切块",
    "向量数据库",
    "向量检索",
    "余弦相似度",
    "语义检索",
    "重排序",
    "稠密检索",
)


class BM25Retriever:
    """使用双语词法分词和 BM25 评分的内存检索器。"""

    def __init__(
        self,
        chunks: Sequence[Chunk],
        tokenizer: Tokenizer | None = None,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        if not chunks:
            raise ValueError("at least one chunk is required")
        if not math.isfinite(k1) or k1 <= 0:
            raise ValueError("k1 must be positive")
        if not math.isfinite(b) or not 0 <= b <= 1:
            raise ValueError("b must be between 0 and 1")

        self._chunks = tuple(chunks)
        self._tokenizer = tokenizer or tokenize_bilingual
        self.k1 = float(k1)
        self.b = float(b)

        tokenized_chunks = tuple(tuple(self._tokenizer(chunk.text)) for chunk in self._chunks)
        self._document_lengths = tuple(len(tokens) for tokens in tokenized_chunks)
        total_tokens = sum(self._document_lengths)
        if total_tokens == 0:
            raise ValueError("chunks must contain at least one searchable token")
        self._average_document_length = total_tokens / len(self._chunks)

        document_frequencies: Counter[str] = Counter()
        postings: defaultdict[str, list[tuple[int, int]]] = defaultdict(list)
        for index, tokens in enumerate(tokenized_chunks):
            term_frequencies = Counter(tokens)
            for term, frequency in term_frequencies.items():
                document_frequencies[term] += 1
                postings[term].append((index, frequency))

        document_count = len(self._chunks)
        self._idf = {
            term: math.log(1 + (document_count - frequency + 0.5) / (frequency + 0.5))
            for term, frequency in document_frequencies.items()
        }
        self._postings = {term: tuple(entries) for term, entries in postings.items()}

    def retrieve(self, query: str, top_k: int = 5) -> tuple[RetrievedChunk, ...]:
        query = query.strip()
        if not query:
            raise ValueError("query must not be empty")
        if top_k < 1:
            raise ValueError("top_k must be positive")

        query_terms = tuple(dict.fromkeys(self._tokenizer(query)))
        if not query_terms:
            raise ValueError("query must contain at least one searchable token")

        scores: defaultdict[int, float] = defaultdict(float)
        for term in query_terms:
            idf = self._idf.get(term)
            if idf is None:
                continue
            for chunk_index, term_frequency in self._postings[term]:
                document_length = self._document_lengths[chunk_index]
                length_normalization = 1 - self.b + self.b * document_length / self._average_document_length
                denominator = term_frequency + self.k1 * length_normalization
                scores[chunk_index] += idf * term_frequency * (self.k1 + 1) / denominator

        scored_chunks = [
            (score, self._chunks[index])
            for index, score in scores.items()
            if score > 0
        ]
        scored_chunks.sort(key=lambda item: (-item[0], item[1].chunk_id))

        return tuple(
            RetrievedChunk(chunk=chunk, score=score, rank=rank)
            for rank, (score, chunk) in enumerate(scored_chunks[:top_k], start=1)
        )


def tokenize_bilingual(text: str) -> tuple[str, ...]:
    """保留英文技术词，并使用领域词典对连续中文做词级切分。"""

    normalized = unicodedata.normalize("NFKC", text).lower()
    tokens: list[str] = []
    for match in _TOKEN_PATTERN.finditer(normalized):
        value = match.group(0)
        if _is_han(value[0]):
            tokens.extend(_segment_chinese(value))
        else:
            tokens.append(value)
    return tuple(tokens)


def tokenize_char_ngrams(text: str, n_values: Sequence[int] = (2, 3)) -> tuple[str, ...]:
    """为中文生成字符 n-gram，同时保留完整的英文/技术词。"""

    sizes = tuple(dict.fromkeys(n_values))
    if not sizes or any(isinstance(size, bool) or not isinstance(size, int) or size < 1 for size in sizes):
        raise ValueError("n_values must contain positive integers")

    normalized = unicodedata.normalize("NFKC", text).lower()
    tokens: list[str] = []
    for match in _TOKEN_PATTERN.finditer(normalized):
        value = match.group(0)
        if not _is_han(value[0]):
            tokens.append(value)
            continue

        start_count = len(tokens)
        for size in sizes:
            tokens.extend(value[index : index + size] for index in range(len(value) - size + 1))
        if len(tokens) == start_count:
            tokens.append(value)
    return tuple(tokens)


def _segment_chinese(text: str) -> tuple[str, ...]:
    """生成基础词和领域短语，兼顾宽召回与专业词精确匹配。"""

    tokens = [token for token in _get_chinese_tokenizer().cut(text, HMM=False) if token.strip()]
    tokens.extend(term for term in _RAG_DOMAIN_TERMS if term in text and term not in tokens)
    return tuple(tokens)


@lru_cache(maxsize=1)
def _get_chinese_tokenizer() -> jieba.Tokenizer:
    """延迟创建独立 jieba 实例，避免修改进程级全局词典。"""

    return jieba.Tokenizer()


def _is_han(character: str) -> bool:
    return "\u3400" <= character <= "\u4dbf" or "\u4e00" <= character <= "\u9fff"
