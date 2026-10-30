"""不依赖模型的检索指标；空答案样本单独报告，不混入 Recall 分母。"""
from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class EvaluationCase:
    case_id: str
    query: str
    relevant_document_ids: tuple[str, ...]
    tags: tuple[str, ...] = ()
    human_reviewed: bool = False

    def __post_init__(self) -> None:
        if not self.case_id.strip() or not self.query.strip():
            raise ValueError("case id and query must not be empty")
        if len(set(self.relevant_document_ids)) != len(self.relevant_document_ids):
            raise ValueError("relevant document ids must be unique")


def recall_at_k(ranked_ids: Sequence[str], relevant_ids: Sequence[str], k: int) -> float:
    """doc-level Recall；多个相关文档时分母是相关文档总数。"""
    if k < 1:
        raise ValueError("k must be positive")
    relevant = set(relevant_ids)
    if not relevant:
        raise ValueError("unanswerable cases must be evaluated separately")
    ranked = tuple(dict.fromkeys(ranked_ids))
    return len(set(ranked[:k]) & relevant) / len(relevant)


def reciprocal_rank(ranked_ids: Sequence[str], relevant_ids: Sequence[str]) -> float:
    if not relevant_ids:
        raise ValueError("unanswerable cases must be evaluated separately")
    relevant = set(relevant_ids)
    for rank, document_id in enumerate(dict.fromkeys(ranked_ids), start=1):
        if document_id in relevant:
            return 1.0 / rank
    return 0.0


def percentile(values: Sequence[float], percent: float) -> float:
    """使用线性插值，单样本仍返回该样本。"""
    if not values or not 0 <= percent <= 100:
        raise ValueError("nonempty values and percent in [0,100] required")
    if not all(math.isfinite(value) and value >= 0 for value in values):
        raise ValueError("timings must be finite and nonnegative")
    ordered = sorted(values)
    position = (len(ordered) - 1) * percent / 100
    lower = math.floor(position)
    upper = math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def citation_id_precision(cited_ids: Sequence[str], allowed_ids: Sequence[str]) -> float | None:
    """只检查引用 ID 是否属于供给模型的证据，不衡量语义支持/答案正确性。"""
    unique = set(cited_ids)
    if not unique:
        return None
    return len(unique & set(allowed_ids)) / len(unique)


def summarize(cases: Sequence[EvaluationCase], rankings: Sequence[Sequence[str]], timings: Sequence[float]) -> dict:
    if not cases or len(cases) != len(rankings) or len(cases) != len(timings):
        raise ValueError("cases, rankings, timings must have equal nonzero length")
    answerable = [(case, ranking) for case, ranking in zip(cases, rankings, strict=True) if case.relevant_document_ids]
    unanswerable = [(case, ranking) for case, ranking in zip(cases, rankings, strict=True) if not case.relevant_document_ids]
    metrics: dict = {"cases": len(cases), "answerable_cases": len(answerable), "unanswerable_cases": len(unanswerable), "human_reviewed_cases": sum(case.human_reviewed for case in cases)}
    for k in (1, 3, 5):
        metrics[f"recall@{k}"] = sum(recall_at_k(ranking, case.relevant_document_ids, k) for case, ranking in answerable) / len(answerable) if answerable else None
        metrics[f"hit_rate@{k}"] = sum(bool(set(tuple(dict.fromkeys(ranking))[:k]) & set(case.relevant_document_ids)) for case, ranking in answerable) / len(answerable) if answerable else None
    metrics["mrr@5"] = sum(reciprocal_rank(tuple(dict.fromkeys(ranking))[:5], case.relevant_document_ids) for case, ranking in answerable) / len(answerable) if answerable else None
    metrics["unanswerable_empty_retrieval_rate"] = sum(not ranking for _, ranking in unanswerable) / len(unanswerable) if unanswerable else None
    metrics["unanswerable_note"] = "Empty retrieval is a diagnostic only; dense retrieval always returns nearest neighbours. Generation abstention needs separate evaluation."
    metrics["query_seconds_mean"] = sum(timings) / len(timings)
    metrics["query_seconds_p50"] = percentile(timings, 50)
    metrics["query_seconds_p95"] = percentile(timings, 95)
    groups = sorted({tag for case in cases for tag in case.tags})
    metrics["subsets"] = {
        tag: {
            "answerable_cases": sum(tag in case.tags for case, _ in answerable),
            "recall@3": sum(recall_at_k(ranking, case.relevant_document_ids, 3) for case, ranking in answerable if tag in case.tags) / sum(tag in case.tags for case, _ in answerable)
            if any(tag in case.tags for case, _ in answerable) else None,
        }
        for tag in groups
    }
    return metrics
