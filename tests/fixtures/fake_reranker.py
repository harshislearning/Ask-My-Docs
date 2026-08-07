"""Deterministic stand-in for the ms-marco cross-encoder.

Scores by query-term overlap, which is enough to exercise every ordering,
filtering and fallback path without loading a model.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

_TOKEN = re.compile(r"[a-z0-9]+")


class FakeReranker:
    """Implements the :class:`askmydocs.retrieval.reranker.Reranker` protocol."""

    def __init__(self, scores: Sequence[float] | None = None) -> None:
        #: Fixed scores to return, in candidate order. When None, scores are
        #: computed from term overlap.
        self.fixed_scores = list(scores) if scores is not None else None
        self.calls: list[tuple[str, list[str]]] = []

    @property
    def model_name(self) -> str:
        return "fake-reranker"

    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        self.calls.append((query, list(passages)))
        if self.fixed_scores is not None:
            return list(self.fixed_scores)

        query_terms = set(_TOKEN.findall(query.lower()))
        return [
            float(len(query_terms & set(_TOKEN.findall(passage.lower()))))
            for passage in passages
        ]


class BrokenReranker:
    """A reranker whose model fails at query time."""

    @property
    def model_name(self) -> str:
        return "broken-reranker"

    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        raise RuntimeError("reranker model unavailable")


class MiscountingReranker:
    """Returns the wrong number of scores - the failure that would silently
    misalign scores with candidates if it were not detected."""

    @property
    def model_name(self) -> str:
        return "miscounting-reranker"

    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        return [1.0] * (len(passages) - 1)
