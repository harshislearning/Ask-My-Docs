"""Ranking metrics.

Pure functions over a list of booleans - "was the item at this rank relevant?" -
so they can be tested exhaustively without an index, an embedding model, or a
golden set. Every one of these has an off-by-one waiting in it, and a retrieval
metric that is quietly wrong is worse than no metric: it makes a regression look
like an improvement.

All three answer different questions:

* **Recall@k** - did we find the evidence at all? The ceiling on everything
  downstream: what retrieval misses, the model cannot cite.
* **MRR** - how far down was the first good hit? Sensitive to the top of the
  list, which is what a reranker moves.
* **nDCG@k** - how good is the whole ordering, discounted by position? Catches
  the case where recall is unchanged but the useful chunks sank.
"""

from __future__ import annotations

import math
from collections.abc import Sequence


def recall_at_k(relevance: Sequence[bool], total_relevant: int, k: int) -> float:
    """Share of all relevant items that appear in the top k.

    ``total_relevant`` is the number that *exist*, not the number retrieved -
    otherwise a system that returns one correct chunk out of five scores 1.0.
    """
    if total_relevant <= 0:
        return 0.0
    found = sum(1 for is_relevant in relevance[:k] if is_relevant)
    return min(found / total_relevant, 1.0)


def precision_at_k(relevance: Sequence[bool], k: int) -> float:
    """Share of the top k that is relevant."""
    if k <= 0:
        return 0.0
    window = relevance[:k]
    if not window:
        return 0.0
    # Divided by k, not len(window): a short result list is a failure to
    # retrieve, not an excuse for a smaller denominator.
    return sum(1 for is_relevant in window if is_relevant) / k


def reciprocal_rank(relevance: Sequence[bool]) -> float:
    """1 / rank of the first relevant item, 0 if there is none."""
    for index, is_relevant in enumerate(relevance, start=1):
        if is_relevant:
            return 1.0 / index
    return 0.0


def dcg_at_k(relevance: Sequence[bool], k: int) -> float:
    """Discounted cumulative gain with binary relevance."""
    return sum(
        1.0 / math.log2(index + 2)
        for index, is_relevant in enumerate(relevance[:k])
        if is_relevant
    )


def ndcg_at_k(relevance: Sequence[bool], total_relevant: int, k: int) -> float:
    """DCG normalised by the best achievable ordering.

    The ideal ranking puts every relevant item first, capped at k - so a query
    with five relevant chunks evaluated at k=3 can still score 1.0 by returning
    three of them, rather than being punished for a limit it did not set.
    """
    if total_relevant <= 0 or k <= 0:
        return 0.0
    ideal = dcg_at_k([True] * min(total_relevant, k), k)
    if ideal == 0.0:
        return 0.0
    return dcg_at_k(relevance, k) / ideal


def hit_at_k(relevance: Sequence[bool], k: int) -> float:
    """1.0 if anything relevant made the top k."""
    return 1.0 if any(relevance[:k]) else 0.0


def score_ranking(
    relevance: Sequence[bool], total_relevant: int, k_values: Sequence[int]
) -> dict[str, float]:
    """Every metric for one query, keyed for aggregation."""
    scores: dict[str, float] = {"mrr": reciprocal_rank(relevance)}
    for k in k_values:
        scores[f"recall@{k}"] = recall_at_k(relevance, total_relevant, k)
        scores[f"precision@{k}"] = precision_at_k(relevance, k)
        scores[f"ndcg@{k}"] = ndcg_at_k(relevance, total_relevant, k)
        scores[f"hit@{k}"] = hit_at_k(relevance, k)
    return scores


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def aggregate(per_query: Sequence[dict[str, float]]) -> dict[str, float]:
    """Macro-average across queries: every query counts equally.

    Micro-averaging would let one question with many expected sources dominate
    the score for the whole set.
    """
    if not per_query:
        return {}
    keys = sorted({key for scores in per_query for key in scores})
    return {
        key: round(mean([scores[key] for scores in per_query if key in scores]), 4)
        for key in keys
    }
