"""Reciprocal Rank Fusion.

Dense similarity and BM25 produce scores on incomparable scales - cosine sits
in [-1, 1] while BM25 is an unbounded sum of IDF terms that shifts with corpus
statistics. Normalising them into a weighted average means picking a
normalisation that silently changes with the corpus.

RRF sidesteps that by discarding the scores and fusing *ranks*::

    score(d) = sum over retrievers r of  weight_r / (k + rank_r(d))

Only ordering matters, so no calibration is needed. The constant ``k`` damps
the influence of the very top ranks: small k makes rank 1 dominate, large k
flattens the curve so agreement across retrievers matters more than any single
retriever's confidence. k=60 comes from the original paper (Cormack et al.,
2009) and is a reasonable default.

This module is deliberately dependency-free and works on plain id strings, so
the fusion logic can be tested exhaustively without building an index.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from ..logging_setup import get_logger
from ..models import FusedHit

log = get_logger(__name__)

DEFAULT_RRF_K = 60


def reciprocal_rank_fusion(
    ranked_lists: Mapping[str, Sequence[str]],
    *,
    k: int = DEFAULT_RRF_K,
    weights: Mapping[str, float] | None = None,
    top_n: int | None = None,
) -> list[FusedHit]:
    """Fuse ranked id lists into one ordering.

    Args:
        ranked_lists: retriever name -> ids, best first.
        k: RRF damping constant. Must be positive.
        weights: per-retriever multiplier. Missing entries default to 1.0;
            a weight of 0.0 excludes that retriever from the fused score while
            still recording its ranks for debugging.
        top_n: truncate the result. ``None`` returns everything.

    Returns:
        Fused hits, best first. Ties are broken by first appearance across the
        input lists (in the order given), so the output is deterministic - which
        matters because the eval harness compares runs against a baseline.
    """
    if k <= 0:
        raise ValueError(f"rrf k must be positive, got {k}")

    weights = weights or {}
    scores: dict[str, float] = {}
    ranks: dict[str, dict[str, int]] = {}
    first_seen: dict[str, int] = {}
    order = 0

    for retriever, ids in ranked_lists.items():
        weight = float(weights.get(retriever, 1.0))
        seen_in_this_list: set[str] = set()
        rank = 0

        for chunk_id in ids:
            # A retriever should not return the same id twice, but if it does,
            # the first (best) rank is the honest one to use.
            if chunk_id in seen_in_this_list:
                continue
            seen_in_this_list.add(chunk_id)
            rank += 1

            scores[chunk_id] = scores.get(chunk_id, 0.0) + weight / (k + rank)
            ranks.setdefault(chunk_id, {})[retriever] = rank
            if chunk_id not in first_seen:
                first_seen[chunk_id] = order
                order += 1

    ordered = sorted(scores, key=lambda cid: (-scores[cid], first_seen[cid]))
    if top_n is not None:
        ordered = ordered[:top_n]

    fused = [
        FusedHit(chunk_id=cid, score=scores[cid], rank=position, ranks=ranks[cid])
        for position, cid in enumerate(ordered, start=1)
    ]

    log.debug(
        "rrf_fused",
        retrievers=list(ranked_lists),
        input_sizes={name: len(ids) for name, ids in ranked_lists.items()},
        unique_candidates=len(scores),
        returned=len(fused),
        k=k,
    )
    return fused


def rrf_score(rank: int, k: int = DEFAULT_RRF_K, weight: float = 1.0) -> float:
    """Contribution of a single rank. Exposed for tests and for explaining
    a fused score in the UI."""
    if rank < 1:
        raise ValueError(f"rank is 1-based, got {rank}")
    return weight / (k + rank)
