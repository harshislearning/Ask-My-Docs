"""Ranking metrics.

Every one of these has an off-by-one waiting in it, and a retrieval metric that
is quietly wrong is worse than no metric at all: it makes a regression look like
an improvement. Hence the hand-computed expected values.
"""

from __future__ import annotations

import math

import pytest

from askmydocs.evaluation.retrieval_metrics import (
    aggregate,
    dcg_at_k,
    hit_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    score_ranking,
)

T, F = True, False


# -- recall ----------------------------------------------------------------


def test_recall_counts_relevant_items_found_in_the_top_k() -> None:
    assert recall_at_k([T, F, T, F], total_relevant=4, k=4) == 0.5


def test_recall_is_bounded_by_k() -> None:
    assert recall_at_k([T, T, T], total_relevant=3, k=1) == pytest.approx(1 / 3)


def test_recall_divides_by_what_exists_not_what_was_retrieved() -> None:
    # Otherwise a system returning one of five correct chunks scores 1.0.
    assert recall_at_k([T], total_relevant=5, k=5) == 0.2


def test_recall_of_a_perfect_ranking_is_one() -> None:
    assert recall_at_k([T, T], total_relevant=2, k=5) == 1.0


def test_recall_with_nothing_relevant_is_zero() -> None:
    assert recall_at_k([F, F], total_relevant=0, k=5) == 0.0


def test_recall_never_exceeds_one() -> None:
    # A duplicated chunk must not push recall above 1.
    assert recall_at_k([T, T, T], total_relevant=2, k=3) == 1.0


# -- precision -------------------------------------------------------------


def test_precision_is_relevant_over_k() -> None:
    assert precision_at_k([T, F, T, F], k=4) == 0.5


def test_precision_divides_by_k_not_by_the_result_length() -> None:
    # A short result list is a failure to retrieve, not a smaller denominator.
    assert precision_at_k([T], k=5) == 0.2


def test_precision_at_zero_is_zero() -> None:
    assert precision_at_k([T], k=0) == 0.0


# -- MRR -------------------------------------------------------------------


def test_reciprocal_rank_of_the_first_hit() -> None:
    assert reciprocal_rank([F, F, T, T]) == pytest.approx(1 / 3)


def test_reciprocal_rank_is_one_when_the_top_hit_is_relevant() -> None:
    assert reciprocal_rank([T, F]) == 1.0


def test_reciprocal_rank_is_zero_with_no_hits() -> None:
    assert reciprocal_rank([F, F, F]) == 0.0


def test_reciprocal_rank_of_an_empty_ranking_is_zero() -> None:
    assert reciprocal_rank([]) == 0.0


def test_reciprocal_rank_ignores_hits_after_the_first() -> None:
    # MRR measures how far you have to read, not how much is down there.
    assert reciprocal_rank([F, T, T, T]) == reciprocal_rank([F, T, F, F])


# -- nDCG ------------------------------------------------------------------


def test_dcg_discounts_by_log_position() -> None:
    assert dcg_at_k([T, T], k=2) == pytest.approx(1.0 + 1 / math.log2(3))


def test_ndcg_of_the_ideal_ranking_is_one() -> None:
    assert ndcg_at_k([T, T, F], total_relevant=2, k=3) == pytest.approx(1.0)


def test_ndcg_punishes_relevant_items_that_sank() -> None:
    top = ndcg_at_k([T, F, F], total_relevant=1, k=3)
    bottom = ndcg_at_k([F, F, T], total_relevant=1, k=3)
    assert top == pytest.approx(1.0)
    assert bottom < top


def test_ndcg_is_not_punished_for_a_k_it_did_not_choose() -> None:
    # Five relevant chunks evaluated at k=3 can still be a perfect ranking:
    # the ideal DCG is capped at k too.
    assert ndcg_at_k([T, T, T], total_relevant=5, k=3) == pytest.approx(1.0)


def test_ndcg_with_nothing_relevant_is_zero() -> None:
    assert ndcg_at_k([F], total_relevant=0, k=3) == 0.0


def test_ndcg_of_a_ranking_with_no_hits_is_zero() -> None:
    assert ndcg_at_k([F, F, F], total_relevant=2, k=3) == 0.0


# -- hit -------------------------------------------------------------------


def test_hit_is_one_when_anything_relevant_is_in_range() -> None:
    assert hit_at_k([F, T], k=2) == 1.0
    assert hit_at_k([F, T], k=1) == 0.0


# -- combined --------------------------------------------------------------


def test_score_ranking_emits_every_metric_per_k() -> None:
    scores = score_ranking([T, F, T], total_relevant=2, k_values=[1, 3])
    assert set(scores) == {
        "mrr",
        "recall@1",
        "precision@1",
        "ndcg@1",
        "hit@1",
        "recall@3",
        "precision@3",
        "ndcg@3",
        "hit@3",
    }


def test_reranking_that_promotes_a_hit_improves_the_scores() -> None:
    # The comparison the harness reports: same items, better order.
    fused = score_ranking([F, F, T], total_relevant=1, k_values=[1, 3])
    reranked = score_ranking([T, F, F], total_relevant=1, k_values=[1, 3])

    assert reranked["mrr"] > fused["mrr"]
    assert reranked["recall@1"] > fused["recall@1"]
    assert reranked["recall@3"] == fused["recall@3"]  # same items, just reordered


# -- aggregation -----------------------------------------------------------


def test_aggregate_macro_averages_across_queries() -> None:
    # Every query counts equally: micro-averaging would let one question with
    # many expected sources dominate the whole set.
    assert aggregate([{"recall@5": 1.0}, {"recall@5": 0.0}]) == {"recall@5": 0.5}


def test_aggregate_handles_metrics_missing_from_some_queries() -> None:
    result = aggregate([{"a": 1.0, "b": 0.0}, {"a": 0.0}])
    assert result["a"] == 0.5
    assert result["b"] == 0.0


def test_aggregate_of_nothing_is_empty() -> None:
    assert aggregate([]) == {}
