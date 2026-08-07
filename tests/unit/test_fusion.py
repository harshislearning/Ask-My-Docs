"""Reciprocal Rank Fusion.

The fusion step decides what the LLM ever gets to see, and its bugs are silent:
a subtly wrong ordering still produces a fluent, well-cited, wrong answer. So
the arithmetic, the tie-breaking, and every degenerate input are pinned down
explicitly here.
"""

from __future__ import annotations

import pytest

from askmydocs.retrieval.fusion import DEFAULT_RRF_K, reciprocal_rank_fusion, rrf_score


def ids(hits) -> list[str]:
    return [hit.chunk_id for hit in hits]


# -- the formula -----------------------------------------------------------


def test_score_is_one_over_k_plus_rank() -> None:
    fused = reciprocal_rank_fusion({"vector": ["a", "b"]}, k=60)

    assert fused[0].score == pytest.approx(1 / 61)
    assert fused[1].score == pytest.approx(1 / 62)


def test_scores_from_both_retrievers_are_summed() -> None:
    fused = reciprocal_rank_fusion({"vector": ["a"], "bm25": ["a"]}, k=60)

    assert len(fused) == 1
    assert fused[0].score == pytest.approx(1 / 61 + 1 / 61)


def test_rank_is_one_based() -> None:
    fused = reciprocal_rank_fusion({"vector": ["a", "b", "c"]}, k=10)
    assert [hit.rank for hit in fused] == [1, 2, 3]


def test_ranks_from_each_retriever_are_recorded() -> None:
    fused = reciprocal_rank_fusion({"vector": ["a", "b"], "bm25": ["b", "a"]}, k=60)
    by_id = {hit.chunk_id: hit.ranks for hit in fused}

    assert by_id["a"] == {"vector": 1, "bm25": 2}
    assert by_id["b"] == {"vector": 2, "bm25": 1}


def test_fused_score_equals_sum_of_individual_contributions() -> None:
    fused = reciprocal_rank_fusion({"vector": ["x", "a"], "bm25": ["a"]}, k=17)
    a = next(hit for hit in fused if hit.chunk_id == "a")

    assert a.score == pytest.approx(rrf_score(2, k=17) + rrf_score(1, k=17))


# -- what fusion is actually for -------------------------------------------


def test_agreement_beats_a_single_strong_hit() -> None:
    # The whole point: a chunk both retrievers merely like outranks one that
    # only one retriever loves. Cross-retriever agreement is stronger evidence
    # than any single retriever's confidence.
    fused = reciprocal_rank_fusion(
        {
            "vector": ["only_vector", "x", "agreed"],
            "bm25": ["y", "agreed"],
        },
        k=DEFAULT_RRF_K,
    )
    assert ids(fused)[0] == "agreed"


#: "solo" is rank 1 for one retriever only; "agreed" is rank 5 for both.
#: solo scores 1/(k+1), agreed scores 2/(k+5) - so k decides which wins.
_K_SENSITIVE_LISTS = {
    "vector": ["solo", "a", "b", "c", "agreed"],
    "bm25": ["p", "q", "r", "s", "agreed"],
}


def test_small_k_lets_a_single_top_hit_win() -> None:
    # Small k sharpens the curve: being rank 1 anywhere outweighs agreement.
    assert ids(reciprocal_rank_fusion(_K_SENSITIVE_LISTS, k=2))[0] == "solo"


def test_large_k_favours_broad_agreement() -> None:
    # Large k flattens it: appearing in both lists wins even from rank 5.
    assert ids(reciprocal_rank_fusion(_K_SENSITIVE_LISTS, k=60))[0] == "agreed"


# -- weights ---------------------------------------------------------------


def test_weights_scale_a_retrievers_contribution() -> None:
    fused = reciprocal_rank_fusion(
        {"vector": ["a"], "bm25": ["b"]}, k=60, weights={"vector": 3.0}
    )
    assert ids(fused) == ["a", "b"]
    assert fused[0].score == pytest.approx(3 / 61)


def test_zero_weight_removes_influence_but_keeps_the_rank_record() -> None:
    # Useful for ablation runs: turn a retriever off without losing the
    # diagnostic information about where it would have ranked things.
    fused = reciprocal_rank_fusion(
        {"vector": ["a"], "bm25": ["b"]}, k=60, weights={"bm25": 0.0}
    )
    by_id = {hit.chunk_id: hit for hit in fused}

    assert by_id["b"].score == pytest.approx(0.0)
    assert by_id["b"].ranks == {"bm25": 1}
    assert ids(fused)[0] == "a"


def test_missing_weight_defaults_to_one() -> None:
    weighted = reciprocal_rank_fusion({"vector": ["a"]}, k=60, weights={"bm25": 5.0})
    plain = reciprocal_rank_fusion({"vector": ["a"]}, k=60)
    assert weighted[0].score == pytest.approx(plain[0].score)


# -- determinism -----------------------------------------------------------


def test_ties_break_by_first_appearance() -> None:
    # Identical scores must resolve the same way on every run, or the eval
    # harness compares noise against its baseline.
    fused = reciprocal_rank_fusion({"vector": ["a"], "bm25": ["b"]}, k=60)

    assert fused[0].score == pytest.approx(fused[1].score)
    assert ids(fused) == ["a", "b"]


def test_result_is_stable_across_repeated_calls() -> None:
    lists = {"vector": ["a", "b", "c"], "bm25": ["c", "a", "d"]}
    assert ids(reciprocal_rank_fusion(lists)) == ids(reciprocal_rank_fusion(lists))


def test_input_order_of_retrievers_decides_tie_order() -> None:
    forward = reciprocal_rank_fusion({"vector": ["a"], "bm25": ["b"]}, k=60)
    reverse = reciprocal_rank_fusion({"bm25": ["b"], "vector": ["a"]}, k=60)
    assert ids(forward) == ["a", "b"]
    assert ids(reverse) == ["b", "a"]


# -- degenerate input ------------------------------------------------------


def test_no_retrievers_returns_nothing() -> None:
    assert reciprocal_rank_fusion({}) == []


def test_all_lists_empty_returns_nothing() -> None:
    assert reciprocal_rank_fusion({"vector": [], "bm25": []}) == []


def test_one_empty_list_leaves_the_other_ordering_intact() -> None:
    fused = reciprocal_rank_fusion({"vector": ["a", "b", "c"], "bm25": []}, k=60)
    assert ids(fused) == ["a", "b", "c"]


def test_duplicate_ids_are_scored_once_at_their_best_rank() -> None:
    # A retriever repeating an id must not compound its own score.
    duplicated = reciprocal_rank_fusion({"vector": ["a", "a", "b"]}, k=60)
    clean = reciprocal_rank_fusion({"vector": ["a", "b"]}, k=60)

    assert ids(duplicated) == ["a", "b"]
    assert duplicated[0].score == pytest.approx(clean[0].score)
    assert duplicated[1].score == pytest.approx(clean[1].score)


def test_disjoint_lists_keep_everything() -> None:
    fused = reciprocal_rank_fusion({"vector": ["a", "b"], "bm25": ["c", "d"]}, k=60)
    assert set(ids(fused)) == {"a", "b", "c", "d"}


def test_top_n_truncates_after_scoring() -> None:
    lists = {"vector": ["a", "b", "c", "d"], "bm25": ["d", "c"]}
    full = reciprocal_rank_fusion(lists, k=60)
    limited = reciprocal_rank_fusion(lists, k=60, top_n=2)

    assert len(limited) == 2
    assert ids(limited) == ids(full)[:2]
    assert [hit.rank for hit in limited] == [1, 2]


def test_top_n_larger_than_the_candidate_pool_is_harmless() -> None:
    assert len(reciprocal_rank_fusion({"vector": ["a"]}, top_n=99)) == 1


def test_three_retrievers_fuse_the_same_way() -> None:
    # Nothing in the implementation is hardcoded to two retrievers, which is
    # what makes adding one (e.g. a title index) a config change.
    fused = reciprocal_rank_fusion(
        {"vector": ["a"], "bm25": ["a"], "titles": ["a"]}, k=60
    )
    assert fused[0].score == pytest.approx(3 / 61)


# -- validation ------------------------------------------------------------


@pytest.mark.parametrize("bad_k", [0, -1, -60])
def test_non_positive_k_is_rejected(bad_k: int) -> None:
    with pytest.raises(ValueError, match="positive"):
        reciprocal_rank_fusion({"vector": ["a"]}, k=bad_k)


def test_rrf_score_rejects_zero_based_ranks() -> None:
    with pytest.raises(ValueError, match="1-based"):
        rrf_score(0)
