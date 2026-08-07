"""Reranking: ordering, truncation, thresholds, and every failure path.

The reranker decides what the LLM sees. If it silently drops the one chunk that
holds the answer, or misaligns scores with candidates, the result is a fluent
answer built from the wrong evidence - so the fallbacks are tested as carefully
as the happy path.
"""

from __future__ import annotations

import sys
import types
from typing import Any

import pytest

from askmydocs.config import RetrievalConfig
from askmydocs.models import Candidate, Chunk
from askmydocs.retrieval.reranker import CrossEncoderReranker, rerank_candidates
from fixtures.fake_reranker import BrokenReranker, FakeReranker, MiscountingReranker


def _candidate(chunk_id: str, fused_rank: int, text: str = "some passage text") -> Candidate:
    chunk = Chunk(
        chunk_id=chunk_id,
        doc_id="doc-1",
        source_file="handbook.pdf",
        doc_title="Handbook",
        text=text,
        embed_text=f"Handbook > Section\n\n{text}",
        page_start=1,
        page_end=1,
        chunk_index=fused_rank,
        token_count=len(text.split()),
    )
    return Candidate(
        chunk=chunk,
        fused_score=1.0 / fused_rank,
        fused_rank=fused_rank,
        ranks={"vector": fused_rank},
        scores={"vector": 1.0 / fused_rank},
    )


@pytest.fixture
def config() -> RetrievalConfig:
    return RetrievalConfig(rerank_enabled=True, rerank_top_k=3)


@pytest.fixture
def candidates() -> list[Candidate]:
    return [
        _candidate("a", 1, "storage replication across zones"),
        _candidate("b", 2, "the rollback procedure for failed deployments"),
        _candidate("c", 3, "retry budget and backoff behaviour"),
        _candidate("d", 4, "unrelated filler about typography"),
    ]


def ids(result: list[Candidate]) -> list[str]:
    return [c.chunk.chunk_id for c in result]


# -- ordering --------------------------------------------------------------


def test_candidates_are_reordered_by_rerank_score(
    candidates: list[Candidate], config: RetrievalConfig
) -> None:
    # "b" is fused rank 2 but the best answer to this query - promoting it is
    # the entire reason this stage exists.
    result = rerank_candidates("rollback procedure", candidates, FakeReranker(), config)
    assert ids(result)[0] == "b"


def test_rerank_scores_are_attached(
    candidates: list[Candidate], config: RetrievalConfig
) -> None:
    result = rerank_candidates("retry budget", candidates, FakeReranker(), config)
    assert all(c.rerank_score is not None for c in result)


def test_final_rank_is_renumbered_but_fused_rank_survives(
    candidates: list[Candidate], config: RetrievalConfig
) -> None:
    # Keeping both is what makes "the reranker promoted this from #4 to #1"
    # visible in the logs and the UI.
    result = rerank_candidates("rollback procedure", candidates, FakeReranker(), config)

    assert [c.final_rank for c in result] == [1, 2, 3]
    assert result[0].fused_rank == 2


def test_ties_preserve_the_fused_order(config: RetrievalConfig) -> None:
    pool = [_candidate("a", 1), _candidate("b", 2), _candidate("c", 3)]
    result = rerank_candidates("query", pool, FakeReranker(scores=[5.0, 5.0, 5.0]), config)
    assert ids(result) == ["a", "b", "c"]


def test_result_is_stable_across_runs(
    candidates: list[Candidate], config: RetrievalConfig
) -> None:
    first = rerank_candidates("retry budget", candidates, FakeReranker(), config)
    second = rerank_candidates("retry budget", candidates, FakeReranker(), config)
    assert ids(first) == ids(second)


def test_input_candidates_are_not_mutated(
    candidates: list[Candidate], config: RetrievalConfig
) -> None:
    rerank_candidates("rollback", candidates, FakeReranker(), config)
    assert [c.chunk.chunk_id for c in candidates] == ["a", "b", "c", "d"]
    assert all(c.rerank_score is None for c in candidates)


# -- truncation ------------------------------------------------------------


def test_only_top_k_survive(candidates: list[Candidate], config: RetrievalConfig) -> None:
    assert len(rerank_candidates("query", candidates, FakeReranker(), config)) == 3


def test_top_k_larger_than_the_pool_is_harmless(
    candidates: list[Candidate], config: RetrievalConfig
) -> None:
    config.rerank_top_k = 99
    assert len(rerank_candidates("query", candidates, FakeReranker(), config)) == 4


def test_empty_candidate_list_returns_empty(config: RetrievalConfig) -> None:
    assert rerank_candidates("query", [], FakeReranker(), config) == []


# -- the score threshold ---------------------------------------------------


def test_candidates_below_the_threshold_are_dropped(
    candidates: list[Candidate], config: RetrievalConfig
) -> None:
    config.min_rerank_score = 2.0
    result = rerank_candidates(
        "query", candidates, FakeReranker(scores=[5.0, 3.0, 1.0, 0.0]), config
    )
    assert ids(result) == ["a", "b"]


def test_nothing_clearing_the_threshold_yields_an_empty_context(
    candidates: list[Candidate], config: RetrievalConfig
) -> None:
    # Returning the best of an irrelevant pool would invite a confident answer
    # from the wrong evidence; an empty set routes to the refusal path.
    config.min_rerank_score = 100.0
    assert rerank_candidates("query", candidates, FakeReranker(), config) == []


def test_no_threshold_keeps_the_top_k_regardless_of_score(
    candidates: list[Candidate], config: RetrievalConfig
) -> None:
    config.min_rerank_score = None
    result = rerank_candidates("query", candidates, FakeReranker(scores=[0.0] * 4), config)
    assert len(result) == 3


# -- disabling -------------------------------------------------------------


def test_disabled_reranking_keeps_the_fused_order(
    candidates: list[Candidate], config: RetrievalConfig
) -> None:
    config.rerank_enabled = False
    result = rerank_candidates("rollback procedure", candidates, FakeReranker(), config)
    assert ids(result) == ["a", "b", "c"]


def test_no_reranker_keeps_the_fused_order(
    candidates: list[Candidate], config: RetrievalConfig
) -> None:
    result = rerank_candidates("rollback procedure", candidates, None, config)
    assert ids(result) == ["a", "b", "c"]


def test_disabled_reranking_still_truncates(
    candidates: list[Candidate], config: RetrievalConfig
) -> None:
    config.rerank_enabled = False
    assert len(rerank_candidates("query", candidates, None, config)) == 3


# -- failure paths ---------------------------------------------------------


def test_a_failing_model_falls_back_to_the_fused_order(
    candidates: list[Candidate], config: RetrievalConfig
) -> None:
    # An optional refinement stage must never take down the answer path.
    result = rerank_candidates("rollback", candidates, BrokenReranker(), config)
    assert ids(result) == ["a", "b", "c"]


def test_wrong_number_of_scores_is_detected(
    candidates: list[Candidate], config: RetrievalConfig
) -> None:
    # Zipping mismatched lists would attach each score to the wrong candidate -
    # silent, and disastrous for answer quality.
    result = rerank_candidates("rollback", candidates, MiscountingReranker(), config)
    assert ids(result) == ["a", "b", "c"]
    assert all(c.rerank_score is None for c in result)


# -- what the model is shown -----------------------------------------------


def test_the_reranker_sees_the_query_and_the_breadcrumbed_text(
    candidates: list[Candidate], config: RetrievalConfig
) -> None:
    reranker = FakeReranker()
    rerank_candidates("how do rollbacks work", candidates, reranker, config)

    query, passages = reranker.calls[0]
    assert query == "how do rollbacks work"
    assert len(passages) == len(candidates)
    assert all(passage.startswith("Handbook > Section") for passage in passages)


def test_every_candidate_is_scored_not_just_the_top_k(
    candidates: list[Candidate], config: RetrievalConfig
) -> None:
    reranker = FakeReranker()
    rerank_candidates("query", candidates, reranker, config)
    assert len(reranker.calls[0][1]) == 4


# -- the sentence-transformers wrapper -------------------------------------


class _StubCrossEncoder:
    def __init__(self, name: str, max_length: int | None = None) -> None:
        self.name = name
        self.max_length = max_length
        self.calls: list[dict[str, Any]] = []

    def predict(self, pairs: list[tuple[str, str]], **kwargs: Any) -> list[float]:
        self.calls.append({"pairs": list(pairs), **kwargs})
        return [0.5] * len(pairs)


@pytest.fixture
def stub_cross_encoder(monkeypatch: pytest.MonkeyPatch) -> list[_StubCrossEncoder]:
    created: list[_StubCrossEncoder] = []

    def factory(name: str, max_length: int | None = None) -> _StubCrossEncoder:
        model = _StubCrossEncoder(name, max_length)
        created.append(model)
        return model

    module = types.ModuleType("sentence_transformers")
    module.CrossEncoder = factory  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)
    return created


def test_cross_encoder_is_not_loaded_until_first_use(
    config: RetrievalConfig, stub_cross_encoder: list[_StubCrossEncoder]
) -> None:
    CrossEncoderReranker(config)
    assert stub_cross_encoder == []


def test_cross_encoder_scores_query_passage_pairs(
    config: RetrievalConfig, stub_cross_encoder: list[_StubCrossEncoder]
) -> None:
    scores = CrossEncoderReranker(config).score("a query", ["first", "second"])

    assert scores == [0.5, 0.5]
    assert stub_cross_encoder[0].calls[0]["pairs"] == [
        ("a query", "first"),
        ("a query", "second"),
    ]


def test_max_length_is_passed_to_the_model(
    config: RetrievalConfig, stub_cross_encoder: list[_StubCrossEncoder]
) -> None:
    config.rerank_max_length = 384
    CrossEncoderReranker(config).score("q", ["p"])
    assert stub_cross_encoder[0].max_length == 384


def test_batch_size_is_passed_to_predict(
    config: RetrievalConfig, stub_cross_encoder: list[_StubCrossEncoder]
) -> None:
    config.rerank_batch_size = 7
    CrossEncoderReranker(config).score("q", ["p"])
    assert stub_cross_encoder[0].calls[0]["batch_size"] == 7


def test_scoring_no_passages_skips_the_model(
    config: RetrievalConfig, stub_cross_encoder: list[_StubCrossEncoder]
) -> None:
    assert CrossEncoderReranker(config).score("q", []) == []
    assert stub_cross_encoder == []


def test_model_is_loaded_once(
    config: RetrievalConfig, stub_cross_encoder: list[_StubCrossEncoder]
) -> None:
    reranker = CrossEncoderReranker(config)
    reranker.score("q", ["p"])
    reranker.score("q2", ["p2"])
    assert len(stub_cross_encoder) == 1
