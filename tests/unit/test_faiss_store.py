from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from askmydocs.errors import IndexingError, IndexNotFoundError
from askmydocs.indexing.embedder import l2_normalise
from askmydocs.indexing.faiss_store import FaissStore


@pytest.fixture
def store() -> FaissStore:
    # Three orthogonal-ish vectors so nearest-neighbour order is unambiguous.
    vectors = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.9, 0.1, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    return FaissStore.build(vectors, ["a", "b", "c"])


def test_search_returns_nearest_first(store: FaissStore) -> None:
    hits = store.search(np.array([1.0, 0.0, 0.0], dtype=np.float32), top_n=3)
    assert [hit.chunk_id for hit in hits] == ["a", "b", "c"]
    assert [hit.rank for hit in hits] == [1, 2, 3]


def test_scores_are_cosine_similarities(store: FaissStore) -> None:
    hits = store.search(np.array([1.0, 0.0, 0.0], dtype=np.float32), top_n=1)
    assert hits[0].score == pytest.approx(1.0, abs=1e-5)


def test_unnormalised_input_is_normalised_before_indexing() -> None:
    # IndexFlatIP computes a raw inner product, so an un-normalised vector
    # would rank by magnitude rather than by direction.
    scaled = np.array([[10.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    store = FaissStore.build(scaled, ["big", "small"])

    hits = store.search(np.array([0.0, 1.0], dtype=np.float32), top_n=2)
    assert hits[0].chunk_id == "small"
    assert hits[0].score == pytest.approx(1.0, abs=1e-5)


def test_unnormalised_query_is_normalised_too(store: FaissStore) -> None:
    hits = store.search(np.array([50.0, 0.0, 0.0], dtype=np.float32), top_n=1)
    assert hits[0].score == pytest.approx(1.0, abs=1e-5)


def test_top_n_beyond_the_corpus_returns_everything(store: FaissStore) -> None:
    assert len(store.search(np.array([1.0, 0.0, 0.0], dtype=np.float32), top_n=99)) == 3


def test_non_positive_top_n_returns_nothing(store: FaissStore) -> None:
    assert store.search(np.array([1.0, 0.0, 0.0], dtype=np.float32), top_n=0) == []


def test_empty_index_returns_nothing() -> None:
    store = FaissStore.build(np.zeros((0, 8), dtype=np.float32), [])
    assert len(store) == 0
    assert store.search(np.ones(8, dtype=np.float32), top_n=5) == []


def test_id_count_must_match_vector_count() -> None:
    with pytest.raises(IndexingError):
        FaissStore.build(np.zeros((2, 4), dtype=np.float32), ["only-one"])


def test_zero_vector_does_not_produce_nan() -> None:
    store = FaissStore.build(np.zeros((1, 4), dtype=np.float32), ["z"])
    hits = store.search(np.ones(4, dtype=np.float32), top_n=1)
    assert not np.isnan(hits[0].score)


def test_save_and_load_round_trip(store: FaissStore, tmp_path: Path) -> None:
    store.save(tmp_path)
    reloaded = FaissStore.load(tmp_path)

    query = np.array([0.5, 0.5, 0.0], dtype=np.float32)
    before = store.search(query, top_n=3)
    after = reloaded.search(query, top_n=3)

    assert reloaded.chunk_ids == store.chunk_ids
    assert reloaded.dimension == store.dimension
    assert [h.chunk_id for h in before] == [h.chunk_id for h in after]
    assert [h.score for h in before] == pytest.approx([h.score for h in after])


def test_loading_a_missing_index_raises(tmp_path: Path) -> None:
    with pytest.raises(IndexNotFoundError):
        FaissStore.load(tmp_path)


def test_normalisation_helper_leaves_unit_vectors_alone() -> None:
    unit = np.array([[0.0, 1.0]], dtype=np.float32)
    assert l2_normalise(unit) == pytest.approx(unit)


def test_normalisation_helper_handles_a_zero_row() -> None:
    result = l2_normalise(np.zeros((1, 3), dtype=np.float32))
    assert not np.isnan(result).any()
