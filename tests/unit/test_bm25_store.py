from __future__ import annotations

from pathlib import Path

import pytest

from askmydocs.config import RetrievalConfig
from askmydocs.errors import IndexNotFoundError
from askmydocs.indexing.bm25_store import Bm25Store, tokenize
from askmydocs.models import Chunk


@pytest.fixture
def retrieval_config() -> RetrievalConfig:
    return RetrievalConfig()


@pytest.fixture
def store(sample_chunks: list[Chunk], retrieval_config: RetrievalConfig) -> Bm25Store:
    return Bm25Store.build(sample_chunks, retrieval_config)


# -- tokenisation ----------------------------------------------------------


def test_identifiers_are_indexed_whole_and_split() -> None:
    # Both forms are needed: the exact name must score highest, but a query for
    # "timeout" alone should still find the chunk.
    tokens = tokenize("request_timeout")
    assert tokens == ["request_timeout", "request", "timeout"]


def test_version_strings_survive_tokenisation() -> None:
    assert "v2.1.0" in tokenize("upgrade to v2.1.0 first")


def test_hyphenated_identifiers_are_handled() -> None:
    tokens = tokenize("set the x-request-id header")
    assert "x-request-id" in tokens
    assert "request" in tokens


def test_tokenisation_is_case_insensitive() -> None:
    assert tokenize("REQUEST Timeout") == tokenize("request timeout")


def test_punctuation_only_text_yields_no_tokens() -> None:
    assert tokenize("!!! ??? ...") == []


def test_empty_text_yields_no_tokens() -> None:
    assert tokenize("") == []


# -- search ----------------------------------------------------------------


def test_exact_identifier_query_ranks_its_chunk_first(store: Bm25Store) -> None:
    # This is the case dense retrieval gets wrong: `request_timeout` and
    # "connection deadline" are semantically close but only one is correct.
    hits = store.search("request_timeout", top_n=5)
    assert hits[0].chunk_id == "chunk-0"


def test_results_are_ranked_by_descending_score(store: Bm25Store) -> None:
    hits = store.search("retry budget error rate", top_n=5)
    scores = [hit.score for hit in hits]
    assert scores == sorted(scores, reverse=True)
    assert [hit.rank for hit in hits] == list(range(1, len(hits) + 1))


def test_chunks_sharing_no_query_term_are_excluded(store: Bm25Store) -> None:
    # BM25 scores them 0; returning them would let RRF reward pure noise.
    hits = store.search("replicated availability zones", top_n=5)
    assert hits
    assert all(hit.score > 0 for hit in hits)
    assert "chunk-4" in {hit.chunk_id for hit in hits}


def test_query_with_no_matching_terms_returns_nothing(store: Bm25Store) -> None:
    assert store.search("xylophone quokka zeppelin", top_n=5) == []


def test_query_with_no_tokens_returns_nothing(store: Bm25Store) -> None:
    assert store.search("!!!", top_n=5) == []


def test_top_n_limits_results(store: Bm25Store) -> None:
    assert len(store.search("the", top_n=2)) <= 2


def test_non_positive_top_n_returns_nothing(store: Bm25Store) -> None:
    assert store.search("timeout", top_n=0) == []


def test_breadcrumb_is_searchable(store: Bm25Store) -> None:
    # embed_text is indexed, so the section heading is part of the evidence.
    hits = store.search("Rollback", top_n=5)
    assert hits[0].chunk_id == "chunk-3"


# -- empty corpus ----------------------------------------------------------


def test_empty_corpus_builds_and_returns_nothing(retrieval_config: RetrievalConfig) -> None:
    store = Bm25Store.build([], retrieval_config)
    assert len(store) == 0
    assert store.search("anything", top_n=5) == []


# -- persistence -----------------------------------------------------------


def test_save_and_load_round_trip(
    store: Bm25Store,
    sample_chunks: list[Chunk],
    retrieval_config: RetrievalConfig,
    tmp_path: Path,
) -> None:
    store.save(tmp_path)
    reloaded = Bm25Store.load(tmp_path, sample_chunks, retrieval_config)

    before = store.search("request_timeout", top_n=5)
    after = reloaded.search("request_timeout", top_n=5)
    assert [h.chunk_id for h in before] == [h.chunk_id for h in after]
    assert [h.score for h in before] == pytest.approx([h.score for h in after])


def test_loading_a_missing_index_raises(
    sample_chunks: list[Chunk], retrieval_config: RetrievalConfig, tmp_path: Path
) -> None:
    with pytest.raises(IndexNotFoundError):
        Bm25Store.load(tmp_path, sample_chunks, retrieval_config)


def test_corrupt_index_falls_back_to_rebuilding(
    store: Bm25Store,
    sample_chunks: list[Chunk],
    retrieval_config: RetrievalConfig,
    tmp_path: Path,
) -> None:
    # A pickle written by a different library version must degrade into a
    # slower startup, not a crash at query time.
    store.save(tmp_path)
    (tmp_path / "bm25.pkl").write_bytes(b"not a pickle")

    reloaded = Bm25Store.load(tmp_path, sample_chunks, retrieval_config)
    assert len(reloaded) == len(sample_chunks)
    assert reloaded.search("request_timeout", top_n=1)[0].chunk_id == "chunk-0"


def test_stale_index_is_rebuilt_from_chunks(
    store: Bm25Store,
    sample_chunks: list[Chunk],
    retrieval_config: RetrievalConfig,
    tmp_path: Path,
) -> None:
    Bm25Store.build(sample_chunks[:2], retrieval_config).save(tmp_path)

    reloaded = Bm25Store.load(tmp_path, sample_chunks, retrieval_config)
    assert len(reloaded) == len(sample_chunks)
