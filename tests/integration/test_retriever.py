"""Hybrid retrieval end to end: build both indexes, query, fuse."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from askmydocs.config import AppConfig
from askmydocs.errors import IndexingError, IndexNotFoundError
from askmydocs.indexing import Bm25Store, FaissStore, IndexBuilder, load_index_bundle
from askmydocs.indexing.build import IndexBundle
from askmydocs.ingestion.pipeline import write_chunks
from askmydocs.models import Chunk
from askmydocs.retrieval import HybridRetriever
from fixtures.fake_embedder import FakeEmbedder


@pytest.fixture
def indexed(
    config: AppConfig, sample_chunks: list[Chunk], embedder: FakeEmbedder
) -> AppConfig:
    """A config whose chunks.jsonl and indexes are built and on disk."""
    write_chunks(config.paths.chunks_file, sample_chunks)
    IndexBuilder(config, embedder=embedder).build()
    return config


@pytest.fixture
def retriever(indexed: AppConfig, embedder: FakeEmbedder) -> HybridRetriever:
    return HybridRetriever.from_config(indexed, embedder=embedder)


# -- building --------------------------------------------------------------


def test_build_writes_both_indexes_and_a_manifest(
    indexed: AppConfig, sample_chunks: list[Chunk]
) -> None:
    directory = indexed.paths.indexes
    assert (directory / "faiss.index").is_file()
    assert (directory / "faiss_ids.json").is_file()
    assert (directory / "bm25.pkl").is_file()

    bundle = load_index_bundle(indexed)
    assert bundle.manifest is not None
    assert bundle.manifest.chunk_count == len(sample_chunks)
    assert bundle.manifest.document_count == 1
    assert bundle.manifest.embedding_model == "fake-embedder"


def test_both_indexes_cover_every_chunk(indexed: AppConfig) -> None:
    bundle = load_index_bundle(indexed)
    assert len(bundle.faiss) == len(bundle.chunks)
    assert len(bundle.bm25) == len(bundle.chunks)


def test_documents_are_embedded_without_the_query_prefix(
    config: AppConfig, sample_chunks: list[Chunk], embedder: FakeEmbedder
) -> None:
    # bge degrades measurably if the query instruction is prepended to documents.
    write_chunks(config.paths.chunks_file, sample_chunks)
    IndexBuilder(config, embedder=embedder).build()
    assert embedder.encoded_queries == []
    assert embedder.encoded_document_batches == [len(sample_chunks)]


def test_building_without_chunks_is_a_clear_error(
    config: AppConfig, embedder: FakeEmbedder
) -> None:
    with pytest.raises(IndexingError, match=re.escape("run scripts/ingest.py")):
        IndexBuilder(config, embedder=embedder).build()


def test_loading_before_building_raises(
    config: AppConfig, sample_chunks: list[Chunk]
) -> None:
    write_chunks(config.paths.chunks_file, sample_chunks)
    with pytest.raises(IndexNotFoundError):
        load_index_bundle(config)


# -- retrieval -------------------------------------------------------------


def test_exact_identifier_is_retrieved(retriever: HybridRetriever) -> None:
    candidates = retriever.retrieve("request_timeout")
    assert candidates[0].chunk.chunk_id == "chunk-0"


def test_candidates_carry_per_retriever_provenance(retriever: HybridRetriever) -> None:
    # Debugging retrieval quality means knowing which retriever found what.
    candidates = retriever.retrieve("request_timeout default value")
    top = candidates[0]

    assert set(top.ranks) <= {"vector", "bm25"}
    assert set(top.scores) == set(top.ranks)
    assert all(rank >= 1 for rank in top.ranks.values())


def test_a_chunk_found_by_both_retrievers_is_flagged(retriever: HybridRetriever) -> None:
    candidates = retriever.retrieve("retry budget exponential backoff")
    assert any(candidate.found_by_both for candidate in candidates)


def test_fused_ranks_are_sequential_and_scores_descend(
    retriever: HybridRetriever,
) -> None:
    candidates = retriever.retrieve("timeout")
    assert [c.fused_rank for c in candidates] == list(range(1, len(candidates) + 1))
    scores = [c.fused_score for c in candidates]
    assert scores == sorted(scores, reverse=True)


def test_dense_retrieval_still_answers_when_keywords_miss(
    retriever: HybridRetriever,
) -> None:
    # No shared vocabulary with any chunk: BM25 returns nothing, the vector
    # index still returns its nearest neighbours rather than failing.
    candidates = retriever.retrieve("zzz")
    assert candidates
    assert all("bm25" not in c.ranks for c in candidates)


def test_top_n_caps_the_fused_list(retriever: HybridRetriever) -> None:
    assert len(retriever.retrieve("timeout", top_n=2)) == 2


def test_empty_query_returns_nothing(retriever: HybridRetriever) -> None:
    assert retriever.retrieve("") == []
    assert retriever.retrieve("   ") == []


def test_query_is_embedded_with_the_configured_prefix(
    indexed: AppConfig, embedder: FakeEmbedder
) -> None:
    retriever = HybridRetriever.from_config(indexed, embedder=embedder)
    retriever.retrieve("how long is the timeout")
    assert embedder.encoded_queries == ["how long is the timeout"]


# -- ablation via weights --------------------------------------------------


def test_zero_bm25_weight_gives_vector_only_retrieval(
    indexed: AppConfig, embedder: FakeEmbedder
) -> None:
    indexed.retrieval.bm25_weight = 0.0
    retriever = HybridRetriever.from_config(indexed, embedder=embedder)

    candidates = retriever.retrieve("request_timeout")
    assert candidates
    assert all(set(c.ranks) == {"vector"} for c in candidates)


def test_zero_vector_weight_gives_keyword_only_retrieval(
    indexed: AppConfig, embedder: FakeEmbedder
) -> None:
    indexed.retrieval.vector_weight = 0.0
    retriever = HybridRetriever.from_config(indexed, embedder=embedder)

    candidates = retriever.retrieve("request_timeout")
    assert candidates
    assert all(set(c.ranks) == {"bm25"} for c in candidates)


# -- degraded states -------------------------------------------------------


def test_retrieval_on_an_empty_corpus_returns_nothing(
    config: AppConfig, embedder: FakeEmbedder
) -> None:
    bundle = IndexBundle(
        chunks=[],
        chunks_by_id={},
        faiss=FaissStore.build(embedder.encode_documents([]), []),
        bm25=Bm25Store.build([], config.retrieval),
    )
    retriever = HybridRetriever(bundle, embedder, config.retrieval)
    assert retriever.retrieve("anything") == []


def test_a_failing_embedder_degrades_to_keyword_only(
    indexed: AppConfig, embedder: FakeEmbedder
) -> None:
    # A model that fails to load mid-session must not take retrieval down.
    class BrokenEmbedder(FakeEmbedder):
        def encode_query(self, text: str):  # type: ignore[no-untyped-def]
            raise RuntimeError("model unavailable")

    retriever = HybridRetriever.from_config(indexed, embedder=BrokenEmbedder())
    candidates = retriever.retrieve("request_timeout")

    assert candidates
    assert all(set(c.ranks) == {"bm25"} for c in candidates)


def test_index_referencing_a_deleted_chunk_is_skipped(
    indexed: AppConfig, sample_chunks: list[Chunk], embedder: FakeEmbedder
) -> None:
    # Simulate chunks.jsonl being re-written without rebuilding the index.
    write_chunks(indexed.paths.chunks_file, sample_chunks[:-1])
    retriever = HybridRetriever.from_config(indexed, embedder=embedder)

    candidates = retriever.retrieve("availability zones replication")
    assert all(c.chunk.chunk_id != "chunk-4" for c in candidates)


def test_stale_index_is_detected(
    indexed: AppConfig, sample_chunks: list[Chunk], caplog: pytest.LogCaptureFixture
) -> None:
    write_chunks(indexed.paths.chunks_file, sample_chunks[:2])
    load_index_bundle(indexed)
    assert "index_out_of_date" in caplog.text


def test_index_survives_a_missing_manifest(
    indexed: AppConfig, embedder: FakeEmbedder
) -> None:
    (indexed.paths.indexes / "index_manifest.json").unlink()
    bundle = load_index_bundle(indexed)

    assert bundle.manifest is None
    retriever = HybridRetriever(bundle, embedder, indexed.retrieval)
    assert retriever.retrieve("timeout")


def test_chunks_and_index_stay_addressable_by_id(indexed: AppConfig) -> None:
    bundle = load_index_bundle(indexed)
    assert set(bundle.faiss.chunk_ids) == set(bundle.chunks_by_id)


def test_index_directory_is_reusable_after_rebuild(
    indexed: AppConfig, sample_chunks: list[Chunk], embedder: FakeEmbedder
) -> None:
    write_chunks(indexed.paths.chunks_file, sample_chunks[:3])
    manifest = IndexBuilder(indexed, embedder=embedder).build()

    assert manifest.chunk_count == 3
    bundle = load_index_bundle(indexed)
    assert len(bundle.faiss) == 3


def test_saved_index_files_land_in_the_configured_directory(
    indexed: AppConfig, tmp_path: Path
) -> None:
    assert tmp_path in indexed.paths.indexes.parents
