"""The composed path: retrieve both indexes, fuse, rerank."""

from __future__ import annotations

import pytest

from askmydocs.config import AppConfig
from askmydocs.indexing import IndexBuilder
from askmydocs.ingestion.pipeline import write_chunks
from askmydocs.models import Chunk
from askmydocs.retrieval import RetrievalPipeline
from fixtures.fake_embedder import FakeEmbedder
from fixtures.fake_reranker import BrokenReranker, FakeReranker


@pytest.fixture
def indexed(
    config: AppConfig, sample_chunks: list[Chunk], embedder: FakeEmbedder
) -> AppConfig:
    write_chunks(config.paths.chunks_file, sample_chunks)
    IndexBuilder(config, embedder=embedder).build()
    return config


@pytest.fixture
def pipeline(indexed: AppConfig, embedder: FakeEmbedder) -> RetrievalPipeline:
    return RetrievalPipeline.from_config(
        indexed, embedder=embedder, reranker=FakeReranker()
    )


def test_search_returns_at_most_rerank_top_k(
    pipeline: RetrievalPipeline, indexed: AppConfig
) -> None:
    results = pipeline.search("timeout")
    assert 0 < len(results) <= indexed.retrieval.rerank_top_k


def test_results_carry_the_whole_provenance_chain(pipeline: RetrievalPipeline) -> None:
    # One object explains the full journey: which retrievers found it, where
    # fusion put it, what the cross-encoder thought, where it ended up.
    top = pipeline.search("request_timeout")[0]

    assert top.ranks
    assert top.scores
    assert top.fused_rank >= 1
    assert top.rerank_score is not None
    assert top.final_rank == 1


def test_reranking_can_reorder_the_fused_list(
    indexed: AppConfig, embedder: FakeEmbedder
) -> None:
    fused_only = RetrievalPipeline.from_config(indexed, embedder=embedder, reranker=None)
    indexed.retrieval.rerank_enabled = False
    baseline = [c.chunk.chunk_id for c in fused_only.search("availability zones")]

    indexed.retrieval.rerank_enabled = True
    reranked = RetrievalPipeline.from_config(
        indexed, embedder=embedder, reranker=FakeReranker(scores=[0.0] * 10)
    )
    # With flat scores the reranker must not invent an ordering of its own.
    assert [c.chunk.chunk_id for c in reranked.search("availability zones")] == baseline


def test_disabled_reranking_returns_the_fused_top_k(
    indexed: AppConfig, embedder: FakeEmbedder
) -> None:
    indexed.retrieval.rerank_enabled = False
    pipeline = RetrievalPipeline.from_config(indexed, embedder=embedder, reranker=None)

    results = pipeline.search("timeout")
    assert all(c.rerank_score is None for c in results)
    assert [c.fused_rank for c in results] == list(range(1, len(results) + 1))


def test_a_broken_reranker_still_returns_answers(
    indexed: AppConfig, embedder: FakeEmbedder
) -> None:
    pipeline = RetrievalPipeline.from_config(
        indexed, embedder=embedder, reranker=BrokenReranker()
    )
    assert pipeline.search("request_timeout")


def test_empty_query_produces_no_context(pipeline: RetrievalPipeline) -> None:
    assert pipeline.search("") == []


def test_threshold_can_produce_an_empty_context(
    indexed: AppConfig, embedder: FakeEmbedder
) -> None:
    # This is the input the generator needs in order to refuse.
    indexed.retrieval.min_rerank_score = 999.0
    pipeline = RetrievalPipeline.from_config(
        indexed, embedder=embedder, reranker=FakeReranker()
    )
    assert pipeline.search("request_timeout") == []


def test_the_reranker_scores_the_whole_fused_pool(
    indexed: AppConfig, embedder: FakeEmbedder, sample_chunks: list[Chunk]
) -> None:
    reranker = FakeReranker()
    pipeline = RetrievalPipeline.from_config(
        indexed, embedder=embedder, reranker=reranker
    )
    pipeline.search("timeout")

    _, passages = reranker.calls[0]
    assert len(passages) > indexed.retrieval.rerank_top_k
