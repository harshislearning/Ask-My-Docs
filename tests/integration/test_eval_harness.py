"""The evaluation harness end to end, against fakes.

Runs the real retrieval path over a real index; only the embedder, reranker and
LLM are substituted, so the numbers come from the actual pipeline rather than
from a mock of it.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from askmydocs.config import AppConfig
from askmydocs.evaluation import run_evaluation
from askmydocs.evaluation.golden import ExpectedSource, GoldenItem, write_golden_set
from askmydocs.generation import Answerer
from askmydocs.indexing import IndexBuilder
from askmydocs.ingestion.pipeline import write_chunks
from askmydocs.models import Chunk
from askmydocs.retrieval import RetrievalPipeline
from askmydocs.verification import Verifier
from fixtures.fake_embedder import FakeEmbedder
from fixtures.fake_llm import FakeLlmClient
from fixtures.fake_reranker import FakeReranker

#: sample_chunks has chunk-0 on page 1 (request_timeout) and chunk-3 on page 4
#: (rollback). See tests/conftest.py.
TIMEOUT_SOURCE = ExpectedSource(source_file="handbook.pdf", page=1)
ROLLBACK_SOURCE = ExpectedSource(source_file="handbook.pdf", page=4)


@pytest.fixture
def indexed(config: AppConfig, sample_chunks: list[Chunk]) -> AppConfig:
    write_chunks(config.paths.chunks_file, sample_chunks)
    IndexBuilder(config, embedder=FakeEmbedder()).build()
    return config


@pytest.fixture
def golden_path(tmp_path: Path) -> Path:
    path = tmp_path / "golden.jsonl"
    write_golden_set(
        path,
        [
            GoldenItem(
                id="q001",
                question="What is the request_timeout default?",
                expected_answer="30 seconds",
                expected_sources=[TIMEOUT_SOURCE],
                must_contain=["30"],
                tags=["parameter"],
            ),
            GoldenItem(
                id="q002",
                question="What triggers an automatic rollback?",
                expected_sources=[ROLLBACK_SOURCE],
                tags=["prose"],
            ),
            GoldenItem(
                id="q003",
                question="What is the parental leave policy?",
                answerable=False,
                tags=["unanswerable"],
            ),
        ],
    )
    return path


def _run(config: AppConfig, golden_path: Path, **kwargs: object):  # type: ignore[no-untyped-def]
    embedder = FakeEmbedder()
    pipeline = RetrievalPipeline.from_config(
        config, embedder=embedder, reranker=FakeReranker()
    )
    answerer = Answerer(
        FakeLlmClient(text="The default is 30 seconds [1]."), config.generation
    )
    return run_evaluation(
        config,
        golden_set=golden_path,
        pipeline=pipeline,
        answerer=answerer,
        verifier=Verifier(config.verification, client=answerer.client),
        output_dir=golden_path.parent / "runs",
        **kwargs,  # type: ignore[arg-type]
    )


# -- the report ------------------------------------------------------------


def test_the_report_covers_every_item(indexed: AppConfig, golden_path: Path) -> None:
    report = _run(indexed, golden_path)
    assert report["items_evaluated"] == 3
    assert report["golden_stats"] == {
        "total": 3,
        "answerable": 2,
        "unanswerable": 1,
        "with_must_contain": 1,
    }


def test_retrieval_metrics_are_reported_at_every_k(
    indexed: AppConfig, golden_path: Path
) -> None:
    report = _run(indexed, golden_path)
    for k in indexed.evaluation.k_values:
        assert f"recall@{k}" in report["retrieval"]
        assert f"ndcg@{k}" in report["retrieval"]
    assert "mrr" in report["retrieval"]


def test_fused_and_reranked_metrics_are_reported_separately(
    indexed: AppConfig, golden_path: Path
) -> None:
    # The difference between these two blocks is the reranker's justification.
    report = _run(indexed, golden_path)
    assert report["retrieval"]
    assert report["reranked"]
    assert set(report["retrieval"]) == set(report["reranked"])


def test_generation_metrics_are_reported(indexed: AppConfig, golden_path: Path) -> None:
    report = _run(indexed, golden_path)
    assert set(report["generation"]) >= {
        "citation_precision",
        "citation_recall",
        "context_recall",
        "groundedness",
        "refusal_correct",
    }


def test_the_refusal_breakdown_is_reported(indexed: AppConfig, golden_path: Path) -> None:
    report = _run(indexed, golden_path)
    breakdown = report["refusal"]
    assert breakdown["correctly_answered"] + breakdown["wrongly_refused"] == 2
    assert breakdown["correctly_refused"] + breakdown["wrongly_answered"] == 1


def test_the_summary_holds_the_headline_numbers(
    indexed: AppConfig, golden_path: Path
) -> None:
    summary = _run(indexed, golden_path)["summary"]
    assert set(summary) >= {"items", "recall@5", "mrr", "citation_precision", "groundedness"}


def test_the_config_is_recorded_with_the_run(
    indexed: AppConfig, golden_path: Path
) -> None:
    # A metric is meaningless without the settings that produced it.
    config_block = _run(indexed, golden_path)["config"]
    assert config_block["embedding_model"] == indexed.embedding.model
    assert config_block["rerank_top_k"] == indexed.retrieval.rerank_top_k


# -- retrieval-only mode ---------------------------------------------------


def test_generation_can_be_skipped(indexed: AppConfig, golden_path: Path) -> None:
    # What CI runs: retrieval metrics, no API key, no cost, deterministic.
    report = _run(indexed, golden_path, generate=False)

    assert report["generated"] is False
    assert report["retrieval"]
    assert report["generation"] == {}
    assert "refusal" not in report


def test_retrieval_only_mode_needs_no_answerer(
    indexed: AppConfig, golden_path: Path
) -> None:
    report = run_evaluation(
        indexed,
        golden_set=golden_path,
        generate=False,
        pipeline=RetrievalPipeline.from_config(
            indexed, embedder=FakeEmbedder(), reranker=FakeReranker()
        ),
        output_dir=golden_path.parent / "runs",
    )
    assert report["items_evaluated"] == 3


# -- limits and errors -----------------------------------------------------


def test_limit_truncates_the_run(indexed: AppConfig, golden_path: Path) -> None:
    assert _run(indexed, golden_path, limit=1)["items_evaluated"] == 1


def test_an_empty_golden_set_is_an_error(indexed: AppConfig, tmp_path: Path) -> None:
    path = tmp_path / "empty.jsonl"
    path.write_text("", encoding="utf-8")

    with pytest.raises(Exception, match="empty"):
        _run(indexed, path)


def test_unreachable_expected_sources_are_flagged(
    indexed: AppConfig, tmp_path: Path
) -> None:
    # A label pointing at a page that was never ingested scores as a permanent
    # retrieval miss and looks exactly like a retrieval bug.
    path = tmp_path / "golden.jsonl"
    write_golden_set(
        path,
        [
            GoldenItem(
                id="q001",
                question="Anything?",
                expected_sources=[ExpectedSource(source_file="missing.pdf", page=1)],
            )
        ],
    )

    report = _run(indexed, path)
    assert report["coverage"]["unreachable"] == 1
    assert "missing.pdf" in report["coverage"]["unreachable_examples"][0]


# -- artifacts -------------------------------------------------------------


def test_the_run_is_written_to_disk(indexed: AppConfig, golden_path: Path) -> None:
    _run(indexed, golden_path)
    runs = list((golden_path.parent / "runs").glob("run-*.json"))

    assert len(runs) == 1
    payload = json.loads(runs[0].read_text(encoding="utf-8"))
    assert payload["report"]["items_evaluated"] == 3
    assert len(payload["records"]) == 3


def test_per_item_records_explain_a_regression(
    indexed: AppConfig, golden_path: Path
) -> None:
    # An aggregate that moved is useless without the item that moved it.
    _run(indexed, golden_path)
    payload = json.loads(
        next((golden_path.parent / "runs").glob("run-*.json")).read_text(encoding="utf-8")
    )
    record = payload["records"][0]

    assert record["id"] == "q001"
    assert record["question"]
    assert record["retrieved"]
    assert "relevant" in record["retrieved"][0]


def test_answer_objects_are_not_serialised(
    indexed: AppConfig, golden_path: Path
) -> None:
    _run(indexed, golden_path)
    payload = json.loads(
        next((golden_path.parent / "runs").glob("run-*.json")).read_text(encoding="utf-8")
    )
    assert all(not key.startswith("_") for record in payload["records"] for key in record)


# -- ragas -----------------------------------------------------------------


def test_ragas_is_off_by_default_and_says_so(
    indexed: AppConfig, golden_path: Path
) -> None:
    report = _run(indexed, golden_path)
    assert report["ragas"]["enabled"] is False


def test_a_ragas_failure_does_not_fail_the_run(
    indexed: AppConfig, golden_path: Path
) -> None:
    # RAGAS is optional and version-fragile; it must never take the harness down.
    indexed.evaluation.use_ragas = True
    report = _run(indexed, golden_path)

    assert report["items_evaluated"] == 3
    assert "reason" in report["ragas"] or "scores" in report["ragas"]
