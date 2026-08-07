"""The evaluation harness.

One pass over the golden set, measuring three things that fail independently:

1. **Retrieval** - did the right chunks come back, and where in the ranking?
   Measured twice, once on the fused list and once after reranking, because the
   difference between those two numbers is the reranker's entire justification.
2. **Generation** - did the answer cite the expected evidence, and is every
   claim grounded?
3. **Refusal** - did it decline exactly when the documents could not answer?

Generation can be switched off so retrieval can be tuned without spending API
calls, and so CI has something meaningful to gate on without a key.
"""

from __future__ import annotations

import json
import time
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ..config import AppConfig
from ..errors import AskMyDocsError
from ..generation import Answerer
from ..logging_setup import bind_run, clear_run, get_logger, new_run_id
from ..models import Answer, Candidate
from ..retrieval import RetrievalPipeline
from ..retrieval.reranker import rerank_candidates
from ..verification import Verifier
from . import generation_metrics as gen
from . import retrieval_metrics as rank
from .golden import GoldenItem, GoldenSet, coverage_report, load_golden_set
from .ragas_runner import run_ragas

log = get_logger(__name__)


def _runs_dir(config: AppConfig) -> Path:
    directory = config.evaluation.runs_dir
    return directory if directory.is_absolute() else Path.cwd() / directory


def run_evaluation(
    config: AppConfig,
    *,
    golden_set: str | Path | None = None,
    limit: int | None = None,
    generate: bool = True,
    pipeline: RetrievalPipeline | None = None,
    answerer: Answerer | None = None,
    verifier: Verifier | None = None,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Evaluate the system against the golden set and return the metrics."""
    run_id = new_run_id()
    bind_run(run_id=run_id, stage="evaluation")

    try:
        golden = load_golden_set(golden_set or config.evaluation.golden_set)
        items = list(golden.items)[:limit] if limit else list(golden.items)
        if not items:
            raise AskMyDocsError("the golden set is empty")

        pipeline = pipeline or RetrievalPipeline.from_config(config)
        if generate:
            answerer = answerer or Answerer.from_config(config)
            verifier = verifier or Verifier(config.verification, client=answerer.client)

        coverage = coverage_report(golden, pipeline.retriever.bundle.chunks)
        log.info("evaluation_started", items=len(items), generate=generate, **golden.stats())

        started = time.perf_counter()
        records = [
            _evaluate_item(item, config, pipeline, answerer if generate else None, verifier)
            for item in items
        ]
        elapsed = time.perf_counter() - started

        report = _aggregate(records, golden, coverage, config, run_id, elapsed, generate)
        if generate:
            report["ragas"] = run_ragas(records, config)

        _write_run(report, records, output_dir or _runs_dir(config))

        log.info("evaluation_finished", **report["summary"])
        return report
    finally:
        clear_run()


# --------------------------------------------------------------------------
# One item
# --------------------------------------------------------------------------


def _evaluate_item(
    item: GoldenItem,
    config: AppConfig,
    pipeline: RetrievalPipeline,
    answerer: Answerer | None,
    verifier: Verifier | None,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "id": item.id,
        "question": item.question,
        "answerable": item.answerable,
        "tags": item.tags,
    }

    fused = pipeline.retriever.retrieve(item.question)
    reranked = rerank_candidates(item.question, fused, pipeline.reranker, config.retrieval)

    if item.answerable:
        total = len(item.expected_sources)
        k_values = config.evaluation.k_values
        record["retrieval"] = rank.score_ranking(_relevance(fused, item), total, k_values)
        record["reranked"] = rank.score_ranking(_relevance(reranked, item), total, k_values)

    record["retrieved"] = [
        {
            "rank": candidate.fused_rank,
            "source_file": candidate.chunk.source_file,
            "page": candidate.chunk.page_label,
            "relevant": item.is_relevant(candidate.chunk),
        }
        for candidate in reranked
    ]

    if answerer is None:
        return record

    try:
        answer = answerer.answer(item.question, reranked)
        if verifier is not None:
            answer = verifier.verify(answer)
    except AskMyDocsError as exc:
        # One failed question must not abandon the run; a scored zero is a
        # worse lie than an explicit error on the record.
        log.warning("evaluation_item_failed", id=item.id, error=str(exc))
        record["error"] = str(exc)
        return record

    record["answer"] = answer.text
    record["refused"] = answer.refused
    record["generation"] = gen.score_answer(answer, item)
    record["_answer"] = answer  # dropped before serialisation
    return record


def _relevance(candidates: Sequence[Candidate], item: GoldenItem) -> list[bool]:
    return item.relevance_vector([candidate.chunk for candidate in candidates])


# --------------------------------------------------------------------------
# Aggregation
# --------------------------------------------------------------------------


def _aggregate(
    records: list[dict[str, Any]],
    golden: GoldenSet,
    coverage: dict[str, Any],
    config: AppConfig,
    run_id: str,
    elapsed: float,
    generated: bool,
) -> dict[str, Any]:
    retrieval = rank.aggregate([r["retrieval"] for r in records if "retrieval" in r])
    reranked = rank.aggregate([r["reranked"] for r in records if "reranked" in r])
    generation = rank.aggregate([r["generation"] for r in records if "generation" in r])

    pairs: list[tuple[Answer, GoldenItem]] = [
        (r["_answer"], item)
        for r, item in zip(records, golden.items, strict=False)
        if "_answer" in r
    ]

    report: dict[str, Any] = {
        "run_id": run_id,
        "created_at": datetime.now(UTC).isoformat(),
        "golden_set": golden.path,
        "items_evaluated": len(records),
        "seconds": round(elapsed, 1),
        "generated": generated,
        "config": {
            "embedding_model": config.embedding.model,
            "reranker_model": config.retrieval.reranker_model,
            "generation_model": config.generation.model,
            "rerank_enabled": config.retrieval.rerank_enabled,
            "rerank_top_k": config.retrieval.rerank_top_k,
            "rrf_k": config.retrieval.rrf_k,
            "entailment_mode": config.verification.entailment_mode,
        },
        "golden_stats": golden.stats(),
        "coverage": {
            "expected_sources": coverage["expected_sources"],
            "unreachable": len(coverage["unreachable"]),
            "unreachable_examples": coverage["unreachable"][:5],
        },
        "retrieval": retrieval,
        "reranked": reranked,
        "generation": generation,
        "errors": [r["id"] for r in records if "error" in r],
    }

    if pairs:
        report["refusal"] = gen.refusal_breakdown(pairs)

    report["summary"] = _summary(report)
    return report


def _summary(report: dict[str, Any]) -> dict[str, Any]:
    """The handful of numbers worth putting in a CI comment."""
    reranked = report.get("reranked") or report.get("retrieval") or {}
    generation = report.get("generation") or {}
    refusal = report.get("refusal") or {}

    summary = {
        "items": report["items_evaluated"],
        "recall@5": reranked.get("recall@5", 0.0),
        "mrr": reranked.get("mrr", 0.0),
        "ndcg@5": reranked.get("ndcg@5", 0.0),
    }
    if generation:
        summary |= {
            "citation_precision": generation.get("citation_precision", 0.0),
            "citation_recall": generation.get("citation_recall", 0.0),
            "groundedness": generation.get("groundedness", 0.0),
            "refusal_correct": generation.get("refusal_correct", 0.0),
        }
    if refusal:
        summary["wrongly_answered"] = refusal["wrongly_answered"]
    return summary


def _write_run(report: dict[str, Any], records: list[dict[str, Any]], directory: Path) -> None:
    """Persist the run so a regression can be diffed against the item that caused it."""
    directory.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")

    serialisable = [{k: v for k, v in record.items() if not k.startswith("_")} for record in records]
    (directory / f"run-{stamp}-{report['run_id']}.json").write_text(
        json.dumps({"report": report, "records": serialisable}, indent=2, default=str),
        encoding="utf-8",
    )
    log.info("evaluation_run_written", directory=str(directory))
