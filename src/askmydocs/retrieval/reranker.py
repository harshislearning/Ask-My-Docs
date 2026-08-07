"""Cross-encoder reranking.

Retrieval and reranking answer different questions. The bi-encoder behind FAISS
embeds the query and each chunk *separately* - it never sees them together, so
it can only compare two independently-formed summaries. BM25 does not model
meaning at all. Both are cheap enough to run over the whole corpus, and both
are approximations.

A cross-encoder reads the query and the passage *jointly* in one forward pass
and scores how well this passage answers this query. That is far more accurate
and far too slow to run over a corpus - which is exactly why it goes last, over
a few dozen fused candidates rather than thousands of chunks.

The scoring model is kept behind a protocol, and the ordering logic lives in a
free function, so the part most likely to be subtly wrong (what gets dropped,
what order survives) is testable without loading a model.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from typing import Any, Protocol, runtime_checkable

from ..config import RetrievalConfig
from ..logging_setup import get_logger
from ..models import Candidate

log = get_logger(__name__)


@runtime_checkable
class Reranker(Protocol):
    """Scores how well each passage answers the query. Higher is better."""

    @property
    def model_name(self) -> str: ...

    def score(self, query: str, passages: Sequence[str]) -> list[float]: ...


class CrossEncoderReranker:
    """ms-marco cross-encoder via sentence-transformers, run locally."""

    def __init__(self, config: RetrievalConfig) -> None:
        self.config = config
        self._model: Any = None

    def _load(self) -> Any:
        if self._model is None:
            from sentence_transformers import CrossEncoder

            log.info("loading_reranker_model", model=self.config.reranker_model)
            self._model = CrossEncoder(
                self.config.reranker_model, max_length=self.config.rerank_max_length
            )
            log.info("reranker_model_loaded", model=self.config.reranker_model)
        return self._model

    @property
    def model_name(self) -> str:
        return self.config.reranker_model

    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        if not passages:
            return []
        model = self._load()
        pairs = [(query, passage) for passage in passages]
        scores = model.predict(
            pairs,
            batch_size=self.config.rerank_batch_size,
            show_progress_bar=False,
        )
        # Raw logits, not probabilities - only the ordering is meaningful, and
        # any threshold must be interpreted on that scale.
        return [float(score) for score in scores]


def rerank_candidates(
    query: str,
    candidates: Sequence[Candidate],
    reranker: Reranker | None,
    config: RetrievalConfig,
) -> list[Candidate]:
    """Reorder ``candidates`` by cross-encoder relevance and keep the top k.

    Degrades rather than fails: if reranking is disabled or the model errors,
    the fused order is kept and simply truncated. A retrieval stage that throws
    would take down the whole answer path over an optional refinement.
    """
    if not candidates:
        return []

    top_k = config.rerank_top_k

    if not config.rerank_enabled or reranker is None:
        log.debug("reranking_skipped", enabled=config.rerank_enabled, kept=min(top_k, len(candidates)))
        return list(candidates[:top_k])

    # The reranker sees the same text the retrievers indexed, breadcrumb
    # included: "Deployment > 2.2 Rollback" is genuine evidence about whether a
    # passage answers a question about rollbacks.
    passages = [candidate.chunk.embed_text for candidate in candidates]

    started = time.perf_counter()
    try:
        scores = reranker.score(query, passages)
    except Exception as exc:
        log.exception("reranking_failed", error=str(exc), fallback="fused order")
        return list(candidates[:top_k])

    if len(scores) != len(candidates):
        log.error(
            "reranker_returned_wrong_number_of_scores",
            expected=len(candidates),
            received=len(scores),
            fallback="fused order",
        )
        return list(candidates[:top_k])

    scored = [
        candidate.model_copy(update={"rerank_score": score})
        for candidate, score in zip(candidates, scores, strict=True)
    ]

    # Ties keep fusion order, so repeated runs produce identical output.
    scored.sort(key=lambda c: (-(c.rerank_score or 0.0), c.fused_rank))

    if config.min_rerank_score is not None:
        kept = [c for c in scored if (c.rerank_score or 0.0) >= config.min_rerank_score]
        if not kept:
            # Nothing cleared the bar. Returning the best of a bad pool would
            # invite a confident answer from irrelevant context; an empty set
            # routes to the "not enough information" path instead.
            log.info(
                "all_candidates_below_rerank_threshold",
                threshold=config.min_rerank_score,
                best_score=round(scored[0].rerank_score or 0.0, 4),
                candidates=len(scored),
            )
        scored = kept

    top = scored[:top_k]
    _renumber(top)

    log.info(
        "reranking_completed",
        model=reranker.model_name,
        candidates=len(candidates),
        kept=len(top),
        top_score=round(top[0].rerank_score or 0.0, 4) if top else None,
        moved_to_top=_promotion_distance(top),
        ms=round((time.perf_counter() - started) * 1000, 1),
    )
    return top


def _renumber(candidates: list[Candidate]) -> None:
    """Restate final positions 1..n while leaving fused_rank intact.

    Both numbers are worth keeping: the gap between them is exactly how much
    the reranker changed its mind about the retrievers' ordering.
    """
    for position, candidate in enumerate(candidates, start=1):
        candidate.final_rank = position


def _promotion_distance(candidates: Sequence[Candidate]) -> int | None:
    """How far the reranker promoted its new top result. 0 means it agreed."""
    return candidates[0].fused_rank - 1 if candidates else None
