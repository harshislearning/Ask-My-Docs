"""The full retrieval path: hybrid retrieval, fusion, reranking.

This is the single entry point the API (Phase 6) and the eval harness (Phase 8)
call. Keeping it separate from :class:`HybridRetriever` means the fusion stage
stays independently testable and the reranker stays genuinely optional - which
matters, because measuring what the reranker contributes requires being able to
turn it off.
"""

from __future__ import annotations

import time

from ..config import AppConfig, RetrievalConfig
from ..indexing.embedder import Embedder
from ..logging_setup import get_logger
from ..models import Candidate
from .reranker import CrossEncoderReranker, Reranker, rerank_candidates
from .retriever import HybridRetriever

log = get_logger(__name__)


class RetrievalPipeline:
    """Query -> fused candidates -> reranked top-k."""

    def __init__(
        self,
        retriever: HybridRetriever,
        reranker: Reranker | None,
        config: RetrievalConfig,
    ) -> None:
        self.retriever = retriever
        self.reranker = reranker
        self.config = config

    @classmethod
    def from_config(
        cls,
        config: AppConfig,
        embedder: Embedder | None = None,
        reranker: Reranker | None = None,
    ) -> RetrievalPipeline:
        if reranker is None and config.retrieval.rerank_enabled:
            reranker = CrossEncoderReranker(config.retrieval)
        return cls(
            retriever=HybridRetriever.from_config(config, embedder=embedder),
            reranker=reranker,
            config=config.retrieval,
        )

    def search(self, query: str) -> list[Candidate]:
        """Return the chunks that should be shown to the model, best first."""
        started = time.perf_counter()

        candidates = self.retriever.retrieve(query)
        if not candidates:
            # A legitimate outcome, not an error: an empty context set is what
            # makes the generator refuse instead of inventing an answer.
            log.info("search_returned_no_candidates", query_chars=len(query))
            return []

        top = rerank_candidates(query, candidates, self.reranker, self.config)

        log.info(
            "search_completed",
            fused_candidates=len(candidates),
            returned=len(top),
            reranked=self.config.rerank_enabled and self.reranker is not None,
            ms=round((time.perf_counter() - started) * 1000, 1),
        )
        return top
