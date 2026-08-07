"""Application state and FastAPI dependencies.

The embedding model, the reranker and the indexes take seconds to load, so they
are built once at startup and shared - loading them per request would dominate
latency entirely.

Startup is deliberately tolerant. A service that refuses to boot without an
index is unusable: you cannot call the ingest endpoint that would create one.
Instead the components that failed are recorded, ``/health`` reports
``degraded``, and ``/query`` returns 503 with a message saying what to run.
"""

from __future__ import annotations

import time
from typing import Annotated

from fastapi import Depends, Request

from ..config import AppConfig, load_config
from ..errors import AskMyDocsError
from ..generation import Answerer
from ..indexing.embedder import Embedder, SentenceTransformerEmbedder
from ..logging_setup import get_logger
from ..retrieval import CrossEncoderReranker, Reranker, RetrievalPipeline
from ..verification import Verifier
from .jobs import JobRegistry

log = get_logger(__name__)


class AppState:
    """Everything the request handlers need, built once."""

    def __init__(
        self,
        config: AppConfig,
        embedder: Embedder | None = None,
        reranker: Reranker | None = None,
        answerer: Answerer | None = None,
    ) -> None:
        self.config = config
        self.jobs = JobRegistry()

        # Injectable so tests can run the whole API without downloading models
        # or holding an API key.
        self._embedder = embedder
        self._reranker = reranker
        self._answerer = answerer

        self.pipeline: RetrievalPipeline | None = None
        self.index_error: str | None = None

    # -- construction ----------------------------------------------------

    @property
    def embedder(self) -> Embedder:
        if self._embedder is None:
            self._embedder = SentenceTransformerEmbedder(self.config.embedding)
        return self._embedder

    @property
    def reranker(self) -> Reranker | None:
        if self._reranker is None and self.config.retrieval.rerank_enabled:
            self._reranker = CrossEncoderReranker(self.config.retrieval)
        return self._reranker

    @property
    def answerer(self) -> Answerer:
        if self._answerer is None:
            self._answerer = Answerer.from_config(self.config)
        return self._answerer

    @property
    def verifier(self) -> Verifier:
        # The entailment judge reuses the generation client, so llm mode needs
        # no separate configuration.
        return Verifier(self.config.verification, client=self.answerer.client)

    @property
    def ready(self) -> bool:
        return self.pipeline is not None

    # -- lifecycle -------------------------------------------------------

    def load_indexes(self) -> bool:
        """(Re)load the indexes. Returns whether the service can serve queries."""
        try:
            self.pipeline = RetrievalPipeline.from_config(
                self.config, embedder=self.embedder, reranker=self.reranker
            )
            self.index_error = None
            log.info("indexes_loaded", chunks=len(self.pipeline.retriever.bundle.chunks))
            return True
        except AskMyDocsError as exc:
            # Expected before the first ingest; the message tells the operator
            # exactly which script to run.
            self.pipeline = None
            self.index_error = str(exc)
            log.warning("indexes_unavailable", error=str(exc))
            return False
        except Exception as exc:
            self.pipeline = None
            self.index_error = f"{type(exc).__name__}: {exc}"
            log.exception("index_load_failed")
            return False

    def warm(self) -> None:
        """Force the models to load now rather than during the first request.

        sentence-transformers loads weights on first use, not at construction,
        so without this the first user after a restart waits for bge *and* the
        cross-encoder to come off disk - measured at 13 seconds against 0.4 for
        every request after it.
        """
        if self.pipeline is None:
            return

        started = time.perf_counter()
        try:
            self.embedder.encode_query("warmup")
            if self.reranker is not None:
                self.reranker.score("warmup", ["warmup passage"])
        except Exception as exc:
            log.warning("warmup_failed", error=str(exc), impact="first query will be slow")
            return

        log.info("models_warmed", seconds=round(time.perf_counter() - started, 2))

    @classmethod
    def build(
        cls,
        config: AppConfig | None = None,
        embedder: Embedder | None = None,
        reranker: Reranker | None = None,
        answerer: Answerer | None = None,
    ) -> AppState:
        state = cls(
            config or load_config(),
            embedder=embedder,
            reranker=reranker,
            answerer=answerer,
        )
        state.load_indexes()
        return state


def get_state(request: Request) -> AppState:
    state: AppState = request.app.state.askmydocs
    return state


def get_config(state: Annotated[AppState, Depends(get_state)]) -> AppConfig:
    return state.config


StateDep = Annotated[AppState, Depends(get_state)]
ConfigDep = Annotated[AppConfig, Depends(get_config)]
