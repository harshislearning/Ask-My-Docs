"""The query endpoint: retrieve, generate, verify."""

from __future__ import annotations

import time

from fastapi import APIRouter, HTTPException, Request, status

from ...logging_setup import get_logger
from ...retrieval.reranker import rerank_candidates
from ..deps import StateDep
from ..schemas import QueryRequest, QueryResponse

log = get_logger(__name__)

router = APIRouter(tags=["query"])


@router.post("/query", response_model=QueryResponse)
def query(payload: QueryRequest, state: StateDep, request: Request) -> QueryResponse:
    """Answer a question from the indexed documents.

    A question the documents cannot answer is a 200 with ``refused: true``, not
    an error - refusing is correct behaviour, and a client should render it
    differently from a failure it might retry.
    """
    if state.pipeline is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                state.index_error
                or "no index loaded - run scripts/ingest.py then scripts/build_index.py"
            ),
        )

    started = time.perf_counter()
    config = state.config.retrieval

    # Per-request overrides are applied to a copy: mutating the shared config
    # would leak one caller's settings into every other request.
    request_config = config.model_copy(
        update={
            key: value
            for key, value in (
                ("rerank_top_k", payload.top_k),
                ("rerank_enabled", payload.rerank),
            )
            if value is not None
        }
    )

    candidates = state.pipeline.retriever.retrieve(payload.question)
    candidates = rerank_candidates(
        payload.question, candidates, state.pipeline.reranker, request_config
    )
    retrieval_ms = (time.perf_counter() - started) * 1000

    answer = state.answerer.answer(payload.question, candidates)
    answer = state.verifier.verify(answer)

    total_ms = (time.perf_counter() - started) * 1000
    log.info(
        "query_served",
        refused=answer.refused,
        sources=len(answer.sources),
        verified=answer.verification.passed if answer.verification else None,
        total_ms=round(total_ms, 1),
    )

    return QueryResponse.from_answer(
        answer,
        include_source_text=payload.include_source_text,
        retrieval_ms=retrieval_ms,
        total_ms=total_ms,
        request_id=getattr(request.state, "request_id", None),
    )
