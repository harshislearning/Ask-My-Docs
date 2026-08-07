"""Health and readiness."""

from __future__ import annotations

from fastapi import APIRouter

from ... import __version__
from ...indexing.build import read_index_manifest
from ..deps import StateDep
from ..schemas import ComponentHealth, HealthResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health(state: StateDep) -> HealthResponse:
    """Whether the service can answer questions, and why not if it cannot.

    Returns 200 either way: the process being up and the service being *useful*
    are different questions, and a load balancer should not restart a container
    whose only problem is that nobody has run ingestion yet.
    """
    manifest = read_index_manifest(state.config.paths.indexes)

    components = [
        ComponentHealth(
            name="index",
            ready=state.ready,
            detail=state.index_error,
        ),
        ComponentHealth(
            name="reranker",
            ready=not state.config.retrieval.rerank_enabled or state.reranker is not None,
            detail=None if state.config.retrieval.rerank_enabled else "disabled by config",
        ),
        ComponentHealth(
            name="llm",
            ready=bool(state.config.groq_api_key),
            detail=None if state.config.groq_api_key else "GROQ_API_KEY is not set",
        ),
    ]

    return HealthResponse(
        status="ok" if all(component.ready for component in components) else "degraded",
        version=__version__,
        index_built=manifest is not None,
        chunk_count=manifest.chunk_count if manifest else 0,
        document_count=manifest.document_count if manifest else 0,
        embedding_model=manifest.embedding_model if manifest else None,
        indexed_at=manifest.created_at if manifest else None,
        components=components,
    )
