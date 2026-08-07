"""FastAPI application.

    uvicorn askmydocs.api.main:app --reload

Two things here are worth more than the routing. First, every provider failure
is translated into an HTTP status that says what the caller should do about it -
a rate limit is a 429 they can retry, a missing index is a 503 with the command
that fixes it, and a bad API key is a 500 with no details leaked. Second, every
log line emitted while handling a request carries that request's id, so a single
slow or wrong answer can be traced across retrieval, generation and verification.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse

from .. import __version__
from ..config import AppConfig
from ..errors import (
    AskMyDocsError,
    ConfigError,
    DocumentError,
    GenerationError,
    IndexingError,
    LlmAuthError,
    LlmRateLimitError,
    LlmTimeoutError,
)
from ..logging_setup import bind_run, clear_run, configure_logging, get_logger
from .deps import AppState
from .routes import evaluation, health, ingest, jobs, query
from .schemas import ErrorResponse

log = get_logger(__name__)

#: Which HTTP status each failure maps to, most specific first.
ERROR_STATUS: list[tuple[type[Exception], int, str]] = [
    (LlmRateLimitError, status.HTTP_429_TOO_MANY_REQUESTS, "rate_limited"),
    (LlmTimeoutError, status.HTTP_504_GATEWAY_TIMEOUT, "upstream_timeout"),
    # The key is a server-side configuration problem. Reporting it as 401 would
    # tell the caller to fix credentials they do not have.
    (LlmAuthError, status.HTTP_500_INTERNAL_SERVER_ERROR, "llm_auth_failed"),
    (GenerationError, status.HTTP_502_BAD_GATEWAY, "generation_failed"),
    (IndexingError, status.HTTP_503_SERVICE_UNAVAILABLE, "index_unavailable"),
    (DocumentError, status.HTTP_400_BAD_REQUEST, "document_error"),
    (ConfigError, status.HTTP_500_INTERNAL_SERVER_ERROR, "misconfigured"),
    (AskMyDocsError, status.HTTP_500_INTERNAL_SERVER_ERROR, "internal_error"),
]


def create_app(config: AppConfig | None = None, state: AppState | None = None) -> FastAPI:
    """Build the application.

    ``state`` is injectable so tests can run the full API against fake models
    and a fake LLM, with no downloads and no API key.
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app_state = state or AppState.build(config)
        configure_logging(app_state.config.logging)
        app.state.askmydocs = app_state

        if not app_state.ready:
            # Not fatal: refusing to start without an index would make the
            # ingest endpoint that creates one unreachable.
            log.warning("starting_without_index", detail=app_state.index_error)
        else:
            app_state.warm()

        log.info("api_started", version=__version__, ready=app_state.ready)

        yield
        log.info("api_stopping")

    app = FastAPI(
        title="Ask My Docs",
        version=__version__,
        summary="Hybrid-retrieval RAG over a PDF knowledge base, with cited answers.",
        lifespan=lifespan,
    )

    app.middleware("http")(_request_context)
    _install_error_handlers(app)

    app.include_router(health.router)
    app.include_router(query.router)
    app.include_router(ingest.router)
    app.include_router(jobs.router)
    app.include_router(evaluation.router)
    return app


async def _request_context(
    request: Request, call_next: Callable[[Request], Awaitable[JSONResponse]]
) -> JSONResponse:
    """Tag the request and bind its id to every log line it produces."""
    request_id = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]
    request.state.request_id = request_id
    bind_run(request_id=request_id, path=request.url.path)

    started = time.perf_counter()
    try:
        response = await call_next(request)
    finally:
        clear_run()

    duration_ms = (time.perf_counter() - started) * 1000
    response.headers["x-request-id"] = request_id
    log.info(
        "http_request",
        method=request.method,
        path=request.url.path,
        status=response.status_code,
        ms=round(duration_ms, 1),
        request_id=request_id,
    )
    return response


def _install_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(AskMyDocsError)
    async def handle_known(request: Request, exc: AskMyDocsError) -> JSONResponse:
        http_status, code = _classify(exc)
        request_id = getattr(request.state, "request_id", None)

        # An auth failure means our key, not theirs. Log the detail, return none.
        detail = (
            "the language model rejected this deployment's credentials"
            if isinstance(exc, LlmAuthError)
            else str(exc)
        )
        log.warning(
            "request_failed",
            error=type(exc).__name__,
            code=code,
            status=http_status,
            detail=str(exc),
            request_id=request_id,
        )
        return JSONResponse(
            status_code=http_status,
            content=ErrorResponse(
                error=code, detail=detail, request_id=request_id
            ).model_dump(),
        )

    @app.exception_handler(NotImplementedError)
    async def handle_not_implemented(
        request: Request, exc: NotImplementedError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            content=ErrorResponse(
                error="not_implemented",
                detail=str(exc),
                request_id=getattr(request.state, "request_id", None),
            ).model_dump(),
        )


def _classify(exc: Exception) -> tuple[int, str]:
    for error_type, http_status, code in ERROR_STATUS:
        if isinstance(exc, error_type):
            return http_status, code
    return status.HTTP_500_INTERNAL_SERVER_ERROR, "internal_error"


app = create_app()
