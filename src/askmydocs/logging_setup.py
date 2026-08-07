"""Structured logging.

Every log line is a dict of fields, not an interpolated string. That is what
makes retrieval debugging tractable later: you can filter on ``doc_id``,
``chunk_id`` or ``query_id`` instead of grepping prose.

Third-party stdlib loggers (PyMuPDF, httpx, uvicorn) are routed through the
same formatter so one run produces one consistent stream.
"""

from __future__ import annotations

import logging
import sys
import uuid
from typing import Any, TextIO

import structlog

from .config import LoggingConfig

_CONFIGURED = False


def configure_logging(config: LoggingConfig | None = None, stream: TextIO | None = None) -> None:
    """Install the logging configuration. Safe to call more than once."""
    global _CONFIGURED
    config = config or LoggingConfig()
    level = getattr(logging, config.level)

    shared_processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
    ]

    structlog.configure(
        processors=[
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    if config.format == "json":
        renderer: Any = structlog.processors.JSONRenderer()
        exception_processor: Any = structlog.processors.dict_tracebacks
    else:
        renderer = structlog.dev.ConsoleRenderer(colors=stream is None and sys.stderr.isatty())
        exception_processor = structlog.processors.format_exc_info

    formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=shared_processors,
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            exception_processor,
            renderer,
        ],
    )

    handler = logging.StreamHandler(stream or sys.stderr)
    handler.setFormatter(formatter)

    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level)

    # These are chatty and rarely useful at INFO.
    for noisy in ("httpx", "httpcore", "urllib3", "sentence_transformers", "filelock"):
        logging.getLogger(noisy).setLevel(max(level, logging.WARNING))

    _CONFIGURED = True


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Return a bound logger, configuring defaults on first use."""
    if not _CONFIGURED:
        configure_logging()
    return structlog.stdlib.get_logger(name)


def new_run_id() -> str:
    """Short id used to correlate every log line from a single ingest/query run."""
    return uuid.uuid4().hex[:12]


def bind_run(**fields: Any) -> None:
    """Attach fields to every subsequent log line on this thread/task."""
    structlog.contextvars.bind_contextvars(**fields)


def clear_run() -> None:
    structlog.contextvars.clear_contextvars()
