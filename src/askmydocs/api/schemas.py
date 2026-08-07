"""Request and response models for the HTTP API.

Separate from the domain models on purpose. The wire format is a contract with
the front end and with anyone scripting against this service; letting internal
types leak onto it would mean every refactor of a chunk or a candidate becomes a
breaking API change.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

from ..models import Answer, CitationIssue, Source, VerificationReport


class ErrorResponse(BaseModel):
    """Every failure comes back in this shape."""

    error: str = Field(description="Machine-readable error code")
    detail: str = Field(description="Human-readable explanation")
    request_id: str | None = None


# --------------------------------------------------------------------------
# Query
# --------------------------------------------------------------------------


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    #: Per-request overrides. Omitted values fall back to the server config, so
    #: the front end never has to know the defaults.
    top_k: int | None = Field(default=None, gt=0, le=50)
    rerank: bool | None = None
    include_source_text: bool = True

    model_config = {
        "json_schema_extra": {
            "examples": [{"question": "What is the default request timeout?"}]
        }
    }


class SourceOut(BaseModel):
    number: int
    chunk_id: str
    doc_title: str
    source_file: str
    page_label: str
    page_start: int
    page_end: int
    section_path: list[str] = Field(default_factory=list)
    text: str | None = None
    rerank_score: float | None = None

    @classmethod
    def from_source(cls, source: Source, include_text: bool = True) -> SourceOut:
        start, end = _page_bounds(source.page_label)
        return cls(
            number=source.number,
            chunk_id=source.chunk_id,
            doc_title=source.doc_title,
            source_file=source.source_file,
            page_label=source.page_label,
            page_start=start,
            page_end=end,
            section_path=source.section_path,
            text=source.text if include_text else None,
            rerank_score=source.rerank_score,
        )


class IssueOut(BaseModel):
    type: str
    severity: str
    detail: str
    sentence: str | None = None
    citation: int | None = None

    @classmethod
    def from_issue(cls, issue: CitationIssue) -> IssueOut:
        return cls(
            type=issue.type.value,
            severity=issue.severity.value,
            detail=issue.detail,
            sentence=issue.sentence,
            citation=issue.citation,
        )


class VerificationOut(BaseModel):
    passed: bool
    citation_precision: float
    claim_coverage: float
    total_claims: int
    cited_claims: int
    uncited_claims: int
    unsupported_claims: int
    invalid_citations: list[int] = Field(default_factory=list)
    unused_sources: list[int] = Field(default_factory=list)
    entailment_mode: str = "off"
    issues: list[IssueOut] = Field(default_factory=list)

    @classmethod
    def from_report(cls, report: VerificationReport) -> VerificationOut:
        return cls(
            passed=report.passed,
            citation_precision=round(report.citation_precision, 4),
            claim_coverage=round(report.claim_coverage, 4),
            total_claims=report.total_claims,
            cited_claims=report.cited_claims,
            uncited_claims=report.uncited_claims,
            unsupported_claims=report.unsupported_claims,
            invalid_citations=report.invalid_citations,
            unused_sources=report.unused_sources,
            entailment_mode=report.entailment_mode,
            issues=[IssueOut.from_issue(issue) for issue in report.issues],
        )


class QueryResponse(BaseModel):
    question: str
    answer: str
    #: True when the model declined because the sources did not support an
    #: answer. Clients should present this differently from a real answer.
    refused: bool
    sources: list[SourceOut] = Field(default_factory=list)
    verification: VerificationOut | None = None
    model: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    retrieval_ms: float = 0.0
    generation_ms: float = 0.0
    total_ms: float = 0.0
    request_id: str | None = None

    @classmethod
    def from_answer(
        cls,
        answer: Answer,
        *,
        include_source_text: bool,
        retrieval_ms: float,
        total_ms: float,
        request_id: str | None = None,
    ) -> QueryResponse:
        return cls(
            question=answer.question,
            answer=answer.text,
            refused=answer.refused,
            sources=[
                SourceOut.from_source(source, include_source_text)
                for source in answer.sources
            ],
            verification=(
                VerificationOut.from_report(answer.verification)
                if answer.verification is not None
                else None
            ),
            model=answer.model,
            prompt_tokens=answer.prompt_tokens,
            completion_tokens=answer.completion_tokens,
            retrieval_ms=round(retrieval_ms, 1),
            generation_ms=answer.latency_ms,
            total_ms=round(total_ms, 1),
            request_id=request_id,
        )


# --------------------------------------------------------------------------
# Health
# --------------------------------------------------------------------------


class ComponentHealth(BaseModel):
    name: str
    ready: bool
    detail: str | None = None


class HealthResponse(BaseModel):
    #: "ok" means queries will work. "degraded" means the service is up but
    #: something (usually a missing index) will make queries fail.
    status: Literal["ok", "degraded"]
    version: str
    index_built: bool
    chunk_count: int = 0
    document_count: int = 0
    embedding_model: str | None = None
    indexed_at: datetime | None = None
    components: list[ComponentHealth] = Field(default_factory=list)


# --------------------------------------------------------------------------
# Ingestion and jobs
# --------------------------------------------------------------------------


class IngestRequest(BaseModel):
    #: Re-parse every document, ignoring the unchanged-file cache.
    force: bool = False
    #: Rebuild FAISS and BM25 after ingesting. Almost always what you want -
    #: without it, new chunks exist on disk but are invisible to queries.
    rebuild_index: bool = True


class UploadResponse(BaseModel):
    uploaded: list[str] = Field(default_factory=list)
    rejected: list[str] = Field(default_factory=list)
    detail: str


class DocumentSummary(BaseModel):
    source_file: str
    title: str
    status: str
    page_count: int = 0
    chunk_count: int = 0
    structure_source: str | None = None
    reason: str | None = None


class JobResponse(BaseModel):
    """A slow operation, tracked so the caller does not hold a connection open."""

    id: str
    kind: str
    status: str
    created_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None
    detail: str | None = None
    error: str | None = None
    result: dict[str, object] | None = None


class DocumentListResponse(BaseModel):
    documents: list[DocumentSummary] = Field(default_factory=list)
    total_chunks: int = 0
    ingested_at: datetime | None = None


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------


class EvalRequest(BaseModel):
    golden_set: str | None = Field(
        default=None, description="Path to a golden set; defaults to the configured one"
    )
    limit: int | None = Field(default=None, gt=0, description="Evaluate only the first N items")


def _page_bounds(page_label: str) -> tuple[int, int]:
    """Recover numeric page bounds from 'p. 3' or 'pp. 3-5'."""
    digits = [int(part) for part in "".join(
        char if char.isdigit() else " " for char in page_label
    ).split()]
    if not digits:
        return (0, 0)
    return (digits[0], digits[-1])
