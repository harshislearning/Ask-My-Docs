"""Domain types shared across the pipeline.

Parse-time internals (millions of spans) are dataclasses with ``slots`` for
speed. Anything that crosses a module boundary or gets serialised is a Pydantic
model so the schema is validated and self-documenting.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field

BBox = tuple[float, float, float, float]


# --------------------------------------------------------------------------
# Parse layer (in-memory only)
# --------------------------------------------------------------------------


@dataclass(slots=True)
class Line:
    """One visual line of text with the font metadata structure detection needs."""

    text: str
    size: float
    font: str
    is_bold: bool
    bbox: BBox
    page_no: int  # 1-based, as printed in a citation

    @property
    def top(self) -> float:
        return self.bbox[1]

    @property
    def height(self) -> float:
        return self.bbox[3] - self.bbox[1]


@dataclass(slots=True)
class TableBlock:
    """A detected table, already serialised to markdown."""

    markdown: str
    bbox: BBox
    page_no: int
    rows: int
    cols: int


PageElement = Line | TableBlock


@dataclass(slots=True)
class ParsedPage:
    page_no: int
    elements: list[PageElement] = field(default_factory=list)
    width: float = 0.0
    height: float = 0.0

    @property
    def char_count(self) -> int:
        return sum(
            len(e.text) if isinstance(e, Line) else len(e.markdown) for e in self.elements
        )


@dataclass(slots=True)
class ParsedDocument:
    doc_id: str
    source_path: str
    filename: str
    title: str
    pages: list[ParsedPage]
    metadata: dict[str, str] = field(default_factory=dict)

    @property
    def page_count(self) -> int:
        return len(self.pages)

    def lines(self) -> list[Line]:
        return [e for p in self.pages for e in p.elements if isinstance(e, Line)]


# --------------------------------------------------------------------------
# Structure layer
# --------------------------------------------------------------------------


class StructureSource(StrEnum):
    """How a document's sections were derived - recorded on every chunk so eval
    can compare retrieval quality on structured vs unstructured documents."""

    HEADINGS = "headings"
    PAGE_FALLBACK = "page_fallback"


@dataclass(slots=True)
class Heading:
    text: str
    level: int
    page_no: int
    order: int  # index into the document's flat element stream


@dataclass(slots=True)
class Block:
    """A unit of section content: a prose paragraph or a whole table."""

    kind: Literal["prose", "table"]
    text: str
    page_no: int


@dataclass(slots=True)
class Section:
    """Contiguous content under one heading, with its ancestor trail."""

    path: list[str]  # e.g. ["3. Deployment", "3.2 Rollback"]
    level: int
    blocks: list[Block] = field(default_factory=list)

    @property
    def page_start(self) -> int:
        return min((b.page_no for b in self.blocks), default=1)

    @property
    def page_end(self) -> int:
        return max((b.page_no for b in self.blocks), default=1)

    @property
    def breadcrumb(self) -> str:
        return " > ".join(self.path)

    @property
    def text(self) -> str:
        return "\n\n".join(b.text for b in self.blocks if b.text.strip())

    def is_empty(self) -> bool:
        return not self.text.strip()


# --------------------------------------------------------------------------
# Chunk layer (persisted to chunks.jsonl - the ingestion/indexing boundary)
# --------------------------------------------------------------------------


class Chunk(BaseModel):
    """A retrievable unit.

    ``text`` is what the user and the LLM see. ``embed_text`` is what gets
    embedded and BM25-indexed: same content, prefixed with the breadcrumb so an
    isolated chunk still carries the context it was written under.
    """

    chunk_id: str
    doc_id: str
    source_file: str
    doc_title: str
    text: str
    embed_text: str
    section_path: list[str] = Field(default_factory=list)
    heading_level: int = 0
    page_start: int
    page_end: int
    chunk_index: int
    token_count: int
    content_type: Literal["prose", "table"] = "prose"
    structure_source: StructureSource = StructureSource.HEADINGS

    @property
    def breadcrumb(self) -> str:
        return " > ".join(self.section_path)

    @property
    def page_label(self) -> str:
        """Human-readable page reference for citations in the UI."""
        if self.page_start == self.page_end:
            return f"p. {self.page_start}"
        return f"pp. {self.page_start}-{self.page_end}"

    @staticmethod
    def make_id(doc_id: str, section_path: list[str], chunk_index: int, text: str) -> str:
        """Content-derived id: re-ingesting unchanged input reproduces it exactly,
        so indexes can be updated incrementally instead of rebuilt."""
        payload = "|".join([doc_id, " > ".join(section_path), str(chunk_index), text])
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


# --------------------------------------------------------------------------
# Manifest (what happened during a run - the thing you read when a doc is missing)
# --------------------------------------------------------------------------


# --------------------------------------------------------------------------
# Retrieval layer
# --------------------------------------------------------------------------


class RetrieverName(StrEnum):
    VECTOR = "vector"
    BM25 = "bm25"


class Hit(BaseModel):
    """One result from a single retriever, before fusion."""

    chunk_id: str
    score: float
    rank: int  # 1-based, within that retriever's own result list


class FusedHit(BaseModel):
    """Output of reciprocal rank fusion: an id, its fused score, and the
    per-retriever ranks that produced it (kept for debugging retrieval)."""

    chunk_id: str
    score: float
    rank: int  # 1-based, in the fused ordering
    ranks: dict[str, int] = Field(default_factory=dict)


class Candidate(BaseModel):
    """A retrieved chunk with the full provenance of how it was found.

    Every intermediate score is retained rather than collapsed, because
    diagnosing bad retrieval means asking *which* retriever surfaced a chunk
    and where the other one ranked it.
    """

    chunk: Chunk
    fused_score: float
    fused_rank: int
    ranks: dict[str, int] = Field(default_factory=dict)
    scores: dict[str, float] = Field(default_factory=dict)
    rerank_score: float | None = None
    #: Position after reranking. The gap between this and ``fused_rank`` is how
    #: much the cross-encoder disagreed with the retrievers.
    final_rank: int | None = None

    @property
    def retrievers(self) -> list[str]:
        """Which retrievers found this chunk at all."""
        return sorted(self.ranks)

    @property
    def found_by_both(self) -> bool:
        return len(self.ranks) > 1


class DocumentStatus(StrEnum):
    OK = "ok"
    SKIPPED = "skipped"
    FAILED = "failed"
    UNCHANGED = "unchanged"


class DocumentRecord(BaseModel):
    doc_id: str
    source_file: str
    title: str
    status: DocumentStatus
    page_count: int = 0
    chunk_count: int = 0
    heading_count: int = 0
    structure_source: StructureSource | None = None
    table_count: int = 0
    reason: str | None = None  # populated for SKIPPED / FAILED
    error_code: str | None = None


# --------------------------------------------------------------------------
# Generation layer
# --------------------------------------------------------------------------


class Source(BaseModel):
    """A chunk as presented to the model, under the number it must cite.

    ``number`` is 1-based and stable for one answer. It is the only handle the
    model has on a chunk, and the only thing Phase 5 can verify a citation
    against - so the mapping is stored on the answer, never recomputed.
    """

    number: int
    chunk_id: str
    doc_title: str
    source_file: str
    page_label: str
    section_path: list[str] = Field(default_factory=list)
    text: str
    rerank_score: float | None = None

    @property
    def breadcrumb(self) -> str:
        return " > ".join(self.section_path)

    @property
    def label(self) -> str:
        """Human-readable attribution shown in the prompt and the UI."""
        parts = [self.source_file, self.page_label]
        if self.section_path:
            parts.append(self.breadcrumb)
        return " - ".join(parts)


class LlmResponse(BaseModel):
    """Raw result of one model call, before it becomes an Answer."""

    text: str
    model: str
    finish_reason: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    attempts: int = 1

    @property
    def total_tokens(self) -> int | None:
        if self.prompt_tokens is None or self.completion_tokens is None:
            return None
        return self.prompt_tokens + self.completion_tokens

    @property
    def truncated(self) -> bool:
        """True when the model hit the token ceiling mid-answer.

        Worth surfacing: a truncated answer can lose its closing citation and
        then look like an uncited claim to the verifier.
        """
        return self.finish_reason == "length"


class IssueType(StrEnum):
    """What a verifier found wrong with an answer."""

    #: The answer cited a number that was never in the source list.
    UNKNOWN_SOURCE = "unknown_source"
    #: A sentence asserts something and carries no citation at all.
    UNCITED_CLAIM = "uncited_claim"
    #: A cited chunk does not actually support the claim. (entailment checking)
    UNSUPPORTED_CLAIM = "unsupported_claim"


class Severity(StrEnum):
    ERROR = "error"
    WARNING = "warning"


class CitationIssue(BaseModel):
    """One problem found in an answer, located precisely enough to act on."""

    type: IssueType
    severity: Severity = Severity.ERROR
    detail: str
    sentence: str | None = None
    citation: int | None = None

    def __str__(self) -> str:  # pragma: no cover - diagnostics only
        where = f" ({self.sentence[:60]}...)" if self.sentence else ""
        return f"[{self.severity}] {self.type}: {self.detail}{where}"


class VerificationReport(BaseModel):
    """The result of checking an answer against the sources it was given.

    Reported, never enforced by rewriting the answer: silently editing a model's
    output hides the failure from both the user and the eval harness.
    """

    citations: list[int] = Field(default_factory=list)
    valid_citations: list[int] = Field(default_factory=list)
    invalid_citations: list[int] = Field(default_factory=list)
    unused_sources: list[int] = Field(default_factory=list)
    total_claims: int = 0
    cited_claims: int = 0
    uncited_claims: int = 0
    #: Cited claims the entailment layer judged unsupported by their own source.
    unsupported_claims: int = 0
    issues: list[CitationIssue] = Field(default_factory=list)
    entailment_mode: str = "off"
    passed: bool = True

    @property
    def citation_precision(self) -> float:
        """Share of citations that point at a source that actually exists."""
        if not self.citations:
            return 1.0
        return len(self.valid_citations) / len(self.citations)

    @property
    def claim_coverage(self) -> float:
        """Share of asserting sentences that carry at least one citation."""
        if not self.total_claims:
            return 1.0
        return self.cited_claims / self.total_claims

    @property
    def source_usage(self) -> float:
        """Share of supplied sources the answer actually drew on.

        Low usage is not a defect in the answer - it is a signal that
        ``rerank_top_k`` may be feeding the model more context than it needs.
        """
        total = len(self.unused_sources) + len(set(self.valid_citations))
        if not total:
            return 0.0
        return len(set(self.valid_citations)) / total

    def issues_of(self, issue_type: IssueType) -> list[CitationIssue]:
        return [issue for issue in self.issues if issue.type is issue_type]

    def summary(self) -> dict[str, object]:
        return {
            "passed": self.passed,
            "citations": len(self.citations),
            "invalid_citations": len(self.invalid_citations),
            "claims": self.total_claims,
            "uncited_claims": self.uncited_claims,
            "unsupported_claims": self.unsupported_claims,
            "citation_precision": round(self.citation_precision, 3),
            "claim_coverage": round(self.claim_coverage, 3),
            "issues": len(self.issues),
        }


class Answer(BaseModel):
    """A generated answer together with everything needed to audit it."""

    question: str
    text: str
    sources: list[Source] = Field(default_factory=list)
    refused: bool = False
    model: str = ""
    finish_reason: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    latency_ms: float = 0.0
    attempts: int = 1
    verification: VerificationReport | None = None

    def source_by_number(self, number: int) -> Source | None:
        return next((s for s in self.sources if s.number == number), None)

    @property
    def valid_citation_numbers(self) -> set[int]:
        return {s.number for s in self.sources}


class IndexManifest(BaseModel):
    """What the indexes on disk were built from.

    ``chunks_fingerprint`` is what lets the retriever detect that ingestion has
    run since the index was built - otherwise you get silently stale answers
    that are very hard to trace back to their cause.
    """

    run_id: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    embedding_model: str
    dimension: int
    chunk_count: int
    document_count: int
    chunks_file: str
    chunks_fingerprint: str
    bm25_k1: float
    bm25_b: float


class IngestionManifest(BaseModel):
    run_id: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    config_fingerprint: str
    chunks_file: str
    documents: list[DocumentRecord] = Field(default_factory=list)

    @property
    def total_chunks(self) -> int:
        return sum(d.chunk_count for d in self.documents)

    def by_status(self, status: DocumentStatus) -> list[DocumentRecord]:
        return [d for d in self.documents if d.status is status]

    def summary(self) -> dict[str, int]:
        return {
            "documents_total": len(self.documents),
            "documents_ok": len(self.by_status(DocumentStatus.OK)),
            "documents_unchanged": len(self.by_status(DocumentStatus.UNCHANGED)),
            "documents_skipped": len(self.by_status(DocumentStatus.SKIPPED)),
            "documents_failed": len(self.by_status(DocumentStatus.FAILED)),
            "chunks_total": self.total_chunks,
        }
