"""The golden evaluation set.

A golden item pins an expected source by **file and page**, not by chunk id.
That choice is what makes the set survive the system it evaluates: chunk ids are
content-derived, so changing ``chunk_tokens`` or fixing a heading rule would
invalidate every label overnight and quietly turn a tuning experiment into a
measurement of nothing. File and page are properties of the document, so a
label written once stays true.

Unanswerable questions are first-class. A set containing only answerable ones
cannot distinguish a system that knows things from a system that answers
everything, and the refusal path is the one most likely to regress unnoticed.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

from pydantic import BaseModel, Field, model_validator

from ..logging_setup import get_logger
from ..models import Chunk

log = get_logger(__name__)


class ExpectedSource(BaseModel):
    """Where the answer to a question actually lives."""

    source_file: str
    #: 1-based page as printed. Omit to accept the document anywhere.
    page: int | None = None

    def matches(self, chunk: Chunk) -> bool:
        if Path(self.source_file).name.lower() != Path(chunk.source_file).name.lower():
            return False
        if self.page is None:
            return True
        # A chunk can span pages, so containment rather than equality.
        return chunk.page_start <= self.page <= chunk.page_end

    def label(self) -> str:
        return f"{self.source_file}" + (f" p.{self.page}" if self.page else "")


class GoldenItem(BaseModel):
    id: str
    question: str = Field(min_length=1)
    #: What a correct answer says. Free text - used for reporting and for the
    #: optional RAGAS metrics, never for exact-match scoring.
    expected_answer: str | None = None
    expected_sources: list[ExpectedSource] = Field(default_factory=list)
    #: False means the documents do not contain the answer and the system is
    #: expected to refuse.
    answerable: bool = True
    #: Values a correct answer must contain verbatim - exact numbers, parameter
    #: names, error codes. The cheapest signal that survives rewording.
    #:
    #: An entry may be a list of alternatives, any one of which satisfies it.
    #: This matters more than it looks: documents write the same value as "30"
    #: in a table and "thirty" in prose, and a single required spelling turns a
    #: correct answer into a failing one.
    #:
    #:     "must_contain": ["request_timeout", ["30", "thirty"]]
    must_contain: list[str | list[str]] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _unanswerable_has_no_sources(self) -> GoldenItem:
        if not self.answerable and self.expected_sources:
            raise ValueError(
                f"{self.id}: an unanswerable question cannot have expected_sources"
            )
        if self.answerable and not self.expected_sources:
            raise ValueError(
                f"{self.id}: an answerable question needs at least one expected source"
            )
        return self

    def is_relevant(self, chunk: Chunk) -> bool:
        return any(expected.matches(chunk) for expected in self.expected_sources)

    def relevance_vector(self, chunks: Sequence[Chunk]) -> list[bool]:
        """Mark each retrieved chunk relevant only if it covers *new* evidence.

        Chunking regularly produces several chunks on one page - a prose lead-in
        and the table it introduces, say - and all of them match a single
        ``(file, page)`` label. Counting each as a separate hit lets the number
        of relevant results exceed the number that exist, which pushes nDCG
        above 1.0 and makes the metric meaningless.

        Each expected source can therefore be found exactly once, by the
        highest-ranked chunk that satisfies it.
        """
        covered: set[int] = set()
        flags: list[bool] = []

        for chunk in chunks:
            newly = {
                index
                for index, expected in enumerate(self.expected_sources)
                if index not in covered and expected.matches(chunk)
            }
            covered |= newly
            flags.append(bool(newly))

        return flags


class GoldenSet(BaseModel):
    items: list[GoldenItem] = Field(default_factory=list)
    path: str | None = None

    def __len__(self) -> int:
        return len(self.items)

    @property
    def answerable(self) -> list[GoldenItem]:
        return [item for item in self.items if item.answerable]

    @property
    def unanswerable(self) -> list[GoldenItem]:
        return [item for item in self.items if not item.answerable]

    def stats(self) -> dict[str, int]:
        return {
            "total": len(self.items),
            "answerable": len(self.answerable),
            "unanswerable": len(self.unanswerable),
            "with_must_contain": sum(1 for item in self.items if item.must_contain),
        }


def load_golden_set(path: str | Path) -> GoldenSet:
    """Read a golden set from JSON Lines.

    A malformed line fails the load rather than being skipped: a silently
    shrinking evaluation set would make metrics improve for no reason.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"golden set not found: {path}")

    items: list[GoldenItem] = []
    seen: set[str] = set()

    # utf-8-sig, not utf-8: this file is hand-authored, often on Windows, where
    # editors and PowerShell readily prepend a byte-order mark. Failing to load
    # a golden set because of an invisible byte is a miserable way to lose an
    # afternoon. The codec is a no-op when no BOM is present.
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line or line.startswith("//"):
                continue
            try:
                item = GoldenItem.model_validate_json(line)
            except Exception as exc:
                raise ValueError(f"{path}:{line_number} is not a valid golden item: {exc}") from exc

            if item.id in seen:
                raise ValueError(f"{path}:{line_number} duplicate id {item.id!r}")
            seen.add(item.id)
            items.append(item)

    golden = GoldenSet(items=items, path=str(path))
    log.info("golden_set_loaded", path=str(path), **golden.stats())
    return golden


def write_golden_set(path: str | Path, items: list[GoldenItem]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "\n".join(item.model_dump_json(exclude_none=True) for item in items)
    path.write_text(payload + "\n" if payload else "", encoding="utf-8")


def coverage_report(golden: GoldenSet, chunks: list[Chunk]) -> dict[str, object]:
    """Check that every expected source actually exists in the index.

    A label pointing at a page that was never ingested scores as a permanent
    retrieval miss, and looks exactly like a retrieval bug.
    """
    unreachable: list[str] = []
    for item in golden.answerable:
        for expected in item.expected_sources:
            if not any(expected.matches(chunk) for chunk in chunks):
                unreachable.append(f"{item.id}: {expected.label()}")

    if unreachable:
        log.warning(
            "golden_sources_unreachable",
            count=len(unreachable),
            examples=unreachable[:5],
        )
    return {
        "expected_sources": sum(len(i.expected_sources) for i in golden.answerable),
        "unreachable": unreachable,
    }


def to_jsonl_template() -> str:
    """A commented starter file, written by scripts/make_golden.py."""
    examples = [
        GoldenItem(
            id="q001",
            question="What is the default request timeout?",
            expected_answer="30 seconds.",
            expected_sources=[ExpectedSource(source_file="handbook.pdf", page=3)],
            must_contain=["30"],
            tags=["parameter"],
        ),
        GoldenItem(
            id="q002",
            question="What is the company's parental leave policy?",
            answerable=False,
            tags=["unanswerable"],
        ),
    ]
    return "\n".join(item.model_dump_json(exclude_none=True) for item in examples) + "\n"


def read_jsonl(path: Path) -> list[dict[str, object]]:
    """Plain JSONL reader used by the harness for run artifacts."""
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text("utf-8").splitlines() if line.strip()]
