"""Builds synthetic PDFs at test time.

Checking binary fixtures into git makes tests opaque - you cannot see why a
document is "unstructured" by reading the test. Here the input is declared in
Python, so every assertion is traceable to the typography that produced it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import fitz

#: Gap between items, as a multiple of font size. LINE_SPACING is small enough
#: that adjacent items read as one paragraph; PARAGRAPH_GAP is large enough to
#: split them. See structure._paragraph_break.
LINE_SPACING = 0.2
PARAGRAPH_GAP = 1.4

BODY_SIZE = 10.0
H1_SIZE = 18.0
H2_SIZE = 14.0

PAGE_WIDTH = 595.0
PAGE_HEIGHT = 842.0
MARGIN_X = 60.0
MARGIN_TOP = 60.0


@dataclass
class Text:
    """One line of text to place on a page."""

    text: str
    size: float = BODY_SIZE
    bold: bool = False
    new_paragraph: bool = False


@dataclass
class Table:
    """A grid drawn with real cell borders so PyMuPDF's detector finds it."""

    rows: list[list[str]]
    new_paragraph: bool = True


Element = Text | Table


def make_pdf(path: Path, pages: list[list[Element]]) -> Path:
    """Render ``pages`` to a PDF at ``path``."""
    doc = fitz.open()
    for elements in pages:
        page = doc.new_page(width=PAGE_WIDTH, height=PAGE_HEIGHT)
        cursor = MARGIN_TOP
        for element in elements:
            if isinstance(element, Table):
                cursor = _draw_table(page, element, cursor)
            else:
                cursor = _draw_text(page, element, cursor)
    path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(path))
    doc.close()
    return path


def _draw_text(page: fitz.Page, item: Text, cursor: float) -> float:
    """Draw one paragraph, wrapped inside the margins.

    Wrapping matters: text drawn past the page edge is clipped on render and
    silently lost on extraction, which would make fixtures lie about what the
    parser sees.
    """
    cursor += item.size * (PARAGRAPH_GAP if item.new_paragraph else LINE_SPACING)
    box = fitz.Rect(MARGIN_X, cursor, PAGE_WIDTH - MARGIN_X, PAGE_HEIGHT - MARGIN_TOP)
    leftover = page.insert_textbox(
        box,
        item.text,
        fontsize=item.size,
        fontname="hebo" if item.bold else "helv",
    )
    if leftover < 0:  # pragma: no cover - fixture authoring error
        raise ValueError(f"text does not fit on the page: {item.text[:40]!r}")
    return box.y1 - leftover


def _draw_table(page: fitz.Page, table: Table, cursor: float) -> float:
    cursor += BODY_SIZE * PARAGRAPH_GAP
    col_width = (PAGE_WIDTH - 2 * MARGIN_X) / max(len(table.rows[0]), 1)
    row_height = 22.0

    for r, row in enumerate(table.rows):
        top = cursor + r * row_height
        for c, cell in enumerate(row):
            left = MARGIN_X + c * col_width
            rect = fitz.Rect(left, top, left + col_width, top + row_height)
            page.draw_rect(rect, color=(0, 0, 0), width=0.7)
            inner = fitz.Rect(left + 3, top + 4, left + col_width - 3, top + row_height - 2)
            page.insert_textbox(inner, cell, fontsize=8, fontname="helv")

    return cursor + len(table.rows) * row_height


# --------------------------------------------------------------------------
# Ready-made documents
# --------------------------------------------------------------------------


def structured_pdf(path: Path) -> Path:
    """Numbered headings in larger, bolder type across three pages."""
    body = (
        "The deployment service coordinates rollout across every regional cluster. "
        "It waits for health checks to pass before advancing to the next stage."
    )
    detail = (
        "Rollback is triggered automatically when the error rate exceeds the configured "
        "budget for two consecutive evaluation windows. Operators can also trigger it "
        "manually from the control plane."
    )
    return make_pdf(
        path,
        [
            [
                Text("Deployment Handbook", size=H1_SIZE, bold=True),
                Text("1. Overview", size=H2_SIZE, bold=True, new_paragraph=True),
                Text(body, new_paragraph=True),
                Text("Each stage is gated on the previous one completing cleanly."),
            ],
            [
                Text("2. Rollout Stages", size=H2_SIZE, bold=True),
                Text("Stages advance from canary to partial to full fleet.", new_paragraph=True),
                Text("2.1 Canary", size=BODY_SIZE * 1.2, bold=True, new_paragraph=True),
                Text("The canary stage routes one percent of traffic.", new_paragraph=True),
                Text("2.2 Rollback", size=BODY_SIZE * 1.2, bold=True, new_paragraph=True),
                Text(detail, new_paragraph=True),
            ],
            [
                Text("3. Timeouts", size=H2_SIZE, bold=True),
                Text("Default request timeout is thirty seconds.", new_paragraph=True),
                Text("Health check timeout is five seconds and is not configurable."),
            ],
        ],
    )


def unstructured_pdf(path: Path) -> Path:
    """Uniform body type throughout - nothing for heading detection to latch onto.

    Page content differs per page, as in a real document; identical text on
    every page would (correctly) be treated as running furniture.
    """
    topics = [
        "capacity planning for the shared ingest tier during quarter close",
        "an incident where the retry budget was exhausted by a downstream timeout",
        "the migration of batch jobs onto the newer scheduler and what regressed",
    ]
    return make_pdf(
        path,
        [
            [
                Text(
                    f"These operating notes record {topic}. Nothing here follows a formal "
                    "template and the text simply runs on without any headings to separate "
                    "one topic from the next.",
                    new_paragraph=True,
                ),
                Text(f"Further detail on {topic} continues in the same uniform typeface."),
            ]
            for topic in topics
        ],
    )


def table_pdf(path: Path) -> Path:
    """A document whose real content is a parameter table."""
    return make_pdf(
        path,
        [
            [
                Text("Configuration Reference", size=H1_SIZE, bold=True),
                Text("4. Timeout Parameters", size=H2_SIZE, bold=True, new_paragraph=True),
                Text("The table below lists every tunable timeout.", new_paragraph=True),
                Table(
                    rows=[
                        ["Parameter", "Default", "Unit"],
                        ["request_timeout", "30", "seconds"],
                        ["health_timeout", "5", "seconds"],
                        ["drain_timeout", "120", "seconds"],
                    ]
                ),
            ]
        ],
    )


def paged_pdf_with_furniture(path: Path, pages: int = 6) -> Path:
    """Every page carries the same header and a 'Page N of M' footer."""
    doc = fitz.open()
    for n in range(1, pages + 1):
        page = doc.new_page(width=PAGE_WIDTH, height=PAGE_HEIGHT)
        page.insert_text((MARGIN_X, 30), "ACME Internal - Confidential", fontsize=8)
        # Identical on every page and mid-page: repetition alone must not be
        # enough to strip it, or boilerplate body text would disappear.
        page.insert_text(
            (MARGIN_X, 300),
            "Body content describing the service behaviour in detail.",
            fontsize=BODY_SIZE,
        )
        page.insert_text((MARGIN_X, PAGE_HEIGHT - 25), f"Page {n} of {pages}", fontsize=8)
    path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(path))
    doc.close()
    return path


def scanned_pdf(path: Path, pages: int = 3) -> Path:
    """Pages with no text layer at all - the OCR case we skip."""
    doc = fitz.open()
    for _ in range(pages):
        doc.new_page(width=PAGE_WIDTH, height=PAGE_HEIGHT)
    path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(path))
    doc.close()
    return path


def corrupt_pdf(path: Path) -> Path:
    """A file that claims to be a PDF and is not."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"%PDF-1.7\nthis is not actually a pdf\n")
    return path
