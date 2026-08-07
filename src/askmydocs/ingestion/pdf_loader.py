"""PDF parsing with PyMuPDF.

Produces typed :class:`Line` objects that keep font size and weight, because
that metadata is the only reliable signal for heading detection in PDFs (there
is no semantic markup to read). Tables are pulled out separately and serialised
to markdown before they can be flattened into unreadable column soup.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections import defaultdict
from pathlib import Path

import fitz  # PyMuPDF

from ..config import IngestionConfig
from ..errors import (
    EmptyDocumentError,
    EncryptedPdfError,
    FileTooLargeError,
    PdfParseError,
    ScannedPdfError,
)
from ..logging_setup import get_logger
from ..models import BBox, Line, ParsedDocument, ParsedPage, TableBlock

log = get_logger(__name__)

#: PyMuPDF packs font styling into a bitfield; bit 4 is bold.
_BOLD_FLAG = 1 << 4
_DIGITS = re.compile(r"\d+")
_WS = re.compile(r"\s+")
_READ_CHUNK = 1 << 20


def compute_doc_id(path: Path) -> str:
    """Content hash of the file.

    Identity is the bytes, not the filename: renaming a PDF does not re-ingest
    it, and editing one invalidates only that document's chunks.
    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(_READ_CHUNK):
            digest.update(block)
    return digest.hexdigest()[:20]


def load_pdf(path: Path, config: IngestionConfig) -> ParsedDocument:
    """Parse one PDF into pages of lines and tables.

    Raises a :class:`~askmydocs.errors.DocumentError` subclass for anything
    unusable; the pipeline catches those and records them in the manifest
    rather than failing the run.
    """
    if not path.is_file():
        raise PdfParseError(f"not a file: {path}", path=str(path))

    size_mb = path.stat().st_size / (1024 * 1024)
    if size_mb > config.max_file_mb:
        raise FileTooLargeError(
            f"{size_mb:.1f}MB exceeds max_file_mb={config.max_file_mb}", path=str(path)
        )

    doc_id = compute_doc_id(path)

    try:
        document = fitz.open(path)
    except Exception as exc:
        raise PdfParseError(f"could not open PDF: {exc}", path=str(path)) from exc

    try:
        if document.needs_pass:
            raise EncryptedPdfError("PDF is password protected", path=str(path))

        pages: list[ParsedPage] = []
        table_total = 0
        for index in range(document.page_count):
            try:
                page = document.load_page(index)
                parsed = _parse_page(page, index + 1, config)
            except Exception as exc:
                log.warning(
                    "page_parse_failed", doc_id=doc_id, page_no=index + 1, error=str(exc)
                )
                parsed = ParsedPage(page_no=index + 1)
            table_total += sum(1 for e in parsed.elements if isinstance(e, TableBlock))
            pages.append(parsed)

        _assert_has_text_layer(pages, config, path)
        _strip_furniture(pages, config)

        title = _document_title(document, pages, path)
        metadata = {k: str(v) for k, v in (document.metadata or {}).items() if v}
    finally:
        document.close()

    parsed_doc = ParsedDocument(
        doc_id=doc_id,
        source_path=str(path),
        filename=path.name,
        title=title,
        pages=pages,
        metadata=metadata,
    )

    if not parsed_doc.lines() and not _tables(parsed_doc):
        raise EmptyDocumentError("no text extracted after cleanup", path=str(path))

    log.info(
        "pdf_parsed",
        doc_id=doc_id,
        file=path.name,
        pages=len(pages),
        lines=len(parsed_doc.lines()),
        tables=table_total,
    )
    return parsed_doc


# --------------------------------------------------------------------------
# Page parsing
# --------------------------------------------------------------------------


def _parse_page(page: fitz.Page, page_no: int, config: IngestionConfig) -> ParsedPage:
    tables = _extract_tables(page, page_no) if config.extract_tables else []
    table_boxes = [t.bbox for t in tables]

    lines = _extract_lines(page, page_no, exclude=table_boxes)

    # Reading order: interleave tables back into the prose flow by vertical
    # position, so a table stays attached to the section that introduces it.
    elements: list[Line | TableBlock] = [*lines, *tables]
    elements.sort(key=lambda e: (round(e.bbox[1], 1), round(e.bbox[0], 1)))

    rect = page.rect
    return ParsedPage(page_no=page_no, elements=elements, width=rect.width, height=rect.height)


def _extract_lines(page: fitz.Page, page_no: int, exclude: list[BBox]) -> list[Line]:
    try:
        payload = page.get_text("dict")
    except Exception as exc:
        log.warning("text_extraction_failed", page_no=page_no, error=str(exc))
        return []

    lines: list[Line] = []
    for block in payload.get("blocks", []):
        if block.get("type") != 0:  # 1 == image
            continue
        for raw_line in block.get("lines", []):
            line = _build_line(raw_line, page_no)
            if line is None:
                continue
            if _inside_any(line.bbox, exclude):
                continue  # already captured as part of a table
            lines.append(line)
    return lines


def _build_line(raw_line: dict, page_no: int) -> Line | None:
    spans = [s for s in raw_line.get("spans", []) if s.get("text", "").strip()]
    if not spans:
        return None

    text = _WS.sub(" ", "".join(s["text"] for s in raw_line.get("spans", []))).strip()
    if not text:
        return None

    # Weight font attributes by how much text each span contributes, so a
    # trailing footnote marker cannot redefine the line's style.
    dominant = max(spans, key=lambda s: len(s["text"]))
    bold_chars = sum(len(s["text"]) for s in spans if _is_bold(s))
    total_chars = sum(len(s["text"]) for s in spans)

    return Line(
        text=text,
        size=round(max(float(s.get("size", 0.0)) for s in spans), 2),
        font=str(dominant.get("font", "")),
        is_bold=bold_chars * 2 > total_chars,
        bbox=tuple(float(v) for v in raw_line["bbox"]),  # type: ignore[arg-type]
        page_no=page_no,
    )


def _is_bold(span: dict) -> bool:
    if int(span.get("flags", 0)) & _BOLD_FLAG:
        return True
    # Flags are unreliable for some embedded fonts; the name usually still says so.
    return "bold" in str(span.get("font", "")).lower()


def _extract_tables(page: fitz.Page, page_no: int) -> list[TableBlock]:
    try:
        finder = page.find_tables()
        found = list(getattr(finder, "tables", finder))
    except Exception as exc:
        log.debug("table_detection_failed", page_no=page_no, error=str(exc))
        return []

    blocks: list[TableBlock] = []
    for table in found:
        markdown = _table_to_markdown(table)
        if not markdown:
            continue
        rows, cols = _table_shape(table)
        if rows < 2 or cols < 2:
            continue  # a 1xN "table" is almost always a false positive
        blocks.append(
            TableBlock(
                markdown=markdown,
                bbox=tuple(float(v) for v in table.bbox),  # type: ignore[arg-type]
                page_no=page_no,
                rows=rows,
                cols=cols,
            )
        )
    if blocks:
        log.debug("tables_detected", page_no=page_no, count=len(blocks))
    return blocks


def _table_to_markdown(table: object) -> str:
    to_markdown = getattr(table, "to_markdown", None)
    if callable(to_markdown):
        try:
            return str(to_markdown()).strip()
        except Exception:
            pass
    try:
        rows = table.extract()  # type: ignore[attr-defined]
    except Exception:
        return ""
    return _rows_to_markdown(rows)


def _rows_to_markdown(rows: list[list[str | None]]) -> str:
    cleaned = [[_WS.sub(" ", (cell or "").strip()) for cell in row] for row in rows if row]
    if not cleaned:
        return ""
    width = max(len(row) for row in cleaned)
    cleaned = [row + [""] * (width - len(row)) for row in cleaned]
    header, *body = cleaned
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * width) + "|"]
    lines += ["| " + " | ".join(row) + " |" for row in body]
    return "\n".join(lines)


def _table_shape(table: object) -> tuple[int, int]:
    try:
        rows = table.extract()  # type: ignore[attr-defined]
        return len(rows), max((len(r) for r in rows), default=0)
    except Exception:
        return (
            int(getattr(table, "row_count", 0) or 0),
            int(getattr(table, "col_count", 0) or 0),
        )


def _inside_any(bbox: BBox, boxes: list[BBox]) -> bool:
    cx = (bbox[0] + bbox[2]) / 2
    cy = (bbox[1] + bbox[3]) / 2
    return any(x0 <= cx <= x1 and y0 <= cy <= y1 for x0, y0, x1, y1 in boxes)


# --------------------------------------------------------------------------
# Document-level cleanup
# --------------------------------------------------------------------------


def _assert_has_text_layer(
    pages: list[ParsedPage], config: IngestionConfig, path: Path
) -> None:
    if not pages:
        raise EmptyDocumentError("PDF has no pages", path=str(path))
    blank = sum(1 for p in pages if p.char_count < config.scanned_page_char_threshold)
    if blank / len(pages) >= config.scanned_doc_page_ratio:
        raise ScannedPdfError(
            f"{blank}/{len(pages)} pages have no text layer; needs OCR", path=str(path)
        )


def _strip_furniture(pages: list[ParsedPage], config: IngestionConfig) -> None:
    """Drop running headers and footers.

    They repeat on every page, so left in they would add hundreds of
    near-identical chunks and skew BM25 term statistics.
    """
    if len(pages) < 3:
        return  # too little evidence to distinguish furniture from content

    occurrences: dict[str, set[int]] = defaultdict(set)
    for page in pages:
        for line in _margin_lines(page, config):
            occurrences[_furniture_key(line.text)].add(page.page_no)

    threshold = max(2, math.ceil(len(pages) * config.furniture_page_ratio))
    furniture = {key for key, seen in occurrences.items() if len(seen) >= threshold}
    if not furniture:
        return

    removed = 0
    for page in pages:
        margin = {id(line) for line in _margin_lines(page, config)}
        keep: list[Line | TableBlock] = []
        for element in page.elements:
            if (
                isinstance(element, Line)
                and id(element) in margin
                and _furniture_key(element.text) in furniture
            ):
                removed += 1
                continue
            keep.append(element)
        page.elements = keep

    log.debug("furniture_stripped", patterns=len(furniture), lines_removed=removed)


def _margin_lines(page: ParsedPage, config: IngestionConfig) -> list[Line]:
    """The only lines eligible to be stripped as furniture.

    Deliberately narrow: at most the single topmost and single bottommost line
    of the page, and only if it sits in the margin and is short. A wider rule
    eats real content - a body paragraph wraps into several short lines, and if
    the whole margin band were eligible, a page whose text starts high would
    lose its opening sentences.
    """
    lines = [e for e in page.elements if isinstance(e, Line)]
    if not lines or page.height <= 0:
        return []

    def eligible(line: Line) -> bool:
        return len(line.text) <= config.furniture_max_chars

    candidates: list[Line] = []
    top_limit = page.height * config.furniture_margin_ratio
    bottom_limit = page.height * (1 - config.furniture_margin_ratio)

    first = min(lines, key=lambda line: line.bbox[1])
    if first.bbox[1] <= top_limit and eligible(first):
        candidates.append(first)

    last = max(lines, key=lambda line: line.bbox[3])
    if last is not first and last.bbox[3] >= bottom_limit and eligible(last):
        candidates.append(last)

    return candidates


def _furniture_key(text: str) -> str:
    """Normalise so 'Page 3 of 40' and 'Page 4 of 40' collapse to one pattern."""
    return _DIGITS.sub("#", _WS.sub(" ", text).strip().lower())


def _document_title(document: fitz.Document, pages: list[ParsedPage], path: Path) -> str:
    meta_title = (document.metadata or {}).get("title", "")
    if meta_title and meta_title.strip() and not meta_title.lower().endswith(".pdf"):
        return meta_title.strip()

    # Fall back to the largest text on page 1 - almost always the cover title.
    if pages:
        first_page_lines = [e for e in pages[0].elements if isinstance(e, Line)]
        if first_page_lines:
            largest = max(first_page_lines, key=lambda line: line.size)
            if len(largest.text) >= 4:
                return largest.text

    return path.stem.replace("_", " ").replace("-", " ").strip()


def _tables(document: ParsedDocument) -> list[TableBlock]:
    return [e for p in document.pages for e in p.elements if isinstance(e, TableBlock)]
