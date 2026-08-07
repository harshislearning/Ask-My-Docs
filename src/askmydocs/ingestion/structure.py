"""Heading detection and section building.

PDFs carry no semantic markup, so structure has to be inferred from typography:
a line is a heading if it is set larger than body text, or bold and short, or
matches a numbering convention. Documents where that inference fails fall back
to page-bounded sections, and the choice is recorded on every chunk so
retrieval quality can be compared across the two modes.
"""

from __future__ import annotations

import re
import statistics
from collections import Counter

from ..config import StructureConfig
from ..logging_setup import get_logger
from ..models import (
    Block,
    Heading,
    Line,
    ParsedDocument,
    ParsedPage,
    Section,
    StructureSource,
    TableBlock,
)

log = get_logger(__name__)

_TERMINAL_PUNCT = (".", ",", ";")
_NUMBER_PREFIX = re.compile(r"^(\d+(?:\.\d+)*)")
#: A heading in a larger font may run longer than a bold one; this multiplies
#: heading_max_words for the font-size rule only.
_LARGE_FONT_WORD_SLACK = 2


def body_font_size(lines: list[Line]) -> float:
    """The document's dominant body text size.

    Character-weighted: a 40pt cover title on one line must not outvote
    thousands of characters of 10pt prose.
    """
    if not lines:
        return 0.0
    weights: Counter[float] = Counter()
    for line in lines:
        weights[round(line.size, 1)] += len(line.text)
    return weights.most_common(1)[0][0]


def detect_headings(document: ParsedDocument, config: StructureConfig) -> list[Heading]:
    """Find heading lines and assign them nesting levels."""
    lines = document.lines()
    body_size = body_font_size(lines)
    patterns = [re.compile(p) for p in config.numbered_heading_patterns]

    candidates: list[tuple[int, Line, str | None]] = []
    for order, (_, element) in enumerate(_flatten(document)):
        if not isinstance(element, Line):
            continue
        number = _numbering(element.text, patterns)
        if _is_heading(element, body_size, config, matched_number=number is not None):
            candidates.append((order, element, number))

    if not candidates:
        return []

    size_levels = _levels_from_font_size(
        [line.size for _, line, _ in candidates], config.max_heading_depth
    )

    headings: list[Heading] = []
    for order, line, number in candidates:
        # Explicit numbering ("3.2.1") states the depth outright; trust it over
        # font-size ranking, which is only a proxy.
        if number:
            level = min(number.count(".") + 1, config.max_heading_depth)
        else:
            level = size_levels[round(line.size, 2)]
        headings.append(Heading(text=line.text, level=level, page_no=line.page_no, order=order))

    log.debug(
        "headings_detected",
        doc_id=document.doc_id,
        count=len(headings),
        body_font_size=body_size,
    )
    return headings


def build_sections(
    document: ParsedDocument, config: StructureConfig
) -> tuple[list[Section], StructureSource, int]:
    """Split a document into sections.

    Returns the sections, which strategy produced them, and how many headings
    were detected (recorded in the manifest so you can spot documents whose
    structure was missed).
    """
    headings = detect_headings(document, config)
    density = len(headings) / max(document.page_count, 1)

    if density < config.min_headings_per_page:
        log.info(
            "structure_fallback",
            doc_id=document.doc_id,
            headings=len(headings),
            pages=document.page_count,
            headings_per_page=round(density, 3),
            reason="heading density below threshold",
        )
        return _page_sections(document), StructureSource.PAGE_FALLBACK, len(headings)

    return _heading_sections(document, headings), StructureSource.HEADINGS, len(headings)


# --------------------------------------------------------------------------
# Heading classification
# --------------------------------------------------------------------------


def _is_heading(
    line: Line, body_size: float, config: StructureConfig, *, matched_number: bool
) -> bool:
    text = line.text.strip()
    if not text or len(text) > 200:
        return False

    words = len(text.split())
    ends_like_sentence = text.endswith(_TERMINAL_PUNCT)

    if matched_number and words <= config.heading_max_words * _LARGE_FONT_WORD_SLACK:
        return True

    set_larger_than_body = body_size > 0 and line.size >= body_size * config.heading_size_ratio
    if (
        set_larger_than_body
        and words <= config.heading_max_words * _LARGE_FONT_WORD_SLACK
        and not ends_like_sentence
    ):
        return True

    # Bold alone is weak evidence - require it to be short and unpunctuated,
    # otherwise every emphasised sentence becomes a section break.
    return bool(
        config.bold_qualifies_as_heading
        and line.is_bold
        and words <= config.heading_max_words
        and not ends_like_sentence
    )


def _numbering(text: str, patterns: list[re.Pattern[str]]) -> str | None:
    """Return the numeric prefix ('3.2') if the line looks like a numbered heading."""
    if not any(p.match(text.strip()) for p in patterns):
        return None
    match = _NUMBER_PREFIX.match(text.strip())
    return match.group(1) if match else ""


def _levels_from_font_size(sizes: list[float], max_depth: int) -> dict[float, int]:
    """Rank distinct heading font sizes; largest becomes level 1."""
    distinct = sorted({round(s, 2) for s in sizes}, reverse=True)
    return {size: min(rank + 1, max_depth) for rank, size in enumerate(distinct)}


# --------------------------------------------------------------------------
# Section assembly
# --------------------------------------------------------------------------


def _heading_sections(document: ParsedDocument, headings: list[Heading]) -> list[Section]:
    heading_orders = {h.order: h for h in headings}
    sections: list[Section] = []
    stack: list[Heading] = []

    # Content before the first heading still belongs to the document.
    current = Section(path=[], level=0)
    buffer: list[Line] = []

    for order, (page, element) in enumerate(_flatten(document)):
        heading = heading_orders.get(order)

        if heading is not None:
            _flush(buffer, current)
            buffer = []
            if not current.is_empty() or current.path:
                sections.append(current)

            while stack and stack[-1].level >= heading.level:
                stack.pop()
            stack.append(heading)
            current = Section(path=[h.text for h in stack], level=heading.level)
            continue

        if isinstance(element, TableBlock):
            _flush(buffer, current)
            buffer = []
            current.blocks.append(
                Block(kind="table", text=element.markdown, page_no=element.page_no)
            )
        elif isinstance(element, Line):
            if buffer and _paragraph_break(buffer[-1], element, page):
                _flush(buffer, current)
                buffer = []
            buffer.append(element)

    _flush(buffer, current)
    if not current.is_empty() or current.path:
        sections.append(current)

    return [s for s in sections if not s.is_empty()]


def _page_sections(document: ParsedDocument) -> list[Section]:
    """Fallback for documents with no usable heading structure.

    Each page becomes its own section: citations stay exact to a single page,
    and unrelated content either side of a page break never merges into one
    chunk. Paths stay empty so embed_text is prefixed with the document title
    alone rather than a meaningless 'Page 7'.
    """
    sections: list[Section] = []
    for page in document.pages:
        section = Section(path=[], level=0)
        buffer: list[Line] = []
        for element in page.elements:
            if isinstance(element, TableBlock):
                _flush(buffer, section)
                buffer = []
                section.blocks.append(
                    Block(kind="table", text=element.markdown, page_no=element.page_no)
                )
            else:
                if buffer and _paragraph_break(buffer[-1], element, page):
                    _flush(buffer, section)
                    buffer = []
                buffer.append(element)
        _flush(buffer, section)
        if not section.is_empty():
            sections.append(section)
    return sections


def _flush(buffer: list[Line], section: Section) -> None:
    """Turn buffered lines into one prose block."""
    if not buffer:
        return
    text = _join_lines(buffer)
    if text.strip():
        section.blocks.append(Block(kind="prose", text=text, page_no=buffer[0].page_no))


def _join_lines(lines: list[Line]) -> str:
    """Rebuild a paragraph from hard-wrapped PDF lines, repairing hyphenation."""
    parts: list[str] = []
    for line in lines:
        text = line.text.strip()
        if not text:
            continue
        if parts and parts[-1].endswith("-") and not parts[-1].endswith(("--", " -")):
            parts[-1] = parts[-1][:-1] + text
        else:
            parts.append(text)
    return " ".join(parts)


def _paragraph_break(previous: Line, current: Line, page: ParsedPage) -> bool:
    """True when vertical whitespace says these lines are separate paragraphs."""
    if previous.page_no != current.page_no:
        return True
    gap = current.bbox[1] - previous.bbox[3]
    typical = _typical_line_height(page)
    return typical > 0 and gap > typical * 0.6


def _typical_line_height(page: ParsedPage) -> float:
    heights = [e.height for e in page.elements if isinstance(e, Line) and e.height > 0]
    return statistics.median(heights) if heights else 0.0


def _flatten(document: ParsedDocument) -> list[tuple[ParsedPage, Line | TableBlock]]:
    """Document-order stream of elements, paired with the page they came from."""
    return [(page, element) for page in document.pages for element in page.elements]
