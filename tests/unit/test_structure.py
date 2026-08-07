from __future__ import annotations

from pathlib import Path

from askmydocs.config import AppConfig, StructureConfig
from askmydocs.ingestion.pdf_loader import load_pdf
from askmydocs.ingestion.structure import (
    body_font_size,
    build_sections,
    detect_headings,
)
from askmydocs.models import Line, ParsedDocument, ParsedPage, StructureSource
from fixtures import pdf_factory as pf


def _line(text: str, size: float = 10.0, bold: bool = False, page: int = 1, top: float = 0.0):
    return Line(
        text=text, size=size, font="helv", is_bold=bold, bbox=(0, top, 400, top + size), page_no=page
    )


def _document(lines: list[Line]) -> ParsedDocument:
    pages: dict[int, ParsedPage] = {}
    for line in lines:
        page = pages.setdefault(line.page_no, ParsedPage(page_no=line.page_no, height=800))
        page.elements.append(line)
    return ParsedDocument(
        doc_id="test",
        source_path="test.pdf",
        filename="test.pdf",
        title="Test",
        pages=[pages[k] for k in sorted(pages)],
    )


# -- body font size --------------------------------------------------------


def test_body_font_size_is_weighted_by_characters() -> None:
    # One huge title must not outvote the bulk of the prose.
    lines = [_line("A GIANT TITLE", size=30.0)] + [
        _line("ordinary body text that goes on for a while", size=10.0) for _ in range(20)
    ]
    assert body_font_size(lines) == 10.0


def test_body_font_size_of_empty_document_is_zero() -> None:
    assert body_font_size([]) == 0.0


# -- heading detection -----------------------------------------------------


def test_larger_font_is_a_heading() -> None:
    config = StructureConfig()
    doc = _document(
        [
            _line("Introduction", size=18.0),
            *[_line(f"body sentence number {i} here", size=10.0) for i in range(10)],
        ]
    )
    headings = detect_headings(doc, config)
    assert [h.text for h in headings] == ["Introduction"]


def test_numbered_heading_depth_sets_level() -> None:
    config = StructureConfig(numbered_heading_patterns=[r"^\d+(\.\d+)*[\.\)]?\s+\S"])
    doc = _document(
        [
            _line("1 Overview"),
            _line("1.2 Details"),
            _line("1.2.3 Specifics"),
            *[_line("filler body text here for size stats") for _ in range(10)],
        ]
    )
    levels = {h.text: h.level for h in detect_headings(doc, config)}
    assert levels["1 Overview"] == 1
    assert levels["1.2 Details"] == 2
    assert levels["1.2.3 Specifics"] == 3


def test_numbering_depth_is_capped_at_max_depth() -> None:
    config = StructureConfig(
        max_heading_depth=2, numbered_heading_patterns=[r"^\d+(\.\d+)*[\.\)]?\s+\S"]
    )
    doc = _document(
        [_line("1.2.3.4.5 Deep"), *[_line("filler body text here") for _ in range(10)]]
    )
    assert detect_headings(doc, config)[0].level == 2


def test_bold_sentence_is_not_a_heading() -> None:
    # Bold alone is weak evidence; a full punctuated sentence stays body text.
    config = StructureConfig()
    doc = _document(
        [
            _line("This entire sentence is emphasised for effect.", bold=True),
            *[_line("regular body text line here") for _ in range(10)],
        ]
    )
    assert detect_headings(doc, config) == []


def test_long_line_in_large_font_is_not_a_heading() -> None:
    config = StructureConfig(heading_max_words=4)
    doc = _document(
        [
            _line("one two three four five six seven eight nine ten", size=18.0),
            *[_line("regular body text line") for _ in range(10)],
        ]
    )
    assert detect_headings(doc, config) == []


def test_headings_found_in_a_real_pdf(tmp_path: Path, config: AppConfig) -> None:
    document = load_pdf(pf.structured_pdf(tmp_path / "doc.pdf"), config.ingestion)
    texts = [h.text for h in detect_headings(document, config.structure)]
    assert "1. Overview" in texts
    assert "2. Rollout Stages" in texts
    assert "2.2 Rollback" in texts


# -- section building ------------------------------------------------------


def test_sections_carry_ancestor_breadcrumbs(tmp_path: Path, config: AppConfig) -> None:
    document = load_pdf(pf.structured_pdf(tmp_path / "doc.pdf"), config.ingestion)
    sections, source, heading_count = build_sections(document, config.structure)

    assert source is StructureSource.HEADINGS
    assert heading_count >= 3

    rollback = next(s for s in sections if s.path and s.path[-1].startswith("2.2"))
    assert rollback.path[0].startswith("2.")
    assert "Rollback is triggered automatically" in rollback.text


def test_section_content_does_not_leak_across_headings(
    tmp_path: Path, config: AppConfig
) -> None:
    document = load_pdf(pf.structured_pdf(tmp_path / "doc.pdf"), config.ingestion)
    sections, _, _ = build_sections(document, config.structure)

    canary = next(s for s in sections if s.path and s.path[-1].startswith("2.1"))
    assert "one percent" in canary.text
    assert "Rollback is triggered" not in canary.text


def test_unstructured_document_falls_back_to_pages(tmp_path: Path, config: AppConfig) -> None:
    document = load_pdf(pf.unstructured_pdf(tmp_path / "doc.pdf"), config.ingestion)
    sections, source, _ = build_sections(document, config.structure)

    assert source is StructureSource.PAGE_FALLBACK
    assert len(sections) == document.page_count
    for section in sections:
        pages = {b.page_no for b in section.blocks}
        assert len(pages) == 1, "a fallback section must not span pages"


def test_tables_are_preserved_as_blocks(tmp_path: Path, config: AppConfig) -> None:
    document = load_pdf(pf.table_pdf(tmp_path / "doc.pdf"), config.ingestion)
    sections, _, _ = build_sections(document, config.structure)

    table_blocks = [b for s in sections for b in s.blocks if b.kind == "table"]
    assert table_blocks
    assert "request_timeout" in table_blocks[0].text


def test_hyphenated_line_breaks_are_repaired() -> None:
    config = StructureConfig()
    doc = _document(
        [
            _line("The configura-", page=1, top=100),
            _line("tion is applied at startup.", page=1, top=112),
            *[_line("more body text here", page=1, top=200 + i * 12) for i in range(10)],
        ]
    )
    sections, _, _ = build_sections(doc, config)
    assert "configuration is applied" in " ".join(s.text for s in sections)
