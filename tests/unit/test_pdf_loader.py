from __future__ import annotations

from pathlib import Path

import pytest

from askmydocs.config import AppConfig
from askmydocs.errors import EncryptedPdfError, PdfParseError, ScannedPdfError
from askmydocs.ingestion.pdf_loader import compute_doc_id, load_pdf
from askmydocs.models import Line, TableBlock
from fixtures import pdf_factory as pf


def test_doc_id_is_content_derived(tmp_path: Path) -> None:
    first = pf.structured_pdf(tmp_path / "a.pdf")
    renamed = tmp_path / "renamed.pdf"
    renamed.write_bytes(first.read_bytes())

    assert compute_doc_id(first) == compute_doc_id(renamed)

    different = pf.table_pdf(tmp_path / "b.pdf")
    assert compute_doc_id(first) != compute_doc_id(different)


def test_extracts_lines_with_font_metadata(tmp_path: Path, config: AppConfig) -> None:
    document = load_pdf(pf.structured_pdf(tmp_path / "doc.pdf"), config.ingestion)

    assert document.page_count == 3
    lines = document.lines()
    assert lines

    title = next(line for line in lines if line.text.startswith("Deployment Handbook"))
    body = next(line for line in lines if "deployment service" in line.text)
    assert title.size > body.size
    assert title.is_bold
    assert not body.is_bold
    assert title.page_no == 1


def test_title_falls_back_to_largest_first_page_text(tmp_path: Path, config: AppConfig) -> None:
    document = load_pdf(pf.structured_pdf(tmp_path / "doc.pdf"), config.ingestion)
    assert document.title == "Deployment Handbook"


def test_repeating_headers_and_footers_are_stripped(tmp_path: Path, config: AppConfig) -> None:
    document = load_pdf(pf.paged_pdf_with_furniture(tmp_path / "doc.pdf"), config.ingestion)
    texts = [line.text for line in document.lines()]

    assert not any("Confidential" in t for t in texts)
    assert not any(t.startswith("Page ") and " of " in t for t in texts)
    assert any("Body content describing" in t for t in texts)


def test_repeated_body_text_away_from_the_margins_survives(
    tmp_path: Path, config: AppConfig
) -> None:
    # The body line is identical on all six pages. Repetition alone must not
    # strip it - only position in the margin makes something furniture.
    document = load_pdf(pf.paged_pdf_with_furniture(tmp_path / "doc.pdf"), config.ingestion)
    kept = [line.text for line in document.lines() if "service behaviour" in line.text]
    assert len(kept) == 6


def test_page_opening_lines_are_not_stripped_as_furniture(
    tmp_path: Path, config: AppConfig
) -> None:
    # A paragraph starting high on every page wraps into several short lines.
    # Only the topmost one is ever eligible, so the rest of the text survives.
    boilerplate = (
        "This notice is repeated verbatim on every page of the document because it "
        "carries a safety warning that the publisher is obliged to restate in full "
        "at the top of each and every printed page of the manual."
    )
    path = pf.make_pdf(tmp_path / "doc.pdf", [[pf.Text(boilerplate)] for _ in range(6)])

    document = load_pdf(path, config.ingestion)
    assert any("obliged to restate" in line.text for line in document.lines())


def test_tables_become_markdown_blocks(tmp_path: Path, config: AppConfig) -> None:
    document = load_pdf(pf.table_pdf(tmp_path / "doc.pdf"), config.ingestion)
    tables = [e for p in document.pages for e in p.elements if isinstance(e, TableBlock)]

    assert tables, "expected the bordered grid to be detected as a table"
    markdown = tables[0].markdown
    assert "|" in markdown
    assert "request_timeout" in markdown
    assert "30" in markdown


def test_table_text_is_not_duplicated_as_prose(tmp_path: Path, config: AppConfig) -> None:
    document = load_pdf(pf.table_pdf(tmp_path / "doc.pdf"), config.ingestion)
    prose = " ".join(e.text for p in document.pages for e in p.elements if isinstance(e, Line))
    assert "request_timeout" not in prose


def test_reading_order_is_top_to_bottom(tmp_path: Path, config: AppConfig) -> None:
    document = load_pdf(pf.structured_pdf(tmp_path / "doc.pdf"), config.ingestion)
    for page in document.pages:
        tops = [e.bbox[1] for e in page.elements]
        assert tops == sorted(tops)


def test_scanned_pdf_is_rejected(tmp_path: Path, config: AppConfig) -> None:
    with pytest.raises(ScannedPdfError):
        load_pdf(pf.scanned_pdf(tmp_path / "scan.pdf"), config.ingestion)


def test_corrupt_pdf_raises_parse_error(tmp_path: Path, config: AppConfig) -> None:
    with pytest.raises(PdfParseError):
        load_pdf(pf.corrupt_pdf(tmp_path / "broken.pdf"), config.ingestion)


def test_missing_file_raises_parse_error(tmp_path: Path, config: AppConfig) -> None:
    with pytest.raises(PdfParseError):
        load_pdf(tmp_path / "nope.pdf", config.ingestion)


def test_encrypted_pdf_raises(tmp_path: Path, config: AppConfig) -> None:
    import fitz

    source = pf.structured_pdf(tmp_path / "plain.pdf")
    encrypted = tmp_path / "locked.pdf"
    doc = fitz.open(source)
    doc.save(
        str(encrypted),
        encryption=fitz.PDF_ENCRYPT_AES_256,
        owner_pw="owner",
        user_pw="user",
    )
    doc.close()

    with pytest.raises(EncryptedPdfError):
        load_pdf(encrypted, config.ingestion)


def test_oversized_file_is_rejected(tmp_path: Path, config: AppConfig) -> None:
    from askmydocs.errors import FileTooLargeError

    path = pf.structured_pdf(tmp_path / "doc.pdf")
    config.ingestion.max_file_mb = 0.0001  # type: ignore[assignment]
    with pytest.raises(FileTooLargeError):
        load_pdf(path, config.ingestion)
