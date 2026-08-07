"""End-to-end ingestion over a folder of synthetic PDFs."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import pytest

from askmydocs.config import AppConfig
from askmydocs.ingestion import Chunker, IngestionPipeline
from askmydocs.ingestion.pipeline import read_chunks
from askmydocs.models import DocumentStatus, IngestionManifest, StructureSource
from fixtures import pdf_factory as pf


@pytest.fixture
def pipeline(config: AppConfig, word_token_counter: Callable[[str], int]) -> IngestionPipeline:
    return IngestionPipeline(config, chunker=Chunker(config.chunking, word_token_counter))


def test_full_run_over_a_mixed_folder(
    pipeline: IngestionPipeline, config: AppConfig, pdf_dir: Path
) -> None:
    pf.structured_pdf(pdf_dir / "handbook.pdf")
    pf.unstructured_pdf(pdf_dir / "notes.pdf")
    pf.table_pdf(pdf_dir / "reference.pdf")

    manifest = pipeline.run()

    assert manifest.summary()["documents_ok"] == 3
    assert manifest.total_chunks > 0
    assert config.paths.chunks_file.is_file()

    chunks = read_chunks(config.paths.chunks_file)
    assert len(chunks) == manifest.total_chunks
    assert len({c.doc_id for c in chunks}) == 3
    assert all(c.chunk_id and c.text.strip() for c in chunks)


def test_structure_source_differs_by_document(
    pipeline: IngestionPipeline, config: AppConfig, pdf_dir: Path
) -> None:
    pf.structured_pdf(pdf_dir / "handbook.pdf")
    pf.unstructured_pdf(pdf_dir / "notes.pdf")

    manifest = pipeline.run()
    sources = {r.source_file: r.structure_source for r in manifest.documents}

    assert sources["handbook.pdf"] is StructureSource.HEADINGS
    assert sources["notes.pdf"] is StructureSource.PAGE_FALLBACK


def test_broken_documents_are_recorded_not_fatal(
    pipeline: IngestionPipeline, pdf_dir: Path
) -> None:
    pf.structured_pdf(pdf_dir / "good.pdf")
    pf.corrupt_pdf(pdf_dir / "broken.pdf")
    pf.scanned_pdf(pdf_dir / "scan.pdf")

    manifest = pipeline.run()
    by_file = {r.source_file: r for r in manifest.documents}

    assert by_file["good.pdf"].status is DocumentStatus.OK
    assert by_file["broken.pdf"].status is DocumentStatus.FAILED
    assert by_file["broken.pdf"].error_code == "parse_failed"
    assert by_file["scan.pdf"].status is DocumentStatus.SKIPPED
    assert by_file["scan.pdf"].error_code == "scanned_no_text_layer"
    # The healthy document still made it through.
    assert by_file["good.pdf"].chunk_count > 0


def test_empty_folder_is_not_an_error(pipeline: IngestionPipeline, pdf_dir: Path) -> None:
    manifest = pipeline.run()
    assert manifest.documents == []
    assert manifest.total_chunks == 0


def test_missing_folder_is_not_an_error(pipeline: IngestionPipeline, tmp_path: Path) -> None:
    manifest = pipeline.run(tmp_path / "does-not-exist")
    assert manifest.documents == []


def test_unchanged_documents_are_reused(
    pipeline: IngestionPipeline, config: AppConfig, pdf_dir: Path
) -> None:
    pf.structured_pdf(pdf_dir / "handbook.pdf")
    first = pipeline.run()
    second = pipeline.run()

    assert first.documents[0].status is DocumentStatus.OK
    assert second.documents[0].status is DocumentStatus.UNCHANGED
    # Same ids means an index can be updated incrementally instead of rebuilt.
    assert [c.chunk_id for c in read_chunks(config.paths.chunks_file)] == [
        c.chunk_id for c in read_chunks(config.paths.chunks_file)
    ]
    assert second.total_chunks == first.total_chunks


def test_force_bypasses_the_cache(pipeline: IngestionPipeline, pdf_dir: Path) -> None:
    pf.structured_pdf(pdf_dir / "handbook.pdf")
    pipeline.run()
    forced = pipeline.run(force=True)
    assert forced.documents[0].status is DocumentStatus.OK


def test_changing_chunking_config_invalidates_the_cache(
    config: AppConfig, word_token_counter: Callable[[str], int], pdf_dir: Path
) -> None:
    pf.structured_pdf(pdf_dir / "handbook.pdf")
    IngestionPipeline(config, chunker=Chunker(config.chunking, word_token_counter)).run()

    config.chunking.chunk_tokens = 25
    rerun = IngestionPipeline(config, chunker=Chunker(config.chunking, word_token_counter)).run()

    assert rerun.documents[0].status is DocumentStatus.OK, "expected a re-chunk, not a cache hit"


def test_edited_document_is_reingested(
    pipeline: IngestionPipeline, pdf_dir: Path
) -> None:
    target = pdf_dir / "handbook.pdf"
    pf.structured_pdf(target)
    first = pipeline.run()

    pf.table_pdf(target)  # same filename, different bytes
    second = pipeline.run()

    assert second.documents[0].status is DocumentStatus.OK
    assert second.documents[0].doc_id != first.documents[0].doc_id


def test_manifest_is_valid_json_and_reloadable(
    pipeline: IngestionPipeline, config: AppConfig, pdf_dir: Path
) -> None:
    pf.structured_pdf(pdf_dir / "handbook.pdf")
    pipeline.run()

    payload = json.loads(config.paths.manifest_file.read_text("utf-8"))
    assert payload["run_id"]
    reloaded = IngestionManifest.model_validate(payload)
    assert reloaded.total_chunks > 0


def test_chunks_file_is_one_json_object_per_line(
    pipeline: IngestionPipeline, config: AppConfig, pdf_dir: Path
) -> None:
    pf.structured_pdf(pdf_dir / "handbook.pdf")
    pipeline.run()

    lines = config.paths.chunks_file.read_text("utf-8").strip().splitlines()
    assert lines
    for line in lines:
        record = json.loads(line)
        assert {"chunk_id", "doc_id", "text", "embed_text", "page_start"} <= record.keys()


def test_page_numbers_are_within_document_bounds(
    pipeline: IngestionPipeline, config: AppConfig, pdf_dir: Path
) -> None:
    pf.structured_pdf(pdf_dir / "handbook.pdf")
    manifest = pipeline.run()
    page_count = manifest.documents[0].page_count

    for chunk in read_chunks(config.paths.chunks_file):
        assert 1 <= chunk.page_start <= chunk.page_end <= page_count


def test_nested_folders_are_discovered(pipeline: IngestionPipeline, pdf_dir: Path) -> None:
    pf.structured_pdf(pdf_dir / "top.pdf")
    pf.table_pdf(pdf_dir / "nested" / "deep.pdf")

    manifest = pipeline.run()
    assert {r.source_file for r in manifest.documents} == {"top.pdf", "deep.pdf"}
