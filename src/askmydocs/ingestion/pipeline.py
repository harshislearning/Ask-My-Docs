"""Ingestion orchestration: a folder of PDFs in, ``chunks.jsonl`` out.

Two properties matter more than anything else here:

* **One bad PDF never fails the run.** Every document is isolated; failures are
  recorded in the manifest with a reason so you can see what was skipped
  instead of wondering why an answer is missing.
* **Re-ingesting is cheap and idempotent.** Documents are identified by content
  hash, so unchanged files keep their existing chunks (and therefore their
  chunk ids, so indexes can be updated incrementally). Changing a chunking
  parameter invalidates that cache automatically.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

from ..config import AppConfig
from ..errors import (
    DocumentError,
    EmptyDocumentError,
    EncryptedPdfError,
    FileTooLargeError,
    ScannedPdfError,
)
from ..logging_setup import bind_run, clear_run, get_logger, new_run_id
from ..models import (
    Chunk,
    DocumentRecord,
    DocumentStatus,
    IngestionManifest,
    ParsedDocument,
)
from .chunker import Chunker
from .pdf_loader import compute_doc_id, load_pdf
from .structure import build_sections

log = get_logger(__name__)

#: These mean "we understood the file and cannot use it" - not a bug.
_SKIP_ERRORS = (ScannedPdfError, EncryptedPdfError, FileTooLargeError, EmptyDocumentError)


class IngestionPipeline:
    def __init__(self, config: AppConfig, chunker: Chunker | None = None) -> None:
        self.config = config
        self.chunker = chunker or Chunker(config.chunking)

    def run(
        self,
        input_dir: Path | None = None,
        *,
        force: bool = False,
        run_id: str | None = None,
    ) -> IngestionManifest:
        """Ingest every PDF under ``input_dir`` and write chunks + manifest."""
        run_id = run_id or new_run_id()
        source_dir = Path(input_dir) if input_dir else self.config.paths.raw_pdfs
        fingerprint = self.config_fingerprint()

        bind_run(run_id=run_id, stage="ingestion")
        try:
            pdf_files = self.discover(source_dir)
            log.info(
                "ingestion_started",
                source_dir=str(source_dir),
                files=len(pdf_files),
                force=force,
                config_fingerprint=fingerprint,
            )
            if not pdf_files:
                log.warning("no_pdfs_found", source_dir=str(source_dir))

            cached = {} if force else self._load_cached_chunks(fingerprint)

            records: list[DocumentRecord] = []
            all_chunks: list[Chunk] = []

            for path in pdf_files:
                record, chunks = self._ingest_one(path, cached)
                records.append(record)
                all_chunks.extend(chunks)

            manifest = IngestionManifest(
                run_id=run_id,
                created_at=datetime.now(UTC),
                config_fingerprint=fingerprint,
                chunks_file=str(self.config.paths.chunks_file),
                documents=records,
            )
            self._persist(all_chunks, manifest)
            log.info("ingestion_finished", **manifest.summary())
            return manifest
        finally:
            clear_run()

    # -- discovery -------------------------------------------------------

    @staticmethod
    def discover(source_dir: Path) -> list[Path]:
        """All PDFs under ``source_dir``, recursively, in stable order."""
        if not source_dir.is_dir():
            log.warning("source_dir_missing", source_dir=str(source_dir))
            return []
        return sorted(
            (p for p in source_dir.rglob("*") if p.is_file() and p.suffix.lower() == ".pdf"),
            key=lambda p: str(p).lower(),
        )

    def config_fingerprint(self) -> str:
        """Hash of every setting that affects chunk content.

        When it changes, cached chunks are invalid and everything is re-chunked.
        """
        payload = {
            "ingestion": self.config.ingestion.model_dump(),
            "structure": self.config.structure.model_dump(),
            "chunking": self.config.chunking.model_dump(),
        }
        blob = json.dumps(payload, sort_keys=True, default=str)
        return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:16]

    # -- per-document ----------------------------------------------------

    def _ingest_one(
        self, path: Path, cached: dict[str, list[Chunk]]
    ) -> tuple[DocumentRecord, list[Chunk]]:
        try:
            doc_id = compute_doc_id(path)
        except OSError as exc:
            log.error("doc_id_failed", file=str(path), error=str(exc))
            return (
                DocumentRecord(
                    doc_id="",
                    source_file=path.name,
                    title=path.stem,
                    status=DocumentStatus.FAILED,
                    reason=f"could not read file: {exc}",
                    error_code="unreadable",
                ),
                [],
            )

        if doc_id in cached:
            chunks = cached[doc_id]
            log.info("document_unchanged", doc_id=doc_id, file=path.name, chunks=len(chunks))
            return (
                DocumentRecord(
                    doc_id=doc_id,
                    source_file=path.name,
                    title=chunks[0].doc_title if chunks else path.stem,
                    status=DocumentStatus.UNCHANGED,
                    chunk_count=len(chunks),
                    structure_source=chunks[0].structure_source if chunks else None,
                ),
                chunks,
            )

        try:
            document = load_pdf(path, self.config.ingestion)
        except DocumentError as exc:
            status = (
                DocumentStatus.SKIPPED
                if isinstance(exc, _SKIP_ERRORS)
                else DocumentStatus.FAILED
            )
            log.warning(
                "document_not_ingested",
                file=path.name,
                status=status.value,
                error_code=exc.code,
                reason=str(exc),
            )
            return (
                DocumentRecord(
                    doc_id=doc_id,
                    source_file=path.name,
                    title=path.stem,
                    status=status,
                    reason=str(exc),
                    error_code=exc.code,
                ),
                [],
            )
        except Exception as exc:
            log.exception("document_unexpected_error", file=path.name)
            return (
                DocumentRecord(
                    doc_id=doc_id,
                    source_file=path.name,
                    title=path.stem,
                    status=DocumentStatus.FAILED,
                    reason=f"unexpected error: {exc}",
                    error_code="unexpected",
                ),
                [],
            )

        return self._chunk_document(document, path)

    def _chunk_document(
        self, document: ParsedDocument, path: Path
    ) -> tuple[DocumentRecord, list[Chunk]]:
        sections, structure_source, heading_count = build_sections(document, self.config.structure)
        chunks = self.chunker.chunk_document(document, sections, structure_source)

        if not chunks:
            log.warning("document_produced_no_chunks", doc_id=document.doc_id, file=path.name)
            return (
                DocumentRecord(
                    doc_id=document.doc_id,
                    source_file=path.name,
                    title=document.title,
                    status=DocumentStatus.SKIPPED,
                    page_count=document.page_count,
                    heading_count=heading_count,
                    structure_source=structure_source,
                    reason="no chunks produced",
                    error_code="no_chunks",
                ),
                [],
            )

        return (
            DocumentRecord(
                doc_id=document.doc_id,
                source_file=path.name,
                title=document.title,
                status=DocumentStatus.OK,
                page_count=document.page_count,
                chunk_count=len(chunks),
                heading_count=heading_count,
                structure_source=structure_source,
                table_count=sum(1 for c in chunks if c.content_type == "table"),
            ),
            chunks,
        )

    # -- persistence -----------------------------------------------------

    def _load_cached_chunks(self, fingerprint: str) -> dict[str, list[Chunk]]:
        manifest_path = self.config.paths.manifest_file
        chunks_path = self.config.paths.chunks_file
        if not manifest_path.is_file() or not chunks_path.is_file():
            return {}

        try:
            previous = IngestionManifest.model_validate_json(manifest_path.read_text("utf-8"))
        except Exception as exc:
            log.warning("manifest_unreadable", path=str(manifest_path), error=str(exc))
            return {}

        if previous.config_fingerprint != fingerprint:
            log.info(
                "cache_invalidated",
                reason="chunking config changed",
                previous=previous.config_fingerprint,
                current=fingerprint,
            )
            return {}

        grouped: dict[str, list[Chunk]] = defaultdict(list)
        for chunk in read_chunks(chunks_path):
            grouped[chunk.doc_id].append(chunk)
        log.debug("cache_loaded", documents=len(grouped))
        return dict(grouped)

    def _persist(self, chunks: list[Chunk], manifest: IngestionManifest) -> None:
        chunks_path = self.config.paths.chunks_file
        manifest_path = self.config.paths.manifest_file
        chunks_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)

        write_chunks(chunks_path, chunks)
        _atomic_write(manifest_path, manifest.model_dump_json(indent=2))
        log.info(
            "artifacts_written", chunks_file=str(chunks_path), manifest_file=str(manifest_path)
        )


# --------------------------------------------------------------------------
# chunks.jsonl - the boundary between ingestion and indexing
# --------------------------------------------------------------------------


def write_chunks(path: Path, chunks: list[Chunk]) -> None:
    """Write chunks as JSON Lines, atomically."""
    payload = "\n".join(chunk.model_dump_json() for chunk in chunks)
    _atomic_write(path, payload + "\n" if payload else "")


def read_chunks(path: Path) -> list[Chunk]:
    """Read chunks.jsonl. Malformed lines are logged and skipped, not fatal."""
    if not Path(path).is_file():
        return []
    chunks: list[Chunk] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                chunks.append(Chunk.model_validate_json(line))
            except Exception as exc:
                log.warning("chunk_line_invalid", path=str(path), line=line_no, error=str(exc))
    return chunks


def _atomic_write(path: Path, content: str) -> None:
    """Write via a temp file + rename so readers never see a half-written file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(content)
        os.replace(tmp_name, path)
    except Exception:
        Path(tmp_name).unlink(missing_ok=True)
        raise
