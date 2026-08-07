"""Uploading documents and (re)building the indexes."""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from typing import Annotated, Any

from fastapi import APIRouter, BackgroundTasks, File, HTTPException, UploadFile, status

from ...indexing import IndexBuilder
from ...ingestion import IngestionPipeline
from ...ingestion.pipeline import read_chunks
from ...logging_setup import get_logger
from ..deps import StateDep
from ..schemas import (
    DocumentListResponse,
    DocumentSummary,
    IngestRequest,
    JobResponse,
    UploadResponse,
)

log = get_logger(__name__)

router = APIRouter(tags=["documents"])

#: PDFs only. The parser handles nothing else, and accepting other types would
#: turn a clear rejection here into a confusing skip in the manifest later.
ALLOWED_SUFFIX = ".pdf"
MAX_UPLOAD_BYTES = 100 * 1024 * 1024


@router.post("/documents/upload", response_model=UploadResponse)
async def upload(
    state: StateDep, files: Annotated[list[UploadFile], File()]
) -> UploadResponse:
    """Store uploaded PDFs in the ingestion folder.

    Uploading does not index anything - call ``POST /ingest`` afterwards. Kept
    separate so several uploads can be batched into a single expensive
    embedding pass.
    """
    target = state.config.paths.raw_pdfs
    target.mkdir(parents=True, exist_ok=True)

    uploaded: list[str] = []
    rejected: list[str] = []

    for upload_file in files:
        name = safe_filename(upload_file.filename or "")
        if not name:
            rejected.append(upload_file.filename or "(unnamed)")
            continue

        payload = await upload_file.read()
        if not payload or len(payload) > MAX_UPLOAD_BYTES:
            rejected.append(name)
            log.warning("upload_rejected", file=name, bytes=len(payload))
            continue

        (target / name).write_bytes(payload)
        uploaded.append(name)
        log.info("document_uploaded", file=name, bytes=len(payload))

    detail = f"{len(uploaded)} stored in {target}"
    if rejected:
        detail += f"; {len(rejected)} rejected (must be a PDF under 100MB)"
    return UploadResponse(uploaded=uploaded, rejected=rejected, detail=detail)


@router.post("/ingest", response_model=JobResponse, status_code=status.HTTP_202_ACCEPTED)
def ingest(
    payload: IngestRequest, state: StateDep, background: BackgroundTasks
) -> JobResponse:
    """Start ingestion, and optionally rebuild the indexes afterwards.

    Returns 202 with a job id: embedding a real corpus takes minutes, well past
    any sensible HTTP timeout. Poll ``GET /jobs/{id}`` for the outcome.
    """
    if state.jobs.is_busy():
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="an ingestion job is already running",
        )

    job = state.jobs.create(
        kind="ingest",
        detail=f"force={payload.force}, rebuild_index={payload.rebuild_index}",
    )
    background.add_task(
        state.jobs.run, job, lambda: _run_ingest(state, payload), exclusive=True
    )
    return JobResponse(**job.as_dict())


@router.get("/documents", response_model=DocumentListResponse)
def list_documents(state: StateDep) -> DocumentListResponse:
    """What is currently ingested, including anything that was skipped and why."""
    manifest = _read_manifest(state)
    if manifest is None:
        return DocumentListResponse()

    return DocumentListResponse(
        documents=[
            DocumentSummary(
                source_file=record.source_file,
                title=record.title,
                status=record.status.value,
                page_count=record.page_count,
                chunk_count=record.chunk_count,
                structure_source=(
                    record.structure_source.value if record.structure_source else None
                ),
                reason=record.reason,
            )
            for record in manifest.documents
        ],
        total_chunks=manifest.total_chunks,
        ingested_at=manifest.created_at,
    )


def _run_ingest(state: StateDep, payload: IngestRequest) -> dict[str, Any]:
    """The work behind the ingest job. Runs in a background thread."""
    manifest = IngestionPipeline(state.config).run(force=payload.force)
    result: dict[str, Any] = dict(manifest.summary())

    if payload.rebuild_index and manifest.total_chunks:
        index_manifest = IndexBuilder(state.config, embedder=state.embedder).build()
        result["indexed_chunks"] = index_manifest.chunk_count
        # Swapping the pipeline reference is atomic, so in-flight queries finish
        # against the old indexes rather than seeing a half-built one.
        state.load_indexes()
        result["index_reloaded"] = state.ready
    elif payload.rebuild_index:
        result["indexed_chunks"] = 0
        result["note"] = "nothing to index - no chunks were produced"

    return result


def _read_manifest(state: StateDep):  # type: ignore[no-untyped-def]
    from ...models import IngestionManifest

    path = state.config.paths.manifest_file
    if not path.is_file():
        return None
    try:
        return IngestionManifest.model_validate_json(path.read_text(encoding="utf-8"))
    except Exception as exc:
        log.warning("manifest_unreadable", path=str(path), error=str(exc))
        return None


def safe_filename(name: str) -> str:
    """Reduce an uploaded filename to something safe to write.

    Uploads are attacker-controlled input even on an internal tool: without
    this, a name like ``../../config/default.yaml`` would escape the upload
    directory entirely.
    """
    name = unicodedata.normalize("NFKD", name).strip()
    name = Path(name).name  # discards any directory component, including ..
    if not name.lower().endswith(ALLOWED_SUFFIX):
        return ""
    stem = re.sub(r"[^A-Za-z0-9._ -]+", "_", name[: -len(ALLOWED_SUFFIX)]).strip()
    stem = stem.strip(". ")
    return f"{stem[:100]}{ALLOWED_SUFFIX}" if stem else ""


def chunk_count(state: StateDep) -> int:
    return len(read_chunks(state.config.paths.chunks_file))
