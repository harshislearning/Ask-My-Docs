"""Index construction: chunks.jsonl -> FAISS + BM25 on disk.

Embedding is the expensive step (seconds per thousand chunks on CPU), so it
happens once here rather than at query time. Both indexes are written together
with a manifest recording exactly which chunks they were built from.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from ..config import AppConfig
from ..errors import IndexingError, IndexNotFoundError
from ..ingestion.pipeline import read_chunks
from ..logging_setup import bind_run, clear_run, get_logger, new_run_id
from ..models import Chunk, IndexManifest
from .bm25_store import Bm25Store
from .embedder import Embedder, SentenceTransformerEmbedder
from .faiss_store import FaissStore

log = get_logger(__name__)

MANIFEST_FILENAME = "index_manifest.json"


@dataclass(slots=True)
class IndexBundle:
    """Everything the retriever needs, loaded together and known consistent."""

    chunks: list[Chunk]
    chunks_by_id: dict[str, Chunk]
    faiss: FaissStore
    bm25: Bm25Store
    manifest: IndexManifest | None = None


class IndexBuilder:
    def __init__(self, config: AppConfig, embedder: Embedder | None = None) -> None:
        self.config = config
        self.embedder = embedder or SentenceTransformerEmbedder(config.embedding)

    def build(self, run_id: str | None = None) -> IndexManifest:
        run_id = run_id or new_run_id()
        chunks_path = self.config.paths.chunks_file
        directory = self.config.paths.indexes

        bind_run(run_id=run_id, stage="indexing")
        try:
            chunks = read_chunks(chunks_path)
            if not chunks:
                raise IndexingError(
                    f"no chunks found in {chunks_path} - run scripts/ingest.py first"
                )

            log.info(
                "index_build_started",
                chunks=len(chunks),
                documents=len({c.doc_id for c in chunks}),
                embedding_model=self.embedder.model_name,
            )

            embeddings = self.embedder.encode_documents([c.embed_text for c in chunks])
            if embeddings.shape[0] != len(chunks):
                raise IndexingError(
                    f"embedder returned {embeddings.shape[0]} vectors for {len(chunks)} chunks"
                )

            faiss_store = FaissStore.build(embeddings, [c.chunk_id for c in chunks])
            bm25_store = Bm25Store.build(chunks, self.config.retrieval)

            directory.mkdir(parents=True, exist_ok=True)
            faiss_store.save(directory)
            bm25_store.save(directory)

            manifest = IndexManifest(
                run_id=run_id,
                created_at=datetime.now(UTC),
                embedding_model=self.embedder.model_name,
                dimension=faiss_store.dimension,
                chunk_count=len(chunks),
                document_count=len({c.doc_id for c in chunks}),
                chunks_file=str(chunks_path),
                chunks_fingerprint=fingerprint_file(chunks_path),
                bm25_k1=self.config.retrieval.bm25_k1,
                bm25_b=self.config.retrieval.bm25_b,
            )
            (directory / MANIFEST_FILENAME).write_text(
                manifest.model_dump_json(indent=2), encoding="utf-8"
            )

            log.info(
                "index_build_finished",
                chunks=manifest.chunk_count,
                documents=manifest.document_count,
                dim=manifest.dimension,
                directory=str(directory),
            )
            return manifest
        finally:
            clear_run()


def load_index_bundle(config: AppConfig) -> IndexBundle:
    """Load both indexes plus the chunks they refer to.

    Warns loudly when the indexes were built from a different chunks.jsonl than
    the one currently on disk - stale indexes produce answers that look fine
    and cite the wrong thing.
    """
    directory = config.paths.indexes
    chunks_path = config.paths.chunks_file

    chunks = read_chunks(chunks_path)
    if not chunks:
        raise IndexNotFoundError(f"no chunks found in {chunks_path} - run ingestion first")

    manifest = read_index_manifest(directory)
    if manifest is not None:
        current = fingerprint_file(chunks_path)
        if current != manifest.chunks_fingerprint:
            log.warning(
                "index_out_of_date",
                indexed_chunks=manifest.chunk_count,
                current_chunks=len(chunks),
                action="rebuild with scripts/build_index.py",
            )

    faiss_store = FaissStore.load(directory)
    bm25_store = Bm25Store.load(directory, chunks, config.retrieval)

    known = {c.chunk_id for c in chunks}
    missing = [cid for cid in faiss_store.chunk_ids if cid not in known]
    if missing:
        log.warning(
            "index_references_unknown_chunks", count=len(missing), example=missing[0]
        )

    return IndexBundle(
        chunks=chunks,
        chunks_by_id={c.chunk_id: c for c in chunks},
        faiss=faiss_store,
        bm25=bm25_store,
        manifest=manifest,
    )


def read_index_manifest(directory: Path) -> IndexManifest | None:
    path = directory / MANIFEST_FILENAME
    if not path.is_file():
        return None
    try:
        return IndexManifest.model_validate_json(path.read_text(encoding="utf-8"))
    except Exception as exc:
        log.warning("index_manifest_unreadable", path=str(path), error=str(exc))
        return None


def fingerprint_file(path: Path) -> str:
    """Content hash used to detect that chunks changed after indexing."""
    if not path.is_file():
        return ""
    digest = hashlib.sha1()
    with path.open("rb") as handle:
        while block := handle.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()[:16]


def index_summary(directory: Path) -> dict[str, object]:
    """Small dict for the API's health endpoint (Phase 6)."""
    manifest = read_index_manifest(directory)
    if manifest is None:
        return {"built": False}
    return {
        "built": True,
        **json.loads(manifest.model_dump_json()),
    }
