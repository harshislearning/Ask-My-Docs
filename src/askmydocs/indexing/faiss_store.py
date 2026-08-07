"""FAISS vector index, persisted to disk.

``IndexFlatIP`` over L2-normalised vectors gives *exact* cosine similarity.
Approximate indexes (IVF, HNSW) only start paying off in the hundreds of
thousands of vectors; below that they add tuning parameters, a training step,
and recall loss for no measurable latency win. An internal knowledge base is
comfortably in exact-search territory, and exact search means retrieval
evaluation measures the retriever rather than the index's approximation error.

FAISS addresses vectors by position, so the id mapping is stored alongside the
index and the two are always written and loaded together.
"""

from __future__ import annotations

import json
from pathlib import Path

import faiss
import numpy as np

from ..errors import IndexingError, IndexNotFoundError
from ..logging_setup import get_logger
from ..models import Hit
from .embedder import l2_normalise

log = get_logger(__name__)

INDEX_FILENAME = "faiss.index"
IDS_FILENAME = "faiss_ids.json"


class FaissStore:
    """Dense index over chunk embeddings."""

    def __init__(self, index: faiss.Index, chunk_ids: list[str]) -> None:
        if index.ntotal != len(chunk_ids):
            raise IndexingError(
                f"index has {index.ntotal} vectors but {len(chunk_ids)} ids were supplied"
            )
        self.index = index
        self.chunk_ids = chunk_ids

    def __len__(self) -> int:
        return len(self.chunk_ids)

    @property
    def dimension(self) -> int:
        return int(self.index.d)

    # -- construction ----------------------------------------------------

    @classmethod
    def build(cls, embeddings: np.ndarray, chunk_ids: list[str]) -> FaissStore:
        matrix = l2_normalise(embeddings)
        if matrix.shape[0] != len(chunk_ids):
            raise IndexingError(f"{matrix.shape[0]} embeddings for {len(chunk_ids)} chunks")

        dimension = int(matrix.shape[1]) if matrix.ndim == 2 else 0
        if dimension == 0:
            raise IndexingError("cannot build an index with zero-dimensional vectors")

        index = faiss.IndexFlatIP(dimension)
        if len(chunk_ids):
            index.add(matrix)

        log.info("faiss_index_built", vectors=index.ntotal, dim=dimension)
        return cls(index, list(chunk_ids))

    # -- search ----------------------------------------------------------

    def search(self, query_vector: np.ndarray, top_n: int) -> list[Hit]:
        """Nearest neighbours by cosine similarity, best first."""
        if not len(self.chunk_ids) or top_n <= 0:
            return []

        query = l2_normalise(query_vector)
        similarities, positions = self.index.search(query, min(top_n, len(self.chunk_ids)))

        hits: list[Hit] = []
        for rank, (position, score) in enumerate(
            zip(positions[0], similarities[0], strict=True), start=1
        ):
            if position < 0:  # FAISS pads with -1 when fewer results exist
                continue
            hits.append(
                Hit(chunk_id=self.chunk_ids[int(position)], score=float(score), rank=rank)
            )
        return hits

    # -- persistence -----------------------------------------------------

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(directory / INDEX_FILENAME))
        (directory / IDS_FILENAME).write_text(
            json.dumps(self.chunk_ids, indent=0), encoding="utf-8"
        )
        log.info("faiss_index_saved", directory=str(directory), vectors=len(self.chunk_ids))

    @classmethod
    def load(cls, directory: Path) -> FaissStore:
        index_path = directory / INDEX_FILENAME
        ids_path = directory / IDS_FILENAME
        if not index_path.is_file() or not ids_path.is_file():
            raise IndexNotFoundError(
                f"no FAISS index in {directory} - run scripts/build_index.py first"
            )
        index = faiss.read_index(str(index_path))
        chunk_ids = json.loads(ids_path.read_text(encoding="utf-8"))
        log.info("faiss_index_loaded", directory=str(directory), vectors=len(chunk_ids))
        return cls(index, chunk_ids)
