"""Embedding models.

Everything downstream depends on the :class:`Embedder` protocol rather than on
sentence-transformers, so indexes and retrievers can be tested with a fake that
needs no model download and no torch.

bge models have two quirks that are easy to get wrong and expensive to debug:

* Queries take an instruction prefix; documents do **not**. Prefixing documents
  degrades retrieval measurably.
* Vectors must be L2-normalised so inner product equals cosine similarity -
  FAISS ``IndexFlatIP`` does no normalisation of its own.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol, runtime_checkable

import numpy as np

from ..config import EmbeddingConfig
from ..logging_setup import get_logger

log = get_logger(__name__)


@runtime_checkable
class Embedder(Protocol):
    """What the indexing and retrieval layers need from an embedding model."""

    @property
    def dimension(self) -> int: ...

    @property
    def model_name(self) -> str: ...

    def encode_documents(self, texts: Sequence[str]) -> np.ndarray:
        """Return an (n, dim) float32 array of unit vectors."""
        ...

    def encode_query(self, text: str) -> np.ndarray:
        """Return a (dim,) float32 unit vector."""
        ...


class SentenceTransformerEmbedder:
    """bge-base-en-v1.5 via sentence-transformers, run locally."""

    def __init__(self, config: EmbeddingConfig) -> None:
        self.config = config
        self._model: Any = None
        self._dimension: int | None = None

    # sentence-transformers pulls in torch (~GBs) and takes seconds to import,
    # so it is loaded on first use rather than at module import.
    def _load(self) -> Any:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            device = None if self.config.device == "auto" else self.config.device
            log.info("loading_embedding_model", model=self.config.model, device=device or "auto")
            self._model = SentenceTransformer(self.config.model, device=device)
            self._dimension = _embedding_dimension(self._model)
            log.info("embedding_model_loaded", model=self.config.model, dim=self._dimension)
        return self._model

    @property
    def dimension(self) -> int:
        if self._dimension is None:
            self._load()
        assert self._dimension is not None
        return self._dimension

    @property
    def model_name(self) -> str:
        return self.config.model

    def encode_documents(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)
        model = self._load()
        vectors = model.encode(
            list(texts),
            batch_size=self.config.batch_size,
            normalize_embeddings=self.config.normalize,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype=np.float32)

    def encode_query(self, text: str) -> np.ndarray:
        model = self._load()
        # The instruction prefix belongs on queries only.
        prompt = f"{self.config.query_prefix}{text}" if self.config.query_prefix else text
        vector = model.encode(
            [prompt],
            batch_size=1,
            normalize_embeddings=self.config.normalize,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return np.asarray(vector, dtype=np.float32)[0]


def _embedding_dimension(model: Any) -> int:
    """Read the output dimension across sentence-transformers versions.

    v5 renamed ``get_sentence_embedding_dimension`` to ``get_embedding_dimension``
    and deprecated the old name.
    """
    for attribute in ("get_embedding_dimension", "get_sentence_embedding_dimension"):
        getter = getattr(model, attribute, None)
        if callable(getter):
            return int(getter())
    raise AttributeError("embedding model exposes no dimension accessor")


def as_float32_matrix(vectors: np.ndarray) -> np.ndarray:
    """Coerce to the contiguous float32 layout FAISS requires."""
    matrix = np.ascontiguousarray(np.asarray(vectors, dtype=np.float32))
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    return matrix


def l2_normalise(vectors: np.ndarray) -> np.ndarray:
    """Scale rows to unit length so inner product == cosine similarity.

    Applied defensively even when the model already normalises: a silently
    un-normalised index produces plausible-looking but subtly wrong rankings.
    """
    matrix = as_float32_matrix(vectors)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0  # a zero vector stays zero rather than becoming NaN
    return matrix / norms
