"""A deterministic stand-in for bge-base-en-v1.5.

Hashed bag-of-words into a small dense space: texts sharing vocabulary get
similar vectors, so retrieval tests exercise real ranking behaviour without
downloading a model or importing torch. Hashing uses md5 rather than Python's
``hash``, which is salted per process and would make results irreproducible.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence

import numpy as np

_TOKEN = re.compile(r"[a-z0-9]+")


class FakeEmbedder:
    """Implements the :class:`askmydocs.indexing.embedder.Embedder` protocol."""

    def __init__(self, dimension: int = 64, query_prefix: str = "") -> None:
        self._dimension = dimension
        self.query_prefix = query_prefix
        self.encoded_queries: list[str] = []
        self.encoded_document_batches: list[int] = []

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def model_name(self) -> str:
        return "fake-embedder"

    def _vector(self, text: str) -> np.ndarray:
        vector = np.zeros(self._dimension, dtype=np.float32)
        for token in _TOKEN.findall(text.lower()):
            digest = hashlib.md5(token.encode("utf-8")).hexdigest()
            vector[int(digest[:8], 16) % self._dimension] += 1.0
        norm = float(np.linalg.norm(vector))
        return vector / norm if norm else vector

    def encode_documents(self, texts: Sequence[str]) -> np.ndarray:
        self.encoded_document_batches.append(len(texts))
        if not texts:
            return np.zeros((0, self._dimension), dtype=np.float32)
        return np.vstack([self._vector(text) for text in texts]).astype(np.float32)

    def encode_query(self, text: str) -> np.ndarray:
        self.encoded_queries.append(text)
        return self._vector(f"{self.query_prefix}{text}")
