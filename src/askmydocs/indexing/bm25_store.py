"""BM25 keyword index.

This is the half of retrieval that catches what embeddings miss: exact
identifiers, error codes, flag names, version strings. A dense model maps
``request_timeout`` and ``connection_deadline`` to nearby vectors, which is
useful for paraphrase and actively harmful when the user asked about one
specific parameter.

Tokenisation is therefore identifier-aware. ``request_timeout`` is indexed both
whole *and* as its parts, so a query for the exact name scores highest while a
query for "timeout" still matches.

The index is persisted, with a rebuild-from-chunks fallback: pickled objects
are fragile across library versions, and a stale pickle must degrade into a
slower startup rather than a crash.
"""

from __future__ import annotations

import pickle
import re
import time
from collections.abc import Sequence
from pathlib import Path

from rank_bm25 import BM25Okapi

from ..config import RetrievalConfig
from ..errors import IndexNotFoundError
from ..logging_setup import get_logger
from ..models import Chunk, Hit

log = get_logger(__name__)

BM25_FILENAME = "bm25.pkl"
CORPUS_FILENAME = "bm25_ids.json"

#: Keep dots, underscores and hyphens: they are part of the token in
#: `request_timeout`, `v2.1.0`, `x-request-id`.
_TOKEN = re.compile(r"[a-z0-9]+(?:[._-][a-z0-9]+)*")
_SPLIT_COMPOUND = re.compile(r"[._-]")


def tokenize(text: str) -> list[str]:
    """Lowercase, identifier-aware tokenisation.

    Compound identifiers are emitted whole and split, so both an exact query
    (``request_timeout``) and a partial one (``timeout``) hit the same chunk.
    """
    tokens: list[str] = []
    for match in _TOKEN.finditer(text.lower()):
        token = match.group(0)
        tokens.append(token)
        parts = [p for p in _SPLIT_COMPOUND.split(token) if p]
        if len(parts) > 1:
            tokens.extend(parts)
    return tokens


class Bm25Store:
    """Keyword index over chunk text."""

    def __init__(self, bm25: BM25Okapi, chunk_ids: list[str]) -> None:
        self.bm25 = bm25
        self.chunk_ids = chunk_ids

    def __len__(self) -> int:
        return len(self.chunk_ids)

    # -- construction ----------------------------------------------------

    @classmethod
    def build(cls, chunks: Sequence[Chunk], config: RetrievalConfig) -> Bm25Store:
        if not chunks:
            # BM25Okapi divides by the average document length and cannot be
            # built from nothing; an empty store that returns no hits is the
            # honest representation of an empty corpus.
            return cls(_EmptyBm25(), [])

        started = time.perf_counter()
        # embed_text, not text: the breadcrumb is part of what makes a chunk
        # findable, and both retrievers must see the same content.
        corpus = [tokenize(chunk.embed_text) for chunk in chunks]
        bm25 = BM25Okapi(corpus, k1=config.bm25_k1, b=config.bm25_b)
        elapsed = time.perf_counter() - started

        log.info(
            "bm25_index_built",
            documents=len(corpus),
            tokens=sum(len(doc) for doc in corpus),
            seconds=round(elapsed, 3),
        )
        return cls(bm25, [chunk.chunk_id for chunk in chunks])

    # -- search ----------------------------------------------------------

    def search(self, query: str, top_n: int) -> list[Hit]:
        """Best-matching chunks for a keyword query, best first."""
        if not self.chunk_ids or top_n <= 0:
            return []

        tokens = tokenize(query)
        if not tokens:
            log.debug("bm25_query_had_no_tokens", query=query)
            return []

        scores = self.bm25.get_scores(tokens)
        ordered = sorted(range(len(scores)), key=lambda i: (-scores[i], i))

        hits: list[Hit] = []
        for rank, position in enumerate(ordered[:top_n], start=1):
            score = float(scores[position])
            # BM25 gives 0 to documents sharing no query term. Returning them
            # would pad the candidate list with noise that RRF then rewards
            # purely for existing.
            if score <= 0.0:
                break
            hits.append(Hit(chunk_id=self.chunk_ids[position], score=score, rank=rank))
        return hits

    # -- persistence -----------------------------------------------------

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        with (directory / BM25_FILENAME).open("wb") as handle:
            pickle.dump({"bm25": self.bm25, "chunk_ids": self.chunk_ids}, handle)
        log.info("bm25_index_saved", directory=str(directory), documents=len(self.chunk_ids))

    @classmethod
    def load(
        cls, directory: Path, chunks: Sequence[Chunk], config: RetrievalConfig
    ) -> Bm25Store:
        """Load the index, rebuilding from ``chunks`` if the artifact is
        missing or unreadable."""
        path = directory / BM25_FILENAME
        if not path.is_file():
            raise IndexNotFoundError(
                f"no BM25 index in {directory} - run scripts/build_index.py first"
            )
        try:
            with path.open("rb") as handle:
                payload = pickle.load(handle)
            store = cls(payload["bm25"], list(payload["chunk_ids"]))
        except Exception as exc:
            log.warning(
                "bm25_index_unreadable",
                path=str(path),
                error=str(exc),
                action="rebuilding from chunks",
            )
            return cls.build(chunks, config)

        if len(store) != len(chunks):
            log.warning(
                "bm25_index_stale",
                indexed=len(store),
                chunks=len(chunks),
                action="rebuilding from chunks",
            )
            return cls.build(chunks, config)

        log.info("bm25_index_loaded", directory=str(directory), documents=len(store))
        return store


class _EmptyBm25:
    """Stand-in for an index over an empty corpus."""

    def get_scores(self, query: Sequence[str]) -> list[float]:
        return []
