"""Hybrid retrieval: query both indexes independently, then fuse.

The two retrievers run on the *same* query and are deliberately not allowed to
influence each other - each produces its own ranking from its own view of the
corpus, and RRF reconciles them afterwards. That independence is what makes the
fusion meaningful: a chunk ranked highly by both is strong evidence, and a
chunk found by only one still gets a fair hearing.
"""

from __future__ import annotations

import time

from ..config import AppConfig, RetrievalConfig
from ..indexing.build import IndexBundle, load_index_bundle
from ..indexing.embedder import Embedder, SentenceTransformerEmbedder
from ..logging_setup import get_logger
from ..models import Candidate, Hit, RetrieverName
from .fusion import reciprocal_rank_fusion

log = get_logger(__name__)


class HybridRetriever:
    """Vector + BM25 retrieval fused with RRF."""

    def __init__(
        self,
        bundle: IndexBundle,
        embedder: Embedder,
        config: RetrievalConfig,
    ) -> None:
        self.bundle = bundle
        self.embedder = embedder
        self.config = config

    @classmethod
    def from_config(cls, config: AppConfig, embedder: Embedder | None = None) -> HybridRetriever:
        """Load indexes from disk and wire up a retriever."""
        return cls(
            bundle=load_index_bundle(config),
            embedder=embedder or SentenceTransformerEmbedder(config.embedding),
            config=config.retrieval,
        )

    # -- retrieval -------------------------------------------------------

    def retrieve(self, query: str, top_n: int | None = None) -> list[Candidate]:
        """Return fused candidates for ``query``, best first.

        ``top_n`` caps the fused list; the per-retriever depths stay as
        configured, because fusion needs a deep enough pool to work with.
        """
        query = query.strip()
        if not query:
            log.warning("empty_query")
            return []
        if not self.bundle.chunks:
            log.warning("retrieval_on_empty_index")
            return []

        started = time.perf_counter()
        vector_hits = self._vector_search(query)
        bm25_hits = self._bm25_search(query)

        if not vector_hits and not bm25_hits:
            log.warning("no_candidates_retrieved", query=query)
            return []

        fused = reciprocal_rank_fusion(
            {
                RetrieverName.VECTOR.value: [hit.chunk_id for hit in vector_hits],
                RetrieverName.BM25.value: [hit.chunk_id for hit in bm25_hits],
            },
            k=self.config.rrf_k,
            weights={
                RetrieverName.VECTOR.value: self.config.vector_weight,
                RetrieverName.BM25.value: self.config.bm25_weight,
            },
            top_n=top_n,
        )

        scores_by_retriever = {
            RetrieverName.VECTOR.value: {hit.chunk_id: hit.score for hit in vector_hits},
            RetrieverName.BM25.value: {hit.chunk_id: hit.score for hit in bm25_hits},
        }

        candidates: list[Candidate] = []
        for hit in fused:
            chunk = self.bundle.chunks_by_id.get(hit.chunk_id)
            if chunk is None:
                # Index and chunks.jsonl disagree; skip rather than fabricate.
                log.warning("candidate_chunk_missing", chunk_id=hit.chunk_id)
                continue
            candidates.append(
                Candidate(
                    chunk=chunk,
                    fused_score=hit.score,
                    fused_rank=len(candidates) + 1,
                    ranks=hit.ranks,
                    scores={
                        name: scores[hit.chunk_id]
                        for name, scores in scores_by_retriever.items()
                        if hit.chunk_id in scores
                    },
                )
            )

        log.info(
            "retrieval_completed",
            query_chars=len(query),
            vector_hits=len(vector_hits),
            bm25_hits=len(bm25_hits),
            fused_candidates=len(candidates),
            found_by_both=sum(1 for c in candidates if c.found_by_both),
            ms=round((time.perf_counter() - started) * 1000, 1),
        )
        return candidates

    # -- individual retrievers -------------------------------------------

    def _vector_search(self, query: str) -> list[Hit]:
        if self.config.vector_weight == 0.0:
            return []
        try:
            vector = self.embedder.encode_query(query)
        except Exception as exc:
            log.exception("query_embedding_failed", error=str(exc))
            return []
        return self.bundle.faiss.search(vector, self.config.vector_top_n)

    def _bm25_search(self, query: str) -> list[Hit]:
        if self.config.bm25_weight == 0.0:
            return []
        return self.bundle.bm25.search(query, self.config.bm25_top_n)
