"""Index construction over the chunks produced by ingestion.

Two parallel indexes are built from the same chunk set:

* :class:`~askmydocs.indexing.faiss_store.FaissStore` - dense, semantic
* :class:`~askmydocs.indexing.bm25_store.Bm25Store` - sparse, lexical

They fail in opposite directions, which is exactly why both exist: embeddings
handle paraphrase but blur exact identifiers, BM25 nails identifiers but misses
anything worded differently.
"""

from .bm25_store import Bm25Store, tokenize
from .build import IndexBuilder, IndexBundle, load_index_bundle, read_index_manifest
from .embedder import Embedder, SentenceTransformerEmbedder
from .faiss_store import FaissStore

__all__ = [
    "Bm25Store",
    "Embedder",
    "FaissStore",
    "IndexBuilder",
    "IndexBundle",
    "SentenceTransformerEmbedder",
    "load_index_bundle",
    "read_index_manifest",
    "tokenize",
]
