"""PDF ingestion: parse -> detect structure -> chunk -> chunks.jsonl.

This package deliberately imports no ML libraries. Chunk-size accounting takes
an injected token counter, so the whole ingestion path (and its tests) runs
without torch or sentence-transformers installed.
"""

from .chunker import Chunker
from .pdf_loader import compute_doc_id, load_pdf
from .pipeline import IngestionPipeline
from .structure import build_sections, detect_headings

__all__ = [
    "Chunker",
    "IngestionPipeline",
    "build_sections",
    "compute_doc_id",
    "detect_headings",
    "load_pdf",
]
