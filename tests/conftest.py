"""Shared fixtures.

Tests never touch the network and never load an ML model: chunk sizing takes an
injected token counter, so the whole ingestion path is exercised offline.
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from askmydocs.config import AppConfig, load_config  # noqa: E402
from askmydocs.models import Chunk  # noqa: E402
from fixtures.fake_embedder import FakeEmbedder  # noqa: E402

TEST_CONFIG_FILE = REPO_ROOT / "config" / "test.yaml"


@pytest.fixture
def word_token_counter() -> Callable[[str], int]:
    """Deterministic stand-in for the bge tokenizer: one token per word.

    Real subword counts drift between library versions, which would make size
    assertions flaky for no benefit - what we are testing is the chunker's
    accounting, not the tokenizer's.
    """
    return lambda text: len(text.split()) if text else 0


@pytest.fixture
def config(tmp_path: Path) -> AppConfig:
    """Test config with every path redirected into a temp directory.

    ``groq_api_key`` is overridden with a placeholder. Pydantic Settings reads
    the real ``.env``, and a config object appears in full in every pytest
    failure report - so without this, one failing test prints a live API key
    into the terminal, and into the CI log of a public repository.
    """
    return load_config(
        TEST_CONFIG_FILE,
        groq_api_key="test-key-not-real",
        paths={
            "raw_pdfs": tmp_path / "raw_pdfs",
            "processed": tmp_path / "processed",
            "indexes": tmp_path / "indexes",
            "chunks_file": tmp_path / "processed" / "chunks.jsonl",
            "manifest_file": tmp_path / "processed" / "manifest.json",
        },
    )


@pytest.fixture
def pdf_dir(config: AppConfig) -> Path:
    path = config.paths.raw_pdfs
    path.mkdir(parents=True, exist_ok=True)
    return path


@pytest.fixture
def embedder() -> FakeEmbedder:
    """Deterministic embeddings - no model download, no torch."""
    return FakeEmbedder()


@pytest.fixture
def sample_chunks() -> list[Chunk]:
    """A miniature technical knowledge base.

    Deliberately contains near-synonyms (`request_timeout` vs
    `connection deadline`) so tests can show what each retriever is good at.
    """
    specs = [
        (
            "1. Timeouts",
            "The request_timeout parameter controls how long the gateway waits for an "
            "upstream response before giving up. It defaults to 30 seconds.",
            1,
        ),
        (
            "2. Connections",
            "Each connection has a deadline after which the socket is closed and the "
            "caller receives an error describing the failure.",
            2,
        ),
        (
            "3. Retries",
            "Failed requests are retried three times with exponential backoff. The retry "
            "budget is shared across the whole service.",
            3,
        ),
        (
            "4. Rollback",
            "Deployments roll back automatically when the error rate exceeds the budget "
            "for two consecutive evaluation windows.",
            4,
        ),
        (
            "5. Storage",
            "Objects are replicated across three availability zones before a write is "
            "acknowledged to the client.",
            5,
        ),
    ]
    return [
        Chunk(
            chunk_id=f"chunk-{index}",
            doc_id="doc-1",
            source_file="handbook.pdf",
            doc_title="Service Handbook",
            text=text,
            embed_text=f"Service Handbook > {heading}\n\n{text}",
            section_path=[heading],
            heading_level=1,
            page_start=page,
            page_end=page,
            chunk_index=index,
            token_count=len(text.split()),
        )
        for index, (heading, text, page) in enumerate(specs)
    ]
