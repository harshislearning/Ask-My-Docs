"""SentenceTransformerEmbedder.

The fake embedder used elsewhere bypasses this class entirely, so the two
bge-specific rules it enforces are pinned down here against a stubbed model:
queries get the instruction prefix, documents never do.
"""

from __future__ import annotations

import sys
import types
from typing import Any

import numpy as np
import pytest

from askmydocs.config import EmbeddingConfig
from askmydocs.indexing.embedder import SentenceTransformerEmbedder


class _StubModel:
    def __init__(self, name: str, device: str | None = None) -> None:
        self.name = name
        self.device = device
        self.calls: list[dict[str, Any]] = []

    def get_sentence_embedding_dimension(self) -> int:
        return 4

    def encode(self, texts: list[str], **kwargs: Any) -> np.ndarray:
        self.calls.append({"texts": list(texts), **kwargs})
        return np.ones((len(texts), 4), dtype=np.float32)


@pytest.fixture
def stub_sentence_transformers(monkeypatch: pytest.MonkeyPatch) -> list[_StubModel]:
    """Install a fake `sentence_transformers` module for the duration of a test."""
    created: list[_StubModel] = []

    def factory(name: str, device: str | None = None) -> _StubModel:
        model = _StubModel(name, device)
        created.append(model)
        return model

    module = types.ModuleType("sentence_transformers")
    module.SentenceTransformer = factory  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)
    return created


@pytest.fixture
def config() -> EmbeddingConfig:
    return EmbeddingConfig(
        model="BAAI/bge-base-en-v1.5",
        batch_size=8,
        normalize=True,
        query_prefix="Represent this sentence for searching relevant passages: ",
        device="cpu",
    )


def test_query_gets_the_instruction_prefix(
    config: EmbeddingConfig, stub_sentence_transformers: list[_StubModel]
) -> None:
    SentenceTransformerEmbedder(config).encode_query("how long is the timeout")
    sent = stub_sentence_transformers[0].calls[0]["texts"][0]

    assert sent.startswith("Represent this sentence for searching relevant passages: ")
    assert sent.endswith("how long is the timeout")


def test_documents_do_not_get_the_prefix(
    config: EmbeddingConfig, stub_sentence_transformers: list[_StubModel]
) -> None:
    # Prefixing documents measurably degrades bge retrieval quality.
    SentenceTransformerEmbedder(config).encode_documents(["a chunk of text"])
    assert stub_sentence_transformers[0].calls[0]["texts"] == ["a chunk of text"]


def test_empty_prefix_leaves_the_query_untouched(
    stub_sentence_transformers: list[_StubModel],
) -> None:
    embedder = SentenceTransformerEmbedder(EmbeddingConfig(query_prefix=""))
    embedder.encode_query("plain query")
    assert stub_sentence_transformers[0].calls[0]["texts"] == ["plain query"]


def test_normalisation_flag_is_passed_through(
    config: EmbeddingConfig, stub_sentence_transformers: list[_StubModel]
) -> None:
    SentenceTransformerEmbedder(config).encode_documents(["x"])
    assert stub_sentence_transformers[0].calls[0]["normalize_embeddings"] is True


def test_batch_size_is_passed_through(
    config: EmbeddingConfig, stub_sentence_transformers: list[_StubModel]
) -> None:
    SentenceTransformerEmbedder(config).encode_documents(["x", "y"])
    assert stub_sentence_transformers[0].calls[0]["batch_size"] == 8


def test_model_is_loaded_once_and_reused(
    config: EmbeddingConfig, stub_sentence_transformers: list[_StubModel]
) -> None:
    embedder = SentenceTransformerEmbedder(config)
    embedder.encode_query("one")
    embedder.encode_query("two")
    assert len(stub_sentence_transformers) == 1


def test_model_is_not_loaded_until_first_use(
    config: EmbeddingConfig, stub_sentence_transformers: list[_StubModel]
) -> None:
    # Importing torch costs seconds; constructing a retriever must not pay it.
    SentenceTransformerEmbedder(config)
    assert stub_sentence_transformers == []


def test_empty_document_list_skips_the_model(
    config: EmbeddingConfig, stub_sentence_transformers: list[_StubModel]
) -> None:
    result = SentenceTransformerEmbedder(config).encode_documents([])
    assert result.shape == (0, 4)


def test_auto_device_is_left_to_sentence_transformers(
    stub_sentence_transformers: list[_StubModel],
) -> None:
    SentenceTransformerEmbedder(EmbeddingConfig(device="auto")).encode_query("q")
    assert stub_sentence_transformers[0].device is None


def test_explicit_device_is_honoured(
    config: EmbeddingConfig, stub_sentence_transformers: list[_StubModel]
) -> None:
    SentenceTransformerEmbedder(config).encode_query("q")
    assert stub_sentence_transformers[0].device == "cpu"


def test_returned_vectors_are_float32(
    config: EmbeddingConfig, stub_sentence_transformers: list[_StubModel]
) -> None:
    # FAISS rejects float64 silently in some builds and loudly in others.
    embedder = SentenceTransformerEmbedder(config)
    assert embedder.encode_documents(["x"]).dtype == np.float32
    assert embedder.encode_query("x").dtype == np.float32


def test_query_vector_is_one_dimensional(
    config: EmbeddingConfig, stub_sentence_transformers: list[_StubModel]
) -> None:
    assert SentenceTransformerEmbedder(config).encode_query("x").shape == (4,)
