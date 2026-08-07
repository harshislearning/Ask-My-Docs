"""The UI's API client, driven against the real ASGI app.

No server and no port: httpx talks to the FastAPI app in-process, so these
tests exercise the actual request and response shapes the front end depends on.
If the API's contract changes, this fails rather than the UI breaking at runtime.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from askmydocs.api import AppState, create_app
from askmydocs.config import AppConfig
from askmydocs.errors import LlmRateLimitError
from askmydocs.generation import Answerer
from askmydocs.indexing import IndexBuilder
from askmydocs.ingestion.pipeline import write_chunks
from askmydocs.models import Chunk
from askmydocs.ui import ApiError, AskMyDocsClient
from fixtures.fake_embedder import FakeEmbedder
from fixtures.fake_llm import FailingLlmClient, FakeLlmClient
from fixtures.fake_reranker import FakeReranker
from fixtures.pdf_factory import structured_pdf


def _app(config: AppConfig, llm: object | None = None):  # type: ignore[no-untyped-def]
    state = AppState.build(
        config,
        embedder=FakeEmbedder(),
        reranker=FakeReranker(),
        answerer=Answerer(llm or FakeLlmClient(), config.generation),
    )
    return create_app(state=state)


@pytest.fixture
def indexed_config(config: AppConfig, sample_chunks: list[Chunk]) -> AppConfig:
    write_chunks(config.paths.chunks_file, sample_chunks)
    IndexBuilder(config, embedder=FakeEmbedder()).build()
    return config


@pytest.fixture
def client(indexed_config: AppConfig) -> Iterator[AskMyDocsClient]:
    with TestClient(_app(indexed_config)) as http_client:
        yield AskMyDocsClient("http://testserver", http_client=http_client)


# -- health ----------------------------------------------------------------


def test_health_is_readable(client: AskMyDocsClient) -> None:
    health = client.health()
    assert health["status"] == "ok"
    assert health["chunk_count"] == 5


def test_an_unreachable_api_gives_actionable_advice() -> None:
    # The most common failure for a UI user: they started Streamlit but not the
    # backend. The message must say what to run.
    with (
        AskMyDocsClient("http://127.0.0.1:9") as api_client,
        pytest.raises(ApiError) as exc_info,
    ):
        api_client.health()

    assert "uvicorn" in exc_info.value.message


# -- query -----------------------------------------------------------------


def test_query_returns_the_shape_the_ui_renders(client: AskMyDocsClient) -> None:
    response = client.query("What is the request timeout?")

    assert {"question", "answer", "refused", "sources", "verification"} <= response.keys()
    assert response["sources"][0].keys() >= {"number", "source_file", "page_label", "text"}


def test_query_passes_the_retrieval_overrides(client: AskMyDocsClient) -> None:
    assert len(client.query("timeout", top_k=1)["sources"]) == 1


def test_source_text_can_be_left_out(client: AskMyDocsClient) -> None:
    response = client.query("timeout", include_source_text=False)
    assert all(source["text"] is None for source in response["sources"])


def test_an_empty_question_surfaces_a_readable_validation_error(
    client: AskMyDocsClient,
) -> None:
    # FastAPI returns a list of field errors; the UI must not render that raw.
    with pytest.raises(ApiError) as exc_info:
        client.query("")

    assert exc_info.value.status == 422
    assert "question" in exc_info.value.message


# -- error translation -----------------------------------------------------


def test_a_rate_limit_becomes_a_friendly_message(indexed_config: AppConfig) -> None:
    app = _app(indexed_config, llm=FailingLlmClient(LlmRateLimitError("slow down")))
    with TestClient(app, raise_server_exceptions=False) as http_client:
        api_client = AskMyDocsClient("http://testserver", http_client=http_client)
        with pytest.raises(ApiError) as exc_info:
            api_client.query("timeout")

    assert exc_info.value.status == 429
    assert "rate-limiting" in exc_info.value.message


def test_a_missing_index_is_flagged_as_not_ready(config: AppConfig) -> None:
    with TestClient(_app(config)) as http_client:
        api_client = AskMyDocsClient("http://testserver", http_client=http_client)
        with pytest.raises(ApiError) as exc_info:
            api_client.query("anything")

    # The UI keys off this to point the user at the Documents tab.
    assert exc_info.value.is_not_ready is True


# -- documents and ingestion ----------------------------------------------


def test_upload_then_ingest_then_list(
    client: AskMyDocsClient, indexed_config: AppConfig
) -> None:
    # The exact sequence the Documents tab drives.
    pdf = structured_pdf(indexed_config.paths.processed / "source.pdf")

    upload = client.upload([("handbook.pdf", pdf.read_bytes())])
    assert upload["uploaded"] == ["handbook.pdf"]

    job = client.ingest(rebuild_index=False)
    assert job["id"]

    finished = client.job(job["id"])
    assert finished["status"] == "succeeded"

    documents = client.documents()["documents"]
    assert any(doc["source_file"] == "handbook.pdf" for doc in documents)


def test_a_rejected_upload_is_reported(client: AskMyDocsClient) -> None:
    result = client.upload([("notes.txt", b"not a pdf")])
    assert result["rejected"] == ["notes.txt"]


def test_an_unknown_job_raises(client: AskMyDocsClient) -> None:
    with pytest.raises(ApiError) as exc_info:
        client.job("nope")
    assert exc_info.value.status == 404
