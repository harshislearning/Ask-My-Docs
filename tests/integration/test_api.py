"""The HTTP API, exercised end to end against fakes.

No model downloads, no API key, no network: the whole app runs with an injected
embedder, reranker and LLM client, which is the point of keeping those behind
protocols.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from askmydocs.api import AppState, create_app
from askmydocs.config import AppConfig
from askmydocs.errors import LlmRateLimitError
from askmydocs.evaluation.golden import ExpectedSource, GoldenItem, write_golden_set
from askmydocs.generation import Answerer
from askmydocs.indexing import IndexBuilder
from askmydocs.ingestion.pipeline import write_chunks
from askmydocs.models import Chunk
from fixtures.fake_embedder import FakeEmbedder
from fixtures.fake_llm import FailingLlmClient, FakeLlmClient
from fixtures.fake_reranker import FakeReranker
from fixtures.pdf_factory import structured_pdf


def _state(config: AppConfig, llm: object | None = None) -> AppState:
    return AppState.build(
        config,
        embedder=FakeEmbedder(),
        reranker=FakeReranker(),
        answerer=Answerer(llm or FakeLlmClient(), config.generation),
    )


@pytest.fixture
def indexed_config(
    config: AppConfig, sample_chunks: list[Chunk]
) -> AppConfig:
    write_chunks(config.paths.chunks_file, sample_chunks)
    IndexBuilder(config, embedder=FakeEmbedder()).build()
    return config


@pytest.fixture
def client(indexed_config: AppConfig) -> Iterator[TestClient]:
    with TestClient(create_app(state=_state(indexed_config))) as test_client:
        yield test_client


@pytest.fixture
def bare_client(config: AppConfig) -> Iterator[TestClient]:
    """A service that booted with no index - the state before first ingest."""
    with TestClient(create_app(state=_state(config))) as test_client:
        yield test_client


# -- health ----------------------------------------------------------------


def test_health_reports_ok_when_indexed(client: TestClient) -> None:
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["index_built"] is True
    assert body["chunk_count"] == 5


def test_health_reports_degraded_without_an_index(bare_client: TestClient) -> None:
    response = bare_client.get("/health")
    body = response.json()

    # 200, not 503: the process is fine, it just has nothing to serve yet, and
    # restarting the container would not help.
    assert response.status_code == 200
    assert body["status"] == "degraded"
    assert body["index_built"] is False

    index = next(c for c in body["components"] if c["name"] == "index")
    assert index["ready"] is False
    assert index["detail"]


def test_health_names_the_component_that_is_not_ready(bare_client: TestClient) -> None:
    components = {c["name"]: c for c in bare_client.get("/health").json()["components"]}
    assert set(components) == {"index", "reranker", "llm"}


def test_startup_warms_the_models(indexed_config: AppConfig) -> None:
    # Without this, the first user after a restart pays for loading bge and the
    # cross-encoder - 13 seconds measured, against 0.4 for every query after.
    embedder = FakeEmbedder()
    reranker = FakeReranker()
    state = AppState.build(
        indexed_config,
        embedder=embedder,
        reranker=reranker,
        answerer=Answerer(FakeLlmClient(), indexed_config.generation),
    )

    with TestClient(create_app(state=state)):
        pass

    assert embedder.encoded_queries == ["warmup"]
    assert reranker.calls


def test_warming_is_skipped_without_an_index(config: AppConfig) -> None:
    embedder = FakeEmbedder()
    state = AppState.build(
        config,
        embedder=embedder,
        reranker=FakeReranker(),
        answerer=Answerer(FakeLlmClient(), config.generation),
    )

    with TestClient(create_app(state=state)):
        pass

    assert embedder.encoded_queries == []


# -- query -----------------------------------------------------------------


def test_query_returns_a_cited_answer(client: TestClient) -> None:
    response = client.post("/query", json={"question": "What is the request timeout?"})
    body = response.json()

    assert response.status_code == 200
    assert body["answer"]
    assert body["sources"]
    assert body["sources"][0]["number"] == 1
    assert body["model"] == "llama-3.3-70b-versatile"


def test_query_includes_the_verification_report(client: TestClient) -> None:
    body = client.post("/query", json={"question": "What is the request timeout?"}).json()
    verification = body["verification"]

    assert verification is not None
    assert "passed" in verification
    assert "citation_precision" in verification
    assert "claim_coverage" in verification


def test_query_reports_timings(client: TestClient) -> None:
    body = client.post("/query", json={"question": "timeout"}).json()
    assert body["retrieval_ms"] >= 0
    assert body["total_ms"] >= body["retrieval_ms"]


def test_source_text_can_be_omitted(client: TestClient) -> None:
    # The UI lists sources before the user expands any of them; sending every
    # chunk body on that first request is pure waste.
    body = client.post(
        "/query", json={"question": "timeout", "include_source_text": False}
    ).json()
    assert body["sources"]
    assert all(source["text"] is None for source in body["sources"])


def test_top_k_override_is_honoured(client: TestClient) -> None:
    body = client.post("/query", json={"question": "timeout", "top_k": 1}).json()
    assert len(body["sources"]) == 1


def test_a_request_override_does_not_leak_into_later_requests(
    client: TestClient,
) -> None:
    # The handler copies the config; mutating the shared one would silently
    # reconfigure every other caller.
    client.post("/query", json={"question": "timeout", "top_k": 1})
    body = client.post("/query", json={"question": "timeout"}).json()
    assert len(body["sources"]) > 1


def test_page_numbers_survive_the_wire_format(client: TestClient) -> None:
    source = client.post("/query", json={"question": "timeout"}).json()["sources"][0]
    assert source["page_start"] >= 1
    assert source["page_end"] >= source["page_start"]
    assert source["page_label"].startswith("p")


def test_a_refusal_is_a_200_not_an_error(indexed_config: AppConfig) -> None:
    # Refusing is correct behaviour. A client must be able to tell it apart
    # from a failure it might retry.
    refusal = indexed_config.generation.refusal_text
    state = _state(indexed_config, llm=FakeLlmClient(text=refusal))

    with TestClient(create_app(state=state)) as client:
        response = client.post("/query", json={"question": "anything"})

    assert response.status_code == 200
    assert response.json()["refused"] is True


def test_query_without_an_index_is_503(bare_client: TestClient) -> None:
    response = bare_client.post("/query", json={"question": "anything"})
    assert response.status_code == 503
    assert "ingest" in response.json()["detail"].lower()


def test_an_empty_question_is_rejected(client: TestClient) -> None:
    assert client.post("/query", json={"question": ""}).status_code == 422


def test_a_missing_question_is_rejected(client: TestClient) -> None:
    assert client.post("/query", json={}).status_code == 422


# -- error mapping ---------------------------------------------------------


def test_a_rate_limit_becomes_429(indexed_config: AppConfig) -> None:
    state = _state(indexed_config, llm=FailingLlmClient(LlmRateLimitError("slow down")))
    with TestClient(create_app(state=state), raise_server_exceptions=False) as client:
        response = client.post("/query", json={"question": "timeout"})

    assert response.status_code == 429
    assert response.json()["error"] == "rate_limited"


def test_an_auth_failure_does_not_leak_details(indexed_config: AppConfig) -> None:
    from askmydocs.errors import LlmAuthError

    state = _state(indexed_config, llm=FailingLlmClient(LlmAuthError("bad key gsk_xyz")))
    with TestClient(create_app(state=state), raise_server_exceptions=False) as client:
        response = client.post("/query", json={"question": "timeout"})

    assert response.status_code == 500
    assert "gsk_xyz" not in response.text


def test_a_timeout_becomes_504(indexed_config: AppConfig) -> None:
    from askmydocs.errors import LlmTimeoutError

    state = _state(indexed_config, llm=FailingLlmClient(LlmTimeoutError("too slow")))
    with TestClient(create_app(state=state), raise_server_exceptions=False) as client:
        assert client.post("/query", json={"question": "t"}).status_code == 504


def test_errors_carry_the_request_id(indexed_config: AppConfig) -> None:
    state = _state(indexed_config, llm=FailingLlmClient(LlmRateLimitError("nope")))
    with TestClient(create_app(state=state), raise_server_exceptions=False) as client:
        body = client.post("/query", json={"question": "t"}).json()
    assert body["request_id"]


# -- request tracing -------------------------------------------------------


def test_every_response_carries_a_request_id(client: TestClient) -> None:
    assert client.get("/health").headers["x-request-id"]


def test_a_supplied_request_id_is_preserved(client: TestClient) -> None:
    # Lets a trace started upstream continue through this service.
    response = client.get("/health", headers={"x-request-id": "abc123"})
    assert response.headers["x-request-id"] == "abc123"


# -- documents and ingestion ----------------------------------------------


def test_uploading_a_pdf_stores_it(client: TestClient, indexed_config: AppConfig) -> None:
    pdf = structured_pdf(indexed_config.paths.processed / "upload.pdf")
    response = client.post(
        "/documents/upload",
        files={"files": ("handbook.pdf", pdf.read_bytes(), "application/pdf")},
    )

    assert response.status_code == 200
    assert response.json()["uploaded"] == ["handbook.pdf"]
    assert (indexed_config.paths.raw_pdfs / "handbook.pdf").is_file()


def test_a_non_pdf_upload_is_rejected(client: TestClient) -> None:
    response = client.post(
        "/documents/upload", files={"files": ("notes.txt", b"hello", "text/plain")}
    )
    body = response.json()

    assert body["uploaded"] == []
    assert body["rejected"] == ["notes.txt"]


def test_upload_filenames_cannot_escape_the_directory(
    client: TestClient, indexed_config: AppConfig, tmp_path: Path
) -> None:
    # Uploads are attacker-controlled input even on an internal tool.
    client.post(
        "/documents/upload",
        files={"files": ("../../pwned.pdf", b"%PDF-1.7\n", "application/pdf")},
    )
    assert not (indexed_config.paths.raw_pdfs.parent.parent / "pwned.pdf").exists()


def test_listing_documents_reflects_the_manifest(
    client: TestClient, indexed_config: AppConfig, pdf_dir: Path
) -> None:
    structured_pdf(pdf_dir / "handbook.pdf")
    client.post("/ingest", json={"rebuild_index": False})

    documents = client.get("/documents").json()["documents"]
    assert any(d["source_file"] == "handbook.pdf" for d in documents)


def test_ingest_returns_a_job(client: TestClient, pdf_dir: Path) -> None:
    structured_pdf(pdf_dir / "handbook.pdf")
    response = client.post("/ingest", json={"rebuild_index": False})

    assert response.status_code == 202
    body = response.json()
    assert body["kind"] == "ingest"
    assert body["id"]


def test_a_finished_job_reports_its_result(client: TestClient, pdf_dir: Path) -> None:
    # TestClient runs background tasks synchronously, so the job is already done.
    structured_pdf(pdf_dir / "handbook.pdf")
    job_id = client.post("/ingest", json={"rebuild_index": False}).json()["id"]

    job = client.get(f"/jobs/{job_id}").json()
    assert job["status"] == "succeeded"
    assert job["result"]["documents_ok"] == 1
    assert job["finished_at"]


def test_ingest_can_rebuild_the_index(client: TestClient, pdf_dir: Path) -> None:
    structured_pdf(pdf_dir / "handbook.pdf")
    job_id = client.post("/ingest", json={"rebuild_index": True}).json()["id"]

    job = client.get(f"/jobs/{job_id}").json()
    assert job["status"] == "succeeded"
    assert job["result"]["indexed_chunks"] > 0
    assert job["result"]["index_reloaded"] is True


def test_ingesting_nothing_is_not_a_failure(client: TestClient) -> None:
    job_id = client.post("/ingest", json={"rebuild_index": True}).json()["id"]
    job = client.get(f"/jobs/{job_id}").json()

    assert job["status"] == "succeeded"
    assert job["result"]["documents_total"] == 0


def test_a_failing_job_is_recorded_not_raised(
    client: TestClient, indexed_config: AppConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(*args: object, **kwargs: object) -> None:
        raise RuntimeError("disk on fire")

    monkeypatch.setattr("askmydocs.api.routes.ingest.IngestionPipeline", boom)
    job_id = client.post("/ingest", json={"rebuild_index": False}).json()["id"]

    job = client.get(f"/jobs/{job_id}").json()
    assert job["status"] == "failed"
    assert "disk on fire" in job["error"]


def test_an_unknown_job_is_404(client: TestClient) -> None:
    response = client.get("/jobs/deadbeef")
    assert response.status_code == 404
    assert "restart" in response.json()["detail"]


def test_jobs_can_be_listed(client: TestClient) -> None:
    client.post("/ingest", json={"rebuild_index": False})
    assert len(client.get("/jobs").json()) == 1


# -- evaluation ------------------------------------------------------------


def test_eval_runs_the_harness(
    client: TestClient, indexed_config: AppConfig, tmp_path: Path
) -> None:
    golden = tmp_path / "golden.jsonl"
    write_golden_set(
        golden,
        [
            GoldenItem(
                id="q001",
                question="What is the request_timeout default?",
                expected_sources=[ExpectedSource(source_file="handbook.pdf", page=1)],
            )
        ],
    )

    job_id = client.post("/eval", json={"golden_set": str(golden)}).json()["id"]
    job = client.get(f"/jobs/{job_id}").json()

    assert job["status"] == "succeeded", job.get("error")
    assert job["result"]["items_evaluated"] == 1
    assert "retrieval" in job["result"]


def test_eval_reports_a_missing_golden_set(client: TestClient, tmp_path: Path) -> None:
    job_id = client.post("/eval", json={"golden_set": str(tmp_path / "nope.jsonl")}).json()["id"]
    job = client.get(f"/jobs/{job_id}").json()

    assert job["status"] == "failed"
    assert "golden set not found" in job["error"]


def test_eval_without_an_index_is_503(bare_client: TestClient) -> None:
    assert bare_client.post("/eval", json={}).status_code == 503


# -- openapi ---------------------------------------------------------------


def test_the_schema_documents_every_endpoint(client: TestClient) -> None:
    paths = client.get("/openapi.json").json()["paths"]
    assert {"/health", "/query", "/ingest", "/documents", "/eval"} <= set(paths)
