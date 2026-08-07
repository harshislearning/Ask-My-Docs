"""HTTP client for the Ask My Docs API.

The front end talks to the backend over HTTP rather than importing the pipeline
directly. That costs a hop and is worth it: there is exactly one code path into
retrieval and generation, the UI cannot drift from what the API actually does,
and using the UI genuinely exercises the API contract.

Error responses are translated into one exception type carrying a message fit to
show a user. A Streamlit app should never render a raw stack trace.
"""

from __future__ import annotations

from typing import Any

import httpx

from ..logging_setup import get_logger

log = get_logger(__name__)

DEFAULT_BASE_URL = "http://127.0.0.1:8000"
DEFAULT_TIMEOUT = 120.0


class ApiError(Exception):
    """A request failed. ``message`` is safe to show a user."""

    def __init__(self, message: str, *, status: int | None = None, code: str | None = None):
        super().__init__(message)
        self.message = message
        self.status = status
        self.code = code

    @property
    def is_not_ready(self) -> bool:
        """True when the service is up but has nothing indexed yet."""
        return self.status == 503


class AskMyDocsClient:
    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = DEFAULT_TIMEOUT,
        http_client: httpx.Client | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        # Injectable so tests can pass FastAPI's TestClient - itself an
        # httpx.Client - and exercise the real app with no server and no port.
        self._owns_client = http_client is None
        self._client = http_client or httpx.Client(base_url=self.base_url, timeout=timeout)

    def close(self) -> None:
        # An injected client belongs to whoever created it.
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> AskMyDocsClient:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    # -- endpoints -------------------------------------------------------

    def health(self) -> dict[str, Any]:
        return self._request("GET", "/health")

    def query(
        self,
        question: str,
        *,
        top_k: int | None = None,
        rerank: bool | None = None,
        include_source_text: bool = True,
    ) -> dict[str, Any]:
        return self._request(
            "POST",
            "/query",
            json={
                "question": question,
                "top_k": top_k,
                "rerank": rerank,
                "include_source_text": include_source_text,
            },
        )

    def upload(self, files: list[tuple[str, bytes]]) -> dict[str, Any]:
        return self._request(
            "POST",
            "/documents/upload",
            files=[("files", (name, payload, "application/pdf")) for name, payload in files],
        )

    def ingest(self, *, force: bool = False, rebuild_index: bool = True) -> dict[str, Any]:
        return self._request(
            "POST", "/ingest", json={"force": force, "rebuild_index": rebuild_index}
        )

    def documents(self) -> dict[str, Any]:
        return self._request("GET", "/documents")

    def job(self, job_id: str) -> dict[str, Any]:
        return self._request("GET", f"/jobs/{job_id}")

    # -- transport -------------------------------------------------------

    def _request(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        try:
            response = self._client.request(method, path, **kwargs)
        except httpx.ConnectError as exc:
            raise ApiError(
                f"Cannot reach the API at {self.base_url}. "
                "Start it with: uvicorn askmydocs.api.main:app"
            ) from exc
        except httpx.TimeoutException as exc:
            raise ApiError(
                "The API took too long to respond. It may still be loading its models."
            ) from exc

        if response.is_success:
            payload: dict[str, Any] = response.json()
            return payload

        raise _to_error(response)


def _to_error(response: httpx.Response) -> ApiError:
    """Turn an error response into something worth showing a user."""
    try:
        body = response.json()
    except ValueError:
        body = {}

    code = body.get("error")
    detail = body.get("detail")

    if isinstance(detail, list):
        # FastAPI's own validation errors arrive as a list of field problems.
        detail = "; ".join(
            f"{'.'.join(str(p) for p in item.get('loc', [])[1:])}: {item.get('msg')}"
            for item in detail
        )

    message = _FRIENDLY.get(code or "", detail or response.text or "The request failed.")
    log.warning("api_error", status=response.status_code, code=code, detail=detail)
    return ApiError(str(message), status=response.status_code, code=code)


#: Codes worth rewording. Everything else shows the API's own detail, which is
#: already written for a human.
_FRIENDLY = {
    "rate_limited": "Groq is rate-limiting this key. Wait a moment and try again.",
    "upstream_timeout": "The language model took too long. Try a shorter question.",
    "llm_auth_failed": "The server's GROQ_API_KEY was rejected. Check the server's .env.",
    "generation_failed": "The language model could not produce an answer. Try again.",
}
