"""Groq chat completions for llama-3.3-70b-versatile.

Wrapped rather than called directly for three reasons: the rest of the system
depends on a protocol it can fake in tests, every retry is logged as structured
data rather than swallowed by the SDK, and provider errors are translated into
this project's own exception types so callers never import `groq` to handle a
rate limit.

Retries cover exactly the failures that are worth retrying - 429s, 5xx,
timeouts, connection drops - with exponential backoff and jitter, honouring a
``Retry-After`` header when the provider sends one. A 401 or a malformed
request is never retried; repeating it just burns quota.
"""

from __future__ import annotations

import random
import time
from collections.abc import Callable, Sequence
from typing import Any, Protocol, runtime_checkable

from ..config import GenerationConfig
from ..errors import (
    ConfigError,
    GenerationError,
    LlmAuthError,
    LlmRateLimitError,
    LlmTimeoutError,
)
from ..logging_setup import get_logger
from ..models import LlmResponse

log = get_logger(__name__)

#: Status codes worth another attempt.
_RETRYABLE_STATUS = {408, 409, 429, 500, 502, 503, 504}
_AUTH_STATUS = {401, 403}


@runtime_checkable
class LlmClient(Protocol):
    """What the answerer needs from a language model."""

    @property
    def model_name(self) -> str: ...

    def complete(self, messages: Sequence[dict[str, str]]) -> LlmResponse: ...


class GroqClient:
    def __init__(
        self,
        config: GenerationConfig,
        api_key: str | None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if not api_key:
            raise ConfigError(
                "GROQ_API_KEY is not set - add it to .env or the environment"
            )
        self.config = config
        self._api_key = api_key
        self._sleep = sleep
        self._client: Any = None

    @property
    def model_name(self) -> str:
        return self.config.model

    def _load(self) -> Any:
        if self._client is None:
            from groq import Groq

            # Retries are handled here so each attempt can be logged; letting
            # the SDK retry as well would multiply the attempts silently.
            self._client = Groq(
                api_key=self._api_key,
                timeout=float(self.config.request_timeout_s),
                max_retries=0,
            )
        return self._client

    def complete(self, messages: Sequence[dict[str, str]]) -> LlmResponse:
        client = self._load()
        attempts = self.config.max_retries + 1
        last_error: Exception | None = None

        for attempt in range(1, attempts + 1):
            try:
                completion = client.chat.completions.create(
                    model=self.config.model,
                    messages=list(messages),
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens,
                )
                return _to_response(completion, self.config.model, attempt)
            except Exception as exc:
                last_error = exc
                status = _status_code(exc)

                if status in _AUTH_STATUS:
                    raise LlmAuthError(f"Groq rejected the API key ({status})") from exc

                if not _is_retryable(exc, status) or attempt == attempts:
                    break

                delay = self._backoff(attempt, exc)
                log.warning(
                    "llm_call_retrying",
                    attempt=attempt,
                    of=attempts,
                    status=status,
                    error=type(exc).__name__,
                    sleep_s=round(delay, 2),
                )
                self._sleep(delay)

        raise _translate(last_error, self.config.max_retries)

    def _backoff(self, attempt: int, exc: Exception) -> float:
        """Exponential backoff with jitter, or whatever the provider asked for."""
        retry_after = _retry_after(exc)
        if retry_after is not None:
            return min(retry_after, self.config.retry_max_delay_s)

        delay = self.config.retry_base_delay_s * (2 ** (attempt - 1))
        # Jitter keeps concurrent callers from retrying in lockstep.
        return min(delay, self.config.retry_max_delay_s) * random.uniform(0.6, 1.0)


# --------------------------------------------------------------------------
# Provider-shape helpers
#
# Deliberately duck-typed: the same code path works against the real SDK and
# against a stub in tests, and a new SDK exception class does not silently stop
# being retried.
# --------------------------------------------------------------------------


def _status_code(exc: Exception) -> int | None:
    status = getattr(exc, "status_code", None)
    if isinstance(status, int):
        return status
    response = getattr(exc, "response", None)
    status = getattr(response, "status_code", None)
    return status if isinstance(status, int) else None


def _is_retryable(exc: Exception, status: int | None) -> bool:
    if status is not None:
        return status in _RETRYABLE_STATUS
    name = type(exc).__name__.lower()
    return any(word in name for word in ("timeout", "connection", "ratelimit"))


def _retry_after(exc: Exception) -> float | None:
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if not headers:
        return None
    try:
        value = headers.get("retry-after")
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _translate(exc: Exception | None, max_retries: int) -> GenerationError:
    """Map a provider failure onto this project's error types."""
    if exc is None:  # pragma: no cover - defensive
        return GenerationError("the model call failed for an unknown reason")

    status = _status_code(exc)
    name = type(exc).__name__.lower()
    detail = f"{type(exc).__name__}: {exc}"

    if status == 429 or "ratelimit" in name:
        return LlmRateLimitError(
            f"Groq rate limit not cleared after {max_retries} retries - {detail}"
        )
    if "timeout" in name:
        return LlmTimeoutError(f"Groq request timed out - {detail}")
    return GenerationError(f"Groq request failed - {detail}")


def _to_response(completion: Any, model: str, attempt: int) -> LlmResponse:
    choice = completion.choices[0] if getattr(completion, "choices", None) else None
    if choice is None:
        raise GenerationError("Groq returned no choices")

    message = getattr(choice, "message", None)
    text = getattr(message, "content", None) or ""
    usage = getattr(completion, "usage", None)

    return LlmResponse(
        text=text.strip(),
        model=getattr(completion, "model", None) or model,
        finish_reason=getattr(choice, "finish_reason", None),
        prompt_tokens=getattr(usage, "prompt_tokens", None),
        completion_tokens=getattr(usage, "completion_tokens", None),
        attempts=attempt,
    )
