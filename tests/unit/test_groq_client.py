"""The Groq client's retry and error-translation behaviour.

Rate limits are the normal case on Groq's free tiers, so the backoff path is
tested as a feature rather than an edge case. No test sleeps: the delay
function is injected.
"""

from __future__ import annotations

import sys
import types
from typing import Any

import pytest

from askmydocs.config import GenerationConfig
from askmydocs.errors import (
    ConfigError,
    GenerationError,
    LlmAuthError,
    LlmRateLimitError,
    LlmTimeoutError,
)
from askmydocs.generation.groq_client import GroqClient

MESSAGES = [{"role": "user", "content": "hi"}]


class _Error(Exception):
    """Provider error with a status code, as the SDK raises."""

    def __init__(self, status_code: int, message: str = "boom", retry_after: str | None = None):
        super().__init__(message)
        self.status_code = status_code
        if retry_after is not None:
            self.response = types.SimpleNamespace(
                status_code=status_code, headers={"retry-after": retry_after}
            )


class APITimeoutError(Exception):
    """Named to match the SDK's timeout class - classification is by name."""


class APIConnectionError(Exception):
    pass


def _completion(text: str = "an answer [1]", finish_reason: str = "stop") -> Any:
    return types.SimpleNamespace(
        model="llama-3.3-70b-versatile",
        choices=[
            types.SimpleNamespace(
                message=types.SimpleNamespace(content=text), finish_reason=finish_reason
            )
        ],
        usage=types.SimpleNamespace(prompt_tokens=100, completion_tokens=20),
    )


class _StubGroq:
    """Replays a script of results/exceptions, one per call."""

    def __init__(self, api_key: str, timeout: float | None = None, max_retries: int = 0):
        self.api_key = api_key
        self.timeout = timeout
        self.max_retries = max_retries
        self.script: list[Any] = []
        self.calls: list[dict[str, Any]] = []
        self.chat = types.SimpleNamespace(
            completions=types.SimpleNamespace(create=self._create)
        )

    def _create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        outcome = self.script.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


@pytest.fixture
def groq_module(monkeypatch: pytest.MonkeyPatch) -> list[_StubGroq]:
    created: list[_StubGroq] = []

    def factory(**kwargs: Any) -> _StubGroq:
        client = _StubGroq(**kwargs)
        created.append(client)
        return client

    module = types.ModuleType("groq")
    module.Groq = factory  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "groq", module)
    return created


@pytest.fixture
def config() -> GenerationConfig:
    return GenerationConfig(max_retries=3, retry_base_delay_s=1.0, retry_max_delay_s=10.0)


@pytest.fixture
def slept() -> list[float]:
    return []


def _client(config: GenerationConfig, slept: list[float], script: list[Any]) -> GroqClient:
    client = GroqClient(config, api_key="gsk-test", sleep=slept.append)
    client._load()  # materialise the stub so the script can be attached
    client._client.script = script
    return client


# -- construction ----------------------------------------------------------


def test_a_missing_api_key_fails_immediately(config: GenerationConfig) -> None:
    # Better here than as a 401 four retries deep.
    with pytest.raises(ConfigError, match="GROQ_API_KEY"):
        GroqClient(config, api_key=None)


def test_sdk_retries_are_disabled(
    config: GenerationConfig, groq_module: list[_StubGroq], slept: list[float]
) -> None:
    # The SDK retrying as well would silently multiply the attempts.
    _client(config, slept, [_completion()])
    assert groq_module[0].max_retries == 0


def test_timeout_is_passed_to_the_sdk(
    config: GenerationConfig, groq_module: list[_StubGroq], slept: list[float]
) -> None:
    config.request_timeout_s = 42
    _client(config, slept, [_completion()])
    assert groq_module[0].timeout == 42.0


# -- the happy path --------------------------------------------------------


def test_a_successful_call_returns_the_text(
    config: GenerationConfig, groq_module: list[_StubGroq], slept: list[float]
) -> None:
    response = _client(config, slept, [_completion("the answer [1]")]).complete(MESSAGES)

    assert response.text == "the answer [1]"
    assert response.model == "llama-3.3-70b-versatile"
    assert response.finish_reason == "stop"
    assert response.attempts == 1
    assert slept == []


def test_usage_is_captured(
    config: GenerationConfig, groq_module: list[_StubGroq], slept: list[float]
) -> None:
    response = _client(config, slept, [_completion()]).complete(MESSAGES)
    assert response.prompt_tokens == 100
    assert response.completion_tokens == 20
    assert response.total_tokens == 120


def test_model_parameters_are_sent(
    config: GenerationConfig, groq_module: list[_StubGroq], slept: list[float]
) -> None:
    config.temperature = 0.2
    config.max_tokens = 512
    _client(config, slept, [_completion()]).complete(MESSAGES)

    call = groq_module[0].calls[0]
    assert call["model"] == "llama-3.3-70b-versatile"
    assert call["temperature"] == 0.2
    assert call["max_tokens"] == 512
    assert call["messages"] == MESSAGES


def test_a_length_finish_reason_marks_the_response_truncated(
    config: GenerationConfig, groq_module: list[_StubGroq], slept: list[float]
) -> None:
    response = _client(config, slept, [_completion(finish_reason="length")]).complete(MESSAGES)
    assert response.truncated is True


# -- retrying --------------------------------------------------------------


def test_a_rate_limit_is_retried_then_succeeds(
    config: GenerationConfig, groq_module: list[_StubGroq], slept: list[float]
) -> None:
    client = _client(config, slept, [_Error(429), _Error(429), _completion("ok [1]")])
    response = client.complete(MESSAGES)

    assert response.text == "ok [1]"
    assert response.attempts == 3
    assert len(slept) == 2


@pytest.mark.parametrize("status", [429, 500, 502, 503, 504, 408])
def test_transient_statuses_are_retried(
    status: int, config: GenerationConfig, groq_module: list[_StubGroq], slept: list[float]
) -> None:
    client = _client(config, slept, [_Error(status), _completion()])
    assert client.complete(MESSAGES).attempts == 2


@pytest.mark.parametrize("error", [APITimeoutError("slow"), APIConnectionError("dropped")])
def test_network_failures_without_a_status_are_retried(
    error: Exception, config: GenerationConfig, groq_module: list[_StubGroq], slept: list[float]
) -> None:
    client = _client(config, slept, [error, _completion()])
    assert client.complete(MESSAGES).attempts == 2


def test_backoff_grows_between_attempts(
    config: GenerationConfig, groq_module: list[_StubGroq], slept: list[float]
) -> None:
    client = _client(config, slept, [_Error(500), _Error(500), _completion()])
    client.complete(MESSAGES)
    assert slept[1] > slept[0]


def test_backoff_is_capped(
    config: GenerationConfig, groq_module: list[_StubGroq], slept: list[float]
) -> None:
    config.retry_max_delay_s = 2.0
    config.max_retries = 4
    client = _client(config, slept, [_Error(500)] * 4 + [_completion()])
    client.complete(MESSAGES)
    assert all(delay <= 2.0 for delay in slept)


def test_a_retry_after_header_is_honoured(
    config: GenerationConfig, groq_module: list[_StubGroq], slept: list[float]
) -> None:
    # The provider knows better than our backoff curve when it tells us.
    client = _client(config, slept, [_Error(429, retry_after="7"), _completion()])
    client.complete(MESSAGES)
    assert slept == [7.0]


def test_retry_after_is_still_capped(
    config: GenerationConfig, groq_module: list[_StubGroq], slept: list[float]
) -> None:
    config.retry_max_delay_s = 3.0
    client = _client(config, slept, [_Error(429, retry_after="600"), _completion()])
    client.complete(MESSAGES)
    assert slept == [3.0]


def test_retries_are_exhausted_into_a_typed_error(
    config: GenerationConfig, groq_module: list[_StubGroq], slept: list[float]
) -> None:
    config.max_retries = 2
    client = _client(config, slept, [_Error(429)] * 3)

    with pytest.raises(LlmRateLimitError, match="2 retries"):
        client.complete(MESSAGES)
    assert len(slept) == 2


def test_zero_retries_means_one_attempt(
    config: GenerationConfig, groq_module: list[_StubGroq], slept: list[float]
) -> None:
    config.max_retries = 0
    client = _client(config, slept, [_Error(500)])

    with pytest.raises(GenerationError):
        client.complete(MESSAGES)
    assert slept == []


# -- what is never retried -------------------------------------------------


@pytest.mark.parametrize("status", [401, 403])
def test_auth_failures_fail_fast(
    status: int, config: GenerationConfig, groq_module: list[_StubGroq], slept: list[float]
) -> None:
    # Repeating a rejected key just burns quota and delays a clear message.
    client = _client(config, slept, [_Error(status)])

    with pytest.raises(LlmAuthError):
        client.complete(MESSAGES)
    assert slept == []


@pytest.mark.parametrize("status", [400, 404, 422])
def test_client_errors_are_not_retried(
    status: int, config: GenerationConfig, groq_module: list[_StubGroq], slept: list[float]
) -> None:
    client = _client(config, slept, [_Error(status)])

    with pytest.raises(GenerationError):
        client.complete(MESSAGES)
    assert slept == []


# -- error translation -----------------------------------------------------


def test_timeouts_translate_to_a_timeout_error(
    config: GenerationConfig, groq_module: list[_StubGroq], slept: list[float]
) -> None:
    config.max_retries = 0
    client = _client(config, slept, [APITimeoutError("too slow")])
    with pytest.raises(LlmTimeoutError):
        client.complete(MESSAGES)


def test_an_empty_choices_list_is_an_error(
    config: GenerationConfig, groq_module: list[_StubGroq], slept: list[float]
) -> None:
    empty = types.SimpleNamespace(model="m", choices=[], usage=None)
    with pytest.raises(GenerationError, match="no choices"):
        _client(config, slept, [empty]).complete(MESSAGES)


def test_a_null_message_content_becomes_empty_text(
    config: GenerationConfig, groq_module: list[_StubGroq], slept: list[float]
) -> None:
    response = _client(config, slept, [_completion(text=None)]).complete(MESSAGES)  # type: ignore[arg-type]
    assert response.text == ""
