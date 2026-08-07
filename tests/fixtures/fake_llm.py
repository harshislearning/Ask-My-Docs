"""Stand-ins for the Groq client.

Generation tests must never hit the network: they would be slow, flaky, cost
money, and - because the model is non-deterministic - would test the model
rather than this code.
"""

from __future__ import annotations

from collections.abc import Sequence

from askmydocs.models import LlmResponse


class FakeLlmClient:
    """Returns canned responses and records what it was asked."""

    def __init__(
        self,
        text: str = "The default is 30 seconds [1].",
        finish_reason: str = "stop",
        model: str = "llama-3.3-70b-versatile",
    ) -> None:
        self.text = text
        self.finish_reason = finish_reason
        self._model = model
        self.calls: list[list[dict[str, str]]] = []

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def last_system_prompt(self) -> str:
        return self.calls[-1][0]["content"]

    @property
    def last_user_prompt(self) -> str:
        return self.calls[-1][1]["content"]

    def complete(self, messages: Sequence[dict[str, str]]) -> LlmResponse:
        self.calls.append([dict(message) for message in messages])
        return LlmResponse(
            text=self.text,
            model=self._model,
            finish_reason=self.finish_reason,
            prompt_tokens=120,
            completion_tokens=25,
        )


class FailingLlmClient:
    """Raises whatever it is given, to exercise error propagation."""

    def __init__(self, error: Exception) -> None:
        self.error = error

    @property
    def model_name(self) -> str:
        return "failing-model"

    def complete(self, messages: Sequence[dict[str, str]]) -> LlmResponse:
        raise self.error
