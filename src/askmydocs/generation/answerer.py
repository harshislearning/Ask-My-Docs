"""Turning retrieved chunks into a cited answer."""

from __future__ import annotations

import re
import time
from collections.abc import Sequence

from ..config import AppConfig, GenerationConfig
from ..errors import GenerationError
from ..logging_setup import get_logger
from ..models import Answer, Candidate, LlmResponse
from .groq_client import GroqClient, LlmClient
from .prompts import build_messages, build_sources

log = get_logger(__name__)

_PUNCTUATION = re.compile(r"[^\w\s]")
_WHITESPACE = re.compile(r"\s+")


class Answerer:
    """Generates an answer from reranked candidates."""

    def __init__(self, client: LlmClient, config: GenerationConfig) -> None:
        self.client = client
        self.config = config

    @classmethod
    def from_config(cls, config: AppConfig, client: LlmClient | None = None) -> Answerer:
        return cls(
            client=client or GroqClient(config.generation, config.groq_api_key),
            config=config.generation,
        )

    def answer(self, question: str, candidates: Sequence[Candidate]) -> Answer:
        """Answer ``question`` using only ``candidates`` as evidence."""
        question = question.strip()
        if not question:
            raise GenerationError("question is empty")

        if not candidates:
            # No evidence means there is nothing to reason over. Calling the
            # model anyway would spend a request to be told what we already
            # know, and invites an answer from its own parametric memory.
            log.info("generation_skipped_no_context", question_chars=len(question))
            return Answer(
                question=question,
                text=self.config.refusal_text,
                refused=True,
                model=self.client.model_name,
                finish_reason="no_context",
            )

        sources = build_sources(candidates, self.config)
        messages = build_messages(question, sources, self.config)

        started = time.perf_counter()
        response = self.client.complete(messages)
        latency_ms = (time.perf_counter() - started) * 1000

        answer = Answer(
            question=question,
            text=response.text,
            sources=sources,
            refused=is_refusal(response.text, self.config.refusal_text),
            model=response.model,
            finish_reason=response.finish_reason,
            prompt_tokens=response.prompt_tokens,
            completion_tokens=response.completion_tokens,
            latency_ms=round(latency_ms, 1),
            attempts=response.attempts,
        )

        _log_answer(answer, response)
        return answer


def is_refusal(text: str, refusal_text: str) -> bool:
    """Whether the model declined to answer.

    Matched on normalised text rather than equality: models reliably reproduce
    the sentence but vary the trailing punctuation and capitalisation, and a
    refusal misread as an answer would be scored as a hallucination by eval.
    """
    if not text.strip():
        return True
    return _normalise(refusal_text) in _normalise(text)


def _normalise(text: str) -> str:
    return _WHITESPACE.sub(" ", _PUNCTUATION.sub("", text.lower())).strip()


def _log_answer(answer: Answer, response: LlmResponse) -> None:
    if response.truncated:
        # A cut-off answer often loses its final citation, which then looks
        # like an uncited claim to the verifier in Phase 5.
        log.warning(
            "answer_truncated_at_token_limit",
            completion_tokens=response.completion_tokens,
        )

    log.info(
        "answer_generated",
        model=answer.model,
        sources=len(answer.sources),
        refused=answer.refused,
        answer_chars=len(answer.text),
        prompt_tokens=answer.prompt_tokens,
        completion_tokens=answer.completion_tokens,
        finish_reason=answer.finish_reason,
        attempts=answer.attempts,
        latency_ms=answer.latency_ms,
    )
