from __future__ import annotations

import pytest

from askmydocs.config import GenerationConfig
from askmydocs.errors import GenerationError, LlmRateLimitError
from askmydocs.generation.answerer import Answerer, is_refusal
from askmydocs.models import Candidate, Chunk
from fixtures.fake_llm import FailingLlmClient, FakeLlmClient

REFUSAL = "I don't have enough information in the provided sources to answer that."


def _candidate(chunk_id: str = "a", rank: int = 1) -> Candidate:
    chunk = Chunk(
        chunk_id=chunk_id,
        doc_id="doc-1",
        source_file="handbook.pdf",
        doc_title="Handbook",
        text="The request_timeout parameter defaults to 30 seconds.",
        embed_text="Handbook > 4. Timeouts\n\nThe request_timeout parameter defaults to 30 seconds.",
        section_path=["4. Timeouts"],
        page_start=14,
        page_end=14,
        chunk_index=rank,
        token_count=8,
    )
    return Candidate(
        chunk=chunk,
        fused_score=1.0 / rank,
        fused_rank=rank,
        ranks={"vector": rank},
        rerank_score=4.2,
    )


@pytest.fixture
def config() -> GenerationConfig:
    return GenerationConfig(refusal_text=REFUSAL)


# -- the happy path --------------------------------------------------------


def test_answer_carries_the_model_text(config: GenerationConfig) -> None:
    client = FakeLlmClient(text="The default is 30 seconds [1].")
    answer = Answerer(client, config).answer("How long?", [_candidate()])

    assert answer.text == "The default is 30 seconds [1]."
    assert answer.refused is False


def test_answer_records_the_sources_it_was_given(config: GenerationConfig) -> None:
    answer = Answerer(FakeLlmClient(), config).answer(
        "How long?", [_candidate("a", 1), _candidate("b", 2)]
    )
    assert [s.number for s in answer.sources] == [1, 2]
    assert answer.valid_citation_numbers == {1, 2}


def test_answer_records_usage_and_latency(config: GenerationConfig) -> None:
    answer = Answerer(FakeLlmClient(), config).answer("How long?", [_candidate()])

    assert answer.prompt_tokens == 120
    assert answer.completion_tokens == 25
    assert answer.latency_ms >= 0
    assert answer.model == "llama-3.3-70b-versatile"


def test_source_lookup_by_number(config: GenerationConfig) -> None:
    answer = Answerer(FakeLlmClient(), config).answer(
        "How long?", [_candidate("a", 1), _candidate("b", 2)]
    )
    assert answer.source_by_number(2) is not None
    assert answer.source_by_number(9) is None


def test_the_question_reaches_the_model(config: GenerationConfig) -> None:
    client = FakeLlmClient()
    Answerer(client, config).answer("What is the default timeout?", [_candidate()])
    assert "What is the default timeout?" in client.last_user_prompt


def test_the_source_text_reaches_the_model(config: GenerationConfig) -> None:
    client = FakeLlmClient()
    Answerer(client, config).answer("How long?", [_candidate()])
    assert "request_timeout parameter defaults to 30 seconds" in client.last_user_prompt


# -- no context ------------------------------------------------------------


def test_no_candidates_refuses_without_calling_the_model(
    config: GenerationConfig,
) -> None:
    # Calling the model with no evidence spends a request to be told what we
    # already know, and invites an answer from its parametric memory.
    client = FakeLlmClient()
    answer = Answerer(client, config).answer("Anything?", [])

    assert client.calls == []
    assert answer.refused is True
    assert answer.text == REFUSAL
    assert answer.sources == []
    assert answer.finish_reason == "no_context"


def test_an_empty_question_is_rejected(config: GenerationConfig) -> None:
    with pytest.raises(GenerationError):
        Answerer(FakeLlmClient(), config).answer("   ", [_candidate()])


# -- refusal detection -----------------------------------------------------


def test_the_exact_refusal_is_detected(config: GenerationConfig) -> None:
    answer = Answerer(FakeLlmClient(text=REFUSAL), config).answer("?", [_candidate()])
    assert answer.refused is True


def test_refusal_detection_tolerates_punctuation_and_case() -> None:
    # Models reproduce the sentence reliably but vary the trailing punctuation.
    # A refusal misread as an answer is scored as a hallucination by eval.
    assert is_refusal("I don't have enough information.", "I don't have enough information")
    assert is_refusal("I DO NOT... no wait", "I do not") is True


def test_a_refusal_embedded_in_a_longer_reply_still_counts() -> None:
    assert is_refusal(f"Sorry - {REFUSAL}", REFUSAL) is True


def test_an_empty_reply_counts_as_a_refusal() -> None:
    assert is_refusal("", REFUSAL) is True
    assert is_refusal("   ", REFUSAL) is True


def test_a_normal_answer_is_not_a_refusal() -> None:
    assert is_refusal("The default is 30 seconds [1].", REFUSAL) is False


# -- failures --------------------------------------------------------------


def test_provider_errors_propagate(config: GenerationConfig) -> None:
    # The API layer (Phase 6) decides the status code; swallowing the error
    # here would turn a rate limit into a silently empty answer.
    client = FailingLlmClient(LlmRateLimitError("rate limited"))
    with pytest.raises(LlmRateLimitError):
        Answerer(client, config).answer("How long?", [_candidate()])


def test_a_truncated_answer_is_still_returned(config: GenerationConfig) -> None:
    client = FakeLlmClient(text="The default is 30 sec", finish_reason="length")
    answer = Answerer(client, config).answer("How long?", [_candidate()])

    assert answer.finish_reason == "length"
    assert answer.text == "The default is 30 sec"
