"""Entailment checking: does the cited chunk actually say this?

Citation validity proves a number points at a real source. This layer is what
catches the answer that cites correctly and states the wrong value - the failure
that reads as completely authoritative.

Both layers must fail *open*: a parser hiccup or a rate limit must never
manufacture an accusation against an answer that may be perfectly sound.
"""

from __future__ import annotations

import pytest

from askmydocs.config import VerificationConfig
from askmydocs.models import Answer, LlmResponse, Source
from askmydocs.verification import Verifier
from askmydocs.verification.entailment import (
    Verdict,
    build_judge_messages,
    check_entailment,
    check_lexically,
    extract_probes,
    normalise_numbers,
    parse_judge_response,
)
from askmydocs.verification.sentences import split_sentences


def _source(number: int, text: str, section: str = "4. Timeouts") -> Source:
    return Source(
        number=number,
        chunk_id=f"chunk-{number}",
        doc_title="Handbook",
        source_file="handbook.pdf",
        page_label=f"p. {number}",
        section_path=[section],
        text=text,
    )


def _claim(text: str):
    return split_sentences(text)[0]


@pytest.fixture
def config() -> VerificationConfig:
    return VerificationConfig(entailment_mode="heuristic")


# -- number normalisation --------------------------------------------------


def test_number_words_become_digits() -> None:
    # Without this, "thirty seconds" looks ungrounded against a table reading 30.
    assert normalise_numbers("thirty seconds") == "30 seconds"
    assert normalise_numbers("five seconds") == "5 seconds"


def test_compound_number_words_become_digits() -> None:
    assert normalise_numbers("twenty-five retries") == "25 retries"


def test_digits_are_left_alone() -> None:
    assert normalise_numbers("30 seconds") == "30 seconds"


def test_unrelated_words_are_untouched() -> None:
    assert normalise_numbers("the deployment service") == "the deployment service"


# -- probe extraction ------------------------------------------------------


def test_numbers_are_extracted(config: VerificationConfig) -> None:
    numbers, _ = extract_probes("The timeout is 30 seconds.", config)
    assert numbers == {"30"}


def test_identifiers_are_extracted(config: VerificationConfig) -> None:
    _, identifiers = extract_probes("Set request_timeout in config.yaml.", config)
    assert identifiers == {"request_timeout", "config.yaml"}


def test_citation_markers_are_not_probes(config: VerificationConfig) -> None:
    numbers, _ = extract_probes("The timeout is 30 seconds [1][2].", config)
    assert numbers == {"30"}


def test_version_digits_are_not_counted_separately(config: VerificationConfig) -> None:
    # v2.1.0 is one identifier, not the numbers 2, 1 and 0.
    numbers, identifiers = extract_probes("Upgrade to v2.1.0 first.", config)
    assert identifiers == {"v2.1.0"}
    assert numbers == set()


def test_short_words_are_not_identifiers(config: VerificationConfig) -> None:
    _, identifiers = extract_probes("Use a.b for that.", config)
    assert identifiers == set()


def test_prose_without_values_yields_no_probes(config: VerificationConfig) -> None:
    numbers, identifiers = extract_probes("The service coordinates the rollout.", config)
    assert not numbers and not identifiers


# -- lexical grounding -----------------------------------------------------


def test_a_grounded_number_is_supported(config: VerificationConfig) -> None:
    sources = [_source(1, "The request_timeout parameter defaults to 30 seconds.")]
    check = check_lexically(_claim("The timeout is 30 seconds [1]."), sources, config)
    assert check.verdict is Verdict.SUPPORTED


def test_a_fabricated_number_is_caught(config: VerificationConfig) -> None:
    # The failure this whole layer exists for: valid citation, wrong value.
    sources = [_source(1, "The request_timeout parameter defaults to 30 seconds.")]
    check = check_lexically(_claim("The timeout is 210 seconds [1]."), sources, config)

    assert check.verdict is Verdict.NOT_SUPPORTED
    assert check.missing == ["210"]


def test_a_spelled_out_number_still_matches_digits(config: VerificationConfig) -> None:
    sources = [_source(1, "The request_timeout parameter defaults to 30 seconds.")]
    check = check_lexically(_claim("The timeout is thirty seconds [1]."), sources, config)
    assert check.verdict is Verdict.SUPPORTED


def test_a_digit_matches_a_spelled_out_source(config: VerificationConfig) -> None:
    sources = [_source(1, "The canary stage routes one percent of traffic.")]
    check = check_lexically(_claim("The canary routes 1 percent of traffic [1]."), sources, config)
    assert check.verdict is Verdict.SUPPORTED


def test_a_fabricated_identifier_is_caught(config: VerificationConfig) -> None:
    sources = [_source(1, "The request_timeout parameter defaults to 30 seconds.")]
    check = check_lexically(
        _claim("Set connection_deadline to 30 seconds [1]."), sources, config
    )
    assert check.verdict is Verdict.NOT_SUPPORTED
    assert "connection_deadline" in check.missing


def test_the_source_label_counts_as_grounding(config: VerificationConfig) -> None:
    # The model is shown the filename and section, so a claim naming them is
    # legitimately supported even though the chunk body never mentions them.
    sources = [_source(1, "Body text with no filename in it.")]
    check = check_lexically(
        _claim("The handbook.pdf documents this behaviour [1]."), sources, config
    )
    assert check.verdict is Verdict.SUPPORTED


def test_values_may_come_from_any_cited_source(config: VerificationConfig) -> None:
    sources = [_source(1, "The default is 30 seconds."), _source(2, "Health checks use 5.")]
    check = check_lexically(_claim("Timeouts are 30 and 5 seconds [1][2]."), sources, config)
    assert check.verdict is Verdict.SUPPORTED


def test_an_uncited_source_does_not_ground_a_claim(config: VerificationConfig) -> None:
    # Citing [1] while the value lives in [2] is exactly a misattribution.
    sources = [_source(1, "The default is 30 seconds."), _source(2, "Drain takes 120.")]
    check = check_lexically(_claim("Drain takes 120 seconds [1]."), sources, config)
    assert check.verdict is Verdict.NOT_SUPPORTED


def test_prose_with_nothing_checkable_is_inconclusive(config: VerificationConfig) -> None:
    sources = [_source(1, "The deployment service coordinates rollout.")]
    check = check_lexically(
        _claim("The service handles rollouts carefully [1]."), sources, config
    )
    assert check.verdict is Verdict.INCONCLUSIVE


def test_a_claim_citing_an_unknown_source_is_inconclusive(
    config: VerificationConfig,
) -> None:
    check = check_lexically(_claim("The timeout is 30 seconds [9]."), [_source(1, "x")], config)
    assert check.verdict is Verdict.INCONCLUSIVE


# -- the judge's prompt ----------------------------------------------------


def test_the_judge_sees_only_the_cited_source(config: VerificationConfig) -> None:
    # A judge shown the whole context could mark a claim supported by evidence
    # the answer never pointed at - the very error being looked for.
    sources = [_source(1, "Cited body text."), _source(2, "Uncited body text.")]
    checks = [check_lexically(_claim("A claim about things [1]."), sources, config)]

    content = build_judge_messages(checks, sources)[1]["content"]
    assert "Cited body text." in content
    assert "Uncited body text." not in content


def test_the_judge_prompt_strips_citation_markers(config: VerificationConfig) -> None:
    sources = [_source(1, "Body.")]
    checks = [check_lexically(_claim("A claim about things [1]."), sources, config)]
    content = build_judge_messages(checks, sources)[1]["content"]
    assert "[1]" not in content.split("SOURCE:")[0]


def test_the_judge_prompt_numbers_its_items(config: VerificationConfig) -> None:
    sources = [_source(1, "Body.")]
    checks = [
        check_lexically(_claim("First claim about things [1]."), sources, config),
        check_lexically(_claim("Second claim about things [1]."), sources, config),
    ]
    content = build_judge_messages(checks, sources)[1]["content"]
    assert "ITEM 1" in content and "ITEM 2" in content


# -- parsing the judge's reply ---------------------------------------------


def test_verdict_lines_are_parsed() -> None:
    parsed = parse_judge_response("1: SUPPORTED\n2: NOT_SUPPORTED\n3: PARTIAL", 3)
    assert parsed == {1: Verdict.SUPPORTED, 2: Verdict.NOT_SUPPORTED, 3: Verdict.PARTIAL}


def test_parsing_tolerates_formatting_drift() -> None:
    assert parse_judge_response("1. supported\n2) NOT_SUPPORTED", 2) == {
        1: Verdict.SUPPORTED,
        2: Verdict.NOT_SUPPORTED,
    }


def test_out_of_range_items_are_ignored() -> None:
    assert parse_judge_response("1: SUPPORTED\n9: NOT_SUPPORTED", 1) == {1: Verdict.SUPPORTED}


def test_unparseable_output_yields_nothing() -> None:
    # Fail open: a malformed reply must not become an accusation.
    assert parse_judge_response("I think they all look fine to me!", 2) == {}
    assert parse_judge_response("", 2) == {}


# -- layering --------------------------------------------------------------


class _Judge:
    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.calls: list[list[dict[str, str]]] = []

    @property
    def model_name(self) -> str:
        return "judge"

    def complete(self, messages):  # type: ignore[no-untyped-def]
        self.calls.append(list(messages))
        return LlmResponse(text=self.reply, model="judge")


class _BrokenJudge:
    @property
    def model_name(self) -> str:
        return "broken"

    def complete(self, messages):  # type: ignore[no-untyped-def]
        raise RuntimeError("rate limited")


def test_entailment_is_skipped_when_off() -> None:
    config = VerificationConfig(entailment_mode="off")
    claims = [_claim("The timeout is 210 seconds [1].")]
    assert check_entailment(claims, [_source(1, "30 seconds")], config) == []


def test_heuristic_mode_never_calls_the_judge(config: VerificationConfig) -> None:
    judge = _Judge("1: SUPPORTED")
    claims = [_claim("The timeout is 210 seconds [1].")]
    checks = check_entailment(claims, [_source(1, "30 seconds")], config, judge)

    assert judge.calls == []
    assert checks[0].verdict is Verdict.NOT_SUPPORTED


def test_llm_mode_escalates_only_unresolved_claims() -> None:
    config = VerificationConfig(entailment_mode="llm")
    sources = [_source(1, "The request_timeout defaults to 30 seconds.")]
    claims = [
        _claim("The timeout is 30 seconds [1]."),  # grounded, not escalated
        _claim("The service behaves reliably here [1]."),  # inconclusive, escalated
    ]
    judge = _Judge("1: SUPPORTED")

    checks = check_entailment(claims, sources, config, judge)

    assert len(judge.calls) == 1
    assert "ITEM 2" not in judge.calls[0][1]["content"]
    assert checks[0].checked_by == "lexical"
    assert checks[1].checked_by == "llm"


def test_the_judge_can_clear_a_lexical_false_positive() -> None:
    # "half a minute" has no digits to match, so lexical flags nothing; a
    # paraphrase that *does* trip it must be rescuable by the judge.
    config = VerificationConfig(entailment_mode="llm")
    sources = [_source(1, "The request_timeout defaults to 30 seconds.")]
    claims = [_claim("The timeout is 0.5 minutes [1].")]

    checks = check_entailment(claims, sources, config, _Judge("1: SUPPORTED"))
    assert checks[0].verdict is Verdict.SUPPORTED


def test_the_judge_can_confirm_a_contradiction() -> None:
    config = VerificationConfig(entailment_mode="llm")
    sources = [_source(1, "Rollback must be triggered manually by an operator.")]
    claims = [_claim("Rollback happens automatically without operator action [1].")]

    checks = check_entailment(claims, sources, config, _Judge("1: NOT_SUPPORTED"))
    assert checks[0].verdict is Verdict.NOT_SUPPORTED
    assert checks[0].checked_by == "llm"


def test_a_failing_judge_falls_back_to_lexical_verdicts() -> None:
    config = VerificationConfig(entailment_mode="llm")
    sources = [_source(1, "The request_timeout defaults to 30 seconds.")]
    claims = [_claim("The timeout is 210 seconds [1].")]

    checks = check_entailment(claims, sources, config, _BrokenJudge())
    assert checks[0].verdict is Verdict.NOT_SUPPORTED
    assert checks[0].checked_by == "lexical"


def test_an_unanswered_item_becomes_inconclusive() -> None:
    # The judge replied, but not about this claim. Reporting the lexical verdict
    # anyway would attribute a judgement nothing made.
    config = VerificationConfig(entailment_mode="llm")
    sources = [_source(1, "The request_timeout defaults to 30 seconds.")]
    claims = [_claim("The timeout is 210 seconds [1].")]

    checks = check_entailment(claims, sources, config, _Judge("nothing useful"))
    assert checks[0].verdict is Verdict.INCONCLUSIVE


def test_llm_mode_without_a_client_stays_lexical() -> None:
    config = VerificationConfig(entailment_mode="llm")
    sources = [_source(1, "The request_timeout defaults to 30 seconds.")]
    claims = [_claim("The timeout is 210 seconds [1].")]

    checks = check_entailment(claims, sources, config, None)
    assert checks[0].verdict is Verdict.NOT_SUPPORTED


def test_the_batch_is_capped() -> None:
    config = VerificationConfig(entailment_mode="llm", entailment_max_claims=2)
    sources = [_source(1, "Body text about the service.")]
    claims = [_claim(f"Claim number {i} about the service [1].") for i in range(5)]

    judge = _Judge("1: SUPPORTED\n2: SUPPORTED")
    check_entailment(claims, sources, config, judge)
    assert "ITEM 3" not in judge.calls[0][1]["content"]


# -- through the verifier --------------------------------------------------


def test_an_unsupported_claim_fails_verification() -> None:
    config = VerificationConfig(entailment_mode="heuristic")
    answer = Answer(
        question="How long?",
        text="The drain timeout is 210 seconds [1].",
        sources=[_source(1, "The request_timeout defaults to 30 seconds.")],
    )

    verified = Verifier(config).verify(answer)
    assert verified.verification is not None
    assert verified.verification.passed is False
    assert verified.verification.unsupported_claims == 1


def test_a_grounded_answer_passes_with_entailment_on() -> None:
    config = VerificationConfig(entailment_mode="heuristic")
    answer = Answer(
        question="How long?",
        text="The request_timeout is 30 seconds [1].",
        sources=[_source(1, "The request_timeout parameter defaults to 30 seconds.")],
    )

    verified = Verifier(config).verify(answer)
    assert verified.verification is not None
    assert verified.verification.passed is True
    assert verified.verification.unsupported_claims == 0


def test_entailment_is_not_run_when_disabled() -> None:
    config = VerificationConfig(entailment_mode="off")
    answer = Answer(
        question="How long?",
        text="The drain timeout is 210 seconds [1].",
        sources=[_source(1, "The request_timeout defaults to 30 seconds.")],
    )

    verified = Verifier(config).verify(answer)
    assert verified.verification is not None
    assert verified.verification.unsupported_claims == 0
    assert verified.verification.passed is True
