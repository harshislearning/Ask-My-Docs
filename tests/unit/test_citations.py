"""Citation verification.

The failure this catches is the dangerous one: a fluent, confident, well-formed
answer citing a source that was never supplied, or asserting a fact with no
citation at all. Both look correct to a reader.
"""

from __future__ import annotations

import pytest

from askmydocs.config import GenerationConfig, VerificationConfig
from askmydocs.generation.prompts import build_sources
from askmydocs.models import Answer, Candidate, Chunk, IssueType, Severity, Source
from askmydocs.verification import Verifier, check_citations


def _source(number: int) -> Source:
    return Source(
        number=number,
        chunk_id=f"chunk-{number}",
        doc_title="Handbook",
        source_file="handbook.pdf",
        page_label=f"p. {number}",
        section_path=["4. Timeouts"],
        text=f"Body text for source {number}.",
    )


@pytest.fixture
def sources() -> list[Source]:
    return [_source(1), _source(2), _source(3)]


@pytest.fixture
def config() -> VerificationConfig:
    return VerificationConfig(enabled=True, flag_uncited_claims=True)


# -- citation validity -----------------------------------------------------


def test_a_fully_cited_answer_passes(
    sources: list[Source], config: VerificationConfig
) -> None:
    report = check_citations(
        "The default request timeout is 30 seconds [1]. Retries are capped at three [2].",
        sources,
        config,
    )
    assert report.passed is True
    assert report.issues == []
    assert report.citation_precision == 1.0


def test_a_fabricated_citation_is_flagged(
    sources: list[Source], config: VerificationConfig
) -> None:
    # Three sources were supplied; [7] cannot exist.
    report = check_citations("The value is 30 seconds [7].", sources, config)

    assert report.passed is False
    assert report.invalid_citations == [7]
    issue = report.issues_of(IssueType.UNKNOWN_SOURCE)[0]
    assert issue.severity is Severity.ERROR
    assert issue.citation == 7


def test_the_offending_sentence_is_reported(
    sources: list[Source], config: VerificationConfig
) -> None:
    report = check_citations("Fine [1]. The value is 30 seconds [9].", sources, config)
    issue = report.issues_of(IssueType.UNKNOWN_SOURCE)[0]
    assert "30 seconds" in (issue.sentence or "")


def test_citation_precision_mixes_valid_and_invalid(
    sources: list[Source], config: VerificationConfig
) -> None:
    report = check_citations("A [1]. B [2]. C [8]. D [9].", sources, config)
    assert report.citation_precision == pytest.approx(0.5)


def test_citing_zero_is_invalid(sources: list[Source], config: VerificationConfig) -> None:
    # Numbering starts at 1, so [0] is fabricated.
    report = check_citations("The value is 30 seconds [0].", sources, config)
    assert report.invalid_citations == [0]


def test_an_answer_with_no_sources_rejects_every_citation(
    config: VerificationConfig,
) -> None:
    report = check_citations("The value is 30 seconds [1].", [], config)
    assert report.passed is False
    assert report.invalid_citations == [1]


# -- claim coverage --------------------------------------------------------


def test_an_uncited_claim_is_flagged(
    sources: list[Source], config: VerificationConfig
) -> None:
    report = check_citations(
        "The default timeout is 30 seconds [1]. The retry budget is shared globally.",
        sources,
        config,
    )
    assert report.uncited_claims == 1
    assert report.passed is False
    issue = report.issues_of(IssueType.UNCITED_CLAIM)[0]
    assert issue.severity is Severity.WARNING
    assert "retry budget" in (issue.sentence or "")


def test_claim_coverage_is_reported(
    sources: list[Source], config: VerificationConfig
) -> None:
    report = check_citations(
        "First claim here [1]. Second claim here [2]. Third claim has no source.",
        sources,
        config,
    )
    assert report.total_claims == 3
    assert report.cited_claims == 2
    assert report.claim_coverage == pytest.approx(2 / 3)


def test_uncited_claim_flagging_can_be_switched_off(sources: list[Source]) -> None:
    config = VerificationConfig(flag_uncited_claims=False)
    report = check_citations("An uncited assertion about timeouts.", sources, config)

    assert report.uncited_claims == 1  # still measured
    assert report.issues == []  # but not raised
    assert report.passed is True


def test_a_refusal_sentence_is_not_treated_as_an_uncited_claim(
    sources: list[Source], config: VerificationConfig
) -> None:
    report = check_citations(
        "The sources do not cover the parental leave policy.", sources, config
    )
    assert report.uncited_claims == 0
    assert report.passed is True


def test_a_partial_answer_passes_when_the_gap_is_declared(
    sources: list[Source], config: VerificationConfig
) -> None:
    # The exact shape Phase 4 produces for half-answerable questions.
    report = check_citations(
        "Rollback triggers when the error rate exceeds the budget [1]. "
        "The sources do not cover how many retries happen first.",
        sources,
        config,
    )
    assert report.passed is True


def test_a_list_answer_with_cited_items_passes(
    sources: list[Source], config: VerificationConfig
) -> None:
    report = check_citations(
        "The timeouts are as follows:\n"
        "- request_timeout is 30 seconds [1]\n"
        "- health_timeout is 5 seconds [2]",
        sources,
        config,
    )
    assert report.passed is True
    assert report.total_claims == 2


# -- source usage ----------------------------------------------------------


def test_unused_sources_are_reported(
    sources: list[Source], config: VerificationConfig
) -> None:
    # Not a defect - a signal that rerank_top_k may be oversized.
    report = check_citations("Only the first was needed [1].", sources, config)

    assert report.unused_sources == [2, 3]
    assert report.source_usage == pytest.approx(1 / 3)
    assert report.passed is True


def test_using_every_source_reports_none_unused(
    sources: list[Source], config: VerificationConfig
) -> None:
    report = check_citations("A [1]. B [2]. C [3].", sources, config)
    assert report.unused_sources == []
    assert report.source_usage == 1.0


# -- code in answers -------------------------------------------------------


def test_code_samples_do_not_produce_phantom_citations(
    sources: list[Source], config: VerificationConfig
) -> None:
    report = check_citations(
        "Read the first entry with `values[0]` as shown [1].", sources, config
    )
    assert report.citations == [1]
    assert report.passed is True


# -- the verifier ----------------------------------------------------------


def _answer(text: str, source_count: int = 2) -> Answer:
    candidates = [
        Candidate(
            chunk=Chunk(
                chunk_id=f"chunk-{i}",
                doc_id="doc-1",
                source_file="handbook.pdf",
                doc_title="Handbook",
                text=f"Body {i}",
                embed_text=f"Handbook\n\nBody {i}",
                page_start=i,
                page_end=i,
                chunk_index=i,
                token_count=2,
            ),
            fused_score=1.0 / i,
            fused_rank=i,
        )
        for i in range(1, source_count + 1)
    ]
    return Answer(
        question="How long?",
        text=text,
        sources=build_sources(candidates, GenerationConfig()),
    )


def test_verifier_attaches_a_report(config: VerificationConfig) -> None:
    answer = Verifier(config).verify(_answer("The default is 30 seconds [1]."))
    assert answer.verification is not None
    assert answer.verification.passed is True


def test_verifier_never_edits_the_answer_text(config: VerificationConfig) -> None:
    # Silently repairing output would hide the failure from the user and eval.
    original = "The default is 30 seconds [9]."
    answer = Verifier(config).verify(_answer(original))

    assert answer.text == original
    assert answer.verification is not None
    assert answer.verification.passed is False


def test_a_refusal_is_not_checked_for_claims(config: VerificationConfig) -> None:
    answer = _answer("I don't have enough information to answer that.", source_count=2)
    answer.refused = True

    verified = Verifier(config).verify(answer)
    assert verified.verification is not None
    assert verified.verification.passed is True
    assert verified.verification.total_claims == 0


def test_verification_can_be_disabled() -> None:
    answer = Verifier(VerificationConfig(enabled=False)).verify(_answer("Anything [9]."))
    assert answer.verification is None


def test_entailment_mode_is_recorded(config: VerificationConfig) -> None:
    # Eval needs to know which verification regime produced a number.
    config.entailment_mode = "off"
    report = check_citations("A claim with a source [1].", [_source(1)], config)
    assert report.entailment_mode == "off"


def test_summary_is_loggable(sources: list[Source], config: VerificationConfig) -> None:
    summary = check_citations("A [1]. B has no source.", sources, config).summary()
    assert summary["citations"] == 1
    assert summary["uncited_claims"] == 1
    assert summary["passed"] is False
