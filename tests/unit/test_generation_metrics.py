from __future__ import annotations

import pytest

from askmydocs.evaluation.generation_metrics import (
    citation_precision,
    citation_recall,
    contains_value,
    context_recall,
    groundedness,
    must_contain_coverage,
    refusal_breakdown,
    refusal_correct,
    score_answer,
)
from askmydocs.evaluation.golden import ExpectedSource, GoldenItem
from askmydocs.models import Answer, Source, VerificationReport


def _source(number: int, source_file: str = "handbook.pdf", page: str = "p. 3") -> Source:
    return Source(
        number=number,
        chunk_id=f"c{number}",
        doc_title="Handbook",
        source_file=source_file,
        page_label=page,
        section_path=["4. Timeouts"],
        text="body text",
    )


def _answer(
    text: str = "The default is 30 seconds [1].",
    sources: list[Source] | None = None,
    refused: bool = False,
    verification: VerificationReport | None = None,
) -> Answer:
    return Answer(
        question="How long?",
        text=text,
        sources=sources if sources is not None else [_source(1)],
        refused=refused,
        verification=verification,
    )


def _item(**kwargs: object) -> GoldenItem:
    defaults: dict[str, object] = {
        "id": "q001",
        "question": "How long?",
        "expected_sources": [ExpectedSource(source_file="handbook.pdf", page=3)],
    }
    return GoldenItem(**{**defaults, **kwargs})  # type: ignore[arg-type]


# -- citation recall -------------------------------------------------------


def test_citing_the_expected_source_scores_one() -> None:
    answer = _answer(
        verification=VerificationReport(citations=[1], valid_citations=[1], passed=True)
    )
    assert citation_recall(answer, _item()) == 1.0


def test_citing_the_wrong_source_scores_zero() -> None:
    # Retrieval can put the right chunk in front of the model and the model can
    # still answer from a different one. This is where that shows up.
    answer = _answer(
        sources=[_source(1, source_file="other.pdf", page="p. 9")],
        verification=VerificationReport(citations=[1], valid_citations=[1], passed=True),
    )
    assert citation_recall(answer, _item()) == 0.0


def test_partial_coverage_of_multiple_expected_sources() -> None:
    item = _item(
        expected_sources=[
            ExpectedSource(source_file="handbook.pdf", page=3),
            ExpectedSource(source_file="reference.pdf", page=1),
        ]
    )
    answer = _answer(
        sources=[_source(1), _source(2, "reference.pdf", "p. 1")],
        verification=VerificationReport(citations=[1], valid_citations=[1], passed=True),
    )
    assert citation_recall(answer, item) == 0.5


def test_a_source_spanning_the_expected_page_counts() -> None:
    answer = _answer(
        sources=[_source(1, page="pp. 2-5")],
        verification=VerificationReport(citations=[1], valid_citations=[1], passed=True),
    )
    assert citation_recall(answer, _item()) == 1.0


def test_an_uncited_source_does_not_count_for_citation_recall() -> None:
    # It was in the prompt but the answer never pointed at it.
    answer = _answer(
        text="The default is 30 seconds.",
        verification=VerificationReport(citations=[], valid_citations=[], passed=True),
    )
    assert citation_recall(answer, _item()) == 0.0
    # ...but it still counts as context: the retrieval half of the chain worked.
    assert context_recall(answer, _item()) == 1.0


def test_context_recall_is_the_ceiling_on_citation_recall() -> None:
    # The model cannot cite what it was never shown.
    answer = _answer(sources=[_source(1, "other.pdf", "p. 9")])
    assert context_recall(answer, _item()) == 0.0


# -- citation precision ----------------------------------------------------


def test_citation_precision_comes_from_verification() -> None:
    answer = _answer(
        verification=VerificationReport(
            citations=[1, 7], valid_citations=[1], invalid_citations=[7], passed=False
        )
    )
    assert citation_precision(answer) == 0.5


def test_citation_precision_without_a_report_is_one() -> None:
    assert citation_precision(_answer()) == 1.0


# -- groundedness ----------------------------------------------------------


def test_a_fully_cited_answer_is_grounded() -> None:
    answer = _answer(
        verification=VerificationReport(total_claims=3, cited_claims=3, passed=True)
    )
    assert groundedness(answer) == 1.0


def test_uncited_claims_reduce_groundedness() -> None:
    answer = _answer(
        verification=VerificationReport(total_claims=4, cited_claims=3, uncited_claims=1)
    )
    assert groundedness(answer) == 0.75


def test_unsupported_claims_reduce_groundedness() -> None:
    answer = _answer(
        verification=VerificationReport(total_claims=2, cited_claims=2, unsupported_claims=1)
    )
    assert groundedness(answer) == 0.5


def test_an_answer_with_no_claims_is_trivially_grounded() -> None:
    assert groundedness(_answer(verification=VerificationReport())) == 1.0


def test_groundedness_never_goes_negative() -> None:
    answer = _answer(
        verification=VerificationReport(
            total_claims=1, uncited_claims=1, unsupported_claims=1
        )
    )
    assert groundedness(answer) == 0.0


# -- must_contain ----------------------------------------------------------


def test_required_values_present_score_one() -> None:
    item = _item(must_contain=["30", "request_timeout"])
    answer = _answer(text="Set request_timeout to 30 seconds [1].")
    assert must_contain_coverage(answer, item) == 1.0


def test_a_missing_value_lowers_coverage() -> None:
    item = _item(must_contain=["30", "5"])
    answer = _answer(text="The timeout is 30 seconds [1].")
    assert must_contain_coverage(answer, item) == 0.5


def test_a_transposed_number_does_not_count_as_a_match() -> None:
    # The exact failure this metric exists to catch: fluent, cited, wrong value.
    # Naive substring matching would let "300" satisfy a requirement for "30".
    item = _item(must_contain=["30"])
    answer = _answer(text="Set request_timeout to 300 seconds [1].")
    assert must_contain_coverage(answer, item) == 0.0


@pytest.mark.parametrize(
    ("text", "needle", "expected"),
    [
        ("the timeout is 30 seconds", "30", True),
        ("the timeout is 300 seconds", "30", False),
        ("the timeout is 1.30 seconds", "30", False),
        ("set request_timeout now", "request_timeout", True),
        ("set request_timeout_ms now", "request_timeout", False),
        ("upgrade to v2.1.0 first", "v2.1.0", True),
        ("returns ERR_CONN_RESET", "err_conn_reset", True),
        ("uses 30% of the budget", "30%", True),
    ],
)
def test_values_match_on_word_boundaries(text: str, needle: str, expected: bool) -> None:
    assert contains_value(text, needle) is expected


def test_any_alternative_spelling_satisfies_a_requirement() -> None:
    # Documents write the same value as "30" in a table and "thirty" in prose.
    # Demanding one spelling would fail a correct answer for using the other.
    item = _item(must_contain=[["30", "thirty"]])
    assert must_contain_coverage(_answer(text="It is thirty seconds [1]."), item) == 1.0
    assert must_contain_coverage(_answer(text="It is 30 seconds [1]."), item) == 1.0


def test_no_alternative_matching_still_fails() -> None:
    item = _item(must_contain=[["30", "thirty"]])
    assert must_contain_coverage(_answer(text="It is 300 seconds [1]."), item) == 0.0


def test_alternatives_and_plain_values_mix() -> None:
    item = _item(must_contain=["request_timeout", ["30", "thirty"]])
    answer = _answer(text="Set request_timeout to thirty seconds [1].")
    assert must_contain_coverage(answer, item) == 1.0


def test_matching_is_case_insensitive() -> None:
    item = _item(must_contain=["Request_Timeout"])
    assert must_contain_coverage(_answer(text="request_timeout is set [1]."), item) == 1.0


def test_no_requirements_means_no_score() -> None:
    assert must_contain_coverage(_answer(), _item()) is None


# -- refusal ---------------------------------------------------------------


def test_refusing_an_unanswerable_question_is_correct() -> None:
    item = GoldenItem(id="q1", question="Parental leave?", answerable=False)
    assert refusal_correct(_answer(refused=True), item) is True


def test_answering_an_unanswerable_question_is_wrong() -> None:
    item = GoldenItem(id="q1", question="Parental leave?", answerable=False)
    assert refusal_correct(_answer(refused=False), item) is False


def test_refusing_an_answerable_question_is_wrong() -> None:
    assert refusal_correct(_answer(refused=True), _item()) is False


def test_the_refusal_breakdown_separates_the_two_failure_modes() -> None:
    # Over-refusing looks like caution and destroys usefulness; under-refusing
    # looks like helpfulness and is how a RAG system misleads people.
    unanswerable = GoldenItem(id="q2", question="Unrelated?", answerable=False)
    results = [
        (_answer(refused=False), _item()),          # correctly answered
        (_answer(refused=True), _item()),           # wrongly refused
        (_answer(refused=True), unanswerable),      # correctly refused
        (_answer(refused=False), unanswerable),     # wrongly answered
    ]
    breakdown = refusal_breakdown(results)

    assert breakdown["correctly_answered"] == 1
    assert breakdown["wrongly_refused"] == 1
    assert breakdown["correctly_refused"] == 1
    assert breakdown["wrongly_answered"] == 1
    assert breakdown["refusal_recall"] == 0.5
    assert breakdown["answer_rate"] == 0.5


def test_the_breakdown_of_an_empty_run_does_not_divide_by_zero() -> None:
    assert refusal_breakdown([])["refusal_recall"] == 1.0


# -- combined --------------------------------------------------------------


def test_score_answer_covers_an_answerable_item() -> None:
    scores = score_answer(
        _answer(verification=VerificationReport(total_claims=1, cited_claims=1, passed=True)),
        _item(must_contain=["30"]),
    )
    assert set(scores) >= {
        "citation_precision",
        "citation_recall",
        "context_recall",
        "groundedness",
        "refusal_correct",
        "answered",
        "must_contain_coverage",
    }


def test_recall_metrics_are_omitted_for_unanswerable_items() -> None:
    # There is no source to cite and no evidence to have retrieved.
    item = GoldenItem(id="q1", question="Unrelated?", answerable=False)
    scores = score_answer(_answer(refused=True), item)

    assert "citation_recall" not in scores
    assert "context_recall" not in scores
    assert scores["refusal_correct"] == 1.0


def test_scores_are_all_in_the_unit_interval() -> None:
    scores = score_answer(_answer(), _item(must_contain=["30"]))
    assert all(0.0 <= value <= 1.0 for value in scores.values())


@pytest.mark.parametrize("refused", [True, False])
def test_scoring_never_raises_on_a_refusal(refused: bool) -> None:
    assert score_answer(_answer(refused=refused), _item())
