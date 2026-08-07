"""Metrics over a generated answer.

These cover what RAGAS does not: whether the citations point at the *expected*
evidence, and whether the system refused when it should have. Both are specific
to this system's contract - RAGAS has no notion of a numbered citation, and no
notion of a question that is supposed to go unanswered.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from ..models import Answer, Source
from .golden import GoldenItem


def citation_precision(answer: Answer) -> float:
    """Share of the answer's citations that point at a real source.

    Comes straight from verification. 1.0 when nothing was cited - there is
    nothing wrong to measure, and penalising a refusal for not citing would
    reward it for citing something.
    """
    if answer.verification is None:
        return 1.0
    return answer.verification.citation_precision


def citation_recall(answer: Answer, item: GoldenItem) -> float:
    """Share of the expected sources the answer actually cited.

    Distinct from retrieval recall in the way that matters: retrieval can put
    the right chunk in front of the model, and the model can still answer from
    a different one. This measures the end of that chain.
    """
    if not item.expected_sources:
        return 1.0

    cited = _cited_sources(answer)
    covered = sum(
        1
        for expected in item.expected_sources
        if any(_source_matches(expected, source) for source in cited)
    )
    return covered / len(item.expected_sources)


def context_recall(answer: Answer, item: GoldenItem) -> float:
    """Share of expected sources that made it into the prompt at all.

    The ceiling on citation recall: the model cannot cite what it was not shown.
    Comparing the two separates a retrieval problem from a generation one.
    """
    if not item.expected_sources:
        return 1.0
    covered = sum(
        1
        for expected in item.expected_sources
        if any(_source_matches(expected, source) for source in answer.sources)
    )
    return covered / len(item.expected_sources)


def must_contain_coverage(answer: Answer, item: GoldenItem) -> float | None:
    """Share of required values present in the answer.

    Deliberately literal. For technical documentation the values that matter -
    ``30``, ``request_timeout``, ``ERR_CONN_RESET`` - are exactly the ones a
    paraphrase-tolerant metric would let slide.

    Matched on word boundaries, not as raw substrings: plain ``in`` would let
    ``"300 seconds"`` satisfy a requirement for ``30``, which is precisely the
    transposed-value error this metric exists to catch.
    """
    if not item.must_contain:
        return None
    found = sum(
        1 for requirement in item.must_contain if _satisfied(answer.text, requirement)
    )
    return found / len(item.must_contain)


def _satisfied(text: str, requirement: str | list[str]) -> bool:
    """One requirement: a value, or a list of acceptable spellings of it."""
    if isinstance(requirement, str):
        return contains_value(text, requirement)
    return any(contains_value(text, alternative) for alternative in requirement)


def contains_value(text: str, needle: str) -> bool:
    """Whether ``needle`` appears in ``text`` as a whole value."""
    needle = needle.strip()
    if not needle:
        return False

    # A digit edge needs more than \b: "." is a non-word character, so \b30\b
    # happily matches inside "1.30". Digits and dots are both excluded there.
    # A needle like "30%" or "(deprecated)" has no word edge to anchor at all.
    pattern = re.escape(needle)
    if needle[0].isdigit():
        pattern = r"(?<![\d.])" + pattern
    elif needle[0].isalnum() or needle[0] == "_":
        pattern = r"\b" + pattern

    if needle[-1].isdigit():
        pattern = pattern + r"(?![\d.])"
    elif needle[-1].isalnum() or needle[-1] == "_":
        pattern = pattern + r"\b"

    return re.search(pattern, text, re.IGNORECASE) is not None


def groundedness(answer: Answer) -> float:
    """Share of the answer's claims that are cited and supported.

    Reuses the Phase 5 verification report, so it means the same thing here as
    it does in production rather than being a second, differently-wrong
    definition of the same idea.
    """
    report = answer.verification
    if report is None or report.total_claims == 0:
        return 1.0
    unsupported = report.uncited_claims + report.unsupported_claims
    return max(0.0, (report.total_claims - unsupported) / report.total_claims)


def refusal_correct(answer: Answer, item: GoldenItem) -> bool:
    """Did the system refuse exactly when it should have?"""
    return answer.refused == (not item.answerable)


def score_answer(answer: Answer, item: GoldenItem) -> dict[str, float]:
    """Every generation metric for one item."""
    scores: dict[str, float] = {
        "citation_precision": citation_precision(answer),
        "groundedness": groundedness(answer),
        "refusal_correct": 1.0 if refusal_correct(answer, item) else 0.0,
    }

    if item.answerable:
        # Meaningless for a question with no answer: there is no source to
        # cite and no evidence to have retrieved.
        scores["citation_recall"] = citation_recall(answer, item)
        scores["context_recall"] = context_recall(answer, item)
        scores["answered"] = 0.0 if answer.refused else 1.0

    coverage = must_contain_coverage(answer, item)
    if coverage is not None:
        scores["must_contain_coverage"] = coverage

    return scores


def refusal_breakdown(
    results: Sequence[tuple[Answer, GoldenItem]],
) -> dict[str, float | int]:
    """Confusion matrix for the refusal decision.

    The single most informative table in the report: over-refusing looks like
    caution and destroys usefulness, under-refusing looks like helpfulness and
    is how a RAG system misleads people.
    """
    correctly_refused = sum(
        1 for answer, item in results if not item.answerable and answer.refused
    )
    wrongly_answered = sum(
        1 for answer, item in results if not item.answerable and not answer.refused
    )
    wrongly_refused = sum(
        1 for answer, item in results if item.answerable and answer.refused
    )
    correctly_answered = sum(
        1 for answer, item in results if item.answerable and not answer.refused
    )

    unanswerable = correctly_refused + wrongly_answered
    answerable = correctly_answered + wrongly_refused

    return {
        "correctly_refused": correctly_refused,
        "wrongly_answered": wrongly_answered,
        "wrongly_refused": wrongly_refused,
        "correctly_answered": correctly_answered,
        "refusal_recall": round(correctly_refused / unanswerable, 4) if unanswerable else 1.0,
        "answer_rate": round(correctly_answered / answerable, 4) if answerable else 1.0,
    }


# --------------------------------------------------------------------------


def _cited_sources(answer: Answer) -> list[Source]:
    if answer.verification is None:
        return list(answer.sources)
    cited = set(answer.verification.valid_citations)
    return [source for source in answer.sources if source.number in cited]


def _source_matches(expected: object, source: Source) -> bool:
    """Whether a prompt source satisfies an expected (file, page) label."""
    from pathlib import Path

    expected_file = getattr(expected, "source_file", "")
    expected_page = getattr(expected, "page", None)

    if Path(expected_file).name.lower() != Path(source.source_file).name.lower():
        return False
    if expected_page is None:
        return True

    # Sources carry a printed label ("p. 3", "pp. 3-5"); recover the bounds.
    digits = [int(part) for part in "".join(
        char if char.isdigit() else " " for char in source.page_label
    ).split()]
    if not digits:
        return False
    return digits[0] <= expected_page <= digits[-1]
