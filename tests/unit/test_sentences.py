"""Sentence segmentation and claim detection.

Every false positive here becomes a spurious "uncited claim" warning the user
learns to ignore, and every false negative lets an unsupported assertion pass -
so the awkward cases in technical prose are pinned down individually.
"""

from __future__ import annotations

import pytest

from askmydocs.verification.sentences import (
    is_claim,
    mask_code,
    parse_citations,
    split_sentences,
)


def texts(text: str) -> list[str]:
    return [sentence.text for sentence in split_sentences(text)]


# -- citation parsing ------------------------------------------------------


def test_simple_citations_are_parsed() -> None:
    assert parse_citations("The default is 30s [1].") == [1]


def test_adjacent_citations_are_parsed() -> None:
    assert parse_citations("Both agree [1][2].") == [1, 2]


def test_grouped_citations_are_parsed() -> None:
    assert parse_citations("Both agree [1, 2].") == [1, 2]
    assert parse_citations("Both agree [1,2].") == [1, 2]


def test_repeated_citations_are_all_counted() -> None:
    # Citation precision must be computed over what the model actually wrote.
    assert parse_citations("A [1]. B [1]. C [1].") == [1, 1, 1]


def test_multi_digit_citations_are_parsed() -> None:
    assert parse_citations("See [12].") == [12]


def test_text_without_citations_yields_none() -> None:
    assert parse_citations("No citations here at all.") == []


def test_non_numeric_brackets_are_not_citations() -> None:
    assert parse_citations("See [note] and [TODO].") == []


# -- code is not a citation ------------------------------------------------


def test_array_indexing_in_a_code_span_is_not_a_citation() -> None:
    # A documentation assistant emits code constantly; `items[0]` must not
    # register as a citation to source 0.
    assert parse_citations("Use `items[0]` to read the first entry [1].") == [1]


def test_array_indexing_in_a_fenced_block_is_not_a_citation() -> None:
    text = "Example:\n\n```python\nvalues[1]\nvalues[2]\n```\n\nThat is the pattern [3]."
    assert parse_citations(text) == [3]


def test_masking_preserves_length() -> None:
    # Offsets into the masked text must still address the original.
    original = "Use `arr[0]` here."
    assert len(mask_code(original)) == len(original)


# -- sentence splitting ----------------------------------------------------


def test_sentences_split_on_terminal_punctuation() -> None:
    assert texts("First one. Second one! Third one?") == [
        "First one.",
        "Second one!",
        "Third one?",
    ]


def test_decimals_do_not_split_a_sentence() -> None:
    assert texts("The timeout is 30.5 seconds by default.") == [
        "The timeout is 30.5 seconds by default."
    ]


def test_version_numbers_do_not_split_a_sentence() -> None:
    assert texts("Upgrade to v2.1.0 before enabling it.") == [
        "Upgrade to v2.1.0 before enabling it."
    ]


def test_filenames_do_not_split_a_sentence() -> None:
    assert texts("Edit config.yaml and restart.") == ["Edit config.yaml and restart."]


@pytest.mark.parametrize("abbreviation", ["e.g.", "i.e.", "etc.", "vs."])
def test_abbreviations_do_not_split_a_sentence(abbreviation: str) -> None:
    assert len(split_sentences(f"Use a timeout, {abbreviation} 30 seconds, always.")) == 1


def test_newlines_end_a_sentence() -> None:
    # Bullets and headings often carry no terminal punctuation.
    assert texts("- First point [1]\n- Second point [2]") == [
        "- First point [1]",
        "- Second point [2]",
    ]


def test_blank_lines_are_ignored() -> None:
    assert texts("First.\n\n\nSecond.") == ["First.", "Second."]


def test_empty_text_yields_no_sentences() -> None:
    assert split_sentences("") == []
    assert split_sentences("   \n  ") == []


def test_text_without_terminal_punctuation_is_one_sentence() -> None:
    assert texts("A statement with no full stop") == ["A statement with no full stop"]


# -- attaching citations to sentences --------------------------------------


def test_citation_inside_a_sentence_belongs_to_it() -> None:
    sentences = split_sentences("The default is 30s [1]. Retries are capped [2].")
    assert [s.citations for s in sentences] == [[1], [2]]


def test_citation_after_the_full_stop_belongs_to_the_same_sentence() -> None:
    # Models write both "X [1]." and "X. [1]" - the citation refers backwards.
    sentences = split_sentences("The default is 30s. [1] Retries are capped. [2]")
    assert [s.citations for s in sentences] == [[1], [2]]
    assert len(sentences) == 2


def test_a_citation_on_its_own_line_folds_into_the_previous_sentence() -> None:
    sentences = split_sentences("The default is 30 seconds\n[1]")
    assert len(sentences) == 1
    assert sentences[0].citations == [1]


def test_an_uncited_sentence_reports_no_citations() -> None:
    sentences = split_sentences("This one has a source [1]. This one does not.")
    assert sentences[0].is_cited is True
    assert sentences[1].is_cited is False


# -- claim detection -------------------------------------------------------


def _only(text: str):
    return split_sentences(text)[0]


def test_an_assertion_is_a_claim() -> None:
    assert is_claim(_only("The default request timeout is 30 seconds [1].")) is True


def test_a_short_fragment_is_not_a_claim() -> None:
    assert is_claim(_only("Yes.")) is False


def test_a_list_lead_in_is_not_a_claim() -> None:
    # The citations belong on the items that follow, not on the colon line.
    assert is_claim(_only("The timeouts are as follows:")) is False


def test_a_question_is_not_a_claim() -> None:
    assert is_claim(_only("What is the default timeout here?")) is False


def test_a_sentence_that_is_only_a_citation_is_not_a_claim() -> None:
    assert is_claim(_only("[1]")) is False


@pytest.mark.parametrize(
    "text",
    [
        "The sources do not cover the retry count.",
        "The provided sources don't mention parental leave.",
        "I don't have enough information to answer that.",
        "That detail is not specified in the provided sources.",
        "No information is available about the rollout schedule.",
    ],
)
def test_statements_about_missing_evidence_are_not_claims(text: str) -> None:
    # Flagging these as uncited would penalise exactly the honest behaviour the
    # prompt asks for.
    assert is_claim(_only(text)) is False


def test_citing_the_sources_as_evidence_is_still_a_claim() -> None:
    # "The sources do not cover X" is meta; "According to the sources, X is 30s"
    # is an assertion that needs support.
    assert is_claim(_only("According to the sources, the timeout is 30 seconds [1].")) is True


def test_a_bulleted_assertion_is_a_claim() -> None:
    assert is_claim(_only("- The retry budget is shared across the service [2]")) is True


def test_the_word_minimum_is_configurable() -> None:
    sentence = _only("Timeout is 30s [1].")
    assert is_claim(sentence, min_words=3) is True
    assert is_claim(sentence, min_words=10) is False


def test_citations_do_not_count_towards_the_word_minimum() -> None:
    # Otherwise "[1][2][3][4] ok" would qualify as a claim on citation count.
    assert is_claim(_only("It works [1][2][3][4].")) is False
