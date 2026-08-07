from __future__ import annotations

from pathlib import Path

import pytest

from askmydocs.evaluation.golden import (
    ExpectedSource,
    GoldenItem,
    GoldenSet,
    coverage_report,
    load_golden_set,
    write_golden_set,
)
from askmydocs.models import Chunk


def _chunk(source_file: str = "handbook.pdf", page_start: int = 3, page_end: int = 3) -> Chunk:
    return Chunk(
        chunk_id="c1",
        doc_id="d1",
        source_file=source_file,
        doc_title="Handbook",
        text="body",
        embed_text="body",
        page_start=page_start,
        page_end=page_end,
        chunk_index=0,
        token_count=1,
    )


def _item(**kwargs: object) -> GoldenItem:
    defaults: dict[str, object] = {
        "id": "q001",
        "question": "What is the default request timeout?",
        "expected_sources": [ExpectedSource(source_file="handbook.pdf", page=3)],
    }
    return GoldenItem(**{**defaults, **kwargs})  # type: ignore[arg-type]


# -- matching expected sources ---------------------------------------------


def test_a_chunk_on_the_expected_page_matches() -> None:
    assert _item().is_relevant(_chunk(page_start=3, page_end=3))


def test_a_chunk_spanning_the_expected_page_matches() -> None:
    # Chunks can cross page boundaries, so containment rather than equality.
    assert _item().is_relevant(_chunk(page_start=2, page_end=4))


def test_a_chunk_on_another_page_does_not_match() -> None:
    assert not _item().is_relevant(_chunk(page_start=9, page_end=9))


def test_a_chunk_from_another_file_does_not_match() -> None:
    assert not _item().is_relevant(_chunk(source_file="other.pdf"))


def test_omitting_the_page_accepts_the_whole_document() -> None:
    item = _item(expected_sources=[ExpectedSource(source_file="handbook.pdf")])
    assert item.is_relevant(_chunk(page_start=42, page_end=42))


def test_matching_ignores_directories_and_case() -> None:
    # Labels are written by hand from the PDF; a path prefix should not break them.
    item = _item(expected_sources=[ExpectedSource(source_file="docs/Handbook.PDF", page=3)])
    assert item.is_relevant(_chunk(source_file="handbook.pdf"))


def test_any_expected_source_matching_is_enough() -> None:
    item = _item(
        expected_sources=[
            ExpectedSource(source_file="other.pdf", page=1),
            ExpectedSource(source_file="handbook.pdf", page=3),
        ]
    )
    assert item.is_relevant(_chunk())


# -- relevance vectors -----------------------------------------------------


def test_each_expected_source_can_only_be_found_once() -> None:
    # Chunking regularly puts several chunks on one page - a prose lead-in and
    # the table it introduces. Counting each as a separate hit would let the
    # relevant count exceed the number that exist and push nDCG above 1.0.
    item = _item(expected_sources=[ExpectedSource(source_file="handbook.pdf", page=3)])
    chunks = [_chunk(), _chunk(), _chunk()]

    assert item.relevance_vector(chunks) == [True, False, False]


def test_the_highest_ranked_chunk_claims_the_source() -> None:
    item = _item()
    chunks = [_chunk(source_file="other.pdf"), _chunk(), _chunk()]

    assert item.relevance_vector(chunks) == [False, True, False]


def test_distinct_expected_sources_are_counted_separately() -> None:
    item = _item(
        expected_sources=[
            ExpectedSource(source_file="handbook.pdf", page=3),
            ExpectedSource(source_file="reference.pdf", page=1),
        ]
    )
    chunks = [_chunk(), _chunk("reference.pdf", 1, 1), _chunk()]

    assert item.relevance_vector(chunks) == [True, True, False]


def test_relevant_count_never_exceeds_the_number_of_expected_sources() -> None:
    item = _item()
    flags = item.relevance_vector([_chunk() for _ in range(10)])
    assert sum(flags) <= len(item.expected_sources)


def test_a_chunk_covering_two_sources_at_once_counts_once() -> None:
    # One chunk spanning pages 2-4 satisfies labels for both pages; it is a
    # single retrieved result, so it contributes one hit.
    item = _item(
        expected_sources=[
            ExpectedSource(source_file="handbook.pdf", page=2),
            ExpectedSource(source_file="handbook.pdf", page=4),
        ]
    )
    assert item.relevance_vector([_chunk(page_start=2, page_end=4)]) == [True]


def test_an_empty_ranking_yields_an_empty_vector() -> None:
    assert _item().relevance_vector([]) == []


# -- validation ------------------------------------------------------------


def test_an_answerable_item_requires_a_source() -> None:
    # Without one it can never score above zero, silently dragging recall down.
    with pytest.raises(ValueError, match="expected source"):
        GoldenItem(id="q1", question="Anything?", expected_sources=[])


def test_an_unanswerable_item_must_not_have_sources() -> None:
    with pytest.raises(ValueError, match="cannot have expected_sources"):
        GoldenItem(
            id="q1",
            question="Anything?",
            answerable=False,
            expected_sources=[ExpectedSource(source_file="handbook.pdf")],
        )


def test_an_unanswerable_item_needs_no_sources() -> None:
    item = GoldenItem(id="q1", question="Parental leave policy?", answerable=False)
    assert item.answerable is False


def test_an_empty_question_is_rejected() -> None:
    with pytest.raises(ValueError):
        GoldenItem(id="q1", question="", answerable=False)


# -- loading ---------------------------------------------------------------


def test_a_set_round_trips_through_jsonl(tmp_path: Path) -> None:
    items = [_item(), GoldenItem(id="q002", question="Unrelated?", answerable=False)]
    path = tmp_path / "golden.jsonl"

    write_golden_set(path, items)
    loaded = load_golden_set(path)

    assert len(loaded) == 2
    assert loaded.items[0].expected_sources[0].page == 3
    assert loaded.items[1].answerable is False


def test_blank_lines_and_comments_are_skipped(tmp_path: Path) -> None:
    path = tmp_path / "golden.jsonl"
    path.write_text(
        "// a comment\n\n" + _item().model_dump_json() + "\n\n", encoding="utf-8"
    )
    assert len(load_golden_set(path)) == 1


def test_a_byte_order_mark_does_not_break_the_load(tmp_path: Path) -> None:
    # Windows editors and PowerShell prepend one readily. Losing an afternoon
    # to an invisible byte is not an acceptable failure mode for a file people
    # hand-author.
    path = tmp_path / "golden.jsonl"
    path.write_bytes(b"\xef\xbb\xbf" + _item().model_dump_json().encode("utf-8") + b"\n")

    assert len(load_golden_set(path)) == 1


def test_a_malformed_line_fails_the_load(tmp_path: Path) -> None:
    # Skipping it would shrink the set silently and make metrics improve for
    # no reason at all.
    path = tmp_path / "golden.jsonl"
    path.write_text(_item().model_dump_json() + "\nnot json\n", encoding="utf-8")

    with pytest.raises(ValueError, match=r"not a valid golden item"):
        load_golden_set(path)


def test_duplicate_ids_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "golden.jsonl"
    line = _item().model_dump_json()
    path.write_text(f"{line}\n{line}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate id"):
        load_golden_set(path)


def test_a_missing_file_is_a_clear_error(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_golden_set(tmp_path / "nope.jsonl")


# -- stats and coverage ----------------------------------------------------


def test_stats_split_answerable_from_unanswerable() -> None:
    golden = GoldenSet(
        items=[
            _item(),
            GoldenItem(id="q002", question="Unrelated?", answerable=False),
            GoldenItem(id="q003", question="Also unrelated?", answerable=False),
        ]
    )
    assert golden.stats() == {
        "total": 3,
        "answerable": 1,
        "unanswerable": 2,
        "with_must_contain": 0,
    }


def test_coverage_flags_labels_that_are_not_in_the_index() -> None:
    # A label pointing at a page that was never ingested scores as a permanent
    # retrieval miss and looks exactly like a retrieval bug.
    golden = GoldenSet(
        items=[_item(expected_sources=[ExpectedSource(source_file="missing.pdf", page=1)])]
    )
    report = coverage_report(golden, [_chunk()])

    assert report["unreachable"] == ["q001: missing.pdf p.1"]


def test_coverage_is_clean_when_every_label_resolves() -> None:
    report = coverage_report(GoldenSet(items=[_item()]), [_chunk()])
    assert report["unreachable"] == []
    assert report["expected_sources"] == 1
