from __future__ import annotations

from collections.abc import Callable

import pytest

from askmydocs.config import ChunkingConfig
from askmydocs.ingestion.chunker import Chunker, heuristic_token_count
from askmydocs.models import Block, Chunk, ParsedDocument, ParsedPage, Section, StructureSource


def _document(title: str = "Ops Handbook") -> ParsedDocument:
    return ParsedDocument(
        doc_id="doc123",
        source_path="ops.pdf",
        filename="ops.pdf",
        title=title,
        pages=[ParsedPage(page_no=1)],
    )


def _section(path: list[str], text: str, page: int = 1, kind: str = "prose") -> Section:
    return Section(
        path=path,
        level=len(path),
        blocks=[Block(kind=kind, text=text, page_no=page)],  # type: ignore[arg-type]
    )


def _words(n: int, word: str = "alpha") -> str:
    return " ".join(f"{word}{i}" for i in range(n))


@pytest.fixture
def chunker(word_token_counter: Callable[[str], int]) -> Chunker:
    config = ChunkingConfig(
        chunk_tokens=50,
        chunk_overlap_tokens=10,
        min_section_tokens=8,
        prepend_breadcrumb=True,
        separators=["\n\n", "\n", ". ", " "],
    )
    return Chunker(config, token_counter=word_token_counter)


# -- size accounting -------------------------------------------------------


def test_chunks_respect_the_token_budget(chunker: Chunker) -> None:
    section = _section(["1. Overview"], _words(400))
    chunks = chunker.chunk_document(_document(), [section])

    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk.token_count <= chunker.config.chunk_tokens


def test_breadcrumb_budget_keeps_embed_text_under_the_limit(
    word_token_counter: Callable[[str], int],
) -> None:
    # A deep breadcrumb must shrink the body, not push embed_text over the
    # model's hard limit - bge truncates silently past 512 tokens.
    config = ChunkingConfig(chunk_tokens=50, chunk_overlap_tokens=5, min_section_tokens=0)
    chunker = Chunker(config, token_counter=word_token_counter)
    deep_path = ["Section One", "Subsection Two", "Sub Sub Section Three"]

    chunks = chunker.chunk_document(_document(), [_section(deep_path, _words(300))])

    for chunk in chunks:
        assert word_token_counter(chunk.embed_text) <= config.chunk_tokens


def test_overlap_is_applied_between_adjacent_chunks(chunker: Chunker) -> None:
    chunks = chunker.chunk_document(_document(), [_section(["S"], _words(300))])
    first_words = set(chunks[0].text.split())
    second_words = set(chunks[1].text.split())
    assert first_words & second_words, "adjacent chunks should share overlap tokens"


def test_short_document_produces_a_single_chunk(chunker: Chunker) -> None:
    chunks = chunker.chunk_document(_document(), [_section(["S"], _words(20))])
    assert len(chunks) == 1


# -- section boundaries ----------------------------------------------------


def test_chunks_never_span_two_sections(chunker: Chunker) -> None:
    sections = [
        _section(["1. Networking"], "Networking uses BGP for route exchange between peers."),
        _section(["2. Storage"], "Storage replicates every object across three availability zones."),
    ]
    chunks = chunker.chunk_document(_document(), sections)

    for chunk in chunks:
        assert not ("BGP" in chunk.text and "availability zones" in chunk.text)
    assert {tuple(c.section_path) for c in chunks} == {("1. Networking",), ("2. Storage",)}


def test_tiny_sections_are_merged_forward(chunker: Chunker) -> None:
    sections = [
        _section(["1. Scope"], "Applies to all clusters."),  # under min_section_tokens
        _section(["2. Detail"], _words(30)),
    ]
    chunks = chunker.chunk_document(_document(), sections)

    assert len(chunks) == 1
    assert chunks[0].section_path == ["2. Detail"]
    # The absorbed heading survives in the body so it stays searchable.
    assert "1. Scope" in chunks[0].text
    assert "Applies to all clusters" in chunks[0].text


def test_merging_is_skipped_when_disabled(word_token_counter: Callable[[str], int]) -> None:
    config = ChunkingConfig(chunk_tokens=50, chunk_overlap_tokens=5, min_section_tokens=0)
    chunker = Chunker(config, token_counter=word_token_counter)
    sections = [_section(["A"], "Short one."), _section(["B"], "Short two.")]
    assert len(chunker.chunk_document(_document(), sections)) == 2


def test_all_tiny_sections_still_produce_a_chunk(chunker: Chunker) -> None:
    sections = [_section(["A"], "One."), _section(["B"], "Two."), _section(["C"], "Three.")]
    chunks = chunker.chunk_document(_document(), sections)
    assert len(chunks) == 1
    assert "One." in chunks[0].text and "Three." in chunks[0].text


# -- tables ----------------------------------------------------------------


def test_table_is_kept_whole_even_when_oversized(
    word_token_counter: Callable[[str], int],
) -> None:
    config = ChunkingConfig(chunk_tokens=10, chunk_overlap_tokens=2, min_section_tokens=0)
    chunker = Chunker(config, token_counter=word_token_counter)
    table = "| Parameter | Default |\n|---|---|\n" + "\n".join(
        f"| param_{i} | value_{i} |" for i in range(30)
    )
    section = Section(
        path=["4. Reference"], level=1, blocks=[Block(kind="table", text=table, page_no=7)]
    )

    chunks = chunker.chunk_document(_document(), [section])

    assert len(chunks) == 1
    assert chunks[0].content_type == "table"
    assert "param_29" in chunks[0].text


def test_table_and_prose_stay_in_separate_chunks(chunker: Chunker) -> None:
    section = Section(
        path=["4. Reference"],
        level=1,
        blocks=[
            Block(kind="prose", text="The table below lists the timeouts.", page_no=3),
            Block(kind="table", text="| a | b |\n|---|---|\n| 1 | 2 |", page_no=3),
            Block(kind="prose", text="Values are given in seconds throughout.", page_no=3),
        ],
    )
    chunks = chunker.chunk_document(_document(), [section])

    kinds = [c.content_type for c in chunks]
    assert kinds == ["prose", "table", "prose"]


# -- metadata --------------------------------------------------------------


def test_embed_text_carries_title_and_breadcrumb(chunker: Chunker) -> None:
    section = _section(["2. Rollout", "2.2 Rollback"], _words(20))
    chunk = chunker.chunk_document(_document("Ops Handbook"), [section])[0]

    assert chunk.embed_text.startswith("Ops Handbook > 2. Rollout > 2.2 Rollback")
    assert chunk.text in chunk.embed_text
    assert not chunk.text.startswith("Ops Handbook")


def test_breadcrumb_can_be_disabled(word_token_counter: Callable[[str], int]) -> None:
    config = ChunkingConfig(chunk_tokens=50, chunk_overlap_tokens=5, prepend_breadcrumb=False)
    chunker = Chunker(config, token_counter=word_token_counter)
    chunk = chunker.chunk_document(_document(), [_section(["A"], _words(20))])[0]
    assert chunk.embed_text == chunk.text


def test_page_range_tracks_the_source_pages(chunker: Chunker) -> None:
    section = Section(
        path=["1. Intro"],
        level=1,
        blocks=[
            Block(kind="prose", text=_words(30, "page4"), page_no=4),
            Block(kind="prose", text=_words(30, "page5"), page_no=5),
        ],
    )
    chunks = chunker.chunk_document(_document(), [section])

    assert min(c.page_start for c in chunks) == 4
    assert max(c.page_end for c in chunks) == 5
    for chunk in chunks:
        assert chunk.page_start <= chunk.page_end


def test_chunk_ids_are_stable_across_runs(chunker: Chunker) -> None:
    sections = [_section(["1. Overview"], _words(120))]
    first = chunker.chunk_document(_document(), sections)
    second = chunker.chunk_document(_document(), sections)
    assert [c.chunk_id for c in first] == [c.chunk_id for c in second]


def test_chunk_ids_change_when_content_changes(chunker: Chunker) -> None:
    a = chunker.chunk_document(_document(), [_section(["S"], "Original text here.")])[0]
    b = chunker.chunk_document(_document(), [_section(["S"], "Different text here.")])[0]
    assert a.chunk_id != b.chunk_id


def test_chunk_indices_are_sequential(chunker: Chunker) -> None:
    sections = [_section(["A"], _words(120)), _section(["B"], _words(120))]
    chunks = chunker.chunk_document(_document(), sections)
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))


def test_page_fallback_sections_are_never_merged(chunker: Chunker) -> None:
    # Each fallback section is one page. Merging them would produce a chunk
    # spanning pages, defeating the point of page-bounded splitting.
    sections = [
        Section(path=[], level=0, blocks=[Block(kind="prose", text="Page one.", page_no=1)]),
        Section(path=[], level=0, blocks=[Block(kind="prose", text="Page two.", page_no=2)]),
    ]
    chunks = chunker.chunk_document(_document(), sections, StructureSource.PAGE_FALLBACK)

    assert len(chunks) == 2
    assert [(c.page_start, c.page_end) for c in chunks] == [(1, 1), (2, 2)]


def test_structure_source_is_recorded(chunker: Chunker) -> None:
    chunks = chunker.chunk_document(
        _document(), [_section([], _words(30))], StructureSource.PAGE_FALLBACK
    )
    assert all(c.structure_source is StructureSource.PAGE_FALLBACK for c in chunks)


def test_page_label_formats_single_and_multi_page(chunker: Chunker) -> None:
    chunk = Chunk(
        chunk_id="x",
        doc_id="d",
        source_file="f.pdf",
        doc_title="T",
        text="t",
        embed_text="t",
        page_start=3,
        page_end=3,
        chunk_index=0,
        token_count=1,
    )
    assert chunk.page_label == "p. 3"
    assert chunk.model_copy(update={"page_end": 5}).page_label == "pp. 3-5"


# -- empty input -----------------------------------------------------------


def test_empty_sections_produce_no_chunks(chunker: Chunker) -> None:
    assert chunker.chunk_document(_document(), []) == []
    assert chunker.chunk_document(_document(), [_section(["A"], "   ")]) == []


def test_heuristic_token_count_is_sane() -> None:
    assert heuristic_token_count("") == 0
    assert heuristic_token_count("a" * 400) == 100
