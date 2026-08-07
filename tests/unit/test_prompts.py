"""Prompt construction.

The prompt is the only thing making the model cite evidence and refuse without
it, so it is tested like code: the numbering contract, the exact refusal string,
attribution, and what happens when the context does not fit.
"""

from __future__ import annotations

import pytest

from askmydocs.config import GenerationConfig
from askmydocs.generation.prompts import (
    build_messages,
    build_sources,
    format_context,
    format_source,
    system_prompt,
)
from askmydocs.models import Candidate, Chunk


def _candidate(
    chunk_id: str,
    rank: int,
    text: str = "The request_timeout parameter defaults to 30 seconds.",
    page: int = 14,
    section: list[str] | None = None,
) -> Candidate:
    chunk = Chunk(
        chunk_id=chunk_id,
        doc_id="doc-1",
        source_file="handbook.pdf",
        doc_title="Service Handbook",
        text=text,
        embed_text=f"Service Handbook\n\n{text}",
        section_path=section if section is not None else ["4. Timeouts"],
        page_start=page,
        page_end=page,
        chunk_index=rank,
        token_count=len(text.split()),
    )
    return Candidate(
        chunk=chunk, fused_score=1.0 / rank, fused_rank=rank, ranks={"vector": rank}
    )


@pytest.fixture
def config() -> GenerationConfig:
    return GenerationConfig(max_context_tokens=6000)


# -- the numbering contract ------------------------------------------------


def test_sources_are_numbered_from_one_in_rank_order(config: GenerationConfig) -> None:
    sources = build_sources([_candidate("a", 1), _candidate("b", 2)], config)
    assert [s.number for s in sources] == [1, 2]
    assert [s.chunk_id for s in sources] == ["a", "b"]


def test_source_numbers_appear_in_the_context_block(config: GenerationConfig) -> None:
    context = format_context(build_sources([_candidate("a", 1), _candidate("b", 2)], config))
    assert context.startswith("[1]")
    assert "[2]" in context


def test_numbering_stays_contiguous_after_truncation() -> None:
    # A gap in the numbering would make the model cite a number that is not
    # there, or skip one that is.
    config = GenerationConfig(max_context_tokens=40)
    sources = build_sources([_candidate(str(i), i) for i in range(1, 6)], config)
    assert [s.number for s in sources] == list(range(1, len(sources) + 1))


def test_no_sources_produces_an_empty_context(config: GenerationConfig) -> None:
    assert build_sources([], config) == []
    assert format_context([]) == ""


# -- attribution -----------------------------------------------------------


def test_each_source_carries_file_page_and_section(config: GenerationConfig) -> None:
    block = format_source(build_sources([_candidate("a", 1)], config)[0])
    assert "handbook.pdf" in block
    assert "p. 14" in block
    assert "4. Timeouts" in block


def test_a_source_without_a_section_still_renders(config: GenerationConfig) -> None:
    source = build_sources([_candidate("a", 1, section=[])], config)[0]
    assert source.label == "handbook.pdf - p. 14"


def test_multi_page_chunks_show_a_page_range(config: GenerationConfig) -> None:
    candidate = _candidate("a", 1)
    candidate.chunk.page_end = 16
    assert "pp. 14-16" in format_source(build_sources([candidate], config)[0])


def test_chunk_text_is_included_verbatim(config: GenerationConfig) -> None:
    block = format_source(build_sources([_candidate("a", 1)], config)[0])
    assert "The request_timeout parameter defaults to 30 seconds." in block


def test_sources_are_visually_separated(config: GenerationConfig) -> None:
    context = format_context(build_sources([_candidate("a", 1), _candidate("b", 2)], config))
    assert "---" in context


# -- the instructions ------------------------------------------------------


def test_system_prompt_quotes_the_configured_refusal_verbatim() -> None:
    config = GenerationConfig(refusal_text="No idea, sorry.")
    assert "No idea, sorry." in system_prompt(config)


def test_system_prompt_demands_citations(config: GenerationConfig) -> None:
    prompt = system_prompt(config).lower()
    assert "[1]" in prompt
    assert "citation" in prompt


def test_system_prompt_forbids_inventing_source_numbers(
    config: GenerationConfig,
) -> None:
    assert "never invent" in system_prompt(config).lower()


def test_system_prompt_forbids_outside_knowledge(config: GenerationConfig) -> None:
    assert "outside knowledge" in system_prompt(config).lower()


def test_system_prompt_requires_verbatim_identifiers(config: GenerationConfig) -> None:
    assert "exactly as" in system_prompt(config).lower()


# -- message assembly ------------------------------------------------------


def test_messages_are_a_system_and_a_user_turn(config: GenerationConfig) -> None:
    messages = build_messages("How long?", build_sources([_candidate("a", 1)], config), config)
    assert [m["role"] for m in messages] == ["system", "user"]


def test_question_appears_in_the_user_message(config: GenerationConfig) -> None:
    messages = build_messages(
        "What is the default timeout?", build_sources([_candidate("a", 1)], config), config
    )
    assert "What is the default timeout?" in messages[1]["content"]


def test_sources_precede_the_question(config: GenerationConfig) -> None:
    # Instructions and evidence first, task last - the ordering models follow best.
    content = build_messages("Q?", build_sources([_candidate("a", 1)], config), config)[1][
        "content"
    ]
    assert content.index("SOURCES") < content.index("QUESTION")


def test_surrounding_whitespace_is_stripped_from_the_question(
    config: GenerationConfig,
) -> None:
    content = build_messages("  padded?  ", build_sources([_candidate("a", 1)], config), config)[
        1
    ]["content"]
    assert content.rstrip().endswith("padded?")


# -- the context budget ----------------------------------------------------


def test_lowest_ranked_sources_are_dropped_first() -> None:
    config = GenerationConfig(max_context_tokens=60)
    sources = build_sources([_candidate(str(i), i) for i in range(1, 6)], config)

    assert len(sources) < 5
    assert [s.chunk_id for s in sources] == [str(i) for i in range(1, len(sources) + 1)]


def test_the_top_source_is_kept_even_if_it_exceeds_the_budget() -> None:
    # Dropping everything would silently turn a good answer into a refusal.
    config = GenerationConfig(max_context_tokens=1)
    sources = build_sources([_candidate("a", 1, text="word " * 500)], config)
    assert len(sources) == 1


def test_a_generous_budget_keeps_everything() -> None:
    config = GenerationConfig(max_context_tokens=100_000)
    assert len(build_sources([_candidate(str(i), i) for i in range(1, 8)], config)) == 7


# -- provenance carried through --------------------------------------------


def test_rerank_score_is_carried_onto_the_source(config: GenerationConfig) -> None:
    candidate = _candidate("a", 1)
    candidate.rerank_score = 5.25
    assert build_sources([candidate], config)[0].rerank_score == 5.25


def test_chunk_id_is_preserved_for_verification(config: GenerationConfig) -> None:
    # Phase 5 verifies citations against exactly this mapping.
    sources = build_sources([_candidate("chunk-xyz", 1)], config)
    assert sources[0].chunk_id == "chunk-xyz"
