"""Prompt construction.

The prompt is the only mechanism that makes the model cite its evidence and
refuse when the evidence is missing, so it is treated as source code: kept in
one place, versioned, and unit-tested rather than tuned by hand in a notebook.

Three properties matter:

* **Sources are numbered, and the numbering is the contract.** The model can
  only refer to a chunk by its number, and Phase 5 verifies citations against
  that same mapping. Nothing recomputes it.
* **The refusal string is exact.** Verification and evaluation both key off it,
  so it comes from config and is quoted verbatim in the instructions.
* **Attribution travels with the text.** Each source carries its file, page and
  section, so the model can be told to cite precisely and the UI can show the
  user where an answer came from.
"""

from __future__ import annotations

from collections.abc import Sequence

from ..config import GenerationConfig
from ..logging_setup import get_logger
from ..models import Candidate, Source
from ..tokens import heuristic_token_count

log = get_logger(__name__)

SOURCE_SEPARATOR = "\n\n---\n\n"

SYSTEM_PROMPT = """\
You are a precise technical documentation assistant. You answer questions using \
ONLY the numbered sources provided in the SOURCES section.

Follow these rules exactly:

1. Every factual claim in your answer must carry a citation to the source it came \
from, written as [1], [2], and so on. Place the citation at the end of the sentence \
or clause it supports. When a claim draws on more than one source, cite each: [1][3].
2. Only use source numbers that appear in the SOURCES section. Never invent a number \
and never cite a source that is not listed.
3. If the sources do not contain enough information to answer the question, reply \
with exactly this sentence and nothing else:
{refusal_text}
4. If the sources answer only part of the question, answer that part with citations, \
then state plainly which part the sources do not cover.
5. Reproduce parameter names, identifiers, error codes and numeric values exactly as \
they appear in the sources. Do not paraphrase them and do not convert units.
6. Do not use outside knowledge, do not speculate, and do not add background the \
sources do not contain.
7. Be concise. Give the shortest answer that is complete and fully cited.\
"""

USER_PROMPT = """\
SOURCES
{context}

QUESTION
{question}\
"""


def build_sources(
    candidates: Sequence[Candidate], config: GenerationConfig
) -> list[Source]:
    """Number the retrieved chunks and drop any that will not fit the budget.

    Candidates arrive best-first, so truncation removes the least relevant
    material. Dropping is logged: a silently shortened context is a plausible
    cause of an unexpectedly incomplete answer.
    """
    sources: list[Source] = []
    used_tokens = 0
    dropped = 0

    for position, candidate in enumerate(candidates, start=1):
        chunk = candidate.chunk
        source = Source(
            number=position,
            chunk_id=chunk.chunk_id,
            doc_title=chunk.doc_title,
            source_file=chunk.source_file,
            page_label=chunk.page_label,
            section_path=list(chunk.section_path),
            text=chunk.text,
            rerank_score=candidate.rerank_score,
        )
        cost = heuristic_token_count(format_source(source))
        if sources and used_tokens + cost > config.max_context_tokens:
            dropped += 1
            continue
        sources.append(source)
        used_tokens += cost

    if dropped:
        log.warning(
            "context_truncated",
            kept=len(sources),
            dropped=dropped,
            budget_tokens=config.max_context_tokens,
        )

    # Renumber so the model always sees a contiguous 1..n with no gaps.
    for position, source in enumerate(sources, start=1):
        source.number = position
    return sources


def format_source(source: Source) -> str:
    """One numbered source block, header first."""
    return f"[{source.number}] {source.label}\n{source.text.strip()}"


def format_context(sources: Sequence[Source]) -> str:
    return SOURCE_SEPARATOR.join(format_source(source) for source in sources)


def build_messages(
    question: str, sources: Sequence[Source], config: GenerationConfig
) -> list[dict[str, str]]:
    """The chat messages sent to the model."""
    return [
        {"role": "system", "content": system_prompt(config)},
        {
            "role": "user",
            "content": USER_PROMPT.format(
                context=format_context(sources), question=question.strip()
            ),
        },
    ]


def system_prompt(config: GenerationConfig) -> str:
    return SYSTEM_PROMPT.format(refusal_text=config.refusal_text)
