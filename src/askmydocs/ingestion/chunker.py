"""Section-aware chunking.

Two rules drive everything here:

* **Chunks never cross a section boundary.** Splitting purely by size merges
  the end of one topic with the start of the next, which is the single most
  common cause of confidently wrong RAG answers.
* **Size is measured in tokens, not characters.** bge-base-en-v1.5 truncates at
  512 tokens silently; character-based sizing overflows on dense technical
  prose and you lose the tail of the chunk without any error.

The token counter is injected, so tests run without downloading a tokenizer and
ingestion never imports torch.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from functools import lru_cache
from typing import Any

from langchain_text_splitters import RecursiveCharacterTextSplitter

from ..config import ChunkingConfig
from ..logging_setup import get_logger
from ..models import Block, Chunk, ParsedDocument, Section, StructureSource
from ..tokens import heuristic_token_count

log = get_logger(__name__)

TokenCounter = Callable[[str], int]

_WS = re.compile(r"[ \t]+")
_BLANKS = re.compile(r"\n{3,}")
#: Never let the breadcrumb prefix eat more than this share of the budget.
_MAX_BREADCRUMB_SHARE = 0.5


class Chunker:
    """Turns sections into retrievable chunks."""

    def __init__(self, config: ChunkingConfig, token_counter: TokenCounter | None = None) -> None:
        self.config = config
        self.count_tokens: TokenCounter = token_counter or default_token_counter(
            config.tokenizer_model
        )
        self._splitters: dict[int, RecursiveCharacterTextSplitter] = {}

    # -- public API ------------------------------------------------------

    def chunk_document(
        self,
        document: ParsedDocument,
        sections: list[Section],
        structure_source: StructureSource = StructureSource.HEADINGS,
    ) -> list[Chunk]:
        chunks: list[Chunk] = []
        index = 0

        # In fallback mode a section *is* a page. Merging undersized ones would
        # produce chunks spanning pages, which is exactly what page-bounded
        # splitting exists to prevent - so citations stay exact to one page.
        if structure_source is StructureSource.PAGE_FALLBACK:
            prepared = list(sections)
        else:
            prepared = self._merge_small_sections(sections)

        for section in prepared:
            for text, content_type, page_start, page_end in self._split_section(
                section, document.title
            ):
                clean = _normalise(text)
                if not clean:
                    continue
                chunk = self._build_chunk(
                    document=document,
                    section=section,
                    text=clean,
                    content_type=content_type,
                    page_start=page_start,
                    page_end=page_end,
                    index=index,
                    structure_source=structure_source,
                )
                chunks.append(chunk)
                index += 1

        log.info(
            "document_chunked",
            doc_id=document.doc_id,
            sections=len(sections),
            chunks=len(chunks),
            structure_source=structure_source.value,
        )
        return chunks

    # -- section handling ------------------------------------------------

    def _merge_small_sections(self, sections: list[Section]) -> list[Section]:
        """Fold undersized sections into the following one.

        A two-line section becomes a chunk that matches keyword queries on
        almost no evidence, so it crowds out real answers. Merging forward
        (rather than backward) keeps at most ``min_section_tokens`` of content
        labelled under a neighbouring heading, and the absorbed heading text is
        preserved inline so it stays searchable.
        """
        if self.config.min_section_tokens <= 0 or not sections:
            return list(sections)

        merged: list[Section] = []
        carry: list[Block] = []

        for position, section in enumerate(sections):
            combined = carry + list(section.blocks)
            is_last = position == len(sections) - 1
            size = self.count_tokens(_blocks_text(combined))

            if not is_last and size < self.config.min_section_tokens:
                own = list(section.blocks)
                if section.path:
                    own.insert(
                        0,
                        Block(kind="prose", text=section.path[-1], page_no=section.page_start),
                    )
                carry = carry + own
                continue

            carry = []
            merged.append(Section(path=section.path, level=section.level, blocks=combined))

        if carry:  # every section was undersized
            merged.append(Section(path=sections[-1].path, level=sections[-1].level, blocks=carry))

        if len(merged) != len(sections):
            log.debug("sections_merged", before=len(sections), after=len(merged))
        return merged

    def _split_section(
        self, section: Section, doc_title: str
    ) -> list[tuple[str, str, int, int]]:
        """Split one section, preserving document order and table integrity.

        Returns ``(text, content_type, page_start, page_end)`` tuples.
        """
        pieces: list[tuple[str, str, int, int]] = []
        prose_run: list[Block] = []
        splitter = self._splitter_for(self._breadcrumb_budget(doc_title, section.path))

        def flush_prose() -> None:
            if not prose_run:
                return
            pieces.extend(self._split_prose(prose_run, splitter))
            prose_run.clear()

        for block in section.blocks:
            if block.kind == "table" and self.config.keep_tables_atomic:
                flush_prose()
                # Emitted whole even when oversized: half a table is worse than
                # a long one, because the header row carries the column meaning.
                if self.count_tokens(block.text) > self.config.chunk_tokens:
                    log.warning(
                        "oversized_table_kept_whole",
                        page_no=block.page_no,
                        tokens=self.count_tokens(block.text),
                        limit=self.config.chunk_tokens,
                    )
                pieces.append((block.text, "table", block.page_no, block.page_no))
            else:
                prose_run.append(block)

        flush_prose()
        return pieces

    def _split_prose(
        self, blocks: list[Block], splitter: RecursiveCharacterTextSplitter
    ) -> list[tuple[str, str, int, int]]:
        """Split a run of paragraphs, tracking which pages each piece came from."""
        text, offsets = _concat_with_offsets(blocks)
        if not text.strip():
            return []

        results: list[tuple[str, str, int, int]] = []
        cursor = 0
        for piece in splitter.split_text(text):
            if not piece.strip():
                continue
            start = _locate(text, piece, cursor)
            end = start + len(piece)
            cursor = max(cursor, start + 1)
            page_start, page_end = _page_range(offsets, start, end)
            results.append((piece, "prose", page_start, page_end))
        return results

    # -- chunk construction ----------------------------------------------

    def _build_chunk(
        self,
        *,
        document: ParsedDocument,
        section: Section,
        text: str,
        content_type: str,
        page_start: int,
        page_end: int,
        index: int,
        structure_source: StructureSource,
    ) -> Chunk:
        embed_text = self._embed_text(document.title, section.path, text)
        return Chunk(
            chunk_id=Chunk.make_id(document.doc_id, section.path, index, text),
            doc_id=document.doc_id,
            source_file=document.filename,
            doc_title=document.title,
            text=text,
            embed_text=embed_text,
            section_path=list(section.path),
            heading_level=section.level,
            page_start=page_start,
            page_end=page_end,
            chunk_index=index,
            token_count=self.count_tokens(text),
            content_type="table" if content_type == "table" else "prose",
            structure_source=structure_source,
        )

    def _prefix(self, doc_title: str, section_path: list[str]) -> str:
        """The breadcrumb prepended to embed_text, e.g. 'Deploy Guide > 4.2 Timeouts'."""
        if not self.config.prepend_breadcrumb:
            return ""
        trail = [part for part in [doc_title, *section_path] if part and part.strip()]
        return " > ".join(trail)

    def _embed_text(self, doc_title: str, section_path: list[str], text: str) -> str:
        """What actually gets embedded and BM25-indexed.

        A chunk saying "set this to 30 seconds" is unretrievable on its own;
        prefixed with "Deploy Guide > 4.2 Timeouts" it is not.
        """
        prefix = self._prefix(doc_title, section_path)
        return f"{prefix}\n\n{text}" if prefix else text

    # -- splitter management ---------------------------------------------

    def _breadcrumb_budget(self, doc_title: str, section_path: list[str]) -> int:
        """Tokens the breadcrumb prefix will consume, so the body can be sized
        to leave room for it."""
        prefix = self._prefix(doc_title, section_path)
        return self.count_tokens(prefix) if prefix else 0

    def _splitter_for(self, breadcrumb_tokens: int) -> RecursiveCharacterTextSplitter:
        """Reserve room for the breadcrumb so embed_text stays under the model limit."""
        reserve = min(breadcrumb_tokens, int(self.config.chunk_tokens * _MAX_BREADCRUMB_SHARE))
        size = max(self.config.chunk_tokens - reserve, 1)
        if size not in self._splitters:
            self._splitters[size] = RecursiveCharacterTextSplitter(
                chunk_size=size,
                chunk_overlap=min(self.config.chunk_overlap_tokens, max(size - 1, 0)),
                separators=list(self.config.separators),
                length_function=self.count_tokens,
                keep_separator=True,
            )
        return self._splitters[size]


# --------------------------------------------------------------------------
# Token counting
# --------------------------------------------------------------------------


def default_token_counter(model_name: str) -> TokenCounter:
    """Token counter backed by the embedding model's own tokenizer.

    Falls back to a character heuristic if the tokenizer cannot be loaded (no
    network on a fresh machine), so ingestion degrades instead of failing.
    """

    def count(text: str) -> int:
        tokenizer = _load_tokenizer(model_name)
        if tokenizer is None:
            return heuristic_token_count(text)
        return len(tokenizer.encode(text, add_special_tokens=False))

    return count


@lru_cache(maxsize=4)
def _load_tokenizer(model_name: str) -> Any | None:
    try:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(model_name)
    except Exception as exc:
        log.warning(
            "tokenizer_unavailable",
            model=model_name,
            error=str(exc),
            fallback="character heuristic (~4 chars/token)",
        )
        return None


__all__ = ["Chunker", "TokenCounter", "default_token_counter", "heuristic_token_count"]


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def _normalise(text: str) -> str:
    return _BLANKS.sub("\n\n", _WS.sub(" ", text)).strip()


def _blocks_text(blocks: list[Block]) -> str:
    return "\n\n".join(b.text for b in blocks if b.text.strip())


def _concat_with_offsets(blocks: list[Block]) -> tuple[str, list[tuple[int, int, int]]]:
    """Join blocks and record ``(start, end, page_no)`` so chunks can be traced
    back to the page they came from."""
    parts: list[str] = []
    offsets: list[tuple[int, int, int]] = []
    cursor = 0
    for block in blocks:
        text = block.text
        if not text.strip():
            continue
        if parts:
            cursor += 2  # the "\n\n" separator
        offsets.append((cursor, cursor + len(text), block.page_no))
        parts.append(text)
        cursor += len(text)
    return "\n\n".join(parts), offsets


def _locate(haystack: str, piece: str, from_index: int) -> int:
    """Find where a split piece sits in the source text.

    The splitter strips whitespace, so an exact match can fail; a prefix probe
    is enough to recover the position, and the worst case (not found) only
    costs page-range precision.
    """
    found = haystack.find(piece, from_index)
    if found >= 0:
        return found
    probe = piece[:40].strip()
    if probe:
        found = haystack.find(probe, from_index)
        if found >= 0:
            return found
    return from_index


def _page_range(
    offsets: list[tuple[int, int, int]], start: int, end: int
) -> tuple[int, int]:
    pages = [page for (b_start, b_end, page) in offsets if b_start < end and b_end > start]
    if not pages:
        pages = [offsets[0][2]] if offsets else [1]
    return min(pages), max(pages)
