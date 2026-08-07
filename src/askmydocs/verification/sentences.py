"""Splitting an answer into sentences and deciding which ones assert something.

Every false positive here becomes a spurious "uncited claim" warning, and every
false negative lets an uncited assertion through - so the rules are explicit and
conservative rather than clever.

Two hazards drive the design:

* **Technical text is full of periods that do not end sentences.** Version
  strings (``v2.1.0``), decimals (``30.5 seconds``), abbreviations (``e.g.``)
  and identifiers (``config.yaml``) all break a naive split-on-period.
* **Code samples contain square brackets.** ``items[0]`` inside a code span is
  not a citation, and a documentation assistant produces code constantly. Code
  regions are masked before anything is parsed.

No NLP dependency: spaCy or NLTK would add a model download to a step that runs
on every answer, for a task a few dozen lines of careful regex handle.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

#: A citation marker: [1], [2], or a grouped [1, 2].
CITATION = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")

#: Fenced code blocks and inline code spans, in that order.
_CODE_FENCE = re.compile(r"```.*?```", re.DOTALL)
_CODE_SPAN = re.compile(r"`[^`\n]+`")

#: Abbreviations whose trailing period does not end a sentence.
_ABBREVIATIONS = frozenset(
    ["e.g", "i.e", "etc", "vs", "cf", "approx", "fig", "no", "cols", "col", "sec", "secs", "min", "mins", "max", "ms", "mr", "mrs", "ms", "dr", "prof", "inc", "ltd", "corp", "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept", "oct", "nov", "dec"]
)

#: Leading list markers stripped before a sentence is judged.
_LIST_MARKER = re.compile(r"^\s*(?:[-*+•]|\d+[.)])\s+")

#: Sentences that talk *about* the sources' limits rather than asserting a fact.
#: "According to the sources, X is 30s" is a claim; "the sources do not cover X"
#: is not, and flagging it as uncited would penalise the honest behaviour the
#: prompt asks for.
_META_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in (
        r"\b(?:the\s+)?(?:provided\s+)?sources?\s+(?:do|does)\s*n[o']t\b",
        r"\bi\s+(?:do\s*n[o']t|don't|cannot|can't)\s+have\s+enough\s+information\b",
        r"\bnot\s+(?:covered|mentioned|specified|stated|included|present)\s+in\s+the\b",
        r"\bno\s+information\s+(?:is\s+)?(?:available|provided|given)\b",
        r"\b(?:there\s+is\s+)?nothing\s+in\s+the\s+(?:provided\s+)?sources?\b",
    )
]


@dataclass(slots=True)
class Sentence:
    """One sentence of an answer, with the citations attached to it."""

    text: str
    start: int
    end: int
    citations: list[int] = field(default_factory=list)

    @property
    def is_cited(self) -> bool:
        return bool(self.citations)


def parse_citations(text: str) -> list[int]:
    """Every citation number in ``text``, in order of appearance.

    Duplicates are preserved: citing [1] three times is three citations, and
    citation precision should be computed over what the model actually wrote.
    """
    numbers: list[int] = []
    for match in CITATION.finditer(mask_code(text)):
        numbers.extend(int(part) for part in match.group(1).split(","))
    return numbers


def mask_code(text: str) -> str:
    """Blank out code regions, preserving length so offsets stay valid.

    Without this, ``arr[0]`` in a code sample parses as a citation to source 0
    and a period inside ``config.yaml`` splits a sentence in two.
    """

    def blank(match: re.Match[str]) -> str:
        return "".join(" " if char.isspace() else "x" for char in match.group(0))

    return _CODE_SPAN.sub(blank, _CODE_FENCE.sub(blank, text))


def split_sentences(text: str) -> list[Sentence]:
    """Split ``text`` into sentences carrying their citations."""
    if not text.strip():
        return []

    masked = mask_code(text)
    sentences: list[Sentence] = []

    for line_start, line_end in _line_spans(masked):
        for start, end in _sentence_spans(masked, line_start, line_end):
            raw = text[start:end].strip()
            if not raw:
                continue
            sentences.append(
                Sentence(text=raw, start=start, end=end, citations=parse_citations(raw))
            )

    return _absorb_trailing_citations(sentences)


def is_claim(sentence: Sentence, min_words: int = 4) -> bool:
    """Whether a sentence asserts something that ought to be cited."""
    stripped = _LIST_MARKER.sub("", sentence.text).strip()
    without_citations = CITATION.sub("", stripped).strip()

    if not without_citations:
        return False
    if len(without_citations.split()) < min_words:
        return False
    # A line ending in a colon introduces the list that follows; the citations
    # belong on the items, not on the lead-in.
    if without_citations.endswith(":"):
        return False
    if without_citations.endswith("?"):
        return False
    return not any(pattern.search(without_citations) for pattern in _META_PATTERNS)


# --------------------------------------------------------------------------
# Segmentation internals
# --------------------------------------------------------------------------


def _line_spans(text: str) -> list[tuple[int, int]]:
    """Line boundaries. Newlines always end a sentence - bullets and headings
    frequently carry no terminal punctuation at all."""
    spans: list[tuple[int, int]] = []
    position = 0
    for line in text.split("\n"):
        spans.append((position, position + len(line)))
        position += len(line) + 1
    return [(s, e) for s, e in spans if text[s:e].strip()]


def _sentence_spans(text: str, start: int, end: int) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    cursor = start

    for match in re.finditer(r"[.!?]+", text[start:end]):
        boundary = start + match.end()
        if not _is_sentence_end(text, start + match.start(), boundary, end):
            continue
        # Keep any citation that trails the punctuation with this sentence.
        boundary = _extend_over_citations(text, boundary, end)
        spans.append((cursor, boundary))
        cursor = boundary

    if cursor < end:
        spans.append((cursor, end))
    return spans


def _is_sentence_end(text: str, punct_start: int, punct_end: int, limit: int) -> bool:
    if text[punct_start] != ".":
        return True  # ! and ? are unambiguous

    before = text[:punct_start]
    after = text[punct_end:limit]

    # A period between digits is a decimal or a version number.
    if before[-1:].isdigit() and after[:1].isdigit():
        return False
    # config.yaml, module.function - no space after the dot.
    if after[:1].isalpha():
        return False

    last_word = re.split(r"[\s(]", before)[-1].lower().rstrip(".")
    if last_word in _ABBREVIATIONS:
        return False
    # A single capital letter is an initial (J. Smith), not a sentence end.
    return not (len(last_word) == 1 and last_word.isalpha())


def _extend_over_citations(text: str, position: int, limit: int) -> int:
    """Swallow ``[1]`` markers that follow the full stop.

    Models write both "X is 30s [1]." and "X is 30s. [1]" - the citation belongs
    to the sentence either way.
    """
    cursor = position
    while cursor < limit:
        match = re.match(r"\s*\[\d+(?:\s*,\s*\d+)*\]", text[cursor:limit])
        if not match:
            break
        cursor += match.end()
    return cursor


def _absorb_trailing_citations(sentences: list[Sentence]) -> list[Sentence]:
    """Fold a fragment that is nothing but citations into the sentence before it.

    A citation on its own line refers backwards; left standing it would be a
    zero-claim fragment and would strip the real sentence of its support.
    """
    merged: list[Sentence] = []
    for sentence in sentences:
        only_citations = not CITATION.sub("", sentence.text).strip()
        if only_citations and merged:
            previous = merged[-1]
            previous.citations.extend(sentence.citations)
            previous.text = f"{previous.text} {sentence.text}".strip()
            previous.end = sentence.end
            continue
        merged.append(sentence)
    return merged
