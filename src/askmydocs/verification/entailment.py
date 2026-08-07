"""Does the cited chunk actually support the claim?

Citation *validity* only proves a number points at a real source. It says
nothing about whether that source says what the answer claims it says. An
answer reading "drain_timeout defaults to 120 seconds [5]" passes every
structural check whether the source says 120 or 210.

Closing that gap runs in two layers, cheapest first:

**Lexical grounding** (always, free, deterministic). Numbers and identifiers are
the parts of a technical claim that are both checkable and worth checking - they
are what a reader acts on, and what a model transposes. Every number and
identifier in a claim must appear somewhere in the sources it cites. Number
words are normalised to digits on both sides, so "thirty seconds" matches "30".

**An LLM judge** (only when the first layer cannot decide). Claims with nothing
lexically checkable, and claims the lexical layer flagged, go to one batched
call scoped to just those claims and just the chunks they cite. The judge is
authoritative for what it sees: lexical grounding has real false positives
("half a minute" against "30 seconds"), and a false accusation is worse than a
missed one when the output is shown to a user.

Both layers fail open. A parser hiccup or a rate limit must not manufacture an
accusation against an answer that may be perfectly sound.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import StrEnum

from ..config import VerificationConfig
from ..generation.prompts import format_source
from ..logging_setup import get_logger
from ..models import Source
from .sentences import CITATION, Sentence

log = get_logger(__name__)


class Verdict(StrEnum):
    SUPPORTED = "supported"
    PARTIAL = "partial"
    NOT_SUPPORTED = "not_supported"
    #: Nothing checkable, or the check could not be completed. Never an accusation.
    INCONCLUSIVE = "inconclusive"


@dataclass(slots=True)
class ClaimCheck:
    """One claim, the sources it cites, and what checking concluded."""

    sentence: Sentence
    verdict: Verdict
    reason: str
    checked_by: str = "lexical"
    missing: list[str] = field(default_factory=list)

    @property
    def is_problem(self) -> bool:
        return self.verdict in (Verdict.NOT_SUPPORTED, Verdict.PARTIAL)


# --------------------------------------------------------------------------
# Layer 1: lexical grounding
# --------------------------------------------------------------------------

#: snake_case, dotted.paths, kebab-case, file.names - the things a reader copies.
_IDENTIFIER = re.compile(r"\b[a-zA-Z][a-zA-Z0-9]*(?:[._-][a-zA-Z0-9]+)+\b")
_NUMBER = re.compile(r"\b\d+(?:\.\d+)?\b")

_UNITS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
    "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16,
    "seventeen": 17, "eighteen": 18, "nineteen": 19,
}
_TENS = {
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
    "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
}
_NUMBER_WORD = re.compile(
    r"\b(" + "|".join([*_TENS, *_UNITS]) + r")(?:[-\s](" + "|".join(_UNITS) + r"))?\b",
    re.IGNORECASE,
)


def normalise_numbers(text: str) -> str:
    """Rewrite number words as digits so both sides compare on equal terms.

    Without it, "the timeout is thirty seconds" looks ungrounded against a table
    reading "30" - a false accusation on a perfectly good answer.
    """

    def replace(match: re.Match[str]) -> str:
        head = match.group(1).lower()
        tail = match.group(2)
        value = _TENS.get(head, _UNITS.get(head, 0))
        if tail and head in _TENS:
            value += _UNITS.get(tail.lower(), 0)
        elif tail:
            # "one five" is not a compound number; leave the tail alone.
            return f"{value} {tail}"
        return str(value)

    return _NUMBER_WORD.sub(replace, text)


def extract_probes(claim: str, config: VerificationConfig) -> tuple[set[str], set[str]]:
    """The numbers and identifiers in a claim that can be checked against text."""
    body = CITATION.sub(" ", claim)
    identifiers = {
        token.lower()
        for token in _IDENTIFIER.findall(body)
        if len(token) >= config.min_identifier_length
    }
    # Identifiers such as v2.1.0 also contain digits; drop those digits so a
    # version string is not double-counted as a bare number.
    without_identifiers = _IDENTIFIER.sub(" ", body)
    numbers = set(_NUMBER.findall(normalise_numbers(without_identifiers)))
    return numbers, identifiers


def check_lexically(
    sentence: Sentence, sources: Sequence[Source], config: VerificationConfig
) -> ClaimCheck:
    """Require every number and identifier in the claim to appear in its sources."""
    cited = [source for source in sources if source.number in sentence.citations]
    if not cited:
        return ClaimCheck(sentence, Verdict.INCONCLUSIVE, "no valid citation to check")

    numbers, identifiers = extract_probes(sentence.text, config)
    if not numbers and not identifiers:
        return ClaimCheck(
            sentence, Verdict.INCONCLUSIVE, "no numbers or identifiers to verify"
        )

    # The label is part of what the model was shown, so a claim naming the file
    # or section it came from is legitimately grounded.
    haystack = normalise_numbers(" ".join(format_source(s) for s in cited)).lower()

    missing = [probe for probe in sorted(numbers | identifiers) if probe not in haystack]
    if missing:
        return ClaimCheck(
            sentence,
            Verdict.NOT_SUPPORTED,
            f"not found in the cited source(s): {', '.join(missing)}",
            missing=missing,
        )
    return ClaimCheck(sentence, Verdict.SUPPORTED, "all values found in the cited source(s)")


# --------------------------------------------------------------------------
# Layer 2: the LLM judge
# --------------------------------------------------------------------------

JUDGE_SYSTEM_PROMPT = """\
You check whether a claim is supported by a source text. You are given numbered \
items, each with a CLAIM and the SOURCE text it cites.

For each item, decide:
- SUPPORTED: the source states the claim, or states it in different words.
- PARTIAL: the source supports some of the claim but not all of it.
- NOT_SUPPORTED: the source does not state the claim, or contradicts it.

Judge only from the source text shown. Do not use outside knowledge. Absence of \
information means NOT_SUPPORTED, not SUPPORTED.

Reply with exactly one line per item, in this format and nothing else:
1: SUPPORTED
2: NOT_SUPPORTED\
"""

_VERDICT_LINE = re.compile(r"^\s*(\d+)\s*[:.)-]\s*(SUPPORTED|PARTIAL|NOT_SUPPORTED)", re.I | re.M)

_VERDICTS = {
    "SUPPORTED": Verdict.SUPPORTED,
    "PARTIAL": Verdict.PARTIAL,
    "NOT_SUPPORTED": Verdict.NOT_SUPPORTED,
}


def build_judge_messages(
    checks: Sequence[ClaimCheck], sources: Sequence[Source]
) -> list[dict[str, str]]:
    """One prompt covering every escalated claim.

    Each claim carries only the chunks it actually cites - a judge shown the
    whole context could mark a claim supported by evidence the answer never
    pointed at, which is precisely the error being looked for.
    """
    by_number = {source.number: source for source in sources}
    blocks: list[str] = []

    for position, check in enumerate(checks, start=1):
        cited = [by_number[n] for n in check.sentence.citations if n in by_number]
        source_text = "\n\n".join(s.text.strip() for s in cited) or "(no source text)"
        claim = CITATION.sub("", check.sentence.text).strip()
        blocks.append(f"ITEM {position}\nCLAIM: {claim}\nSOURCE:\n{source_text}")

    return [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": "\n\n---\n\n".join(blocks)},
    ]


def parse_judge_response(text: str, expected: int) -> dict[int, Verdict]:
    """Read the judge's verdict lines.

    Anything unparseable is simply absent from the result, and the caller leaves
    that claim inconclusive - a malformed reply must never become an accusation.
    """
    verdicts: dict[int, Verdict] = {}
    for match in _VERDICT_LINE.finditer(text or ""):
        position = int(match.group(1))
        if 1 <= position <= expected:
            verdicts[position] = _VERDICTS[match.group(2).upper()]
    return verdicts


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------


def check_entailment(
    claims: Sequence[Sentence],
    sources: Sequence[Source],
    config: VerificationConfig,
    client: object | None = None,
) -> list[ClaimCheck]:
    """Run the layered check over every cited claim."""
    if config.entailment_mode == "off" or not claims:
        return []

    checks = [
        check_lexically(claim, sources, config) for claim in claims if claim.is_cited
    ]

    if config.entailment_mode != "llm" or client is None:
        if config.entailment_mode == "llm":
            log.warning("entailment_judge_unavailable", fallback="lexical verdicts only")
        return checks

    escalate = [
        check
        for check in checks
        if check.verdict in (Verdict.NOT_SUPPORTED, Verdict.INCONCLUSIVE)
    ][: config.entailment_max_claims]

    if not escalate:
        return checks

    _adjudicate(escalate, sources, client)
    return checks


def _adjudicate(
    escalate: list[ClaimCheck], sources: Sequence[Source], client: object
) -> None:
    """Ask the judge and overwrite the escalated verdicts in place."""
    try:
        response = client.complete(build_judge_messages(escalate, sources))  # type: ignore[attr-defined]
        verdicts = parse_judge_response(response.text, len(escalate))
    except Exception as exc:
        log.warning("entailment_judge_failed", error=str(exc), fallback="lexical verdicts")
        return

    for position, check in enumerate(escalate, start=1):
        verdict = verdicts.get(position)
        if verdict is None:
            # Unparsed: downgrade a lexical accusation to inconclusive rather
            # than report a verdict nothing confirmed.
            check.verdict = Verdict.INCONCLUSIVE
            check.reason = "judge did not return a verdict for this claim"
            check.checked_by = "llm"
            continue
        check.verdict = verdict
        check.checked_by = "llm"
        check.reason = {
            Verdict.SUPPORTED: "the cited source states this",
            Verdict.PARTIAL: "the cited source supports only part of this claim",
            Verdict.NOT_SUPPORTED: "the cited source does not state this",
        }[verdict]

    log.info(
        "entailment_adjudicated",
        claims=len(escalate),
        verdicts={v.value: sum(1 for c in escalate if c.verdict is v) for v in Verdict},
    )
