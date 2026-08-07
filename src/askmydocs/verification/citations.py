"""Post-hoc citation checking.

The prompt asks the model to cite every claim. This module checks whether it
did, and it exists because prompting is a request, not a guarantee - a fluent,
well-formatted answer with a citation to a source that was never supplied looks
exactly like a correct one.

Three deterministic checks run on every answer:

1. **Citation validity** - every ``[n]`` refers to a source that was actually in
   the prompt. A number outside the range is fabricated.
2. **Claim coverage** - every sentence that asserts something carries at least
   one citation.
3. **Source usage** - which supplied sources went unused. Not a defect, but a
   direct signal that ``rerank_top_k`` may be oversized.

Findings are *reported*, never repaired. Rewriting a model's output to hide a
missing citation would conceal the failure from the user and from eval.
"""

from __future__ import annotations

from collections.abc import Sequence

from ..config import VerificationConfig
from ..logging_setup import get_logger
from ..models import (
    CitationIssue,
    IssueType,
    Severity,
    Source,
    VerificationReport,
)
from .sentences import Sentence, is_claim, parse_citations, split_sentences

log = get_logger(__name__)


def check_citations(
    answer_text: str,
    sources: Sequence[Source],
    config: VerificationConfig,
) -> VerificationReport:
    """Run the deterministic checks over one answer."""
    valid_numbers = {source.number for source in sources}
    sentences = split_sentences(answer_text)
    citations = parse_citations(answer_text)

    issues: list[CitationIssue] = []
    valid = [number for number in citations if number in valid_numbers]
    invalid = [number for number in citations if number not in valid_numbers]

    issues.extend(_unknown_source_issues(sentences, valid_numbers))

    claims = [s for s in sentences if is_claim(s, config.min_claim_words)]
    cited_claims = [s for s in claims if s.is_cited]
    uncited_claims = [s for s in claims if not s.is_cited]

    if config.flag_uncited_claims:
        issues.extend(
            CitationIssue(
                type=IssueType.UNCITED_CLAIM,
                severity=Severity.WARNING,
                detail="sentence asserts something but carries no citation",
                sentence=sentence.text,
            )
            for sentence in uncited_claims
        )

    unused = sorted(valid_numbers - set(valid))

    report = VerificationReport(
        citations=citations,
        valid_citations=valid,
        invalid_citations=invalid,
        unused_sources=unused,
        total_claims=len(claims),
        cited_claims=len(cited_claims),
        uncited_claims=len(uncited_claims),
        issues=issues,
        entailment_mode=config.entailment_mode,
        passed=_passed(invalid, uncited_claims, config),
    )

    log.info("citations_verified", **report.summary())
    return report


def claims_in(answer_text: str, config: VerificationConfig) -> list[Sentence]:
    """The sentences of ``answer_text`` that assert something.

    Shared with entailment checking so both layers agree on what a claim is.
    """
    return [
        sentence
        for sentence in split_sentences(answer_text)
        if is_claim(sentence, config.min_claim_words)
    ]


def _unknown_source_issues(
    sentences: Sequence[Sentence], valid_numbers: set[int]
) -> list[CitationIssue]:
    """One issue per fabricated citation, located in its sentence."""
    issues: list[CitationIssue] = []
    for sentence in sentences:
        for number in sentence.citations:
            if number in valid_numbers:
                continue
            issues.append(
                CitationIssue(
                    type=IssueType.UNKNOWN_SOURCE,
                    severity=Severity.ERROR,
                    detail=(
                        f"cited [{number}] but only "
                        f"{_range_description(valid_numbers)} were provided"
                    ),
                    sentence=sentence.text,
                    citation=number,
                )
            )
    return issues


def _range_description(valid_numbers: set[int]) -> str:
    if not valid_numbers:
        return "no sources"
    if len(valid_numbers) == 1:
        return "source [1]"
    return f"sources [1]-[{max(valid_numbers)}]"


def _passed(
    invalid: Sequence[int],
    uncited_claims: Sequence[Sentence],
    config: VerificationConfig,
) -> bool:
    """A fabricated citation always fails.

    An uncited claim fails only when the check is switched on, because sentence
    segmentation is heuristic and a borderline false positive should not mark an
    otherwise sound answer as broken.
    """
    if invalid:
        return False
    return not (config.flag_uncited_claims and uncited_claims)
