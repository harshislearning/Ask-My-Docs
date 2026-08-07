"""Verification entry point."""

from __future__ import annotations

from ..config import VerificationConfig
from ..logging_setup import get_logger
from ..models import (
    Answer,
    CitationIssue,
    IssueType,
    Severity,
    VerificationReport,
)
from .citations import check_citations, claims_in
from .entailment import ClaimCheck, check_entailment

log = get_logger(__name__)


class Verifier:
    """Checks a generated answer against the sources it was given.

    ``client`` is only used when ``entailment_mode`` is ``llm``; without one the
    check silently stays lexical rather than failing.
    """

    def __init__(self, config: VerificationConfig, client: object | None = None) -> None:
        self.config = config
        self.client = client

    def verify(self, answer: Answer) -> Answer:
        """Attach a :class:`VerificationReport` to ``answer`` and return it.

        The answer text is never modified. A verifier that quietly rewrote
        output would hide exactly the failures it exists to surface.
        """
        if not self.config.enabled:
            return answer

        if answer.refused:
            # A refusal makes no claims, so there is nothing to check. Running
            # the claim detector over it would flag the refusal sentence itself.
            answer.verification = VerificationReport(
                entailment_mode=self.config.entailment_mode, passed=True
            )
            return answer

        report = check_citations(answer.text, answer.sources, self.config)
        self._add_entailment(report, answer)
        answer.verification = report

        if not report.passed:
            log.warning(
                "answer_failed_verification",
                invalid_citations=report.invalid_citations,
                uncited_claims=report.uncited_claims,
                issues=[issue.detail for issue in report.issues][:5],
            )
        return answer

    def _add_entailment(self, report: VerificationReport, answer: Answer) -> None:
        """Layer entailment findings onto the deterministic report."""
        if self.config.entailment_mode == "off":
            return

        valid = {source.number for source in answer.sources}
        claims = [
            claim
            for claim in claims_in(answer.text, self.config)
            if claim.is_cited and any(n in valid for n in claim.citations)
        ]
        checks = check_entailment(claims, answer.sources, self.config, self.client)

        unsupported = [check for check in checks if check.is_problem]
        report.issues.extend(_to_issues(unsupported))
        report.unsupported_claims = len(unsupported)
        if unsupported:
            report.passed = False


def _to_issues(checks: list[ClaimCheck]) -> list[CitationIssue]:
    return [
        CitationIssue(
            type=IssueType.UNSUPPORTED_CLAIM,
            # A partial match is a hint to a reviewer; a flat contradiction is a
            # defect in the answer.
            severity=Severity.WARNING if check.verdict == "partial" else Severity.ERROR,
            detail=f"{check.reason} (checked by {check.checked_by})",
            sentence=check.sentence.text,
            citation=check.sentence.citations[0] if check.sentence.citations else None,
        )
        for check in checks
    ]
