"""Presentation helpers.

Kept out of the Streamlit module so they can be unit-tested. Streamlit scripts
re-execute top to bottom on every interaction, which makes anything defined
inside them awkward to test and easy to break silently.
"""

from __future__ import annotations

import html
import re
from typing import Any

from .theme import citation_badge_style

CITATION = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")


def cited_numbers(answer_text: str) -> set[int]:
    """Every source number the answer refers to."""
    numbers: set[int] = set()
    for match in CITATION.finditer(answer_text):
        numbers.update(int(part) for part in match.group(1).split(","))
    return numbers


def highlight_citations(answer_text: str) -> str:
    """Render the answer with citation markers as visible badges.

    The text is HTML-escaped first: an answer quoting a config file containing
    angle brackets would otherwise inject markup into the page.
    """
    escaped = html.escape(answer_text)

    def badge(match: re.Match[str]) -> str:
        # inline-block + margin matters: adjacent badges for [1][2] would
        # otherwise sit flush and read as the single number 12.
        return f'<span style="{citation_badge_style()}">{match.group(1)}</span>'

    return CITATION.sub(badge, escaped).replace("\n", "<br>")


def source_label(source: dict[str, Any]) -> str:
    """The one-line heading of a source expander."""
    parts = [f"[{source['number']}]", source.get("source_file", "?")]
    if source.get("page_label"):
        parts.append(source["page_label"])
    breadcrumb = " > ".join(source.get("section_path") or [])
    if breadcrumb:
        parts.append(breadcrumb)
    return "  ·  ".join(parts)


def verification_badges(report: dict[str, Any] | None) -> list[tuple[str, str]]:
    """Compact ``(label, tone)`` pairs for the badge row under an answer.

    Tone is one of good / warn / bad / plain. A row of small badges says the
    same thing a full-width alert box does, without dominating the answer it is
    commenting on.
    """
    if not report:
        return [("not verified", "plain")]

    badges: list[tuple[str, str]] = []

    if report.get("passed"):
        badges.append(("verified", "good"))
    elif report.get("invalid_citations"):
        numbers = ", ".join(f"[{n}]" for n in report["invalid_citations"])
        badges.append((f"cites {numbers} — no such source", "bad"))
    else:
        badges.append(("needs review", "warn"))

    claims = report.get("total_claims", 0)
    if claims:
        badges.append((f"{report.get('cited_claims', 0)}/{claims} claims cited", "plain"))

    if report.get("uncited_claims"):
        badges.append((f"{report['uncited_claims']} uncited", "warn"))
    if report.get("unsupported_claims"):
        badges.append((f"{report['unsupported_claims']} unsupported", "bad"))

    return badges


def format_duration(milliseconds: float) -> str:
    if milliseconds < 1000:
        return f"{milliseconds:.0f} ms"
    return f"{milliseconds / 1000:.1f} s"


def health_caption(health: dict[str, Any]) -> str:
    """One line describing what the backend currently has indexed."""
    if not health.get("index_built"):
        return "No index yet - upload documents and run ingestion."
    return (
        f"{health.get('chunk_count', 0)} chunks from "
        f"{health.get('document_count', 0)} documents"
    )


def unready_components(health: dict[str, Any]) -> list[str]:
    """Human-readable reasons the service is not fully ready."""
    return [
        f"{component['name']}: {component.get('detail') or 'not ready'}"
        for component in health.get("components", [])
        if not component.get("ready")
    ]
