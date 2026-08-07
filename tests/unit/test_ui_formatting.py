from __future__ import annotations

import pytest

from askmydocs.ui.formatting import (
    cited_numbers,
    format_duration,
    health_caption,
    highlight_citations,
    source_label,
    unready_components,
    verification_badges,
)

# -- citations -------------------------------------------------------------


def test_cited_numbers_are_collected() -> None:
    assert cited_numbers("A [1]. B [2][3]. C [1].") == {1, 2, 3}


def test_grouped_citations_are_collected() -> None:
    assert cited_numbers("Both agree [1, 2].") == {1, 2}


def test_text_without_citations_yields_none() -> None:
    assert cited_numbers("No sources referenced here.") == set()


def test_citations_become_badges() -> None:
    rendered = highlight_citations("The default is 30s [1].")
    assert "<span" in rendered
    assert ">1</span>" in rendered


def test_adjacent_badges_are_visually_separated() -> None:
    # Without a gap, the pills for [1][2] sit flush and read as "12".
    rendered = highlight_citations("Both sources agree [1][2].")
    assert rendered.count("<span") == 2
    assert "margin" in rendered
    assert "inline-block" in rendered


def test_answer_html_is_escaped() -> None:
    # An answer quoting XML config would otherwise inject markup into the page.
    rendered = highlight_citations("Set <timeout>30</timeout> in the file [1].")
    assert "&lt;timeout&gt;" in rendered
    assert "<timeout>" not in rendered


def test_script_tags_cannot_survive_rendering() -> None:
    rendered = highlight_citations("<script>alert('x')</script>")
    assert "<script>" not in rendered


def test_newlines_become_line_breaks() -> None:
    assert "<br>" in highlight_citations("First line.\nSecond line.")


# -- source labels ---------------------------------------------------------


def test_source_label_includes_number_file_page_and_section() -> None:
    label = source_label(
        {
            "number": 2,
            "source_file": "handbook.pdf",
            "page_label": "p. 14",
            "section_path": ["3. Deploy", "3.2 Rollback"],
        }
    )
    assert label.startswith("[2]")
    assert "handbook.pdf" in label
    assert "p. 14" in label
    assert "3. Deploy > 3.2 Rollback" in label


def test_source_label_without_a_section_still_renders() -> None:
    label = source_label(
        {"number": 1, "source_file": "notes.pdf", "page_label": "p. 3", "section_path": []}
    )
    assert label == "[1]  ·  notes.pdf  ·  p. 3"


# -- verification badges ---------------------------------------------------


def _tones(report: dict | None) -> list[str]:
    return [tone for _, tone in verification_badges(report)]


def test_a_passing_report_leads_with_a_good_badge() -> None:
    badges = verification_badges(
        {"passed": True, "total_claims": 3, "cited_claims": 3, "citation_precision": 1.0}
    )
    assert badges[0] == ("verified", "good")
    assert ("3/3 claims cited", "plain") in badges


def test_a_fabricated_citation_leads_with_a_bad_badge() -> None:
    label, tone = verification_badges({"passed": False, "invalid_citations": [7]})[0]
    assert tone == "bad"
    assert "[7]" in label


def test_an_uncited_claim_leads_with_a_warning_badge() -> None:
    badges = verification_badges(
        {"passed": False, "invalid_citations": [], "uncited_claims": 2, "total_claims": 5}
    )
    assert badges[0] == ("needs review", "warn")
    assert ("2 uncited", "warn") in badges


def test_unsupported_claims_get_their_own_badge() -> None:
    badges = verification_badges(
        {"passed": False, "invalid_citations": [], "unsupported_claims": 1}
    )
    assert ("1 unsupported", "bad") in badges


def test_a_missing_report_says_not_verified() -> None:
    assert verification_badges(None) == [("not verified", "plain")]


def test_badge_tones_are_all_known_css_classes() -> None:
    # Every tone maps to a class in theme.CSS; a typo would render unstyled.
    report = {"passed": False, "invalid_citations": [7], "uncited_claims": 1, "total_claims": 2}
    assert set(_tones(report)) <= {"good", "warn", "bad", "plain"}


# -- misc ------------------------------------------------------------------


@pytest.mark.parametrize(
    ("milliseconds", "expected"), [(0, "0 ms"), (412.7, "413 ms"), (1500, "1.5 s")]
)
def test_durations_are_human_readable(milliseconds: float, expected: str) -> None:
    assert format_duration(milliseconds) == expected


def test_health_caption_describes_the_index() -> None:
    caption = health_caption({"index_built": True, "chunk_count": 42, "document_count": 3})
    assert "42 chunks" in caption
    assert "3 documents" in caption


def test_health_caption_says_what_to_do_when_empty() -> None:
    assert "upload" in health_caption({"index_built": False}).lower()


def test_unready_components_are_listed_with_reasons() -> None:
    reasons = unready_components(
        {
            "components": [
                {"name": "index", "ready": False, "detail": "no chunks found"},
                {"name": "llm", "ready": True, "detail": None},
            ]
        }
    )
    assert reasons == ["index: no chunks found"]


def test_a_component_without_a_detail_still_reports() -> None:
    reasons = unready_components({"components": [{"name": "llm", "ready": False}]})
    assert reasons == ["llm: not ready"]
