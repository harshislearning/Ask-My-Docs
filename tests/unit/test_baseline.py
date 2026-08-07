"""The regression gate.

A gate that fires on noise gets disabled within a week, and one that stays
silent through a real regression is worse than none. Both failure modes are
pinned down here, along with the awkward cases: a metric that is new, one that
vanished, and one where lower is better.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from askmydocs.evaluation.baseline import (
    Gate,
    compare_to_baseline,
    format_comparison,
    format_markdown,
    load_baseline,
    read_metric,
    write_baseline,
)


def _report(**blocks: dict[str, Any]) -> dict[str, Any]:
    """A report, with named blocks merged over the defaults.

    e.g. ``_report(reranked={"recall@5": 0.80})``
    """
    report: dict[str, Any] = {
        "created_at": "2026-01-01T00:00:00Z",
        "items_evaluated": 9,
        "reranked": {"recall@5": 0.90, "mrr": 0.85, "ndcg@5": 0.88},
        "generation": {"citation_precision": 1.0, "groundedness": 0.95},
        "refusal": {"wrongly_answered": 0},
        "config": {"rerank_top_k": 6},
    }
    for name, values in blocks.items():
        report[name] = {**report.get(name, {}), **values}
    return report


RECALL = Gate(metric="reranked.recall@5", max_drop=0.05)
PRECISION = Gate(metric="generation.citation_precision", max_drop=0.01)
WRONGLY_ANSWERED = Gate(metric="refusal.wrongly_answered", max_drop=0, higher_is_better=False)


# -- reading metrics -------------------------------------------------------


def test_a_dotted_path_resolves() -> None:
    assert read_metric(_report(), "reranked.recall@5") == 0.90


def test_a_missing_path_is_none() -> None:
    assert read_metric(_report(), "reranked.nonsense") is None
    assert read_metric(_report(), "nothing.at.all") is None


def test_a_non_numeric_value_is_none() -> None:
    assert read_metric({"a": {"b": "text"}}, "a.b") is None


def test_a_boolean_is_not_a_metric() -> None:
    # bool is a subclass of int; treating True as 1.0 would silently gate on it.
    assert read_metric({"a": {"b": True}}, "a.b") is None


# -- passing and failing ---------------------------------------------------


def test_an_unchanged_run_passes() -> None:
    comparison = compare_to_baseline(_report(), _report(), [RECALL, PRECISION])
    assert comparison.passed
    assert comparison.failures == []


def test_a_drop_inside_the_tolerance_passes() -> None:
    current = _report(reranked={"recall@5": 0.87})  # -0.03, tolerance 0.05
    comparison = compare_to_baseline(current, _report(), [RECALL])
    assert comparison.passed
    assert comparison.results[0].reason == "within tolerance"


def test_a_drop_past_the_tolerance_fails() -> None:
    current = _report(reranked={"recall@5": 0.80})  # -0.10
    comparison = compare_to_baseline(current, _report(), [RECALL])

    assert not comparison.passed
    assert "regressed by 0.1" in comparison.failures[0].reason


def test_a_drop_exactly_at_the_tolerance_passes() -> None:
    # The boundary is inclusive: "may drop by 0.05" has to permit 0.05.
    current = _report(reranked={"recall@5": 0.85})
    assert compare_to_baseline(current, _report(), [RECALL]).passed


def test_an_improvement_passes_and_is_marked() -> None:
    current = _report(reranked={"recall@5": 0.95})
    comparison = compare_to_baseline(current, _report(), [RECALL])

    assert comparison.passed
    assert comparison.improvements
    assert comparison.results[0].delta == pytest.approx(0.05)


def test_a_zero_tolerance_gate_fails_on_any_regression() -> None:
    current = _report(generation={"citation_precision": 0.999})
    strict = Gate(metric="generation.citation_precision", max_drop=0.0)
    assert not compare_to_baseline(current, _report(), [strict]).passed


# -- lower-is-better metrics -----------------------------------------------


def test_an_increase_in_a_lower_is_better_metric_fails() -> None:
    # Answering a question the documents cannot support is the failure that
    # actively misleads people; any increase must fail.
    current = _report(refusal={"wrongly_answered": 1})
    comparison = compare_to_baseline(current, _report(), [WRONGLY_ANSWERED])

    assert not comparison.passed
    assert "regressed by 1" in comparison.failures[0].reason


def test_a_decrease_in_a_lower_is_better_metric_is_an_improvement() -> None:
    baseline = _report(refusal={"wrongly_answered": 3})
    comparison = compare_to_baseline(_report(), baseline, [WRONGLY_ANSWERED])

    assert comparison.passed
    assert comparison.improvements


# -- awkward cases ---------------------------------------------------------


def test_a_metric_absent_from_the_baseline_passes_and_says_so() -> None:
    # A newly added gate, or a baseline from before the metric existed. Not a
    # regression, but not silently ignored either.
    baseline = {"created_at": "x", "reranked": {"mrr": 0.85}}
    comparison = compare_to_baseline(_report(), baseline, [RECALL])

    assert comparison.passed
    assert "not in baseline" in comparison.results[0].reason


def test_a_metric_missing_from_the_run_fails() -> None:
    # Almost always a broken run rather than a deliberate change; skipping it
    # would hide the breakage behind a green build.
    current = {"created_at": "x", "reranked": {}}
    comparison = compare_to_baseline(current, _report(), [RECALL])

    assert not comparison.passed
    assert "missing from this run" in comparison.failures[0].reason


def test_every_gate_is_reported_not_only_the_failures() -> None:
    current = _report(reranked={"recall@5": 0.50})
    comparison = compare_to_baseline(current, _report(), [RECALL, PRECISION, WRONGLY_ANSWERED])

    assert len(comparison.results) == 3
    assert len(comparison.failures) == 1


def test_no_gates_means_a_vacuous_pass() -> None:
    # The CLI rejects this separately; the comparison itself has nothing to say.
    assert compare_to_baseline(_report(), _report(), []).passed


# -- persistence -----------------------------------------------------------


def test_a_baseline_round_trips(tmp_path: Path) -> None:
    path = tmp_path / "baseline.json"
    write_baseline(path, _report(), reason="initial")

    loaded = load_baseline(path)
    assert loaded["reranked"]["recall@5"] == 0.90
    assert loaded["baseline_reason"] == "initial"
    assert loaded["baseline_written_at"]


def test_the_baseline_keeps_the_config_that_produced_it(tmp_path: Path) -> None:
    # A metric means nothing without the settings behind it, and a baseline
    # whose provenance is unknown cannot be argued with later.
    path = tmp_path / "baseline.json"
    write_baseline(path, _report(), reason="initial")
    assert load_baseline(path)["config"]["rerank_top_k"] == 6


def test_per_item_noise_is_not_stored(tmp_path: Path) -> None:
    path = tmp_path / "baseline.json"
    write_baseline(path, {**_report(), "errors": ["q1"], "coverage": {"unreachable": []}}, "x")

    stored = json.loads(path.read_text(encoding="utf-8"))
    assert "errors" not in stored
    assert "coverage" not in stored


def test_a_missing_baseline_says_how_to_create_one(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="update-baseline"):
        load_baseline(tmp_path / "nope.json")


def test_a_baseline_with_a_byte_order_mark_loads(tmp_path: Path) -> None:
    path = tmp_path / "baseline.json"
    path.write_bytes(b"\xef\xbb\xbf" + json.dumps(_report()).encode("utf-8"))
    assert load_baseline(path)["reranked"]["mrr"] == 0.85


# -- rendering -------------------------------------------------------------


def test_the_text_report_shows_every_gate_and_the_verdict() -> None:
    current = _report(reranked={"recall@5": 0.50})
    text = format_comparison(compare_to_baseline(current, _report(), [RECALL, PRECISION]))

    assert "reranked.recall@5" in text
    assert "generation.citation_precision" in text
    assert "FAIL" in text
    assert "-0.4000" in text


def test_a_failing_report_explains_how_to_accept_the_change() -> None:
    current = _report(reranked={"recall@5": 0.50})
    text = format_comparison(compare_to_baseline(current, _report(), [RECALL]))
    assert "--update-baseline" in text


def test_a_passing_report_says_pass() -> None:
    text = format_comparison(compare_to_baseline(_report(), _report(), [RECALL]))
    assert "PASS" in text
    assert "--update-baseline" not in text


def test_the_markdown_summary_is_a_table() -> None:
    comparison = compare_to_baseline(_report(), _report(), [RECALL])
    markdown = format_markdown(comparison, _report())

    assert markdown.startswith("## Evaluation gate: PASSED")
    assert "| `reranked.recall@5` |" in markdown


def test_the_markdown_summary_lists_regressions() -> None:
    current = _report(reranked={"recall@5": 0.50})
    markdown = format_markdown(compare_to_baseline(current, _report(), [RECALL]), current)

    assert "FAILED" in markdown
    assert "### Regressions" in markdown
