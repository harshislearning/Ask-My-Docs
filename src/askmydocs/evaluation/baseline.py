"""Comparing an evaluation run against a stored baseline.

A gate is only useful if it fires on real regressions and stays silent
otherwise. One that fires on noise gets disabled within a week, and a gate
nobody trusts is worse than none - so every threshold is explicit, every
comparison is shown whether it passed or not, and a metric that vanished from
the report fails loudly rather than being skipped.

Thresholds are expressed as *how much worse a metric may get*, not as absolute
floors. An absolute floor has to be rewritten every time the system improves;
a tolerance keeps meaning as the numbers move.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from ..logging_setup import get_logger

log = get_logger(__name__)

#: Slack for binary floating point when comparing a drop against its tolerance.
_FLOAT_TOLERANCE = 1e-9


class Gate(BaseModel):
    """One metric the build is allowed to fail on."""

    #: Dotted path into the evaluation report, e.g. "reranked.recall@5".
    metric: str
    #: How far the metric may move in the bad direction before failing.
    #: 0.0 means any regression at all fails.
    max_drop: float = Field(0.0, ge=0.0)
    higher_is_better: bool = True

    def label(self) -> str:
        arrow = "higher is better" if self.higher_is_better else "lower is better"
        return f"{self.metric} ({arrow}, tolerance {self.max_drop:g})"


@dataclass(slots=True)
class GateResult:
    gate: Gate
    baseline: float | None
    current: float | None
    passed: bool
    reason: str

    @property
    def delta(self) -> float | None:
        if self.baseline is None or self.current is None:
            return None
        return self.current - self.baseline

    @property
    def improved(self) -> bool:
        delta = self.delta
        if delta is None or delta == 0:
            return False
        return delta > 0 if self.gate.higher_is_better else delta < 0


@dataclass(slots=True)
class Comparison:
    results: list[GateResult]
    baseline_created_at: str | None
    baseline_path: str

    @property
    def passed(self) -> bool:
        return all(result.passed for result in self.results)

    @property
    def failures(self) -> list[GateResult]:
        return [result for result in self.results if not result.passed]

    @property
    def improvements(self) -> list[GateResult]:
        return [result for result in self.results if result.improved]


def read_metric(report: dict[str, Any], path: str) -> float | None:
    """Look up a dotted path in a report. Missing or non-numeric returns None."""
    node: Any = report
    for part in path.split("."):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return float(node) if isinstance(node, int | float) and not isinstance(node, bool) else None


def compare_to_baseline(
    report: dict[str, Any], baseline: dict[str, Any], gates: list[Gate], baseline_path: str = ""
) -> Comparison:
    """Check every gate. Returns all results, not just the failures."""
    results = [_check(gate, report, baseline) for gate in gates]

    comparison = Comparison(
        results=results,
        baseline_created_at=baseline.get("created_at"),
        baseline_path=baseline_path,
    )
    log.info(
        "baseline_compared",
        gates=len(results),
        failed=len(comparison.failures),
        improved=len(comparison.improvements),
    )
    return comparison


def _check(gate: Gate, report: dict[str, Any], baseline: dict[str, Any]) -> GateResult:
    current = read_metric(report, gate.metric)
    previous = read_metric(baseline, gate.metric)

    if previous is None:
        # A new gate, or a baseline from before this metric existed. Not a
        # regression - but say so rather than silently passing.
        return GateResult(gate, None, current, True, "not in baseline (nothing to compare)")

    if current is None:
        # The metric disappeared. Almost always a broken run rather than a
        # deliberate change, and silently skipping it would hide the breakage.
        return GateResult(gate, previous, None, False, "missing from this run")

    drop = (previous - current) if gate.higher_is_better else (current - previous)
    # The epsilon is not cosmetic: 0.90 - 0.85 is 0.05000000000000004 in binary
    # floating point, so a drop of exactly the stated tolerance would fail a
    # gate that promised to allow it.
    if drop > gate.max_drop + _FLOAT_TOLERANCE:
        return GateResult(
            gate,
            previous,
            current,
            False,
            f"regressed by {drop:.4f}, tolerance is {gate.max_drop:g}",
        )
    return GateResult(gate, previous, current, True, "within tolerance")


# --------------------------------------------------------------------------
# Persistence
# --------------------------------------------------------------------------


def load_baseline(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"no baseline at {path} - create one with "
            "scripts/check_regression.py --update-baseline"
        )
    payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8-sig"))
    return payload


def write_baseline(path: str | Path, report: dict[str, Any], reason: str = "") -> None:
    """Store a run as the new baseline.

    The per-item records are dropped and the config snapshot is kept: a metric
    means nothing without the settings that produced it, and a baseline whose
    provenance is unknown cannot be argued with later.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    baseline = {
        key: value for key, value in report.items() if key not in {"errors", "coverage"}
    }
    baseline["baseline_written_at"] = datetime.now(UTC).isoformat()
    baseline["baseline_reason"] = reason or "(no reason given)"

    path.write_text(json.dumps(baseline, indent=2, default=str), encoding="utf-8")
    log.info("baseline_written", path=str(path), reason=reason)


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------


def format_comparison(comparison: Comparison) -> str:
    """A plain-text table for terminal output."""
    lines = [
        "",
        f"Baseline: {comparison.baseline_path or '(unknown)'}",
        f"Recorded: {comparison.baseline_created_at or '(unknown)'}",
        "",
        f"  {'metric':<34} {'baseline':>9} {'current':>9} {'delta':>9}  ",
        "  " + "-" * 72,
    ]

    for result in comparison.results:
        marker = "ok  " if result.passed else "FAIL"
        baseline = f"{result.baseline:.4f}" if result.baseline is not None else "-"
        current = f"{result.current:.4f}" if result.current is not None else "-"
        delta = f"{result.delta:+.4f}" if result.delta is not None else "-"
        note = "" if result.passed and not result.improved else f"  {result.reason}"
        if result.improved:
            note = "  improved"
        lines.append(
            f"  {result.gate.metric:<34} {baseline:>9} {current:>9} {delta:>9}  {marker}{note}"
        )

    lines.append("  " + "-" * 72)
    if comparison.passed:
        lines.append(f"  PASS - {len(comparison.results)} gates within tolerance")
    else:
        lines.append(f"  FAIL - {len(comparison.failures)} gate(s) regressed:")
        lines.extend(
            f"      {result.gate.metric}: {result.reason}" for result in comparison.failures
        )
        lines += [
            "",
            "  If this regression is intentional, record it deliberately:",
            "      python scripts/check_regression.py --update-baseline "
            '--reason "why this is acceptable"',
        ]
    lines.append("")
    return "\n".join(lines)


def format_markdown(comparison: Comparison, report: dict[str, Any]) -> str:
    """A GitHub step summary, so a reviewer sees the numbers without opening logs."""
    verdict = "PASSED" if comparison.passed else "FAILED"
    lines = [
        f"## Evaluation gate: {verdict}",
        "",
        f"`{report.get('items_evaluated', '?')}` golden items · "
        f"baseline recorded {comparison.baseline_created_at or 'unknown'}",
        "",
        "| Metric | Baseline | Current | Delta | |",
        "|---|---:|---:|---:|:--|",
    ]

    for result in comparison.results:
        baseline = f"{result.baseline:.4f}" if result.baseline is not None else "—"
        current = f"{result.current:.4f}" if result.current is not None else "—"
        delta = f"{result.delta:+.4f}" if result.delta is not None else "—"
        status = "PASS" if result.passed else "**FAIL**"
        if result.improved:
            status = "improved"
        lines.append(f"| `{result.gate.metric}` | {baseline} | {current} | {delta} | {status} |")

    if not comparison.passed:
        lines += [
            "",
            "### Regressions",
            "",
            *(f"- `{r.gate.metric}` — {r.reason}" for r in comparison.failures),
            "",
            "If this is an intentional trade-off, update the baseline on this branch:",
            "",
            "```bash",
            'python scripts/check_regression.py --update-baseline --reason "why"',
            "```",
        ]
    return "\n".join(lines)
