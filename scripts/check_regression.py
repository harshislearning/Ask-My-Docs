"""CLI: run the evaluation and fail if it regressed against the baseline.

    python scripts/check_regression.py --no-generate        # what CI runs
    python scripts/check_regression.py --update-baseline --reason "..."

Exit codes: 0 passed, 1 a gate regressed, 2 could not run.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from askmydocs.config import load_config
from askmydocs.errors import AskMyDocsError
from askmydocs.evaluation import run_evaluation
from askmydocs.evaluation.baseline import (
    Gate,
    compare_to_baseline,
    format_comparison,
    format_markdown,
    load_baseline,
    write_baseline,
)
from askmydocs.logging_setup import configure_logging


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Gate a build on evaluation metrics")
    parser.add_argument("--config", help="Path to a config YAML")
    parser.add_argument("--golden-set", help="Path to a golden set JSONL")
    parser.add_argument("--baseline", help="Path to the baseline JSON")
    parser.add_argument(
        "--no-generate",
        action="store_true",
        help="Retrieval metrics only: no API key, no cost, deterministic",
    )
    parser.add_argument(
        "--update-baseline",
        action="store_true",
        help="Record this run as the new baseline instead of gating on it",
    )
    parser.add_argument(
        "--reason", default="", help="Why the baseline is being updated (recorded in the file)"
    )
    parser.add_argument("--run-file", help="Compare a saved run instead of evaluating again")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        config = load_config(args.config)
    except AskMyDocsError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    config.logging.level = args.log_level or "WARNING"
    configure_logging(config.logging)

    baseline_path = Path(args.baseline) if args.baseline else config.evaluation.baseline_file

    try:
        report = _get_report(args, config)
    except (AskMyDocsError, FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.update_baseline:
        if not args.reason:
            # A baseline without a stated reason is indistinguishable from one
            # updated to make a red build green.
            print(
                "error: --update-baseline requires --reason explaining why the "
                "new numbers are acceptable",
                file=sys.stderr,
            )
            return 2
        write_baseline(baseline_path, report, args.reason)
        print(f"\nBaseline updated: {baseline_path}")
        print(f"  reason: {args.reason}")
        print("  Commit this file with the change that caused it.\n")
        return 0

    try:
        baseline = load_baseline(baseline_path)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    gates = [Gate(**entry) for entry in config.evaluation.gates]
    if not gates:
        print("error: no gates configured under evaluation.gates", file=sys.stderr)
        return 2

    comparison = compare_to_baseline(report, baseline, gates, str(baseline_path))
    print(format_comparison(comparison))
    _write_step_summary(comparison, report)

    return 0 if comparison.passed else 1


def _get_report(args: argparse.Namespace, config: object) -> dict:
    if args.run_file:
        payload = json.loads(Path(args.run_file).read_text(encoding="utf-8-sig"))
        # Accept either a bare report or a full run artifact.
        return payload.get("report", payload)  # type: ignore[no-any-return]

    return run_evaluation(
        config,  # type: ignore[arg-type]
        golden_set=args.golden_set,
        generate=not args.no_generate,
    )


def _write_step_summary(comparison: object, report: dict) -> None:
    """Put the table in the PR's checks tab, not only in the log."""
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if not summary_path:
        return
    with open(summary_path, "a", encoding="utf-8") as handle:
        handle.write(format_markdown(comparison, report))  # type: ignore[arg-type]
        handle.write("\n")


if __name__ == "__main__":
    raise SystemExit(main())
