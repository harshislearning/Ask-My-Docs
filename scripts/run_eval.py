"""CLI: evaluate the system against the golden set.

    python scripts/run_eval.py
    python scripts/run_eval.py --no-generate      # retrieval metrics only, no API calls
    python scripts/run_eval.py --limit 10         # quick smoke run
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from askmydocs.config import load_config
from askmydocs.errors import AskMyDocsError
from askmydocs.evaluation import run_evaluation
from askmydocs.logging_setup import configure_logging


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate retrieval and generation")
    parser.add_argument("--config", help="Path to a config YAML")
    parser.add_argument("--golden-set", help="Path to a golden set JSONL")
    parser.add_argument("--limit", type=int, help="Evaluate only the first N items")
    parser.add_argument(
        "--no-generate",
        action="store_true",
        help="Skip generation: retrieval metrics only, no API calls",
    )
    parser.add_argument("--output-dir", help="Where to write the run artifact")
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

    try:
        report = run_evaluation(
            config,
            golden_set=args.golden_set,
            limit=args.limit,
            generate=not args.no_generate,
            output_dir=Path(args.output_dir) if args.output_dir else None,
        )
    except (AskMyDocsError, FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    _print_report(report)
    return 0


def _print_report(report: dict) -> None:
    stats = report["golden_stats"]
    print("\nEvaluation report")
    print("=" * 68)
    print(
        f"  {report['items_evaluated']} items "
        f"({stats['answerable']} answerable, {stats['unanswerable']} unanswerable)"
        f"  ·  {report['seconds']}s"
    )

    if report["coverage"]["unreachable"]:
        # A label pointing at a page that was never ingested scores as a
        # permanent retrieval miss and looks exactly like a retrieval bug.
        print(f"\n  ! {report['coverage']['unreachable']} expected sources are not in the index")
        for example in report["coverage"]["unreachable_examples"]:
            print(f"      {example}")

    _table("Retrieval (after fusion)", report.get("retrieval"))
    _table("Retrieval (after reranking)", report.get("reranked"))
    _table("Generation", report.get("generation"))

    refusal = report.get("refusal")
    if refusal:
        print("\n  Refusal")
        print("  " + "-" * 66)
        print(f"    correctly refused    {refusal['correctly_refused']}")
        print(f"    wrongly answered     {refusal['wrongly_answered']}   <- misleading answers")
        print(f"    wrongly refused      {refusal['wrongly_refused']}   <- lost usefulness")
        print(f"    correctly answered   {refusal['correctly_answered']}")

    ragas = report.get("ragas") or {}
    if ragas.get("scores"):
        _table("RAGAS", ragas["scores"])
    elif ragas.get("reason"):
        print(f"\n  RAGAS: skipped ({ragas['reason']})")

    if report["errors"]:
        print(f"\n  ! {len(report['errors'])} items errored: {', '.join(report['errors'][:5])}")
    print()


def _table(title: str, metrics: dict | None) -> None:
    if not metrics:
        return
    print(f"\n  {title}")
    print("  " + "-" * 66)
    for name, value in sorted(metrics.items()):
        bar = "#" * round(float(value) * 30) if 0 <= float(value) <= 1 else ""
        print(f"    {name:<24} {float(value):>6.3f}  {bar}")


if __name__ == "__main__":
    raise SystemExit(main())
