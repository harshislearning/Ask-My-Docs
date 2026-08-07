"""CLI: ask a question against the indexed knowledge base.

    python scripts/ask.py "what is the default request timeout?"

Runs the whole path - hybrid retrieval, RRF fusion, cross-encoder reranking,
Groq generation - and prints the answer with its numbered sources. Requires
GROQ_API_KEY in .env or the environment.
"""

from __future__ import annotations

import argparse
import sys
import textwrap
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from askmydocs.config import load_config
from askmydocs.errors import AskMyDocsError
from askmydocs.generation import Answerer
from askmydocs.logging_setup import configure_logging
from askmydocs.models import Answer
from askmydocs.retrieval import RetrievalPipeline
from askmydocs.verification import Verifier


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Ask a question against your documents")
    parser.add_argument("question", help="The question to answer")
    parser.add_argument("--config", help="Path to a config YAML")
    parser.add_argument(
        "--show-sources",
        action="store_true",
        help="Print the full text of every source, not just its heading",
    )
    parser.add_argument(
        "--no-rerank", action="store_true", help="Skip cross-encoder reranking"
    )
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    parser.add_argument("--log-format", choices=["console", "json"])
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        config = load_config(args.config)
    except AskMyDocsError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    config.logging.level = args.log_level or "WARNING"
    if args.log_format:
        config.logging.format = args.log_format
    if args.no_rerank:
        config.retrieval.rerank_enabled = False
    configure_logging(config.logging)

    try:
        pipeline = RetrievalPipeline.from_config(config)
        answerer = Answerer.from_config(config)
        candidates = pipeline.search(args.question)
        answer = answerer.answer(args.question, candidates)
        # The judge reuses the generation client, so llm mode needs no extra setup.
        answer = Verifier(config.verification, client=answerer.client).verify(answer)
    except AskMyDocsError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    _print_answer(answer, show_text=args.show_sources)
    # A failed check is worth a non-zero exit so scripted callers notice.
    return 0 if answer.verification is None or answer.verification.passed else 3


def _print_answer(answer: Answer, *, show_text: bool) -> None:
    print("\n" + "=" * 78)
    print(f"Q: {answer.question}")
    print("=" * 78 + "\n")
    print(textwrap.fill(answer.text, width=78, replace_whitespace=False))

    if answer.refused:
        print("\n  (the model declined to answer from these sources)")

    if answer.sources:
        print("\n" + "-" * 78)
        print("SOURCES")
        for source in answer.sources:
            score = f"  rerank {source.rerank_score:+.2f}" if source.rerank_score else ""
            print(f"\n  [{source.number}] {source.label}{score}")
            if show_text:
                print(textwrap.indent(textwrap.fill(source.text, width=72), "      "))

    report = answer.verification
    if report is not None:
        print("\n" + "-" * 78)
        verdict = "PASSED" if report.passed else "FAILED"
        print(
            f"VERIFICATION: {verdict}"
            f"   citation precision {report.citation_precision:.0%}"
            f"   claim coverage {report.claim_coverage:.0%}"
            f"   ({report.cited_claims}/{report.total_claims} claims cited)"
        )
        for issue in report.issues:
            print(f"\n  ! {issue.type}: {issue.detail}")
            if issue.sentence:
                print(textwrap.indent(textwrap.fill(issue.sentence, width=70), "    > "))
        if report.unused_sources:
            unused = ", ".join(f"[{n}]" for n in report.unused_sources)
            print(f"\n  unused sources: {unused}")

    print("\n" + "-" * 78)
    usage = f"{answer.prompt_tokens or '?'} in / {answer.completion_tokens or '?'} out"
    print(
        f"  {answer.model}  |  {usage} tokens  |  {answer.latency_ms:.0f} ms"
        f"  |  {len(answer.sources)} sources\n"
    )


if __name__ == "__main__":
    raise SystemExit(main())
