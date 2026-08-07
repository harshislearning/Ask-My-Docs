"""CLI: bootstrap a golden set from the indexed corpus.

    python scripts/make_golden.py --per-doc 8 --unanswerable 10

Writes *draft* items to review. It samples real chunks and asks the model to
write questions those chunks answer, so every draft arrives with its
``expected_sources`` already filled in from the chunk it came from - which is
the tedious, error-prone half of building a golden set.

The drafts are not a golden set until a human has read them. Questions a model
writes from a chunk tend to be answerable by that exact chunk in that exact
wording, which flatters retrieval. Rewrite them the way a colleague would
actually ask, and delete the ones that are really just the chunk read aloud.
"""

from __future__ import annotations

import argparse
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from askmydocs.config import load_config
from askmydocs.errors import AskMyDocsError
from askmydocs.evaluation.golden import ExpectedSource, GoldenItem, write_golden_set
from askmydocs.generation.groq_client import GroqClient
from askmydocs.ingestion.pipeline import read_chunks
from askmydocs.logging_setup import configure_logging
from askmydocs.models import Chunk

DRAFT_PROMPT = """\
You are helping build an evaluation set for a documentation search system.

Below is one passage from an internal technical document. Write {count} distinct \
questions that this passage answers, as an engineer looking for this information \
would actually phrase them.

Rules:
- Each question must be answerable from this passage alone.
- Do not mention "the passage", "the document" or "the text" in the question.
- Prefer specific questions about values, parameters and behaviour over vague ones.
- Vary the phrasing: do not simply reuse the passage's own wording.

Reply with one question per line and nothing else.

PASSAGE
{passage}\
"""

UNANSWERABLE_PROMPT = """\
You are helping build an evaluation set for a documentation search system.

The system indexes the documents listed below. Write {count} questions that are \
plausible things someone might ask this system, but that these documents almost \
certainly do NOT answer - adjacent topics, unrelated policies, information a \
different department would own.

Rules:
- Each question must sound like a genuine question for this kind of system.
- Do not write nonsense or trick questions.
- Do not ask about anything the listed topics cover.

Reply with one question per line and nothing else.

DOCUMENTS
{documents}\
"""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Draft golden-set questions from the corpus")
    parser.add_argument("--config", help="Path to a config YAML")
    parser.add_argument("--output", help="Where to write drafts (default: eval/golden/drafts.jsonl)")
    parser.add_argument("--per-doc", type=int, default=6, help="Questions per document")
    parser.add_argument(
        "--questions-per-chunk", type=int, default=2, help="Questions per sampled chunk"
    )
    parser.add_argument(
        "--unanswerable", type=int, default=10, help="How many unanswerable questions to draft"
    )
    parser.add_argument("--seed", type=int, default=0, help="Sampling seed, for reproducibility")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        config = load_config(args.config)
    except AskMyDocsError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    config.logging.level = "WARNING"
    configure_logging(config.logging)

    chunks = read_chunks(config.paths.chunks_file)
    if not chunks:
        print("error: no chunks found - run scripts/ingest.py first", file=sys.stderr)
        return 1

    try:
        client = GroqClient(config.generation, config.groq_api_key)
    except AskMyDocsError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    random.seed(args.seed)
    items: list[GoldenItem] = []

    for chunk in _sample(chunks, args.per_doc, args.questions_per_chunk):
        items.extend(_draft_for_chunk(client, chunk, args.questions_per_chunk, len(items)))
        print(f"  drafted from {chunk.source_file} {chunk.page_label}")

    if args.unanswerable:
        items.extend(_draft_unanswerable(client, chunks, args.unanswerable, len(items)))
        print(f"  drafted {args.unanswerable} unanswerable questions")

    output = Path(args.output) if args.output else config.paths.raw_pdfs.parent.parent / (
        "eval/golden/drafts.jsonl"
    )
    write_golden_set(output, items)

    print(f"\n{len(items)} drafts written to {output}")
    print("\nNEXT: read every line before using it.")
    print("  - rewrite questions the way a colleague would actually ask")
    print("  - delete any that just read the chunk back")
    print("  - fill in expected_answer and must_contain for the exact values")
    print("  - check each expected_sources page against the real PDF")
    print(f"  - then rename it to {config.evaluation.golden_set}")
    return 0


def _sample(chunks: list[Chunk], per_doc: int, per_chunk: int) -> list[Chunk]:
    """Spread the sample across documents rather than over-fitting to the longest."""
    by_document: dict[str, list[Chunk]] = {}
    for chunk in chunks:
        by_document.setdefault(chunk.doc_id, []).append(chunk)

    sampled: list[Chunk] = []
    wanted = max(1, per_doc // max(per_chunk, 1))
    for document_chunks in by_document.values():
        # Prefer substantial chunks: a two-line fragment yields a trivial question.
        candidates = sorted(document_chunks, key=lambda c: c.token_count, reverse=True)
        pool = candidates[: max(wanted * 3, wanted)]
        sampled.extend(random.sample(pool, min(wanted, len(pool))))
    return sampled


def _draft_for_chunk(
    client: GroqClient, chunk: Chunk, count: int, offset: int
) -> list[GoldenItem]:
    messages = [
        {"role": "user", "content": DRAFT_PROMPT.format(count=count, passage=chunk.text)}
    ]
    try:
        response = client.complete(messages)
    except AskMyDocsError as exc:
        print(f"  ! skipped {chunk.chunk_id}: {exc}", file=sys.stderr)
        return []

    return [
        GoldenItem(
            id=f"q{offset + index + 1:03d}",
            question=question,
            expected_sources=[
                ExpectedSource(source_file=chunk.source_file, page=chunk.page_start)
            ],
            answerable=True,
            tags=["draft", *(["table"] if chunk.content_type == "table" else [])],
        )
        for index, question in enumerate(_questions(response.text, count))
    ]


def _draft_unanswerable(
    client: GroqClient, chunks: list[Chunk], count: int, offset: int
) -> list[GoldenItem]:
    documents = "\n".join(
        sorted({f"- {chunk.doc_title} ({chunk.source_file})" for chunk in chunks})
    )
    messages = [
        {
            "role": "user",
            "content": UNANSWERABLE_PROMPT.format(count=count, documents=documents),
        }
    ]
    try:
        response = client.complete(messages)
    except AskMyDocsError as exc:
        print(f"  ! skipped unanswerable batch: {exc}", file=sys.stderr)
        return []

    return [
        GoldenItem(
            id=f"q{offset + index + 1:03d}",
            question=question,
            answerable=False,
            tags=["draft", "unanswerable"],
        )
        for index, question in enumerate(_questions(response.text, count))
    ]


def _questions(text: str, limit: int) -> list[str]:
    """Pull questions out of the reply, tolerating numbering and bullets."""
    questions: list[str] = []
    for line in text.splitlines():
        cleaned = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", line).strip().strip('"')
        if len(cleaned.split()) >= 4:
            questions.append(cleaned)
    return questions[:limit]


if __name__ == "__main__":
    raise SystemExit(main())
