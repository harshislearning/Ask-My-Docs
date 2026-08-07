"""CLI: ingest a folder of PDFs into chunks.jsonl.

    python scripts/ingest.py --input data/raw_pdfs

Ingestion is a batch job, not a request handler, so it lives outside the API
and can run without booting FastAPI or loading any ML model.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running straight from a checkout without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from askmydocs.config import load_config
from askmydocs.errors import AskMyDocsError
from askmydocs.ingestion import IngestionPipeline
from askmydocs.logging_setup import configure_logging
from askmydocs.models import DocumentStatus, IngestionManifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Ingest PDFs into chunks.jsonl")
    parser.add_argument("--config", help="Path to a config YAML (default: config/default.yaml)")
    parser.add_argument("--input", help="Folder of PDFs (default: paths.raw_pdfs from config)")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-parse every document, ignoring the unchanged-file cache",
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

    if args.log_level:
        config.logging.level = args.log_level
    if args.log_format:
        config.logging.format = args.log_format
    configure_logging(config.logging)

    pipeline = IngestionPipeline(config)
    manifest = pipeline.run(Path(args.input) if args.input else None, force=args.force)

    _print_report(manifest)
    # Failures are worth a non-zero exit so CI notices; skips are expected.
    return 1 if manifest.by_status(DocumentStatus.FAILED) else 0


def _print_report(manifest: IngestionManifest) -> None:
    summary = manifest.summary()
    print("\nIngestion report")
    print("-" * 68)
    for record in manifest.documents:
        marker = {
            DocumentStatus.OK: "ok      ",
            DocumentStatus.UNCHANGED: "cached  ",
            DocumentStatus.SKIPPED: "skipped ",
            DocumentStatus.FAILED: "FAILED  ",
        }[record.status]
        detail = f"{record.chunk_count:>5} chunks"
        if record.status is DocumentStatus.OK:
            source = record.structure_source.value if record.structure_source else "-"
            detail += f"  [{source}, {record.heading_count} headings]"
        elif record.reason:
            detail = f"        {record.reason}"
        print(f"  {marker} {record.source_file[:34]:<34} {detail}")

    print("-" * 68)
    print(
        f"  {summary['documents_ok']} ingested, "
        f"{summary['documents_unchanged']} cached, "
        f"{summary['documents_skipped']} skipped, "
        f"{summary['documents_failed']} failed"
    )
    print(f"  {summary['chunks_total']} chunks -> {manifest.chunks_file}\n")


if __name__ == "__main__":
    raise SystemExit(main())
