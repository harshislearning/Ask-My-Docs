"""CLI: build the FAISS and BM25 indexes from chunks.jsonl.

    python scripts/build_index.py

Requires the `ml` extra (sentence-transformers, faiss-cpu, rank_bm25). The
first run downloads the embedding model.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from askmydocs.config import load_config
from askmydocs.errors import AskMyDocsError
from askmydocs.indexing import IndexBuilder
from askmydocs.logging_setup import configure_logging


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build FAISS + BM25 indexes")
    parser.add_argument("--config", help="Path to a config YAML (default: config/default.yaml)")
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

    try:
        manifest = IndexBuilder(config).build()
    except AskMyDocsError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print("\nIndex build complete")
    print("-" * 68)
    print(f"  chunks       {manifest.chunk_count} from {manifest.document_count} documents")
    print(f"  embeddings   {manifest.embedding_model} ({manifest.dimension} dims)")
    print(f"  bm25         k1={manifest.bm25_k1} b={manifest.bm25_b}")
    print(f"  written to   {config.paths.indexes}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
