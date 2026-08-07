"""CLI: write the synthetic corpus CI evaluates against.

    python scripts/make_ci_corpus.py --output .ci/raw_pdfs

CI cannot use your real documents - they are not in the repository and should
not be. It uses the same PDFs the test suite builds, generated at run time
rather than committed as binaries, so the corpus is reproducible and the
typography that drives every assertion stays visible in Python.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "tests"))

from fixtures import pdf_factory as pf  # noqa: E402  (needs the sys.path above)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate the CI evaluation corpus")
    parser.add_argument("--output", default=".ci/raw_pdfs", help="Where to write the PDFs")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    # The same three documents the golden set is written against, plus the two
    # broken ones - so CI also proves that a corrupt or scanned file is skipped
    # rather than taking the run down.
    written = [
        pf.structured_pdf(output / "deployment_handbook.pdf"),
        pf.table_pdf(output / "config_reference.pdf"),
        pf.unstructured_pdf(output / "field_notes.pdf"),
        pf.scanned_pdf(output / "scanned_scan.pdf"),
        pf.corrupt_pdf(output / "broken.pdf"),
    ]

    for path in written:
        print(f"  {path.name}")
    print(f"\n{len(written)} files written to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
