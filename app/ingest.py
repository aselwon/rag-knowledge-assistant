"""Ingest a directory: python -m app.ingest data/handbook."""

import argparse
import json
from pathlib import Path

from app.config import get_settings
from app.db import initialize
from app.service import RAGService


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, nargs="?", default=Path("data/handbook"))
    args = parser.parse_args()
    settings = get_settings()
    initialize(settings)
    service = RAGService(settings)
    files = sorted(p for p in args.directory.rglob("*") if p.suffix.lower() in {".md", ".txt"})
    if not files:
        parser.error("No Markdown/text documents found")
    for path in files:
        # Relative paths preserve identity for nested directories.
        print(
            json.dumps(
                service.ingest(
                    path.relative_to(args.directory).as_posix(),
                    path.read_text(encoding="utf-8-sig"),
                )
            )
        )


if __name__ == "__main__":
    main()
