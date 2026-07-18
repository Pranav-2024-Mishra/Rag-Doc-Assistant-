#!/usr/bin/env python
"""Standalone CLI to (re)ingest the corpus directory into the vector store.

Usage:
    python scripts/ingest.py                  # ingest ./corpus (default)
    python scripts/ingest.py --dir path/to/docs
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.ingestion import ingest_directory  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Ingest technical docs into the vector store")
    parser.add_argument("--dir", default=None, help="Directory of .md/.txt files (defaults to CORPUS_DIR / ./corpus)")
    args = parser.parse_args()

    result = ingest_directory(args.dir)
    print(f"\nIngested {result['files_ingested']} files -> {result['chunks_indexed']} chunks total\n")
    for src, count in result["chunks_by_source"].items():
        print(f"  - {src}: {count} chunks")


if __name__ == "__main__":
    main()
