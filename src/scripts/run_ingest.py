#!/usr/bin/env python3
"""CLI: dataset -> chunks -> embeddings -> Qdrant."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.ingest import ingest_dataset


def main():
    parser = argparse.ArgumentParser(description="Ingest MSMARCO-XI into Qdrant.")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    counts = ingest_dataset(limit=args.limit)
    print("Ingest complete:")
    for strategy, n in counts.items():
        print(f"  {strategy}: {n} chunks")


if __name__ == "__main__":
    main()