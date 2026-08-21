#!/usr/bin/env python3
"""Samples real queries, runs the full pipeline (skips STT), reports retrieval-leg
vs full-pipeline percentiles separately, writes data/latency_results.csv."""
import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import settings
from src.latency import percentiles
from src.pipeline import run_pipeline


def _sample_queries(n: int) -> list[str]:
    from datasets import load_dataset
    path = f"train/{settings.hf_dataset_lang}train.parquet"
    ds = load_dataset(settings.hf_dataset, data_files=path, split="train")
    n = min(n, len(ds))
    return [ds[i]["query"] for i in range(n)]


def main():
    parser = argparse.ArgumentParser(description="Benchmark retrieval + full pipeline latency.")
    parser.add_argument("--n", type=int, default=50)
    args = parser.parse_args()

    queries = _sample_queries(args.n)
    retrieval_ms, total_ms, rows = [], [], []

    for query in queries:
        result = run_pipeline(text_query=query)
        r_ms = result.stage_timings_ms.get("retrieval", 0.0) + result.stage_timings_ms.get("ontopic_guardrail", 0.0)
        retrieval_ms.append(r_ms)
        total_ms.append(result.total_latency_ms)
        rows.append({"query": query, "retrieval_ms": r_ms, "total_ms": result.total_latency_ms})

    Path("data").mkdir(exist_ok=True)
    with open("data/latency_results.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["query", "retrieval_ms", "total_ms"])
        writer.writeheader()
        writer.writerows(rows)

    print("=== Retrieval leg only (embed + vector search + guardrail checks) ===")
    print(percentiles(retrieval_ms))
    print("\n=== Full pipeline (incl. LLM generation) ===")
    print(percentiles(total_ms))


if __name__ == "__main__":
    main()