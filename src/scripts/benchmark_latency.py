#!/usr/bin/env python3
"""Samples real queries, runs the full pipeline (skips STT), reports retrieval-leg
vs full-pipeline percentiles separately, writes data/latency_results.csv."""
import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.config import settings
from src.latency import percentiles
from src.pipeline import run_pipeline


def _sample_queries(n: int) -> list[str]:
    # Reads via pyarrow's iter_batches() directly rather than
    # datasets.load_dataset(): this file has a single ~780k-row row
    # group, and pyarrow can't concatenate the nested `passages`
    # struct/list column across the internal chunks that produces in
    # one shot (ArrowNotImplementedError: "Nested data conversions not
    # implemented for chunked array outputs"). Batched reads sidestep
    # it since no cross-batch concatenation is needed.
    import huggingface_hub
    import pyarrow.parquet as pq

    filename = f"train/{settings.hf_dataset_lang}train.parquet"
    local_path = huggingface_hub.hf_hub_download(settings.hf_dataset, filename, repo_type="dataset")
    pf = pq.ParquetFile(local_path)

    queries = []
    for batch in pf.iter_batches(batch_size=500, columns=["query"]):
        for row in batch.to_pylist():
            if row.get("query"):
                queries.append(row["query"])
            if len(queries) >= n:
                return queries
    return queries


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