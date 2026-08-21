"""Dataset -> chunks (3 strategies) -> enrichment -> embeddings -> Qdrant."""
from src.chunking import Chunk
from src.chunking.fixed_size import chunk_row as chunk_fixed_size
from src.chunking.metadata_aware import enrich_all
from src.chunking.passage_native import chunk_row as chunk_passage_native
from src.chunking.semantic import chunk_row as chunk_semantic
from src.config import settings
from src.embeddings import embed_passages
from src.vector_store import upsert_chunks


def build_chunks_for_row(row: dict) -> list[Chunk]:
    chunks = chunk_fixed_size(row) + chunk_passage_native(row) + chunk_semantic(row)
    return enrich_all(chunks)


def ingest_dataset(limit: int | None = None) -> dict[str, int]:
    from datasets import load_dataset

    limit = limit or settings.ingest_limit
    path = f"train/{settings.hf_dataset_lang}train.parquet"
    ds = load_dataset(settings.hf_dataset, data_files=path, split="train")
    if limit:
        ds = ds.select(range(min(limit, len(ds))))

    by_strategy: dict[str, list[Chunk]] = {"fixed_size": [], "passage_native": [], "semantic": []}
    for row in ds:
        for chunk in build_chunks_for_row(row):
            by_strategy[chunk.strategy].append(chunk)

    counts = {}
    for strategy, chunks in by_strategy.items():
        if chunks:
            vectors = embed_passages([c.text for c in chunks])
            upsert_chunks(strategy, chunks, vectors)
        counts[strategy] = len(chunks)
    return counts