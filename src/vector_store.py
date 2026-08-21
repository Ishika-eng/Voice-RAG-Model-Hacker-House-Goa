"""Qdrant wrapper: one collection per chunking strategy (scores aren't
comparable across strategies -- see retrieval.py's rank-based RRF)."""
import hashlib
from functools import lru_cache

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from src.chunking import Chunk
from src.config import settings
from src.embeddings import embedding_dim

STRATEGIES = ("fixed_size", "semantic", "passage_native")


@lru_cache(maxsize=1)
def _client() -> QdrantClient:
    return QdrantClient(path=settings.qdrant_path)


def _collection_name(strategy: str) -> str:
    return f"msmarco_xi_{strategy}"


def _stable_id(doc_id: str) -> int:
    return int(hashlib.sha1(doc_id.encode()).hexdigest()[:16], 16)


def ensure_collections() -> None:
    client = _client()
    dim = embedding_dim()
    existing = {c.name for c in client.get_collections().collections}
    for strategy in STRATEGIES:
        name = _collection_name(strategy)
        if name not in existing:
            client.create_collection(
                collection_name=name,
                vectors_config=qmodels.VectorParams(size=dim, distance=qmodels.Distance.COSINE),
            )


def upsert_chunks(strategy: str, chunks: list[Chunk], vectors: list[list[float]]) -> None:
    if not chunks:
        return
    ensure_collections()
    points = [
        qmodels.PointStruct(
            id=_stable_id(chunk.doc_id),
            vector=vector,
            payload={"text": chunk.text, "doc_id": chunk.doc_id, **chunk.metadata},
        )
        for chunk, vector in zip(chunks, vectors)
    ]
    _client().upsert(collection_name=_collection_name(strategy), points=points)


def search(strategy: str, query_vector: list[float], top_k: int):
    return _client().search(collection_name=_collection_name(strategy), query_vector=query_vector, limit=top_k)