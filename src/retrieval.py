"""Fan the query across all 3 Qdrant collections, merge with RRF (rank-based,
since scores aren't comparable across chunking strategies)."""
from dataclasses import dataclass

from src.config import settings
from src.embeddings import embed_query
from src.vector_store import STRATEGIES, search

RRF_K = 60
IS_SELECTED_BOOST = 0.05


@dataclass
class RetrievedChunk:
    text: str
    score: float
    strategy: str
    doc_id: str


def retrieve(query: str, top_k: int | None = None) -> list[RetrievedChunk]:
    top_k = top_k or settings.top_k
    query_vector = embed_query(query)

    fused: dict[str, dict] = {}
    for strategy in STRATEGIES:
        hits = search(strategy, query_vector, top_k=max(top_k * 2, 10))
        for rank, hit in enumerate(hits):
            payload = hit.payload or {}
            doc_id = payload.get("doc_id", str(hit.id))
            rrf_score = 1.0 / (RRF_K + rank + 1)
            if payload.get("is_selected"):
                rrf_score += IS_SELECTED_BOOST
            entry = fused.setdefault(doc_id, {
                "text": payload.get("text", ""), "strategy": strategy,
                "doc_id": doc_id, "rrf": 0.0, "raw_sim": hit.score,
            })
            entry["rrf"] += rrf_score
            entry["raw_sim"] = max(entry["raw_sim"], hit.score)

    ranked = sorted(fused.values(), key=lambda e: e["rrf"], reverse=True)[:top_k]
    return [RetrievedChunk(text=e["text"], score=e["raw_sim"], strategy=e["strategy"], doc_id=e["doc_id"]) for e in ranked]


def max_similarity(retrieved: list[RetrievedChunk]) -> float:
    return max((c.score for c in retrieved), default=0.0)