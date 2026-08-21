"""Passage-native chunking: trusts MS MARCO's own passage segmentation."""
from src.chunking import Chunk


def chunk_row(row: dict) -> list[Chunk]:
    query_id = row["query_id"]
    passages = row["passages"]["Translated_passages"]
    is_selected = row["passages"].get("is_selected", [0] * len(passages))

    return [
        Chunk(
            text=passage,
            strategy="passage_native",
            doc_id=f"{query_id}_p{p_idx}",
            metadata={"query_id": query_id, "passage_idx": p_idx, "is_selected": is_selected[p_idx]},
        )
        for p_idx, passage in enumerate(passages)
        if passage.strip()
    ]