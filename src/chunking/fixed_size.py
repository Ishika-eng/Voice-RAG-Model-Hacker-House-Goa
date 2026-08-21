"""Fixed-size sliding-window chunking: baseline. 60-word window, 15-word overlap."""
from src.chunking import Chunk

WINDOW_WORDS = 60
OVERLAP_WORDS = 15


def chunk_row(row: dict) -> list[Chunk]:
    query_id = row["query_id"]
    passages = row["passages"]["Translated_passages"]
    is_selected = row["passages"].get("is_selected", [0] * len(passages))

    chunks: list[Chunk] = []
    for p_idx, passage in enumerate(passages):
        words = passage.split()
        if not words:
            continue
        step = WINDOW_WORDS - OVERLAP_WORDS
        start, c_idx = 0, 0
        while start < len(words):
            window = words[start:start + WINDOW_WORDS]
            chunks.append(Chunk(
                text=" ".join(window),
                strategy="fixed_size",
                doc_id=f"{query_id}_p{p_idx}_c{c_idx}",
                metadata={"query_id": query_id, "passage_idx": p_idx, "is_selected": is_selected[p_idx]},
            ))
            if start + WINDOW_WORDS >= len(words):
                break
            start += step
            c_idx += 1
    return chunks