"""Semantic chunking: sentence-split (aware of Hindi '।'), embed each sentence,
greedily merge into a chunk while cosine sim to the running centroid stays >= threshold."""
import re
import numpy as np

from src.chunking import Chunk
from src.embeddings import embed_passages

SIM_THRESHOLD = 0.6
_SENTENCE_RE = re.compile(r"(?<=[।.!?])\s+")


def _split_sentences(text: str) -> list[str]:
    sentences = [s.strip() for s in _SENTENCE_RE.split(text) if s.strip()]
    return sentences or ([text.strip()] if text.strip() else [])


def chunk_row(row: dict) -> list[Chunk]:
    query_id = row["query_id"]
    passages = row["passages"]["Translated_passages"]
    is_selected = row["passages"].get("is_selected", [0] * len(passages))

    chunks: list[Chunk] = []
    for p_idx, passage in enumerate(passages):
        sentences = _split_sentences(passage)
        if not sentences:
            continue
        vectors = np.array(embed_passages(sentences))

        c_idx = 0
        cur_sents, cur_vecs = [sentences[0]], [vectors[0]]
        for sent, vec in zip(sentences[1:], vectors[1:]):
            centroid = np.mean(cur_vecs, axis=0)
            sim = float(np.dot(centroid, vec) / (np.linalg.norm(centroid) * np.linalg.norm(vec) + 1e-8))
            if sim >= SIM_THRESHOLD:
                cur_sents.append(sent)
                cur_vecs.append(vec)
            else:
                chunks.append(Chunk(
                    text=" ".join(cur_sents), strategy="semantic",
                    doc_id=f"{query_id}_p{p_idx}_c{c_idx}",
                    metadata={"query_id": query_id, "passage_idx": p_idx, "is_selected": is_selected[p_idx]},
                ))
                c_idx += 1
                cur_sents, cur_vecs = [sent], [vec]
        chunks.append(Chunk(
            text=" ".join(cur_sents), strategy="semantic",
            doc_id=f"{query_id}_p{p_idx}_c{c_idx}",
            metadata={"query_id": query_id, "passage_idx": p_idx, "is_selected": is_selected[p_idx]},
        ))
    return chunks