"""Enrichment pass: token count, has-numbers flag, keywords -- attached as
Qdrant payload, used for is_selected retrieval boost + off-topic guardrail."""
import re
from src.chunking import Chunk

_NUMBER_RE = re.compile(r"\d")
_WORD_RE = re.compile(r"[\wऀ-ॿ]+", re.UNICODE)
_STOPWORDS = {"है", "के", "की", "का", "में", "से", "को", "और", "यह", "एक", "थे", "था"}


def _keywords(text: str, top_n: int = 5) -> list[str]:
    words = [w.lower() for w in _WORD_RE.findall(text) if len(w) > 2 and w.lower() not in _STOPWORDS]
    seen, out = set(), []
    for w in sorted(words, key=len, reverse=True):
        if w not in seen:
            seen.add(w)
            out.append(w)
        if len(out) >= top_n:
            break
    return out


def enrich(chunk: Chunk) -> Chunk:
    chunk.metadata["token_count"] = len(chunk.text.split())
    chunk.metadata["has_numbers"] = bool(_NUMBER_RE.search(chunk.text))
    chunk.metadata["keywords"] = _keywords(chunk.text)
    return chunk


def enrich_all(chunks: list[Chunk]) -> list[Chunk]:
    return [enrich(c) for c in chunks]