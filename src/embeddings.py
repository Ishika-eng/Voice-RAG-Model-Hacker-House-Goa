"""Shared multilingual-e5 embedding model. E5 needs a 'query:'/'passage:'
prefix and normalized embeddings -- both handled here."""
from functools import lru_cache
from sentence_transformers import SentenceTransformer
from src.config import settings


@lru_cache(maxsize=1)
def _model() -> SentenceTransformer:
    return SentenceTransformer(settings.embedding_model)


def embed_passages(texts: list[str]) -> list[list[float]]:
    prefixed = [f"passage: {t}" for t in texts]
    return _model().encode(prefixed, normalize_embeddings=True, show_progress_bar=False).tolist()


def embed_query(text: str) -> list[float]:
    return _model().encode(f"query: {text}", normalize_embeddings=True, show_progress_bar=False).tolist()


def embedding_dim() -> int:
    return _model().get_sentence_embedding_dimension()