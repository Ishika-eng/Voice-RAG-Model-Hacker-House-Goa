"""Guardrails applied on both sides of generation:

- pre-retrieval: block unsafe/inappropriate input outright.
- post-retrieval: if nothing relevant came back, refuse instead of
  guessing (off-topic detection).
- post-generation: check the answer is actually grounded in the retrieved
  context (lexical overlap heuristic) before returning it -- a cheap,
  dependency-free stand-in for an NLI entailment model, fast enough to sit
  on the latency-critical path.
"""
import re
from dataclasses import dataclass

from src.config import settings
from src.retrieval import RetrievedChunk

_UNSAFE_PATTERNS = [
    r"\bhow to (make|build|synthesize)\b.*\b(bomb|explosive|weapon|poison)\b",
    r"\b(kill|harm|hurt)\s+(myself|yourself|someone)\b",
    r"\bchild\s+(sexual|abuse)\b",
    r"\bhack\s+into\b",
]
_UNSAFE_RE = re.compile("|".join(_UNSAFE_PATTERNS), re.IGNORECASE)

_WORD_RE = re.compile(r"[\wऀ-ॿ]+", re.UNICODE)


@dataclass
class GuardrailResult:
    allowed: bool
    reason: str | None = None


def check_input_safety(query: str) -> GuardrailResult:
    if not query or not query.strip():
        return GuardrailResult(False, "empty_query")
    if _UNSAFE_RE.search(query):
        return GuardrailResult(False, "unsafe_content")
    if len(query) > 2000:
        return GuardrailResult(False, "query_too_long")
    return GuardrailResult(True)


def check_on_topic(retrieved: list[RetrievedChunk]) -> GuardrailResult:
    if not retrieved:
        return GuardrailResult(False, "no_results")
    top_score = max(c.score for c in retrieved)
    if top_score < settings.offtopic_sim_threshold:
        return GuardrailResult(False, "off_topic_low_similarity")
    return GuardrailResult(True)


def _tokenize(text: str) -> set[str]:
    return {w.lower() for w in _WORD_RE.findall(text) if len(w) > 2}


def check_groundedness(answer: str, retrieved: list[RetrievedChunk]) -> GuardrailResult:
    """Fraction of the answer's content words that also appear somewhere
    in the retrieved context. Cheap, explainable, no extra model call --
    trades recall for latency, which is the right trade-off under a
    200ms retrieval budget."""
    if not answer or not answer.strip():
        return GuardrailResult(False, "empty_answer")

    answer_tokens = _tokenize(answer)
    if not answer_tokens:
        return GuardrailResult(False, "empty_answer")

    context_tokens: set[str] = set()
    for c in retrieved:
        context_tokens |= _tokenize(c.text)

    overlap = len(answer_tokens & context_tokens) / len(answer_tokens)
    if overlap < settings.groundedness_overlap_threshold:
        return GuardrailResult(False, f"ungrounded_overlap_{overlap:.2f}")
    return GuardrailResult(True, f"overlap_{overlap:.2f}")


REFUSAL_MESSAGE = (
    "I don't have enough grounded information in the indexed dataset to answer that "
    "confidently, so I'd rather not guess."
)
