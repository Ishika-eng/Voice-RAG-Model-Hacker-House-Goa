"""Answer generation with Groq (fast Llama inference), constrained to the retrieved context."""
from tenacity import retry, stop_after_attempt, wait_exponential

from src.config import settings
from src.retrieval import RetrievedChunk

SYSTEM_PROMPT = (
    "You are a retrieval-grounded QA assistant. Answer ONLY using the numbered "
    "context passages given to you. If the passages do not contain the answer, "
    "say so plainly instead of guessing. Keep answers concise (2-4 sentences), "
    "and respond in the same language as the question."
)


def _build_prompt(query: str, retrieved: list[RetrievedChunk]) -> str:
    context_block = "\n".join(f"[{i+1}] {c.text}" for i, c in enumerate(retrieved))
    return (
        f"Context passages:\n{context_block}\n\n"
        f"Question: {query}\n\n"
        "Answer using only the context above. Cite passage numbers like [1] where relevant."
    )


class GenerationError(RuntimeError):
    pass


@retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=0.5, max=3))
def generate_answer(query: str, retrieved: list[RetrievedChunk]) -> str:
    if not settings.groq_api_key:
        raise GenerationError("GROQ_API_KEY is not configured")

    from groq import Groq

    client = Groq(api_key=settings.groq_api_key)
    try:
        response = client.chat.completions.create(
            model=settings.generation_model,
            max_tokens=400,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": _build_prompt(query, retrieved)},
            ],
        )
    except Exception as exc:  # noqa: BLE001
        raise GenerationError(f"Groq generation failed: {exc}") from exc

    answer = (response.choices[0].message.content or "").strip()
    if not answer:
        raise GenerationError("Empty generation response")
    return answer
