"""Orchestration harness. Structured stages, typed input/output at each
boundary, retries on transient stage failures (STT/generation already
retry internally via tenacity), and explicit error-recovery paths that
degrade to a refusal response instead of raising to the caller.

Stages:
  audio/text -> STT -> input guardrail -> retrieval -> on-topic guardrail
  -> generation -> groundedness guardrail -> PipelineResult
"""
from dataclasses import dataclass, field

from src.generation import GenerationError, generate_answer
from src.guardrails import (
    REFUSAL_MESSAGE,
    check_groundedness,
    check_input_safety,
    check_on_topic,
)
from src.latency import StageTimer
from src.retrieval import RetrievedChunk, retrieve
from src.stt import STTError, transcribe


@dataclass
class PipelineResult:
    query_text: str = ""
    answer: str = ""
    refused: bool = False
    refusal_reason: str | None = None
    retrieved: list[RetrievedChunk] = field(default_factory=list)
    stage_timings_ms: dict[str, float] = field(default_factory=dict)
    total_latency_ms: float = 0.0
    error: str | None = None

    def to_dict(self) -> dict:
        return {
            "query_text": self.query_text,
            "answer": self.answer,
            "refused": self.refused,
            "refusal_reason": self.refusal_reason,
            "retrieved": [
                {"text": c.text, "score": c.score, "strategy": c.strategy, "doc_id": c.doc_id}
                for c in self.retrieved
            ],
            "stage_timings_ms": self.stage_timings_ms,
            "total_latency_ms": self.total_latency_ms,
            "error": self.error,
        }


def _refuse(result: PipelineResult, reason: str) -> PipelineResult:
    result.refused = True
    result.refusal_reason = reason
    result.answer = REFUSAL_MESSAGE
    return result


def run_pipeline(
    audio_bytes: bytes | None = None,
    text_query: str | None = None,
    language_code: str = "hi-IN",
    top_k: int | None = None,
) -> PipelineResult:
    result = PipelineResult()
    timer = StageTimer()

    # --- Stage: STT (skipped if a text query was passed directly, e.g. benchmarking) ---
    if audio_bytes is not None:
        try:
            with timer.stage("stt"):
                result.query_text = transcribe(audio_bytes, language_code=language_code)
        except STTError as exc:
            result.error = f"stt_failed: {exc}"
            result.stage_timings_ms = timer.timings_ms
            return _refuse(result, "stt_failed")
    else:
        result.query_text = (text_query or "").strip()

    # --- Stage: input guardrail ---
    with timer.stage("input_guardrail"):
        input_check = check_input_safety(result.query_text)
    if not input_check.allowed:
        result.stage_timings_ms = timer.timings_ms
        result.total_latency_ms = timer.total_ms
        return _refuse(result, input_check.reason)

    # --- Stage: retrieval ---
    try:
        with timer.stage("retrieval"):
            result.retrieved = retrieve(result.query_text, top_k=top_k)
    except Exception as exc:  # noqa: BLE001
        result.error = f"retrieval_failed: {exc}"
        result.stage_timings_ms = timer.timings_ms
        result.total_latency_ms = timer.total_ms
        return _refuse(result, "retrieval_failed")

    # --- Stage: on-topic guardrail ---
    with timer.stage("ontopic_guardrail"):
        topic_check = check_on_topic(result.retrieved)
    if not topic_check.allowed:
        result.stage_timings_ms = timer.timings_ms
        result.total_latency_ms = timer.total_ms
        return _refuse(result, topic_check.reason)

    # --- Stage: generation ---
    try:
        with timer.stage("generation"):
            answer = generate_answer(result.query_text, result.retrieved)
    except GenerationError as exc:
        result.error = f"generation_failed: {exc}"
        result.stage_timings_ms = timer.timings_ms
        result.total_latency_ms = timer.total_ms
        return _refuse(result, "generation_failed")

    # --- Stage: groundedness guardrail ---
    with timer.stage("groundedness_guardrail"):
        grounded_check = check_groundedness(answer, result.retrieved)
    if not grounded_check.allowed:
        result.stage_timings_ms = timer.timings_ms
        result.total_latency_ms = timer.total_ms
        return _refuse(result, grounded_check.reason)

    result.answer = answer
    result.stage_timings_ms = timer.timings_ms
    result.total_latency_ms = timer.total_ms
    return result
