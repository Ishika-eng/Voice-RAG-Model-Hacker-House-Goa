"""Speech-to-text via Sarvam (chosen over ElevenLabs because MSMARCO-XI is
an Indic-language corpus and Sarvam's saaras models are tuned for Indic
languages + code-switched speech).
"""
import io

from tenacity import retry, stop_after_attempt, wait_exponential

from src.config import settings


class STTError(RuntimeError):
    pass


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=0.5, max=4))
def transcribe(audio_bytes: bytes, filename: str = "audio.wav", language_code: str = "hi-IN") -> str:
    if not settings.sarvam_api_key:
        raise STTError("SARVAM_API_KEY is not configured")

    from sarvamai import SarvamAI

    client = SarvamAI(api_subscription_key=settings.sarvam_api_key)
    audio_file = io.BytesIO(audio_bytes)
    audio_file.name = filename

    try:
        response = client.speech_to_text.transcribe(
            file=audio_file,
            model="saaras:v3",
            language_code=language_code,
        )
    except Exception as exc:  # noqa: BLE001 - normalize all SDK failures
        raise STTError(f"Sarvam transcription failed: {exc}") from exc

    transcript = getattr(response, "transcript", "") or ""
    if not transcript.strip():
        raise STTError("Empty transcript returned")
    return transcript.strip()
