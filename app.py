import logging

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from src.pipeline import run_pipeline
from src.rate_limit import enforce_rate_limit

log = logging.getLogger("app")

app = FastAPI(title="Voice RAG - MSMARCO-XI")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="frontend"), name="static")

MAX_AUDIO_BYTES = 15 * 1024 * 1024  # 15MB -- generous for a few minutes of speech
ALLOWED_AUDIO_TYPES = {"audio/webm", "audio/wav", "audio/x-wav", "audio/mpeg", "audio/ogg", "audio/mp4"}


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    # Any endpoint's own domain errors are already caught and turned into
    # structured refusals (see src/pipeline.py) -- this is the last-resort
    # net so a genuinely unexpected failure still returns clean JSON
    # instead of leaking a stack trace to the client.
    log.exception("Unhandled error on %s", request.url.path)
    return JSONResponse(
        status_code=500,
        content={"error": "Something went wrong on our end. Please try again."},
    )


# ---------- pages ----------

@app.get("/")
def landing_page():
    return FileResponse("frontend/index.html")


@app.get("/app")
def app_page():
    return FileResponse("frontend/app.html")


@app.get("/history")
def history_page():
    return FileResponse("frontend/history.html")


@app.get("/about")
def about_page():
    return FileResponse("frontend/about.html")


@app.get("/health")
def health():
    return {"status": "ok"}


# ---------- api ----------

@app.post("/ask/audio", dependencies=[Depends(enforce_rate_limit)])
async def ask_audio(file: UploadFile = File(...), language_code: str = Form("hi-IN")):
    if file.content_type and file.content_type not in ALLOWED_AUDIO_TYPES:
        raise HTTPException(status_code=415, detail=f"Unsupported audio type: {file.content_type}")

    audio_bytes = await file.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Empty audio file.")
    if len(audio_bytes) > MAX_AUDIO_BYTES:
        raise HTTPException(status_code=413, detail="Audio file too large (max 15MB).")

    result = run_pipeline(audio_bytes=audio_bytes, language_code=language_code)
    return result.to_dict()


@app.post("/ask/text", dependencies=[Depends(enforce_rate_limit)])
async def ask_text(query: str = Form(...)):
    if not query or not query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    result = run_pipeline(text_query=query)
    return result.to_dict()
