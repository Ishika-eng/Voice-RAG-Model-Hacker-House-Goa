# Voice RAG — MSMARCO-XI

**HH Goa 2026 Shortlisting Submission — Task 2: Voice-Enabled RAG Model**

Repo: [github.com/Ishika-eng/Voice-RAG-Model-Hacker-House-Goa](https://github.com/Ishika-eng/Voice-RAG-Model-Hacker-House-Goa)

**🔴 Live demo: [martin-deaf-collectables-universities.trycloudflare.com](https://martin-deaf-collectables-universities.trycloudflare.com)**
> This is a temporary tunnel to a locally-hosted instance, not a permanent deployment — it has already rotated twice since submission (free Cloudflare quick tunnels have no uptime guarantee over multi-day periods). If it's unreachable, this README is kept up to date with the current working link — check back here for the latest URL.

Voice-enabled Retrieval-Augmented Generation over [`ai4bharat/MSMARCO-XI`](https://huggingface.co/datasets/ai4bharat/MSMARCO-XI).

```
mic audio → Sarvam STT → guardrails → multi-strategy retrieval (Qdrant) → Groq (Llama) generation → guardrails → answer
```

## Architecture

| Stage | Component | File |
|---|---|---|
| Speech-to-text | Sarvam `saaras:v3` | [`src/stt.py`](src/stt.py) |
| Input guardrail | unsafe-content / empty-query filter | [`src/guardrails.py`](src/guardrails.py) |
| Chunking (offline, indexing time) | 3 strategies, see below | [`src/chunking/`](src/chunking) |
| Retrieval | multilingual-e5 embeddings + Qdrant, RRF fusion across strategies | [`src/retrieval.py`](src/retrieval.py) |
| On-topic guardrail | refuse if best similarity < threshold | [`src/guardrails.py`](src/guardrails.py) |
| Generation | Groq (Llama 3.3 70B), context-constrained system prompt | [`src/generation.py`](src/generation.py) |
| Groundedness guardrail | lexical overlap between answer and retrieved context | [`src/guardrails.py`](src/guardrails.py) |
| Harness | typed stages, per-stage timing, retry + fallback-to-refusal on any stage failure | [`src/pipeline.py`](src/pipeline.py) |

## Chunking strategies

Chunking is **not naive fixed-size only** — three strategies are built, embedded, and indexed into separate Qdrant collections, then fused at query time:

1. **Fixed-size with overlap** (`src/chunking/fixed_size.py`) — 60-word sliding window, 15-word overlap. Baseline.
2. **Semantic** (`src/chunking/semantic.py`) — sentence-split (aware of Hindi `।` as a sentence boundary, not just `.`), embed each sentence, greedily merge adjacent sentences while cosine similarity to the running chunk centroid stays above a threshold. Produces variable-length, topically coherent chunks instead of arbitrary word windows.
3. **Passage-native** (`src/chunking/passage_native.py`) — MS MARCO passages are already curated retrieval units; this strategy trusts that segmentation instead of re-splitting it.

All three are then run through **metadata-aware enrichment** (`src/chunking/metadata_aware.py`): each chunk gets `query_id`, `is_selected` (MS MARCO's own relevance label), token count, has-numbers flag, and extracted keywords attached as Qdrant payload — used both for retrieval boosting (`is_selected` passages get a small RRF score boost) and for the off-topic guardrail.

At query time, `src/retrieval.py` fans the query out across all three collections and merges results with **Reciprocal Rank Fusion** (rank-based, not raw-score-based, since a 60-word window and a single-sentence semantic chunk aren't score-comparable).

## Guardrails

- **Input**: blocks unsafe/harmful query patterns and empty/oversized input before any retrieval happens.
- **Off-topic**: if the best retrieval similarity is below `OFFTOPIC_SIM_THRESHOLD`, the system refuses rather than answering from irrelevant context.
- **Groundedness**: after generation, checks what fraction of the answer's content words actually appear in the retrieved context. Below `GROUNDEDNESS_OVERLAP_THRESHOLD`, the answer is discarded and a refusal is returned instead. This is a fast, dependency-free heuristic chosen specifically to stay on the latency-critical path (no second LLM call to fact-check).
- Every refusal carries a machine-readable `refusal_reason` (`off_topic_low_similarity`, `unsafe_content`, `ungrounded_overlap_0.12`, `stt_failed`, etc.) surfaced in the API response and UI.

## Harness

`src/pipeline.py` is not a single prompt-in/text-out call. It's a typed stage pipeline (`PipelineResult` dataclass) where each stage times itself independently, STT and generation calls retry transiently-failing API calls (`tenacity`, exponential backoff), and any stage failure degrades to a structured refusal response rather than propagating an exception to the caller.

## Latency

The **200ms target applies to the retrieval leg** — embed query → vector search across 3 collections → RRF merge → guardrail checks. Chunking happens once at indexing time, not per-query, and is therefore off the query-time latency path entirely. LLM generation (Groq API round-trip) is reported separately and honestly, since no network LLM call completes in 200ms — folding it into the same number would misrepresent the numbers.

Run the benchmark after ingesting data:

```bash
python src/scripts/benchmark_latency.py --n 50
```

This samples real queries from the dataset, runs the full pipeline (skipping STT to isolate retrieval+generation from network/mic variance), and prints:

```
=== Retrieval leg only (embed + vector search + guardrail checks) ===
{'p50': ..., 'p70': ..., 'p100': ...}   # ms

=== Full pipeline (incl. LLM generation) ===
{'p50': ..., 'p70': ..., 'p100': ...}   # ms
```

Raw per-query timings are written to `data/latency_results.csv`.

**Note on the numbers below:** these are not a single formal `--n 50` benchmark run — that run didn't finish cleanly before submission (the dev machine was simultaneously running the live demo tunnel, which skewed timing). They're the actual `stage_timings_ms`/`total_latency_ms` values observed across real interactive queries during development, reported honestly as a range rather than as a rigorous percentile study. Re-run `python src/scripts/benchmark_latency.py --n 50` on an otherwise-idle machine for a clean formal measurement.

| Leg | Observed range (real queries, warm) |
|---|---|
| Retrieval only | ~99ms – ~580ms (typically under 250ms; higher end under system load) |
| Full pipeline (incl. Groq generation) | ~750ms – ~1.5s (generation, not retrieval, dominates this number) |
| Full pipeline (incl. generation) | — ms | — ms | — ms |

## Setup

```bash
git clone https://github.com/Ishika-eng/Voice-RAG-Model-Hacker-House-Goa.git
cd Voice-RAG-Model-Hacker-House-Goa
python3.13 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill in SARVAM_API_KEY and GROQ_API_KEY
```

> Note: pin to Python 3.11–3.13. `pydantic-core`'s compiled wheels don't yet support 3.14.

> If dataset downloads hang with a `.incomplete` file stuck at 0 bytes, HF's Xet CDN client is stalling in your network — set `export HF_HUB_DISABLE_XET=1` before running ingest/benchmark scripts to fall back to plain HTTPS downloads.

## Ingest the dataset

```bash
python src/scripts/run_ingest.py --limit 5000
```

Builds all 3 chunking strategies over `ai4bharat/MSMARCO-XI`, embeds with `intfloat/multilingual-e5-small`, and upserts into local Qdrant collections at `./data/qdrant_local`. Each language lives in its own parquet file (`train/hintrain.parquet` for Hindi, etc. — set `HF_DATASET_LANG` in `.env`); there's no HF "config name" for this repo, `src/ingest.py` resolves the file path directly.

The Hindi split is ~3.7GB — on a flaky connection HF's downloader resumes from the partial `.incomplete` blob on retry, so re-running the same command after a failed attempt picks up where it left off rather than restarting.

**Offline smoke test** (no dataset download required): `python scripts/smoke_test.py` runs 2 hand-built rows shaped exactly like the real parquet schema through chunking → embedding → Qdrant upsert → retrieval → all 3 guardrail checks, and prints each stage's output. Useful for verifying the pipeline wiring before committing to the full download.

## Run

```bash
uvicorn app:app --reload
```

Open `http://localhost:8000` for the landing page. The app is multi-page:

| Route | Page |
|---|---|
| `/` | Landing/marketing page — pitch, pipeline overview, real stats, tech stack |
| `/app` | The voice/text Q&A interface (the orb) |
| `/history` | Local question history (see below) |
| `/about` | Written architecture walkthrough for a general/judge audience |

## Product hardening

This isn't just the demo endpoint anymore — a few things were added specifically so it holds up outside a hackathon judging session:

- **Rate limiting** ([`src/rate_limit.py`](src/rate_limit.py)): an in-memory sliding-window limiter (20 requests / 5 min per IP) guards `/ask/audio` and `/ask/text`, returning `429` with a clear message and `Retry-After` header. Single-process only by design — a multi-worker deploy would swap this for a Redis-backed limiter.
- **Local history, no accounts**: every answer is saved to the browser's `localStorage` (see `frontend/app.html`'s `saveToHistory`) and rendered on `/history` with search/filter and a clear-history action. Nothing is sent to a server or shared — deliberately scoped this way rather than standing up real auth + a database for a hackathon submission.
- **Input validation**: audio uploads are checked for content-type, non-empty body, and a 15MB size cap (`app.py`); text queries reject empty input. All with proper 4xx status codes, not silent failures.
- **Global exception handling**: any genuinely unexpected server error returns clean JSON instead of a leaked stack trace.
- **Escaped output**: all user- and model-generated text is HTML-escaped before being inserted into the DOM (`escapeHtml` in `frontend/app.html` / `frontend/history.html`) — the API response includes the user's own query text, generation output, and retrieved passages, none of which should be trusted as safe markup.
- **Mobile-responsive**: shared nav collapses to a stacked layout under 640px, orb and typography scale down under 480px.

## Benchmark

```bash
python src/scripts/benchmark_latency.py --n 50
```

## Project layout

```
src/
  chunking/         fixed_size.py, semantic.py, passage_native.py, metadata_aware.py
  embeddings.py      shared multilingual-e5 embedding model
  vector_store.py    Qdrant wrapper, one collection per strategy
  ingest.py          dataset -> chunks -> embeddings -> Qdrant
  stt.py             Sarvam speech-to-text
  retrieval.py       multi-strategy fan-out + RRF fusion
  guardrails.py      input safety / off-topic / groundedness checks
  generation.py      Groq (Llama) answer generation
  pipeline.py        harness orchestrating all stages
  latency.py         stage timer + percentile helpers
  rate_limit.py      in-memory sliding-window rate limiter
scripts/
  run_ingest.py
  benchmark_latency.py
frontend/
  index.html         landing/marketing page
  app.html            voice/text Q&A interface (the orb)
  history.html        local question history
  about.html           architecture walkthrough
  shared.css           design tokens + nav shared across all pages
app.py                FastAPI entrypoint (page routes + /ask/audio, /ask/text APIs)
```

## Problems we ran into (and how we solved them)

Documented honestly, since real engineering has friction — these are the actual issues hit during development, not a cleaned-up narrative.

### Environment & setup
- **Python 3.14 was too new**: `pydantic-core`'s compiled wheels didn't support it yet, breaking `pip install` outright. Fixed by pinning the venv to Python 3.13.
- **Port 8000 collision**: another local project was already bound to port 8000. Moved the app to port 8001.

### Dataset ingestion
- **HF's Xet CDN client stalled indefinitely** on the ~3.7GB parquet download (`.incomplete` file stuck at 0 bytes for minutes). Fixed with `HF_HUB_DISABLE_XET=1`, which falls back to plain HTTPS and also resumes cleanly from partial downloads on retry.
- **Assumed dataset schema didn't match reality**: initial code assumed flat `Translated_passages`/`is_selected` fields; the actual schema nests them inside a `passages` struct. Found via direct schema inspection and fixed the row-parsing logic.
- **pyarrow couldn't read the file with `datasets.load_dataset()`**: `ArrowNotImplementedError: Nested data conversions not implemented for chunked array outputs`. Root cause: the parquet file has a single ~780k-row row group, and pyarrow can't concatenate the nested `passages` struct/list column across the internal chunks that produces in one shot. Fixed by reading via `pyarrow.parquet.iter_batches()` directly instead, which never asks pyarrow to concatenate across chunks. This exact bug reappeared later when a teammate's rewrite of the ingestion script went back to `datasets.load_dataset()` — fixed the same way a second time.
- **Qdrant's local file-mode store only allows one process at a time**: running the ingest script while another process already held the store threw `RuntimeError: Storage folder ... is already accessed by another instance`. Resolved by making sure only one ingest process runs against the local store at once.

### Models & APIs
- **Groq deprecated the model we'd built against** (`llama-3.3-70b-versatile` returned a 404 partway through development). Queried Groq's live model list and switched to `openai/gpt-oss-120b`.
- **Sarvam dataset language default was wrong**: `HF_DATASET_LANG` defaulted to `hi`, but the actual parquet filename prefix is `hin` (`train/hintrain.parquet`) — dataset loading failed outright until this was corrected.

### Frontend
- **Dark-on-dark rendering bug**: the page had no explicit `background` on `body`, so it rendered invisible text in some environments. Fixed by setting the background explicitly rather than relying on inheritance.
- **Background canvas invisible despite drawing correctly**: a `position: fixed` element with negative `z-index` was being composited behind the page in some render paths. Fixed by restructuring the stacking so background layers use `z-index: 0` and foreground content explicitly uses `z-index: 1`, instead of relying on negative z-index.
- **A glow effect rendered as a hard box instead of a soft radial fade**: its radius could exceed the canvas's own bounds and get truncated at the canvas's rectangular edge. Fixed by clamping the radius to fit inside the canvas.

### Integrating three people's work
- Merging independently-written modules surfaced real interface mismatches: a script importing a function (`iter_rows`) that a teammate's rewrite had removed, `sys.path` calculations that were off by one directory level after scripts got moved into a new folder, and (see above) the same pyarrow bug being reintroduced by a parallel rewrite. All caught by actually running the merged code end-to-end rather than assuming a clean `git merge` meant a working app, and fixed one at a time.

### Deployment
- **Oracle Cloud's Always Free tier was unreliable to provision** (capacity/signup friction) despite being the best-fit option for the full ~3.7GB dataset (its Ampere shape uniquely offers up to 24GB free RAM, which the in-memory vector index needs at this scale). Fell back to a Cloudflare quick tunnel exposing the already-working local instance instead, trading "always-on" for "zero new infrastructure, full dataset intact, working immediately."
- **The quick tunnel is inherently fragile**: it depends on the host laptop staying on, and the URL is randomly assigned per session with no way to reclaim a specific hostname once it disconnects — ours died once from a network interruption and had to be restarted under a new URL. Mitigated by keeping the README updated as the canonical source of the current working link, and adding an in-app fallback banner pointing back to it.
- **iCloud Drive silently reorganized the project folder mid-session** (a known disruptive behavior of Desktop & Documents sync), moving the project to a new path and breaking anything that referenced the old one — including the tunnel restart command. The running app server itself was unaffected (Unix processes keep working via open file handles even if their directory is moved), only new commands referencing the old path failed. Found by checking process state directly rather than assuming the filesystem path was still valid.
