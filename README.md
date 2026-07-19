
# RAG-Based Technical Documentation Assistant

A self-corrective Retrieval-Augmented Generation system built with **LangGraph**, **ChromaDB**,
and **FastAPI**. It answers natural-language questions over a small technical documentation
corpus (FastAPI, Pydantic, LangGraph, ChromaDB, and RAG/Self-RAG concepts), grading its own
retrievals and generations before returning an answer.

## Demo video 
[▶ Watch Demo](https://drive.google.com/file/d/16JP4Yp0hO-LCPYaRFMeCsiBk4HTMg6pO/view?usp=sharing)


## Table of Contents

- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Setup](#setup)
- [Running the App](#running-the-app)
- [Streamlit UI](#streamlit-ui)
- [API Reference & Examples](#api-reference--examples)
- [Chunking & Embedding Strategy](#chunking--embedding-strategy)
- [Design Decisions & Tradeoffs](#design-decisions--tradeoffs)
- [Assumptions](#assumptions)
- [What I'd Improve With More Time](#what-id-improve-with-more-time)
- [Testing](#testing)

## Architecture

### The LangGraph Workflow

```
                    ┌────────────────┐
                    │  analyze_query │   rewrite question + classify type
                    └────────┬───────┘
                             ▼
                    ┌────────────────┐
              ┌────▶│    retrieve    │   vector similarity search (top-k)
              │     └────────┬───────┘
              │              ▼
              │     ┌────────────────┐
              │     │ grade_documents│   LLM grades each chunk relevant/irrelevant
              │     └────────┬───────┘
              │              │
              │    ┌─────────┴─────────┐
              │    │   any relevant?   │
              │    └─────────┬─────────┘
              │      no │         │ yes
              │         ▼         ▼
     ┌────────┴─────┐      ┌────────────┐
     │transform_query│      │  generate  │  LLM answers using only relevant chunks
     └───────────────┘      └──────┬─────┘
     (retries < max)                ▼
              │            ┌──────────────────┐
              │            │hallucination_check│  is the answer grounded in context?
              │            └────────┬──────────┘
     (retries                       │
      exhausted)           ┌────────┴────────┐
              │             grounded │  not grounded
              ▼                      │         │
      ┌───────────────┐              ▼    (retries < max)
      │ cannot_answer │             END        │
      └───────┬───────┘                        ▼
              ▼                    ┌─────────────────────────┐
             END                   │bump_retry_and_regenerate│──▶ generate (loop)
                                    └─────────────────────────┘
                              (retries exhausted)
                                        │
                                        ▼
                               ┌────────────────┐
                               │ flag_ungrounded │──▶ END (answer + disclaimer)
                               └────────────────┘
```

This is two independent, bounded retry loops sharing one `retries` counter in state:

1. **Retrieval loop** (`grade_documents → transform_query → retrieve`): if none of the
   retrieved chunks are graded relevant, the query is rewritten and retrieval is retried.
2. **Generation loop** (`hallucination_check → bump_retry_and_regenerate → generate`): if the
   generated answer isn't grounded in the retrieved context, generation is retried.

Both loops are capped by `max_retries` (default `2`, configurable per-request). Once
exhausted, the graph always terminates deterministically — either an explicit "I don't know"
(`cannot_answer`) or a flagged, possibly-imperfect answer (`flag_ungrounded`) — it never loops
forever.

### State Schema (`app/state.py`)

The state is the core design artifact of this assignment — it's what makes the graph's control
flow explicit and debuggable instead of implicit in a chain of function calls.

```python
class GraphState(TypedDict, total=False):
    question: str                       # original user question, never mutated
    chat_history: List[dict]            # prior turns, for follow-up query rewriting

    search_query: str                   # rewritten query actually used for retrieval
    query_type: str                     # conceptual | how-to | troubleshooting | api-reference

    documents: List[DocChunk]           # all retrieved chunks this run, with grades attached
    relevant_documents: List[DocChunk]  # subset that passed grading -> fed to generation
    retries: int                        # shared counter for both retry loops
    max_retries: int

    generation: str                     # final answer text
    citations: List[str]                # source filenames actually cited

    hallucination_grade: str            # "grounded" | "not_grounded"
    route: str                          # last routing decision (debugging/telemetry)
    answered: bool                      # False => fallback path was used
```

Each node returns only the keys it changes; LangGraph merges the partial update into the
running state. `retries` and `max_retries` are threaded through every node so the two
conditional-edge functions (`decide_after_grading`, `decide_after_hallucination_check`) can
make a stateless routing decision purely from what's already in state.

### Nodes

| Node | Responsibility |
|---|---|
| `analyze_query` | Rewrites the question into a self-contained search query (resolving follow-up references using `chat_history`) and classifies query type. LLM call, structured JSON output. |
| `retrieve` | Runs `similarity_search(search_query, k=TOP_K)` against Chroma. |
| `grade_documents` | One LLM call per retrieved chunk: `relevant` / `irrelevant`. This is the self-corrective core. |
| `transform_query` | Rewrites the query differently (broader/rephrased) when nothing relevant was found, and increments `retries`. |
| `generate` | Answers using **only** the `relevant_documents`, with inline `[source_filename]` citations. |
| `hallucination_check` | *(bonus, Self-RAG-inspired)* LLM checks whether the generated answer is actually supported by the retrieved context. |
| `bump_retry_and_regenerate` | Increments `retries` and loops back to `generate` for another attempt. |
| `flag_ungrounded` | Appends a disclaimer when hallucination retries are exhausted rather than silently returning a possibly-wrong answer. |
| `cannot_answer` | Explicit "I don't know" fallback when retrieval retries are exhausted. |

## Project Structure

```
rag-doc-assistant/
├── app/
│   ├── api.py            # FastAPI routes
│   ├── config.py         # env-driven configuration
│   ├── graph.py           # the LangGraph workflow (the core of the assignment)
│   ├── ingestion.py       # load -> chunk -> embed -> store
│   ├── llm.py             # chat model factory
│   ├── main.py             # uvicorn entrypoint
│   ├── state.py             # GraphState schema
│   └── vectorstore.py     # Chroma wrapper
├── corpus/                 # 5 original technical docs (FastAPI, Pydantic, LangGraph, Chroma, RAG)
├── scripts/ingest.py       # standalone CLI ingestion
├── tests/test_graph.py     # unit tests for routing logic + graph compilation (mocked LLM)
├── data/                   # chroma_db/ (persisted vectors) + feedback/ (jsonl log)
├── streamlit_app.py         # minimal Streamlit chat UI (talks to the FastAPI backend)
├── requirements.txt
├── .env.example
└── README.md
```

## Setup

**Requirements:** Python 3.11+, and a free [Groq API key](https://console.groq.com/keys)
(sign up, no credit card needed). By default this project runs on **Groq for the LLM** and a
**free local model (via `fastembed`) for embeddings** — so it costs $0 to run. OpenAI is also
supported as a drop-in alternative if you'd rather use it (see `.env.example`).

```bash
git clone <your-repo-url>
cd rag-doc-assistant

python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt

cp .env.example .env
# edit .env and set GROQ_API_KEY=gsk_...  (get one free at https://console.groq.com/keys)
```

**Switching providers:** everything is controlled by `.env`:

| Setting | Free (default) | Paid alternative |
|---|---|---|
| `LLM_PROVIDER` | `groq` (needs `GROQ_API_KEY`) | `openai` (needs `OPENAI_API_KEY`) |
| `EMBEDDING_PROVIDER` | `local` (needs nothing, runs on-device via `fastembed`) | `openai` (needs `OPENAI_API_KEY`) |

The first time you run ingestion with `EMBEDDING_PROVIDER=local`, `fastembed` downloads a
small (~130MB) ONNX embedding model (`BAAI/bge-small-en-v1.5` by default) and caches it
locally — after that it runs fully offline with no API calls.

## Running the App

**1. Ingest the bundled corpus** (one-time, or whenever `corpus/` changes):

```bash
python scripts/ingest.py
```

This loads the 5 markdown files in `corpus/`, chunks them, embeds them, and persists them to
`./data/chroma_db`. Re-running is safe and idempotent — chunks are upserted by a stable
`{source}::{chunk_index}` id.

**2. Start the API server:**

```bash
uvicorn app.main:app --reload
```

The server starts at `http://localhost:8000`. Interactive docs are at
`http://localhost:8000/docs` (Swagger UI) and `http://localhost:8000/redoc`.

## Streamlit UI

A minimal chat UI (`streamlit_app.py`) sits on top of the FastAPI backend — it's a thin
client; all the logic stays in the API/graph layer.

With the backend already running (`uvicorn app.main:app --reload`), in a **second terminal**:

```bash
source .venv/bin/activate   # same venv, streamlit is in requirements.txt
streamlit run streamlit_app.py
```

This opens `http://localhost:8501` in your browser. Features:

- **Chat interface** — ask questions, see the answer with inline `[source.md]` citations
- **Source panel** — each answer has an expandable list of the chunks that were actually graded relevant and used
- **Run metadata** — query type, retries used, and hallucination-check grade shown under each answer
- **Multi-turn memory** — follow-up questions reuse the same `session_id` automatically
- **👍 / 👎 feedback** — posts to `/feedback`, tied to that answer's `query_id`
- **Sidebar** — shows what's currently indexed (`/documents`), lets you upload a new `.md`/`.txt` file to ingest on the fly, and lets you point the UI at a different API URL if the backend isn't on `localhost:8000`

If the backend isn't running yet, the UI shows a clear "can't reach the API" message instead of crashing.

## API Reference & Examples

### `POST /query` — ask a question

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "How does FastAPI validate a request body?"}'
```

```json
{
  "query_id": "3f1c...",
  "session_id": "9a02...",
  "answer": "FastAPI validates request bodies using Pydantic models [fastapi_basics.md]. When a client sends JSON, FastAPI parses it, validates it against the declared model, and converts it into a Python object automatically. If the data doesn't match the model (wrong type, missing field), FastAPI returns a 422 response [fastapi_basics.md]...",
  "answered": true,
  "sources": [
    {"source": "fastapi_basics.md", "chunk_index": 2, "grade": "relevant"},
    {"source": "pydantic_models.md", "chunk_index": 0, "grade": "relevant"}
  ],
  "query_type": "conceptual",
  "rewritten_query": "How does FastAPI validate incoming request body data?",
  "retries_used": 0,
  "hallucination_grade": "grounded",
  "latency_ms": 2140
}
```

Pass the returned `session_id` on the next call to enable multi-turn follow-ups (e.g. "what
about optional fields?" after asking about Pydantic models) — `chat_history` is used by
`analyze_query` to rewrite the follow-up into a self-contained query.

A question outside the corpus (e.g. "What's the weather today?") exercises the fallback path:

```json
{
  "answer": "I couldn't find information relevant to this question in the indexed documentation. Try rephrasing, or this may be outside the current corpus (FastAPI, Pydantic, LangGraph, ChromaDB, RAG concepts).",
  "answered": false,
  "sources": [],
  "retries_used": 2
}
```

### `POST /ingest` — ingest new documents

```bash
# Re-ingest the bundled corpus directory
curl -X POST http://localhost:8000/ingest

# Upload new files
curl -X POST http://localhost:8000/ingest \
  -F "files=@my_new_doc.md"

# Ingest from URLs (requires outbound network access to the target domain)
curl -X POST http://localhost:8000/ingest \
  -F 'urls=["https://raw.githubusercontent.com/example/repo/main/README.md"]'
```

```json
{"files_ingested": 1, "chunks_indexed": 6, "chunks_by_source": {"my_new_doc.md": 6}}
```

### `GET /documents` — list what's indexed

```bash
curl http://localhost:8000/documents
```

```json
{
  "total_chunks": 36,
  "sources": {
    "chromadb_guide.md": 7,
    "fastapi_basics.md": 8,
    "langgraph_overview.md": 8,
    "pydantic_models.md": 7,
    "rag_concepts.md": 6
  }
}
```

### `POST /feedback` — rate an answer

```bash
curl -X POST http://localhost:8000/feedback \
  -H "Content-Type: application/json" \
  -d '{"query_id": "3f1c...", "rating": "up", "comment": "Exactly what I needed"}'
```

```json
{"status": "recorded", "query_id": "3f1c..."}
```

Feedback is appended as JSON lines to `data/feedback/feedback.jsonl`, alongside the original
question/answer for context.

## Chunking & Embedding Strategy

**Chunking:** `RecursiveCharacterTextSplitter` with markdown-aware separators, tried in order
of preference: headings (`##`, `###`, `####`) → code fences → paragraphs → lines → sentences →
words. `chunk_size=800`, `chunk_overlap=120` (15%).

Rationale: technical docs mix prose explanations with code examples. Splitting on generic
character counts risks cutting a code block or a sentence in half, which produces chunks that
are useless as retrieval context (and can look actively confusing when shown to the grading
LLM or cited to the user). Preferring semantic boundaries keeps a concept and its accompanying
example together whenever it fits inside the chunk budget, and only falls back to a harder
split when a section genuinely doesn't fit. 800 characters is roughly 150-200 words — enough
for a concept + one short example, small enough to keep the embedding topically focused. 15%
overlap prevents a sentence right at a chunk boundary from being orphaned without its
preceding context. This was validated empirically against the bundled corpus (`scripts/ingest.py`
output — see `python scripts/ingest.py`) — chunk counts and boundaries look sensible (chunks
land ~330–790 characters, headings and code fences stay intact).

**Embeddings:** Local, via `fastembed` (`BAAI/bge-small-en-v1.5` by default) — a small
ONNX-runtime embedding model that runs on-device with no API key and no per-call cost, chosen
specifically to keep the whole stack free when paired with Groq. It's swappable to OpenAI's
`text-embedding-3-small` via `EMBEDDING_PROVIDER=openai` in `.env` if you'd rather trade the
local download for OpenAI's (paid) embeddings. `app/vectorstore.py` is the only place that
would need to change to add another provider.

**LLM:** Groq (`llama-3.3-70b-versatile` by default) for all reasoning steps — query analysis,
document grading, generation, and the hallucination check. Groq's free tier is generous and
its inference is fast, which matters here since a single `/query` call makes 5-8 sequential
LLM calls. Swappable to OpenAI via `LLM_PROVIDER=openai` in `.env`. Note that **Groq doesn't
offer an embeddings API**, which is why embeddings are handled by a separate, independent
provider setting.

**Vector store:** ChromaDB, persisted locally to `./data/chroma_db`. Chosen over FAISS because
it natively stores metadata (source filename, chunk index) alongside vectors and supports
metadata filtering out of the box, and it needs no separate server process for a project this
size.

## Design Decisions & Tradeoffs

- **Grading is per-chunk, not per-batch.** Grading each retrieved chunk individually (one LLM
  call each) is slower and more expensive than a single "are any of these relevant" call, but
  it lets `relevant_documents` be an exact filtered subset rather than an all-or-nothing
  decision, and it's what enables citing individual chunks accurately. Tradeoff: `TOP_K=4`
  means up to 4 grading calls per query on top of the analysis/retrieval/generation calls —
  fine for a demo, but worth batching into a single structured-output call in a
  higher-throughput setting.

- **One shared `retries` counter for two different loops.** The retrieval-retry loop and the
  generation-retry loop both increment the same `retries` field rather than having separate
  counters. This is a simplification — it means an expensive rewrite-and-retry attempt "spends"
  budget that a slow-to-ground generation could have used, and vice versa. I chose this because
  the assignment explicitly calls out tracking retries as a core evaluation point, and a single,
  legible counter is easier to reason about and log than two independent ones. With more time
  I'd split them (see below).

- **Hallucination check regenerates from the same `relevant_documents`, not fresh retrieval.**
  If an answer isn't grounded, the most likely cause is generation (the LLM added something not
  in the context), not retrieval, so re-running `generate` against the same context is the
  cheaper and usually correct fix. If it's actually a retrieval problem, the generation retry
  loop won't fix it — the fallback disclaimer catches that failure mode instead of an infinite
  loop.

- **In-memory session store for conversation memory**, not LangGraph's checkpointer. LangGraph
  checkpointing (bonus feature territory) is the "correct" way to persist thread state across
  turns, but for a 2-day take-home with a single-process FastAPI server, a plain
  `dict[session_id, history]` is simpler to reason about and test, and is explicitly called out
  in the code as a non-persistent, process-local simplification.

- **Groq for the LLM, a local model for embeddings — two independent providers, not one.**
  Groq has no embeddings API, so unlike a single-provider OpenAI setup, this project always
  needs two separate configuration axes (`LLM_PROVIDER`, `EMBEDDING_PROVIDER`). I chose to make
  both independently swappable (each defaults to the free option, each can be pointed at OpenAI
  instead) rather than hardcoding the pairing, since "Groq LLM + OpenAI embeddings" or
  "OpenAI LLM + local embeddings" are both reasonable setups a reviewer might want. All
  provider-specific code is isolated to `app/llm.py` and `app/vectorstore.py`.

## Assumptions

- The corpus is small (5 documents) and I wrote it myself rather than scraping real
  documentation sites, both to avoid any copyright/reproduction concerns and because this
  sandbox's outbound network access is restricted to package registries (PyPI, npm, GitHub),
  not general documentation sites — a `POST /ingest` with URLs will work in a normal
  unrestricted environment (it uses plain `requests.get`), but couldn't be exercised against a
  real doc site from inside this dev environment.
- "Documents" in `/documents` are reported as indexed source files with chunk counts rather
  than raw file listings, since that's the more useful signal for verifying ingestion worked.
- Feedback (`/feedback`) references a `query_id` returned by a prior `/query` call; there's no
  persistent database, so answer history for feedback lookups only survives for the life of the
  server process (see tradeoffs above).
- I assumed "self-corrective" should cover both retrieval quality (bad documents) and
  generation quality (hallucination) rather than just the required retrieval-grading loop,
  since the assignment explicitly calls out Self-RAG-style hallucination checking as a bonus.

## What I'd Improve With More Time

- **Split the retry budget** into independent `retrieval_retries` / `generation_retries`
  counters instead of one shared counter.
- **Batch document grading** into a single structured-output LLM call (grade all `k` chunks at
  once) instead of `k` separate calls, to cut latency and cost.
- **Real LangGraph checkpointing** (`MemorySaver` or a SQLite checkpointer) for conversation
  memory instead of the in-memory dict, so sessions survive a server restart and so
  `chat_history` doesn't need to be threaded manually through the FastAPI layer.
- **Web search fallback** (Tavily/Serper), wired in as an additional branch off
  `decide_after_grading` when retries are exhausted, instead of going straight to
  `cannot_answer` — the config flag (`ENABLE_WEB_SEARCH_FALLBACK`) and node are stubbed for
  this but not implemented.
- **Score-based grading** in addition to LLM relevance grading — combining the embedding
  distance/similarity score with the LLM's relevance verdict (rather than the LLM verdict
  alone) would let obviously-irrelevant chunks be filtered without a full LLM call.
- **A small eval set** (10-20 question/expected-source pairs) with a script that runs each
  question through the graph and reports retrieval precision/recall and whether the expected
  source was cited, to catch regressions as prompts change.
- **Rate limiting / auth** on the FastAPI endpoints, and persisting the feedback log and answer
  log to a real database instead of a JSONL file and an in-memory dict.

## Testing

```bash
pytest tests/ -v
```

Tests cover the conditional-edge routing logic (`decide_after_grading`,
`decide_after_hallucination_check`), the grading and generation node logic, and confirm the
full graph compiles — all using a fake LLM so the suite runs without a real API key. All 10
tests pass as of this submission.

To manually verify ingestion/chunking without any API key: `python scripts/ingest.py --dir corpus`
will fail at the embedding step without a key, but `app.ingestion.load_directory` +
`chunk_documents` (used directly, see `tests/`) can be exercised standalone.
=======
# Rag-Doc-Assistant-
 065958dae2c336d159a622ee7a4c7186a7cb9a95
