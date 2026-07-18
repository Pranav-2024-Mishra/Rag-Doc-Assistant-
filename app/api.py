"""FastAPI layer exposing the RAG workflow."""
import json
import time
import uuid
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from pydantic import BaseModel, Field

from app import config
from app.graph import run_query
from app.ingestion import ingest_directory, ingest_documents
from app.vectorstore import list_sources
from langchain_core.documents import Document

app = FastAPI(
    title="RAG Technical Documentation Assistant",
    description="A self-corrective LangGraph RAG pipeline over a small technical docs corpus.",
    version="1.0.0",
)

# --- in-memory session store for conversation memory (bonus feature) -------
# NOTE: process-local and non-persistent by design; fine for a take-home demo.
_SESSIONS: dict[str, List[dict]] = {}

# --- in-memory record of past answers, so /feedback can reference them -----
_ANSWER_LOG: dict[str, dict] = {}

Path(config.FEEDBACK_LOG).parent.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, description="The user's natural language question")
    session_id: Optional[str] = Field(None, description="Reuse to maintain multi-turn conversation memory")
    max_retries: Optional[int] = Field(None, ge=0, le=5, description="Override retry limit for this call")


class SourceOut(BaseModel):
    source: str
    chunk_index: int
    grade: Optional[str] = None


class QueryResponse(BaseModel):
    query_id: str
    session_id: str
    answer: str
    answered: bool
    sources: List[SourceOut]
    query_type: Optional[str] = None
    rewritten_query: Optional[str] = None
    retries_used: int
    hallucination_grade: Optional[str] = None
    latency_ms: int


class IngestUrlsRequest(BaseModel):
    urls: List[str] = Field(default_factory=list)


class IngestResponse(BaseModel):
    files_ingested: int
    chunks_indexed: int
    chunks_by_source: dict


class DocumentsResponse(BaseModel):
    total_chunks: int
    sources: dict


class FeedbackRequest(BaseModel):
    query_id: str
    rating: str = Field(..., pattern="^(up|down)$")
    comment: Optional[str] = None


class FeedbackResponse(BaseModel):
    status: str
    query_id: str


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    if not req.question.strip():
        raise HTTPException(status_code=422, detail="question must not be empty")

    session_id = req.session_id or str(uuid.uuid4())
    history = _SESSIONS.get(session_id, [])

    start = time.time()
    try:
        result = run_query(req.question, chat_history=history, max_retries=req.max_retries)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Generation pipeline failed: {e}")
    latency_ms = int((time.time() - start) * 1000)

    answer = result.get("generation", "")
    answered = result.get("answered", False)

    # Update conversation memory
    history = history + [
        {"role": "user", "content": req.question},
        {"role": "assistant", "content": answer},
    ]
    _SESSIONS[session_id] = history[-20:]  # cap history length

    query_id = str(uuid.uuid4())
    sources = [
        SourceOut(source=d["source"], chunk_index=d["chunk_index"], grade=d.get("grade"))
        for d in result.get("relevant_documents", [])
    ]

    _ANSWER_LOG[query_id] = {
        "question": req.question,
        "answer": answer,
        "session_id": session_id,
    }

    return QueryResponse(
        query_id=query_id,
        session_id=session_id,
        answer=answer,
        answered=answered,
        sources=sources,
        query_type=result.get("query_type"),
        rewritten_query=result.get("search_query"),
        retries_used=result.get("retries", 0),
        hallucination_grade=result.get("hallucination_grade"),
        latency_ms=latency_ms,
    )


@app.post("/ingest", response_model=IngestResponse)
async def ingest(
    files: Optional[List[UploadFile]] = File(default=None),
    urls: Optional[str] = Form(default=None),  # JSON-encoded list of URLs, or comma-separated
):
    """Accepts either uploaded files (multipart) or a list of URLs (form field,
    JSON array or comma-separated string). Fetching arbitrary URLs requires
    outbound network access to whatever domain is provided -- see README for
    the sandboxed-network caveat."""
    if not files and not urls:
        # Fall back to re-ingesting the bundled corpus directory.
        result = ingest_directory()
        return IngestResponse(**result)

    documents: List[Document] = []

    if files:
        for f in files:
            raw = await f.read()
            try:
                text = raw.decode("utf-8")
            except UnicodeDecodeError:
                raise HTTPException(status_code=422, detail=f"{f.filename} is not valid UTF-8 text")
            documents.append(Document(page_content=text, metadata={"source": f.filename}))

    if urls:
        import requests
        try:
            url_list = json.loads(urls)
            if isinstance(url_list, str):
                url_list = [url_list]
        except json.JSONDecodeError:
            url_list = [u.strip() for u in urls.split(",") if u.strip()]

        for url in url_list:
            try:
                resp = requests.get(url, timeout=15)
                resp.raise_for_status()
            except Exception as e:
                raise HTTPException(status_code=422, detail=f"Failed to fetch {url}: {e}")
            documents.append(Document(page_content=resp.text, metadata={"source": url}))

    if not documents:
        raise HTTPException(status_code=422, detail="No valid files or URLs provided")

    result = ingest_documents(documents)
    return IngestResponse(**result)


@app.get("/documents", response_model=DocumentsResponse)
def documents():
    sources = list_sources()
    return DocumentsResponse(total_chunks=sum(sources.values()), sources=sources)


@app.post("/feedback", response_model=FeedbackResponse)
def feedback(req: FeedbackRequest):
    record = {
        "query_id": req.query_id,
        "rating": req.rating,
        "comment": req.comment,
        "timestamp": time.time(),
        "context": _ANSWER_LOG.get(req.query_id),
    }
    with open(config.FEEDBACK_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
    return FeedbackResponse(status="recorded", query_id=req.query_id)
