"""The self-corrective RAG workflow, implemented as a LangGraph StateGraph.

Node flow
---------
    analyze_query -> retrieve -> grade_documents --(relevant docs found)--> generate -> hallucination_check --(grounded)--> END
                                       |                                                              |
                                       | (no relevant docs, retries left)                              | (not grounded, retries left)
                                       v                                                              v
                                transform_query -> retrieve (loop)                     bump_retry_and_regenerate -> generate (loop)
                                       |
                                       | (no relevant docs, retries exhausted)
                                       v
                                cannot_answer -> END                    (not grounded, retries exhausted) -> flag_ungrounded -> END

Two independent retry loops exist, both bounded by the same `max_retries` /
`retries` counter in state so we can't loop forever:
  1. Retrieval loop: grade_documents -> transform_query -> retrieve (bad retrieval)
  2. Generation loop: hallucination_check -> generate (ungrounded answer)
"""
import json
from typing import List

from langchain_core.documents import Document
from langgraph.graph import StateGraph, END

from app import config
from app.llm import get_llm
from app.state import GraphState, DocChunk
from app.vectorstore import similarity_search

# ---------------------------------------------------------------------------
# Node 1: Query Analysis
# ---------------------------------------------------------------------------

QUERY_ANALYSIS_PROMPT = """You are a query analysis assistant for a technical documentation \
search system covering FastAPI, Pydantic, LangGraph, ChromaDB, and RAG concepts.

Given the conversation so far and the user's latest question, do two things:
1. Rewrite the question into a self-contained, search-optimized query. Resolve any \
pronouns or follow-up references using the chat history (e.g. "what about async?" after a \
question about FastAPI dependencies becomes "How does FastAPI handle async path operation \
functions?"). Expand obvious abbreviations. Do not answer the question -- only rewrite it.
2. Classify the query type as exactly one of: conceptual, how-to, troubleshooting, api-reference.

Chat history (may be empty):
{history}

User's latest question: {question}

Respond ONLY with valid JSON, no markdown fences, no preamble:
{{"rewritten_query": "...", "query_type": "..."}}
"""


def analyze_query(state: GraphState) -> dict:
    history = state.get("chat_history", [])
    history_str = "\n".join(f"{m['role']}: {m['content']}" for m in history[-6:]) or "(none)"

    llm = get_llm()
    raw = llm.invoke(
        QUERY_ANALYSIS_PROMPT.format(history=history_str, question=state["question"])
    ).content

    try:
        parsed = json.loads(_strip_fences(raw))
        rewritten = parsed.get("rewritten_query", "").strip() or state["question"]
        qtype = parsed.get("query_type", "conceptual").strip()
    except Exception:
        rewritten = state["question"]
        qtype = "conceptual"

    return {
        "search_query": rewritten,
        "query_type": qtype,
        "retries": state.get("retries", 0),
        "max_retries": state.get("max_retries", config.MAX_RETRIES),
    }


def _strip_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
    return text.strip()


# ---------------------------------------------------------------------------
# Node 2: Retrieval
# ---------------------------------------------------------------------------

def retrieve(state: GraphState) -> dict:
    query = state.get("search_query") or state["question"]
    docs: List[Document] = similarity_search(query, k=config.TOP_K)

    chunks: List[DocChunk] = [
        {
            "content": d.page_content,
            "source": d.metadata.get("source", "unknown"),
            "chunk_index": d.metadata.get("chunk_index", -1),
            "grade": None,
        }
        for d in docs
    ]
    return {"documents": chunks}


# ---------------------------------------------------------------------------
# Node 3: Document Grading (self-corrective step)
# ---------------------------------------------------------------------------

GRADING_PROMPT = """You are grading whether a retrieved document chunk is relevant to a user's \
question. This is a binary relevance check for a retrieval pipeline, not a quality judgment.

Question: {question}

Retrieved chunk (source: {source}):
---
{content}
---

Does this chunk contain information that would help answer the question, even partially? \
Respond with exactly one word: "relevant" or "irrelevant"."""


def grade_documents(state: GraphState) -> dict:
    llm = get_llm()
    question = state["question"]
    graded: List[DocChunk] = []

    for chunk in state.get("documents", []):
        verdict = llm.invoke(
            GRADING_PROMPT.format(question=question, source=chunk["source"], content=chunk["content"])
        ).content.strip().lower()
        grade = "relevant" if "relevant" in verdict and "irrelevant" not in verdict else "irrelevant"
        graded.append({**chunk, "grade": grade})

    relevant = [c for c in graded if c["grade"] == "relevant"]
    return {"documents": graded, "relevant_documents": relevant}


def decide_after_grading(state: GraphState) -> str:
    if state.get("relevant_documents"):
        return "generate"
    if state.get("retries", 0) < state.get("max_retries", config.MAX_RETRIES):
        return "transform_query"
    return "cannot_answer"


# ---------------------------------------------------------------------------
# Query transformation (retrieval retry loop)
# ---------------------------------------------------------------------------

TRANSFORM_PROMPT = """The following search query returned no relevant results from a technical \
documentation corpus (covering FastAPI, Pydantic, LangGraph, ChromaDB, and RAG concepts).

Original question: {question}
Query that failed: {failed_query}

Rewrite it as a different, broader or differently-phrased search query that might retrieve \
relevant chunks instead. Respond with ONLY the new query text, nothing else."""


def transform_query(state: GraphState) -> dict:
    llm = get_llm()
    new_query = llm.invoke(
        TRANSFORM_PROMPT.format(
            question=state["question"],
            failed_query=state.get("search_query", state["question"]),
        )
    ).content.strip()

    return {
        "search_query": new_query,
        "retries": state.get("retries", 0) + 1,
        "route": "transform_query",
    }


# ---------------------------------------------------------------------------
# Node 4: Generation
# ---------------------------------------------------------------------------

GENERATION_PROMPT = """You are a technical documentation assistant. Answer the user's question \
using ONLY the provided context chunks. If the context is insufficient to fully answer, say so \
explicitly rather than filling gaps from outside knowledge.

Cite sources inline using the format [source_filename] right after the claim it supports. \
Every substantive claim should be traceable to at least one citation.

Question: {question}

Context chunks:
{context}

Write a clear, accurate answer with inline citations."""


def _format_context(chunks: List[DocChunk]) -> str:
    parts = []
    for c in chunks:
        parts.append(f"[{c['source']}] (chunk {c['chunk_index']}):\n{c['content']}")
    return "\n\n---\n\n".join(parts)


def generate(state: GraphState) -> dict:
    llm = get_llm()
    relevant = state.get("relevant_documents", [])
    context = _format_context(relevant)

    answer = llm.invoke(
        GENERATION_PROMPT.format(question=state["question"], context=context)
    ).content.strip()

    citations = sorted({c["source"] for c in relevant})
    return {"generation": answer, "citations": citations, "answered": True, "route": "generate"}


# ---------------------------------------------------------------------------
# Bonus Node: Hallucination / groundedness check (Self-RAG style)
# ---------------------------------------------------------------------------

HALLUCINATION_PROMPT = """You are fact-checking whether an answer is fully supported by the \
given source context. Ignore style; judge only factual grounding.

Source context:
{context}

Generated answer:
{answer}

Is every factual claim in the answer directly supported by the source context? Respond with \
exactly one word: "grounded" or "not_grounded"."""


def hallucination_check(state: GraphState) -> dict:
    if not config.ENABLE_HALLUCINATION_CHECK:
        return {"hallucination_grade": "grounded"}

    llm = get_llm()
    context = _format_context(state.get("relevant_documents", []))
    verdict = llm.invoke(
        HALLUCINATION_PROMPT.format(context=context, answer=state.get("generation", ""))
    ).content.strip().lower()

    grade = "grounded" if "not" not in verdict else "not_grounded"
    return {"hallucination_grade": grade}


def decide_after_hallucination_check(state: GraphState) -> str:
    if state.get("hallucination_grade") == "grounded":
        return "end"
    if state.get("retries", 0) < state.get("max_retries", config.MAX_RETRIES):
        return "regenerate"
    # Exhausted retries: still return the (possibly imperfect) answer, but flag it.
    return "give_up"


def bump_retry_and_regenerate(state: GraphState) -> dict:
    """Increments the retry counter, then loops back to `generate` to try again
    with the same relevant_documents (a fresh sample from the LLM can fix a
    one-off ungrounded generation without re-doing retrieval)."""
    return {"retries": state.get("retries", 0) + 1, "route": "regenerate"}


def flag_ungrounded(state: GraphState) -> dict:
    note = ("\n\n_Note: this answer could not be fully verified against the retrieved "
            "documentation and may contain unsupported claims._")
    return {"generation": state.get("generation", "") + note, "route": "flag_ungrounded"}


# ---------------------------------------------------------------------------
# Fallback: no relevant documents found after all retries
# ---------------------------------------------------------------------------

def cannot_answer(state: GraphState) -> dict:
    return {
        "generation": (
            "I couldn't find information relevant to this question in the indexed "
            "documentation. Try rephrasing, or this may be outside the current corpus "
            "(FastAPI, Pydantic, LangGraph, ChromaDB, RAG concepts)."
        ),
        "citations": [],
        "answered": False,
        "route": "cannot_answer",
    }


# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------

def build_graph():
    workflow = StateGraph(GraphState)

    workflow.add_node("analyze_query", analyze_query)
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("grade_documents", grade_documents)
    workflow.add_node("transform_query", transform_query)
    workflow.add_node("generate", generate)
    workflow.add_node("hallucination_check", hallucination_check)
    workflow.add_node("bump_retry_and_regenerate", bump_retry_and_regenerate)
    workflow.add_node("flag_ungrounded", flag_ungrounded)
    workflow.add_node("cannot_answer", cannot_answer)

    workflow.set_entry_point("analyze_query")
    workflow.add_edge("analyze_query", "retrieve")
    workflow.add_edge("retrieve", "grade_documents")

    workflow.add_conditional_edges(
        "grade_documents",
        decide_after_grading,
        {"generate": "generate", "transform_query": "transform_query", "cannot_answer": "cannot_answer"},
    )
    workflow.add_edge("transform_query", "retrieve")

    workflow.add_edge("generate", "hallucination_check")
    workflow.add_conditional_edges(
        "hallucination_check",
        decide_after_hallucination_check,
        {"end": END, "regenerate": "bump_retry_and_regenerate", "give_up": "flag_ungrounded"},
    )
    # Loop back to generate for another attempt (bounded by retries).
    workflow.add_edge("bump_retry_and_regenerate", "generate")
    workflow.add_edge("flag_ungrounded", END)

    workflow.add_edge("cannot_answer", END)

    return workflow.compile()


_compiled_graph = None


def get_graph():
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph()
    return _compiled_graph


def run_query(question: str, chat_history: List[dict] = None, max_retries: int = None) -> GraphState:
    graph = get_graph()
    initial_state: GraphState = {
        "question": question,
        "chat_history": chat_history or [],
        "retries": 0,
        "max_retries": max_retries if max_retries is not None else config.MAX_RETRIES,
    }
    return graph.invoke(initial_state)
