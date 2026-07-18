"""State schema shared across every LangGraph node.

Keeping this in one place forces us to be explicit about exactly what data
flows between nodes -- this is the core design artifact of the assignment.
"""
from typing import List, Optional, TypedDict


class DocChunk(TypedDict):
    """A single retrieved chunk plus everything needed to grade/cite it."""
    content: str
    source: str
    chunk_index: int
    grade: Optional[str]  # "relevant" | "irrelevant" | None (not yet graded)


class GraphState(TypedDict, total=False):
    # --- input ---
    question: str                     # original user question, never mutated
    chat_history: List[dict]          # [{"role": "user"/"assistant", "content": str}, ...]

    # --- query analysis ---
    search_query: str                 # the (possibly rewritten) query actually used for retrieval
    query_type: str                   # conceptual | how-to | troubleshooting | api-reference

    # --- retrieval / grading ---
    documents: List[DocChunk]         # raw retrieved chunks (this run)
    relevant_documents: List[DocChunk]  # subset that passed grading, used for generation
    retries: int                      # number of rewrite+re-retrieve cycles used so far
    max_retries: int

    # --- generation ---
    generation: str                   # the final answer text
    citations: List[str]              # human-readable source labels actually cited

    # --- self-corrective / bonus ---
    hallucination_grade: str          # "grounded" | "not_grounded" | None
    used_web_fallback: bool
    route: str                        # last routing decision, useful for debugging/telemetry

    # --- terminal status, surfaced to the API layer ---
    answered: bool                    # False => "I don't know" style fallback was used
