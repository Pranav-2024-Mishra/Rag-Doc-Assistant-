"""Tests for graph topology and node logic, using fakes instead of real
OpenAI calls so the suite runs offline / without an API key.
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("OPENAI_API_KEY", "test-key-not-real")

from app.state import GraphState
from app import graph as graph_mod


class FakeLLMResponse:
    def __init__(self, content):
        self.content = content


class FakeLLM:
    """Returns queued responses in order, cycling the last one if exhausted."""
    def __init__(self, responses):
        self.responses = responses
        self.calls = 0

    def invoke(self, _prompt):
        idx = min(self.calls, len(self.responses) - 1)
        self.calls += 1
        return FakeLLMResponse(self.responses[idx])


def make_chunk(source="doc.md", idx=0, content="some content", grade=None):
    return {"content": content, "source": source, "chunk_index": idx, "grade": grade}


def test_decide_after_grading_routes_to_generate_when_relevant():
    state: GraphState = {"relevant_documents": [make_chunk()], "retries": 0, "max_retries": 2}
    assert graph_mod.decide_after_grading(state) == "generate"


def test_decide_after_grading_routes_to_transform_when_retries_left():
    state: GraphState = {"relevant_documents": [], "retries": 0, "max_retries": 2}
    assert graph_mod.decide_after_grading(state) == "transform_query"


def test_decide_after_grading_routes_to_cannot_answer_when_retries_exhausted():
    state: GraphState = {"relevant_documents": [], "retries": 2, "max_retries": 2}
    assert graph_mod.decide_after_grading(state) == "cannot_answer"


def test_decide_after_hallucination_check_end_when_grounded():
    state: GraphState = {"hallucination_grade": "grounded", "retries": 0, "max_retries": 2}
    assert graph_mod.decide_after_hallucination_check(state) == "end"


def test_decide_after_hallucination_check_regenerate_when_not_grounded_and_retries_left():
    state: GraphState = {"hallucination_grade": "not_grounded", "retries": 0, "max_retries": 2}
    assert graph_mod.decide_after_hallucination_check(state) == "regenerate"


def test_decide_after_hallucination_check_gives_up_when_retries_exhausted():
    state: GraphState = {"hallucination_grade": "not_grounded", "retries": 2, "max_retries": 2}
    assert graph_mod.decide_after_hallucination_check(state) == "give_up"


def test_grade_documents_filters_irrelevant(monkeypatch):
    fake = FakeLLM(["relevant", "irrelevant"])
    monkeypatch.setattr(graph_mod, "get_llm", lambda *a, **k: fake)

    state: GraphState = {
        "question": "How does FastAPI validate request bodies?",
        "documents": [make_chunk(source="a.md"), make_chunk(source="b.md")],
    }
    result = graph_mod.grade_documents(state)
    assert len(result["relevant_documents"]) == 1
    assert result["documents"][0]["grade"] == "relevant"
    assert result["documents"][1]["grade"] == "irrelevant"


def test_cannot_answer_sets_answered_false():
    result = graph_mod.cannot_answer({})
    assert result["answered"] is False
    assert result["citations"] == []


def test_generate_produces_citations(monkeypatch):
    fake = FakeLLM(["FastAPI validates bodies with Pydantic models. [fastapi_basics.md]"])
    monkeypatch.setattr(graph_mod, "get_llm", lambda *a, **k: fake)

    state: GraphState = {
        "question": "How does FastAPI validate request bodies?",
        "relevant_documents": [make_chunk(source="fastapi_basics.md", grade="relevant")],
    }
    result = graph_mod.generate(state)
    assert result["answered"] is True
    assert "fastapi_basics.md" in result["citations"]


def test_graph_compiles():
    """End-to-end structural check: the graph must compile without error."""
    compiled = graph_mod.build_graph()
    assert compiled is not None
