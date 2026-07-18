# LangGraph: Building Stateful, Multi-Step LLM Applications

## What is LangGraph?

LangGraph is a library for building applications as graphs of nodes, where each node is a
step in a computation (often an LLM call, a tool call, or a bit of logic) and edges define
how control passes from one step to the next. It is designed for workflows that are not
strictly linear: loops, conditional branches, retries, and human-in-the-loop checkpoints are
all first-class concepts, which makes it a good fit for "agentic" or self-correcting
pipelines that a simple prompt chain can't express cleanly.

## The State Object

Every LangGraph graph is built around a shared state object, typically defined as a
`TypedDict` or a Pydantic model. Each node receives the current state, does some work, and
returns a (partial) update to that state. LangGraph merges the returned update into the
overall state before passing it to the next node.

```python
from typing import TypedDict, List

class GraphState(TypedDict):
    question: str
    documents: List[str]
    generation: str
    retries: int
```

By default, keys are overwritten by whatever a node returns for them. If you want a key to
accumulate values across nodes instead (for example, a running list of messages), you
annotate it with a reducer function, most commonly via `Annotated[list, add]` or the
`add_messages` helper for chat-style histories.

## Building a Graph

A graph is constructed with `StateGraph(GraphState)`, nodes are registered with `add_node`,
and edges connect them:

```python
from langgraph.graph import StateGraph, END

workflow = StateGraph(GraphState)
workflow.add_node("retrieve", retrieve_fn)
workflow.add_node("generate", generate_fn)

workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "generate")
workflow.add_edge("generate", END)

app = workflow.compile()
```

`END` is a special sentinel node that terminates the graph. `app.invoke(initial_state)` runs
the graph to completion and returns the final state; `app.stream(...)` yields intermediate
state updates as each node finishes, which is useful for showing progress in a UI.

## Conditional Edges

Conditional edges let the graph branch based on the current state. You supply a routing
function that inspects the state and returns the name of the next node:

```python
def decide_next_step(state: GraphState) -> str:
    if not state["documents"]:
        return "fallback"
    return "generate"

workflow.add_conditional_edges(
    "grade_documents",
    decide_next_step,
    {"fallback": "transform_query", "generate": "generate"},
)
```

This is the mechanism used to implement self-correction: a grading node evaluates the
quality of retrieved documents or a generated answer, and the conditional edge decides
whether to proceed, retry with a modified input, or bail out to a fallback path.

## Cycles and Retry Limits

Because LangGraph graphs can contain cycles (a node's conditional edge can route back to an
earlier node), it's important to track loop counters in the state and check them in your
routing function to avoid infinite loops. A common pattern is to increment a `retries`
counter each time a retry edge is taken, and have the conditional function force a fallback
route once `retries` exceeds a configured maximum, regardless of the grading outcome.

## Checkpointing and Memory

LangGraph supports checkpointers (e.g. an in-memory checkpointer or a SQLite/Postgres-backed
one) that persist the state of a graph run keyed by a thread/session id. This allows a graph
to be paused and resumed, and is the mechanism typically used to support multi-turn
conversations: each new user turn is invoked against the same thread id, and the
checkpointer restores the prior state (including message history) before the new turn runs.

## Why Use a Graph Instead of a Chain?

A linear chain (retrieve -> generate) has no way to react to a bad retrieval or a bad
generation except by failing outright. Modeling the same logic as a graph makes the control
flow explicit and inspectable: each decision point is a distinct node or edge, intermediate
state can be logged or streamed, and new branches (a hallucination check, a web-search
fallback, a human approval step) can be added without restructuring the whole pipeline.
