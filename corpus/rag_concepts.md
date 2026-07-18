# Retrieval-Augmented Generation: Core Concepts

## What is RAG?

Retrieval-Augmented Generation (RAG) is a pattern for grounding LLM outputs in an external
knowledge source. Instead of relying solely on what a model learned during training, a RAG
system retrieves relevant passages from a document store at query time and includes them in
the prompt, so the model can generate an answer that cites and is constrained by that
retrieved context. This reduces hallucination and allows the system's knowledge to be
updated by changing the document store, without retraining the model.

## The Basic Pipeline

A minimal RAG pipeline has three stages: (1) index a corpus of documents as embedding
vectors ahead of time, (2) at query time, embed the user's question and retrieve the most
similar chunks, and (3) construct a prompt containing the question plus the retrieved
chunks, and ask an LLM to answer using only that context.

## Why Naive RAG Falls Short

A naive pipeline retrieves a fixed number of chunks and always uses them, even when they are
irrelevant to the question or when the phrasing of the question doesn't closely match the
wording in the documents (a mismatch that hurts embedding similarity search in particular).
This can cause the model to either answer confidently from bad context or to hallucinate
when the context doesn't actually contain the answer.

## Self-Corrective RAG (Self-RAG / Corrective RAG)

Self-corrective RAG architectures add evaluation steps into the pipeline itself. A "document
grading" step uses an LLM (or a smaller classifier) to judge whether each retrieved chunk is
actually relevant to the question before it is used for generation. If none of the retrieved
chunks are judged relevant, the system can rewrite the query and retry retrieval, fall back
to an alternative source such as a web search, or return an explicit "I don't know" rather
than forcing an answer from irrelevant context.

A further refinement adds a "hallucination grading" step after generation: an LLM checks
whether the generated answer is actually supported by the retrieved documents (as opposed to
being plausible-sounding but unsupported text). If the answer is not grounded, the system can
regenerate, retrieve again, or flag the answer as low-confidence to the user.

## Query Rewriting

Query rewriting/expansion improves retrieval by transforming the user's raw question into a
form that is more likely to match how the answer is phrased in the source documents. Common
techniques include: expanding abbreviations and adding synonyms, decomposing a compound
question into sub-questions, and rephrasing a vague follow-up question (e.g. "what about
async?") into a self-contained question using prior conversation context.

## Evaluating RAG Systems

RAG systems are typically evaluated along two axes: retrieval quality (did we fetch the
passages that actually contain the answer, measured with metrics like recall@k or precision)
and generation quality (is the final answer correct, complete, and grounded in the retrieved
context, often measured with LLM-as-judge scoring or human review). A system can fail at
either stage independently: perfect retrieval with poor generation still produces a bad
answer, and vice versa.
