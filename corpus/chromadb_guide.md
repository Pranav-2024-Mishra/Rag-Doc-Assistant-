# ChromaDB: An Embedding Database for RAG

## What is ChromaDB?

Chroma is an open-source embedding database designed to store vector embeddings alongside
their source text and metadata, and to perform fast similarity search over them. It is
commonly used as the retrieval backend in Retrieval-Augmented Generation (RAG) systems
because it is simple to run locally (either in-memory or persisted to disk) without standing
up a separate server.

## Collections

Data in Chroma is organized into "collections," which are roughly analogous to a table. Each
collection holds a set of items, where each item has an id, an embedding vector, the
original document text, and an optional metadata dictionary.

```python
import chromadb

client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(name="docs")

collection.add(
    ids=["doc1-chunk0"],
    documents=["FastAPI is a modern Python web framework..."],
    metadatas=[{"source": "fastapi_basics.md", "chunk_index": 0}],
)
```

## Similarity Search

Querying a collection returns the `n_results` items whose embeddings are closest to the
query embedding, typically using cosine similarity or L2 distance depending on configuration:

```python
results = collection.query(
    query_texts=["How do I validate a request body in FastAPI?"],
    n_results=4,
)
```

The results include the matched documents, their metadata, their ids, and a distance score
for each match, which can be used downstream to filter out low-confidence matches before
they are passed to an LLM.

## Metadata Filtering

Metadata attached at insert time can be used to filter queries, for example restricting
search to a specific source file or document type:

```python
collection.query(
    query_texts=["dependency injection"],
    n_results=4,
    where={"source": "fastapi_basics.md"},
)
```

This is useful in multi-document corpora where a user's question implies a specific document
(e.g. "in the Pydantic docs, how do I...") or where you want to exclude deprecated or
low-quality sources from retrieval.

## Using Chroma with LangChain

LangChain provides a `Chroma` vector store wrapper that handles embedding generation and
persistence for you, so that application code can work with LangChain `Document` objects
directly instead of raw ids/embeddings:

```python
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

vectorstore = Chroma(
    collection_name="docs",
    embedding_function=OpenAIEmbeddings(model="text-embedding-3-small"),
    persist_directory="./chroma_db",
)

vectorstore.add_documents(documents)
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
```

## Chunking Considerations

Because similarity search operates on whole-chunk embeddings, chunk size directly affects
retrieval quality: chunks that are too large dilute the embedding with unrelated content
(hurting precision), while chunks that are too small lose surrounding context (hurting
recall and making generated answers feel fragmented). A common starting point for technical
documentation is 500-1000 characters per chunk with 10-20% overlap between consecutive
chunks, splitting preferentially on semantic boundaries (headings, paragraphs, code blocks)
rather than at a fixed character offset, so that a chunk doesn't get cut in the middle of a
code example or a sentence.

## Persistence

A `PersistentClient` (or a LangChain `Chroma` instance configured with `persist_directory`)
writes its data to disk automatically, so a collection built once can be reloaded in later
process runs without re-embedding the entire corpus. Re-running ingestion against an
existing collection with the same ids will typically upsert (overwrite) those ids rather
than duplicate them, which makes idempotent ingestion scripts straightforward to write.
