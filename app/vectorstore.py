"""Wraps ChromaDB (via LangChain) so ingestion.py and graph.py share one store."""
from functools import lru_cache
from typing import List

from langchain_core.documents import Document
from langchain_chroma import Chroma

from app import config


@lru_cache(maxsize=1)
def get_embeddings():
    """Groq has no embeddings API, so this is independent of LLM_PROVIDER.
    Defaults to a free local embedding model via fastembed (ONNX-based --
    no PyTorch, small download, no API key or network needed after the
    first run).
    """
    if config.EMBEDDING_PROVIDER == "local":
        from langchain_community.embeddings import FastEmbedEmbeddings
        return FastEmbedEmbeddings(model_name=config.EMBEDDING_MODEL)

    from langchain_openai import OpenAIEmbeddings
    return OpenAIEmbeddings(
        model=config.EMBEDDING_MODEL,
        api_key=config.OPENAI_API_KEY or None,
    )


@lru_cache(maxsize=1)
def get_vectorstore() -> Chroma:
    return Chroma(
        collection_name=config.COLLECTION_NAME,
        embedding_function=get_embeddings(),
        persist_directory=config.CHROMA_DIR,
    )


def add_documents(documents: List[Document]) -> int:
    """Adds documents to the store. Ids are derived from source+chunk_index so
    re-ingesting the same file upserts rather than duplicates."""
    store = get_vectorstore()
    ids = [
        f"{d.metadata.get('source', 'unknown')}::{d.metadata.get('chunk_index', i)}"
        for i, d in enumerate(documents)
    ]
    store.add_documents(documents, ids=ids)
    return len(documents)


def similarity_search(query: str, k: int) -> List[Document]:
    store = get_vectorstore()
    return store.similarity_search(query, k=k)


def list_sources() -> dict:
    """Returns {source_filename: chunk_count} for everything currently indexed."""
    store = get_vectorstore()
    raw = store.get(include=["metadatas"])
    counts: dict = {}
    for meta in raw.get("metadatas", []) or []:
        src = meta.get("source", "unknown")
        counts[src] = counts.get(src, 0) + 1
    return counts
