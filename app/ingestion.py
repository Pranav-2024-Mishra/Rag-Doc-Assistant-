"""Document ingestion: load -> chunk -> embed -> store.

Chunking strategy
------------------
We use LangChain's `RecursiveCharacterTextSplitter` configured with markdown-aware
separators (headings, then paragraphs, then sentences, then words). It tries the
first separator; if a resulting chunk is still bigger than `chunk_size`, it falls
back to the next, more granular separator. This keeps headings/paragraphs/code
blocks intact whenever they fit, and only cuts mid-paragraph as a last resort --
which matters for technical docs where a code example split across two chunks
becomes useless as retrieval context.

Defaults: 800 characters per chunk, 120 character overlap (15%). This is a
reasonable middle ground for prose-heavy technical documentation: small enough
that a chunk stays topically focused (good embedding precision), large enough
to keep a concept + a short example together (good recall/context), with enough
overlap that a sentence isn't orphaned right at a chunk boundary.
"""
import os
from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app import config
from app.vectorstore import add_documents

MARKDOWN_SEPARATORS = [
    "\n## ", "\n### ", "\n#### ",   # headings
    "\n```\n",                      # code fences (keep code blocks whole when possible)
    "\n\n", "\n", ". ", " ", "",     # paragraphs -> lines -> sentences -> words -> chars
]


def _splitter() -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=MARKDOWN_SEPARATORS,
        length_function=len,
    )


def load_file(path: str) -> Document:
    text = Path(path).read_text(encoding="utf-8")
    return Document(page_content=text, metadata={"source": Path(path).name})


def load_directory(directory: str) -> List[Document]:
    docs = []
    for name in sorted(os.listdir(directory)):
        if name.lower().endswith((".md", ".txt")):
            docs.append(load_file(os.path.join(directory, name)))
    return docs


def chunk_documents(documents: List[Document]) -> List[Document]:
    splitter = _splitter()
    chunks: List[Document] = []
    for doc in documents:
        pieces = splitter.split_text(doc.page_content)
        for i, piece in enumerate(pieces):
            chunks.append(
                Document(
                    page_content=piece,
                    metadata={**doc.metadata, "chunk_index": i, "total_chunks": len(pieces)},
                )
            )
    return chunks


def ingest_documents(documents: List[Document]) -> dict:
    chunks = chunk_documents(documents)
    n = add_documents(chunks)
    by_source: dict = {}
    for c in chunks:
        by_source[c.metadata["source"]] = by_source.get(c.metadata["source"], 0) + 1
    return {"files_ingested": len(documents), "chunks_indexed": n, "chunks_by_source": by_source}


def ingest_directory(directory: str = None) -> dict:
    directory = directory or config.CORPUS_DIR
    documents = load_directory(directory)
    if not documents:
        return {"files_ingested": 0, "chunks_indexed": 0, "chunks_by_source": {}}
    return ingest_documents(documents)


def ingest_texts(texts: List[str], sources: List[str]) -> dict:
    documents = [Document(page_content=t, metadata={"source": s}) for t, s in zip(texts, sources)]
    return ingest_documents(documents)


if __name__ == "__main__":
    result = ingest_directory()
    print(f"Ingested {result['files_ingested']} files -> {result['chunks_indexed']} chunks")
    for src, count in result["chunks_by_source"].items():
        print(f"  - {src}: {count} chunks")
