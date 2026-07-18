"""Centralized configuration, loaded from environment variables / .env."""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent

# --- LLM ---
# "groq" (free, fast, default) or "openai"
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0"))

if LLM_PROVIDER == "groq":
    LLM_MODEL = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")
else:
    LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")

# --- Embeddings ---
# "local" (free, runs on your machine via sentence-transformers, default)
# or "openai" (requires OPENAI_API_KEY, costs a small amount)
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "local")

if EMBEDDING_PROVIDER == "local":
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
else:
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

# --- Vector store ---
CHROMA_DIR = os.getenv("CHROMA_DIR", str(BASE_DIR / "data" / "chroma_db"))
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "tech_docs")

# --- Corpus ---
CORPUS_DIR = os.getenv("CORPUS_DIR", str(BASE_DIR / "corpus"))

# --- Chunking ---
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "120"))

# --- Retrieval / graph behavior ---
TOP_K = int(os.getenv("TOP_K", "4"))
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "2"))  # query rewrite + re-retrieve attempts
RELEVANCE_THRESHOLD = float(os.getenv("RELEVANCE_THRESHOLD", "0.0"))  # reserved for future score-based filtering

# --- Feedback storage ---
FEEDBACK_LOG = os.getenv("FEEDBACK_LOG", str(BASE_DIR / "data" / "feedback" / "feedback.jsonl"))

# --- Bonus features ---
ENABLE_HALLUCINATION_CHECK = os.getenv("ENABLE_HALLUCINATION_CHECK", "true").lower() == "true"
ENABLE_WEB_SEARCH_FALLBACK = os.getenv("ENABLE_WEB_SEARCH_FALLBACK", "false").lower() == "true"
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")
