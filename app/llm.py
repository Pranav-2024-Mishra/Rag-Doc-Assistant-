"""Thin factory around the chat model so the rest of the app never imports a
provider SDK directly. Swapping providers only requires editing this file.

Supports two providers via LLM_PROVIDER in .env:
  - "groq"   (default): free, fast inference over open models (Llama, etc.)
  - "openai": OpenAI chat models

Note: Groq does not offer an embeddings API, so embeddings are handled
separately in vectorstore.py regardless of which LLM provider is selected.
"""
from functools import lru_cache

from app import config


@lru_cache(maxsize=1)
def get_llm(temperature: float | None = None):
    temp = config.LLM_TEMPERATURE if temperature is None else temperature

    if config.LLM_PROVIDER == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(
            model=config.LLM_MODEL,
            temperature=temp,
            api_key=config.GROQ_API_KEY or None,
        )

    if config.LLM_PROVIDER == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=config.LLM_MODEL,
            temperature=temp,
            api_key=config.OPENAI_API_KEY or None,
        )

    raise ValueError(f"Unknown LLM_PROVIDER: {config.LLM_PROVIDER!r} (expected 'groq' or 'openai')")
