"""Local-first helpers for the Hugging Face Agents Course (CS6961)."""

from .backends import (
    chat,
    describe_backend,
    get_langchain_chat,
    get_langchain_ollama_chat,
    get_llamaindex_embedding,
    get_llamaindex_llm,
    get_openai_client,
    get_smolagents_litellm_model,
    get_smolagents_model,
)
from .config import BACKENDS, get_backend

__version__ = "1.0.0"

__all__ = [
    "BACKENDS",
    "chat",
    "describe_backend",
    "get_backend",
    "get_langchain_chat",
    "get_langchain_ollama_chat",
    "get_llamaindex_embedding",
    "get_llamaindex_llm",
    "get_openai_client",
    "get_smolagents_litellm_model",
    "get_smolagents_model",
]
