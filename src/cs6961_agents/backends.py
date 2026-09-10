"""
One backend, three frameworks.

The Agents Course uses a different LLM wrapper class in every unit:

    Unit 1 & 2.1  smolagents   -> InferenceClientModel
    Unit 2.2      LangGraph    -> ChatHuggingFace(HuggingFaceEndpoint(...))
    Unit 2.3      LlamaIndex   -> HuggingFaceInferenceAPI

All three of those point at Hugging Face Inference Providers, which is metered.
This module gives you a drop-in replacement for each one, all reading the same
`.env`. In the course notebooks, replace the model line with an import:

    # from smolagents import InferenceClientModel
    # model = InferenceClientModel(model_id="Qwen/Qwen2.5-Coder-32B-Instruct")
    from cs6961_agents.backends import get_smolagents_model
    model = get_smolagents_model()

That is the only edit. Everything downstream is untouched.
"""

from __future__ import annotations

import os
from typing import Any

from .config import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_TEMPERATURE,
    REQUEST_TIMEOUT,
    BackendSpec,
    get_api_key,
    get_backend,
)

__all__ = [
    "chat",
    "describe_backend",
    "get_openai_client",
    "get_smolagents_model",
    "get_langchain_chat",
    "get_llamaindex_llm",
]


def describe_backend(name: str | None = None) -> str:
    """Human-readable one-liner, handy at the top of every example script."""
    spec = get_backend(name)
    cost = "free" if spec.is_free else "METERED — costs credits"
    where = "local" if spec.is_local else "cloud"
    return f"[{spec.name}] {spec.default_model} @ {spec.base_url} ({where}, {cost})"


# --------------------------------------------------------------------------
# Layer 0 — the raw OpenAI SDK. Everything below is built on this.
# --------------------------------------------------------------------------
def get_openai_client(name: str | None = None):
    """Return a plain `openai.OpenAI` client aimed at the active backend."""
    from openai import OpenAI

    spec = get_backend(name)
    return OpenAI(
        base_url=spec.base_url,
        api_key=get_api_key(spec),
        timeout=REQUEST_TIMEOUT,
    )


def chat(messages: list[dict], name: str | None = None, **kwargs: Any):
    """One chat completion against the active backend, with its quirks applied.

    `get_openai_client()` hands you a bare OpenAI client, which is the right
    thing for learning -- but it does not know about per-backend request
    tweaks such as Ollama's `reasoning_effort`. This wrapper fills in the model
    name and merges `spec.extra_body`, so short replies do not come back empty
    from a hybrid Qwen3 tag.
    """
    spec = get_backend(name)
    client = get_openai_client(name)

    extra_body = dict(spec.extra_body)
    extra_body.update(kwargs.pop("extra_body", None) or {})

    return client.chat.completions.create(
        model=kwargs.pop("model", spec.default_model),
        messages=messages,
        extra_body=extra_body or None,
        **kwargs,
    )


# --------------------------------------------------------------------------
# Layer 1 — smolagents  (Unit 1, Unit 2.1, and most of Unit 4)
# --------------------------------------------------------------------------
def _import_smolagents_openai_model():
    """smolagents renamed this class; support both spellings."""
    import smolagents

    for attr in ("OpenAIServerModel", "OpenAIModel"):
        cls = getattr(smolagents, attr, None)
        if cls is not None:
            return cls
    raise ImportError(
        "Your smolagents version exposes neither OpenAIServerModel nor "
        "OpenAIModel. Run: pip install -U 'smolagents[toolkit,openai]'"
    )


def get_smolagents_model(name: str | None = None, **overrides: Any):
    """Drop-in replacement for `smolagents.InferenceClientModel`."""
    spec = get_backend(name)
    model_cls = _import_smolagents_openai_model()

    kwargs: dict[str, Any] = dict(
        model_id=spec.default_model,
        api_base=spec.base_url,
        api_key=get_api_key(spec),
        temperature=DEFAULT_TEMPERATURE,
        max_tokens=DEFAULT_MAX_TOKENS,
        flatten_messages_as_text=spec.flatten_messages_as_text,
    )
    if spec.extra_body:
        kwargs["extra_body"] = spec.extra_body
    kwargs.update(overrides)
    return model_cls(**kwargs)


def get_smolagents_litellm_model(name: str | None = None, **overrides: Any):
    """Alternative smolagents path that routes through LiteLLM.

    Useful when a provider needs quirk handling that the plain OpenAI client
    does not do. For Ollama, LiteLLM wants the `ollama_chat/` prefix and the
    *root* URL (no `/v1`).
    """
    from smolagents import LiteLLMModel

    spec = get_backend(name)
    if spec.name == "ollama":
        model_id = f"ollama_chat/{spec.default_model}"
        api_base = spec.base_url.removesuffix("/v1")
    else:
        model_id = f"openai/{spec.default_model}"
        api_base = spec.base_url

    kwargs: dict[str, Any] = dict(
        model_id=model_id,
        api_base=api_base,
        api_key=get_api_key(spec),
        temperature=DEFAULT_TEMPERATURE,
        max_tokens=DEFAULT_MAX_TOKENS,
        num_ctx=int(os.getenv("OLLAMA_NUM_CTX", "32768")),
    )
    kwargs.update(overrides)
    return LiteLLMModel(**kwargs)


# --------------------------------------------------------------------------
# Layer 2 — LangChain / LangGraph  (Unit 2.2)
# --------------------------------------------------------------------------
def get_langchain_chat(name: str | None = None, **overrides: Any):
    """Drop-in replacement for `ChatHuggingFace(HuggingFaceEndpoint(...))`.

    Returns a `ChatOpenAI` pointed at whatever backend `.env` selects, so
    `.bind_tools(...)` and every LangGraph prebuilt keep working.
    """
    from langchain_openai import ChatOpenAI

    spec = get_backend(name)
    kwargs: dict[str, Any] = dict(
        model=spec.default_model,
        base_url=spec.base_url,
        api_key=get_api_key(spec),
        temperature=DEFAULT_TEMPERATURE,
        max_tokens=DEFAULT_MAX_TOKENS,
        timeout=REQUEST_TIMEOUT,
    )
    if spec.extra_body:
        kwargs["extra_body"] = spec.extra_body
    kwargs.update(overrides)
    return ChatOpenAI(**kwargs)


def get_langchain_ollama_chat(**overrides: Any):
    """Native Ollama LangChain client.

    Only needed when you must set Ollama-specific knobs that the OpenAI
    protocol has no field for -- above all `num_ctx`, see docs/08.
    """
    from langchain_ollama import ChatOllama

    spec = get_backend("ollama")
    kwargs: dict[str, Any] = dict(
        model=spec.default_model,
        base_url=spec.base_url.removesuffix("/v1"),
        temperature=DEFAULT_TEMPERATURE,
        num_ctx=int(os.getenv("OLLAMA_NUM_CTX", "32768")),
    )
    kwargs.update(overrides)
    return ChatOllama(**kwargs)


# --------------------------------------------------------------------------
# Layer 3 — LlamaIndex  (Unit 2.3)
# --------------------------------------------------------------------------
def get_llamaindex_llm(name: str | None = None, **overrides: Any):
    """Drop-in replacement for `HuggingFaceInferenceAPI`.

    `is_function_calling_model=True` matters: without it LlamaIndex refuses to
    build a `FunctionAgent` and silently downgrades you to a ReAct text agent.
    """
    from llama_index.llms.openai_like import OpenAILike

    spec = get_backend(name)
    kwargs: dict[str, Any] = dict(
        model=spec.default_model,
        api_base=spec.base_url,
        api_key=get_api_key(spec),
        is_chat_model=True,
        is_function_calling_model=True,
        temperature=DEFAULT_TEMPERATURE,
        max_tokens=DEFAULT_MAX_TOKENS,
        timeout=REQUEST_TIMEOUT,
        context_window=int(os.getenv("OLLAMA_NUM_CTX", "32768")),
    )
    if spec.extra_body:
        # OpenAILike merges additional_kwargs into the request body.
        kwargs["additional_kwargs"] = dict(spec.extra_body)
    kwargs.update(overrides)
    return OpenAILike(**kwargs)


def get_llamaindex_embedding(model_name: str | None = None):
    """Local sentence-transformers embeddings, so RAG costs nothing either.

    Unit 2.3's RAG examples default to an OpenAI or HF embedding endpoint.
    This runs on your CPU; the model is ~90 MB and downloads once.
    """
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding

    return HuggingFaceEmbedding(
        model_name=model_name or os.getenv("EMBED_MODEL_ID", "BAAI/bge-small-en-v1.5")
    )
