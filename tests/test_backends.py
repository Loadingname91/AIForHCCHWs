"""
Tests for the backend abstraction.

    pytest -q                 # offline tests only
    pytest -q -m live         # also hit the configured LLM server

The offline tests are the useful ones for a TA: they catch a broken `.env`,
a renamed upstream class, or a typo in the backend catalogue, without needing
a model loaded.
"""

from __future__ import annotations

import os

import pytest

from cs6961_agents import config
from cs6961_agents.config import BACKENDS, get_api_key, get_backend


# --------------------------------------------------------------------------
# The catalogue itself
# --------------------------------------------------------------------------
def test_every_backend_has_a_v1_style_base_url():
    for name, spec in BACKENDS.items():
        assert spec.base_url.startswith("http"), name
        assert not spec.base_url.endswith("/"), f"{name}: trailing slash breaks urljoin"


def test_every_backend_declares_a_default_model():
    for name, spec in BACKENDS.items():
        assert spec.default_model, name


def test_local_backends_point_at_localhost():
    for name, spec in BACKENDS.items():
        if spec.is_local:
            assert "localhost" in spec.base_url or "127.0.0.1" in spec.base_url, name


def test_local_backends_need_no_api_key():
    for name, spec in BACKENDS.items():
        if spec.is_local:
            assert spec.api_key_env is None, f"{name} should not require a key"
            assert get_api_key(spec)  # returns a placeholder, does not raise


def test_metered_backends_are_flagged_as_such():
    """A regression guard: nobody should ever mark these free by accident."""
    for name in ("hf", "openai"):
        assert not BACKENDS[name].is_free, f"{name} costs money and must say so"


# --------------------------------------------------------------------------
# Resolution from the environment
# --------------------------------------------------------------------------
def test_unknown_backend_fails_loudly():
    with pytest.raises(ValueError, match="Unknown LLM_BACKEND"):
        get_backend("does-not-exist")


def test_env_overrides_the_default_model(monkeypatch):
    monkeypatch.setenv("OLLAMA_MODEL_ID", "qwen3:14b")
    assert get_backend("ollama").default_model == "qwen3:14b"


def test_env_overrides_the_base_url(monkeypatch):
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://gpu-node:11434/v1")
    assert get_backend("ollama").base_url == "http://gpu-node:11434/v1"


def test_backend_specific_override_beats_the_generic_one(monkeypatch):
    monkeypatch.setenv("LLM_MODEL_ID", "generic")
    monkeypatch.setenv("OLLAMA_MODEL_ID", "specific")
    assert get_backend("ollama").default_model == "specific"


def test_missing_cloud_key_raises_a_helpful_error(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="GROQ_API_KEY"):
        get_api_key(get_backend("groq"))


# --------------------------------------------------------------------------
# The framework adapters. These construct objects but send no requests, so
# they catch upstream renames without needing a running server.
# --------------------------------------------------------------------------
def test_smolagents_adapter_constructs():
    from cs6961_agents import get_smolagents_model

    model = get_smolagents_model("ollama")
    assert model.model_id == get_backend("ollama").default_model


def test_langchain_adapter_constructs_and_can_bind_tools():
    from langchain_core.tools import tool

    from cs6961_agents import get_langchain_chat

    @tool
    def add(a: float, b: float) -> float:
        """Add two numbers."""
        return a + b

    chat = get_langchain_chat("ollama")
    assert chat.bind_tools([add]) is not None


def test_llamaindex_adapter_advertises_function_calling():
    """Without this flag LlamaIndex silently downgrades FunctionAgent."""
    from cs6961_agents import get_llamaindex_llm

    llm = get_llamaindex_llm("ollama")
    assert llm.metadata.is_function_calling_model is True


def test_describe_backend_says_whether_it_costs_money(monkeypatch):
    from cs6961_agents import describe_backend

    monkeypatch.setenv("LLM_BACKEND", "ollama")
    assert "free" in describe_backend()

    monkeypatch.setenv("LLM_BACKEND", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    assert "METERED" in describe_backend()


# --------------------------------------------------------------------------
# Live tests — only run with `pytest -m live` and a server actually up.
# --------------------------------------------------------------------------
def test_ollama_disables_thinking_by_default(monkeypatch):
    """Hybrid Qwen3 tags return empty `content` unless reasoning is switched off."""
    monkeypatch.delenv("OLLAMA_DISABLE_THINKING", raising=False)
    assert get_backend("ollama").extra_body == {"reasoning_effort": "none"}


def test_thinking_can_be_re_enabled(monkeypatch):
    monkeypatch.setenv("OLLAMA_DISABLE_THINKING", "false")
    assert get_backend("ollama").extra_body == {}


def test_cloud_backends_do_not_get_the_ollama_switch():
    """`reasoning_effort` is verified on Ollama only; do not send it elsewhere."""
    for name in ("groq", "hf", "openai", "gemini"):
        assert "reasoning_effort" not in BACKENDS[name].extra_body, name


@pytest.mark.live
def test_live_chat_completion():
    from cs6961_agents import get_openai_client

    spec = get_backend()
    response = get_openai_client().chat.completions.create(
        model=spec.default_model,
        messages=[{"role": "user", "content": "Reply with exactly: PONG"}],
        max_tokens=512,   # room for a thinking model, if one slipped through
        temperature=0,
    )
    content = response.choices[0].message.content or ""
    assert content.strip(), (
        "empty content — the model reasoned instead of answering; "
        "see docs/02-local-models.md#thinking-models"
    )
    assert "PONG" in content.upper()


@pytest.mark.live
def test_live_tool_call():
    """The capability the whole course depends on."""
    from cs6961_agents import get_openai_client

    spec = get_backend()
    response = get_openai_client().chat.completions.create(
        model=spec.default_model,
        messages=[{"role": "user", "content": "What is the weather in Paris?"}],
        tools=[{
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the current weather for a city.",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            },
        }],
        tool_choice="auto",
        temperature=0,
        max_tokens=256,
    )
    calls = response.choices[0].message.tool_calls
    assert calls, "model answered in prose instead of calling the tool"
    assert calls[0].function.name == "get_weather"


# --------------------------------------------------------------------------
# Unit 4 answer cleaning. GAIA is scored by exact string match, so this is
# worth more points than most of the agent logic.
# --------------------------------------------------------------------------
def _finalize():
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "examples" / "05_unit4_gaia_agent.py"
    spec = importlib.util.spec_from_file_location("gaia_agent", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.finalize


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("4", "4"),
        ("The answer is 4.", "4"),
        ("Final answer: Paris", "Paris"),
        ('"Paris"', "Paris"),
        ("1,234", "1234"),                              # GAIA wants no separators
        ("<think>let me see</think>7", "7"),            # hybrid Qwen3 leakage
        ("Stdout:\n\nOutput: 395", "395"),              # smolagents execution log
        ("Last output from code snippet:\n42", "42"),
        ("Paris, France", "Paris, France"),             # must NOT strip this comma
    ],
)
def test_finalize_produces_exact_match_answers(raw, expected):
    assert _finalize()(raw) == expected
