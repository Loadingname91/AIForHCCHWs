"""
Central configuration for the CS6961 local-first Agents Course setup.

The whole point of this file
----------------------------
The Hugging Face Agents Course hard-codes one specific way of getting an LLM:
`InferenceClientModel` -> Hugging Face Inference Providers -> your credit card
(or your $0.10/month of free credits, which one Unit-1 exercise can burn).

Almost every LLM server on earth -- Ollama, LM Studio, vLLM, llama.cpp, Groq,
OpenRouter, Cerebras, Together, Google Gemini, and Hugging Face's own router --
speaks the *same* OpenAI-compatible `/v1/chat/completions` protocol.

So instead of learning N different SDKs, you learn one idea:

    a "backend" is just (base_url, api_key, model_name)

Switch backends by editing three lines in `.env`. Nothing else in your agent
code changes. That is the entire trick, and it is worth internalizing well
beyond this course.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

# Load .env from the repository root, no matter where the script is run from.
REPO_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(REPO_ROOT / ".env")


@dataclass(frozen=True)
class BackendSpec:
    """Everything needed to talk to one LLM provider."""

    name: str
    base_url: str
    api_key_env: str | None
    default_model: str
    # Free as in "you will not be billed while doing this course".
    is_free: bool
    # Runs on your own machine (works offline, no rate limits, no signup).
    is_local: bool
    notes: str = ""
    # Some servers reject the OpenAI `stop` parameter or multimodal content
    # blocks; smolagents needs to be told to flatten messages for those.
    flatten_messages_as_text: bool = False
    # Extra JSON merged into every request body (e.g. disabling Qwen3 thinking).
    extra_body: dict = field(default_factory=dict)


# --------------------------------------------------------------------------
# The catalogue. Add your own row if your lab has a private endpoint.
# --------------------------------------------------------------------------
BACKENDS: dict[str, BackendSpec] = {
    # ---------------- LOCAL (recommended default) -------------------------
    "ollama": BackendSpec(
        name="ollama",
        base_url="http://localhost:11434/v1",
        api_key_env=None,  # Ollama ignores the key, but OpenAI SDK requires one.
        default_model="qwen3:8b",
        is_free=True,
        is_local=True,
        flatten_messages_as_text=True,
        # Hybrid Qwen3 tags reason before answering, and Ollama returns that
        # reasoning in a separate field while `content` stays empty. Agents
        # parse `content`, so they see nothing. See _thinking_extra_body().
        extra_body={"reasoning_effort": "none"},
        notes="Easiest local option. Works on Linux/macOS/Windows, CPU or GPU.",
    ),
    "lmstudio": BackendSpec(
        name="lmstudio",
        base_url="http://localhost:1234/v1",
        api_key_env=None,
        default_model="qwen/qwen3-8b",
        is_free=True,
        is_local=True,
        flatten_messages_as_text=True,
        notes="GUI alternative to Ollama. Nice on Windows if the CLI scares you.",
    ),
    "vllm": BackendSpec(
        name="vllm",
        base_url="http://localhost:8000/v1",
        api_key_env=None,
        default_model="Qwen/Qwen3-8B",
        is_free=True,
        is_local=True,
        notes="Fastest local serving, needs a real NVIDIA GPU. See docs/02.",
    ),
    "llamacpp": BackendSpec(
        name="llamacpp",
        base_url="http://localhost:8080/v1",
        api_key_env=None,
        default_model="qwen3-8b",
        is_free=True,
        is_local=True,
        flatten_messages_as_text=True,
        notes="llama.cpp server. Best CPU-only performance, most manual setup.",
    ),
    # ---------------- FREE-TIER CLOUD (no credit card) --------------------
    "groq": BackendSpec(
        name="groq",
        base_url="https://api.groq.com/openai/v1",
        api_key_env="GROQ_API_KEY",
        default_model="llama-3.3-70b-versatile",
        is_free=True,
        is_local=False,
        flatten_messages_as_text=True,
        notes="Very generous free tier and absurdly fast. Daily token cap.",
    ),
    "cerebras": BackendSpec(
        name="cerebras",
        base_url="https://api.cerebras.ai/v1",
        api_key_env="CEREBRAS_API_KEY",
        default_model="qwen-3-32b",
        is_free=True,
        is_local=False,
        flatten_messages_as_text=True,
        notes="Free tier with a daily request cap. Also extremely fast.",
    ),
    "gemini": BackendSpec(
        name="gemini",
        base_url="https://generativelanguage.googleapis.com/v1beta/openai",
        api_key_env="GOOGLE_API_KEY",
        default_model="gemini-2.0-flash",
        is_free=True,
        is_local=False,
        notes="Google's OpenAI-compatible endpoint. Free tier, rate-limited.",
    ),
    "openrouter": BackendSpec(
        name="openrouter",
        base_url="https://openrouter.ai/api/v1",
        api_key_env="OPENROUTER_API_KEY",
        default_model="qwen/qwen3-8b:free",
        is_free=True,
        is_local=False,
        notes="Aggregator. Models with a ':free' suffix cost nothing.",
    ),
    # ---------------- THE COURSE DEFAULT (metered) ------------------------
    "hf": BackendSpec(
        name="hf",
        base_url="https://router.huggingface.co/v1",
        api_key_env="HF_TOKEN",
        default_model="Qwen/Qwen2.5-Coder-32B-Instruct",
        is_free=False,
        is_local=False,
        notes="What the course docs use. Metered against your HF credits.",
    ),
    # ---------------- PAID (only if you already have a key) ---------------
    "openai": BackendSpec(
        name="openai",
        base_url="https://api.openai.com/v1",
        api_key_env="OPENAI_API_KEY",
        default_model="gpt-4o-mini",
        is_free=False,
        is_local=False,
        notes="Costs real money. Never required for this course.",
    ),
}


def get_backend(name: str | None = None) -> BackendSpec:
    """Resolve the active backend, honouring `.env` overrides."""
    name = (name or os.getenv("LLM_BACKEND", "ollama")).strip().lower()
    if name not in BACKENDS:
        known = ", ".join(sorted(BACKENDS))
        raise ValueError(f"Unknown LLM_BACKEND={name!r}. Choose one of: {known}")

    spec = BACKENDS[name]

    # Per-backend overrides let one .env hold settings for several backends,
    # e.g. OLLAMA_BASE_URL / GROQ_MODEL_ID.
    prefix = name.upper()
    base_url = os.getenv(f"{prefix}_BASE_URL") or os.getenv("LLM_BASE_URL") or spec.base_url
    model_id = os.getenv(f"{prefix}_MODEL_ID") or os.getenv("LLM_MODEL_ID") or spec.default_model

    return BackendSpec(
        name=spec.name,
        base_url=base_url,
        api_key_env=spec.api_key_env,
        default_model=model_id,
        is_free=spec.is_free,
        is_local=spec.is_local,
        notes=spec.notes,
        flatten_messages_as_text=spec.flatten_messages_as_text,
        extra_body=_thinking_extra_body(spec),
    )


def _thinking_extra_body(spec: BackendSpec) -> dict:
    """Decide whether to ask the server to skip the model's reasoning phase.

    Qwen3's hybrid tags (`qwen3:8b`, `qwen3:14b`, ...) emit a long reasoning
    pass before answering. Through Ollama's OpenAI-compatible endpoint that
    reasoning arrives in a separate `reasoning` field and `content` comes back
    EMPTY until it finishes -- so with a modest `max_tokens` you get an empty
    string and `finish_reason="length"`, which every agent framework reads as
    "the model said nothing".

    Of the three ways people try to turn this off, only one works on `/v1`:

        {"think": false}            ignored on /v1 (it works on /api/chat)
        "/no_think" in the prompt   ignored by the 2507-era Qwen3 tags
        {"reasoning_effort":"none"} works, and is harmless on models that
                                    have no reasoning phase at all

    Set OLLAMA_DISABLE_THINKING=false in .env if you actually want the
    reasoning (it does improve hard single-shot answers -- it just breaks
    agent loops).
    """
    extra = dict(spec.extra_body)
    if "reasoning_effort" in extra:
        if os.getenv("OLLAMA_DISABLE_THINKING", "true").strip().lower() in {"0", "false", "no"}:
            extra.pop("reasoning_effort")
    return extra


def get_api_key(spec: BackendSpec) -> str:
    """Return the API key for a backend, or a harmless placeholder.

    Local servers do not authenticate, but the OpenAI SDK refuses to start
    without *some* string, so we hand it one.
    """
    if spec.api_key_env is None:
        return os.getenv(f"{spec.name.upper()}_API_KEY", "not-needed")

    key = os.getenv(spec.api_key_env, "")
    if not key:
        raise RuntimeError(
            f"Backend {spec.name!r} needs {spec.api_key_env} to be set.\n"
            f"  1. Copy .env.example to .env\n"
            f"  2. Fill in {spec.api_key_env}=...\n"
            f"  3. Or switch to a local backend: LLM_BACKEND=ollama"
        )
    return key


# Generation defaults. Agents are not creative writing: keep temperature low so
# the model emits parseable tool calls instead of prose about tool calls.
DEFAULT_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.2"))
DEFAULT_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "4096"))
REQUEST_TIMEOUT = float(os.getenv("LLM_TIMEOUT", "300"))
