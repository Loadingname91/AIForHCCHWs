#!/usr/bin/env python3
"""
Environment self-check.

    python scripts/doctor.py

Run this before asking for help, and paste its output into your question.
It checks, in order: python -> packages -> .env -> the LLM server -> a real
chat call -> a real tool call. The first FAIL is your actual problem; the
checks after it are usually just fallout.
"""

from __future__ import annotations

import importlib.metadata as md
import json
import os
import platform
import sys
import urllib.error
import urllib.request

GREEN, RED, YELLOW, BLUE, DIM, RESET = (
    "\033[32m", "\033[31m", "\033[33m", "\033[34m", "\033[2m", "\033[0m"
)

FAILURES: list[str] = []


def section(title: str) -> None:
    print(f"\n{BLUE}{'─' * 68}\n {title}\n{'─' * 68}{RESET}")


def ok(msg: str, detail: str = "") -> None:
    print(f"  {GREEN}PASS{RESET}  {msg}" + (f"  {DIM}{detail}{RESET}" if detail else ""))


def warn(msg: str, detail: str = "") -> None:
    print(f"  {YELLOW}WARN{RESET}  {msg}" + (f"  {DIM}{detail}{RESET}" if detail else ""))


def fail(msg: str, fix: str = "") -> None:
    print(f"  {RED}FAIL{RESET}  {msg}")
    if fix:
        for line in fix.strip().splitlines():
            print(f"        {DIM}{line}{RESET}")
    FAILURES.append(msg)


# ---------------------------------------------------------------------------
section("1. Python")
# ---------------------------------------------------------------------------
print(f"  {DIM}{platform.platform()}{RESET}")
print(f"  {DIM}{sys.executable}{RESET}")

if sys.version_info < (3, 10):
    fail(f"Python {platform.python_version()} is too old",
         "smolagents needs >= 3.10. Run: bash scripts/setup_env.sh")
else:
    ok(f"Python {platform.python_version()}")

env_name = os.getenv("CONDA_DEFAULT_ENV")
if env_name and env_name != "base":
    ok(f"conda environment active: {env_name}")
elif env_name == "base":
    warn("You are in the conda 'base' environment",
         "Prefer an isolated env: conda activate cs6961-agents")
else:
    warn("No conda environment detected", "That is fine if you use venv/uv instead.")

# ---------------------------------------------------------------------------
section("2. Packages")
# ---------------------------------------------------------------------------
REQUIRED = ["openai", "python-dotenv", "requests"]
FRAMEWORKS = {
    "smolagents": "Unit 1, Unit 2.1, Unit 4",
    "langgraph": "Unit 2.2",
    "langchain-openai": "Unit 2.2",
    "llama-index-core": "Unit 2.3",
    "llama-index-llms-openai-like": "Unit 2.3",
    "huggingface-hub": "Hub access: tokenizers, datasets, embeddings",
    }

for pkg in REQUIRED:
    try:
        ok(pkg, md.version(pkg))
    except md.PackageNotFoundError:
        fail(f"{pkg} is not installed", "pip install -r requirements.txt")

for pkg, where in FRAMEWORKS.items():
    try:
        ok(pkg, f"{md.version(pkg)}   used by: {where}")
    except md.PackageNotFoundError:
        warn(f"{pkg} missing (needed for {where})", "pip install -r requirements.txt")

try:
    import cs6961_agents  # noqa: F401
    ok("cs6961_agents importable", cs6961_agents.__version__)
except ImportError:
    fail("cannot import cs6961_agents",
         "Install the repo in editable mode:  pip install -e .")
    print(f"\n{RED}Stopping: the rest of the checks need this package.{RESET}")
    sys.exit(1)

# ---------------------------------------------------------------------------
section("3. Configuration (.env)")
# ---------------------------------------------------------------------------
from cs6961_agents.config import REPO_ROOT, get_api_key, get_backend  # noqa: E402

if (REPO_ROOT / ".env").exists():
    ok(".env found", str(REPO_ROOT / ".env"))
else:
    warn(".env not found — using defaults", "cp .env.example .env")

try:
    spec = get_backend()
except ValueError as exc:
    fail(str(exc))
    sys.exit(1)

ok(f"LLM_BACKEND = {spec.name}", spec.notes)
ok(f"model       = {spec.default_model}")
ok(f"base_url    = {spec.base_url}")

if not spec.is_free:
    warn(f"Backend '{spec.name}' is METERED — it spends credits or money",
         "For coursework set LLM_BACKEND=ollama in .env")

try:
    key = get_api_key(spec)
    if spec.api_key_env:
        ok(f"{spec.api_key_env} is set", f"{key[:6]}…{key[-4:]}" if len(key) > 12 else "short key")
except RuntimeError as exc:
    fail("missing API key", str(exc))

if os.getenv("HF_TOKEN"):
    ok("HF_TOKEN is set", "raises Hub download limits")
else:
    warn("HF_TOKEN is not set",
         "Optional, but Hub downloads are rate-limited without one.\n"
         "https://huggingface.co/settings/tokens")

# ---------------------------------------------------------------------------
section("4. LLM server reachable")
# ---------------------------------------------------------------------------
if spec.is_local:
    root = spec.base_url.removesuffix("/v1")
    probe = f"{root}/api/tags" if spec.name == "ollama" else f"{spec.base_url}/models"
    try:
        with urllib.request.urlopen(probe, timeout=5) as resp:
            body = json.loads(resp.read())
        ok(f"server responding at {spec.base_url}")

        if spec.name == "ollama":
            installed = [m["name"] for m in body.get("models", [])]
            if not installed:
                fail("Ollama is running but has no models",
                     "bash scripts/pull_models.sh")
            elif spec.default_model in installed or any(
                m.split(":")[0] == spec.default_model.split(":")[0] for m in installed
            ):
                ok(f"model '{spec.default_model}' available",
                   f"{len(installed)} model(s) installed")
            else:
                fail(f"model '{spec.default_model}' not pulled",
                     f"installed: {', '.join(installed)}\n"
                     f"fix:  ollama pull {spec.default_model}\n"
                     f"or:   set OLLAMA_MODEL_ID in .env to one of the above")
    except (urllib.error.URLError, OSError) as exc:
        fail(f"cannot reach {spec.base_url} ({exc})",
             "Ollama:   bash scripts/install_ollama.sh   (or: ollama serve)\n"
             "LM Studio: start the app, then 'Developer' -> 'Start Server'\n"
             "vLLM:     see docs/02-setup-ollama.md")
else:
    ok(f"cloud backend '{spec.name}' — connectivity is tested by the next step")

# ---------------------------------------------------------------------------
section("5. A real chat completion")
# ---------------------------------------------------------------------------
if FAILURES:
    warn("skipped — fix the failures above first")
else:
    try:
        from cs6961_agents import chat

        resp = chat(
            [{"role": "user", "content": "Reply with exactly: PONG"}],
            max_tokens=64,
            temperature=0,
        )
        message = resp.choices[0].message
        reply = (message.content or "").strip()
        if reply:
            ok("chat completion works", f"model said: {reply[:60]!r}")
        elif getattr(message, "reasoning", None) or getattr(message, "reasoning_content", None):
            fail("model returned an EMPTY answer — it spent the whole budget thinking",
                 "You are on a hybrid Qwen3 tag. Its reasoning goes to a separate\n"
                 "field and 'content' stays empty, which agents read as silence.\n"
                 "Fix, in order of preference:\n"
                 "  1. keep OLLAMA_DISABLE_THINKING=true in .env (the default)\n"
                 "  2. use an instruct-only tag:  OLLAMA_MODEL_ID=qwen3:4b-instruct\n"
                 "  3. raise LLM_MAX_TOKENS so the reasoning has room to finish\n"
                 "See docs/02-local-models.md#thinking-models")
        else:
            fail(f"model returned nothing (finish_reason={resp.choices[0].finish_reason})",
                 "See docs/08-troubleshooting.md")
    except Exception as exc:  # noqa: BLE001 — students need the raw error
        fail(f"chat completion failed: {type(exc).__name__}: {exc}",
             "See docs/08-troubleshooting.md")

# ---------------------------------------------------------------------------
section("6. Tool calling (the thing agents actually need)")
# ---------------------------------------------------------------------------
if FAILURES:
    warn("skipped — fix the failures above first")
else:
    try:
        from cs6961_agents import chat

        tools = [{
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
        }]
        resp = chat(
            [{"role": "user", "content": "What's the weather in Salt Lake City?"}],
            tools=tools,
            tool_choice="auto",
            temperature=0,
            max_tokens=256,
        )
        calls = resp.choices[0].message.tool_calls
        if calls:
            call = calls[0]
            ok("model emitted a tool call",
               f"{call.function.name}({call.function.arguments})")
        else:
            warn("model answered in prose instead of calling the tool",
                 "Your model is too small or not tool-tuned.\n"
                 "Try a bigger one:  ollama pull qwen3:8b\n"
                 "smolagents' CodeAgent still works without native tool calling.")
    except Exception as exc:  # noqa: BLE001
        warn(f"tool-calling probe failed: {type(exc).__name__}: {exc}",
             "Some servers reject the 'tools' field. CodeAgent is your fallback.")

# ---------------------------------------------------------------------------
section("Summary")
# ---------------------------------------------------------------------------
if FAILURES:
    print(f"  {RED}{len(FAILURES)} check(s) failed:{RESET}")
    for f in FAILURES:
        print(f"    - {f}")
    print(f"\n  Fix the FIRST one, then re-run. See {BLUE}docs/08-troubleshooting.md{RESET}")
    sys.exit(1)

print(f"  {GREEN}Everything works.{RESET} Your local model is set up and calling tools.\n")
print("  Try it:  python examples/00_raw_chat.py")
print("           python examples/02_unit2_smolagents.py")
