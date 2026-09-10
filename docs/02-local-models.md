# 2 · Running a model locally

Goal: an OpenAI-compatible chat API on `localhost` that costs nothing and works
offline. Four ways to get one, easiest first.

---

## Option A — Ollama (recommended)

Ollama is a single binary that downloads open-weight models and serves them at
`http://localhost:11434/v1`. No account, no key, no quota.

### Install

```bash
bash scripts/install_ollama.sh
```

Or by hand:

| OS | Command |
|---|---|
| Linux | `curl -fsSL https://ollama.com/install.sh \| sh` |
| macOS | `brew install --cask ollama`, or the app from <https://ollama.com/download> |
| Windows | Installer from <https://ollama.com/download> (or install inside WSL2) |

Check it is up:

```bash
curl http://localhost:11434/api/tags
```

If that hangs or refuses, start the server: `ollama serve`.

### Pick a model

```bash
bash scripts/pull_models.sh          # detects your VRAM and picks for you
bash scripts/pull_models.sh qwen3:8b # or choose explicitly
```

| Model | VRAM | Disk | Speed | Tool-calling quality |
|---|---|---|---|---|
| `qwen3:1.7b` | ~2 GB | 1.4 GB | fast | poor — demos only |
| `qwen3:4b-instruct` | ~3 GB | 2.5 GB | fast | ok — fine for Units 1–2 |
| `qwen3:8b` | ~6 GB | 5.2 GB | medium | **good — recommended baseline** |
| `qwen3:14b` | ~10 GB | 9.3 GB | slower | very good |
| `qwen3-coder:30b` | ~19 GB | 18 GB | fast (MoE) | best — tuned for agentic tool use |

No GPU? Everything still runs on CPU, just slowly. Start at `qwen3:1.7b` or
`qwen3:4b-instruct` and expect 5–20 seconds per agent step.

### Why Qwen3

Apache-2.0 licensed, strong at emitting well-formed tool calls even at 4B, and
Ollama ships tool-calling support for it. That combination is exactly what an
agents course needs. It is also a nice symmetry: the course's own default model
is `Qwen/Qwen2.5-Coder-32B-Instruct`, so you are running the same family, just
smaller and on your own hardware.

### Thinking models

Qwen3 tags come in flavours, and the difference matters more for agents than
for chat:

- `qwen3:8b`, `qwen3:14b` — **hybrid**. They reason at length before answering.
- `qwen3:4b-instruct`, `qwen3:30b-a3b-instruct` — **instruct-only**. No
  reasoning phase.
- `qwen3:4b-thinking` — always reasons, at length.

Here is the failure it causes. Through Ollama's OpenAI-compatible endpoint, a
hybrid model's reasoning goes into a separate `reasoning` field, and `content`
stays **empty** until the reasoning finishes:

```json
{"message": {"content": "",
             "reasoning": "Okay, the user wants me to reply with exactly..."},
 "finish_reason": "length"}
```

Every agent framework reads `content`. So with a modest `max_tokens` the model
appears to have said nothing at all, and you get an empty step, a parse error,
or a silent retry loop.

**Only one of the three obvious fixes actually works on `/v1`:**

| Attempt | Result |
|---|---|
| `{"think": false}` in the request body | ignored on `/v1` (it does work on the native `/api/chat`) |
| `/no_think` appended to the prompt | ignored by the 2507-era Qwen3 tags |
| `{"reasoning_effort": "none"}` | **works** — and is harmless on models with no reasoning phase |

This repo sends `reasoning_effort: "none"` for the Ollama backend by default.
Turn it off if you want the reasoning back:

```ini
OLLAMA_DISABLE_THINKING=false   # default is true
```

Reasoning genuinely helps on hard one-shot questions. It hurts agent loops,
because every step pays for it again and the extra tokens confuse strict
output parsers. For this course, leave it off.

If you are calling the API yourself rather than through a framework, either
use the helper that applies this for you:

```python
from cs6961_agents import chat
response = chat([{"role": "user", "content": "Reply with exactly: PONG"}], max_tokens=64)
```

or pass it explicitly:

```python
client.chat.completions.create(..., extra_body={"reasoning_effort": "none"})
```

Simplest of all: **prefer an `-instruct` tag** and the problem never arises.

Leftover `<think>…</think>` blocks in text output are the same issue in a
different shape. Strip them before parsing — `extract_json()` in
[`examples/01_unit1_manual_agent.py`](../examples/01_unit1_manual_agent.py) and
`finalize()` in [`examples/05_unit4_gaia_agent.py`](../examples/05_unit4_gaia_agent.py)
both do.

### The context-window trap (read this one)

**This is the single biggest cause of "local models are useless" in this course.**

Ollama chooses a context window based on your VRAM: **4k on small GPUs**, 32k on
larger ones, 256k on very large ones. When a prompt does not fit, Ollama
**truncates it silently**. No error, no warning.

Agent prompts are long: system prompt + every tool's JSON schema + every past
thought, action and observation. They cross 4k tokens after two or three steps.
So the agent works, then abruptly "forgets" its task, loops, or repeats a tool
call forever — and it looks like the model is stupid rather than blindfolded.

Check what you actually got, while a model is loaded:

```bash
ollama ps
# NAME       ID    SIZE    PROCESSOR    CONTEXT    UNTIL
# qwen3:8b   ...   8.0 GB  100% GPU     4096   <-- too small
```

Fix it by setting `OLLAMA_CONTEXT_LENGTH` **for the server**, before it starts:

```bash
# macOS / any manual start
OLLAMA_CONTEXT_LENGTH=32768 ollama serve
```

```bash
# Linux with systemd
sudo systemctl edit ollama
# add these three lines:
#   [Service]
#   Environment="OLLAMA_CONTEXT_LENGTH=32768"
sudo systemctl restart ollama
```

```powershell
# Windows: set a user environment variable, then restart Ollama from the tray
setx OLLAMA_CONTEXT_LENGTH 32768
```

A larger context costs VRAM. If the model stops fitting on your GPU, drop to
16384, or use a smaller model.

Note that `OLLAMA_NUM_CTX` in this repo's `.env` only reaches clients that can
send Ollama-native options (`get_langchain_ollama_chat`,
`get_smolagents_litellm_model`). The plain OpenAI protocol has no field for it,
so the server-side setting above is the one that always works.

### Useful commands

```bash
ollama list                 # what you have
ollama ps                   # what is loaded, and its context size
ollama run qwen3:8b         # interactive chat, good for a quick sanity check
ollama rm qwen3:1.7b        # reclaim disk
OLLAMA_DEBUG=1 ollama serve # verbose logs when something is wrong
```

---

## Option B — LM Studio (a GUI, nice on Windows)

1. Install from <https://lmstudio.ai>.
2. **Discover** tab → search `Qwen3 8B` → download a **GGUF, Q4_K_M** build.
3. **Developer** tab → **Start Server** (default port 1234).
4. In `.env`:

```ini
LLM_BACKEND=lmstudio
LMSTUDIO_MODEL_ID=qwen/qwen3-8b     # copy the exact id the server prints
```

Set the context length in the model's load settings — the same 4k trap applies.

---

## Option C — vLLM (fastest, needs a real NVIDIA GPU)

Best for lab machines and for anything batched.

```bash
pip install vllm
vllm serve Qwen/Qwen3-8B \
  --max-model-len 32768 \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  --port 8000
```

```ini
LLM_BACKEND=vllm
VLLM_MODEL_ID=Qwen/Qwen3-8B
```

`--enable-auto-tool-choice` and `--tool-call-parser` are not optional: without
them vLLM will not return `tool_calls` and every framework's tool binding
silently does nothing.

---

## Option D — llama.cpp (best CPU-only performance)

```bash
# macOS
brew install llama.cpp
# then, downloading a GGUF straight from the Hub:
llama-server -hf Qwen/Qwen3-8B-GGUF:Q4_K_M --port 8080 -c 32768 --jinja
```

```ini
LLM_BACKEND=llamacpp
LLAMACPP_MODEL_ID=qwen3-8b
```

`--jinja` matters: it applies the model's real chat template, which is what
makes tool calling work.

---

## Verifying any of them

```bash
python scripts/doctor.py
```

Section 6 of the output is the one that matters for this course. If it says
*"model emitted a tool call"*, you are done. If it says *"answered in prose
instead"*, your model is too small or is not tool-tuned — move up a size, or
use smolagents' `CodeAgent`, which does not need native tool calling.

---

Next: [3 · Every backend, and free cloud fallbacks](03-backends.md)
