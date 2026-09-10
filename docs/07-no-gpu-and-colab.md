# 7 · No GPU, weak laptop, or Colab

You do not need a GPU to finish this course, and you still do not need to pay.
Pick whichever row describes your machine.

| Your situation | Do this |
|---|---|
| 16 GB RAM, no GPU | Ollama on CPU with `qwen3:4b-instruct`. Slow but complete. |
| 8 GB RAM, no GPU | Ollama on CPU with `qwen3:1.7b`, plus a free cloud backend for anything hard. |
| Chromebook / iPad / locked-down machine | Google Colab, or a free cloud backend only. |
| Any machine, but you want speed | A free cloud backend (Groq / Cerebras / Gemini). |

---

## Option 1 — Ollama on CPU

It works. Everything in this repo was designed to.

```bash
bash scripts/install_ollama.sh
ollama pull qwen3:4b-instruct     # or qwen3:1.7b on 8 GB
```

```ini
LLM_BACKEND=ollama
OLLAMA_MODEL_ID=qwen3:4b-instruct
```

Expect 5–20 seconds per agent step instead of well under one. A 6-step agent
run takes a minute or two. That is annoying, not blocking — and it is a good
reason to develop against a single question rather than all twenty.

Things that help on CPU:

- Prefer `-instruct` tags. Thinking variants generate hundreds of extra tokens
  before they answer, and on CPU you feel every one.
- Close everything else; CPU inference is memory-bandwidth bound.
- Set `LLM_MAX_TOKENS=1024` in `.env` while iterating.
- Keep `max_steps` low.

---

## Option 2 — Free cloud backends, no local model at all

Nothing in this repo requires a local model; `ollama` is just the default.

```ini
LLM_BACKEND=groq
GROQ_API_KEY=gsk_...
```

See [3 · Backends](03-backends.md) for Groq, Cerebras, Gemini, and OpenRouter.
All four are free to sign up for and rate-limited rather than billed. This is
the best option on a Chromebook or a managed university machine where you
cannot install anything.

---

## Option 3 — Google Colab

Colab's free tier gives you a T4 GPU (about 15 GB of VRAM), which is more than
enough for a 7–8B model. Two ways to use it.

### 3a. Ollama inside Colab

```python
# Cell 1 — install and start the server in the background
!curl -fsSL https://ollama.com/install.sh | sh
import subprocess, os, time
os.environ["OLLAMA_CONTEXT_LENGTH"] = "32768"   # the trap from docs/02
subprocess.Popen(["ollama", "serve"])
time.sleep(5)
!ollama pull qwen3:8b
```

```python
# Cell 2 — this repo
!git clone https://github.com/MINSUGLLY/CS6961_setup.git
%cd CS6961_setup
!pip install -q -r requirements.txt && pip install -q -e .

import os
os.environ["LLM_BACKEND"] = "ollama"
os.environ["OLLAMA_MODEL_ID"] = "qwen3:8b"
```

```python
# Cell 3 — check
!python scripts/doctor.py
```

Runtime → Change runtime type → **T4 GPU** first, or it will run on CPU.

### 3b. Transformers directly, no server

Heavier, but it is the closest thing to what the course's own notebooks do:

```python
!pip install -q "smolagents[transformers]"

from smolagents import CodeAgent, TransformersModel
model = TransformersModel(model_id="Qwen/Qwen3-8B", device_map="auto", max_new_tokens=2048)
agent = CodeAgent(tools=[], model=model, max_steps=6)
```

### Colab caveats

- Sessions die after ~90 minutes idle, or ~12 hours total. Anything on disk
  goes with them, so re-pull the model each session (a few minutes).
- Mount Drive if you want your notebooks to survive:
  `from google.colab import drive; drive.mount('/content/drive')`.
- The free tier throttles if you use it heavily all day.

---

## Running the course notebooks in Colab

The course's own notebooks will not find `cs6961_agents`, because it is not on
PyPI. Add one cell at the top:

```python
!git clone https://github.com/MINSUGLLY/CS6961_setup.git /content/CS6961_setup
!pip install -q -e /content/CS6961_setup

import os
os.environ["LLM_BACKEND"] = "groq"          # or "ollama" if you started one above
os.environ["GROQ_API_KEY"] = "gsk_..."      # better: Colab Secrets, see below
```

For keys, use Colab's Secrets panel (🔑 in the sidebar) rather than typing them
into a cell you might later share:

```python
from google.colab import userdata
os.environ["GROQ_API_KEY"] = userdata.get("GROQ_API_KEY")
```

Then patch the model line exactly as in [4 · Patching the course
notebooks](04-patching-the-course.md).

---

## What about Hugging Face Spaces as a dev environment?

Free Spaces run on CPU only, and a Space cannot see your laptop — inside it,
`localhost` means the Space's own machine. They are for publishing something,
not for developing it. Use Colab or a free cloud backend instead.

---

Next: [8 · Troubleshooting](08-troubleshooting.md)
