# 1 · Setting up the environment

Everything here is free and works on Linux, macOS, and Windows.
Budget 20 minutes, most of it downloads.

## 0. Prerequisites

- **Conda** — [Miniconda](https://docs.conda.io/en/latest/miniconda.html) is
  enough (Anaconda works too). If you would rather use `venv` or `uv`, see the
  bottom of this page.
- **Git**.
- **~15 GB free disk** — Python packages take ~4 GB, one Qwen3 model 2–6 GB.
- A **free Hugging Face account** — <https://huggingface.co/join>.

## 1. Get the repo

```bash
git clone https://github.com/MINSUGLLY/CS6961_setup.git
cd CS6961_setup
```

## 2. Create the environment

One command does everything: creates the conda env, installs all
dependencies, installs this repo so `import cs6961_agents` works, and copies
`.env.example` to `.env`.

```bash
bash scripts/setup_env.sh
```

On Windows, run that from **Git Bash**, **WSL**, or use the manual steps below
in PowerShell.

<details>
<summary>What it does, if you prefer to run it yourself</summary>

```bash
conda create -y -n cs6961-agents python=3.12 pip
conda activate cs6961-agents
pip install -r requirements.txt
pip install -e .            # so `import cs6961_agents` works from anywhere
cp .env.example .env
```
</details>

**Every new terminal needs this line first:**

```bash
conda activate cs6961-agents
```

Forgetting it is the single most common cause of `ModuleNotFoundError` in this
course. If a command mysteriously stops working, check your prompt for
`(cs6961-agents)`.

## 3. Install a local model

See [2 · Running a model locally](02-local-models.md) for the details and the
model-size table. The short version:

```bash
bash scripts/install_ollama.sh    # installs Ollama, starts the server
bash scripts/pull_models.sh       # detects your hardware, pulls a fitting Qwen3
```

## 4. Get a Hugging Face token

Even with a local model you need one, for the Hub and for Unit 4.

1. <https://huggingface.co/settings/tokens> → **New token** → type **Read**.
2. Paste it into `.env`:

```ini
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxx
```

Never commit `.env`. It is in `.gitignore` already; keep it that way.

## 5. Verify

```bash
python scripts/doctor.py
```

This checks Python, every package, your `.env`, whether the LLM server answers,
whether a chat completion works, and whether the model can emit a tool call.
Expect all green:

```
  PASS  Python 3.12.14
  PASS  smolagents  1.26.0   used by: Unit 1, Unit 2.1, Unit 4
  PASS  LLM_BACKEND = ollama
  PASS  server responding at http://localhost:11434/v1
  PASS  chat completion works  model said: 'PONG'
  PASS  model emitted a tool call  get_weather({"city":"Salt Lake City"})

  Everything works. You are ready for the course.
```

If you do get stuck and ask for help, the output of `doctor.py` answers most
of the questions someone would otherwise have to ask you first.

## 6. Run something

```bash
python examples/00_raw_chat.py            # no framework: does the model answer?
python examples/01_unit1_manual_agent.py  # a ReAct loop written by hand
python examples/02_unit2_smolagents.py    # smolagents CodeAgent
python examples/03_unit2_langgraph.py     # LangGraph state machine
python examples/04_unit2_llamaindex.py    # LlamaIndex agent + local RAG
```

Read them in that order. Example 1 in particular is worth ten pages of
framework documentation: it shows that an agent is a `while` loop, a prompt
format, and a dict of Python functions.

---

## Alternative: venv or uv instead of conda

Nothing here needs conda specifically.

```bash
# venv
python3.12 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt && pip install -e .

# uv (fast)
uv venv --python 3.12 && source .venv/bin/activate
uv pip install -r requirements.txt && uv pip install -e .
```

## Alternative: no local install at all

If your machine cannot handle it, see [7 · No GPU, weak laptop, or
Colab](07-no-gpu-and-colab.md). You still will not pay anything.

---

Next: [2 · Running a model locally](02-local-models.md)
