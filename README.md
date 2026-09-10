# CS6961 · AI for Human-Centered Computing
## Running the Hugging Face Agents Course without paying for it

The [HF Agents Course](https://huggingface.co/learn/agents-course/unit0/introduction)
is free. Running its code is not — every unit calls a model on someone else's
GPU, and Unit 2.2 calls `gpt-4o` outright. A free Hugging Face account has
about **$0.10 of monthly inference credits**, and one afternoon of debugging a
Unit 4 agent will spend all of it.

This repo is **a reference setup for getting around that**: it runs the whole
course on a model on your own machine — free, offline, no rate limit — with
cloud APIs available as a one-line fallback.

> **It is optional.** Nothing here is a course requirement. Use it whole, take
> one idea from it, or ignore it and configure your own environment. Your
> assignments and their requirements live on Canvas.

```
Course:    model = InferenceClientModel(model_id="Qwen/Qwen2.5-Coder-32B-Instruct")
This repo: model = get_smolagents_model()
```

That is the entire change. Your tools, your agent, your `.run()` call are
untouched.

---

## Quick start (about 20 minutes, mostly downloads)

```bash
git clone https://github.com/MINSUGLLY/CS6961_setup.git
cd CS6961_setup

bash scripts/setup_env.sh        # conda env + all dependencies + .env
conda activate cs6961-agents     # every new terminal needs this

bash scripts/install_ollama.sh   # local model server
bash scripts/pull_models.sh      # detects your hardware, pulls a fitting Qwen3

python scripts/doctor.py         # verify everything, end to end
```

All green means you are ready:

```
  PASS  Python 3.12.14
  PASS  smolagents  1.26.0   used by: Unit 1, Unit 2.1, Unit 4
  PASS  LLM_BACKEND = ollama
  PASS  server responding at http://localhost:11434/v1
  PASS  chat completion works  model said: 'PONG'
  PASS  model emitted a tool call  get_weather({"city":"Salt Lake City"})

  Everything works. You are ready for the course.
```

No GPU? It still works — see [docs/07](docs/07-no-gpu-and-colab.md).

---

## The idea worth taking away

Ollama, LM Studio, vLLM, llama.cpp, Groq, Cerebras, Gemini, OpenRouter, and
Hugging Face's own router **all speak the same OpenAI-compatible protocol.**
So a "backend" is three values:

```
base_url  +  api_key  +  model_name
```

Put those in `.env` and provider choice becomes a one-line edit rather than a
rewrite. Switching from your laptop to a free cloud API for one comparison run
costs you nothing but a config change:

```ini
# .env
LLM_BACKEND=ollama        # free, local, offline   <- default
# LLM_BACKEND=groq        # free tier, very fast
# LLM_BACKEND=hf          # what the course docs use; spends credits
```

---

## Run the examples

Read them in order. Each prints its backend on line one, so you always know
whether you are spending anything.

| File | What it teaches |
|---|---|
| [`00_raw_chat.py`](examples/00_raw_chat.py) | proof of life, no framework at all |
| [`01_unit1_manual_agent.py`](examples/01_unit1_manual_agent.py) | **a ReAct loop written by hand — read this one** |
| [`02_unit2_smolagents.py`](examples/02_unit2_smolagents.py) | smolagents `CodeAgent` |
| [`03_unit2_langgraph.py`](examples/03_unit2_langgraph.py) | LangGraph state machine + tools |
| [`04_unit2_llamaindex.py`](examples/04_unit2_llamaindex.py) | LlamaIndex agent + local RAG |
| [`05_unit4_gaia_agent.py`](examples/05_unit4_gaia_agent.py) | a GAIA agent with real tools |
| [`06_unit4_local_eval.py`](examples/06_unit4_local_eval.py) | run it over the whole GAIA set, locally |

```bash
python examples/01_unit1_manual_agent.py
```

Example 1 is the one to actually read. It shows that an agent is a `while`
loop, a prompt format, and a dict of Python functions — everything the
frameworks add later is ergonomics on top of that.

---

## Documentation

| | |
|---|---|
| [0 · Why this repo exists](docs/00-why-this-repo.md) | the cost problem, and how to choose a backend |
| [1 · Setting up the environment](docs/01-setup-environment.md) | conda, install, verify |
| [2 · Running a model locally](docs/02-local-models.md) | Ollama, LM Studio, vLLM, llama.cpp — **and the context-window trap** |
| [3 · Backends](docs/03-backends.md) | every option, and the free cloud fallbacks |
| [4 · Patching the course notebooks](docs/04-patching-the-course.md) | **the one line to change, unit by unit** |
| [5 · Building a GAIA agent](docs/05-unit4-gaia-agent.md) | Unit 4, local evaluation, and why exact-match output is hard |
| [6 · Cost and safety](docs/06-cost-and-safety.md) | not spending money, not running code you did not read |
| [7 · No GPU, weak laptop, or Colab](docs/07-no-gpu-and-colab.md) | it still works |
| [8 · Troubleshooting](docs/08-troubleshooting.md) | symptom → cause → fix |

---

## The three traps that cost students the most time

1. **Ollama's context window.** It picks 4k on small GPUs and then **truncates
   longer prompts silently.** Agent prompts pass 4k after two or three steps,
   so the agent appears to "forget" its task or loop forever. Check with
   `ollama ps`; fix with `OLLAMA_CONTEXT_LENGTH=32768`.
   → [docs/02](docs/02-local-models.md#the-context-window-trap-read-this-one)

2. **Qwen3's reasoning mode returns an empty answer.** Hybrid tags like
   `qwen3:8b` put their reasoning in a separate field and leave `content`
   empty, which every framework reads as "the model said nothing". Of the
   three fixes you will find online, only `reasoning_effort: "none"` works.
   → [docs/02](docs/02-local-models.md#thinking-models)

3. **Small models need `CodeAgent`, not `ToolCallingAgent`.** A 4B–8B model is
   much better at writing one line of Python than at filling a JSON tool
   schema. → [docs/04](docs/04-patching-the-course.md#prefer-codeagent-over-toolcallingagent-locally)

---

## Repository layout

```
├── scripts/
│   ├── setup_env.sh          conda env + dependencies + .env
│   ├── install_ollama.sh     install and start the local model server
│   ├── pull_models.sh        hardware detection + model download
│   └── doctor.py             end-to-end environment check  <- run when stuck
├── src/cs6961_agents/
│   ├── config.py             the backend catalogue, read from .env
│   └── backends.py           one adapter per framework
├── examples/                 00 → 06, in reading order
├── docs/                     00 → 08
├── tests/                    pytest checks for the adapters
└── .env.example              copy to .env; edit three lines
```

---

## What this needs to run

- Python 3.10+ (3.12 recommended), conda or venv
- ~15 GB disk (packages ~4 GB, one model 2–6 GB)
- A free [Hugging Face account](https://huggingface.co/join) — optional, but it
  raises the download limits for tokenizers and embedding models
- A GPU is **optional**

## License

Reference material for CS6961. The models referenced (Qwen3, Qwen3-Coder) are
Apache-2.0.
