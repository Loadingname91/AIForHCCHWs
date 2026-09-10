# 3 · Every backend, and the free cloud fallbacks

Your laptop is the default. But sometimes it is too slow, or you are on a
borrowed machine, or you want one run on a much larger model to compare
against. Everything below is a three-line change to `.env`.

## How switching works

```ini
# .env
LLM_BACKEND=ollama        # pick a row from the table below
OLLAMA_MODEL_ID=qwen3:8b  # optional; each backend has a sensible default
```

The code never changes:

```python
from cs6961_agents import get_smolagents_model, get_langchain_chat, get_llamaindex_llm
```

Confirm which one is live at any time:

```bash
python -c "from cs6961_agents import describe_backend; print(describe_backend())"
# [ollama] qwen3:8b @ http://localhost:11434/v1 (local, free)
```

Every example script prints this on its first line, on purpose: you should
never be unsure whether you are spending money.

---

## Local backends

| `LLM_BACKEND` | Server | Default port | Set-up |
|---|---|---|---|
| `ollama` | Ollama | 11434 | [docs/02 · A](02-local-models.md#option-a--ollama-recommended) |
| `lmstudio` | LM Studio | 1234 | [docs/02 · B](02-local-models.md#option-b--lm-studio-a-gui-nice-on-windows) |
| `vllm` | vLLM | 8000 | [docs/02 · C](02-local-models.md#option-c--vllm-fastest-needs-a-real-nvidia-gpu) |
| `llamacpp` | llama.cpp | 8080 | [docs/02 · D](02-local-models.md#option-d--llamacpp-best-cpu-only-performance) |

---

## Free-tier cloud backends

All four give you a key without a credit card. All four are rate-limited rather
than billed, so the worst case is a `429`, not a charge. Check the current
limits on the linked pages — providers change them often, so this table
deliberately does not quote numbers that would go stale.

### Groq — the best fallback

Very fast, generous free tier, OpenAI-compatible.

1. Sign up at <https://console.groq.com>, create a key at
   <https://console.groq.com/keys>.
2. `.env`:

```ini
LLM_BACKEND=groq
GROQ_API_KEY=gsk_xxxxxxxxxxxx
GROQ_MODEL_ID=llama-3.3-70b-versatile
```

Model list: <https://console.groq.com/docs/models>. Prefer a model marked as
supporting tool use.

### Cerebras

Also extremely fast, free tier with a daily cap, and it serves Qwen3.

1. Key at <https://cloud.cerebras.ai>.
2. `.env`:

```ini
LLM_BACKEND=cerebras
CEREBRAS_API_KEY=csk-xxxxxxxxxxxx
CEREBRAS_MODEL_ID=qwen-3-32b
```

### Google Gemini

Google exposes an OpenAI-compatible endpoint, so it drops straight in.

1. Key at <https://aistudio.google.com/apikey>.
2. `.env`:

```ini
LLM_BACKEND=gemini
GOOGLE_API_KEY=AIza...
GEMINI_MODEL_ID=gemini-2.0-flash
```

### OpenRouter

An aggregator. Models whose id ends in `:free` cost nothing.

1. Key at <https://openrouter.ai/keys>.
2. `.env`:

```ini
LLM_BACKEND=openrouter
OPENROUTER_API_KEY=sk-or-v1-xxxxxxxx
OPENROUTER_MODEL_ID=qwen/qwen3-8b:free
```

Browse free models: <https://openrouter.ai/models?max_price=0>. Free models are
heavily rate-limited and sometimes queue for minutes.

---

## The metered backends

### Hugging Face Inference Providers — the course default

This is what the course docs use, and what this repo exists to avoid using by
accident.

```ini
LLM_BACKEND=hf
HF_TOKEN=hf_xxxxxxxxxxxx
HF_MODEL_ID=Qwen/Qwen2.5-Coder-32B-Instruct
```

Note that `https://router.huggingface.co/v1` is itself OpenAI-compatible, so
this repo reaches it through the same `ChatOpenAI`/`OpenAIServerModel` path as
everything else. `InferenceClientModel` is not doing anything you cannot do
with three strings.

Before and after any session on this backend, check your usage at
<https://huggingface.co/settings/billing>. Set `max_steps` low. Do not leave a
loop running while you go for coffee.

### OpenAI

```ini
LLM_BACKEND=openai
OPENAI_API_KEY=sk-...
OPENAI_MODEL_ID=gpt-4o-mini
```

Costs real money. **Nothing in this course requires it.** It is here only so
that students who already have a key are not tempted to hard-code it somewhere
worse.

---

## Adding your own backend

If your lab has a private endpoint, add a row to `BACKENDS` in
[`src/cs6961_agents/config.py`](../src/cs6961_agents/config.py):

```python
"mylab": BackendSpec(
    name="mylab",
    base_url="http://gpu-node-3.cs.utah.edu:8000/v1",
    api_key_env="MYLAB_API_KEY",   # or None if it does not authenticate
    default_model="Qwen/Qwen3-14B",
    is_free=True,
    is_local=True,
),
```

Then `LLM_BACKEND=mylab`. Nothing else in the repo needs to know it exists.

---

## Cost discipline, whichever backend you use

1. **Print the backend before every run.** The examples do this already.
2. **Cap `max_steps`.** A runaway agent is an expensive agent. 6–8 is plenty.
3. **Develop on one question**, not all twenty. `--limit 1`.
4. **Cache nothing you can recompute for free** — but do save the LLM outputs
   you already paid for.
5. **Watch your token counts.** `examples/00_raw_chat.py` prints them; every
   framework can. Prompt tokens grow with every agent step.

---

Next: [4 · Patching the course notebooks](04-patching-the-course.md)
