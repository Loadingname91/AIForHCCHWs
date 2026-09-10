# 0 · Why this repo exists

## The problem, stated plainly

The [Hugging Face Agents Course](https://huggingface.co/learn/agents-course/unit0/introduction)
is free. Running the code in it is not.

Every unit builds an agent by calling a model that lives on someone else's GPU:

| Unit | What the course tells you to write | What it hits |
|---|---|---|
| 1, 2.1 | `InferenceClientModel(model_id="Qwen/Qwen2.5-Coder-32B-Instruct")` | HF Inference Providers |
| 2.2 | `ChatOpenAI(model="gpt-4o")` | the OpenAI API — **real money** |
| 2.3 | `HuggingFaceInferenceAPI(model_name="Qwen/Qwen2.5-Coder-32B-Instruct")` | HF Inference Providers |
| 4 | the same, ×20 GAIA questions, ×N debugging runs | HF Inference Providers |

Unit 2.2 is the sharpest version of the problem: it does not use Hugging Face
at all, it calls `gpt-4o` on the OpenAI API, which has no free tier whatsoever.

For the units that do use Hugging Face: a free account comes with roughly
**$0.10 of monthly Inference Provider credits**. PRO ($9/month) raises that. Either way the number is small,
and an *agent* is unusually good at spending it: one agent run is not one API
call, it is one call per reasoning step, each carrying the full system prompt,
every tool schema, and the entire history of previous steps. A single Unit 4
debugging session — 20 questions × 6 steps × a growing prompt — can exhaust a
month of free credits in an afternoon.

The failure mode students report is always the same:

```
HfHubHTTPError: 402 Client Error: Payment Required
You have exceeded your monthly included credits for Inference Providers.
```

## The fix

Run the model yourself. A 4B–8B Qwen3 model on a laptop is good enough for
Units 1–3, costs nothing, works on a plane, and has no rate limit. Save the
cloud for a final run on a bigger model, if you want one at all.

## The one idea that makes it easy

You do not need to learn a different SDK per provider. Ollama, LM Studio,
vLLM, llama.cpp, Groq, Cerebras, Gemini, OpenRouter, and Hugging Face's own
router **all speak the same OpenAI-compatible `/v1/chat/completions` protocol.**

So a "backend" is just three values:

```
base_url  +  api_key  +  model_name
```

This repo puts those three in `.env` and gives you one function per framework
that reads them. Switching from your laptop to Groq to Hugging Face is a
one-line edit, and none of your agent code changes:

```python
# The course:
from smolagents import CodeAgent, InferenceClientModel
model = InferenceClientModel(model_id="Qwen/Qwen2.5-Coder-32B-Instruct")

# This repo:
from cs6961_agents import get_smolagents_model
model = get_smolagents_model()
```

Everything after that line — `CodeAgent(tools=..., model=model)`, `agent.run(...)` —
is untouched. That is worth internalizing well beyond this course: provider
independence is a two-line abstraction, and it is the difference between a
project you can afford to iterate on and one you cannot.

## Choosing a backend

| Backend | Cost | Speed | Offline | Set-up | Use it for |
|---|---|---|---|---|---|
| **Ollama** (local) | free | medium | yes | 5 min | **default — all of Units 1–3** |
| LM Studio (local) | free | medium | yes | 10 min, GUI | same, if you prefer clicking to typing |
| vLLM (local) | free | fast | yes | 30 min, needs NVIDIA GPU | lab machines, batch runs |
| llama.cpp (local) | free | slow–medium | yes | 30 min | best CPU-only option |
| Groq | free tier | very fast | no | 2 min | when your laptop is too slow |
| Cerebras | free tier | very fast | no | 2 min | same |
| Gemini | free tier | fast | no | 2 min | same |
| OpenRouter | free models | varies | no | 2 min | trying many models |
| **HF Inference** | **metered** | fast | no | 0 min | a final run on a big model |
| OpenAI | paid | fast | no | 0 min | never required here |

One path that works well — take it, adapt it, or ignore it:

1. **Units 0–3:** Ollama, locally. Free, unlimited, and you learn more because
   you can afford to run things repeatedly.
2. **Unit 4 development:** still Ollama. This is where credits die otherwise.
3. **A final comparison run:** optionally flip to Groq or HF once, to see what
   a much larger model does differently. Same code, one line in `.env`.

## A Hugging Face account is still useful

A local LLM does not remove Hugging Face from the picture. A free account and
an `HF_TOKEN` still help with:

- downloading tokenizers, datasets, and embedding models, which are
  rate-limited without a token,
- reading the course material and its notebooks on the Hub.

Neither costs credits. Only *inference* does.

---

Next: [1 · Setting up the environment](01-setup-environment.md)
