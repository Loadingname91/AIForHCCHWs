# 4 · Patching the course notebooks

If you are following the course at <https://huggingface.co/learn/agents-course>
as written, this page is the diff: for each unit, the one line you would
replace to point it at a local model instead.

The pattern is always the same.

```python
# 1. add this import at the top of the notebook
from cs6961_agents import get_smolagents_model   # or get_langchain_chat / get_llamaindex_llm

# 2. replace the course's model line with a call to it
model = get_smolagents_model()
```

Nothing else changes. Not the tools, not the agent class, not `.run()`.

> **Running the course notebooks in Colab?** They will not find
> `cs6961_agents`. See [7 · No GPU, weak laptop, or Colab](07-no-gpu-and-colab.md#running-the-course-notebooks-in-colab).

---

## Unit 1 — Agent fundamentals

The course's `dummy_agent_library` notebook calls `InferenceClient` directly to
show the raw message/special-token layer. Point it at Ollama:

```python
# from huggingface_hub import InferenceClient
# client = InferenceClient("Qwen/Qwen2.5-Coder-32B-Instruct")

from cs6961_agents import get_openai_client
from cs6961_agents.config import get_backend

client = get_openai_client()
MODEL = get_backend().default_model
```

Then the calls become ordinary OpenAI-SDK calls:

```python
output = client.chat.completions.create(
    model=MODEL,
    messages=[{"role": "user", "content": "The capital of France is"}],
    max_tokens=100,
)
print(output.choices[0].message.content)
```

For the agent template at the end of Unit 1, use
[`examples/01_unit1_manual_agent.py`](../examples/01_unit1_manual_agent.py).
It is a complete, working, framework-free ReAct loop against your local model,
and it is the best 130 lines in this repo to actually read.

**A caveat worth understanding.** Unit 1 spends time on `chat_template`,
`<|im_start|>`, and other special tokens. Ollama's `/v1` endpoint applies the
chat template for you, so those details become invisible. That is convenient
but it hides the lesson. To see the raw prompt the way the course intends:

```python
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")   # tokenizer only, no weights
print(tok.apply_chat_template(
    [{"role": "user", "content": "hi"}], tokenize=False, add_generation_prompt=True
))
```

That downloads a few hundred kilobytes and costs nothing.

---

## Unit 2.1 — smolagents

```python
# from smolagents import CodeAgent, InferenceClientModel
# model = InferenceClientModel(model_id="Qwen/Qwen2.5-Coder-32B-Instruct")

from smolagents import CodeAgent
from cs6961_agents import get_smolagents_model
model = get_smolagents_model()

agent = CodeAgent(tools=[...], model=model, max_steps=6)
```

Working example: [`examples/02_unit2_smolagents.py`](../examples/02_unit2_smolagents.py).

### Prefer `CodeAgent` over `ToolCallingAgent` locally

`ToolCallingAgent` depends on the server returning well-formed native
`tool_calls` JSON. `CodeAgent` asks the model to write Python that calls the
tools, and smolagents parses and sandboxes that code. Small local models are
noticeably better at writing a line of Python than at filling a JSON schema, so
`CodeAgent` is the robust default at 4B–8B. If you specifically want to study
`ToolCallingAgent`, move up to `qwen3:14b` or `qwen3-coder:30b` first.

### Settings that matter on a local model

```python
agent = CodeAgent(
    tools=[...],
    model=get_smolagents_model(),
    max_steps=6,                       # a runaway loop wastes your evening
    additional_authorized_imports=["math", "statistics", "datetime"],
    verbosity_level=2,                 # you want to see every step while learning
)
```

### If you need LiteLLM instead

Some smolagents features and some providers behave better through LiteLLM. Note
the `ollama_chat/` prefix and that it wants the URL *without* `/v1`:

```python
from cs6961_agents import get_smolagents_litellm_model
model = get_smolagents_litellm_model()   # handles both quirks for you
```

---

## Unit 2.2 — LangGraph

**This is the unit that catches people out.** It does not use Hugging Face at
all — it calls `ChatOpenAI(model="gpt-4o")`, so a free HF account does not help
and there is no free tier to fall back on.

```python
# from langchain_openai import ChatOpenAI
# llm = ChatOpenAI(model="gpt-4o")

from cs6961_agents import get_langchain_chat
llm = get_langchain_chat()

llm_with_tools = llm.bind_tools(tools, parallel_tool_calls=False)
```

`get_langchain_chat()` returns a real `ChatOpenAI` object, just aimed at your
local server, so `.bind_tools()`, `ToolNode`, `tools_condition`, and every
LangGraph prebuilt keep working unchanged.

Keep `parallel_tool_calls=False`. Small models handle one tool call per turn far
more reliably than several.

Working example: [`examples/03_unit2_langgraph.py`](../examples/03_unit2_langgraph.py).

### The vision part of that unit

The document-analysis agent also does `vision_llm = ChatOpenAI(model="gpt-4o")`
to read an image. For that you need a *vision* model, and text-only Qwen3 will
not do. Two options:

```bash
ollama pull qwen3-vl:4b        # ~4 GB, fine for reading a document image
ollama pull qwen3-vl:8b        # better, ~7 GB
ollama pull qwen2.5vl:7b       # solid alternative
```

```python
vision_llm = get_langchain_chat(model="qwen3-vl:8b")   # same server, different model
```

If your machine cannot host a vision model, skip that section. It is a
demonstration of multimodal input, not a prerequisite for anything later.

### When you need Ollama-specific options

`num_ctx` has no field in the OpenAI protocol. To set it per-request rather
than server-wide:

```python
from cs6961_agents import get_langchain_ollama_chat
llm = get_langchain_ollama_chat(num_ctx=32768)
```

---

## Unit 2.3 — LlamaIndex

```python
# from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI
# llm = HuggingFaceInferenceAPI(model_name="Qwen/Qwen2.5-Coder-32B-Instruct")

from cs6961_agents import get_llamaindex_llm
llm = get_llamaindex_llm()
```

Or set it globally, so every component built later inherits it:

```python
from llama_index.core import Settings
from cs6961_agents import get_llamaindex_embedding, get_llamaindex_llm

Settings.llm = get_llamaindex_llm()
Settings.embed_model = get_llamaindex_embedding()
```

The embedding model in this unit (`BAAI/bge-small-en-v1.5`) already runs
locally in the course's own code — worth noticing that the expensive half of
RAG is the LLM, not the embeddings.

`get_llamaindex_llm()` sets `is_function_calling_model=True`. Without it,
LlamaIndex quietly refuses to build a `FunctionAgent` and downgrades you to a
text-parsing ReAct agent, which then fails in confusing ways.

Working example: [`examples/04_unit2_llamaindex.py`](../examples/04_unit2_llamaindex.py),
including a small local RAG index.

---

## Unit 3 — Use cases

Unit 3 composes what you already have. Same substitution, no new cases.

---

## Unit 4 — GAIA agent

Enough of its own detail to need a page:
[5 · Building and testing a GAIA agent](05-unit4-gaia-agent.md).

Note that CS6961 does not use the course's leaderboard submission flow — your
coursework goes through Canvas, and the assignment page there says what to hand
in. Unit 4 here is about building the agent.

---

## Bonus units

- **Bonus 1 (fine-tuning for function calling)** needs a GPU. Use Colab's free
  T4 — see [7](07-no-gpu-and-colab.md).
- **Bonus 2 (observability)** works unchanged; Langfuse and Phoenix both have
  free self-hosted or free-tier options, and both trace local models fine.
- **Bonus 3 (Pokémon)** works with a local model, slowly.

---

Next: [5 · The final assignment](05-unit4-gaia-agent.md)
