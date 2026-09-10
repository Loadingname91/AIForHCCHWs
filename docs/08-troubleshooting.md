# 8 · Troubleshooting

**Run this first, and paste its output when you ask for help:**

```bash
python scripts/doctor.py
```

It checks Python → packages → `.env` → server → chat → tool calling, in that
order. **Fix the first `FAIL`;** the ones after it are usually just fallout.

---

## Environment

### `ModuleNotFoundError: No module named 'cs6961_agents'`

Nine times out of ten you forgot to activate the environment. Look at your
prompt for `(cs6961-agents)`.

```bash
conda activate cs6961-agents
pip install -e .            # only if activating did not fix it
```

### `ModuleNotFoundError: No module named 'smolagents'` (or langgraph, …)

Same cause, or the install did not finish.

```bash
conda activate cs6961-agents
pip install -r requirements.txt
```

### The wrong Python is running

```bash
which python && python -V
# expect: .../envs/cs6961-agents/bin/python and 3.12.x
```

In VS Code or Jupyter, the *kernel* is separate from your terminal. Select the
`cs6961-agents` interpreter explicitly, or:

```bash
python -m ipykernel install --user --name cs6961-agents
```

---

## Connecting to the model

### `APIConnectionError` / `Connection refused` on localhost

The server is not running.

```bash
curl http://localhost:11434/api/tags     # should return JSON
ollama serve                             # if it does not
```

On Linux with systemd: `sudo systemctl status ollama`, then
`sudo systemctl start ollama`.

### `404 model 'xyz' not found`

The name in `.env` does not match anything you have pulled.

```bash
ollama list                       # what you actually have
ollama pull qwen3:8b              # get the one you want
```

Names must match exactly, tag included: `qwen3:8b` ≠ `qwen3-8b` ≠ `qwen3`.

### `401 Unauthorized` / `Invalid API key`

For a cloud backend, the key is missing, wrong, or has a stray space or
newline. Check `.env`, and check you edited `.env` and not `.env.example`.

### `429 Too Many Requests`

Free-tier rate limit. Wait, use a smaller model, or switch back to
`LLM_BACKEND=ollama`.

### `402 Payment Required`

Your Hugging Face Inference credits for the month are gone. Nothing is broken.

```ini
LLM_BACKEND=ollama
```

---

## The agent behaves badly

### It "forgets" its task, or loops on the same tool call

**This is almost always the context window, not the model.** Ollama picks a
window from your VRAM — often 4k — and truncates longer prompts *silently*.
Agent prompts pass 4k after two or three steps.

```bash
ollama ps      # look at the CONTEXT column while a model is loaded
```

If it says 4096, fix it server-side and restart Ollama — full instructions in
[2 · The context-window trap](02-local-models.md#the-context-window-trap-read-this-one).

```bash
OLLAMA_CONTEXT_LENGTH=32768 ollama serve
```

### It answers in prose instead of calling a tool

- Your model is too small or not tool-tuned. Move up: `qwen3:4b-instruct` →
  `qwen3:8b` → `qwen3:14b`.
- Use smolagents' `CodeAgent` rather than `ToolCallingAgent`. Small models are
  much better at writing a line of Python than at filling a JSON schema.
- Drop the temperature: `LLM_TEMPERATURE=0.1`.
- Check `doctor.py` section 6 — it tests exactly this.

### The model returns an empty answer

`content` is `""` and `finish_reason` is `length`. You are on a hybrid Qwen3
tag: it spent the whole token budget reasoning, and Ollama puts that reasoning
in a separate `reasoning` field rather than in `content`.

```ini
OLLAMA_DISABLE_THINKING=true       # the default; sends reasoning_effort="none"
OLLAMA_MODEL_ID=qwen3:4b-instruct  # or just use an instruct-only tag
```

Note that `{"think": false}` and a `/no_think` suffix **do not work** on the
`/v1` endpoint, despite what you will read in forum posts. Full explanation:
[2 · Thinking models](02-local-models.md#thinking-models).

If you call the SDK directly, use `cs6961_agents.chat()` rather than a bare
client — it applies this for you.

### Output has `<think>…</think>` in it, and parsing breaks

Same cause, text-shaped. Switch to an instruct-only tag, or strip the block —
`extract_json()` in [`examples/01_unit1_manual_agent.py`](../examples/01_unit1_manual_agent.py)
and `finalize()` in [`examples/05_unit4_gaia_agent.py`](../examples/05_unit4_gaia_agent.py)
both show how.

### `Error in code parsing` (smolagents)

The model did not produce a clean code block. Try, in order: a bigger model;
`temperature=0.1`; fewer tools (each one adds schema noise); and check the
context window above.

### It is extremely slow

```bash
ollama ps      # PROCESSOR column: "100% GPU" is what you want
```

`100% CPU` on a machine with a GPU means the model did not fit in VRAM. Use a
smaller model or a smaller context. On a genuinely CPU-only machine, see
[7 · No GPU](07-no-gpu-and-colab.md).

### Out of memory / the model will not load

Use a smaller model, lower `OLLAMA_CONTEXT_LENGTH` (32768 → 16384 → 8192), and
unload anything stale: `ollama stop <model>`.

---

## Framework-specific

### LangGraph: `bind_tools` seems to do nothing

The server is not returning native tool calls. With vLLM you must start it with
`--enable-auto-tool-choice --tool-call-parser hermes`. With Ollama, check the
model advertises tool support: `ollama show qwen3:8b` should list `tools` under
Capabilities.

### LlamaIndex: "this LLM does not support function calling"

Use `get_llamaindex_llm()`, which sets `is_function_calling_model=True`. If you
built `OpenAILike` by hand, set it yourself.

### smolagents: `InterpreterError: Import of X is not allowed`

That is the sandbox doing its job. Add the module deliberately:

```python
CodeAgent(..., additional_authorized_imports=["math", "statistics"])
```

Think before adding `os`, `subprocess`, or `shutil` — see
[6 · Cost and safety](06-cost-and-safety.md).

### Web search fails or returns nothing

DuckDuckGo rate-limits. Wait a minute, or install/upgrade the client:
`pip install -U ddgs`.

---

## Unit 4 / GAIA

### `GET /files/{task_id}` returns 404

Known upstream: the attachment endpoint answers
`{"detail": "No file path associated with task_id ..."}` for all five questions
that reference a file. Catch it and move on — do not let one failing download
end a run of twenty. See [5 · The attachments](05-unit4-gaia-agent.md#the-attachments).

### The answers look right but are full of prose

GAIA is graded by exact string match, and `The answer is 4.` does not match
`4`. Use `finalize()` from
[`examples/05_unit4_gaia_agent.py`](../examples/05_unit4_gaia_agent.py), and
print exactly what your agent produced.

### `agent.run()` returns "Stdout: ... Output: 42"

That is smolagents' execution log, returned when the agent finishes by
evaluating an expression instead of calling `final_answer()`. Tell it to call
`final_answer()` in the system prompt; `finalize()` strips it as a backstop.

---

## Still stuck

Include all of this when you ask:

1. the full output of `python scripts/doctor.py`,
2. the complete traceback, not just the last line,
3. `ollama list` and `ollama ps`,
4. what you changed since it last worked.

That is usually enough for someone to answer in one message instead of five.
