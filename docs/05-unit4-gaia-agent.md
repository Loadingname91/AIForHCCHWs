# 5 · Unit 4 · Building and testing a GAIA agent

Unit 4 of the course is where everything comes together: an agent with real
tools, answering questions from
[GAIA](https://huggingface.co/gaia-benchmark) — a benchmark deliberately built
so that a chat model with no tools scores near zero.

> **On submission.** The course's own Unit 4 flow asks you to publish a Gradio
> Space and post answers to a scoring API for its student leaderboard. **This
> repo does not cover that, and CS6961 does not ask you to do it.** Your
> coursework is submitted through Canvas, and the assignment page there is the
> authority on what to hand in. What follows is about building the agent and
> testing it locally.

It is also where cost would have hurt most, because getting an agent right
means running it over and over. Locally, that is free.

---

## Running the benchmark locally

The 20 questions are publicly readable, so you can use them as a practice set
without submitting anything:

```bash
python examples/06_unit4_local_eval.py --list            # read the questions
python examples/06_unit4_local_eval.py --run --limit 3   # try three
python examples/06_unit4_local_eval.py --run             # try all twenty
```

It writes `gaia_report.json` with every answer, every error, and how long each
one took.

**There is no published answer key**, so nothing can tell you your score. That
turns out to matter less than you would think. What the report *does* tell you
is where the agent breaks:

- questions where it crashed,
- questions where it returned nothing,
- questions where it answered in a paragraph instead of a value,
- how much wall-clock each question cost you.

Those are the fixable failures, and reading twenty traces to find them teaches
far more than a single number would.

---

## What the agent needs

[`examples/05_unit4_gaia_agent.py`](../examples/05_unit4_gaia_agent.py) is a
working starting point:

```python
agent = CodeAgent(
    tools=[SearchTool(), PythonInterpreterTool(), read_text_file],
    model=get_smolagents_model(),
    max_steps=8,
    additional_authorized_imports=["math", "statistics", "datetime", "re", "json"],
)
```

Three things carry most of the weight:

1. **Real tools.** GAIA questions need web search, arithmetic, and file
   reading. Without them you are just asking a small model trivia.
2. **A step cap.** `max_steps=8`. Past that a small model is looping, not
   thinking — and you are burning minutes per question.
3. **Never crashing the run.** Catch exceptions per question. One bad question
   must not cost you the other nineteen.

---

## Answer formatting matters more than you expect

GAIA is graded by **exact string match**. This is worth understanding even
though you are not submitting anywhere, because it is a general lesson about
getting structured output out of a language model.

| Produced | Wanted | Result |
|---|---|---|
| `4` | `4` | ✅ |
| `The answer is 4.` | `4` | ❌ |
| `4 albums` | `4` | ❌ |
| `1,234` | `1234` | ❌ |
| `Paris.` | `Paris` | ❌ |

Two defences, and you want both, because instructing a model is not the same
as controlling it:

1. **Tell the model.** `GAIA_SYSTEM_SUFFIX` in
   [`examples/05_unit4_gaia_agent.py`](../examples/05_unit4_gaia_agent.py)
   appends explicit formatting rules to smolagents' system prompt.
2. **Do not trust it.** `finalize()` in the same file strips `<think>` blocks,
   leading "The answer is", surrounding quotes, trailing periods, and thousands
   separators inside bare numbers.

There is a third source of noise that is easy to miss: **smolagents itself.**
If the agent finishes by evaluating an expression rather than calling
`final_answer()`, `agent.run()` returns its execution log:

```
Stdout:

Output: 395
```

`finalize()` handles that too, but the better fix is a system prompt that tells
the agent to call `final_answer()`. Print exactly what you produced, every
time.

---

## The attachments

Five of the twenty questions reference a file (`.png`, `.mp3`, `.py`,
`.xlsx`). The endpoint that would serve them,
`GET https://agents-course-unit4-scoring.hf.space/files/{task_id}`, currently
answers `404 {"detail": "No file path associated with task_id ..."}` for all
five, so those questions cannot be solved from the attachment.

Handle the 404 rather than crashing on it, and spend your effort on the other
fifteen.

---

## What to expect from a local model

| Setup | Roughly how many of 20 it handles sensibly |
|---|---|
| `qwen3:4b-instruct`, search + python | 4–6 |
| `qwen3:8b`, search + python | 6–9 |
| `qwen3:14b` / `qwen3-coder:30b` | 8–11 |
| a large cloud model, well-tooled | 10–14 |

These are impressions from a small local sample, not measurements — there is
no answer key to measure against. Take them as an order of magnitude. The
useful point is that a local 8B model gets meaningfully far, so nothing here
requires you to pay for inference.

If you want one run on something bigger, flip `LLM_BACKEND=groq` in `.env` and
run again. The code does not change. See [3 · Backends](03-backends.md).

---

## If you want the Hugging Face certificate

Separate from CS6961, the course offers a free certificate for completing its
own Unit 4 flow, which does require publishing a Space and posting to the
scoring API. If you go after it on your own, one thing will bite you:

> **A Hugging Face Space cannot reach your laptop's `localhost`.**

The Space runs on Hugging Face's servers; your Ollama runs on your machine, and
`http://localhost:11434` inside the Space means the Space's own localhost. Give
the Space a free-tier cloud key instead (`LLM_BACKEND=groq` plus `GROQ_API_KEY`
as Space secrets) and keep Ollama for local development. Because the backend
lives in the environment rather than in the code, the same agent runs in both
places.

---

Next: [6 · Cost and safety](06-cost-and-safety.md)
