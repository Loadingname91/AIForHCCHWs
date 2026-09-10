"""
Example 5 — Unit 4: a GAIA-style agent you can actually afford to iterate on.

    python examples/05_unit4_gaia_agent.py
    python examples/05_unit4_gaia_agent.py "your own question here"

Unit 4 of the course puts an agent up against a 20-question subset of GAIA.
You will run it dozens of times while debugging. On a metered backend that is
exactly how people burn their credits in an afternoon; locally it is free.

Two things matter:

1. EXACT MATCH. The grader compares strings. "4" passes, "The answer is 4"
   fails. Hence `finalize()` below — never skip that step.
2. The questions need real tools (web search, arithmetic, file reading), so
   this is a genuine agent, not a chat wrapper.

Strategy that works: develop locally on qwen3:8b until the loop is right,
then, if you are curious what a much bigger model does differently, flip
LLM_BACKEND=groq for one run. Same code either way.

Use examples/06_unit4_local_eval.py to run this over the whole question set.
"""

import re
import sys

from smolagents import CodeAgent, PythonInterpreterTool, tool

from cs6961_agents import describe_backend, get_smolagents_model

try:
    from smolagents import WebSearchTool as SearchTool
except ImportError:
    from smolagents import DuckDuckGoSearchTool as SearchTool


@tool
def read_text_file(path: str) -> str:
    """Read a local UTF-8 text file and return its contents.

    Args:
        path: Path to the file on disk.
    """
    try:
        with open(path, encoding="utf-8") as handle:
            return handle.read()[:20_000]
    except OSError as exc:
        return f"error: {exc}"


GAIA_SYSTEM_SUFFIX = """
You are answering benchmark questions that are graded by EXACT STRING MATCH.

Rules for your final answer:
- Give the answer and nothing else. No "The answer is", no units unless the
  question asks for units, no trailing period.
- Numbers: digits only, no thousands separators (write 1234, not 1,234).
- Lists: comma-separated, no "and".
- If you truly cannot determine it, answer with your single best guess anyway.
"""


def finalize(raw: str) -> str:
    """Strip everything that is not the answer itself.

    Two sources of noise, both of which silently cost you the point:

    1. The model, which adds "The answer is ...", quotes, trailing periods,
       and (on hybrid Qwen3 tags) a <think> block.
    2. smolagents, which returns its execution log -- "Stdout: ... Output: 42"
       -- whenever the agent finishes by evaluating an expression instead of
       calling final_answer().
    """
    text = str(raw).strip()

    # 1. reasoning traces
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    # 2. smolagents execution logs: keep only what follows the last "Output:"
    if "Output:" in text:
        text = text.rsplit("Output:", 1)[1].strip()
    text = re.sub(r"^Stdout:\s*", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(
        r"^Last output from code snippet:\s*", "", text, flags=re.IGNORECASE
    ).strip()

    # 3. conversational preambles
    text = re.sub(
        r"^(final answer|answer|the answer is|the final answer is)\s*[:\-]?\s*",
        "", text, flags=re.IGNORECASE,
    ).strip()

    # 4. wrapping punctuation
    text = text.strip().strip('"').strip("'").rstrip(".")

    # 5. thousands separators inside a bare number: GAIA wants 1234, not 1,234
    if re.fullmatch(r"-?[\d,]+(\.\d+)?", text):
        text = text.replace(",", "")

    return text.strip()


def build_agent() -> CodeAgent:
    agent = CodeAgent(
        tools=[SearchTool(), PythonInterpreterTool(), read_text_file],
        model=get_smolagents_model(),
        max_steps=8,
        additional_authorized_imports=["math", "statistics", "datetime", "re", "json"],
        verbosity_level=1,
    )
    agent.prompt_templates["system_prompt"] += GAIA_SYSTEM_SUFFIX
    return agent


def answer(question: str) -> str:
    return finalize(build_agent().run(question))


if __name__ == "__main__":
    print(f"Backend: {describe_backend()}\n")

    questions = sys.argv[1:] or [
        "How many continents are there on Earth?",
        "What is 17 * 23 + 4?",
    ]
    for question in questions:
        print(f"\nQ: {question}")
        print(f"A: {answer(question)}")
