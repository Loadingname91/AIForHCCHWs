"""
Example 1 — Unit 1, by hand: the ReAct loop with no agent library.

    python examples/01_unit1_manual_agent.py

Unit 1 of the course builds a "dummy agent" to show that an agent is not
magic: it is a while-loop around a chat model, a text format the model is
told to follow, and a dictionary of Python functions.

Everything smolagents/LangGraph/LlamaIndex do later is this, plus retries,
parsing, and ergonomics. Read this file once and the frameworks stop being
mysterious.
"""

import json
import re

from cs6961_agents import chat, describe_backend

# --------------------------------------------------------------------------
# 1. Tools: ordinary Python functions plus a description the model can read.
# --------------------------------------------------------------------------
def get_weather(city: str) -> str:
    """Pretend weather service, so the example needs no API key."""
    fake = {"salt lake city": "sunny, 24°C", "paris": "rainy, 14°C", "tokyo": "cloudy, 19°C"}
    return fake.get(city.strip().lower(), f"no data for {city}")


def calculate(expression: str) -> str:
    """Evaluate simple arithmetic. Restricted: an agent must never eval() freely."""
    if not re.fullmatch(r"[0-9+\-*/(). ]+", expression):
        return "error: only digits and + - * / ( ) are allowed"
    try:
        return str(eval(expression, {"__builtins__": {}}, {}))  # noqa: S307
    except Exception as exc:  # noqa: BLE001
        return f"error: {exc}"


TOOLS = {"get_weather": get_weather, "calculate": calculate}

TOOL_DOCS = """
- get_weather(city: str) -> str : current weather for a city
- calculate(expression: str) -> str : evaluate an arithmetic expression
"""

# --------------------------------------------------------------------------
# 2. The system prompt IS the agent framework. It defines a format the model
#    must follow, and we parse that format back out in Python.
# --------------------------------------------------------------------------
SYSTEM_PROMPT = f"""You are an agent that solves tasks step by step.

You have these tools:
{TOOL_DOCS}

Always answer with exactly one JSON object and nothing else.

To use a tool:
{{"thought": "why you need it", "action": "tool_name", "action_input": {{"arg": "value"}}}}

To give the final answer:
{{"thought": "why you are done", "final_answer": "the answer"}}

Use each tool's real parameter names, exactly as listed above. Example:
{{"thought": "I need Paris weather", "action": "get_weather", "action_input": {{"city": "Paris"}}}}
"""


def extract_json(text: str) -> dict | None:
    """Small models wrap JSON in prose or code fences. Dig it out anyway."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)  # Qwen3 reasoning
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    candidate = fenced.group(1) if fenced else None
    if candidate is None:
        brace = re.search(r"\{.*\}", text, re.DOTALL)
        candidate = brace.group(0) if brace else None
    if candidate is None:
        return None
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        return None


def run_agent(task: str, max_steps: int = 6) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": task},
    ]

    for step in range(1, max_steps + 1):
        # `chat()` is the plain OpenAI call plus this backend's quirks --
        # see cs6961_agents/backends.py, it is eight lines.
        response = chat(messages, temperature=0.1, max_tokens=800)
        raw = response.choices[0].message.content or ""
        parsed = extract_json(raw)

        if parsed is None:
            print(f"  [step {step}] unparseable output, asking again:\n    {raw[:200]}")
            messages.append({"role": "assistant", "content": raw})
            messages.append({
                "role": "user",
                "content": "That was not valid JSON. Reply with ONLY the JSON object.",
            })
            continue

        print(f"  [step {step}] thought: {parsed.get('thought', '')}")

        if "final_answer" in parsed:
            return str(parsed["final_answer"])

        name = parsed.get("action")
        args = parsed.get("action_input") or {}
        if name not in TOOLS:
            observation = f"error: unknown tool {name!r}. Available: {list(TOOLS)}"
        else:
            print(f"  [step {step}] action: {name}({args})")
            try:
                observation = TOOLS[name](**args)
            except TypeError as exc:
                observation = f"error: wrong arguments — {exc}"
        print(f"  [step {step}] observation: {observation}")

        # Feeding the observation back is the entire ReAct trick.
        messages.append({"role": "assistant", "content": json.dumps(parsed)})
        messages.append({"role": "user", "content": f"Observation: {observation}"})

    return "gave up: hit the step limit"


if __name__ == "__main__":
    print(f"Backend: {describe_backend()}\n")
    for task in [
        "What is the weather in Tokyo?",
        "If it is 19 degrees in Tokyo, what is that in Fahrenheit? Use the calculator.",
    ]:
        print(f"TASK: {task}")
        print(f"ANSWER: {run_agent(task)}\n")
