"""
Example 2 — Unit 1 / Unit 2.1: smolagents on a local model.

    python examples/02_unit2_smolagents.py

What the course tells you to write:

    from smolagents import CodeAgent, InferenceClientModel
    model = InferenceClientModel(model_id="Qwen/Qwen2.5-Coder-32B-Instruct")
    #       ^^^^^^^^^^^^^^^^^^^^ metered: every step spends HF credits

What you write instead:

    from cs6961_agents import get_smolagents_model
    model = get_smolagents_model()

One line. The agent, the tools and the run loop are identical, because
smolagents only ever asks the model object for a chat completion.

Why CodeAgent and not ToolCallingAgent?
--------------------------------------
CodeAgent asks the model to write Python that calls tools; ToolCallingAgent
relies on the provider's native JSON tool-calling. Small local models are much
better at the former than the latter, so CodeAgent is the robust default when
you are running a 4B or 8B model on your laptop.
"""

from smolagents import CodeAgent, tool

from cs6961_agents import describe_backend, get_smolagents_model

# smolagents renamed its search tool; accept either.
try:
    from smolagents import WebSearchTool as SearchTool
except ImportError:  # older releases
    from smolagents import DuckDuckGoSearchTool as SearchTool


@tool
def celsius_to_fahrenheit(celsius: float) -> float:
    """Convert a temperature from Celsius to Fahrenheit.

    Args:
        celsius: Temperature in degrees Celsius.
    """
    return celsius * 9 / 5 + 32


def main() -> None:
    print(f"Backend: {describe_backend()}\n")

    model = get_smolagents_model()

    agent = CodeAgent(
        tools=[SearchTool(), celsius_to_fahrenheit],
        model=model,
        # Small models ramble; capping steps stops a runaway loop from eating
        # your evening. Raise it once you move to a 14B+ model.
        max_steps=6,
        # Whitelist only what the generated code may import.
        additional_authorized_imports=["math", "statistics", "datetime"],
        verbosity_level=2,
    )

    task = (
        "The average human body temperature is 37 degrees Celsius. "
        "Convert it to Fahrenheit and state the result in one sentence."
    )
    print(f"TASK: {task}\n")
    result = agent.run(task)

    print(f"\nFINAL ANSWER: {result}")


if __name__ == "__main__":
    main()
