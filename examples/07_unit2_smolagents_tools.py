"""
Example 7 — Unit 2.1: building custom smolagents tools.

    python examples/07_unit2_smolagents_tools.py

Adapted from the HF Agents Course "Tools" page, which runs everything
through `InferenceClientModel` and a paid Hub call:

    model = InferenceClientModel(...)   # course
    model = get_smolagents_model()      # this repo, reads .env

Shows both ways smolagents lets you define a tool:
  - `@tool` — a plain function with type hints and a docstring
  - `Tool` subclass — for anything that needs setup/state beyond one function
"""

from smolagents import CodeAgent, Tool, tool

from cs6961_agents import describe_backend, get_smolagents_model


@tool
def catering_service_tool(query: str) -> str:
    """
    Looks up the highest-rated catering service for a given query.

    Args:
        query: What kind of catering to search for.
    """
    services = {
        "Gotham Catering Co.": 4.9,
        "Wayne Manor Catering": 4.8,
        "Gotham City Events": 4.7,
    }
    return max(services, key=services.get)


class SuperheroPartyThemeTool(Tool):
    name = "superhero_party_theme_generator"
    description = (
        "Suggests a superhero-themed party idea for a given category. "
        "Returns a short description of the theme."
    )
    inputs = {
        "category": {
            "type": "string",
            "description": "'classic heroes', 'villain masquerade', or 'futuristic gotham'",
        }
    }
    output_type = "string"

    def forward(self, category: str) -> str:
        themes = {
            "classic heroes": "Justice League Gala, guests dressed as their favorite DC heroes.",
            "villain masquerade": "Gotham Rogues' Ball, a masquerade of classic Batman villains.",
            "futuristic gotham": "Neo-Gotham Night, a cyberpunk party inspired by Batman Beyond.",
        }
        return themes.get(
            category.lower(),
            "Unknown category. Try 'classic heroes', 'villain masquerade', or 'futuristic gotham'.",
        )


def main() -> None:
    print(f"Backend: {describe_backend()}\n")

    agent = CodeAgent(
        tools=[catering_service_tool, SuperheroPartyThemeTool()],
        model=get_smolagents_model(),
    )

    task = (
        "Find the highest-rated catering service, then suggest a superhero "
        "party theme in the 'villain masquerade' category. Call final_answer "
        "with one sentence naming both."
    )
    print(f"TASK: {task}\n")

    result = agent.run(task)
    print(f"\nRESULT: {result}")


if __name__ == "__main__":
    main()
