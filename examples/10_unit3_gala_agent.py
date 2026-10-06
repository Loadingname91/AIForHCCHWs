import sys

from smolagents import CodeAgent

from cs6961_agents import describe_backend, get_smolagents_model
from unit3.retriever import load_guest_dataset
from unit3.tools import DuckDuckGoSearchTool, HubStatsTool, WeatherInfoTool

EXAMPLES = {
    1: ("Finding Guest Information", "Tell me about 'Lady Ada Lovelace'"),
    2: ("Checking the Weather for Fireworks",
        "What's the weather like in Paris tonight? Will it be suitable for our fireworks display?"),
    3: ("Impressing AI Researchers",
        "One of our guests is from Qwen. What can you tell me about their most popular model?"),
    4: ("Combining Multiple Tools",
        "I need to speak with Dr. Nikola Tesla about recent advancements in wireless energy. "
        "Can you help me prepare for this conversation?"),
}


def build_alfred(model, tools) -> CodeAgent:
    return CodeAgent(
        tools=tools,
        model=model,
        add_base_tools=True,  # Add any additional base tools
        planning_interval=3,  # Enable planning every 3 steps
    )


def main() -> None:
    print(f"Backend: {describe_backend()}\n")
    only = int(sys.argv[1]) if len(sys.argv) > 1 else None

    model = get_smolagents_model()
    tools = [load_guest_dataset(), WeatherInfoTool(), HubStatsTool(), DuckDuckGoSearchTool()]
    alfred = build_alfred(model, tools)

    for n, (title, query) in EXAMPLES.items():
        if only not in (None, n):
            continue
        print(f"\n{'=' * 70}\nExample {n}: {title}\nQUERY: {query}\n{'=' * 70}")
        response = alfred.run(query)
        print("🎩 Alfred's Response:")
        print(response)

    if only not in (None, 5):
        return

    # Advanced feature: conversation memory
    print(f"\n{'=' * 70}\nExample 5: Conversation Memory\n{'=' * 70}")
    alfred_with_memory = build_alfred(model, tools)

    # First interaction
    response1 = alfred_with_memory.run("Tell me about Lady Ada Lovelace.")
    print("🎩 Alfred's First Response:")
    print(response1)

    # Second interaction (referencing the first)
    response2 = alfred_with_memory.run("What projects is she currently working on?", reset=False)
    print("🎩 Alfred's Second Response:")
    print(response2)


if __name__ == "__main__":
    main()
