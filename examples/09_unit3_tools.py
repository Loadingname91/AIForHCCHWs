from smolagents import CodeAgent

from cs6961_agents import describe_backend, get_smolagents_model
from unit3.tools import DuckDuckGoSearchTool, HubStatsTool, WeatherInfoTool


def main() -> None:
    print(f"Backend: {describe_backend()}\n")

    search_tool = DuckDuckGoSearchTool()
    weather_info_tool = WeatherInfoTool()
    hub_stats_tool = HubStatsTool()

    print("--- search_tool('Who's the current President of France?')")
    print(search_tool("Who's the current President of France?"), "\n")

    print("--- weather_info_tool('Paris')")
    print(weather_info_tool("Paris"), "\n")

    print("--- hub_stats_tool('facebook')")
    print(hub_stats_tool("facebook"), "\n")

    alfred = CodeAgent(
        tools=[search_tool, weather_info_tool, hub_stats_tool],
        model=get_smolagents_model(),
    )

    response = alfred.run("What is Facebook and what's their most popular model?")

    print("🎩 Alfred's Response:")
    print(response)


if __name__ == "__main__":
    main()
