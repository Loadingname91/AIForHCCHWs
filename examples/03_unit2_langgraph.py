"""
Example 3 — Unit 2.2: LangGraph on a local model.

    python examples/03_unit2_langgraph.py

What the course tells you to write:

    from langchain_openai import ChatOpenAI
    llm = ChatOpenAI(model="gpt-4o")
    #     ^^^^^^^^^^ the OpenAI API. Not metered credits -- an actual bill.
    #     This unit is the one that catches people out: it is not even using
    #     Hugging Face, so a free HF account does not help you here.

What you write instead:

    from cs6961_agents import get_langchain_chat
    chat = get_langchain_chat()

`ChatOpenAI` pointed at Ollama supports `.bind_tools()`, so every LangGraph
node, edge and prebuilt in Unit 2.2 keeps working unchanged.

This file builds the graph by hand rather than calling `create_react_agent`,
because Unit 2.2 is about seeing the state machine: nodes are functions,
edges are `if` statements, and the loop is the whole agent.
"""

from typing import Annotated, TypedDict

from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from cs6961_agents import describe_backend, get_langchain_chat


# --------------------------------------------------------------------------
# Tools
# --------------------------------------------------------------------------
@tool
def multiply(a: float, b: float) -> float:
    """Multiply two numbers together."""
    return a * b


@tool
def add(a: float, b: float) -> float:
    """Add two numbers together."""
    return a + b


@tool
def lookup_course(code: str) -> str:
    """Look up a university course by its code, e.g. 'CS6961'."""
    catalogue = {
        "CS6961": "AI for Human-Centered Computing — 3 credit hours.",
        "CS5340": "Natural Language Processing — 3 credit hours.",
    }
    return catalogue.get(code.upper().replace(" ", ""), f"No course named {code}.")


TOOLS = [multiply, add, lookup_course]


# --------------------------------------------------------------------------
# State: LangGraph passes this dict between nodes. `add_messages` appends
# rather than overwrites, which is what makes the conversation accumulate.
# --------------------------------------------------------------------------
class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


SYSTEM = SystemMessage(
    content=(
        "You are a helpful assistant with tools. Use a tool whenever it can "
        "answer part of the question. Do arithmetic with the tools, not in "
        "your head. When you have everything you need, answer directly."
    )
)


def build_graph():
    chat = get_langchain_chat().bind_tools(TOOLS)

    def assistant(state: AgentState) -> dict:
        """One LLM turn: it either answers, or asks for a tool."""
        return {"messages": [chat.invoke([SYSTEM] + state["messages"])]}

    builder = StateGraph(AgentState)
    builder.add_node("assistant", assistant)
    builder.add_node("tools", ToolNode(TOOLS))

    builder.add_edge(START, "assistant")
    # tools_condition routes to "tools" if the last message has tool_calls,
    # otherwise to END. That single branch is the entire agent loop.
    builder.add_conditional_edges("assistant", tools_condition, {"tools": "tools", END: END})
    builder.add_edge("tools", "assistant")

    return builder.compile()


def main() -> None:
    print(f"Backend: {describe_backend()}\n")

    graph = build_graph()
    task = "How many credit hours is CS6961, and what is that times 4 semesters?"
    print(f"TASK: {task}\n")

    state = graph.invoke({"messages": [HumanMessage(content=task)]})

    for message in state["messages"]:
        message.pretty_print()


if __name__ == "__main__":
    main()
