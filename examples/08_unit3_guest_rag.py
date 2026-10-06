from smolagents import CodeAgent

from cs6961_agents import describe_backend, get_smolagents_model
from unit3.retriever import GuestInfoRetrieverTool, load_guest_docs


def main() -> None:
    print(f"Backend: {describe_backend()}\n")

    # Step 1
    docs = load_guest_docs()
    print(f"Step 1: loaded {len(docs)} guests. First document:\n{docs[0].page_content}\n")

    # Step 2 — call the tool directly, no LLM involved, to show retrieval works
    guest_info_tool = GuestInfoRetrieverTool(docs)
    print("Step 2: guest_info_retriever('Ada Lovelace') ->")
    print(guest_info_tool("Ada Lovelace"), "\n")

    # Step 3
    alfred = CodeAgent(tools=[guest_info_tool], model=get_smolagents_model())

    response = alfred.run("Tell me about our guest named 'Lady Ada Lovelace'.")

    print("🎩 Alfred's Response:")
    print(response)


if __name__ == "__main__":
    main()
