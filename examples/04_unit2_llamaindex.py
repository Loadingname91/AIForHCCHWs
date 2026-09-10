"""
Example 4 — Unit 2.3: LlamaIndex on a local model, including local RAG.

    python examples/04_unit2_llamaindex.py

What the course tells you to write:

    from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI
    llm = HuggingFaceInferenceAPI(model_name="Qwen/Qwen2.5-Coder-32B-Instruct")
    #     ^^^^^^^^^^^^^^^^^^^^^^^ metered

What you write instead:

    from cs6961_agents import get_llamaindex_llm
    llm = get_llamaindex_llm()

Unit 2.3 also builds a RAG query engine. Its embedding model
(`HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")`) already runs
locally in the course's own code, which is a useful thing to notice: the
expensive half of RAG is the LLM, not the embeddings.
`get_llamaindex_embedding()` below is just that same local embedder, wrapped
so the model id lives in `.env` with everything else.
"""

import asyncio

from llama_index.core import Document, Settings, VectorStoreIndex
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.core.tools import FunctionTool, QueryEngineTool

from cs6961_agents import describe_backend, get_llamaindex_embedding, get_llamaindex_llm

# Course syllabus stub, standing in for whatever corpus you index.
COURSE_NOTES = [
    "CS6961 AI for Human-Centered Computing meets Tuesdays and Thursdays at 2pm "
    "in the Warnock Engineering Building.",
    "The CS6961 final project requires building an agent with tools and "
    "evaluating it on a benchmark.",
    "CS6961 grading: 40% assignments, 30% final project, 20% participation, 10% quizzes.",
]


def multiply(a: float, b: float) -> float:
    """Multiply two numbers together."""
    return a * b


async def main() -> None:
    print(f"Backend: {describe_backend()}\n")

    # Setting Settings.llm / Settings.embed_model once means every LlamaIndex
    # component picks up the local model, including ones you construct later.
    Settings.llm = get_llamaindex_llm()
    Settings.embed_model = get_llamaindex_embedding()

    print("Building a local vector index (first run downloads the embedder) ...")
    index = VectorStoreIndex.from_documents([Document(text=t) for t in COURSE_NOTES])

    rag_tool = QueryEngineTool.from_defaults(
        query_engine=index.as_query_engine(similarity_top_k=2),
        name="course_notes",
        description="Answers questions about CS6961: schedule, grading, and projects.",
    )

    agent = FunctionAgent(
        tools=[rag_tool, FunctionTool.from_defaults(fn=multiply)],
        llm=Settings.llm,
        system_prompt=(
            "You are a course assistant. Use the course_notes tool for anything "
            "about CS6961, and the multiply tool for arithmetic."
        ),
    )

    task = "What percentage of the CS6961 grade is the final project, and what is that times 3?"
    print(f"\nTASK: {task}\n")

    response = await agent.run(task)
    print(f"FINAL ANSWER: {response}")


if __name__ == "__main__":
    asyncio.run(main())
