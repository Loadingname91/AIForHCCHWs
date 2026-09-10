"""
Example 0 — proof of life, with no framework in the way.

    python examples/00_raw_chat.py

Before you debug an agent, prove the LLM answers at all. This is the same
`openai` SDK the course uses; only `base_url` points somewhere free.
Read the printed base_url: that URL is the whole difference between
"free and offline" and "billed to your Hugging Face credits".
"""

from cs6961_agents import describe_backend, get_openai_client
from cs6961_agents.config import get_backend

spec = get_backend()
print(f"Backend: {describe_backend()}\n")

client = get_openai_client()

response = client.chat.completions.create(
    model=spec.default_model,
    messages=[
        {"role": "system", "content": "You are a concise teaching assistant."},
        {"role": "user", "content": "In two sentences: what is an LLM agent?"},
    ],
    temperature=0.2,
    max_tokens=300,
    # Per-backend request tweaks. For Ollama this carries
    # {"reasoning_effort": "none"}, which stops hybrid Qwen3 tags from
    # spending the whole token budget thinking and returning empty content.
    # `cs6961_agents.chat()` applies this for you; here it is spelled out.
    extra_body=spec.extra_body or None,
)

print(response.choices[0].message.content)

usage = response.usage
if usage:
    print(
        f"\n[tokens] prompt={usage.prompt_tokens} "
        f"completion={usage.completion_tokens} total={usage.total_tokens}"
    )
    if spec.is_local:
        print("[cost]   $0.00 — this ran on your own machine.")
