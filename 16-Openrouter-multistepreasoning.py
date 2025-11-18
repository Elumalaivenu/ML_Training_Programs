# Example 3: Multi-Step Reasoning Chain using OpenRouter

from openai import OpenAI

client = OpenAI(
    api_key="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",
    base_url="https://openrouter.ai/api/v1"
)

text = """
Artificial Intelligence is revolutionizing multiple industries.
In healthcare, AI assists with disease prediction and drug discovery.
In education, it enables personalized learning experiences.
In finance, it supports fraud detection and portfolio optimization.
"""

# Step 1: Summarize
summary = client.chat.completions.create(
    model="meta-llama/llama-3-8b-instruct",
    messages=[{"role": "user", "content": f"Summarize this text in 3 lines:\n{text}"}],
    max_tokens=120
).choices[0].message.content

# Step 2: Extract domains
domains = client.chat.completions.create(
    model="meta-llama/llama-3-8b-instruct",
    messages=[{"role": "user", "content": f"List the industries mentioned in this summary:\n{summary}"}],
    max_tokens=60
).choices[0].message.content

# Step 3: Create LinkedIn-style post
linkedin_post = client.chat.completions.create(
    model="meta-llama/llama-3-8b-instruct",
    messages=[{"role": "user", "content": f"Write a short LinkedIn post about how AI impacts these industries:\n{domains}"}],
    max_tokens=150
).choices[0].message.content

print("\n🧾 Summary:\n", summary)
print("\n🏭 Industries:\n", domains)
print("\n💼 LinkedIn Post:\n", linkedin_post)
