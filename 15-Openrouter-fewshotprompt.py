# Example 2: Few-Shot Prompting (Classification)

from openai import OpenAI

client = OpenAI(
    api_key="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",
    base_url="https://openrouter.ai/api/v1"
)

prompt = """
You are a sentiment classifier.
Classify the review as Positive, Negative, or Neutral.

Example 1:
Review: "This product works perfectly, I love it!"
Sentiment: Positive

Example 2:
Review: "It stopped working after two days, very disappointed."
Sentiment: Negative

Now classify this:
Review: "The design looks fine, but performance is average."
Sentiment:
"""

response = client.chat.completions.create(
    model="meta-llama/llama-3-8b-instruct",
    messages=[{"role": "user", "content": prompt}],
    max_tokens=100
)

print(response.choices[0].message.content)
