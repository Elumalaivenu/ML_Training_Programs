# Example 4: OpenRouter API Example

from openai import OpenAI

client = OpenAI(
    api_key="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",
    base_url="https://openrouter.ai/api/v1"
)

prompt = "List three applications of Llama 3 in education."

response = client.chat.completions.create(
    model="meta-llama/llama-3-8b-instruct",
    messages=[
        {"role": "user", "content": prompt}
    ]
)

print(response.choices[0].message.content)
