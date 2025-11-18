# Example 4: Context-Aware Response using OpenRouter

from openai import OpenAI

client = OpenAI(
    api_key="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",
    base_url="https://openrouter.ai/api/v1"
)

context = """
Azure Document Intelligence helps extract text, tables, and structure from PDFs.
It supports prebuilt models for invoices, receipts, and identity documents.
Custom models can also be trained on your data to extract specific fields.
"""

question = "How can Azure Document Intelligence improve invoice automation in enterprises?"

prompt = f"""
Use the following context to answer the question.

Context:
{context}

Question:
{question}

Answer:
"""

response = client.chat.completions.create(
    model="meta-llama/llama-3-8b-instruct",
    messages=[{"role": "user", "content": prompt}],
    max_tokens=200
)

print(response.choices[0].message.content)
