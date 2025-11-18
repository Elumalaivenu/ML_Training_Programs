# Example 2: Prompt Engineering Demo

import os
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables from .env file
load_dotenv()

# Initialize client with API key from environment
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY not found in environment variables. Please check your .env file.")

client = OpenAI(api_key=api_key)

prompts = [
    "Write a short poem about rain.",
    "Explain the water cycle as if talking to a 5-year-old.",
    "List three pros and cons of Artificial Intelligence."
]

for p in prompts:
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": p}]
    )
    print("\n📝 Prompt:", p)
    print("🤖 Response:", response.choices[0].message.content)
