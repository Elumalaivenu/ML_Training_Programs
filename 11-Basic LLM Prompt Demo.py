# Example 1: Basic LLM Prompt Demo

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

# Simple prompt to the model
prompt = "Explain Artificial Intelligence in one simple sentence."

try:
    # Send the prompt to GPT model
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",  # You can also use "gpt-4-turbo"
        messages=[
            {"role": "system", "content": "You are a helpful teacher."},
            {"role": "user", "content": prompt}
        ]
    )

    # Print the model's reply
    print("Prompt:", prompt)
    print("Model Response:", response.choices[0].message.content)
    
except Exception as e:
    print(f"Error occurred: {e}")
    print("Please check your API key and internet connection.")
