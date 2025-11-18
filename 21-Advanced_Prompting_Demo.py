# ------------------------------------------------------------
# Streamlit App: Advanced Prompt Engineering Techniques Demo
# ------------------------------------------------------------
# Techniques covered:
# 1. Role-based
# 2. Contextual
# 3. Chain-of-Thought (CoT)
# 4. Tree-of-Thought (ToT)
# 5. ReAct
# ------------------------------------------------------------

import streamlit as st
import requests
import json
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# -----------------------------
# Page Setup
# -----------------------------
st.set_page_config(page_title="Advanced LLM Prompting Demo", layout="centered")

st.title("🧠 Advanced Prompt Engineering Techniques")
st.markdown("""
Compare how different **advanced prompting strategies** guide an open-source LLM's reasoning.

**Techniques:**  
1️⃣ Role-based 2️⃣ Contextual 3️⃣ Chain-of-Thought 4️⃣ Tree-of-Thought 5️⃣ ReAct  
""")

# -----------------------------
# User Inputs
# -----------------------------
# Get API key from environment file
api_key = os.getenv("OPENROUTER_API_KEY")

if not api_key:
    st.error("❌ OpenRouter API key not found in environment file!")
    st.markdown("""
    **Setup Instructions:**
    1. Create a `.env` file in your project directory
    2. Add: `OPENROUTER_API_KEY=your_api_key_here`
    3. Restart the application
    """)
    st.stop()

model = st.selectbox(
    "Select Model:",
    [
        "mistralai/mixtral-8x7b",
        "meta-llama/llama-3-70b-instruct",
        "google/gemma-2-9b-it",
        "nousresearch/hermes-3-llama-3.1-70b"
    ],
    index=0
)

user_query = st.text_area(
    "💬 Enter your Question:",
    "How can we reduce traffic congestion in large cities?"
)

prompt_type = st.radio(
    "Select Advanced Prompt Type:",
    ["Role-based", "Contextual", "Chain-of-Thought", "Tree-of-Thought", "ReAct"]
)

# -----------------------------
# Build Prompt Templates
# -----------------------------
def build_prompt(prompt_type, query):
    if prompt_type == "Role-based":
        return f"""
        You are a **city planning expert** specializing in sustainable urban transport systems.
        Please answer the following question professionally and with practical strategies.

        Question: {query}
        """

    elif prompt_type == "Contextual":
        return f"""
        Context:
        Many large cities face heavy traffic during peak hours due to over-reliance on private vehicles,
        poor public transport, and lack of smart infrastructure.

        Task:
        Based on this context, answer the question below.

        Question: {query}
        """

    elif prompt_type == "Chain-of-Thought":
        return f"""
        You are an intelligent assistant. Think step by step before answering.
        Explain your reasoning clearly.

        Question: {query}

        Let's reason this out step by step:
        """

    elif prompt_type == "Tree-of-Thought":
        return f"""
        You are an expert problem solver.
        Think of multiple possible approaches (branches), evaluate each, and choose the best solution.

        Question: {query}

        Follow this structure:
        1. Generate 2–3 different ideas.
        2. Evaluate pros and cons.
        3. Pick the best one with reasoning.
        """

    elif prompt_type == "ReAct":
        return f"""
        You are an AI assistant that alternates between reasoning and action.
        When reasoning, explain your thought process.
        When acting, provide a clear answer.

        Example format:
        Thought: ...
        Action: ...

        Now solve this:
        Question: {query}
        """

# -----------------------------
# Function to Call OpenRouter API
# -----------------------------
def call_openrouter(prompt):
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    data = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7,
    }

    response = requests.post("https://openrouter.ai/api/v1/chat/completions",
                             headers=headers, data=json.dumps(data))
    result = response.json()
    if "choices" in result:
        return result["choices"][0]["message"]["content"].strip()
    else:
        return f"❌ Error: {result}"

# -----------------------------
# Run the Prompt
# -----------------------------
if st.button("🚀 Generate Response"):
    with st.spinner(f"Calling model using {prompt_type} prompting..."):
        prompt = build_prompt(prompt_type, user_query)
        output = call_openrouter(prompt)

    st.success("✅ Response Generated Successfully!")
    st.markdown(f"### 🧩 {prompt_type} Prompt Result:\n\n{output}")

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.caption("Built with ❤️ for LLM Prompt Engineering Training • Powered by OpenRouter")
