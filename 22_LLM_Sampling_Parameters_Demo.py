# ------------------------------------------------------------
# Streamlit App: LLM Sampling Parameters Demo
# ------------------------------------------------------------
# Demonstrates how Temperature, Top-K, and Top-P influence output diversity
# Works with OpenRouter (open-source models like Mixtral, LLaMA, Gemma, etc.)
# ------------------------------------------------------------

import streamlit as st
import requests
import json
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# -----------------------------
# Streamlit Page Setup
# -----------------------------
st.set_page_config(page_title="LLM Sampling Parameter Demo", layout="centered")

st.title("🎛️ Explore LLM Sampling Parameters: Temperature, Top-K, Top-P")
st.markdown("""
This demo shows how **model configuration** changes an LLM's output for the *same prompt*.

- **Temperature** → randomness / creativity  
- **Top-K** → limits candidate tokens to top-K most probable  
- **Top-P** → cumulative probability cutoff (nucleus sampling)  
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

user_prompt = st.text_area(
    "💬 Enter your prompt:",
    "Write a short motivational quote about teamwork."
)

temperature = st.slider("🔥 Temperature", 0.0, 1.5, 0.7, 0.1)
top_k = st.slider("🎯 Top-K", 0, 200, 50, 10)
top_p = st.slider("🎲 Top-P", 0.1, 1.0, 0.9, 0.05)

# -----------------------------
# Function to Call OpenRouter
# -----------------------------
def call_llm(prompt, temperature, top_p, top_k):
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "top_p": top_p,
    }

    if top_k > 0:
        payload["top_k"] = top_k

    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers=headers,
        data=json.dumps(payload)
    )
    result = response.json()
    if "choices" in result:
        return result["choices"][0]["message"]["content"].strip()
    else:
        return f"❌ Error: {result}"

# -----------------------------
# Generate Response
# -----------------------------
if st.button("🚀 Generate Response"):
    st.markdown("### ⚙️ Current Model Settings")
    st.json({
        "temperature": temperature,
        "top_k": top_k,
        "top_p": top_p
    })

    with st.spinner("Generating creative output..."):
        prompt = f"You are a motivational coach. {user_prompt}"
        output = call_llm(prompt, temperature, top_p, top_k)

    st.success("✅ Response Generated Successfully!")
    st.markdown(f"### 💬 Model Output:\n\n{output}")

# -----------------------------
# Info Section
# -----------------------------
st.markdown("---")
st.markdown("""
### 🧠 What Happens Under the Hood
| Parameter | Description | Effect When Increased |
|------------|--------------|------------------------|
| **Temperature** | Controls randomness | More creative & diverse responses |
| **Top-K** | Chooses from top K tokens | More variation, possible drift |
| **Top-P** | Chooses tokens within a probability mass | Smoother sampling diversity |

👉 Try increasing *temperature* and *top-p* together for creative writing,  
and lowering them for factual or deterministic responses.
""")
st.caption("Built with ❤️ for LLM Training Workshops • Powered by OpenRouter")
