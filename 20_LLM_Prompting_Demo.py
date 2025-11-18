# ------------------------------------------------------------
# Streamlit App: Zero-shot, One-shot, and Few-shot Prompting Demo
# Using OpenRouter (works with open-source LLMs like Mistral, LLaMA, etc.)
# ------------------------------------------------------------

import streamlit as st
import requests
import json

# -----------------------------
# Configuration
# -----------------------------
st.set_page_config(page_title="Prompt Engineering Demo", layout="centered")

st.title("🧠 LLM Prompting Techniques Demo")
st.markdown(
    """
    Explore how **Zero-shot**, **One-shot**, and **Few-shot** prompting 
    affect the response quality of an LLM.
    """
)

# Input fields
API_KEY = st.text_input("🔑 Enter your OpenRouter API Key:", type="password")
MODEL = st.selectbox(
    "Select Model:",
    [
        "mistralai/mistral-7b-instruct-v0.1",
        "meta-llama/llama-3-8b-instruct",
        "meta-llama/llama-3-70b-instruct"
       
    ]
)

user_query = st.text_area("💬 Enter your question:", "Explain what a neural network is in simple terms.")
prompt_type = st.radio(
    "Select Prompt Type:",
    ["Zero-shot", "One-shot", "Few-shot"]
)

if st.button("🚀 Generate Response"):
    if not API_KEY:
        st.warning("Please enter your OpenRouter API key.")
        st.stop()

    # -----------------------------
    # Define prompt templates
    # -----------------------------
    if prompt_type == "Zero-shot":
        prompt = f"You are a helpful AI assistant.\nQuestion: {user_query}"

    elif prompt_type == "One-shot":
        prompt = f"""
        You are a helpful AI assistant.
        Example:
        Q: What is machine learning?
        A: Machine learning is a way for computers to learn from data without being explicitly programmed.
        
        Now, answer this question:
        Q: {user_query}
        """

    elif prompt_type == "Few-shot":
        prompt = f"""
        You are a helpful AI assistant.
        Here are a few examples:
        
        Example 1:
        Q: What is AI?
        A: AI stands for Artificial Intelligence. It refers to machines that can perform tasks that require human-like intelligence.
        
        Example 2:
        Q: What is deep learning?
        A: Deep learning is a subset of machine learning that uses neural networks with many layers to process data and make predictions.
        
        Now, answer this question:
        Q: {user_query}
        """

    # -----------------------------
    # Call OpenRouter API
    # -----------------------------
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }

    data = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7,
    }

    with st.spinner("Generating response..."):
        response = requests.post("https://openrouter.ai/api/v1/chat/completions",
                                 headers=headers, data=json.dumps(data))
        result = response.json()

        if "choices" in result:
            answer = result["choices"][0]["message"]["content"].strip()
            st.success("✅ Response Generated Successfully!")
            st.markdown(f"### 🧩 {prompt_type} Response:\n\n{answer}")
        else:
            st.error("❌ Failed to get a response from the model.")
            st.write(result)
