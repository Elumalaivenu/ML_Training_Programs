# app.py
import streamlit as st
from openai import OpenAI

# -------------------------
# Configuration
# -------------------------
st.set_page_config(page_title="OpenRouter Chat App", page_icon="🤖", layout="centered")

# Sidebar: API & Model selection
st.sidebar.title("⚙️ Settings")

api_key = st.sidebar.text_input("🔑 Enter your OpenRouter API Key:", type="password")

model_choice = st.sidebar.selectbox(
    "Select Model:",
    [
        "meta-llama/llama-3-8b-instruct",
        "mistralai/mistral-7b-instruct-v0.1", 
        "meta-llama/llama-3-70b-instruct"
    ],
)

st.title("💬 OpenRouter Chatbot (Open-Source Models)")
st.write("Chat with open models like **LLaMA 3**, **Mistral**, or **Gemma** through OpenRouter API.")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Input box
user_prompt = st.chat_input("Type your message...")

if user_prompt:
    if not api_key:
        st.warning("⚠️ Please enter your OpenRouter API Key in the sidebar.")
        st.stop()

    # Display user message
    st.chat_message("user").markdown(user_prompt)
    st.session_state.messages.append({"role": "user", "content": user_prompt})

    # Connect to OpenRouter
    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)

    # Add system prompt for model behavior
    system_prompt = {
        "role": "system",
        "content": "You are a helpful and concise AI assistant."
    }

    # Prepare messages
    full_conversation = [system_prompt] + st.session_state.messages

    # Stream response
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""

        try:
            # Use non-streaming for more reliable response
            response = client.chat.completions.create(
                model=model_choice,
                messages=full_conversation,
                max_tokens=300,
                temperature=0.7,
            )
            
            full_response = response.choices[0].message.content
            response_placeholder.markdown(full_response)
            
        except Exception as e:
            st.error(f"Error: {str(e)}")
            st.stop()

        st.session_state.messages.append({"role": "assistant", "content": full_response})
