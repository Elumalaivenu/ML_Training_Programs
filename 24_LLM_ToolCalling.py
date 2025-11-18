# ------------------------------------------------------------
# Streamlit App: LLM Function / Tool Calling Demo
# ------------------------------------------------------------
# Demonstrates how an LLM can decide to call external tools
# (e.g., Weather lookup, Calculator) based on user intent.
# ------------------------------------------------------------

import streamlit as st
import json
import random
import requests
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# -----------------------------
# API Key Setup
# -----------------------------
api_key = os.getenv("OPENROUTER_API_KEY")

# -----------------------------
# Step 1️⃣ Define Simple "Tools"
# -----------------------------
def get_weather(location):
    weather_data = {
        "Chennai": "☀️ 33°C, Sunny",
        "Mumbai": "🌦️ 29°C, Humid",
        "Delhi": "🌤️ 28°C, Clear",
        "Bangalore": "⛅ 26°C, Pleasant",
        "Paris": "🌧️ 17°C, Cloudy"
    }
    return weather_data.get(location, f"Weather for {location} not found.")

def add_numbers(a, b):
    return f"The sum of {a} and {b} is {a + b}."

# -----------------------------
# OpenRouter API Function
# -----------------------------
def call_openrouter_for_decision(user_query):
    """Use OpenRouter LLM to make tool calling decisions."""
    if not api_key:
        return None
    
    prompt = f"""
    You are an AI assistant that can decide when to call external tools.
    
    Available tools:
    1. get_weather(location) - Get weather for a specific location
    2. add_numbers(a, b) - Add two numbers together
    
    User query: "{user_query}"
    
    Analyze the query and respond with ONLY a JSON object in this format:
    {{"action": "tool_name", "parameters": {{"param1": "value1", "param2": "value2"}}}}
    
    If no tool is needed, respond with:
    {{"action": "none", "parameters": {{}}}}
    
    Examples:
    - "What's weather in Paris?" → {{"action": "get_weather", "parameters": {{"location": "Paris"}}}}
    - "Add 5 and 10" → {{"action": "add_numbers", "parameters": {{"a": 5, "b": 10}}}}
    - "Tell me a joke" → {{"action": "none", "parameters": {{}}}}
    """
    
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": "meta-llama/llama-3-8b-instruct",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1
    }
    
    try:
        response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, data=json.dumps(payload))
        result = response.json()
        if "choices" in result:
            llm_response = result["choices"][0]["message"]["content"].strip()
            return json.loads(llm_response)
        return None
    except:
        return None

# -----------------------------
# Step 2️⃣ Simple LLM Router (Decision Logic)
# -----------------------------
def llm_decision(user_query):
    """Simulate an LLM that decides which function to call."""
    user_query_lower = user_query.lower()

    if "weather" in user_query_lower:
        # Extract location name (naive method)
        words = user_query.split()
        location = words[-1].capitalize()
        return {"action": "get_weather", "parameters": {"location": location}}

    elif "add" in user_query_lower or "sum" in user_query_lower:
        # Extract numbers
        numbers = [int(s) for s in user_query.split() if s.isdigit()]
        if len(numbers) >= 2:
            return {"action": "add_numbers", "parameters": {"a": numbers[0], "b": numbers[1]}}

    # Otherwise, no function needed
    return {"action": "none", "parameters": {}}

# -----------------------------
# Step 3️⃣ Streamlit UI Setup
# -----------------------------
st.set_page_config(page_title="🧠 LLM Function Calling Demo", layout="centered")

st.title("🧠 LLM Tool / Function Calling Demo")
st.markdown("""
This demo shows how an **LLM can decide when to call a tool or function** 
based on the user's intent.

Try asking:
- "What's the weather in Chennai?"
- "Add 10 and 25"
- "Tell me a fun fact" (no tool call)
""")

# API Key status
if api_key:
    st.sidebar.success("✅ OpenRouter API key loaded from config")
    use_openrouter = st.sidebar.checkbox("🤖 Use OpenRouter LLM for decisions", value=True)
else:
    st.sidebar.warning("⚠️ No OpenRouter API key found - using simple local logic")
    st.sidebar.markdown("""
    **To use OpenRouter:**
    1. Add `OPENROUTER_API_KEY=your_key` to `.env` file
    2. Restart the app
    """)
    use_openrouter = False

# -----------------------------
# Step 4️⃣ User Input
# -----------------------------
user_query = st.text_input("💬 Enter your query:", "What’s the weather in Paris?")

if st.button("🚀 Run LLM Decision"):
    st.markdown("---")
    
    if use_openrouter and api_key:
        st.subheader("� Step 1: OpenRouter LLM Thinking...")
        with st.spinner("Calling OpenRouter API..."):
            plan = call_openrouter_for_decision(user_query)
        
        if plan:
            st.json(plan)
        else:
            st.error("Failed to get decision from OpenRouter. Using local logic...")
            plan = llm_decision(user_query)
            st.json(plan)
    else:
        st.subheader("🧠 Step 1: Local LLM Thinking...")
        plan = llm_decision(user_query)
        st.json(plan)

    st.markdown("### ⚙️ Step 2: Action Execution")

    # -----------------------------
    # Step 5️⃣ Execute Function
    # -----------------------------
    if plan["action"] == "get_weather":
        result = get_weather(**plan["parameters"])
        st.success(f"🧩 Tool Used → get_weather()")
        st.info(result)

    elif plan["action"] == "add_numbers":
        result = add_numbers(**plan["parameters"])
        st.success(f"🧩 Tool Used → add_numbers()")
        st.info(result)

    else:
        st.warning("💬 No tool call needed.")
        st.write("LLM handled the request directly: *'Here's an interesting fact about AI: it learns from data patterns!'*")

# -----------------------------
# Step 6️⃣ Explanation Section
# -----------------------------
st.markdown("---")
st.markdown("""
### 🧭 How This Works
| Step | What Happens | Example |
|------|---------------|----------|
| **1. Intent Detection** | LLM interprets your input | “Add 5 and 10” → math task |
| **2. Decision** | Decides which *tool* to call | Picks `add_numbers()` |
| **3. Execution** | Calls the function and returns result | `15` |
| **4. No Tool?** | Responds directly | “Tell me a joke” |

✅ **Function Calling** = Letting the model *reason first*, then *act through tools*.
""")

st.caption("Built with ❤️ using Streamlit • For LLM Training Demos • © 2025")
