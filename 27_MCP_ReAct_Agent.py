# ------------------------------------------------------------
# Streamlit App: ReAct-style MCP Agent (Weather + Wikipedia)
# ------------------------------------------------------------
# 🧠 Demonstrates:
# ✅ Agent reasoning & action selection
# ✅ Dynamic tool calling (Weather or Wikipedia)
# ✅ Context fusion and final reasoning
# ✅ Secure API key handling via .env
# ------------------------------------------------------------

import os
import re
import requests
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

# Try to import wikipedia, if not available, we'll use a fallback
try:
    import wikipedia
    WIKIPEDIA_AVAILABLE = True
except ImportError:
    WIKIPEDIA_AVAILABLE = False
    st.warning("⚠️ Wikipedia package not found. Using alternative Wikipedia API.")

# ------------------------------------------------------------
# Step 1️⃣ Load Environment Variables
# ------------------------------------------------------------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")

# ------------------------------------------------------------
# Step 2️⃣ Streamlit Setup
# ------------------------------------------------------------
st.set_page_config(page_title="🧠 MCP ReAct Agent Demo", layout="centered")
st.title("🤖 Model Context Protocol — ReAct Agent Demo (Weather + Wikipedia)")

# Check API keys and display status
if not OPENAI_API_KEY:
    st.error("❌ OpenAI API key not found in environment file!")
    st.markdown("""
    **Setup Instructions:**
    1. Create a `.env` file in your project directory
    2. Add: `OPENAI_API_KEY=your_api_key_here`
    3. Add: `OPENWEATHER_API_KEY=your_weather_api_key_here` (optional, for weather features)
    4. Restart the application
    
    **For OpenWeather API Key:**
    - Visit: https://openweathermap.org/api
    - Sign up for a free account  
    - Get your API key from the dashboard
    - ⚠️ Important: Wait 10-15 minutes after signup for key activation
    """)
    st.stop()
else:
    st.sidebar.success("✅ OpenAI API Key loaded from config")
    
    if OPENWEATHER_API_KEY:
        st.sidebar.success("✅ OpenWeather API Key loaded - Weather queries enabled!")
        st.sidebar.info("🔑 Note: Make sure your OpenWeather API key is valid and activated")
    else:
        st.sidebar.warning("⚠️ OpenWeather API Key not found - Weather queries disabled")
        st.sidebar.info("💡 Add OPENWEATHER_API_KEY to .env file for weather capabilities")
        st.sidebar.markdown("""
        **To get OpenWeather API Key:**
        1. Visit: https://openweathermap.org/api
        2. Sign up for free account
        3. Get your API key
        4. Add to .env: `OPENWEATHER_API_KEY=your_key_here`
        5. Wait ~10-15 minutes for activation
        """)

st.markdown("""
### 🔍 How It Works:
This demo shows **ReAct-style reasoning** — where the LLM:
1. 🧠 **Thinks** about the query  
2. ⚙️ **Acts** by choosing a tool (Weather or Wikipedia)  
3. 🗣️ **Responds** using the tool output + reasoning  

Everything happens automatically — no manual tool selection needed!

**Environment Configuration:**
- All API keys are securely loaded from `.env` file
- No sensitive information in the UI
""")

# Display available tools
st.markdown("### 🛠️ Available Tools:")
col1, col2 = st.columns(2)
with col1:
    if OPENWEATHER_API_KEY:
        st.success("🌦️ **Weather Tool** - Real-time weather data")
    else:
        st.warning("🌦️ **Weather Tool** - Disabled (missing API key)")
        
with col2:
    st.success("📘 **Wikipedia Tool** - Encyclopedic knowledge")

# ------------------------------------------------------------
# Step 3️⃣ Define Tools
# ------------------------------------------------------------
def get_weather(city: str):
    """Fetch weather info from OpenWeather API."""
    if not OPENWEATHER_API_KEY:
        return "⚠️ Weather service unavailable: OpenWeather API key not configured in .env file"

    try:
        url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={OPENWEATHER_API_KEY}&units=metric"
        response = requests.get(url, timeout=10)

        if response.status_code == 401:
            return "❌ Weather service error: Invalid OpenWeather API key. Please check your OPENWEATHER_API_KEY in the .env file. Get a free API key at: https://openweathermap.org/api"
        elif response.status_code == 404:
            return f"❌ Weather service error: City '{city}' not found. Please check the city name spelling."
        elif response.status_code != 200:
            return f"❌ Weather service error (code {response.status_code}): {response.text}"

        data = response.json()
        desc = data["weather"][0]["description"].capitalize()
        temp = data["main"]["temp"]
        feels_like = data["main"]["feels_like"]
        humidity = data["main"]["humidity"]
        
        return f"🌦️ Weather in {city.capitalize()}: {desc}, {temp}°C (feels like {feels_like}°C), Humidity: {humidity}%"
    
    except requests.exceptions.Timeout:
        return f"❌ Weather request timed out for {city}"
    except Exception as e:
        return f"❌ Weather service error for {city}: {str(e)}"

def get_wikipedia_summary(topic: str):
    """Fetch summary from Wikipedia."""
    try:
        if WIKIPEDIA_AVAILABLE:
            # Use wikipedia package if available
            summary = wikipedia.summary(topic, sentences=3)
            return f"📘 Wikipedia summary for **{topic}**:\n{summary}"
        else:
            # Use Wikipedia API directly as fallback
            search_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{topic.replace(' ', '_')}"
            response = requests.get(search_url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                extract = data.get('extract', 'No summary available')
                if extract:
                    return f"📘 Wikipedia summary for **{topic}**:\n{extract}"
                else:
                    return f"📘 No Wikipedia summary found for: {topic}"
            else:
                # Try search API if direct page fails
                search_url = f"https://en.wikipedia.org/api/rest_v1/page/search/{topic}"
                search_response = requests.get(search_url, timeout=10)
                
                if search_response.status_code == 200:
                    search_data = search_response.json()
                    pages = search_data.get('pages', [])
                    
                    if pages:
                        first_page = pages[0]
                        return f"📘 Wikipedia result for **{topic}**:\n{first_page.get('description', 'No description available')}"
                
                return f"📘 Could not find Wikipedia information for: {topic}"
    
    except requests.exceptions.Timeout:
        return f"📘 Wikipedia request timed out for: {topic}"
    except Exception as e:
        return f"⚠️ Wikipedia Error: {str(e)}"

# ------------------------------------------------------------
# Step 4️⃣ Initialize LLM
# ------------------------------------------------------------
try:
    client = OpenAI(api_key=OPENAI_API_KEY)
except Exception as e:
    st.error(f"❌ Failed to initialize OpenAI client: {str(e)}")
    st.stop()

def call_llm(messages):
    """Call OpenAI LLM with error handling."""
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=0.7,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"❌ LLM Error: {str(e)}"

# ------------------------------------------------------------
# Step 5️⃣ ReAct Agent Logic
# ------------------------------------------------------------
def react_agent(query):
    """Simulate reasoning + acting + responding."""
    # 🧠 Step 1: Ask LLM what to do
    available_tools = "1. Wikipedia (for general knowledge)"
    if OPENWEATHER_API_KEY:
        available_tools = "1. Weather API (for real-time weather in a city)\n2. Wikipedia (for general knowledge)"
    
    reasoning_prompt = f"""
You are a helpful AI agent connected to these tools:
{available_tools}

Decide which tool to use for the user query: "{query}"

{"Note: Weather tool is available for weather queries." if OPENWEATHER_API_KEY else "Note: Weather tool is NOT available, use Wikipedia for all queries."}

Respond strictly in this JSON format:
{{
    "reasoning": "...",
    "tool": "{"weather" if OPENWEATHER_API_KEY else "wikipedia"}" or "wikipedia",
    "topic": "..."
}}
"""

    reasoning_response = call_llm([{"role": "system", "content": "You are a reasoning agent."},
                                   {"role": "user", "content": reasoning_prompt}])

    st.info("🧠 LLM Reasoning & Tool Selection")
    st.code(reasoning_response, language="json")

    # Extract tool and topic
    match_tool = re.search(r'"tool":\s*"([^"]+)"', reasoning_response)
    match_topic = re.search(r'"topic":\s*"([^"]+)"', reasoning_response)
    tool_name = match_tool.group(1) if match_tool else "wikipedia"
    topic = match_topic.group(1) if match_topic else query

    # Force Wikipedia if weather tool requested but not available
    if tool_name == "weather" and not OPENWEATHER_API_KEY:
        tool_name = "wikipedia"
        tool_output = f"⚠️ Weather service unavailable. Here's general information about {topic}:\n" + get_wikipedia_summary(topic)
    elif tool_name == "weather":
        weather_result = get_weather(topic)
        # Check if weather API failed due to invalid key, fall back to Wikipedia
        if "Invalid OpenWeather API key" in weather_result or "Weather service error" in weather_result:
            st.warning(f"Weather API failed: {weather_result}")
            st.info("Falling back to Wikipedia for general information...")
            tool_output = f"⚠️ Weather service error. Here's general information about {topic}:\n" + get_wikipedia_summary(topic)
            tool_name = "wikipedia (fallback)"
        else:
            tool_output = weather_result
    else:
        tool_output = get_wikipedia_summary(topic)

    st.success(f"🧩 Tool Used: {tool_name.title()}")
    st.write(tool_output)

    # 🗣️ Step 3: Final answer with tool context
    final_prompt = f"""
User question: {query}
Tool used: {tool_name}
Tool result: {tool_output}

Now give a helpful final answer combining both reasoning and tool output.
"""
    final_answer = call_llm([
        {"role": "system", "content": "You are a smart AI assistant combining reasoning and real data."},
        {"role": "user", "content": final_prompt},
    ])

    return final_answer

# ------------------------------------------------------------
# Step 6️⃣ Streamlit Interaction
# ------------------------------------------------------------
st.markdown("### 💬 Ask Your Question:")

# Provide example queries based on available tools
if OPENWEATHER_API_KEY:
    example_text = "What's the weather in Paris? | Who is Albert Einstein? | Weather in Tokyo | History of machine learning"
    default_query = "What's the weather in Paris?"
else:
    example_text = "Who is Albert Einstein? | History of machine learning | Tell me about Python programming | Explain quantum physics"
    default_query = "Who is Albert Einstein?"

st.caption(f"💡 **Example queries:** {example_text}")

query = st.text_input("Enter your question:", value=default_query, placeholder="Type your question here...")

if st.button("🚀 Run ReAct Agent", type="primary"):
    if not query.strip():
        st.warning("Please enter a question!")
    else:
        with st.spinner("🧠 Agent reasoning and acting..."):
            try:
                answer = react_agent(query.strip())
                st.success("🧠 Final Answer:")
                st.markdown(answer)
            except Exception as e:
                st.error(f"❌ Agent execution failed: {str(e)}")
                st.info("💡 Please try rephrasing your question or check your API configuration.")

st.markdown("---")
st.caption("🔐 **Security Note:** All API keys are loaded from .env file for maximum security")
st.caption("Built for GenAI Training • ReAct MCP Agent Demo • © 2025")
