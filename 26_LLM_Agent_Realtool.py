# ------------------------------------------------------------
# Streamlit App: LLM Agent + Memory + Real Tools Demo
# ------------------------------------------------------------
# Demonstrates how an AI Agent:
# ✅ Uses multiple tools dynamically (Wikipedia, Math, Time)
# ✅ Maintains conversation memory
# ✅ Shows reasoning via ReAct framework
# ------------------------------------------------------------

import streamlit as st
from langchain_openai import OpenAI
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from langchain.tools import Tool
import os
from dotenv import load_dotenv
import requests
import json
from datetime import datetime

# Load environment variables
load_dotenv()

# -----------------------------
# Step 1️⃣ Streamlit Setup
# -----------------------------
st.set_page_config(page_title="🤖 LLM Agent with Tools + Memory", layout="wide")

st.title("🤖 LLM Agent with Real Tools + Memory (LangChain Demo)")
st.markdown("""
This app shows how **LLM Agents** can:
- 🔍 Use **real tools** (Wikipedia, Calculator, Real-time Search)
- 🧠 Remember context between turns
- 🧩 Use **Reasoning + Acting** (ReAct framework)

**Try sample queries:**
- "Who is Elon Musk?"
- "How old is he?"
- "What is 12.5 * 8.3?"
- "What's the latest news about Tesla stock?"
- "What's the current weather in New York?"
- "What are the top trending topics today?"
""")

# -----------------------------
# Step 2️⃣ API Key Setup
# -----------------------------
# Get API keys from environment file
api_key = os.getenv("OPENAI_API_KEY")
serpapi_key = os.getenv("SERPAPI_API_KEY")

if not api_key:
    st.error("❌ OpenAI API key not found in environment file!")
    st.markdown("""
    **Setup Instructions:**
    1. Create a `.env` file in your project directory
    2. Add: `OPENAI_API_KEY=your_api_key_here`
    3. Add: `SERPAPI_API_KEY=your_serpapi_key_here` (optional, for real-time search)
    4. Restart the application
    """)
    st.stop()
else:
    st.sidebar.success("✅ OpenAI API Key loaded from config")
    
    # Display SerpAPI status
    if serpapi_key:
        st.sidebar.success("✅ SerpAPI Key loaded - Real-time search enabled!")
    else:
        st.sidebar.warning("⚠️ SerpAPI Key not found in .env file - Limited to Wikipedia search")
        st.sidebar.info("💡 Add SERPAPI_API_KEY to .env file for real-time Google search capabilities")

# -----------------------------
# Step 3️⃣ Setup LLM, Memory, Tools
# -----------------------------
llm = OpenAI(temperature=0.5, openai_api_key=api_key)

# Create custom tools to avoid dependency issues
def wikipedia_search(query):
    """Search Wikipedia for information"""
    try:
        import requests
        # Use Wikipedia API directly
        url = "https://en.wikipedia.org/api/rest_v1/page/summary/" + query.replace(" ", "_")
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            return f"Wikipedia Summary: {data.get('extract', 'No summary available')}"
        else:
            return f"Could not find Wikipedia information for: {query}"
    except Exception as e:
        return f"Wikipedia search failed: {str(e)}"

def math_calculator(expression):
    """Calculate mathematical expressions"""
    try:
        # Safe evaluation of mathematical expressions
        import math
        import re
        # Only allow safe mathematical operations
        safe_dict = {
            "__builtins__": {},
            "abs": abs, "round": round, "min": min, "max": max,
            "sum": sum, "pow": pow, "sqrt": math.sqrt,
            "sin": math.sin, "cos": math.cos, "tan": math.tan,
            "log": math.log, "exp": math.exp, "pi": math.pi,
            "e": math.e
        }
        # Remove any potentially dangerous characters
        cleaned_expr = re.sub(r'[^0-9+\-*/().\s]', '', str(expression))
        result = eval(cleaned_expr, safe_dict)
        return f"Calculation result: {result}"
    except Exception as e:
        return f"Mathematical calculation failed: {str(e)}"

def serpapi_search(query):
    """Search using SerpAPI for real-time information"""
    try:
        if not serpapi_key:
            return "SerpAPI key not available. Please add SERPAPI_API_KEY to your .env file for real-time search."
        
        import requests
        url = "https://serpapi.com/search"
        params = {
            "engine": "google",
            "q": query,
            "api_key": serpapi_key,
            "num": 5,  # Get more results
            "hl": "en",
            "gl": "us"
        }
        
        response = requests.get(url, params=params, timeout=30)  # Add timeout
        if response.status_code == 200:
            data = response.json()
            
            # Check for organic results
            organic_results = data.get("organic_results", [])
            news_results = data.get("news_results", [])
            
            results = []
            
            # Add news results first for trending topics
            for news in news_results[:2]:
                title = news.get("title", "")
                snippet = news.get("snippet", "")
                date = news.get("date", "")
                source = news.get("source", "")
                results.append(f"[NEWS - {source}] {title}\n{snippet}\n({date})")
            
            # Add organic results
            for result in organic_results[:3]:
                title = result.get("title", "")
                snippet = result.get("snippet", "")
                results.append(f"{title}\n{snippet}")
            
            if results:
                return "Real-time Search Results:\n\n" + "\n\n---\n\n".join(results)
            else:
                return f"No current search results found for: {query}"
        else:
            return f"Search API returned error code {response.status_code} for query: {query}"
    except requests.exceptions.Timeout:
        return "Search request timed out. Please try again with a more specific query."
    except Exception as e:
        return f"Real-time search failed: {str(e)}"

# Create tool objects
wikipedia_tool = Tool(
    name="Wikipedia",
    func=wikipedia_search,
    description="Useful for getting factual information about people, places, concepts, and historical events from Wikipedia"
)

math_tool = Tool(
    name="Calculator",
    func=math_calculator,
    description="Useful for performing mathematical calculations. Input should be a mathematical expression like '2+2' or 'sqrt(16)'"
)

# Create tools list based on available API keys
if serpapi_key:
    # Include SerpAPI for real-time search if key is available
    serpapi_tool = Tool(
        name="RealTimeSearch",
        func=serpapi_search,
        description="Useful for getting current information, news, weather, stock prices, and real-time data from Google search"
    )
    tools = [wikipedia_tool, math_tool, serpapi_tool]
else:
    # Fallback to basic tools without SerpAPI
    tools = [wikipedia_tool, math_tool]

# Initialize chat history in session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "messages" not in st.session_state:
    st.session_state.messages = []

# Create a prompt template that includes chat history
prompt = PromptTemplate.from_template("""Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Previous conversation:
{chat_history}

Begin!

Question: {input}
Thought:{agent_scratchpad}""")

# Create the agent
agent = create_react_agent(llm, tools, prompt)

# Create agent executor
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=10,
    max_execution_time=240,
    return_intermediate_steps=True
)

# -----------------------------
# Step 4️⃣ Streamlit Chat UI
# -----------------------------
user_input = st.text_input("💬 Ask your question:", placeholder="e.g., Who is the president of India?")

if st.button("🚀 Run Agent"):
    if not user_input.strip():
        st.warning("Please enter a question!")
        st.stop()
        
    with st.spinner("Agent is reasoning and acting..."):
        try:
            # Format chat history for the agent
            formatted_chat_history = "\n".join([
                f"{'Human: ' if isinstance(msg, HumanMessage) else 'AI: '}{msg.content}"
                for msg in st.session_state.chat_history
            ]) if st.session_state.chat_history else "No previous conversation."
            
            # Use invoke instead of run for better error handling
            result = agent_executor.invoke({
                "input": user_input,
                "chat_history": formatted_chat_history
            })
            response = result.get("output", "No response generated")
            
            # Store conversation
            st.session_state.messages.append((user_input, response))
            st.session_state.chat_history.append(HumanMessage(content=user_input))
            st.session_state.chat_history.append(AIMessage(content=response))

            # Display
            st.success("🤖 Agent Response:")
            st.write(response)
            
            # Show intermediate steps if available (for debugging)
            if "intermediate_steps" in result and st.checkbox("🔍 Show Agent Reasoning Steps"):
                with st.expander("🧠 Agent Reasoning Process"):
                    for i, (action, observation) in enumerate(result["intermediate_steps"], 1):
                        st.markdown(f"**Step {i}:**")
                        st.code(f"Tool: {action.tool}\nInput: {action.tool_input}\nResult: {observation}")
                        
        except Exception as e:
            st.error(f"❌ Agent execution failed: {str(e)}")
            st.info("💡 Try rephrasing your question or make it more specific.")
            
            # Still store the conversation with error for context
            error_response = f"Sorry, I encountered an error: {str(e)}"
            st.session_state.messages.append((user_input, error_response))
            st.session_state.chat_history.append(HumanMessage(content=user_input))
            st.session_state.chat_history.append(AIMessage(content=error_response))

    # Display memory buffer
    with st.expander("🧠 Agent Memory Buffer"):
        # Format chat history for display
        formatted_history = "\n".join([
            f"{'Human: ' if isinstance(msg, HumanMessage) else 'AI: '}{msg.content}"
            for msg in st.session_state.chat_history
        ])
        st.code(formatted_history if formatted_history else "No conversation history yet.")

    # Show conversation history
    st.markdown("---")
    st.subheader("💬 Chat History")
    for i, (q, a) in enumerate(st.session_state.messages, start=1):
        st.markdown(f"**{i}. You:** {q}")
        st.markdown(f"**🤖 Agent:** {a}")

# -----------------------------
# Notes for Trainers
# -----------------------------
# 🧠 Memory: Chat history maintained using HumanMessage and AIMessage.
# 🛠️ Tools (Custom implementations to avoid dependency issues):
#   - Wikipedia: factual lookup using Wikipedia REST API
#   - Calculator: mathematical calculations with safe eval
#   - SerpAPI: real-time Google search (loaded from SERPAPI_API_KEY in .env file)
# 🔄 create_react_agent:
#   Uses ReAct pattern for Reason + Act + Observation chain-of-thought style reasoning.
# 🔐 Environment Configuration:
#   - OPENAI_API_KEY: Required for LLM functionality
#   - SERPAPI_API_KEY: Optional, enables real-time search capabilities
# 🌐 SerpAPI provides access to:
#   - Current news and events
#   - Real-time data (weather, stocks, etc.)
#   - Latest search results from Google
# 📝 Note: All API keys are securely loaded from .env file, not user input
# ------------------------------------------------------------
st.markdown("---")
st.caption("Built for GenAI Training • LangChain + Streamlit • © 2025")
