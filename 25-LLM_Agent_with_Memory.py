# ------------------------------------------------------------
# Streamlit App: LLM Agent with Memory (LangChain Demo)
# ------------------------------------------------------------
# Demonstrates how an agent can:
# ✅ Remember previous interactions
# ✅ Use tools like calculator or search
# ✅ Use a reasoning chain to answer user queries
# ------------------------------------------------------------

import streamlit as st
from langchain_openai import OpenAI
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from langchain.tools import Tool
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# -----------------------------
# Step 1️⃣ Streamlit Page Setup
# -----------------------------
st.set_page_config(page_title="🧠 LLM Agent + Memory Demo", layout="centered")

st.title("🧠 LLM Agent + Memory Demo (LangChain)")
st.markdown("""
This app demonstrates how **LLM Agents** can:
- Use **memory** to recall past context
- Use **tools** like a calculator
- Respond intelligently across multiple turns  

Try asking:
- “What is 12 + 8?”
- “Now multiply that by 3”
- “Remind me what my first question was?”
""")

# -----------------------------
# Step 2️⃣ API Key Setup
# -----------------------------
# Get API key from environment file
openai_api_key = os.getenv("OPENAI_API_KEY")

if not openai_api_key:
    st.error("❌ OpenAI API key not found in environment file!")
    st.markdown("""
    **Setup Instructions:**
    1. Create a `.env` file in your project directory
    2. Add: `OPENAI_API_KEY=your_api_key_here`
    3. Restart the application
    """)
    st.stop()
else:
    st.sidebar.success("✅ OpenAI API Key loaded from config")

# -----------------------------
# Step 3️⃣ Define Custom Tools (avoiding numexpr dependency)
# -----------------------------
from langchain.tools import Tool

def simple_calculator(expression):
    """Simple calculator that can handle basic math operations."""
    try:
        # Only allow basic operations for safety
        allowed_chars = "0123456789+-*/(). "
        if all(c in allowed_chars for c in expression):
            result = eval(expression)
            return f"The result of {expression} is {result}"
        else:
            return "Error: Only basic math operations (+, -, *, /, parentheses) are allowed."
    except Exception as e:
        return f"Error calculating: {str(e)}"

# Create custom tools
calculator_tool = Tool(
    name="Calculator",
    description="Useful for when you need to answer questions about math. Input should be a math expression like '2+2' or '10*5'.",
    func=simple_calculator
)

tools = [calculator_tool]

# Define LLM
llm = OpenAI(temperature=0.5, openai_api_key=openai_api_key)

# Initialize chat history in session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

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
    max_iterations=5
)

# -----------------------------
# Step 5️⃣ Chat Interface
# -----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

user_query = st.text_input("💬 Enter your question:", "What is 12 + 8?")

if st.button("🚀 Ask Agent"):
    with st.spinner("Agent thinking..."):
        try:
            # Format chat history for the agent
            formatted_chat_history = "\n".join([
                f"{'Human: ' if isinstance(msg, HumanMessage) else 'AI: '}{msg.content}"
                for msg in st.session_state.chat_history
            ]) if st.session_state.chat_history else "No previous conversation."
            
            response = agent_executor.invoke({
                "input": user_query,
                "chat_history": formatted_chat_history
            })
            agent_response = response.get("output", "No response generated")
        except Exception as e:
            agent_response = f"Error: {str(e)}"

    # Save conversation to chat history
    st.session_state.messages.append((user_query, agent_response))
    st.session_state.chat_history.append(HumanMessage(content=user_query))
    st.session_state.chat_history.append(AIMessage(content=agent_response))

    st.success("🧩 Agent Response:")
    st.write(agent_response)

    st.markdown("---")
    st.subheader("🧠 Conversation Memory (Agent remembers):")
    # Format chat history for display
    formatted_history = "\n".join([
        f"{'Human: ' if isinstance(msg, HumanMessage) else 'AI: '}{msg.content}"
        for msg in st.session_state.chat_history
    ])
    st.code(formatted_history if formatted_history else "No conversation history yet.")

    st.markdown("### 💬 Chat History")
    for i, (q, a) in enumerate(st.session_state.messages, start=1):
        st.markdown(f"**{i}. You:** {q}")
        st.markdown(f"**🤖 Agent:** {a}")

# ------------------------------------------------------------
# Notes for Trainers
# ------------------------------------------------------------
# 1. Memory: Chat history is maintained using HumanMessage and AIMessage.
# 2. Tools: Custom calculator tool allows the LLM to perform calculations.
# 3. Agent Type: create_react_agent uses ReAct pattern for reasoning and actions.
# ------------------------------------------------------------
st.markdown("---")
st.caption("Built with ❤️ using LangChain + Streamlit • For LLM Agent Demos • © 2025")
