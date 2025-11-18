# mcp_client.py
import os
import re
import json
import requests
from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE") or "https://api.openai.com/v1"  # or openrouter base
MCP_SERVER_KEY = os.getenv("MCP_SERVER_KEY", "supersecretlocalkey")
MCP_SERVER_URL = os.getenv("MCP_SERVER_URL", "http://127.0.0.1:8000/sse")

# Use OpenAI-compatible library via requests to keep code simple
def call_llm_for_decision(user_query):
    """
    Ask the LLM to choose a tool and a topic. The LLM must reply in JSON.
    Example expected JSON:
      {"reasoning":"...","tool":"weather","topic":"Chennai"}
    """
    prompt = f"""
You are an agent deciding which tool to call for the user query.

Available tools:
- weather: returns live weather for a city (tool name 'weather')
- wikipedia: returns a short wiki summary for a topic (tool name 'wikipedia')
- time: returns current local time (tool name 'time')

User query: "{user_query}"

Respond EXACTLY in JSON with keys:
{{ "reasoning": "...", "tool": "<weather|wikipedia|time>", "topic": "<topic or city (if applicable)>" }}
No extra text.
"""

    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }
    # Use chat completion request compatible with OpenAI or OpenRouter
    payload = {
        "model": "gpt-3.5-turbo",
        "messages": [
            {"role": "system", "content": "You are a decision agent."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.0,
        "max_tokens": 200,
    }
    # If OPENAI_API_BASE is OpenRouter, use that base; else default OpenAI
    url = OPENAI_API_BASE.rstrip("/") + "/chat/completions"
    resp = requests.post(url, headers=headers, json=payload, timeout=20)
    resp.raise_for_status()
    data = resp.json()
    text = data["choices"][0]["message"]["content"]
    # Extract JSON from text (be forgiving)
    try:
        jstart = text.index("{")
        jend = text.rindex("}") + 1
        json_text = text[jstart:jend]
        decision = json.loads(json_text)
        return decision
    except Exception as e:
        # fallback - try to parse with regex
        print("Failed to parse JSON from LLM decision. Raw text:\n", text)
        raise

def call_mcp_tool(decision):
    tool = decision.get("tool")
    topic = decision.get("topic", "")
    headers = {"X-API-KEY": MCP_SERVER_KEY, "Content-Type": "application/json"}
    if tool == "weather":
        url = f"{MCP_SERVER_URL}/tool/weather"
        resp = requests.post(url, headers=headers, json={"city": topic}, timeout=10)
        resp.raise_for_status()
        return resp.json()
    elif tool == "wikipedia":
        url = f"{MCP_SERVER_URL}/tool/wikipedia"
        resp = requests.post(url, headers=headers, json={"topic": topic}, timeout=10)
        resp.raise_for_status()
        return resp.json()
    elif tool == "time":
        url = f"{MCP_SERVER_URL}/tool/time"
        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()
        return resp.json()
    else:
        return {"tool": "none", "output": "No tool selected."}

def call_llm_for_final_answer(user_query, tool_output):
    prompt = f"""
User question: {user_query}

Tool output:
{tool_output}

Using the tool output, answer the user's question clearly and concisely.
"""
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": "gpt-3.5-turbo",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant combining tool output."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.6,
        "max_tokens": 300,
    }
    url = OPENAI_API_BASE.rstrip("/") + "/chat/completions"
    resp = requests.post(url, headers=headers, json=payload, timeout=20)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()

if __name__ == "__main__":
    print("=== MCP Client Demo ===")
    user_query = input("Enter your question (examples: 'What's the weather in Chennai?', 'Tell me about Elon Musk'): ").strip()
    # 1) Ask LLM which tool to call
    decision = call_llm_for_decision(user_query)
    print("\nLLM decision:", json.dumps(decision, indent=2))
    # 2) Call MCP server tool
    tool_resp = call_mcp_tool(decision)
    print("\nTool response:", json.dumps(tool_resp, indent=2))
    # 3) Ask LLM to combine tool output + user query
    final = call_llm_for_final_answer(user_query, tool_resp.get("output", ""))
    print("\nFINAL ANSWER:\n", final)
