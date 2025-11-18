# mcp_langchain_agent.py
import os
import json
import asyncio
from typing import Any

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import initialize_agent
from langchain.tools import tool

from mcp.client.session import ClientSession
from mcp.client.sse import sse_client


load_dotenv()

load_dotenv()

# --- LLM Setup ---
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.3)

# --- MCP Client Setup ---
MCP_URL = os.getenv("MCP_SERVER_URL", "http://127.0.0.1:8000")


async def _call_mcp_tool(tool_name: str, arguments: dict[str, Any] | None = None) -> Any:
    """Call an MCP tool via a short-lived SSE session."""
    sse_endpoint = f"{MCP_URL.rstrip('/')}/sse"
    async with sse_client(sse_endpoint) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            result = await session.call_tool(tool_name, arguments or {})
            if result.isError:
                raise RuntimeError(f"Tool {tool_name} returned an error: {result.content}")
            text_blocks = [block.text for block in result.content if getattr(block, "type", None) == "text"]
            return "\n".join(text_blocks)


async def _read_mcp_resource(uri: str) -> Any:
    """Read an MCP resource via SSE session."""
    sse_endpoint = f"{MCP_URL.rstrip('/')}/sse"
    async with sse_client(sse_endpoint) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            result = await session.read_resource(uri)
            contents = []
            for item in result.contents:
                if item.text is not None:
                    contents.append(item.text)
            return "\n".join(contents)

# --- Tool Wrappers ---
@tool("get_weather", return_direct=True)
def get_weather(city: str) -> str:
    """Get weather for a city via MCP server"""
    response = asyncio.run(_call_mcp_tool("get_weather", {"city": city}))
    if not response:
        return "No weather information returned."
    try:
        data = json.loads(response)
        if isinstance(data, dict):
            city_name = data.get("city", city)
            temperature = data.get("temperature", "?")
            condition = data.get("condition", "?")
            return f"Weather in {city_name}: {temperature}, {condition}"
    except json.JSONDecodeError:
        pass
    return response

@tool("get_resource_info", return_direct=True)
def get_resource_info() -> str:
    """Fetch static info from MCP server"""
    response = asyncio.run(_read_mcp_resource("resource://weather_info"))
    return response or "No resource information returned."

# --- Agent Setup ---
tools = [get_weather, get_resource_info]
agent = initialize_agent(
    tools,
    llm,
    agent_type="zero-shot-react-description",
    verbose=True,
)

# --- Simple CLI Interface ---
print("🤖 LangChain MCP Agent ready! Ask about the weather or MCP info.\n")
while True:
    user_input = input("🧑‍💻 You: ")
    if user_input.lower() in ["exit", "quit"]:
        break
    response = agent.invoke({"input": user_input})
    print("🤖 AI:", response["output"])
