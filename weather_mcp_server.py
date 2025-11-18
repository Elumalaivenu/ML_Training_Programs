# weather_mcp_server.py
from mcp.server.fastmcp import FastMCP
from starlette.requests import Request
from starlette.responses import JSONResponse
import random

# Spin up a FastMCP server instance that exposes HTTP endpoints via SSE transport.
server = FastMCP(name="Weather MCP Server", host="127.0.0.1", port=8000)


# Register a simple tool that returns mocked weather data.
@server.tool("get_weather", description="Get weather details for a city")
async def get_weather(city: str):
    """Return a simulated weather report."""
    temp = random.randint(20, 40)
    return {
        "city": city,
        "temperature": f"{temp}°C",
        "condition": random.choice(["Sunny", "Cloudy", "Rainy"])
    }


# Expose a static resource describing the server.
@server.resource("resource://weather_info", description="Basic weather data info")
async def weather_info():
    return {
        "data": "This server provides fake weather data for demonstration purposes."
    }


# Provide a small landing page for convenience when browsing the base URL.
@server.custom_route("/", methods=["GET"], name="root")
async def root(_: Request):
    return JSONResponse(
        {
            "message": "Weather MCP Server is running.",
            "tools": ["get_weather"],
            "resources": ["resource://weather_info"],
            "sse_endpoint": server.settings.sse_path,
        }
    )


if __name__ == "__main__":
    server.run(transport="sse")
