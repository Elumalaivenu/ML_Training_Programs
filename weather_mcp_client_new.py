# weather_mcp_client_new.py
import asyncio
import json
import os
from typing import Iterable

from mcp.client.session import ClientSession
from mcp.client.sse import sse_client


def _extract_text(blocks: Iterable) -> str:
    """Pull text fields out of a sequence of MCP content blocks."""
    texts = []
    for block in blocks:
        if getattr(block, "type", None) == "text":
            texts.append(block.text)
    return "\n".join(texts)


async def main() -> None:
    base_url = os.getenv("MCP_SERVER_URL", "http://127.0.0.1:8000")
    sse_endpoint = f"{base_url.rstrip('/')}/sse"

    async with sse_client(sse_endpoint) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()

            tools_result = await session.list_tools()
            print("🔧 Available Tools:")
            for tool in tools_result.tools:
                print(f" - {tool.name}: {tool.description}")

            resources_result = await session.list_resources()
            print("\n📚 Available Resources:")
            for resource in resources_result.resources:
                print(f" - {resource.uri}: {resource.description}")

            city_name = "Chennai"
            print(f"\n🌤️ Getting weather for {city_name}...")
            weather_result = await session.call_tool("get_weather", {"city": city_name})

            output = None
            if weather_result.structuredContent is not None:
                output = json.dumps(weather_result.structuredContent, indent=2)
            elif weather_result.content:
                output = _extract_text(weather_result.content)
            print("✅ Weather Result:")
            print(output or "(no output returned)")

            print("\nℹ️ Fetching resource://weather_info ...")
            resource_result = await session.read_resource("resource://weather_info")
            resource_text = []
            for item in resource_result.contents:
                if item.text is not None:
                    resource_text.append(item.text)
            print("Resource Info:")
            print("\n".join(resource_text) or "(no resource content)")


if __name__ == "__main__":
    asyncio.run(main())
