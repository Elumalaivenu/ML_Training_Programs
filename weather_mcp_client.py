# weather_mcp_client.py
import asyncio
import os
from pprint import pprint

from dotenv import load_dotenv

from mcp.client.session import ClientSession
from mcp.client.sse import sse_client


def get_base_url() -> str:
    load_dotenv()
    return os.getenv("MCP_SERVER_URL", "http://127.0.0.1:8000")


async def main():
    base_url = get_base_url()
    sse_endpoint = f"{base_url.rstrip('/')}/sse"

    async with sse_client(sse_endpoint) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()

            tools_result = await session.list_tools()
            print("\n🧰 Available Tools:")
            pprint([tool.model_dump() for tool in tools_result.tools])

            resources_result = await session.list_resources()
            print("\n📦 Available Resources:")
            pprint([resource.model_dump() for resource in resources_result.resources])

            city = "Chennai"
            call_result = await session.call_tool("get_weather", {"city": city})
            print(f"\n🌦️ Weather in {city}:")
            pprint(call_result.model_dump())

            resource_result = await session.read_resource("resource://weather_info")
            print("\n📖 Resource Info:")
            pprint(resource_result.model_dump())


if __name__ == "__main__":
    asyncio.run(main())
