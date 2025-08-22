import asyncio
import sys
from pathlib import Path

from mcp.client.session import ClientSession
from mcp.client.stdio import stdio_client, StdioServerParameters

SCRIPT = Path(__file__).with_name("server_tools.py")

async def main():
    params = StdioServerParameters(
        command=sys.executable,
        args=[str(SCRIPT)],
        cwd=str(SCRIPT.parent),
    )

    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            tools = await session.list_tools()
            print("TOOLS:", [t.name for t in tools.tools])
            
            # вызов math_eval
            res_math = await session.call_tool("math_eval", {"expr": "2 + 2 * 3"})
            if res_math.content:
                print("math_eval result:", res_math.content[0].text)
            else:
                print("math_eval empty response:", res_math)
            
            # вызов read_text
            res_text = await session.call_tool("read_text", {"relative_path": "sample.txt"})

            if res_text.content:
                print("read_text result:", res_text.content[0].text)
            else:
                print("read_text empty response:", res_text)
            

            # Ничего дополнительно вызывать не нужно — контекст-менеджер всё закроет.
            # Если очень хочется явно: await session.close()

if __name__ == "__main__":
    asyncio.run(main())


