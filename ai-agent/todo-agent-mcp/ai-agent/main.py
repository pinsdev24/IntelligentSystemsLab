from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_mcp_adapters.client import MultiServerMCPClient
from dotenv import load_dotenv
import os
import asyncio

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in environment variables")

client = MultiServerMCPClient(
    {
        "todomcp": {
            "transport": "stdio",
            "command": "/Library/Frameworks/Python.framework/Versions/3.13/bin/uv",
            "args": [
                "--directory",
                "/Volumes/CrucialX9/courses/AI/todo-agent-mcp/todo-mcp",
                "run", 
                "server.py"
            ],
        }
    }
)


tools = asyncio.run(client.get_tools())

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

agent = create_agent(
    model,
    tools,
)

# response = asyncio.run(agent.ainvoke(
#     {
#         "messages": [
#             {
#                 "role": "user", 
#                 "content": "List all my pending tasks"
#             }
#         ]
#     }
# ))

# print(response['messages'][-1]['content'])