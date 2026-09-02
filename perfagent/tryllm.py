from perfagent.llm import GroqClient
import os
from dotenv import load_dotenv
load_dotenv()

c = GroqClient(api_key=os.environ["GROQ_API_KEY"])

TOOL = {"type": "function", "function": {
    "name": "get_weather",
    "description": "Get current weather in a city",
    "parameters": {"type": "object",
                   "properties": {"city": {"type": "string"}},
                   "required": ["city"]}}}

r = c.complete(messages=[{"role": "user", "content": "Weather in Chennai?"}],
               tools=[TOOL])

print("stop_reason:", r.stop_reason)
print("tool_uses:", r.tool_uses())
print("text:", r.text())
print("tokens:", r.usage.input_tokens, r.usage.output_tokens)
