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

# r = c.complete(messages=[{"role": "user", "content": "Weather in Chennai?"}],
#                tools=[TOOL])

# print("stop_reason:", r.stop_reason)
# print("tool_uses:", r.tool_uses())
# print("text:", r.text())
# print("tokens:", r.usage.input_tokens, r.usage.output_tokens)


from perfagent.llm import GroqClient
from perfagent.agent import run_agent
from perfagent.tools import (read_file, search_code, emit_endpoint_model,
                             READ_FILE_TOOL, SEARCH_CODE_TOOL, EMIT_TOOL)

ENDPOINT = "POST /api/customers/search"      # ← your real endpoint

TASK = f"""Trace the execution path of this endpoint: {ENDPOINT}

Find the controller method that handles it, then follow the calls through
services and repositories down to the SQL or database access.

Use search_code to locate things, read_file to understand them.
Note: Spring routes may be split between a class-level @RequestMapping and a
method-level @GetMapping/@PostMapping — compose them.

When done, report:
- the full call chain, with file:line for each step
- every database query or repository method involved
- anything you were unsure about

When you have traced everything, call emit_endpoint_model. Do not write a prose report

Cite file:line for every claim."""

c = GroqClient(api_key=os.environ["GROQ_API_KEY"])

answer = run_agent(
    client=c,
    task=TASK,
    tools=[READ_FILE_TOOL, SEARCH_CODE_TOOL, EMIT_TOOL],
    handlers={"read_file": read_file, "search_code": search_code, "emit_endpoint_model": emit_endpoint_model},
    max_steps=25,
)

from perfagent.tools import _resolve

print("This is the answer")
print(answer)

import json, pathlib
pathlib.Path(".perfagent/endpoint_model.json").write_text(json.dumps(answer, indent=2))