from perfagent.llm import GroqClient
import os, time

def run_agent(client, task: str, tools: list[dict],
              handlers: dict[str, callable], max_steps: int = 10) -> str:
    messages = [{"role": "user", "content": task}]

    for step in range(max_steps):
        resp = client.complete(messages=messages, tools=tools)
        time.sleep(2)
        print(step, resp.stop_reason)

        if resp.stop_reason != "tool_use":
            return resp.text()
        
        messages.append(client.last_raw_message) 

        if resp.tool_uses():
            for tool in resp.tool_uses():
                handler = handlers[tool["name"]]
                if callable(handler):
                    result = handler(**tool["input"])
                    messages.append(
                        {"role":"tool", "tool_call_id":tool["id"], "content": result}
                    )
                    
    return "hit max_steps"

TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather in a city",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name, e.g. Chennai"}
            },
            "required": ["city"],
        },
    },
}


def get_weather(city: str) -> str:
    return {"Chennai": "31C, humid", "Bangalore": "24C, pleasant"}.get(city, "unknown")

c = GroqClient(api_key=os.environ["GROQ_API_KEY"])

answer = run_agent(
    client=c,
    task="What's the weather in Chennai and Bangalore? Compare them.",
    tools=[TOOL],
    handlers={"get_weather": get_weather},
)

print(answer)