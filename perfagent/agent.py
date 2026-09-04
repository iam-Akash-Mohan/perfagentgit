from perfagent.llm import GroqClient
import os, time

# ─────────────────────────────────────────────────────────────────────────
# DEBUG SCAFFOLDING — transcript printing only. Delete this block (and the
# _say/_block calls in run_agent) once P02 emits real trace events.
# CLAUDE.md bans print() in library code; this is temporary.
# ─────────────────────────────────────────────────────────────────────────

WIDTH = 78
SHOW_RESULT_LINES = 20      # display cap ONLY — the model still gets the full string


def _rule(char: str = "─") -> None:
    print(char * WIDTH)


def _say(who: str, what: str) -> None:
    """One labelled speech act in the conversation."""
    print()
    print(f"{who}")
    _rule("·")
    print(_indent(what))


def _indent(text: str, pad: str = "  ") -> str:
    return "\n".join(pad + l for l in str(text).splitlines()) or pad + "(empty)"


def _clip(text: str) -> str:
    """Shorten a long tool result for the transcript, never for the model."""
    lines = str(text).splitlines()
    if len(lines) <= SHOW_RESULT_LINES:
        return text
    hidden = len(lines) - SHOW_RESULT_LINES
    return "\n".join(lines[:SHOW_RESULT_LINES]) + f"\n[... {hidden} more lines sent to the model, hidden here ...]"


def _call_sig(name: str, args: dict) -> str:
    inner = ", ".join(f"{k}={v!r}" for k, v in args.items())
    return f"{name}({inner})"


# ─────────────────────────────────────────────────────────────────────────


def run_agent(client, task: str, tools: list[dict],
              handlers: dict[str, callable], max_steps: int = 10) -> str:
    messages = [{"role": "user", "content": task}]

    print()
    _rule("═")
    print(" CONVERSATION START")
    print(f" tools offered to the model: {', '.join(t['function']['name'] for t in tools)}")
    _rule("═")
    _say("YOU → MODEL   (the task)", task)

    for step in range(max_steps):
        resp = client.complete(messages=messages, tools=tools)

        print()
        print()
        _rule("═")
        print(f" STEP {step}   ·   stop_reason={resp.stop_reason}"
              f"   ·   tokens in={resp.usage.input_tokens} out={resp.usage.output_tokens}")
        _rule("═")

        # the model's private thinking, if the provider returned any
        raw = getattr(client, "last_raw_message", {}) or {}
        if raw.get("reasoning"):
            _say("MODEL (thinking to itself)", raw["reasoning"])

        if resp.stop_reason != "tool_use":
            _say("MODEL → YOU   (final answer)", resp.text())
            print()
            _rule("═")
            print(f" CONVERSATION END   ·   {step + 1} model turns")
            _rule("═")
            return resp.text()

        messages.append(client.last_raw_message)

        if resp.tool_uses():
            for tool in resp.tool_uses():
                if tool["name"] == "emit_endpoint_model":
                    return tool["input"]  
                _say("MODEL → YOU   (asking you to run a tool)",
                     _call_sig(tool["name"], tool["input"]))
                handler = handlers[tool["name"]]
                if callable(handler):

                    try:
                        handler = handlers[tool["name"]]
                        result = handler(**tool["input"])
                    except KeyError:
                        result = (f"No such tool '{tool['name']}'. "
                                f"Available tools: {', '.join(handlers)}")
                    except TypeError as e:
                        result = f"Bad arguments for {tool['name']}: {e}"
                    except Exception as e:
                        result = f"{tool['name']} failed: {type(e).__name__}: {e}"

                    _say(f"YOU → MODEL   (tool result · {len(str(result))} chars)",
                         _clip(result))
                    messages.append(
                        {"role":"tool", "tool_call_id":tool["id"], "content": result}
                    )
        print()
        print(f"  (waiting 20s · conversation is now {len(messages)} messages)")
        print()
        print()
        time.sleep(20)

    print()
    _rule("═")
    print(f" CONVERSATION END   ·   hit max_steps ({max_steps})")
    _rule("═")
    return "hit max_steps"

# TOOL = {
#     "type": "function",
#     "function": {
#         "name": "get_weather",
#         "description": "Get the current weather in a city",
#         "parameters": {
#             "type": "object",
#             "properties": {
#                 "city": {"type": "string", "description": "City name, e.g. Chennai"}
#             },
#             "required": ["city"],
#         },
#     },
# }


# def get_weather(city: str) -> str:
#     return {"Chennai": "31C, humid", "Bangalore": "24C, pleasant"}.get(city, "unknown")

# c = GroqClient(api_key=os.environ["GROQ_API_KEY"])

# answer = run_agent(
#     client=c,
#     task="What's the weather in Chennai and Bangalore? Compare them.",
#     tools=[TOOL],
#     handlers={"get_weather": get_weather},
# )

# print(answer)