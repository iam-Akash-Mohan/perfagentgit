"""Cerebras, Groq, Gemini and Ollama — all speak OpenAI chat-completions.

One class, four base URLs. The differences between these providers are
configuration; a subclass per provider would be four copies of the same file.

## Google's compatibility layer, against Cerebras as the reference

Google serves an OpenAI-shaped endpoint at `/v1beta/openai/`. It is close, not
identical, and these are the differences that reach this file. Everything
below was read off Google's published compatibility documentation; the ones
marked UNVERIFIED have not been exercised against a live key here, and the
cassette test in `tests/test_llm.py` pins the shape we believe we will get.

  * **The trailing slash is load-bearing.** The base URL ends in `/openai/`,
    and `endpoint()` strips it before appending, so the request goes to
    `.../v1beta/openai/chat/completions`.

  * **Tool arguments are a JSON string**, as with every other provider here,
    so `_arguments` applies unchanged. Gemini's NATIVE API returns an object
    instead — if a future change talks to `v1beta/models/...` directly, that
    assumption breaks.

  * **`tool_calls[].id` is not guaranteed.** OpenAI and Cerebras always send
    one; Google's layer has been observed omitting it. `parse` therefore
    synthesises an id rather than raising `KeyError` three frames away from
    anything that explains it. A synthesised id is echoed back in the
    `tool_call_id` of the result message, which is the correlation key, so it
    must be stable within one response — it is, being derived from position.

  * **`max_tokens` is accepted** (as the alias for `max_completion_tokens`),
    which is what `build_body` sends.

  * **Thinking tokens count as output.** `gemini-2.5-flash` reasons before it
    answers and bills that to `completion_tokens`, so a tight `max_tokens`
    buys a reply with no content and no tool call. That is the same failure
    this repo already hit on `gpt-oss-120b`; the loop reads `stop_reason ==
    "max_tokens"` and says so rather than reporting a give-up.

  * **UNVERIFIED: forced tool choice.** `forced_tool_choice` sends OpenAI's
    named-function form. Google documents `"auto"`, `"none"` and `"required"`;
    whether it honours a named function is not something this file can prove
    without a key. If it does not, the terminal-tool nudge in `agent.py`
    degrades from "call THIS tool" to an ordinary turn — the loop still
    works, it just loses one lever.

  * **UNVERIFIED: no prompt caching fields.** `usage` carries
    `prompt_tokens` / `completion_tokens` and no cache counters, so
    `cache_read_tokens` and `cache_write_tokens` stay zero and the dollar
    figure is the uncached one.
"""

from __future__ import annotations

import json
from typing import Any

from ..contracts import LLMResponse, ToolCallResult, Usage
from .base import HTTPLLMClient

# OpenAI's finish_reason vocabulary, mapped onto Anthropic's stop_reason, so
# the agent loop has one word for "it wants a tool" regardless of provider.
STOP_MAP = {
    "tool_calls": "tool_use",
    "stop": "end_turn",
    "length": "max_tokens",
    "content_filter": "content_filter",
    "function_call": "tool_use",
}

BASE_URLS = {
    "cerebras": "https://api.cerebras.ai/v1",
    "groq": "https://api.groq.com/openai/v1",
    "gemini": "https://generativelanguage.googleapis.com/v1beta/openai/",
    "ollama": "http://localhost:11434/v1",
}


class OpenAICompatClient(HTTPLLMClient):
    """An OpenAI-shaped chat-completions provider."""

    def __init__(self, *, provider: str = "cerebras", **kwargs: Any) -> None:
        if provider not in BASE_URLS:
            raise ValueError(
                f"unknown OpenAI-compatible provider {provider!r}; "
                f"known: {', '.join(sorted(BASE_URLS))}"
            )
        self.provider = provider
        super().__init__(**kwargs)

    def default_base_url(self) -> str:
        return BASE_URLS[self.provider]

    def endpoint(self) -> str:
        return f"{self.base_url.rstrip('/')}/chat/completions"

    def headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def build_body(
        self, *, system: str | list[dict[str, Any]] | None,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        tool_choice: dict[str, Any] | None,
        max_tokens: int,
    ) -> dict[str, Any]:
        wire: list[dict[str, Any]] = []
        if system:
            wire.append({"role": "system", "content": _system_text(system)})
        wire.extend(messages)

        body: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": wire,
        }
        # Omit rather than send null: some providers reject `"tools": null`,
        # and every provider charges for the bytes.
        if tools:
            body["tools"] = [
                {"type": "function",
                 "function": {"name": t["name"],
                              "description": t["description"],
                              "parameters": t["parameters"]}}
                for t in tools
            ]
        if tool_choice:
            body["tool_choice"] = tool_choice
        return body

    def parse(self, raw: dict[str, Any], latency_ms: float,
              from_cassette: bool) -> LLMResponse:
        choice = raw["choices"][0]
        message = choice["message"]

        content: list[dict[str, Any]] = []
        if message.get("content"):
            content.append({"type": "text", "text": message["content"]})

        for position, call in enumerate(message.get("tool_calls") or []):
            content.append({
                "type": "tool_use",
                # Not call["id"]: Google's layer may omit it, and a KeyError
                # here would surface as a dead phase rather than as a provider
                # quirk. Position is stable within one response, which is all
                # the correlation key has to be.
                "id": call.get("id") or f"{self.provider}_call_{position}",
                "name": call["function"]["name"],
                "input": _arguments(call["function"].get("arguments")),
            })

        raw_usage = raw.get("usage") or {}
        usage = Usage(
            input_tokens=int(raw_usage.get("prompt_tokens", 0) or 0),
            output_tokens=int(raw_usage.get("completion_tokens", 0) or 0),
        )
        usage.usd = self.price(usage)

        finish = choice.get("finish_reason")
        return LLMResponse(
            content=content,
            stop_reason=STOP_MAP.get(str(finish), finish),
            usage=usage,
            latency_ms=latency_ms,
            from_cassette=from_cassette,
            raw_message=message,
            model=raw.get("model") or self.model,
        )

    def forced_tool_choice(self, name: str) -> dict[str, Any]:
        return {"type": "function", "function": {"name": name}}

    def tool_result_messages(
        self, results: list[ToolCallResult]
    ) -> list[dict[str, Any]]:
        """One message per call. There is no is_error flag in this protocol,
        so the error is stated in the text the model reads."""
        return [
            {"role": "tool",
             "tool_call_id": result.id,
             "name": result.name,
             "content": (f"ERROR: {result.content}" if result.is_error
                         else result.content)}
            for result in results
        ]


def _system_text(system: str | list[dict[str, Any]]) -> str:
    """Anthropic-style system blocks flattened for a provider without them."""
    if isinstance(system, str):
        return system
    return "\n\n".join(block.get("text", "") for block in system)


def _arguments(raw: Any) -> dict[str, Any]:
    """Tool arguments arrive as a JSON string. A malformed one is the model's
    mistake, and the loop must be able to tell it so rather than crash."""
    if isinstance(raw, dict):
        return raw
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {"__malformed_arguments__": str(raw)}
    return parsed if isinstance(parsed, dict) else {"__malformed_arguments__": str(raw)}
