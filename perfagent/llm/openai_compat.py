"""Cerebras, Groq and Ollama. All three speak OpenAI's chat-completions shape.

One class, three base URLs. The differences between these providers are
configuration; a subclass per provider would be three copies of the same file.
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

        for call in message.get("tool_calls") or []:
            content.append({
                "type": "tool_use",
                "id": call["id"],
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
