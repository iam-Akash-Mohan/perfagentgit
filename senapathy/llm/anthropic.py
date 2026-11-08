"""The Anthropic Messages API.

Two things here are not cosmetic differences from the OpenAI shape:

  * `system` is a top-level field, not a message. That is what makes prompt
    caching possible, because the cached prefix is addressable.
  * tool results all travel in ONE user message. Sending them as separate
    messages is a 400, not a style preference.
"""

from __future__ import annotations

from typing import Any

from ..contracts import LLMResponse, ToolCallResult, Usage
from .base import HTTPLLMClient

API_VERSION = "2023-06-01"
DEFAULT_BASE_URL = "https://api.anthropic.com/v1"

CACHE_CONTROL: dict[str, str] = {"type": "ephemeral"}


class AnthropicClient(HTTPLLMClient):
    provider = "anthropic"

    def default_base_url(self) -> str:
        return DEFAULT_BASE_URL

    def endpoint(self) -> str:
        return f"{self.base_url.rstrip('/')}/messages"

    def headers(self) -> dict[str, str]:
        return {
            "content-type": "application/json",
            "anthropic-version": API_VERSION,
            "x-api-key": self.api_key or "",
        }

    def build_body(
        self, *, system: str | list[dict[str, Any]] | None,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        tool_choice: dict[str, Any] | None,
        max_tokens: int,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": messages,
        }
        if system:
            body["system"] = _cached_system(system)
        if tools:
            body["tools"] = _cached_tools(tools)
        if tool_choice:
            body["tool_choice"] = tool_choice
        return body

    def parse(self, raw: dict[str, Any], latency_ms: float,
              from_cassette: bool) -> LLMResponse:
        content = list(raw.get("content") or [])

        raw_usage = raw.get("usage") or {}
        usage = Usage(
            input_tokens=int(raw_usage.get("input_tokens", 0) or 0),
            output_tokens=int(raw_usage.get("output_tokens", 0) or 0),
            cache_read_tokens=int(raw_usage.get("cache_read_input_tokens", 0) or 0),
            cache_write_tokens=int(
                raw_usage.get("cache_creation_input_tokens", 0) or 0),
        )
        usage.usd = self.price(usage)

        return LLMResponse(
            content=content,
            stop_reason=raw.get("stop_reason"),
            usage=usage,
            latency_ms=latency_ms,
            from_cassette=from_cassette,
            raw_message={"role": "assistant", "content": content},
            model=raw.get("model") or self.model,
        )

    def forced_tool_choice(self, name: str) -> dict[str, Any]:
        return {"type": "tool", "name": name}

    def tool_result_messages(
        self, results: list[ToolCallResult]
    ) -> list[dict[str, Any]]:
        if not results:
            return []
        return [{
            "role": "user",
            "content": [
                {"type": "tool_result",
                 "tool_use_id": result.id,
                 "content": result.content,
                 "is_error": result.is_error}
                for result in results
            ],
        }]


def _cached_system(system: str | list[dict[str, Any]]) -> list[dict[str, Any]]:
    """System prompt as blocks, with the last one marked cacheable.

    The system prompt is byte-identical on every turn of a loop. Not caching
    it means paying full input price for the same tokens thirty times — the
    single biggest avoidable cost in an agent run.
    """
    blocks: list[dict[str, Any]] = (
        [{"type": "text", "text": system}] if isinstance(system, str)
        else [dict(block) for block in system]
    )
    if blocks:
        blocks[-1] = {**blocks[-1], "cache_control": CACHE_CONTROL}
    return blocks


def _cached_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Tool schemas, with cache_control on the last one.

    A cache breakpoint covers everything before it, so one marker on the final
    tool caches the whole tool block. Marking every tool would burn four of
    the four available breakpoints for no extra benefit.
    """
    wire = [
        {"name": t["name"],
         "description": t["description"],
         "input_schema": t["parameters"]}
        for t in tools
    ]
    if wire:
        wire[-1] = {**wire[-1], "cache_control": CACHE_CONTROL}
    return wire
