"""Pull the tool-call history out of a cassette's own message list.

Used by live_freeze.py/redo_plan.py right after a live diagnose/plan
conversation finishes: that conversation's own final cassette file already
holds the complete OpenAI-format message history, including every
run_probe/explain/list_indexes/... call and its exact result, verbatim.
This reads them back out rather than re-deriving anything.
"""

from __future__ import annotations

import json
from typing import Any


def extract_tool_calls(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """[{"tool", "args", "result"}] for every tool call/result pair, in order."""
    by_id: dict[str, dict[str, Any]] = {}
    out: list[dict[str, Any]] = []
    for message in messages:
        if message.get("role") == "assistant":
            for call in message.get("tool_calls") or []:
                function = call["function"]
                entry = {
                    "tool": function["name"],
                    "args": json.loads(function["arguments"]),
                }
                by_id[call["id"]] = entry
        elif message.get("role") == "tool":
            entry = by_id.get(message.get("tool_call_id"))
            if entry is not None and "result" not in entry:
                entry["result"] = message.get("content")
                out.append(entry)
    return out
