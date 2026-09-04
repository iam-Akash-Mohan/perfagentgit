"""The conversation, rendered for a human, from trace events only.

This used to live inside the agent loop as a pile of `print` calls. It is
genuinely useful — watching a model pick tools is how you learn what your tool
descriptions actually say — so it is kept, moved out of library code, and
driven by the trace so that what you watch live and what you read afterwards
cannot disagree.
"""

from __future__ import annotations

from typing import Any, Callable

from ..trace import Trace

WIDTH = 78
RESULT_LINES = 20      # display cap ONLY — the model always gets the full string


class TranscriptRenderer:
    """Turns trace events into the transcript. Stateless apart from the sink."""

    def __init__(self, echo: Callable[[str], None] = print,
                 *, result_lines: int = RESULT_LINES) -> None:
        self.echo = echo
        self.result_lines = result_lines

    # ── event handling ───────────────────────────────────────────────────

    def on_event(self, event: dict[str, Any]) -> None:
        handler = getattr(self, f"_on_{event.get('type', '')}", None)
        if handler is not None:
            handler(event)

    def _on_llm_request(self, event: dict[str, Any]) -> None:
        if event.get("step") == 0:
            self._rule("═")
            self.echo(" CONVERSATION START")
            tools = ", ".join(event.get("tools") or [])
            self.echo(f" tools offered to the model: {tools}")
            self._rule("═")

    def _on_llm_response(self, event: dict[str, Any]) -> None:
        self.echo("")
        self._rule("═")
        cached = "  ·   FROM CASSETTE" if event.get("from_cassette") else ""
        self.echo(
            f" STEP {event.get('step')}   ·   stop_reason={event.get('stop_reason')}"
            f"   ·   tokens in={event.get('input_tokens')} "
            f"out={event.get('output_tokens')}"
            f"   ·   ${event.get('usd', 0):.4f}"
            f"   ·   {event.get('duration_ms', 0):.0f} ms{cached}"
        )
        self._rule("═")
        if event.get("reasoning"):
            self._say("MODEL (thinking to itself)", event["reasoning"])
        if event.get("text"):
            self._say("MODEL → YOU", event["text"])

    def _on_tool_call(self, event: dict[str, Any]) -> None:
        self._say("MODEL → YOU   (asking you to run a tool)",
                  f"{event.get('tool')}({event.get('args', '')})")

    def _on_tool_result(self, event: dict[str, Any]) -> None:
        if event.get("guardrail"):
            label = "YOU → MODEL   (REFUSED by a guardrail)"
        elif event.get("is_error"):
            label = "YOU → MODEL   (tool error)"
        else:
            label = f"YOU → MODEL   (tool result · {event.get('bytes', 0)} chars)"
        self._say(label, self._clip(str(event.get("preview", ""))))

    def _on_guardrail_triggered(self, event: dict[str, Any]) -> None:
        self._say("GUARDRAIL", f"{event.get('tool')}: {event.get('reason')}")

    def _on_note(self, event: dict[str, Any]) -> None:
        if event.get("note"):
            self._say("LOOP", str(event["note"]))

    # ── formatting ───────────────────────────────────────────────────────

    def _rule(self, char: str = "─") -> None:
        self.echo(char * WIDTH)

    def _say(self, who: str, what: str) -> None:
        self.echo("")
        self.echo(who)
        self._rule("·")
        self.echo(_indent(what))

    def _clip(self, text: str) -> str:
        lines = text.splitlines()
        if len(lines) <= self.result_lines:
            return text
        hidden = len(lines) - self.result_lines
        return ("\n".join(lines[:self.result_lines])
                + f"\n[... {hidden} more lines sent to the model, hidden here ...]")


def _indent(text: str, pad: str = "  ") -> str:
    return "\n".join(pad + line for line in str(text).splitlines()) or pad + "(empty)"


def attach_transcript(trace: Trace,
                      echo: Callable[[str], None] = print) -> TranscriptRenderer:
    """Subscribe a renderer to a trace. This is what --verbose does."""
    renderer = TranscriptRenderer(echo)
    trace.add_listener(renderer.on_event)
    return renderer
