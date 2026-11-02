"""The agent loop: one conversation, many tools, a typed outcome.

Three properties matter more than anything else here.

  * It is **reusable**. The terminal tool is a parameter, not a hardcoded
    name, so every phase gets the same loop rather than a copy of it.
  * It **cannot be killed by a tool**. Every dispatch failure becomes a tool
    result the model can read and act on. A hallucinated tool name is a
    correctable mistake, not a crash.
  * It **stops for a stated reason**. Budget, no-progress and completion are
    different outcomes with different names, and all three carry whatever was
    proven before the stop.

There is no `print` and no `sleep` in this file. Rate limiting lives in the
client's retry policy, where it can be conditional on an actual 429 instead of
costing twenty seconds a turn unconditionally.
"""

from __future__ import annotations

import json
import time
import warnings
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Callable, Iterable

from .contracts import LLMResponse, ToolCallResult, Usage
from .errors import GuardrailError, ToolError
from .trace import EventType, Trace

# Three identical dispatches in a row is a loop, not deliberation. Two is a
# retry after a rejection, which is exactly what a good error message asks for.
NO_PROGRESS_REPEATS = 3

DEFAULT_MAX_STEPS = 20

# What the loop says when a model writes its artifact as prose instead of
# calling the tool that accepts one. Weaker models do this often, and the
# whole phase is worthless when they do.
FORCE_TERMINAL_NUDGE = (
    "You stopped without calling {tool}. Prose is not the output of this "
    "phase — {tool} is, and it is the only way to finish. Call it now with "
    "the analysis you just described. If a field is missing, use the tools "
    "again to find it first."
)

# How many times the loop will force the terminal call. More than one matters:
# a forced call that is REJECTED by its validator is progress, not failure, and
# the model must be able to be pushed over the line again after fixing it.
MAX_TERMINAL_FORCES = 3

# A reply cut off at the token ceiling is a TRUNCATED reply, not a finished
# one. Reasoning models spend their first thousand-odd tokens thinking, so a
# tight max_tokens produces a response with no content and no tool call —
# indistinguishable from "the model gave up" unless stop_reason is read.
TRUNCATED_NUDGE = (
    "Your last reply was cut off at the output token limit before you "
    "produced anything usable. Reply again, and keep it short: make one tool "
    "call and nothing else."
)
MAX_TRUNCATIONS = 3

# A tool call the PROVIDER refused before it reached us. The loop already
# absorbs a bad call it can see; this is the same mistake arriving as an HTTP
# error, and it gets the same treatment — tell the model what was wrong and
# ask again. Bounded, because a model that cannot read a schema after three
# corrections is not going to on the fourth.
MAX_TOOL_CORRECTIONS = 3


class LoopStatus(StrEnum):
    COMPLETED = "completed"
    BUDGET_EXHAUSTED = "budget_exhausted"
    NO_PROGRESS = "no_progress"
    ERROR = "error"


@dataclass
class LoopBudget:
    """A ceiling on one agent conversation.

    `None` means unlimited and `0` means zero. They are not the same thing,
    and conflating them once let a run given no budget at all run forever.
    """

    max_steps: int | None = DEFAULT_MAX_STEPS
    max_usd: float | None = None
    max_wall_s: float | None = None
    started_at: float = field(default_factory=time.monotonic)

    def elapsed_s(self) -> float:
        return time.monotonic() - self.started_at

    def exceeded(self, *, steps: int, usd: float) -> str | None:
        """The reason this conversation must stop, or None."""
        if self.max_steps is not None and steps >= self.max_steps:
            return f"reached the step ceiling of {self.max_steps}"
        if self.max_usd is not None and usd >= self.max_usd:
            return f"spent ${usd:.4f} of the ${self.max_usd:.4f} ceiling"
        if self.max_wall_s is not None and self.elapsed_s() >= self.max_wall_s:
            return (f"ran for {self.elapsed_s():.0f}s of the "
                    f"{self.max_wall_s:.0f}s ceiling")
        return None


@dataclass
class LoopOutcome:
    """Everything one conversation produced, and why it stopped."""

    status: LoopStatus
    terminal_payload: dict[str, Any] | None
    final_text: str
    steps: int
    usage: Usage
    messages: list[dict[str, Any]]
    stopped_because: str = ""

    def __str__(self) -> str:
        # Callers written against the old `-> str` signature interpolate the
        # return value as the model's last words. Keep that true.
        return self.final_text

    @property
    def completed(self) -> bool:
        return self.status is LoopStatus.COMPLETED


def run_agent(
    *,
    client: Any,
    task: str,
    tools: list[dict[str, Any]],
    handlers: dict[str, Callable[..., Any]],
    system: str | list[dict[str, Any]] | None = None,
    terminal_tool: str | None = None,
    budget: LoopBudget | None = None,
    trace: Trace | None = None,
    phase: str = "agent",
    max_steps: int | None = None,
) -> LoopOutcome:
    """Run one tool-using conversation to a stated conclusion.

    `terminal_tool` is the name of the call that ends the phase. When its
    handler SUCCEEDS the loop returns immediately with the payload. When it
    raises `ToolError` the loop keeps going: a rejected artifact must be
    retryable, and a validator that ends the run on its first complaint is a
    validator the model can never satisfy.

    `max_steps` is deprecated; it is accepted so that callers written against
    the pre-S1 signature keep working, and folded into a `LoopBudget`.
    """
    if max_steps is not None:
        warnings.warn(
            "run_agent(max_steps=...) is deprecated; pass "
            "budget=LoopBudget(max_steps=..., max_usd=..., max_wall_s=...). "
            "A step count is only one of three ways a loop can overrun.",
            DeprecationWarning, stacklevel=2,
        )
    if budget is None:
        budget = LoopBudget(max_steps=max_steps if max_steps is not None
                            else DEFAULT_MAX_STEPS)

    messages: list[dict[str, Any]] = [{"role": "user", "content": task}]
    total = Usage()
    final_text = ""
    steps = 0
    last_signature: str | None = None
    repeats = 0
    force_next: str | None = None
    forces = 0
    truncations = 0
    corrections = 0

    def emit(kind: EventType, **fields: Any) -> None:
        if trace is not None:
            trace.emit(kind, phase=phase, **fields)

    while True:
        stop = budget.exceeded(steps=steps, usd=total.usd)
        if stop is not None:
            emit(EventType.NOTE, note=f"loop stopped: {stop}", steps=steps)
            return LoopOutcome(LoopStatus.BUDGET_EXHAUSTED, None, final_text,
                               steps, total, messages, stop)

        choice = _tool_choice(client, force_next)
        emit(EventType.LLM_REQUEST, step=steps, messages=len(messages),
             tools=[t for t in _tool_names(tools)], forcing=force_next)
        try:
            response = client.complete(system=system, messages=messages,
                                       tools=tools, tool_choice=choice)
        except Exception as exc:  # noqa: BLE001 — reported, never propagated
            correction = _correction_for(client, exc)
            if correction is not None and corrections < MAX_TOOL_CORRECTIONS:
                corrections += 1
                messages.append({"role": "user", "content": correction})
                emit(EventType.NOTE, step=steps,
                     note=f"the provider rejected a malformed tool call "
                          f"({corrections}/{MAX_TOOL_CORRECTIONS}); asking "
                          f"again with the schema error")
                continue
            emit(EventType.NOTE, note=f"LLM call failed: "
                 f"{type(exc).__name__}: {exc}", step=steps)
            return LoopOutcome(LoopStatus.ERROR, None, final_text, steps,
                               total, messages,
                               f"{type(exc).__name__}: {exc}")

        steps += 1
        total = _add(total, response.usage)
        emit(EventType.LLM_RESPONSE, step=steps,
             stop_reason=response.stop_reason,
             input_tokens=response.usage.input_tokens,
             output_tokens=response.usage.output_tokens,
             cache_read_tokens=response.usage.cache_read_tokens,
             cache_write_tokens=response.usage.cache_write_tokens,
             usd=round(response.usage.usd, 6),
             duration_ms=round(response.latency_ms, 1),
             from_cassette=response.from_cassette,
             text=response.text(),
             reasoning=_reasoning(response),
             tool_calls=[u["name"] for u in response.tool_uses()])

        text = response.text()
        if text:
            final_text = text

        force_next = None
        calls = response.tool_uses()
        if not calls:
            if response.stop_reason == "max_tokens":
                truncations += 1
                if truncations > MAX_TRUNCATIONS:
                    reason = (f"{truncations} replies in a row were cut off at "
                              f"the output token limit. Raise llm.max_tokens.")
                    emit(EventType.NOTE, step=steps, note=reason)
                    return LoopOutcome(LoopStatus.ERROR, None, final_text,
                                       steps, total, messages, reason)
                _append_assistant(messages, response)
                messages.append({"role": "user", "content": TRUNCATED_NUDGE})
                emit(EventType.NOTE, step=steps,
                     note="the reply was truncated at the token limit; asking "
                          "again (this is not the model giving up)")
                continue

            # A model that describes its artifact instead of submitting it has
            # done the work and failed the handshake. Push it over the line
            # with the provider's own forcing mechanism before calling it a
            # give-up.
            if terminal_tool is not None and forces < MAX_TERMINAL_FORCES:
                forces += 1
                force_next = terminal_tool
                _append_assistant(messages, response)
                messages.append({"role": "user",
                                 "content": FORCE_TERMINAL_NUDGE.format(
                                     tool=terminal_tool)})
                emit(EventType.NOTE, step=steps,
                     note=f"the model stopped without calling {terminal_tool}; "
                          f"forcing that call ({forces}/{MAX_TERMINAL_FORCES})")
                continue
            return LoopOutcome(LoopStatus.COMPLETED, None, final_text, steps,
                               total, messages, "the model stopped calling tools")

        messages.append(_assistant_turn(response))

        results: list[ToolCallResult] = []
        terminal_payload: dict[str, Any] | None = None

        for call in calls:
            signature = _signature(call)
            repeats = repeats + 1 if signature == last_signature else 1
            last_signature = signature

            emit(EventType.TOOL_CALL, step=steps, tool=call["name"],
                 args=_clip_args(call["input"]))

            content, is_error, guardrail = _dispatch(handlers, call)

            if guardrail:
                emit(EventType.GUARDRAIL_TRIGGERED, step=steps,
                     tool=call["name"], args=_clip_args(call["input"]),
                     reason=content)
            emit(EventType.TOOL_RESULT, step=steps, tool=call["name"],
                 is_error=is_error, guardrail=guardrail,
                 bytes=len(content), preview=content[:400])

            results.append(ToolCallResult(id=call["id"], name=call["name"],
                                          content=content, is_error=is_error))

            if (terminal_tool is not None and call["name"] == terminal_tool
                    and not is_error):
                terminal_payload = dict(call["input"])

        messages.extend(_tool_results(client, results))

        if terminal_payload is not None:
            return LoopOutcome(LoopStatus.COMPLETED, terminal_payload,
                               final_text, steps, total, messages,
                               f"{terminal_tool} accepted")

        if repeats >= NO_PROGRESS_REPEATS:
            reason = (f"the same call was made {repeats} times in a row "
                      f"({last_signature}) — the loop is not progressing")
            emit(EventType.NOTE, note=reason, step=steps)
            return LoopOutcome(LoopStatus.NO_PROGRESS, None, final_text, steps,
                               total, messages, reason)


# ─────────────────────────────  dispatch  ────────────────────────────────


def _dispatch(handlers: dict[str, Callable[..., Any]],
              call: dict[str, Any]) -> tuple[str, bool, bool]:
    """Run one tool call. Returns (content, is_error, was_guardrail).

    The handler lookup is INSIDE the guard. It used to sit outside, where a
    hallucinated tool name raised KeyError somewhere the `except KeyError`
    could not see it and killed the run — the exact failure the guard existed
    to prevent.
    """
    name = call["name"]
    args = call.get("input") or {}

    if "__malformed_arguments__" in args:
        return (f"The arguments for {name} were not valid JSON. Received: "
                f"{args['__malformed_arguments__'][:300]}. Send the arguments "
                f"as a JSON object.", True, False)

    try:
        handler = handlers[name]
    except KeyError:
        return (f"No such tool '{name}'. Available tools: "
                f"{', '.join(sorted(handlers))}.", True, False)

    try:
        return str(handler(**args)), False, False
    except GuardrailError as exc:
        return str(exc), True, True
    except ToolError as exc:
        return str(exc), True, False
    except TypeError as exc:
        return f"Bad arguments for {name}: {exc}", True, False
    except Exception as exc:  # noqa: BLE001 — a tool must never kill the loop
        return f"{name} failed: {type(exc).__name__}: {exc}", True, False


def _tool_results(client: Any, results: list[ToolCallResult]) -> list[dict[str, Any]]:
    """Ask the client to format results; fall back to the OpenAI shape.

    The fallback exists for hand-rolled test doubles, not for real providers.
    """
    formatter = getattr(client, "tool_result_messages", None)
    if callable(formatter):
        return list(formatter(results))
    return [{"role": "tool", "tool_call_id": r.id, "name": r.name,
             "content": (f"ERROR: {r.content}" if r.is_error else r.content)}
            for r in results]


def _append_assistant(messages: list[dict[str, Any]],
                      response: LLMResponse) -> None:
    """Echo the assistant turn back, unless it is empty.

    A truncated reply can carry neither content nor tool calls, and some
    providers reject a message with both missing. Dropping it loses nothing:
    there was nothing in it.
    """
    turn = _assistant_turn(response)
    if turn.get("content") or turn.get("tool_calls"):
        messages.append(turn)


def _assistant_turn(response: LLMResponse) -> dict[str, Any]:
    """The provider-shaped assistant message to echo back."""
    if response.raw_message:
        return dict(response.raw_message)
    # A client that populates neither is a client that cannot hold a
    # conversation; say so in the transcript rather than sending nothing.
    return {"role": "assistant", "content": response.text()}


def _signature(call: dict[str, Any]) -> str:
    """Identity of a dispatch, for no-progress detection."""
    args = json.dumps(call.get("input") or {}, sort_keys=True,
                      separators=(",", ":"), default=str)
    return f"{call['name']}({args})"


def _correction_for(client: Any, error: Exception) -> str | None:
    """Ask the client whether this failure is the model's to fix."""
    hook = getattr(client, "correction_for", None)
    if not callable(hook):
        return None
    try:
        result = hook(error)
    except Exception:  # noqa: BLE001 — a broken hook is not a reason to die
        return None
    return str(result) if result else None


def _tool_choice(client: Any, force: str | None) -> dict[str, Any] | None:
    """Ask the client for its way of forcing one named tool, if it has one."""
    if force is None:
        return None
    builder = getattr(client, "forced_tool_choice", None)
    return builder(force) if callable(builder) else None


def _tool_names(tools: Iterable[dict[str, Any]] | None) -> list[str]:
    names: list[str] = []
    for tool in tools or []:
        nested = tool.get("function")
        inner: dict[str, Any] = nested if isinstance(nested, dict) else tool
        name = inner.get("name")
        if name:
            names.append(str(name))
    return names


def _reasoning(response: LLMResponse) -> str | None:
    """Some providers return the model's private reasoning alongside content."""
    raw: dict[str, Any] = response.raw_message or {}
    value = raw.get("reasoning")
    return str(value) if value else None


def _clip_args(args: dict[str, Any], limit: int = 600) -> str:
    rendered = ", ".join(f"{k}={v!r}" for k, v in (args or {}).items())
    return rendered if len(rendered) <= limit else rendered[:limit] + " …"


def _add(a: Usage, b: Usage) -> Usage:
    return Usage(
        input_tokens=a.input_tokens + b.input_tokens,
        output_tokens=a.output_tokens + b.output_tokens,
        cache_read_tokens=a.cache_read_tokens + b.cache_read_tokens,
        cache_write_tokens=a.cache_write_tokens + b.cache_write_tokens,
        usd=round(a.usd + b.usd, 8),
    )
