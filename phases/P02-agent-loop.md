# PHASE P02 — Agent loop, tool registry, dispatcher, budgets

**Mode: HAND-WRITE** · **Est:** 4h · **Preconditions:** P00, P01 complete

> **Agent instruction:** Do not implement this. Explain mechanics, review the human's
> code, point out what's wrong. If asked for the implementation, restate this line.

---

## 1. GOAL

The loop. Given an `LLMClient`, a set of tools, a system prompt and a budget, run
turns until the agent finishes, hits a terminal tool, or runs out of budget.

## 2. WHY THIS PHASE

This file is the answer to "what is an agent." It's about 120 lines. Every product in
this category — Claude Code, Devin, Cursor's agent mode, this one — is this loop plus
tools plus prompts. Once you've written it, "agentic AI" stops being a category and
becomes a `while` statement you can reason about.

It's also where four of the invariants in `CLAUDE.md` get physically enforced:
capability scoping, budget caps, no-progress detection, and tool errors as context.

## 3. PRECONDITIONS

`AnthropicClient` from P01 with cassette support. `ToolResult`, `Usage` in contracts.

## 4. DELIVERABLES

```
perfagent/tools/__init__.py        # ToolSpec, ToolRegistry
perfagent/agent.py                 # LoopBudget, AgentLoop, LoopOutcome
tests/test_agent_loop.py           # from section 6
```

## 5. SPEC

### 5.1 `ToolSpec` and `ToolRegistry`

```python
@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    input_schema: dict
    handler: Callable[..., str]      # returns content; raises ToolError on failure
    phases: frozenset[str]           # which phases may use it — capability scoping
    terminal: bool = False           # calling it ends the loop

class ToolRegistry:
    def register(self, spec: ToolSpec) -> None: ...
    def schemas_for(self, phase: str) -> list[dict]:
        """Anthropic-shaped tool schemas for tools available in this phase.
        A tool not in this list does not exist to the model."""
    def dispatch(self, phase: str, name: str, args: dict) -> ToolResult: ...
```

`dispatch` rules:

- Tool not registered → `ToolResult(is_error=True)` listing available tool names.
  Never raise. The model recovers from a good error message.
- Tool not permitted in this phase → `is_error=True`, and **emit a
  `GUARDRAIL_TRIGGERED` trace event**. This is the enforcement point for "the
  comprehension agent cannot write files."
- Handler raises `ToolError` → `is_error=True` with its message and suggestion.
- Handler raises anything else → `is_error=True` with the exception repr, plus a
  trace event. A bug in a tool must not kill a run.
- Success → `ToolResult(content=..., bytes_returned=len(content))`.

### 5.2 `LoopBudget`

```python
@dataclass
class LoopBudget:
    max_steps: int = 40
    max_usd: float = 2.0
    max_seconds: float = 900.0
    max_output_tokens_total: int | None = None
    no_progress_threshold: int = 3
```

### 5.3 `LoopOutcome`

```python
class LoopStatus(str, Enum):
    COMPLETED = "COMPLETED"                 # terminal tool called, or end_turn
    BUDGET_EXHAUSTED = "BUDGET_EXHAUSTED"
    NO_PROGRESS = "NO_PROGRESS"
    ERROR = "ERROR"

@dataclass
class LoopOutcome:
    status: LoopStatus
    terminal_tool_name: str | None
    terminal_args: dict | None               # the artifact payload, if terminal
    final_text: str
    steps: int
    messages: list[dict]                     # full transcript, for the trace
    usage: Usage
```

### 5.4 The loop

```python
class AgentLoop:
    def __init__(self, *, llm: LLMClient, registry: ToolRegistry, phase: str,
                 system_prompt: str | list[dict], budget: LoopBudget,
                 trace: Trace, tool_choice: dict | None = None) -> None: ...

    def run(self, task: str) -> LoopOutcome: ...
```

Algorithm:

1. `messages = [{"role": "user", "content": task}]`
2. Loop while budget allows:
   - `resp = llm.complete(system=..., messages=messages, tools=registry.schemas_for(phase), tool_choice=...)`
   - Trace `LLM_RESPONSE` with usage, cost, latency, stop_reason.
   - Append `{"role": "assistant", "content": resp.content}` **verbatim** — including
     any `thinking` blocks. Mutating assistant content breaks the next request.
   - If `stop_reason != "tool_use"` → `COMPLETED` with `final_text`.
   - For **each** `tool_use` block (there may be several):
     - Trace `TOOL_CALL`.
     - `result = registry.dispatch(phase, name, input)`
     - Trace `TOOL_RESULT` with bytes and duration.
     - Collect `{"type": "tool_result", "tool_use_id": id, "content": result.content, "is_error": result.is_error}`
     - If the tool is `terminal` and not an error → return `COMPLETED` with
       `terminal_tool_name` and `terminal_args`, **after** appending the results.
   - Append one user message whose content is the list of all tool_result blocks.
     One message, all results. Not one message per result.
3. Budget exceeded → `BUDGET_EXHAUSTED`.

### 5.5 No-progress detection

Hash `(tool_name, canonical_json(args))` for every dispatch. If the same hash occurs
`no_progress_threshold` times, stop with `NO_PROGRESS`.

An agent repeating an identical call is stuck, and the loop must notice rather than
burn the whole budget. When it happens, the fix is almost never the prompt — it's
that the tool's error message didn't tell the model how to move forward.

### 5.6 Budget enforcement

Check before each LLM call: steps, wall clock, and `llm.usage.usd` against
`max_usd`. Exceeding any is a normal outcome with partial results, not an exception.

### 5.7 What this class must NOT do

- No prompt construction beyond passing through the system prompt.
- No knowledge of `EndpointModel` or any specific artifact.
- No retry of the LLM call (P01 owns that).
- No summarisation or compaction (that's a later concern, and when it arrives it
  arrives as an injected policy object, not as code in here).

Keeping it ignorant is what lets P04, P09 and P12 all reuse it unchanged.

## 6. ACCEPTANCE TEST

`tests/test_agent_loop.py`. No network. A `ScriptedLLM` replaces the client, which
means you can test agent *behaviour* deterministically — the technique that makes
agent development tractable.

```python
from __future__ import annotations
import pytest
from perfagent.contracts import Usage, LLMResponse
from perfagent.agent import AgentLoop, LoopBudget, LoopStatus
from perfagent.tools import ToolSpec, ToolRegistry
from perfagent.trace import Trace, EventType
from perfagent.errors import ToolError


class ScriptedLLM:
    """Returns pre-baked responses in order. Records what it was sent."""
    def __init__(self, script): self.script = list(script); self.seen = []
    def complete(self, **kw):
        self.seen.append(kw)
        item = self.script.pop(0)
        return LLMResponse(content=item[0], stop_reason=item[1],
                           usage=Usage(input_tokens=10, output_tokens=5, usd=0.01),
                           latency_ms=1.0)
    @property
    def usage(self):
        return Usage(input_tokens=10 * len(self.seen), output_tokens=5 * len(self.seen),
                     usd=0.01 * len(self.seen))


def _use(name, args, id="tu_1"):
    return {"type": "tool_use", "id": id, "name": name, "input": args}

def _text(t):
    return {"type": "text", "text": t}


def _registry(**handlers):
    r = ToolRegistry()
    for name, (fn, phases, terminal) in handlers.items():
        r.register(ToolSpec(name=name, description="d",
                            input_schema={"type": "object"},
                            handler=fn, phases=frozenset(phases), terminal=terminal))
    return r


def _loop(llm, registry, tmp_path, budget=None, phase="P1"):
    return AgentLoop(llm=llm, registry=registry, phase=phase,
                     system_prompt="s", budget=budget or LoopBudget(),
                     trace=Trace(tmp_path, "run1"))


# ── the happy path ───────────────────────────────────────────────────────

def test_single_tool_call_then_finish(tmp_path):
    calls = []
    reg = _registry(peek=(lambda path: calls.append(path) or "file contents",
                          ["P1"], False))
    llm = ScriptedLLM([([_use("peek", {"path": "A.java"})], "tool_use"),
                       ([_text("done")], "end_turn")])
    out = _loop(llm, reg, tmp_path).run("look at A.java")
    assert out.status is LoopStatus.COMPLETED
    assert calls == ["A.java"]
    assert out.final_text == "done"
    assert out.steps == 2


def test_tool_result_is_fed_back_correctly(tmp_path):
    reg = _registry(peek=(lambda path: "CONTENTS_XYZ", ["P1"], False))
    llm = ScriptedLLM([([_use("peek", {"path": "A"})], "tool_use"),
                       ([_text("ok")], "end_turn")])
    _loop(llm, reg, tmp_path).run("go")
    second = llm.seen[1]["messages"]
    # assistant turn preserved verbatim, then ONE user message of tool_results
    assert second[1]["role"] == "assistant"
    assert second[2]["role"] == "user"
    blocks = second[2]["content"]
    assert isinstance(blocks, list) and blocks[0]["type"] == "tool_result"
    assert blocks[0]["tool_use_id"] == "tu_1"
    assert "CONTENTS_XYZ" in blocks[0]["content"]


def test_parallel_tool_uses_batched_into_one_user_message(tmp_path):
    reg = _registry(peek=(lambda path: f"c:{path}", ["P1"], False))
    llm = ScriptedLLM([([_use("peek", {"path": "A"}, "t1"),
                         _use("peek", {"path": "B"}, "t2")], "tool_use"),
                       ([_text("ok")], "end_turn")])
    _loop(llm, reg, tmp_path).run("go")
    blocks = llm.seen[1]["messages"][2]["content"]
    assert len(blocks) == 2
    assert {b["tool_use_id"] for b in blocks} == {"t1", "t2"}


# ── termination ──────────────────────────────────────────────────────────

def test_terminal_tool_ends_loop_and_returns_args(tmp_path):
    reg = _registry(emit=(lambda **kw: "accepted", ["P1"], True))
    llm = ScriptedLLM([([_use("emit", {"route": "POST /x", "path": []})], "tool_use")])
    out = _loop(llm, reg, tmp_path).run("go")
    assert out.status is LoopStatus.COMPLETED
    assert out.terminal_tool_name == "emit"
    assert out.terminal_args["route"] == "POST /x"


def test_erroring_terminal_tool_does_not_end_loop(tmp_path):
    """A rejected artifact must let the agent retry, not silently succeed."""
    def emit(**kw): raise ToolError("citation A.java:99 does not exist",
                                    suggestion="re-read the file and retry")
    reg = _registry(emit=(emit, ["P1"], True))
    llm = ScriptedLLM([([_use("emit", {"x": 1})], "tool_use"),
                       ([_text("giving up")], "end_turn")])
    out = _loop(llm, reg, tmp_path).run("go")
    assert out.terminal_args is None
    assert out.status is LoopStatus.COMPLETED and out.final_text == "giving up"


# ── errors as context ────────────────────────────────────────────────────

def test_tool_error_becomes_is_error_result_and_loop_continues(tmp_path):
    def boom(**kw): raise ToolError("no such symbol", suggestion="try CustomerSvc")
    reg = _registry(find=(boom, ["P1"], False))
    llm = ScriptedLLM([([_use("find", {"n": "Nope"})], "tool_use"),
                       ([_text("recovered")], "end_turn")])
    out = _loop(llm, reg, tmp_path).run("go")
    assert out.status is LoopStatus.COMPLETED
    block = llm.seen[1]["messages"][2]["content"][0]
    assert block["is_error"] is True
    assert "no such symbol" in block["content"]
    assert "CustomerSvc" in block["content"]          # suggestion reaches the model


def test_unexpected_exception_does_not_kill_the_run(tmp_path):
    def boom(**kw): raise ZeroDivisionError("bug in my tool")
    reg = _registry(find=(boom, ["P1"], False))
    llm = ScriptedLLM([([_use("find", {})], "tool_use"),
                       ([_text("ok")], "end_turn")])
    out = _loop(llm, reg, tmp_path).run("go")
    assert out.status is LoopStatus.COMPLETED


def test_unknown_tool_returns_available_names(tmp_path):
    reg = _registry(peek=(lambda **kw: "c", ["P1"], False))
    llm = ScriptedLLM([([_use("nonexistent", {})], "tool_use"),
                       ([_text("ok")], "end_turn")])
    _loop(llm, reg, tmp_path).run("go")
    block = llm.seen[1]["messages"][2]["content"][0]
    assert block["is_error"] is True
    assert "peek" in block["content"]


# ── capability scoping ───────────────────────────────────────────────────

def test_tool_not_permitted_in_phase_is_refused_and_traced(tmp_path):
    """The comprehension agent cannot write files because the tool does not
    exist to it. This is enforcement, not instruction."""
    written = []
    reg = _registry(write=(lambda **kw: written.append(kw) or "ok", ["P7"], False))
    llm = ScriptedLLM([([_use("write", {"path": "x", "body": "y"})], "tool_use"),
                       ([_text("ok")], "end_turn")])
    trace = Trace(tmp_path, "run1")
    loop = AgentLoop(llm=llm, registry=reg, phase="P1", system_prompt="s",
                     budget=LoopBudget(), trace=trace)
    loop.run("try to write")
    assert written == []
    kinds = [e["type"] for e in trace.read_all()]
    assert EventType.GUARDRAIL_TRIGGERED.value in kinds


def test_schemas_for_phase_hides_other_phases_tools(tmp_path):
    reg = _registry(read=(lambda **kw: "c", ["P1"], False),
                    write=(lambda **kw: "c", ["P7"], False))
    assert [s["name"] for s in reg.schemas_for("P1")] == ["read"]
    assert [s["name"] for s in reg.schemas_for("P7")] == ["write"]


# ── budgets and stuck loops ──────────────────────────────────────────────

def test_max_steps_exhausts_budget(tmp_path):
    reg = _registry(peek=(lambda i=0: f"c{i}", ["P1"], False))
    script = [([_use("peek", {"i": n})], "tool_use") for n in range(20)]
    out = _loop(ScriptedLLM(script), reg, tmp_path,
                budget=LoopBudget(max_steps=3)).run("go")
    assert out.status is LoopStatus.BUDGET_EXHAUSTED
    assert out.steps == 3


def test_max_usd_exhausts_budget(tmp_path):
    reg = _registry(peek=(lambda i=0: "c", ["P1"], False))
    script = [([_use("peek", {"i": n})], "tool_use") for n in range(20)]
    out = _loop(ScriptedLLM(script), reg, tmp_path,
                budget=LoopBudget(max_usd=0.025)).run("go")
    assert out.status is LoopStatus.BUDGET_EXHAUSTED


def test_no_progress_detection_on_identical_calls(tmp_path):
    """Same tool, same args, three times: the agent is stuck. Break out."""
    reg = _registry(peek=(lambda path: "same", ["P1"], False))
    script = [([_use("peek", {"path": "A"})], "tool_use")] * 10
    out = _loop(ScriptedLLM(script), reg, tmp_path,
                budget=LoopBudget(no_progress_threshold=3)).run("go")
    assert out.status is LoopStatus.NO_PROGRESS
    assert out.steps <= 4


def test_varied_calls_are_not_flagged_as_no_progress(tmp_path):
    reg = _registry(peek=(lambda path: "c", ["P1"], False))
    script = ([([_use("peek", {"path": f"F{n}"})], "tool_use") for n in range(4)]
              + [([_text("done")], "end_turn")])
    out = _loop(ScriptedLLM(script), reg, tmp_path).run("go")
    assert out.status is LoopStatus.COMPLETED


# ── tracing ──────────────────────────────────────────────────────────────

def test_every_step_is_traced(tmp_path):
    reg = _registry(peek=(lambda path: "c", ["P1"], False))
    llm = ScriptedLLM([([_use("peek", {"path": "A"})], "tool_use"),
                       ([_text("ok")], "end_turn")])
    trace = Trace(tmp_path, "run1")
    AgentLoop(llm=llm, registry=reg, phase="P1", system_prompt="s",
              budget=LoopBudget(), trace=trace).run("go")
    kinds = [e["type"] for e in trace.read_all()]
    for expected in (EventType.LLM_RESPONSE, EventType.TOOL_CALL,
                     EventType.TOOL_RESULT):
        assert expected.value in kinds
```

`pytest tests/test_agent_loop.py -q` must pass with no network and no API key.

## 7. OUT OF SCOPE

No real tools (P03). No prompts for any specific agent (P04). No context compaction.
No sub-agents. No parallel execution of tool handlers — dispatch them in order. No
`EndpointModel` or any other artifact type referenced anywhere in these two files.

## 8. DEMO

Wire a throwaway calculator tool to the real client and let it solve
`(4813 * 27) - 991` using only the tool. Watch the trace: two tool calls, one final
answer. Then set `max_steps=1` and watch it terminate mid-thought.

## 9. LEARNING

**Concept: this is what an agent is.**

An agent is: a loop, a stop condition, a set of tools, and a budget. Nothing else.
Every "AI agent framework" is a wrapper around what you just wrote. You will never
again be unsure whether something is an agent or a workflow — you'll ask whether
there's a loop whose step count depends on runtime information.

**Four things in the spec that are doing more work than they look like:**

`registry.schemas_for(phase)` — capability scoping. The comprehension agent is safe
not because it was told to be, but because `apply_patch` is not in its schema list.
This is the whole guardrail philosophy in one method call.

`is_error=True` instead of raising — errors are context. A tool failure is
information the model can act on, and the loop's job is to deliver it well. The
quality of your error messages *is* the quality of your agent.

No-progress detection — agents get stuck in ways that don't look like errors. When
this trips, resist the urge to fix the prompt. Read the tool result the model kept
getting and ask why it didn't tell the model what to do next.

Assistant content appended verbatim — including `thinking` blocks. Rewrite them and
the next request is invalid. The transcript is a protocol, not a log.

### Experiments to run

1. **Delete the no-progress check** and give the model a tool that always returns
   `"not found"`. Watch it burn the full budget calling the same thing. That's the
   most common way an unattended agent wastes money.

2. **Return a deliberately bad error** (`"error"`) from a tool, then a good one
   (`"symbol 'CustomerSvc' not found; did you mean 'CustomerService'?"`). Same model,
   same prompt. Count the steps to success in each case. This is the most valuable
   number you'll measure all week — it's the empirical proof that tool design beats
   prompt engineering.

3. **Set `max_steps=2`** on a task needing five and read the partial outcome. Note
   that you get a *usable* partial result, not an exception. Graceful degradation is
   a design choice you just made.

4. **Give the model two tools where one is obviously right** and read which it picks.
   Then make the descriptions vague and try again. Tool descriptions are prompts.

### Before moving to P03, you should be able to answer

- Where does the loop decide it's finished, and how many distinct ways can it end?
- Why is a terminal tool a more reliable stop condition than waiting for `end_turn`?
- Why are tool errors returned rather than raised?
- If an agent is looping uselessly, what's the *first* thing you'd inspect?
- Why does `AgentLoop` know nothing about `EndpointModel`?
