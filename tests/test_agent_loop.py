"""The agent loop. Every client here is a scripted double; nothing goes out."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import pytest

from perfagent.agent import (DEFAULT_MAX_STEPS, LoopBudget, LoopOutcome,
                             LoopStatus, run_agent)
from perfagent.contracts import LLMResponse, Usage
from perfagent.errors import GuardrailError, ToolError
from perfagent.trace import Trace

TOOLS = [{"name": "read_file", "description": "read", "parameters": {}},
         {"name": "submit", "description": "finish", "parameters": {}}]


def text(body: str, usd: float = 0.0) -> LLMResponse:
    return LLMResponse(content=[{"type": "text", "text": body}],
                       stop_reason="end_turn", usage=Usage(usd=usd),
                       latency_ms=1.0,
                       raw_message={"role": "assistant", "content": body})


def truncated() -> LLMResponse:
    return LLMResponse(content=[], stop_reason="max_tokens", usage=Usage(),
                       latency_ms=1.0,
                       raw_message={"role": "assistant", "content": None})


def call(name: str, args: dict | None = None, ident: str = "c1",
         usd: float = 0.0) -> LLMResponse:
    return LLMResponse(
        content=[{"type": "tool_use", "id": ident, "name": name,
                  "input": args or {}}],
        stop_reason="tool_use", usage=Usage(usd=usd), latency_ms=1.0,
        raw_message={"role": "assistant", "tool_calls": [{"id": ident}]})


class Scripted:
    def __init__(self, *responses: LLMResponse) -> None:
        self.responses = list(responses)
        self.requests: list[dict[str, Any]] = []

    def complete(self, **kwargs: Any) -> LLMResponse:
        self.requests.append(kwargs)
        if not self.responses:
            return text("nothing left to say")
        return self.responses.pop(0)

    def forced_tool_choice(self, name: str) -> dict[str, Any]:
        return {"type": "function", "function": {"name": name}}

    def tool_result_messages(self, results):
        return [{"role": "tool", "tool_call_id": r.id, "content": r.content}
                for r in results]


def trace_for(tmp_path: Path) -> Trace:
    return Trace(tmp_path / "run", "test-run")


# ────────────────────────────  the outcome  ──────────────────────────────


def test_the_terminal_tool_is_a_parameter_not_a_hardcoded_name() -> None:
    client = Scripted(call("finish_here", {"x": 1}))
    outcome = run_agent(client=client, task="go", tools=TOOLS,
                        handlers={"finish_here": lambda **k: "ok"},
                        terminal_tool="finish_here")
    assert outcome.status is LoopStatus.COMPLETED
    assert outcome.terminal_payload == {"x": 1}


def test_a_rejected_submission_is_retryable() -> None:
    """A validator that ends the run on its first complaint is one the model
    can never satisfy."""
    attempts: list[dict] = []

    def submit(**payload):
        attempts.append(payload)
        if len(attempts) == 1:
            raise ToolError("line 99 does not exist; the file has 40 lines")
        return "accepted"

    client = Scripted(call("submit", {"line": 99}), call("submit", {"line": 12}))
    outcome = run_agent(client=client, task="go", tools=TOOLS,
                        handlers={"submit": submit}, terminal_tool="submit")
    assert outcome.status is LoopStatus.COMPLETED
    assert outcome.terminal_payload == {"line": 12}
    assert len(attempts) == 2


def test_the_rejection_text_is_handed_back_to_the_model() -> None:
    def submit(**payload):
        raise ToolError("line 99 does not exist")

    client = Scripted(call("submit"), text("giving up"))
    run_agent(client=client, task="go", tools=TOOLS,
              handlers={"submit": submit}, terminal_tool="submit")
    second = client.requests[1]["messages"]
    assert any("line 99 does not exist" in str(m.get("content"))
               for m in second)


def test_the_outcome_is_typed_and_still_reads_as_the_final_text() -> None:
    outcome = run_agent(client=Scripted(text("all done")), task="go",
                        tools=TOOLS, handlers={})
    assert isinstance(outcome, LoopOutcome)
    assert str(outcome) == "all done"


# ─────────────────────────  the latent crash  ────────────────────────────


def test_a_hallucinated_tool_name_does_not_kill_the_run() -> None:
    """The lookup used to sit outside the guard, where `except KeyError`
    could not see it."""
    client = Scripted(call("no_such_tool"), text("sorry"))
    outcome = run_agent(client=client, task="go", tools=TOOLS,
                        handlers={"read_file": lambda **k: "x"})
    assert outcome.status is LoopStatus.COMPLETED
    assert any("No such tool 'no_such_tool'" in str(m.get("content"))
               for m in outcome.messages)


def test_bad_arguments_come_back_as_a_readable_tool_result() -> None:
    client = Scripted(call("read_file", {"nonexistent": 1}), text("ok"))
    outcome = run_agent(client=client, task="go", tools=TOOLS,
                        handlers={"read_file": lambda path: path})
    assert any("Bad arguments for read_file" in str(m.get("content"))
               for m in outcome.messages)


def test_an_unexpected_exception_becomes_a_tool_result_not_a_crash() -> None:
    def explode(**kwargs):
        raise ZeroDivisionError("nope")

    client = Scripted(call("read_file"), text("ok"))
    outcome = run_agent(client=client, task="go", tools=TOOLS,
                        handlers={"read_file": explode})
    assert outcome.status is LoopStatus.COMPLETED
    assert any("ZeroDivisionError" in str(m.get("content"))
               for m in outcome.messages)


def test_an_llm_failure_is_an_error_outcome_not_an_exception() -> None:
    class Broken:
        def complete(self, **kwargs):
            raise RuntimeError("provider fell over")

    outcome = run_agent(client=Broken(), task="go", tools=TOOLS, handlers={})
    assert outcome.status is LoopStatus.ERROR
    assert "provider fell over" in outcome.stopped_because


# ─────────────────────────────  guardrails  ──────────────────────────────


def test_a_guardrail_is_distinguishable_from_an_ordinary_error(
    tmp_path: Path
) -> None:
    def refuse(**kwargs):
        raise GuardrailError("path '../../etc/passwd' resolves outside the repository")

    def ordinary(**kwargs):
        raise ToolError("file not found")

    trace = trace_for(tmp_path)
    client = Scripted(call("read_file", {"path": "../../etc/passwd"}),
                      call("submit", {}, ident="c2"), text("done"))
    run_agent(client=client, task="go", tools=TOOLS,
              handlers={"read_file": refuse, "submit": ordinary}, trace=trace)

    events = trace.read_all()
    guardrails = [e for e in events if e["type"] == "guardrail_triggered"]
    assert len(guardrails) == 1
    assert guardrails[0]["tool"] == "read_file"

    results = [e for e in events if e["type"] == "tool_result"]
    assert [(r["tool"], r["is_error"], r["guardrail"]) for r in results] == [
        ("read_file", True, True), ("submit", True, False)]


# ──────────────────────────────  budget  ─────────────────────────────────


def test_a_step_ceiling_is_a_normal_outcome_with_partial_results() -> None:
    client = Scripted(*[call("read_file", {"path": f"F{n}.java"})
                        for n in range(10)])
    outcome = run_agent(client=client, task="go", tools=TOOLS,
                        handlers={"read_file": lambda **k: "x"},
                        budget=LoopBudget(max_steps=3))
    assert outcome.status is LoopStatus.BUDGET_EXHAUSTED
    assert outcome.steps == 3
    assert "step ceiling of 3" in outcome.stopped_because


def test_zero_is_zero_and_none_is_unlimited() -> None:
    """`if self.max_steps` treated a 0 budget as unlimited once already."""
    zero = LoopBudget(max_steps=0, max_usd=0.0)
    assert zero.exceeded(steps=0, usd=0.0) is not None

    unlimited = LoopBudget(max_steps=None, max_usd=None, max_wall_s=None)
    assert unlimited.exceeded(steps=10_000, usd=999.0) is None


def test_a_spend_ceiling_stops_the_loop() -> None:
    client = Scripted(*[call("read_file", {"path": f"F{n}.java"}, usd=0.5)
                        for n in range(10)])
    outcome = run_agent(client=client, task="go", tools=TOOLS,
                        handlers={"read_file": lambda **k: "x"},
                        budget=LoopBudget(max_steps=None, max_usd=1.0))
    assert outcome.status is LoopStatus.BUDGET_EXHAUSTED
    assert outcome.usage.usd == pytest.approx(1.0)


def test_a_wall_clock_ceiling_stops_the_loop() -> None:
    budget = LoopBudget(max_steps=None, max_wall_s=0.001)
    budget.started_at = time.monotonic() - 10
    outcome = run_agent(client=Scripted(call("read_file")), task="go",
                        tools=TOOLS, handlers={"read_file": lambda **k: "x"},
                        budget=budget)
    assert outcome.status is LoopStatus.BUDGET_EXHAUSTED


def test_the_deprecated_max_steps_argument_still_works_but_warns() -> None:
    """Kept so pre-S1 callers keep running; warned so it does not survive
    indefinitely by accident."""
    client = Scripted(*[call("read_file", {"path": f"F{n}.java"})
                        for n in range(10)])
    with pytest.deprecated_call(match="max_steps"):
        outcome = run_agent(client=client, task="go", tools=TOOLS,
                            handlers={"read_file": lambda **k: "x"}, max_steps=2)
    assert outcome.steps == 2


# ────────────────────────────  no progress  ──────────────────────────────


def test_three_identical_calls_in_a_row_end_the_loop() -> None:
    client = Scripted(*[call("read_file", {"path": "A.java"}) for _ in range(6)])
    outcome = run_agent(client=client, task="go", tools=TOOLS,
                        handlers={"read_file": lambda **k: "same"})
    assert outcome.status is LoopStatus.NO_PROGRESS
    assert outcome.steps == 3


def test_two_identical_calls_are_a_retry_not_a_loop() -> None:
    client = Scripted(call("read_file", {"path": "A.java"}),
                      call("read_file", {"path": "A.java"}),
                      call("read_file", {"path": "B.java"}),
                      text("done"))
    outcome = run_agent(client=client, task="go", tools=TOOLS,
                        handlers={"read_file": lambda **k: "x"})
    assert outcome.status is LoopStatus.COMPLETED


def test_argument_order_does_not_defeat_no_progress_detection() -> None:
    same = [call("read_file", {"a": 1, "b": 2}), call("read_file", {"b": 2, "a": 1}),
            call("read_file", {"a": 1, "b": 2}), text("done")]
    outcome = run_agent(client=Scripted(*same), task="go", tools=TOOLS,
                        handlers={"read_file": lambda **k: "x"})
    assert outcome.status is LoopStatus.NO_PROGRESS


# ─────────────────────  finishing without the tool  ──────────────────────


def test_prose_instead_of_a_terminal_call_is_forced_once_before_giving_up() -> None:
    client = Scripted(text("here is my analysis in prose"),
                      call("submit", {"route": "GET /x"}))
    outcome = run_agent(client=client, task="go", tools=TOOLS,
                        handlers={"submit": lambda **k: "accepted"},
                        terminal_tool="submit")
    assert outcome.terminal_payload == {"route": "GET /x"}
    assert client.requests[1]["tool_choice"] == {
        "type": "function", "function": {"name": "submit"}}


def test_giving_up_for_good_is_completed_with_no_payload() -> None:
    client = Scripted(*[text("prose") for _ in range(8)])
    outcome = run_agent(client=client, task="go", tools=TOOLS,
                        handlers={"submit": lambda **k: "ok"},
                        terminal_tool="submit")
    assert outcome.status is LoopStatus.COMPLETED
    assert outcome.terminal_payload is None
    assert outcome.final_text == "prose"


def test_a_truncated_reply_is_not_treated_as_giving_up() -> None:
    """A reasoning model can spend its whole output budget thinking. That is a
    cut-off reply, not a finished one, and it must not burn the one forced
    terminal call."""
    client = Scripted(truncated(), call("submit", {"ok": True}))
    outcome = run_agent(client=client, task="go", tools=TOOLS,
                        handlers={"submit": lambda **k: "accepted"},
                        terminal_tool="submit")
    assert outcome.terminal_payload == {"ok": True}
    assert client.requests[1]["tool_choice"] is None, "forcing was not consumed"


def test_endless_truncation_is_an_error_that_names_the_setting() -> None:
    outcome = run_agent(client=Scripted(*[truncated() for _ in range(9)]),
                        task="go", tools=TOOLS, handlers={},
                        terminal_tool="submit")
    assert outcome.status is LoopStatus.ERROR
    assert "max_tokens" in outcome.stopped_because


# ──────────────────────────────  tracing  ────────────────────────────────


def test_the_loop_emits_the_events_the_transcript_is_built_from(
    tmp_path: Path
) -> None:
    trace = trace_for(tmp_path)
    client = Scripted(call("read_file", {"path": "A.java"}), text("done"))
    run_agent(client=client, task="go", tools=TOOLS,
              handlers={"read_file": lambda **k: "contents"}, trace=trace,
              phase="understand")
    kinds = [e["type"] for e in trace.read_all()]
    for expected in ("llm_request", "llm_response", "tool_call", "tool_result"):
        assert expected in kinds
    assert {e["phase"] for e in trace.read_all()} == {"understand"}


def test_there_is_no_sleep_left_in_the_loop() -> None:
    """~5.7 minutes of an 8-minute run went to an unconditional sleep(20)."""
    source = (Path(__file__).resolve().parent.parent
              / "perfagent" / "agent.py").read_text()
    assert "time.sleep" not in source
    assert "sleep(" not in source


def test_the_default_step_ceiling_is_not_infinite() -> None:
    assert DEFAULT_MAX_STEPS > 0


# ───────────────  a provider that rejects tool calls server-side  ────────


def test_a_provider_rejected_tool_call_is_corrected_not_fatal() -> None:
    """Groq validates tool calls SERVER-SIDE and returns HTTP 400 instead of
    handing the malformed call back. The loop's "a bad tool call is
    correctable, not a crash" guarantee has to survive that — otherwise the
    phase dies on a mistake the model could have fixed in one turn."""
    import json as _json

    from perfagent.llm.base import LLMHTTPError

    body = _json.dumps({"error": {
        "code": "tool_use_failed",
        "message": "parameters for tool search_code did not match schema: "
                   "errors: [missing properties: 'pattern']",
        "failed_generation": '{"name":"search_code","arguments":{"query":"x"}}',
    }})

    class Rejecting:
        """Fails the first call the way Groq does, then behaves."""

        def __init__(self) -> None:
            self.calls = 0
            self.seen: list[list[dict]] = []

        def complete(self, **kwargs):
            self.calls += 1
            self.seen.append(list(kwargs["messages"]))
            if self.calls == 1:
                raise LLMHTTPError("groq", 400, body, "https://api.groq.com/x")
            return LLMResponse(
                content=[{"type": "tool_use", "id": "t1", "name": "finish",
                          "input": {"ok": True}}],
                stop_reason="tool_use", usage=Usage(), latency_ms=1.0,
                raw_message={"role": "assistant", "tool_calls": [{"id": "t1"}]})

        def correction_for(self, error):
            from perfagent.llm.base import TOOL_ERROR_CODES

            payload = getattr(error, "payload", {})
            if str(payload.get("code")) not in TOOL_ERROR_CODES:
                return None
            return f"schema error: {payload.get('message')}"

        def tool_result_messages(self, results):
            return [{"role": "tool", "tool_call_id": r.id, "content": r.content}
                    for r in results]

    client = Rejecting()
    outcome = run_agent(
        client=client, task="go",
        tools=[{"name": "finish", "description": "d",
                "parameters": {"type": "object", "properties": {}}}],
        handlers={"finish": lambda ok: "done"},
        terminal_tool="finish", budget=LoopBudget(max_steps=5))

    assert outcome.status is LoopStatus.COMPLETED
    assert client.calls == 2, "it asked again instead of dying"
    # The correction reached the model, with the provider's own words in it.
    second = client.seen[1]
    assert any("missing properties: 'pattern'" in str(m.get("content"))
               for m in second)


def test_an_error_the_model_cannot_fix_still_ends_the_loop() -> None:
    """A 401 is not corrected by saying something different, and retrying it
    would spend the budget discovering that repeatedly."""
    from perfagent.llm.base import LLMHTTPError

    class Unauthorised:
        calls = 0

        def complete(self, **kwargs):
            type(self).calls += 1
            raise LLMHTTPError("groq", 401, '{"error":{"code":"invalid_api_key"}}',
                               "https://api.groq.com/x")

        def correction_for(self, error):
            return None

    outcome = run_agent(client=Unauthorised(), task="go", tools=[], handlers={},
                        budget=LoopBudget(max_steps=5))
    assert outcome.status is LoopStatus.ERROR
    assert Unauthorised.calls == 1, "no retry on an uncorrectable failure"
    assert "401" in outcome.stopped_because
