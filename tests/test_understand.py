"""The comprehension phase: citation validation, the terminal tool, the runner.

No network and no database. The client is scripted; `get_schema` is not
registered, so nothing here can reach MySQL even by accident.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from perfagent.agent import LoopBudget
from perfagent.contracts import (Citation, EndpointModel, IndexInfo,
                                 LLMResponse, TableInfo, Usage)
from perfagent.errors import ContractViolation, ToolError
from perfagent.prompts import load_prompt
from perfagent.tools import SourceRoot
from perfagent.tools.submit import make_submit_endpoint_analysis
from perfagent.trace import Trace
from perfagent.understand.agent import understand
from perfagent.understand.validate import validate_citations

# Lines in the fixture's OrderController.java:
#   12  public Page<OrderDto> search(@RequestBody SearchRequest req) {
#   17  public OrderDto findOne(@PathVariable Long id) {
CONTROLLER = "src/main/java/com/ex/web/OrderController.java"
SERVICE = "src/main/java/com/ex/service/OrderService.java"


def payload(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "route": "POST /api/orders/search",
        "call_chain": [
            {"method": "OrderController.search", "file": CONTROLLER,
             "line": 12, "does": "delegates to OrderService.search"},
            {"method": "OrderService.search", "file": SERVICE,
             "line": 5, "does": "queries a page then maps each row"},
        ],
        "data_access": [
            {"trigger": "repo.findByFilter", "kind": "jpql_query",
             "file": SERVICE, "line": 6, "tables": ["orders"],
             "executes_per_row": False},
            {"trigger": "lazy load of Order.lines", "kind": "lazy_collection",
             "file": SERVICE, "line": 11, "tables": ["order_line"],
             "executes_per_row": True},
        ],
        "tables": [{"table": "orders", "entity": "Order",
                    "columns_read": ["id", "name"]}],
        "uncertainties": ["how many rows a page holds by default"],
    }
    base.update(overrides)
    return base


@pytest.fixture
def root(fixture_repo: Path) -> SourceRoot:
    return SourceRoot.of(fixture_repo)


# ─────────────────────────  citation validation  ─────────────────────────


def test_a_correct_model_has_no_citation_problems(root: SourceRoot) -> None:
    assert validate_citations(EndpointModel.model_validate(payload()), root) == []


def test_a_line_past_the_end_of_the_file_is_rejected(root: SourceRoot) -> None:
    bad = payload()
    bad["call_chain"][0]["line"] = 9999
    problems = validate_citations(EndpointModel.model_validate(bad), root)
    assert len(problems) == 1
    assert "outside the file" in problems[0].problem


def test_a_file_that_does_not_exist_is_rejected_with_a_suggestion(
    root: SourceRoot
) -> None:
    bad = payload()
    bad["call_chain"][0]["file"] = "src/main/java/com/ex/web/OrdreController.java"
    problems = validate_citations(EndpointModel.model_validate(bad), root)
    assert "does not exist" in problems[0].problem
    assert "OrderController.java" in (problems[0].found_instead or "")


def test_a_symbol_on_the_wrong_line_is_rejected_and_the_right_one_named(
    root: SourceRoot
) -> None:
    bad = payload()
    bad["call_chain"][0]["line"] = 1
    problems = validate_citations(EndpointModel.model_validate(bad), root)
    # Line 11 is `@PostMapping("/search")` — the nearest line that really does
    # contain the token, which is exactly what the model needs to be told.
    assert problems[0].nearest_line == 11
    assert "package com.ex.web;" in (problems[0].found_instead or "")


def test_an_annotation_above_the_method_is_within_tolerance(
    root: SourceRoot
) -> None:
    """Java puts annotations above a method and wraps long signatures, so a
    strict single-line check produces false rejections."""
    near = payload()
    near["call_chain"][0]["line"] = 11        # the @PostMapping line
    assert validate_citations(EndpointModel.model_validate(near), root) == []

    far = payload()
    far["call_chain"][0]["line"] = 8          # three lines away: too far
    assert validate_citations(EndpointModel.model_validate(far), root) != []


def test_every_problem_is_reported_at_once_not_just_the_first(
    root: SourceRoot
) -> None:
    """One retry should be able to fix everything."""
    bad = payload()
    bad["call_chain"][0]["line"] = 9999
    bad["call_chain"][1]["file"] = "src/main/java/com/ex/Nope.java"
    bad["data_access"][0]["line"] = 4242
    problems = validate_citations(EndpointModel.model_validate(bad), root)
    assert len(problems) == 3
    assert {p.where.split(" ")[0] for p in problems} == {
        "call_chain[0]", "call_chain[1]", "data_access[0]"}


def test_derived_fields_are_not_validated_twice(root: SourceRoot) -> None:
    """`entry` and `path` come from `call_chain`; reporting the same mistake
    under a name the model never typed would just confuse it."""
    bad = payload()
    bad["call_chain"][0]["line"] = 9999
    problems = validate_citations(EndpointModel.model_validate(bad), root)
    assert all(not p.where.startswith(("entry", "path")) for p in problems)


# ───────────────────────────  the terminal tool  ─────────────────────────


def test_submit_accepts_a_valid_analysis(root: SourceRoot) -> None:
    sink: dict[str, Any] = {}
    handler = make_submit_endpoint_analysis(root, sink, prompt_hash="abc123")
    message = handler(**payload())
    assert "accepted" in message
    assert "1 of them per-row" in message.replace("(", "").replace(")", "")
    assert sink["model"].prompt_hash == "abc123"


def test_submit_rejects_a_bad_citation_and_lists_every_problem(
    root: SourceRoot
) -> None:
    bad = payload()
    bad["call_chain"][0]["line"] = 9999
    bad["data_access"][1]["file"] = "src/main/java/com/ex/Ghost.java"
    with pytest.raises(ToolError) as caught:
        make_submit_endpoint_analysis(root, {})(**bad)
    message = str(caught.value)
    assert "2 citation(s) could not be verified" in message
    assert "call_chain[0]" in message and "data_access[1]" in message


def test_submit_quotes_pydantic_when_the_schema_is_violated(
    root: SourceRoot
) -> None:
    broken = payload()
    del broken["data_access"][0]["executes_per_row"]
    with pytest.raises(ToolError) as caught:
        make_submit_endpoint_analysis(root, {})(**broken)
    message = str(caught.value)
    assert "did not match the required schema" in message
    assert "executes_per_row" in message


def test_submit_rejects_an_unknown_data_access_kind(root: SourceRoot) -> None:
    broken = payload()
    broken["data_access"][0]["kind"] = "jpql_page"
    with pytest.raises(ToolError, match="kind"):
        make_submit_endpoint_analysis(root, {})(**broken)


def test_measured_table_facts_are_templated_in_not_retyped(
    root: SourceRoot
) -> None:
    """Invariant 4: a cardinality in the artifact is a number the database
    reported, never the model's recollection of it."""
    measured = {"orders": TableInfo(
        table="orders", all_columns=["id bigint NOT NULL"],
        existing_indexes=[IndexInfo(name="PRIMARY", columns=["id"],
                                    unique=True, cardinality=51)],
        approx_rows=51)}
    claimed = payload()
    claimed["tables"] = [{"table": "orders", "entity": "Order",
                          "columns_read": ["id", "name"],
                          "approx_rows": 999_999_999}]
    sink: dict[str, Any] = {}
    make_submit_endpoint_analysis(root, sink, measured_tables=measured)(**claimed)

    table = sink["model"].tables[0]
    assert table.approx_rows == 51, "the model's number must not survive"
    assert table.columns_read == ["id", "name"], "the model's judgement does"
    assert table.existing_indexes[0].cardinality == 51


def test_a_table_looked_up_but_not_declared_is_still_recorded(
    root: SourceRoot
) -> None:
    measured = {"order_line": TableInfo(table="order_line", approx_rows=7)}
    sink: dict[str, Any] = {}
    make_submit_endpoint_analysis(root, sink, measured_tables=measured)(**payload())
    assert {t.table for t in sink["model"].tables} == {"orders", "order_line"}


# ─────────────────────────  contract derivation  ─────────────────────────


def test_entry_and_path_are_derived_from_the_call_chain() -> None:
    model = EndpointModel.model_validate(payload())
    assert model.entry == Citation(file=CONTROLLER, line=12,
                                   symbol="OrderController.search")
    assert [c.ref() for c in model.path] == [f"{CONTROLLER}:12", f"{SERVICE}:5"]


def test_sql_sources_are_derived_from_data_access() -> None:
    model = EndpointModel.model_validate(payload())
    assert [s.kind for s in model.sql_sources] == ["@Query JPQL",
                                                   "lazy collection"]


def test_per_row_accesses_are_counted_in_code_not_by_a_model() -> None:
    model = EndpointModel.model_validate(payload())
    assert len(model.per_row_accesses()) == 1


def test_the_artifact_round_trips() -> None:
    model = EndpointModel.model_validate(payload())
    again = EndpointModel.model_validate(
        json.loads(json.dumps(model.model_dump(mode="json"))))
    assert again == model


# ──────────────────────────────  prompts  ────────────────────────────────


def test_the_prompt_files_load_and_are_hashed() -> None:
    system = load_prompt("understand.system")
    user = load_prompt("understand.user", repo_root="/repo", stack="spring",
                       endpoint="POST /x")
    assert "submit_endpoint_analysis" in system.text
    assert "POST /x" in user.text and "{endpoint}" not in user.text
    assert len(system.hash) == 16 and system.hash != user.hash


def test_the_prompt_carries_the_lesson_that_was_learned() -> None:
    """The phrase 'down to the SQL or database access' made the model treat
    lazy per-row loads as out of scope and hid the real bottleneck."""
    text = load_prompt("understand.system").text
    assert "down to the SQL" not in text
    assert "ONCE PER RESULT ROW" in text
    assert "lazy load inside a per-row loop" in text


def test_a_missing_placeholder_is_an_error_not_a_silent_blank() -> None:
    with pytest.raises(ContractViolation, match="endpoint"):
        load_prompt("understand.user", repo_root="/r", stack="s")


def test_a_placeholder_the_prompt_does_not_have_is_an_error() -> None:
    with pytest.raises(ContractViolation, match="drifted apart"):
        load_prompt("understand.user", repo_root="/r", stack="s",
                    endpoint="POST /x", typo="oops")


def test_the_prompt_hash_is_recorded_in_the_trace(tmp_path: Path) -> None:
    trace = Trace(tmp_path / "run", "r1")
    prompt = load_prompt("understand.system", trace=trace)
    recorded = [e for e in trace.read_all() if e["type"] == "prompt_loaded"]
    assert recorded[0]["prompt_hash"] == prompt.hash


# ──────────────────────────────  the runner  ─────────────────────────────


class Scripted:
    """A client that plays a fixed script of tool calls."""

    def __init__(self, *turns: dict[str, Any] | str) -> None:
        self.turns = list(turns)

    def complete(self, **kwargs: Any) -> LLMResponse:
        turn = self.turns.pop(0) if self.turns else "I have nothing more."
        if isinstance(turn, str):
            return LLMResponse(content=[{"type": "text", "text": turn}],
                               stop_reason="end_turn", usage=Usage(),
                               latency_ms=1.0,
                               raw_message={"role": "assistant", "content": turn})
        return LLMResponse(
            content=[{"type": "tool_use", "id": "c1", "name": turn["name"],
                      "input": turn["input"]}],
            stop_reason="tool_use", usage=Usage(), latency_ms=1.0,
            raw_message={"role": "assistant", "tool_calls": [{"id": "c1"}]})

    def forced_tool_choice(self, name: str) -> dict[str, Any]:
        return {"type": "function", "function": {"name": name}}

    def tool_result_messages(self, results):
        return [{"role": "tool", "tool_call_id": r.id, "content": r.content}
                for r in results]


def run(client: Any, repo: Path, tmp_path: Path, endpoint: str = "POST /api/orders/search"):
    return understand(
        repo=repo, endpoint=endpoint, client=client,
        trace=Trace(tmp_path / "run", "r1"),
        budget=LoopBudget(max_steps=8), run_dir=tmp_path / "run",
        mysql=None,     # no database in tests; get_schema is not registered
    )


def test_the_phase_writes_a_validated_artifact(fixture_repo: Path,
                                               tmp_path: Path) -> None:
    client = Scripted({"name": "find_route",
                       "input": {"method": "POST", "path": "/api/orders/search"}},
                      {"name": "submit_endpoint_analysis", "input": payload()})
    model = run(client, fixture_repo, tmp_path)

    assert model.route == "POST /api/orders/search"
    assert len(model.per_row_accesses()) == 1
    assert model.prompt_hash is not None

    written = json.loads((tmp_path / "run" / "endpoint_model.json").read_text())
    assert EndpointModel.model_validate(written) == model


def test_the_transcript_does_not_leave_the_phase(fixture_repo: Path,
                                                 tmp_path: Path) -> None:
    """The EndpointModel is the entire output. That boundary is what keeps a
    ten-phase run inside a context window."""
    client = Scripted({"name": "submit_endpoint_analysis", "input": payload()})
    model = run(client, fixture_repo, tmp_path)
    assert isinstance(model, EndpointModel)
    assert not hasattr(model, "messages") and not hasattr(model, "transcript")


def test_a_rejected_submission_is_retried_by_the_agent(fixture_repo: Path,
                                                       tmp_path: Path) -> None:
    bad = payload()
    bad["call_chain"][0]["line"] = 9999
    client = Scripted({"name": "submit_endpoint_analysis", "input": bad},
                      {"name": "submit_endpoint_analysis", "input": payload()})
    model = run(client, fixture_repo, tmp_path)
    assert model.call_chain[0].line == 12


def test_giving_up_raises_and_includes_the_final_text(fixture_repo: Path,
                                                      tmp_path: Path) -> None:
    client = Scripted(*["here is a prose summary instead"] * 6)
    with pytest.raises(ContractViolation) as caught:
        run(client, fixture_repo, tmp_path)
    assert "without submitting an analysis" in str(caught.value)
    assert "here is a prose summary instead" in str(caught.value)


def test_running_out_of_budget_raises_with_the_trace_reference(
    fixture_repo: Path, tmp_path: Path
) -> None:
    turns = [{"name": "find_route",
              "input": {"method": "POST", "path": f"/api/orders/{n}"}}
             for n in range(20)]
    with pytest.raises(ContractViolation) as caught:
        understand(repo=fixture_repo, endpoint="POST /api/orders/search",
                   client=Scripted(*turns), trace=Trace(tmp_path / "run", "r1"),
                   budget=LoopBudget(max_steps=2), run_dir=tmp_path / "run",
                   mysql=None)
    assert "budget_exhausted" in str(caught.value)
    assert "trace.jsonl" in str(caught.value)


def test_a_path_escape_during_a_run_is_traced_as_a_guardrail(
    fixture_repo: Path, tmp_path: Path
) -> None:
    client = Scripted({"name": "read_file", "input": {"path": "../../etc/passwd"}},
                      {"name": "submit_endpoint_analysis", "input": payload()})
    trace = Trace(tmp_path / "run", "r1")
    understand(repo=fixture_repo, endpoint="POST /api/orders/search",
               client=client, trace=trace, budget=LoopBudget(max_steps=8),
               run_dir=tmp_path / "run", mysql=None)

    events = trace.read_all()
    guardrails = [e for e in events if e["type"] == "guardrail_triggered"]
    assert len(guardrails) == 1
    assert guardrails[0]["tool"] == "read_file"
    # Distinguishable from an ordinary error, not merged into one bucket.
    assert "outside the repository" in guardrails[0]["reason"]


def test_the_endpoint_reaches_the_prompt(fixture_repo: Path,
                                         tmp_path: Path) -> None:
    seen: list[str] = []

    class Watching(Scripted):
        def complete(self, **kwargs):
            seen.append(str(kwargs.get("messages")[0]["content"]))
            return super().complete(**kwargs)

    client = Watching({"name": "submit_endpoint_analysis",
                       "input": payload(route="GET /api/orders/{id}")})
    run(client, fixture_repo, tmp_path, endpoint="GET /api/orders/{id}")
    assert "GET /api/orders/{id}" in seen[0]
    assert str(fixture_repo.resolve()) in seen[0]
