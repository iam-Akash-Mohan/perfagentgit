"""Session 1b: the parts of an endpoint that are not database access.

93% of the target endpoint's latency was a loop building one string per
invoice. Nothing in the artifact could express that, so every phase downstream
reasoned about queries because queries were all the map could describe. These
tests are the guard on the fields that fixed it.

No network, no database.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

from senapathy.agent import LoopBudget
from senapathy.contracts import (CacheInfo, Citation, EndpointModel,
                                 EntityRelation, HiddenQueryTrigger,
                                 LLMResponse, PaginationInfo, PerRowWork,
                                 ResponseShape, Usage)
from senapathy.errors import ToolError
from senapathy.prompts import load_prompt
from senapathy.tools import SourceRoot
from senapathy.tools.submit import (SUBMIT_PARAMETERS, git_head,
                                    make_submit_endpoint_analysis)
from senapathy.trace import Trace
from senapathy.understand.agent import PROMPT_ORDER, build_registry, understand
from senapathy.understand.validate import (completeness_warnings,
                                           validate_citations, validate_claims)

CONTROLLER = "src/main/java/com/ex/web/OrderController.java"
SERVICE = "src/main/java/com/ex/service/OrderService.java"


def payload(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "route": "POST /api/orders/search",
        "call_chain": [
            {"method": "OrderController.search", "file": CONTROLLER,
             "line": 12, "does": "delegates to OrderService.search"},
            {"method": "OrderService.enrich", "file": SERVICE,
             "line": 9, "does": "runs once per row"},
        ],
        "data_access": [
            {"trigger": "repo.findByFilter", "kind": "jpql_query",
             "file": SERVICE, "line": 6, "executes_per_row": False},
        ],
        "per_row_work": [
            {"description": "builds one tag string per line item",
             "file": SERVICE, "line": 10, "kind": "string_building",
             "scales_with": "line items per order", "bounded": False},
        ],
        "response_shape": {
            "dto": "OrderDto",
            "fields_returned": ["id", "lineCount", "tags"],
            "computed_fields": ["lineCount"],
            "unbounded_collections": ["tags"],
        },
        "request_shape": {"parameters": []},
        "uncertainties": [],
    }
    base.update(overrides)
    return base


@pytest.fixture
def root(fixture_repo: Path) -> SourceRoot:
    return SourceRoot.of(fixture_repo)


# ──────────────────────────  the contract  ───────────────────────────────


def test_per_row_work_holds_the_finding_the_map_could_not_express() -> None:
    """A loop that builds a string per invoice is not a data access, is not a
    sql source, and was 93% of the measured latency."""
    model = EndpointModel.model_validate(payload())
    assert len(model.per_row_work) == 1
    work = model.per_row_work[0]
    assert work.bounded is False
    assert work.scales_with == "line items per order"
    assert work.kind == "string_building"


def test_unbounded_per_row_work_is_computed_never_counted_by_a_model() -> None:
    model = EndpointModel.model_validate(payload(per_row_work=[
        {"description": "a", "file": SERVICE, "line": 10, "kind": "computation",
         "scales_with": "page size", "bounded": True, "bound": "max 100"},
        {"description": "b", "file": SERVICE, "line": 10,
         "kind": "string_building", "scales_with": "line items",
         "bounded": False},
        {"description": "c", "file": SERVICE, "line": 10,
         "kind": "collection_processing", "scales_with": "line items",
         "bounded": False},
    ]))
    assert len(model.unbounded_per_row_work()) == 2


def test_response_shape_records_what_cannot_be_trimmed() -> None:
    """A returned collection whose size grows with the data is part of the API
    contract. The first patch attempt spent four build iterations learning that."""
    model = EndpointModel.model_validate(payload())
    assert model.response_shape is not None
    assert model.response_shape.unbounded_collections == ["tags"]


def test_pagination_records_the_count_query() -> None:
    model = EndpointModel.model_validate(payload(pagination={
        "mechanism": "offset", "fires_count_query": True}))
    assert model.pagination is not None
    assert model.pagination.fires_count_query is True
    assert model.pagination.max_page_size is None, "None means uncapped"


def test_entity_relations_record_batching_facts_without_judgement() -> None:
    """Whether an already-batched relation still needs work is diagnosis's
    call, so there is deliberately no `already_optimized` flag."""
    relation = EntityRelation(field="invoices", type="OneToMany", fetch="LAZY",
                              target="Invoice", batch_size=50,
                              fetch_mode="SUBSELECT")
    assert relation.batch_size == 50 and relation.fetch_mode == "SUBSELECT"
    assert not hasattr(relation, "already_optimized")

    plain = EntityRelation(field="x", type="ManyToOne", fetch="LAZY", target="Y")
    assert plain.batch_size is None and plain.fetch_mode is None


def test_the_rarely_used_shapes_exist_so_that_empty_is_honest() -> None:
    """An empty list is a statement; a missing field is a blind spot."""
    model = EndpointModel.model_validate(payload())
    assert model.caching == [] and model.hidden_query_triggers == []

    filled = EndpointModel.model_validate(payload(
        caching=[CacheInfo(kind="spring_cacheable", on="OrderService.search",
                           cache_name="orders",
                           citation=Citation(file=SERVICE, line=5)).model_dump()],
        hidden_query_triggers=[HiddenQueryTrigger(
            kind="formula", on="Order.total", detail="@Formula sums line items",
            citation=Citation(file=SERVICE, line=5)).model_dump()]))
    assert filled.caching[0].cache_name == "orders"
    assert filled.hidden_query_triggers[0].kind == "formula"


def test_all_citations_covers_the_new_shapes() -> None:
    model = EndpointModel.model_validate(payload(
        response_shape={"dto": "OrderDto",
                        "citation": {"file": CONTROLLER, "line": 12}},
        pagination={"mechanism": "offset",
                    "citation": {"file": CONTROLLER, "line": 12}},
        caching=[{"kind": "manual", "on": "x",
                  "citation": {"file": SERVICE, "line": 5}}],
        hidden_query_triggers=[{"kind": "post_load", "on": "Order", "detail": "d",
                                "citation": {"file": SERVICE, "line": 5}}]))
    refs = {c.ref() for c in model.all_citations()}
    assert f"{SERVICE}:10" in refs, "per_row_work"
    assert f"{CONTROLLER}:12" in refs
    assert f"{SERVICE}:5" in refs


def test_the_import_path_did_not_change_when_contracts_was_split() -> None:
    import senapathy.contracts as contracts

    for name in ("Citation", "EndpointModel", "PerRowWork", "ResponseShape",
                 "PaginationInfo", "CacheInfo", "HiddenQueryTrigger",
                 "LatencyStats", "GateDecision", "Usage"):
        assert hasattr(contracts, name), name


def test_the_existing_artifact_still_loads_under_the_new_schema() -> None:
    """EndpointModel gained only optional fields with defaults."""
    path = Path(__file__).resolve().parent.parent / ".senapathy" / "endpoint_model.json"
    if not path.exists():
        pytest.skip("no committed artifact to check")
    model = EndpointModel.model_validate(json.loads(path.read_text()))
    assert model.route
    assert model.per_row_work == [] or model.per_row_work


# ──────────────────────────  the submit tool  ────────────────────────────


def test_per_row_work_and_response_shape_are_required() -> None:
    """'None found' must be a statement the model made, not a field it left out."""
    assert "per_row_work" in SUBMIT_PARAMETERS["required"]
    assert "response_shape" in SUBMIT_PARAMETERS["required"]


def test_the_optional_shapes_are_offered_but_not_demanded() -> None:
    for name in ("pagination", "caching", "hidden_query_triggers"):
        assert name in SUBMIT_PARAMETERS["properties"], name
        assert name not in SUBMIT_PARAMETERS["required"], name


def test_base_commit_is_not_in_the_payload_schema() -> None:
    assert "base_commit" not in SUBMIT_PARAMETERS["properties"]


def test_an_omitted_per_row_work_is_rejected(root: SourceRoot) -> None:
    broken = payload()
    del broken["per_row_work"]
    with pytest.raises(ToolError, match="per_row_work"):
        make_submit_endpoint_analysis(root, {})(**broken)


def test_an_explicitly_empty_per_row_work_is_accepted(root: SourceRoot) -> None:
    sink: dict[str, Any] = {}
    message = make_submit_endpoint_analysis(root, sink)(
        **payload(per_row_work=[], data_access=[]))
    assert "accepted" in message
    assert sink["model"].per_row_work == []


def test_bounded_without_a_bound_is_rejected_and_says_what_is_missing(
    root: SourceRoot
) -> None:
    """'Something limits this, I decline to say what' is not a claim."""
    broken = payload(per_row_work=[
        {"description": "maps rows to DTOs", "file": SERVICE, "line": 10,
         "kind": "object_creation", "scales_with": "page size",
         "bounded": True}])
    with pytest.raises(ToolError) as caught:
        make_submit_endpoint_analysis(root, {})(**broken)
    message = str(caught.value)
    assert "bounded=true but gives no `bound`" in message
    assert "Name the limit" in message


def test_bounded_with_a_bound_is_accepted(root: SourceRoot) -> None:
    sink: dict[str, Any] = {}
    make_submit_endpoint_analysis(root, sink)(**payload(per_row_work=[
        {"description": "maps rows to DTOs", "file": SERVICE, "line": 10,
         "kind": "object_creation", "scales_with": "page size",
         "bounded": True, "bound": "page size, max 100"}]))
    assert sink["model"].per_row_work[0].bound == "page size, max 100"


def test_citation_problems_and_claim_problems_arrive_together(
    root: SourceRoot
) -> None:
    """One retry should be able to fix everything."""
    broken = payload(per_row_work=[
        {"description": "bad line", "file": SERVICE, "line": 9999,
         "kind": "computation", "scales_with": "rows", "bounded": True}])
    with pytest.raises(ToolError) as caught:
        make_submit_endpoint_analysis(root, {})(**broken)
    message = str(caught.value)
    assert "outside the file" in message
    assert "gives no `bound`" in message


def test_a_per_row_query_with_no_per_row_work_warns_but_does_not_reject(
    root: SourceRoot
) -> None:
    """A heuristic must not fail a submission — 'almost always' is not
    'always', and an agent right about an unusual endpoint must not be blocked."""
    sink: dict[str, Any] = {}
    message = make_submit_endpoint_analysis(root, sink)(**payload(
        per_row_work=[],
        data_access=[{"trigger": "lazy load of Order.lines",
                      "kind": "lazy_collection", "file": SERVICE, "line": 10,
                      "executes_per_row": True}]))
    assert "accepted" in message
    warnings = [u for u in sink["model"].uncertainties
                if u.startswith("AUTOMATED CHECK")]
    assert len(warnings) == 1
    assert "per_row_work is empty" in warnings[0]


def test_the_warning_is_silent_when_per_row_work_is_present(
    root: SourceRoot
) -> None:
    sink: dict[str, Any] = {}
    make_submit_endpoint_analysis(root, sink)(**payload(
        data_access=[{"trigger": "lazy load", "kind": "lazy_collection",
                      "file": SERVICE, "line": 10, "executes_per_row": True}]))
    assert not any(u.startswith("AUTOMATED CHECK")
                   for u in sink["model"].uncertainties)


def test_per_row_work_citations_are_validated(root: SourceRoot) -> None:
    bad = EndpointModel.model_validate(payload(per_row_work=[
        {"description": "x", "file": "src/main/java/com/ex/Ghost.java",
         "line": 1, "kind": "computation", "scales_with": "rows",
         "bounded": False}]))
    problems = validate_citations(bad, root)
    assert len(problems) == 1
    assert problems[0].where.startswith("per_row_work[0]")


# ────────────────────────────  base_commit  ──────────────────────────────


def test_base_commit_is_stamped_from_git_not_from_the_payload(
    fixture_repo: Path
) -> None:
    subprocess.run(["git", "init", "-q"], cwd=fixture_repo, check=True)
    subprocess.run(["git", "config", "user.email", "t@t"], cwd=fixture_repo,
                   check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=fixture_repo,
                   check=True)
    subprocess.run(["git", "add", "-A"], cwd=fixture_repo, check=True)
    subprocess.run(["git", "commit", "-qm", "fixture"], cwd=fixture_repo,
                   check=True)
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=fixture_repo,
                          capture_output=True, text=True, check=True).stdout.strip()

    root = SourceRoot.of(fixture_repo)
    sink: dict[str, Any] = {}
    message = make_submit_endpoint_analysis(root, sink)(
        **payload(base_commit="0000000000000000000000000000000000000000"))

    assert sink["model"].base_commit == head
    assert "discarded" in message, "the model should be told its sha was dropped"


def test_a_dirty_worktree_is_reported_as_dirty(fixture_repo: Path) -> None:
    """Honest provenance beats a clean-looking sha the artifact does not have."""
    subprocess.run(["git", "init", "-q"], cwd=fixture_repo, check=True)
    subprocess.run(["git", "config", "user.email", "t@t"], cwd=fixture_repo,
                   check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=fixture_repo,
                   check=True)
    subprocess.run(["git", "add", "-A"], cwd=fixture_repo, check=True)
    subprocess.run(["git", "commit", "-qm", "fixture"], cwd=fixture_repo,
                   check=True)
    (fixture_repo / "README.md").write_text("edited\n")

    assert (git_head(SourceRoot.of(fixture_repo)) or "").endswith("+dirty")


def test_a_repo_without_git_still_submits(fixture_repo: Path) -> None:
    sink: dict[str, Any] = {}
    make_submit_endpoint_analysis(SourceRoot.of(fixture_repo), sink)(**payload())
    assert sink["model"].base_commit is None


# ──────────────────────────────  prompt  ─────────────────────────────────


def test_the_phantom_tools_are_gone() -> None:
    """Telling a model about tools it cannot call invites it to try."""
    text = load_prompt("understand.system").text
    for phantom in ("get_callees", "get_callers", "find_symbol"):
        assert phantom not in text, phantom
    assert "Not yet available" not in text


def test_the_prompt_asks_the_bounded_question() -> None:
    text = load_prompt("understand.system").text
    assert "Per-row work that is not a database access" in text
    assert "is the amount of work\nbounded?" in text
    assert "UNBOUNDED means it scales with data and nothing caps it" in text
    assert "invisible in the SQL logs" in text


def test_the_prompt_covers_the_response_and_pagination() -> None:
    text = load_prompt("understand.system").text
    assert "What the endpoint returns" in text
    assert "part of the\nAPI contract" in text
    assert "## Pagination" in text
    assert "count query fires" in text


def test_the_prompt_says_an_empty_array_is_a_finding() -> None:
    text = load_prompt("understand.system").text
    assert "submit an empty array — that is a finding, not an omission" in text


def test_the_prompt_does_not_spend_weight_on_absent_features() -> None:
    """The target codebase has no caching, no @Formula, no @BatchSize. Prompt
    weight spent on absent features is taken from the ones that matter."""
    text = load_prompt("understand.system").text
    for absent in ("@BatchSize", "@Formula", "@Cacheable", "second-level"):
        assert absent not in text, absent


# ─────────────────────  Session 1 review items (Part D)  ─────────────────


def test_the_registry_is_ordered_the_way_the_prompt_presents_it(
    fixture_repo: Path
) -> None:
    """The docstring used to claim this and not do it."""
    registry = build_registry(SourceRoot.of(fixture_repo), {},
                              mysql=object())
    ordered = [n for n in PROMPT_ORDER if n in registry.names()]
    assert registry.names() == ordered


def test_the_registry_order_matches_the_prompt_text(fixture_repo: Path) -> None:
    text = load_prompt("understand.system").text
    positions = [text.index(name) for name in
                 ("find_route", "read_file", "get_schema", "search_code")]
    assert positions == sorted(positions)


class Scripted:
    def __init__(self, *turns: dict[str, Any] | str) -> None:
        self.turns = list(turns)

    def complete(self, **kwargs: Any) -> LLMResponse:
        turn = self.turns.pop(0) if self.turns else "nothing more"
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


def test_the_phase_runs_with_no_database_configured(
    monkeypatch, fixture_repo: Path, tmp_path: Path
) -> None:
    """The phase can do 90% of its job without a database and must not die at
    the door. `from_env()` validates, so a garbled port raised right there."""
    monkeypatch.setenv("SENAPATHY_MYSQL_PORT", "not-a-port")
    trace = Trace(tmp_path / "run", "r1")

    model = understand(
        repo=fixture_repo, endpoint="POST /api/orders/search",
        client=Scripted({"name": "submit_endpoint_analysis", "input": payload()}),
        trace=trace, budget=LoopBudget(max_steps=6), run_dir=tmp_path / "run",
    )

    assert model.tables == []
    notes = [e for e in trace.read_all()
             if e["type"] == "note" and "no database target" in str(e.get("note"))]
    assert len(notes) == 1
    tools = [e for e in trace.read_all() if e["type"] == "phase_start"][0]["tools"]
    assert "get_schema" not in tools


def test_get_schema_is_offered_when_a_target_exists(fixture_repo: Path,
                                                    tmp_path: Path) -> None:
    registry = build_registry(SourceRoot.of(fixture_repo), {}, mysql=object())
    assert "get_schema" in registry.names()


def test_max_steps_is_deprecated_loudly() -> None:
    from senapathy.agent import run_agent

    with pytest.deprecated_call(match="max_steps"):
        run_agent(client=Scripted("done"), task="go", tools=[], handlers={},
                  max_steps=1)


def test_claude_outputs_is_ignored() -> None:
    ignore = (Path(__file__).resolve().parent.parent / ".gitignore").read_text()
    assert "Claude outputs/" in ignore


# ─────────────────────────────  rendering  ───────────────────────────────


def unbounded_model() -> EndpointModel:
    return EndpointModel.model_validate(payload(
        base_commit="a3f91c2d4e5f60718293a4b5c6d7e8f901234567",
        pagination={"mechanism": "offset", "fires_count_query": True},
        per_row_work=[
            {"description": "builds one tag string per line item",
             "file": SERVICE, "line": 10, "kind": "string_building",
             "scales_with": "line items per order", "bounded": False},
            {"description": "maps each row to an OrderDto", "file": SERVICE,
             "line": 9, "kind": "object_creation", "scales_with": "page size",
             "bounded": True, "bound": "page size, max 100"},
        ]))


def test_the_unbounded_marker_is_the_point_of_the_session() -> None:
    from senapathy.render.endpoint import render_endpoint_model

    out = render_endpoint_model(unbounded_model())
    assert "⚠ UNBOUNDED" in out
    assert "PER-ROW WORK      2 item(s), 1 UNBOUNDED" in out
    # Unbounded first: the ordering is itself the finding.
    assert out.index("builds one tag string") < out.index("maps each row")


def test_the_render_shows_the_new_sections() -> None:
    from senapathy.render.endpoint import render_endpoint_model

    out = render_endpoint_model(unbounded_model())
    assert "base a3f91c2" in out
    assert "RESPONSE          OrderDto" in out
    assert "unbounded       tags" in out
    assert "part of the API contract" in out
    assert "PAGINATION        offset · count query fires · page size uncapped" in out


def test_an_absent_response_shape_says_so_rather_than_printing_nothing() -> None:
    from senapathy.render.endpoint import render_endpoint_model

    model = unbounded_model()
    model.response_shape = None
    model.pagination = None
    out = render_endpoint_model(model)
    assert "RESPONSE          (not recorded)" in out
    assert "PAGINATION        (not recorded)" in out


def test_an_empty_per_row_work_is_called_suspect_in_the_render() -> None:
    from senapathy.render.endpoint import render_endpoint_model

    model = unbounded_model()
    model.per_row_work = []
    assert "treat that as suspect" in render_endpoint_model(model)


def test_an_unbounded_collection_that_is_not_a_field_name_is_flagged(
    root: SourceRoot
) -> None:
    """The model's first real submission wrote 'tags (list of strings per
    CustomerDto)'. A later phase matches these against the response body, and
    a description matches nothing."""
    sink: dict[str, Any] = {}
    make_submit_endpoint_analysis(root, sink)(**payload(response_shape={
        "dto": "OrderDto",
        "fields_returned": ["id", "tags"],
        "unbounded_collections": ["tags (list of strings)"],
    }))
    warnings = [u for u in sink["model"].uncertainties
                if u.startswith("AUTOMATED CHECK")]
    assert len(warnings) == 1
    assert "do not appear in\nfields_returned" in warnings[0].replace(" ", " ") \
        or "fields_returned" in warnings[0]


def test_a_bare_field_name_is_not_flagged(root: SourceRoot) -> None:
    sink: dict[str, Any] = {}
    make_submit_endpoint_analysis(root, sink)(**payload(response_shape={
        "dto": "OrderDto",
        "fields_returned": ["id", "tags"],
        "unbounded_collections": ["tags"],
    }))
    assert not any(u.startswith("AUTOMATED CHECK")
                   for u in sink["model"].uncertainties)


def test_the_check_is_silent_when_no_fields_were_listed(root: SourceRoot) -> None:
    """Nothing to cross-check against is not the same as a mismatch."""
    sink: dict[str, Any] = {}
    make_submit_endpoint_analysis(root, sink)(**payload(response_shape={
        "dto": "OrderDto", "unbounded_collections": ["tags"]}))
    assert not any(u.startswith("AUTOMATED CHECK")
                   for u in sink["model"].uncertainties)


def test_an_unreachable_database_is_an_error_the_model_can_read(
    monkeypatch, fixture_repo: Path, tmp_path: Path
) -> None:
    """A target that can be DESCRIBED but not CONNECTED to must fail inside
    get_schema, where the model can read the message and carry on — not at the
    door, where it would abort a phase that can still do most of its job."""
    from senapathy.measure.sql import MySQLTarget

    monkeypatch.setenv("SENAPATHY_MYSQL_HOST", "127.0.0.1")
    monkeypatch.setenv("SENAPATHY_MYSQL_PORT", "1")   # nothing listens here
    monkeypatch.setenv("SENAPATHY_MYSQL_USER", "root")
    monkeypatch.setenv("SENAPATHY_MYSQL_PASSWORD", "root")
    monkeypatch.setenv("SENAPATHY_MYSQL_DATABASE", "mini")
    trace = Trace(tmp_path / "run", "r1")
    seen: list[str] = []

    class Watching(Scripted):
        def tool_result_messages(self, results):
            seen.extend(r.content for r in results)
            return super().tool_result_messages(results)

    model = understand(
        repo=fixture_repo, endpoint="POST /api/orders/search",
        client=Watching({"name": "get_schema", "input": {"tables": ["orders"]}},
                        {"name": "submit_endpoint_analysis", "input": payload()}),
        trace=trace, budget=LoopBudget(max_steps=6), run_dir=tmp_path / "run",
        mysql=MySQLTarget.from_env(),
    )

    assert model.tables == [], "no measured facts, and none invented"
    assert any("cannot connect to MySQL" in message for message in seen)
    assert any(e["type"] == "tool_result" and e["is_error"]
               for e in trace.read_all())


# ─────────────────  measured vs. never-looked (invariant 4)  ─────────────


def test_a_table_nobody_introspected_is_not_reported_as_having_no_indexes(
) -> None:
    """'NO INDEXES' and 'nobody looked' are different facts and only one of
    them is a finding. A run where the model skipped get_schema printed the
    first when the second was true."""
    from senapathy.render.endpoint import render_endpoint_model

    model = EndpointModel.model_validate(payload(
        tables=[{"table": "orders", "columns_read": ["id"]}]))
    out = render_endpoint_model(model)
    assert "not introspected" in out
    assert "NO INDEXES" not in out


def test_a_table_that_really_has_no_indexes_still_says_so() -> None:
    from senapathy.contracts import TableInfo
    from senapathy.render.endpoint import render_endpoint_model

    model = EndpointModel.model_validate(payload(tables=[TableInfo(
        table="orders", columns_read=["id"], all_columns=["id bigint NOT NULL"],
        approx_rows=10, existing_indexes=[]).model_dump()]))
    out = render_endpoint_model(model)
    assert "NO INDEXES" in out
    assert "not introspected" not in out


def test_a_table_with_no_measured_facts_is_flagged(root: SourceRoot) -> None:
    sink: dict[str, Any] = {}
    make_submit_endpoint_analysis(root, sink)(**payload(
        tables=[{"table": "orders", "columns_read": ["id"]}]))
    warnings = [u for u in sink["model"].uncertainties
                if u.startswith("AUTOMATED CHECK") and "get_schema" in u]
    assert len(warnings) == 1
    assert "not absent" in warnings[0]


def test_measured_tables_are_not_flagged(root: SourceRoot) -> None:
    from senapathy.contracts import TableInfo

    measured = {"orders": TableInfo(table="orders", approx_rows=10,
                                    all_columns=["id bigint NOT NULL"])}
    sink: dict[str, Any] = {}
    make_submit_endpoint_analysis(root, sink, measured_tables=measured)(
        **payload(tables=[{"table": "orders", "columns_read": ["id"]}]))
    assert not any("get_schema" in u for u in sink["model"].uncertainties)
