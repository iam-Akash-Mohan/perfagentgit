"""The plan phase: the policy table, the risk tiers, the terminal tool, the guard.

No network, no database, no LLM. The client is scripted and `get_schema` is
never registered, so nothing here can reach a provider or MySQL even by
accident.

The fixture in `recorded()` is the shape of the artifacts from the run that
motivated this phase: 1400 ms attributed to serializing a response whose
`tags` collection is unbounded, 200 ms to per-row ORM work, 100 ms to the
network. Ranking those by milliseconds picks the one the oracle can never
accept.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from perfagent.agent import LoopBudget
from perfagent.contracts import (CandidateFix, EndpointModel, Hypothesis,
                                 HypothesisSet, LLMResponse, Plan, RiskTier,
                                 Usage)
from perfagent.errors import ContractViolation, ToolError
from perfagent.plan.agent import load_plan, plan
from perfagent.plan.fixability import (assess_fixability, covering_index,
                                       rank_hypotheses, uncovered_columns)
from perfagent.plan.risk import (assess_risk, forbidden_reason,
                                 is_additive_index, parse_index,
                                 selection_problem)
from perfagent.plan.submit import make_submit_plan
from perfagent.tools import SourceRoot
from perfagent.trace import Trace

CONTROLLER = "src/main/java/com/ex/web/OrderController.java"
SERVICE = "src/main/java/com/ex/service/OrderService.java"


# ────────────────────────────────  fixtures  ─────────────────────────────


def hypothesis(**overrides: Any) -> Hypothesis:
    base: dict[str, Any] = {
        "id": "h1", "claim": "something is slow", "mechanism": "because",
        "layer": "other", "evidence": ["p50 is 1000 ms"],
        "attributed_ms": 100.0, "prediction": "it would be faster",
        "confidence": 0.5,
    }
    base.update(overrides)
    return Hypothesis.model_validate(base)


def endpoint_model(**overrides: Any) -> EndpointModel:
    """The complete map: per-row work, response shape, tables with indexes."""
    base: dict[str, Any] = {
        "route": "POST /api/orders/search",
        "call_chain": [
            {"method": "OrderController.search", "file": CONTROLLER,
             "line": 12, "does": "delegates"},
            {"method": "OrderService.search", "file": SERVICE, "line": 5,
             "does": "queries then maps each row"},
        ],
        "data_access": [
            {"trigger": "orderRepo.findByFilter", "kind": "jpql_query",
             "file": SERVICE, "line": 6, "tables": ["orders"],
             "executes_per_row": False},
            {"trigger": "lineRepo.sumOpenAmountByOrder", "kind": "jpql_query",
             "file": SERVICE, "line": 11, "tables": ["order_line"],
             "executes_per_row": True},
        ],
        "per_row_work": [
            {"description": "appends one 'overdue' tag string per overdue line",
             "file": SERVICE, "line": 11, "kind": "string_building",
             "scales_with": "lines per order", "bounded": False},
        ],
        "response_shape": {
            "dto": "OrderDto",
            "fields_returned": ["id", "name", "outstanding", "tags"],
            "computed_fields": ["outstanding", "tags"],
            "unbounded_collections": ["tags"],
        },
        "tables": [
            {"table": "orders", "entity": "Order",
             "columns_read": ["id", "name"],
             "all_columns": ["id", "name", "created", "updated", "notes",
                             "region"],
             "approx_rows": 50_000,
             "existing_indexes": [
                 {"name": "PRIMARY", "columns": ["id"], "unique": True},
             ]},
            {"table": "order_line", "entity": "OrderLine",
             "columns_read": ["order_id", "amount", "status"],
             "all_columns": ["id", "order_id", "amount", "status"],
             "approx_rows": 1_200_000,
             "existing_indexes": [
                 {"name": "PRIMARY", "columns": ["id"], "unique": True},
                 {"name": "idx_line_order_id", "columns": ["order_id"],
                  "unique": False},
             ]},
        ],
        "base_commit": "abc123",
    }
    base.update(overrides)
    return EndpointModel.model_validate(base)


def recorded() -> tuple[HypothesisSet, EndpointModel]:
    """The diagnosis that made this phase necessary, in miniature."""
    hypotheses = HypothesisSet(
        measured_total_ms=1779.7,
        ranked_ids=["h1", "h2", "h3"],
        probes_run=[],
        hypotheses=[
            hypothesis(
                id="h1", layer="serialization", attributed_ms=1400.0,
                claim="JSON serialization of the response DTOs dominates the "
                      "non-SQL latency",
                mechanism="each entity is mapped to a DTO and written to JSON; "
                          "the time grows with the number of bytes produced",
                prediction="a slimmer DTO would cut latency proportionally",
                evidence=["page_size=1: body 142297 bytes, p50 3587 ms"],
            ),
            hypothesis(
                id="h2", layer="orm_hydration", attributed_ms=200.0,
                claim="ORM hydration and per-row enrichment, including lazy "
                      "loading of lines, adds CPU per result row",
                mechanism="for each result the service calls "
                          "sumOpenAmountByOrder, which lazily fetches the "
                          "order_line collection",
                prediction="not touching the lazy collection drops latency",
                evidence=["SQL statements per request rise with rows"],
            ),
            hypothesis(
                id="h3", layer="network", attributed_ms=100.0,
                claim="network transfer of the large JSON payload contributes",
                mechanism="the response is written to the socket uncompressed",
                prediction="gzip would reduce it",
                evidence=["zero-row request returns 349 bytes in 14 ms"],
            ),
        ],
    )
    return hypotheses, endpoint_model()


@pytest.fixture
def root(fixture_repo: Path) -> SourceRoot:
    return SourceRoot.of(fixture_repo)


# ──────────────────────  the policy table, row by row  ───────────────────


def test_a_serialization_claim_about_an_unbounded_collection_is_blocked() -> None:
    model = endpoint_model()
    verdict = assess_fixability(
        hypothesis(layer="serialization",
                   claim="serializing the response body dominates latency",
                   mechanism="the response grows with the number of bytes"),
        model,
    )
    assert verdict.fixability == 0.05
    assert verdict.blocked_by is not None and "tags" in verdict.blocked_by
    assert "API contract" in verdict.reason


def test_a_hypothesis_naming_the_unbounded_field_is_blocked_in_any_layer() -> None:
    verdict = assess_fixability(
        hypothesis(layer="application_logic",
                   claim="building the tags list costs the most"),
        endpoint_model(),
    )
    assert verdict.fixability == 0.05
    assert "tags" in (verdict.blocked_by or "")


def test_a_per_row_query_is_batchable() -> None:
    verdict = assess_fixability(
        hypothesis(layer="database_query",
                   claim="sumOpenAmountByOrder runs once per result row"),
        endpoint_model(),
    )
    assert verdict.fixability == 0.90
    assert "fetch join" in verdict.reason
    assert verdict.blocked_by is None


def test_a_missing_index_is_the_most_fixable_thing_there_is() -> None:
    verdict = assess_fixability(
        hypothesis(layer="database_query",
                   claim="the filter query scans the table",
                   mechanism="the plan reports no index used on order_line",
                   evidence=["no_index_used=true, rows examined 1,120,858"]),
        # No per-row access, so the batchable row cannot fire first.
        endpoint_model(data_access=[
            {"trigger": "orderRepo.findByFilter", "kind": "jpql_query",
             "file": SERVICE, "line": 6, "tables": ["orders"],
             "executes_per_row": False},
        ]),
    )
    assert verdict.fixability == 0.95
    assert "additive index" in verdict.reason


def test_a_narrow_projection_over_a_wide_row() -> None:
    verdict = assess_fixability(
        hypothesis(layer="orm_hydration",
                   claim="entity hydration materialises whole rows"),
        endpoint_model(data_access=[]),      # nothing per-row to batch
    )
    assert verdict.fixability == 0.80
    assert "projection" in verdict.reason


def test_per_row_work_that_fills_the_contract_is_blocked() -> None:
    verdict = assess_fixability(
        hypothesis(layer="application_logic",
                   claim="per-row string building dominates",
                   mechanism="it appends one overdue string per overdue line, "
                             "scaling with lines per order"),
        endpoint_model(response_shape={
            "dto": "OrderDto", "fields_returned": ["id", "tags"],
            "unbounded_collections": ["tags"],
        }),
    )
    # The work produces `tags`; the claim does not name the field, so this is
    # row 5 rather than row 1.
    assert verdict.fixability == 0.10
    assert "overdue" in (verdict.blocked_by or "") or "tags" in (verdict.blocked_by or "")


def test_per_row_work_that_never_reaches_the_response_is_removable() -> None:
    verdict = assess_fixability(
        hypothesis(layer="application_logic",
                   claim="per-row audit logging dominates",
                   mechanism="it writes one audit log line per order row, "
                             "scaling with orders returned"),
        endpoint_model(
            per_row_work=[{
                "description": "writes one audit log line per order",
                "file": SERVICE, "line": 9, "kind": "other",
                "scales_with": "orders returned", "bounded": False,
            }],
            response_shape={"dto": "OrderDto", "fields_returned": ["id"],
                            "unbounded_collections": []},
        ),
    )
    assert verdict.fixability == 0.70
    assert "removable" in verdict.reason


def test_serialization_without_an_unbounded_collection_is_merely_hard() -> None:
    verdict = assess_fixability(
        hypothesis(layer="serialization", claim="jackson is slow"),
        endpoint_model(response_shape={"dto": "OrderDto",
                                       "fields_returned": ["id"],
                                       "unbounded_collections": []}),
    )
    assert verdict.fixability == 0.30


def test_the_network_is_out_of_scope() -> None:
    verdict = assess_fixability(hypothesis(layer="network"), endpoint_model())
    assert verdict.fixability == 0.15
    assert verdict.blocked_by is not None


def test_anything_else_is_neither_favoured_nor_excluded() -> None:
    verdict = assess_fixability(
        hypothesis(layer="other", claim="the JVM is warming up"),
        endpoint_model(),
    )
    assert verdict.fixability == 0.40
    assert verdict.blocked_by is None


def test_fixability_can_never_be_attributed_to_a_model() -> None:
    from pydantic import ValidationError

    from perfagent.contracts import FixabilityAssessment

    verdict = assess_fixability(hypothesis(), endpoint_model())
    assert verdict.assessed_by == "code"
    with pytest.raises(ValidationError):
        FixabilityAssessment.model_validate(
            {**verdict.model_dump(), "assessed_by": "model"})


# ─────────────────────────  the ranking, end to end  ─────────────────────


def test_expected_value_puts_the_fixable_cause_above_the_larger_one() -> None:
    """The whole session, in one assertion.

    1400 ms of serialization loses to 200 ms of ORM work, because the response
    body is the API contract and the per-row query is batchable.
    """
    hypotheses, model = recorded()
    ranked = rank_hypotheses(hypotheses, model)

    assert [entry.hypothesis_id for entry in ranked] == ["h2", "h1", "h3"]
    assert [entry.expected_value_ms for entry in ranked] == [180.0, 70.0, 15.0]

    serialization = next(e for e in ranked if e.hypothesis_id == "h1")
    assert serialization.fixability.fixability == 0.05
    assert "tags" in (serialization.fixability.blocked_by or "")


def test_ties_break_on_the_bigger_measured_cost() -> None:
    hypotheses = HypothesisSet(
        measured_total_ms=1000.0, ranked_ids=["a", "b", "c"], probes_run=[],
        hypotheses=[
            hypothesis(id="a", layer="network", attributed_ms=100.0),
            hypothesis(id="b", layer="network", attributed_ms=200.0),
            hypothesis(id="c", layer="network", attributed_ms=50.0),
        ],
    )
    ranked = rank_hypotheses(hypotheses, endpoint_model())
    assert [entry.hypothesis_id for entry in ranked] == ["b", "a", "c"]


# ──────────────────────────  index facts from the map  ───────────────────


def test_an_existing_index_is_recognised_as_a_prefix() -> None:
    model = endpoint_model()
    line = next(t for t in model.tables if t.table == "order_line")
    assert covering_index(line, ["order_id"]) == "idx_line_order_id"
    # A longer list is NOT covered: (order_id) cannot serve (order_id, status).
    assert covering_index(line, ["order_id", "status"]) is None
    assert uncovered_columns(line) == ["amount", "status"]


def test_create_index_parsing_refuses_anything_else() -> None:
    assert is_additive_index("CREATE INDEX idx_a ON invoice (customer_id);")
    assert parse_index("CREATE INDEX idx_a ON invoice (customer_id, status)") == (
        "idx_a", "invoice", ["customer_id", "status"])
    assert not is_additive_index("ALTER TABLE invoice ADD COLUMN x INT")
    assert not is_additive_index("DROP INDEX idx_a ON invoice")
    # The classic smuggle: a legal statement followed by an illegal one.
    assert not is_additive_index(
        "CREATE INDEX idx_a ON invoice (customer_id); DROP TABLE invoice;")


# ──────────────────────────────  risk tiers  ─────────────────────────────


@pytest.mark.parametrize("approach,expected", [
    ("add_index", RiskTier.LOW),
    ("entity_graph", RiskTier.LOW),
    ("fetch_join", RiskTier.LOW),
    ("read_only_transaction", RiskTier.LOW),
    ("dto_projection", RiskTier.LOW),
    ("batch_query", RiskTier.MEDIUM),
    ("query_rewrite", RiskTier.MEDIUM),
    ("remove_redundant_call", RiskTier.MEDIUM),
    ("move_work_out_of_loop", RiskTier.MEDIUM),
    ("other", RiskTier.HIGH),
])
def test_the_tier_table_is_fixed(approach: str, expected: RiskTier) -> None:
    tier, flag, _why = assess_risk(approach=approach, description="a change")
    assert tier is expected
    assert flag is (expected >= RiskTier.HIGH)


def test_caching_is_tier_three_however_it_is_described() -> None:
    tier, flag, why = assess_risk(
        approach="remove_redundant_call",
        description="keep the computed result in a Caffeine cache between "
                    "requests",
    )
    assert tier is RiskTier.HIGH and flag is True
    assert "caching" in why


def test_a_transaction_boundary_change_is_tier_three() -> None:
    tier, flag, _why = assess_risk(
        approach="query_rewrite",
        description="run the sum in a new transaction with REQUIRES_NEW "
                    "propagation",
    )
    assert tier is RiskTier.HIGH and flag is True


def test_read_only_transaction_is_not_mistaken_for_a_boundary_change() -> None:
    tier, flag, _why = assess_risk(
        approach="read_only_transaction",
        description="add @Transactional(readOnly = true) to the search method",
    )
    assert tier is RiskTier.LOW and flag is False


@pytest.mark.parametrize("kwargs,fragment", [
    ({"files_to_touch": ["src/test/java/com/ex/OrderIT.java"]}, "immutable"),
    ({"sql_change": "ALTER TABLE orders ADD COLUMN total DECIMAL(19,2)"},
     "additive CREATE INDEX"),
    ({"description": "drop the unused index and create table order_summary"},
     "schema change"),
    ({"description": "relax the @PreAuthorize check so the query is cheaper"},
     "authentication"),
])
def test_forbidden_changes_are_refused_at_plan_time(kwargs: dict[str, Any],
                                                    fragment: str) -> None:
    args: dict[str, Any] = {"approach": "query_rewrite", "description": "x",
                            "sql_change": None, "files_to_touch": [SERVICE]}
    args.update(kwargs)
    reason = forbidden_reason(**args)
    assert reason is not None and fragment in reason


def test_a_legitimate_candidate_is_not_forbidden() -> None:
    assert forbidden_reason(
        approach="add_index", description="add an index on (order_id, status)",
        sql_change="CREATE INDEX idx_line_order_id_status ON order_line "
                   "(order_id, status);",
        files_to_touch=[SERVICE],
    ) is None


def test_selection_refuses_tier_three_and_byte_changers() -> None:
    def candidate(**overrides: Any) -> CandidateFix:
        base: dict[str, Any] = {
            "id": "c1", "targets_hypothesis": "h2", "approach": "fetch_join",
            "description": "d", "mechanism_changed": "SQL statements per request",
            "predicted_effect": "6 -> 2", "files_to_touch": [SERVICE],
            "what_could_break": ["ordering"], "preserves_response_bytes": True,
            "risk_tier": RiskTier.LOW,
        }
        base.update(overrides)
        return CandidateFix.model_validate(base)

    assert selection_problem(candidate()) is None
    assert "risk tier" in (selection_problem(candidate(risk_tier=RiskTier.HIGH)) or "")
    assert "response bytes" in (
        selection_problem(candidate(preserves_response_bytes=False)) or "")


# ────────────────────────────  the terminal tool  ────────────────────────


def submit_handler(root: SourceRoot, sink: dict[str, Any],
                   model: EndpointModel | None = None):
    hypotheses, default_model = recorded()
    model = model or default_model
    ranked = rank_hypotheses(hypotheses, model)
    return make_submit_plan(
        root=root, model=model, ranked=ranked,
        considered=[r.hypothesis_id for r in ranked[:2]], sink=sink,
        endpoint="POST /api/orders/search",
        measured_total_ms=hypotheses.measured_total_ms,
    )


def candidate_payload(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "id": "c1", "targets_hypothesis": "h2", "approach": "fetch_join",
        "description": "replace the per-row sum with a join fetch",
        "mechanism_changed": "SQL statements per request",
        "predicted_effect": "queries per request drop from 22 to 2",
        "files_to_touch": [SERVICE],
        "what_could_break": ["duplicate rows if the join is not distinct"],
        "preserves_response_bytes": True,
    }
    base.update(overrides)
    return base


INDEX_CANDIDATE = candidate_payload(
    id="c2", approach="add_index",
    description="index order_line on (order_id, status)",
    mechanism_changed="rows examined",
    predicted_effect="rows examined per request drop from 1.1M to 42",
    sql_change="CREATE INDEX idx_line_order_id_status ON order_line "
               "(order_id, status);",
    what_could_break=["slower writes to order_line"],
)


def test_a_two_candidate_plan_is_accepted_and_one_is_selected(
    root: SourceRoot
) -> None:
    sink: dict[str, Any] = {}
    message = submit_handler(root, sink)(
        candidates=[candidate_payload(), INDEX_CANDIDATE],
        reasoning="batching first, index second",
    )
    result: Plan = sink["plan"]

    assert "accepted" in message
    assert len(result.candidates) == 2
    assert {c.approach for c in result.candidates} == {"fetch_join", "add_index"}
    assert result.selected_id is not None
    assert result.selected().risk_tier <= RiskTier.MEDIUM
    # Expected value comes from the hypothesis, not from the model.
    assert result.selected().expected_value_ms == 180.0
    # And the loser is recorded with a reason.
    assert len(result.not_selected) == 1
    assert result.not_selected[0].reason


def test_an_index_that_already_exists_is_refused_and_named(
    root: SourceRoot
) -> None:
    duplicate = candidate_payload(
        id="c2", approach="add_index", description="index the foreign key",
        mechanism_changed="rows examined", predicted_effect="fewer rows",
        sql_change="CREATE INDEX idx_order_line_order_id ON order_line "
                   "(order_id);",
    )
    with pytest.raises(ToolError) as caught:
        submit_handler(root, {})(candidates=[candidate_payload(), duplicate])
    assert "idx_line_order_id" in str(caught.value)
    assert "already covers" in str(caught.value)


def test_an_index_cannot_be_proposed_without_a_measured_index_list(
    root: SourceRoot
) -> None:
    """The comprehension map's index lists are EMPTY when it was built without
    a reachable database. Treating that as "no index exists" is how a
    duplicate gets proposed, so an unmeasured table fails closed."""
    blind = endpoint_model(tables=[
        {"table": "order_line", "entity": "OrderLine",
         "columns_read": ["order_id"], "existing_indexes": []},
    ])
    with pytest.raises(ToolError) as caught:
        submit_handler(root, {}, blind)(
            candidates=[candidate_payload(), INDEX_CANDIDATE])
    assert "no index list has been measured" in str(caught.value)
    assert "get_schema" in str(caught.value)


def test_get_schema_facts_beat_the_map(root: SourceRoot) -> None:
    """What the database said during THIS conversation wins."""
    from perfagent.contracts import TableInfo

    blind = endpoint_model(tables=[
        {"table": "order_line", "columns_read": ["order_id"],
         "existing_indexes": []},
    ])
    hypotheses, _ = recorded()
    ranked = rank_hypotheses(hypotheses, blind)
    handler = make_submit_plan(
        root=root, model=blind, ranked=ranked,
        considered=[r.hypothesis_id for r in ranked[:2]], sink={},
        endpoint="POST /api/orders/search",
        measured_total_ms=hypotheses.measured_total_ms,
        measured_tables={"order_line": TableInfo.model_validate({
            "table": "order_line", "columns_read": ["order_id"],
            "existing_indexes": [{"name": "idx_line_order_id",
                                  "columns": ["order_id", "status"],
                                  "unique": False}],
        })},
    )
    with pytest.raises(ToolError, match="already covers"):
        handler(candidates=[candidate_payload(), INDEX_CANDIDATE])


def test_an_index_on_a_table_off_the_path_is_refused(root: SourceRoot) -> None:
    stray = candidate_payload(
        id="c2", approach="add_index", description="index another table",
        sql_change="CREATE INDEX idx_x ON audit_log (created);",
    )
    with pytest.raises(ToolError, match="not a table on this endpoint"):
        submit_handler(root, {})(candidates=[candidate_payload(), stray])


@pytest.mark.parametrize("count", [1, 5])
def test_the_number_of_candidates_is_bounded(root: SourceRoot,
                                             count: int) -> None:
    candidates = [candidate_payload(id=f"c{n}") for n in range(count)]
    with pytest.raises(ToolError, match="between 2 and 4"):
        submit_handler(root, {})(candidates=candidates)


def test_a_candidate_for_a_hypothesis_nobody_asked_about_is_refused(
    root: SourceRoot
) -> None:
    with pytest.raises(ToolError) as caught:
        submit_handler(root, {})(
            candidates=[candidate_payload(),
                        candidate_payload(id="c2", targets_hypothesis="h3",
                                          approach="add_index")])
    assert "h3" in str(caught.value) and "h2" in str(caught.value)


def test_the_top_ranked_hypothesis_must_be_attacked(root: SourceRoot) -> None:
    with pytest.raises(ToolError, match="highest expected value"):
        submit_handler(root, {})(candidates=[
            candidate_payload(id="c1", targets_hypothesis="h1"),
            candidate_payload(id="c2", targets_hypothesis="h1",
                              approach="dto_projection"),
        ])


def test_one_approach_stated_twice_is_not_a_choice(root: SourceRoot) -> None:
    with pytest.raises(ToolError, match="same approach"):
        submit_handler(root, {})(candidates=[
            candidate_payload(id="c1"),
            candidate_payload(id="c2", description="a slightly different join"),
        ])


def test_a_file_that_does_not_exist_cannot_be_in_the_allowlist(
    root: SourceRoot
) -> None:
    with pytest.raises(ToolError, match="does not exist"):
        submit_handler(root, {})(candidates=[
            candidate_payload(files_to_touch=[
                "src/main/java/com/ex/service/OrdreService.java"]),
            INDEX_CANDIDATE,
        ])


def test_a_new_migration_file_is_the_one_allowed_exception(
    root: SourceRoot
) -> None:
    sink: dict[str, Any] = {}
    submit_handler(root, sink)(candidates=[
        candidate_payload(),
        candidate_payload(
            **{**INDEX_CANDIDATE,
               "files_to_touch": ["src/main/resources/db/migration/"
                                  "V2__add_order_line_index.sql"]}),
    ])
    assert sink["plan"].selected_id is not None


def test_a_test_file_in_the_allowlist_is_refused(root: SourceRoot) -> None:
    with pytest.raises(ToolError, match="immutable"):
        submit_handler(root, {})(candidates=[
            candidate_payload(files_to_touch=["src/test/java/com/ex/OrderIT.java"]),
            INDEX_CANDIDATE,
        ])


def test_a_path_escape_in_the_allowlist_is_refused(root: SourceRoot) -> None:
    from perfagent.errors import GuardrailError

    with pytest.raises(GuardrailError, match="outside the repository"):
        submit_handler(root, {})(candidates=[
            candidate_payload(files_to_touch=["../../etc/passwd"]),
            INDEX_CANDIDATE,
        ])


def test_a_tier_three_candidate_is_recorded_flagged_and_never_selected(
    root: SourceRoot
) -> None:
    sink: dict[str, Any] = {}
    submit_handler(root, sink)(candidates=[
        candidate_payload(id="c1", approach="other",
                          description="cache the outstanding totals in a "
                                      "Caffeine cache"),
        candidate_payload(id="c2", approach="batch_query",
                          description="load every order's lines in one IN query"),
    ])
    result: Plan = sink["plan"]
    cached = next(c for c in result.candidates if c.id == "c1")

    assert cached.risk_tier is RiskTier.HIGH and cached.requires_flag is True
    assert result.selected_id == "c2"
    assert any(r.candidate_id == "c1" and "risk tier" in r.reason
               for r in result.not_selected)


def test_a_candidate_that_changes_the_response_cannot_be_selected(
    root: SourceRoot
) -> None:
    sink: dict[str, Any] = {}
    submit_handler(root, sink)(candidates=[
        candidate_payload(id="c1", preserves_response_bytes=False),
        candidate_payload(id="c2", approach="batch_query",
                          description="one IN query for every order's lines"),
    ])
    result: Plan = sink["plan"]
    assert result.selected_id == "c2"
    assert any("response bytes" in r.reason for r in result.not_selected)


def test_a_plan_where_nothing_is_selectable_says_so(root: SourceRoot) -> None:
    sink: dict[str, Any] = {}
    submit_handler(root, sink)(candidates=[
        candidate_payload(id="c1", preserves_response_bytes=False),
        candidate_payload(id="c2", approach="add_index",
                          preserves_response_bytes=False,
                          sql_change="CREATE INDEX idx_line_status ON "
                                     "order_line (status);"),
    ])
    result: Plan = sink["plan"]
    assert result.selected_id is None
    assert "response bytes" in result.selection_reason
    assert len(result.not_selected) == 2


def test_risk_and_expected_value_are_not_in_the_tool_schema() -> None:
    """A model that could set its own tier would set it to LOW."""
    from perfagent.plan.submit import CANDIDATE_SCHEMA

    properties = set(CANDIDATE_SCHEMA["properties"])
    assert not properties & {"risk_tier", "requires_flag", "expected_value_ms"}


# ────────────────────────────────  the phase  ────────────────────────────


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


def run_phase(client: Any, repo: Path, tmp_path: Path) -> Plan:
    hypotheses, model = recorded()
    return plan(
        repo=repo, endpoint="POST /api/orders/search", hypotheses=hypotheses,
        model=model, client=client, trace=Trace(tmp_path / "run", "r1"),
        budget=LoopBudget(max_steps=8), run_dir=tmp_path / "run",
        mysql=None,     # no database in tests; get_schema is not registered
    )


def test_the_phase_writes_a_validated_artifact(fixture_repo: Path,
                                               tmp_path: Path) -> None:
    client = Scripted(
        {"name": "read_file", "input": {"path": SERVICE}},
        {"name": "submit_plan",
         "input": {"candidates": [candidate_payload(), INDEX_CANDIDATE]}},
    )
    result = run_phase(client, fixture_repo, tmp_path)

    assert result.selected_id is not None
    assert result.prompt_hash is not None
    assert result.base_commit == "abc123"
    assert [r.hypothesis_id for r in result.ranked] == ["h2", "h1", "h3"]

    written = load_plan(tmp_path / "run" / "plan.json")
    assert written == result


def test_the_ranking_reaches_the_prompt_with_its_reasons(
    fixture_repo: Path, tmp_path: Path
) -> None:
    seen: list[str] = []

    class Watching(Scripted):
        def complete(self, **kwargs: Any) -> LLMResponse:
            seen.append(str(kwargs.get("system")) + " "
                        + str(kwargs["messages"][0]["content"]))
            return super().complete(**kwargs)

    run_phase(
        Watching({"name": "submit_plan",
                  "input": {"candidates": [candidate_payload(),
                                           INDEX_CANDIDATE]}}),
        fixture_repo, tmp_path,
    )
    prompt = seen[0]
    # The constraint, the field it protects, the fixability and its reason, and
    # the indexes that already exist — all templated from artifacts.
    assert "BYTE FOR BYTE" in prompt
    assert "tags" in prompt
    assert "0.05" in prompt and "API contract" in prompt
    assert "idx_line_order_id" in prompt
    assert "1779.7" in prompt


def test_a_rejected_plan_is_retried_by_the_agent(fixture_repo: Path,
                                                 tmp_path: Path) -> None:
    duplicate = candidate_payload(
        id="c2", approach="add_index",
        sql_change="CREATE INDEX idx_dup ON order_line (order_id);")
    client = Scripted(
        {"name": "submit_plan",
         "input": {"candidates": [candidate_payload(), duplicate]}},
        {"name": "submit_plan",
         "input": {"candidates": [candidate_payload(), INDEX_CANDIDATE]}},
    )
    result = run_phase(client, fixture_repo, tmp_path)
    assert len(result.candidates) == 2


def test_a_phase_that_never_submits_raises(fixture_repo: Path,
                                           tmp_path: Path) -> None:
    client = Scripted(*["here is a prose plan instead"] * 6)
    with pytest.raises(ContractViolation) as caught:
        run_phase(client, fixture_repo, tmp_path)
    assert "without submitting" in str(caught.value)
    assert "here is a prose plan instead" in str(caught.value)


def test_the_transcript_does_not_leave_the_phase(fixture_repo: Path,
                                                 tmp_path: Path) -> None:
    result = run_phase(
        Scripted({"name": "submit_plan",
                  "input": {"candidates": [candidate_payload(),
                                           INDEX_CANDIDATE]}}),
        fixture_repo, tmp_path,
    )
    assert isinstance(result, Plan)
    assert not hasattr(result, "messages") and not hasattr(result, "transcript")


def test_the_phase_offers_no_tool_that_writes(fixture_repo: Path) -> None:
    from perfagent.plan.agent import build_registry

    hypotheses, model = recorded()
    ranked = rank_hypotheses(hypotheses, model)
    registry = build_registry(
        SourceRoot.of(fixture_repo), {}, model=model, ranked=ranked,
        considered=["h2", "h1"], endpoint="POST /api/orders/search",
        measured_total_ms=hypotheses.measured_total_ms,
    )
    assert registry.terminal_name() == "submit_plan"
    assert set(registry.names()) == {"find_route", "read_file", "search_code",
                                     "list_dir", "submit_plan"}


def test_the_phase_can_read_the_live_schema_when_a_target_exists(
    fixture_repo: Path
) -> None:
    """get_schema is in the surface so a proposed index can be checked against
    the database rather than against the model's memory of it. Constructing a
    target opens no connection."""
    from perfagent.measure.sql import MySQLTarget
    from perfagent.plan.agent import build_registry

    hypotheses, model = recorded()
    registry = build_registry(
        SourceRoot.of(fixture_repo), {}, model=model,
        ranked=rank_hypotheses(hypotheses, model), considered=["h2", "h1"],
        endpoint="POST /api/orders/search",
        measured_total_ms=hypotheses.measured_total_ms,
        mysql=MySQLTarget(),
    )
    assert "get_schema" in registry.names()
    # Order is part of the prompt whether you intended it or not.
    assert registry.names()[-1] == "submit_plan"


# ───────────────────────────  wiring and guards  ─────────────────────────


def test_the_patch_phase_is_refused_without_a_plan(tmp_path: Path) -> None:
    """Acceptance 5. The guard reads the artifact, not the run state."""
    from perfagent import orchestrator as orch
    from perfagent.contracts import Phase, RunState

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "hypotheses.json").write_text("{}")
    ctx = orch.Context(run_id="r1", run_dir=run_dir, repo=tmp_path,
                       endpoint="POST /x", budget=orch.Budget(),
                       trace=Trace(run_dir, "r1"))
    state = RunState(run_id="r1", repo=str(tmp_path), endpoint="POST /x",
                     phase=Phase.DIAGNOSE)

    blocked = orch._preconditions(orch.BY_NAME["patch"], ctx, state)
    assert blocked is not None and "plan.json" in blocked


def test_the_plan_phase_is_refused_without_a_diagnosis(tmp_path: Path) -> None:
    from perfagent import orchestrator as orch
    from perfagent.contracts import Phase, RunState

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    ctx = orch.Context(run_id="r1", run_dir=run_dir, repo=tmp_path,
                       endpoint="POST /x", budget=orch.Budget(),
                       trace=Trace(run_dir, "r1"))
    state = RunState(run_id="r1", repo=str(tmp_path), endpoint="POST /x",
                     phase=Phase.BASELINE)

    blocked = orch._preconditions(orch.BY_NAME["plan"], ctx, state)
    assert blocked is not None and "hypotheses.json" in blocked


def test_a_plan_with_no_selection_does_not_let_the_phase_be_skipped(
    tmp_path: Path
) -> None:
    from perfagent import orchestrator as orch

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    empty = Plan(endpoint="POST /x", measured_total_ms=10.0, ranked=[],
                 selection_reason="nothing cleared the bar")
    path = run_dir / "plan.json"
    path.write_text(json.dumps(empty.model_dump(mode="json")))

    ctx = orch.Context(run_id="r1", run_dir=run_dir, repo=tmp_path,
                       endpoint="POST /x", budget=orch.Budget(),
                       trace=Trace(run_dir, "r1"))
    assert "no selected candidate" in (orch._valid_plan(ctx, path) or "")


def test_a_plan_for_another_endpoint_does_not_satisfy_this_run(
    fixture_repo: Path, tmp_path: Path
) -> None:
    from perfagent import orchestrator as orch

    result = run_phase(
        Scripted({"name": "submit_plan",
                  "input": {"candidates": [candidate_payload(),
                                           INDEX_CANDIDATE]}}),
        fixture_repo, tmp_path,
    )
    assert result.selected_id is not None

    path = tmp_path / "run" / "plan.json"
    ctx = orch.Context(run_id="r1", run_dir=tmp_path / "run", repo=fixture_repo,
                       endpoint="GET /api/orders/{id}", budget=orch.Budget(),
                       trace=Trace(tmp_path / "run", "r1"))
    assert "but this run is about" in (orch._valid_plan(ctx, path) or "")


def test_the_gate_reads_the_plan_artifact_and_refuses_to_invent_one(
    fixture_repo: Path, tmp_path: Path
) -> None:
    """`plan_from_patch` is gone: a plan derived from the patch proves nothing."""
    import perfagent.ship as ship

    assert not hasattr(ship, "plan_from_patch")

    with pytest.raises(ContractViolation, match="no plan at"):
        ship.selected_plan(tmp_path)

    run_phase(
        Scripted({"name": "submit_plan",
                  "input": {"candidates": [candidate_payload(),
                                           INDEX_CANDIDATE]}}),
        fixture_repo, tmp_path,
    )
    candidate = ship.selected_plan(tmp_path / "run")
    assert candidate.files_to_touch == [SERVICE]
    assert candidate.risk_tier is RiskTier.LOW
