from __future__ import annotations
import pytest
from pydantic import ValidationError
from perfagent.contracts import (
    Citation, EndpointModel, HypothesisSet, Hypothesis,
    RiskTier, TerminalState, EquivalenceVerdict, RequestCorpus, RequestSpec,
)


def test_citation_ref():
    assert Citation(file="a/B.java", line=34).ref() == "a/B.java:34"


def test_hypothesis_set_requires_three():
    """Forced competition: fewer than 3 hypotheses is a contract violation,
    because a single hypothesis means the model locked onto its first idea."""
    one = Hypothesis(id="h1", claim="c", mechanism="m", layer="database_query",
                     evidence=["e"], attributed_ms=100, prediction="p",
                     confidence=0.5)
    with pytest.raises(ValidationError):
        HypothesisSet(hypotheses=[one], ranked_ids=["h1"], measured_total_ms=200)


def test_attribution_validity():
    def h(i, ms):
        return Hypothesis(id=f"h{i}", claim="c", mechanism="m",
                          layer="database_query", evidence=["e"],
                          attributed_ms=ms, prediction="p", confidence=0.5)
    ok = HypothesisSet(hypotheses=[h(1, 100), h(2, 50), h(3, 20)],
                       ranked_ids=["h1", "h2", "h3"], measured_total_ms=200)
    assert ok.attribution_valid()
    bad = HypothesisSet(hypotheses=[h(1, 5000), h(2, 50), h(3, 20)],
                        ranked_ids=["h1", "h2", "h3"], measured_total_ms=200)
    assert not bad.attribution_valid()


def test_confidence_bounded():
    with pytest.raises(ValidationError):
        Hypothesis(id="h", claim="c", mechanism="m", layer="database_query",
                   evidence=[], attributed_ms=1, prediction="p", confidence=1.4)


def test_verdict_decided_by_code_only():
    """A verdict must never be attributable to the model."""
    v = EquivalenceVerdict(passed=True, compared=10, identical=10,
                           db_state_matched=True, external_calls_matched=True,
                           held_back_passed=True, oracle_hash="abc")
    assert v.decided_by == "code"
    with pytest.raises(ValidationError):
        EquivalenceVerdict(passed=True, compared=1, identical=1,
                           db_state_matched=True, external_calls_matched=True,
                           held_back_passed=True, oracle_hash="a",
                           decided_by="model")


def test_endpoint_model_collects_citations():
    m = EndpointModel(route="POST /x",
                      entry=Citation(file="C.java", line=1),
                      path=[Citation(file="C.java", line=1),
                            Citation(file="S.java", line=9)])
    assert len(m.all_citations()) >= 2
    assert m.uncertainties == []


def test_corpus_held_back_defaults_empty_but_present():
    c = RequestCorpus(benchmark=[RequestSpec(id="r1", method="GET", path="/x")],
                      hash="h")
    assert c.held_back == []


def test_every_terminal_state_is_distinct_and_named():
    """Ten now. MEASUREMENT_NOT_ISOLATED was added because "we could not
    measure this" and "this did not help" are different facts, and a run that
    reported the second when the first was true would be lying with a
    straight face. READY_TO_SHIP was added because "all gates passed" and
    "all gates passed and it shipped" are also different facts -- a report
    rendered before a dry run's fate was decided had been defaulting to
    BOTTLENECK_NOT_FIXABLE, which is a passing run lying in the other
    direction."""
    assert len(TerminalState) == 10
    assert TerminalState.NO_BOTTLENECK_FOUND in TerminalState
    assert TerminalState.MEASUREMENT_NOT_ISOLATED is not \
           TerminalState.NO_SIGNIFICANT_IMPROVEMENT
    assert TerminalState.READY_TO_SHIP is not TerminalState.PR_OPENED
    assert TerminalState.READY_TO_SHIP is not TerminalState.BOTTLENECK_NOT_FIXABLE
    assert len({s.value for s in TerminalState}) == len(TerminalState)


def test_risk_tiers_ordered():
    assert RiskTier.LOW < RiskTier.HIGH < RiskTier.FORBIDDEN


def test_hypothesis_requires_a_layer():
    """A hypothesis that does not say WHERE the time goes cannot be checked
    for layer diversity, so the field is required rather than defaulted."""
    with pytest.raises(ValidationError):
        Hypothesis(id="h", claim="c", mechanism="m", evidence=[],
                   attributed_ms=1, prediction="p", confidence=0.5)


def test_hypothesis_rejects_an_unknown_layer():
    with pytest.raises(ValidationError):
        Hypothesis(id="h", claim="c", mechanism="m", layer="gpu",
                   evidence=[], attributed_ms=1, prediction="p", confidence=0.5)


# ───────────────────  S1: schema on the EndpointModel  ───────────────────


def test_table_info_carries_existing_indexes():
    """Session 3 must be able to tell a MISSING index from one that is already
    there, and must never propose a duplicate. It can only do that if this
    phase records what exists."""
    from perfagent.contracts import IndexInfo, TableInfo

    table = TableInfo(
        table="invoice",
        entity="Invoice",
        columns_read=["customer_id", "amount"],
        all_columns=["id bigint NOT NULL", "customer_id bigint NULL"],
        existing_indexes=[IndexInfo(name="idx_invoice_customer_id",
                                    columns=["customer_id"], unique=False,
                                    cardinality=28691)],
        approx_rows=1_196_424,
    )
    assert table.existing_indexes[0].columns == ["customer_id"]
    assert table.existing_indexes[0].unique is False


def test_index_columns_keep_their_order():
    """Order matters for a composite index: (a, b) and (b, a) are different
    indexes and only one of them answers a given query."""
    from perfagent.contracts import IndexInfo

    index = IndexInfo(name="idx", columns=["status", "due_date"], unique=False)
    assert index.columns == ["status", "due_date"]


def test_endpoint_model_defaults_tables_and_prompt_hash_empty():
    m = EndpointModel(route="POST /x", entry=Citation(file="C.java", line=1),
                      path=[Citation(file="C.java", line=1)])
    assert m.tables == [] and m.prompt_hash is None


def test_columns_read_is_what_makes_over_fetching_detectable():
    """An entity with thirty columns where the endpoint reads three is a
    projection opportunity, and nothing downstream can see it unless this
    phase records it."""
    from perfagent.contracts import TableInfo

    table = TableInfo(table="customer", columns_read=["id", "name"],
                      all_columns=[f"c{n} varchar(255)" for n in range(30)])
    assert len(table.columns_read) < len(table.all_columns)


def test_entry_path_and_sql_sources_are_derived_not_asked_for():
    """Two fields describing the same fact are two fields that can disagree,
    and the model does not get to decide which is right."""
    m = EndpointModel.model_validate({
        "route": "POST /x",
        "call_chain": [{"method": "C.search", "file": "C.java", "line": 18,
                        "does": "delegates"}],
        "data_access": [{"trigger": "repo.find", "kind": "derived_query",
                         "file": "R.java", "line": 9,
                         "executes_per_row": False}],
    })
    assert m.entry == Citation(file="C.java", line=18, symbol="C.search")
    assert [c.ref() for c in m.path] == ["C.java:18"]
    assert m.sql_sources[0].kind == "derived query"


def test_per_row_accesses_are_computed_never_counted_by_a_model():
    m = EndpointModel.model_validate({
        "route": "POST /x",
        "call_chain": [{"method": "C.s", "file": "C.java", "line": 1,
                        "does": "d"}],
        "data_access": [
            {"trigger": "a", "kind": "jpql_query", "file": "R.java", "line": 1,
             "executes_per_row": False},
            {"trigger": "b", "kind": "lazy_collection", "file": "R.java",
             "line": 2, "executes_per_row": True},
            {"trigger": "c", "kind": "lazy_association", "file": "R.java",
             "line": 3, "executes_per_row": True},
        ],
    })
    assert len(m.per_row_accesses()) == 2


def test_data_access_must_declare_whether_it_runs_per_row():
    """'Does this run once, or once per row?' is the question the whole phase
    exists to answer, so it has no default."""
    from perfagent.contracts import DataAccess

    with pytest.raises(ValidationError):
        DataAccess(trigger="t", kind="jpql_query", file="R.java", line=1)


def test_a_hand_built_model_with_the_older_fields_still_validates():
    m = EndpointModel(route="POST /x", entry=Citation(file="C.java", line=1),
                      path=[Citation(file="C.java", line=1)])
    assert m.call_chain == [] and m.entry.line == 1
