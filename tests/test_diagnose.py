"""Unit tests for the diagnosis phase. No network, no LLM, no database.

The probe and the agent loop are the parts that touch the world; everything
that DECIDES anything — the derived facts and the validation verdict — is pure
and is tested here.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from senapathy.contracts import HypothesisSet, Layer, RequestCorpus, RequestSpec
from senapathy.errors import ContractViolation, ToolError
from senapathy.diagnose.agent import build_task
from senapathy.diagnose.evidence import (
    app_digests,
    estimated_result_rows,
    latest_run_dir,
    load_baseline,
    ms_per_result_row,
    render_evidence,
    repeated_digests,
    unexplained_share_pct,
)
from senapathy.diagnose.probe import (DEFAULT_MAX_PAGE_SIZE, MAX_ITERATIONS,
                                      ProbeCatalog, ProbeLog, make_run_probe)
from senapathy.diagnose.validate import ATTRIBUTION_TOLERANCE, validate_hypothesis_set

P50 = 1000.0


# ────────────────────────────  fixtures  ────────────────────────────────

def _digest(sql: str, calls: int, total_ms: float, **kw: Any) -> dict[str, Any]:
    return {
        "digest": sql[:16], "sample_sql": sql, "calls": calls,
        "total_ms": total_ms, "mean_ms": total_ms / calls, "rows_examined": 0,
        "rows_sent": 0, "no_index_used": False, "sample_params": [], **kw,
    }


def _baseline() -> dict[str, Any]:
    """10 requests, 20 result rows, 100ms of SQL out of 10_000ms measured."""
    return {
        "benchmark": {
            "iterations": 10, "warmup": 5, "corpus_hash": "abc",
            "status_counts": {"200": 10},
            "latency": {"n": 10, "p50_ms": P50, "p95_ms": 1500.0, "p99_ms": 1600.0,
                        "mean_ms": 1000.0, "stdev_ms": 100.0},
        },
        "sql_digests": [
            _digest("SELECT SUM ( `amount` ) FROM `invoice`", 20, 60.0, rows_examined=2000),
            _digest("SELECT `id` FROM `customer`", 10, 40.0, rows_examined=500,
                    no_index_used=True),
            _digest("SET `autocommit` = ?", 40, 5.0),
        ],
        "derived": {
            "measured_requests": 10, "measured_latency_ms": 10000.0,
            "app_statements_per_request": 3.0, "app_sql_statements": 30,
            "total_sql_ms": 105.0, "sql_share_of_latency_pct": 1.05,
            "total_rows_examined": 2500, "total_rows_sent": 100,
            "digests_without_index": 1,
        },
    }


def _endpoint_model() -> dict[str, Any]:
    return {
        "route": "/api/customers/search",
        "call_chain": [
            {"method": "search", "file": "web/CustomerController.java", "line": 17,
             "does": "handle request"},
            {"method": "enrich", "file": "service/CustomerService.java", "line": 35,
             "does": "map each row"},
        ],
        "data_access": [
            {"trigger": "repo.findByFilter", "kind": "jpql_query",
             "file": "repo/CustomerRepository.java", "line": 12, "executes_per_row": False},
            {"trigger": "lazy load", "kind": "lazy_collection",
             "file": "domain/Customer.java", "line": 18, "executes_per_row": True},
        ],
        "uncertainties": ["count query not visible in source"],
    }


def _paginated_endpoint_model() -> dict[str, Any]:
    """A GET endpoint with offset pagination -- run_probe's size axis applies."""
    model = _endpoint_model()
    model["route"] = "GET /api/customers/search"
    model["pagination"] = {"mechanism": "offset", "max_page_size": 100}
    model["response_shape"] = {"dto": "CustomerDto", "fields_returned": ["content"],
                              "unbounded_collections": []}
    return model


def _paginated_corpus() -> RequestCorpus:
    return RequestCorpus(benchmark=[
        RequestSpec(id="dense-req", method="GET",
                   path="/api/customers/search?name=a&page=0&size=50",
                   note="dense: name=a"),
    ], hash="x")


def _unpaginated_endpoint_model() -> dict[str, Any]:
    """A GET endpoint with no size parameter -- UNB-01's shape: the corpus's
    own request_id choices are the only magnitude axis."""
    model = _endpoint_model()
    model["route"] = "GET /api/invoices/{id}/payment-history"
    model["pagination"] = None
    model["response_shape"] = {"dto": "InvoicePaymentHistoryDto",
                              "fields_returned": ["invoiceId", "allocations"],
                              "unbounded_collections": ["allocations"]}
    return model


def _unpaginated_corpus() -> RequestCorpus:
    return RequestCorpus(benchmark=[
        RequestSpec(id="dense-id1", method="GET",
                   path="/api/invoices/1/payment-history",
                   note="dense: id=1 (16 child rows)"),
        RequestSpec(id="sparse-id9", method="GET",
                   path="/api/invoices/9/payment-history",
                   note="sparse: id=9 (0 child rows)"),
    ], hash="y")


def _h(i: int, ms: float, layer: str = "database_query") -> dict[str, Any]:
    return {"id": f"h{i}", "claim": "c", "mechanism": "m", "layer": layer,
            "evidence": ["p50 is 1000.0 ms"], "attributed_ms": ms,
            "prediction": "p", "confidence": 0.5}


def _probe(ids: list[str] | None = None) -> dict[str, Any]:
    return {"tool": "run_probe", "args": {"page_size": 5},
            "discriminates_between": ids if ids is not None else ["h1", "h2"],
            "result": "p50 rose linearly"}


def _payload(**over: Any) -> dict[str, Any]:
    base = {
        "hypotheses": [_h(1, 500), _h(2, 300, "serialization"), _h(3, 100, "orm_hydration")],
        "probes_run": [_probe()],
        "ranked_ids": ["h1", "h2", "h3"],
    }
    base.update(over)
    return base


# ────────────────────  derived facts: computed, not inferred  ────────────

def test_session_control_sql_is_excluded_from_app_digests() -> None:
    digests = app_digests(_baseline())
    assert len(digests) == 2
    assert all("autocommit" not in d.sample_sql for d in digests)


def test_repeated_digests_finds_the_per_row_query_only() -> None:
    repeated = repeated_digests(_baseline())
    assert len(repeated) == 1
    digest, ratio = repeated[0]
    assert "invoice" in digest.sample_sql
    assert ratio == 2.0            # 20 calls / 10 requests


def test_result_rows_derive_from_the_per_row_digest_call_count() -> None:
    """A query firing once per row has calls == rows. That is the derivation."""
    assert estimated_result_rows(_baseline()) == 20
    assert ms_per_result_row(_baseline()) == pytest.approx(500.0)  # 10000/20


def test_no_per_row_digest_means_no_per_row_cost_is_invented() -> None:
    baseline = _baseline()
    baseline["sql_digests"] = [_digest("SELECT `id` FROM `customer`", 10, 40.0)]
    assert estimated_result_rows(baseline) is None
    assert ms_per_result_row(baseline) is None


def test_unexplained_share_is_the_complement_of_sql_share() -> None:
    assert unexplained_share_pct(_baseline()) == pytest.approx(98.95)


# ───────────────────  the rendered evidence is the whole view  ───────────

def test_rendered_evidence_stays_within_budget_and_hides_raw_samples() -> None:
    baseline = _baseline()
    # Raw samples exist in the artifact but must never reach the model.
    baseline["benchmark"]["samples"] = [
        {"i": i, "request_id": "r", "status": 200, "latency_ms": 1000.0 + i, "error": None}
        for i in range(100)
    ]
    text = render_evidence(baseline, _endpoint_model())

    assert len(text) // 4 < 1500, "evidence must fit the ~1500 token budget"
    assert "latency_ms" not in text
    assert "request_id" not in text
    assert "1000.5" not in text


def test_rendered_evidence_carries_every_required_section() -> None:
    text = render_evidence(_baseline(), _endpoint_model())
    for expected in [
        "LATENCY", "p50", "p95", "p99", "mean", "stdev",
        "APPLICATION SQL DIGESTS", "calls/req", "rows exam", "no idx",
        "DERIVED FACTS", "SQL share of measured latency",
        "measured latency per result row", "CANDIDATE N+1", "CALL CHAIN",
    ]:
        assert expected in text, f"missing: {expected}"
    # one line per hop
    assert "1. search" in text and "2. enrich" in text
    # session-control SQL must not be shown as application SQL
    assert "autocommit" not in text


def test_task_states_the_open_question_with_computed_numbers() -> None:
    catalog = ProbeCatalog("http://app", _paginated_endpoint_model(), _paginated_corpus())
    task = build_task(_baseline(), _paginated_endpoint_model(), catalog=catalog)
    assert "98.95" in task          # unexplained share, computed
    assert "500 ms of measured latency per result row" in task
    assert "dense-req" in task      # the corpus's own request id, not invented
    assert "SWEEPING page_size" in task   # this endpoint paginates
    assert "MEASURED FACTS" in task
    assert "must not exceed 1200 ms" in task   # p50 * 1.20


def test_task_tells_a_non_paginated_endpoint_to_vary_request_id_instead() -> None:
    catalog = ProbeCatalog("http://app", _unpaginated_endpoint_model(), _unpaginated_corpus())
    task = build_task(_baseline(), _unpaginated_endpoint_model(), catalog=catalog)
    assert "does not paginate" in task
    assert "dense-id1" in task and "sparse-id9" in task
    assert "SWEEPING page_size" not in task


# ──────────────────────────  validation verdicts  ────────────────────────

def test_a_well_formed_diagnosis_is_accepted_and_p50_is_injected() -> None:
    result = validate_hypothesis_set(_payload(), P50)
    assert isinstance(result, HypothesisSet)
    assert result.measured_total_ms == P50      # from the artifact, not the model
    assert {h.layer for h in result.hypotheses} == {
        Layer.DATABASE_QUERY, Layer.SERIALIZATION, Layer.ORM_HYDRATION
    }


def test_fewer_than_three_hypotheses_is_rejected() -> None:
    with pytest.raises(ToolError, match="at least 3|too_short|min_length"):
        validate_hypothesis_set(_payload(hypotheses=[_h(1, 10), _h(2, 10)],
                                         ranked_ids=["h1", "h2"]), P50)


def test_attribution_beyond_the_tolerance_is_rejected() -> None:
    over = [_h(1, 900), _h(2, 200, "serialization"), _h(3, 200, "orm_hydration")]
    with pytest.raises(ToolError, match="attributed_ms sums to 1300.0"):
        validate_hypothesis_set(_payload(hypotheses=over), P50)


def test_attribution_exactly_at_the_tolerance_is_accepted() -> None:
    """1000ms p50 with 20% tolerance means 1200ms is the ceiling, inclusive."""
    at_limit = [_h(1, 1000), _h(2, 100, "serialization"), _h(3, 100, "orm_hydration")]
    assert sum(h["attributed_ms"] for h in at_limit) == P50 * (1 + ATTRIBUTION_TOLERANCE)
    result = validate_hypothesis_set(_payload(hypotheses=at_limit), P50)
    assert result.measured_total_ms == P50


def test_attribution_far_below_measured_p50_is_rejected() -> None:
    """COL-04: three hypotheses totalling 11 ms on a measured p50 of ~267 ms
    (4%) passed the OLD check cleanly, because that check only ever caught
    OVER-attribution. A diagnosis that explains a sliver of what was
    measured has not found the cause."""
    thin = [_h(1, 5.0, "orm_hydration"), _h(2, 4.0, "serialization"),
           _h(3, 2.0, "application_logic")]
    with pytest.raises(ToolError, match="only 1.1% of the measured p50"):
        validate_hypothesis_set(_payload(hypotheses=thin), P50)


def test_attribution_exactly_at_the_floor_is_accepted() -> None:
    at_floor = [_h(1, 300, "orm_hydration"), _h(2, 100, "serialization"),
               _h(3, 100, "application_logic")]
    assert sum(h["attributed_ms"] for h in at_floor) == P50 * 0.5
    result = validate_hypothesis_set(_payload(hypotheses=at_floor), P50)
    assert result.measured_total_ms == P50


def test_a_high_sql_share_with_no_database_hypothesis_is_rejected() -> None:
    """COL-04: sql_share_of_latency_pct = 95.16, busiest digest doing a full
    scan of the endpoint's own query -- and none of the three hypotheses
    named the database_query layer. That is a contradiction between the
    evidence and the diagnosis, not a competing theory."""
    no_database = [_h(1, 300, "orm_hydration"), _h(2, 200, "serialization"),
                  _h(3, 100, "application_logic")]
    with pytest.raises(ToolError, match="none of your hypotheses names the "
                       "database_query layer"):
        validate_hypothesis_set(_payload(hypotheses=no_database), P50,
                               sql_share_of_latency_pct=95.16)


def test_a_high_sql_share_with_a_database_hypothesis_is_accepted() -> None:
    result = validate_hypothesis_set(_payload(), P50,
                                     sql_share_of_latency_pct=95.16)
    assert result.measured_total_ms == P50


def test_a_low_sql_share_with_a_dominant_database_hypothesis_is_rejected() -> None:
    """The inverse contradiction: claiming the database dominates when the
    measurement shows it barely runs at all."""
    database_dominant = [_h(1, 600, "database_query"),
                         _h(2, 200, "serialization"), _h(3, 100, "network")]
    with pytest.raises(ToolError, match="the measurement shows it does not"):
        validate_hypothesis_set(_payload(hypotheses=database_dominant), P50,
                               sql_share_of_latency_pct=1.05)


def test_sql_share_checks_are_skipped_when_not_supplied() -> None:
    """A caller without a baseline handy -- a hand-built test, an older
    artifact -- still gets every other check; this one just isn't reachable
    without the evidence it reconciles against."""
    no_database = [_h(1, 300, "orm_hydration"), _h(2, 200, "serialization"),
                  _h(3, 100, "application_logic")]
    result = validate_hypothesis_set(_payload(hypotheses=no_database), P50)
    assert result.measured_total_ms == P50


def test_ranked_ids_must_reference_real_hypotheses() -> None:
    with pytest.raises(ToolError, match="unknown hypothesis id\\(s\\): h9"):
        validate_hypothesis_set(_payload(ranked_ids=["h1", "h2", "h9"]), P50)


def test_ranked_ids_must_cover_every_hypothesis() -> None:
    with pytest.raises(ToolError, match="omits hypothesis id\\(s\\): h3"):
        validate_hypothesis_set(_payload(ranked_ids=["h1", "h2"]), P50)


def test_a_diagnosis_without_probes_is_rejected() -> None:
    with pytest.raises(ToolError, match="probes_run is empty"):
        validate_hypothesis_set(_payload(probes_run=[]), P50)


def test_a_probe_that_separates_nothing_is_not_an_experiment() -> None:
    with pytest.raises(ToolError, match="declares discriminates_between"):
        validate_hypothesis_set(_payload(probes_run=[_probe([])]), P50)


def test_probe_cannot_reference_an_unknown_hypothesis() -> None:
    with pytest.raises(ToolError, match="discriminates_between references unknown"):
        validate_hypothesis_set(_payload(probes_run=[_probe(["h7"])]), P50)


def test_hypotheses_must_span_at_least_two_layers() -> None:
    # 200 each, not 100: the sum has to clear the attribution floor (2b) so
    # this test reaches the layer-span check (5) it actually means to
    # exercise, rather than being intercepted by a different, earlier one.
    same = [_h(1, 200), _h(2, 200), _h(3, 200)]
    with pytest.raises(ToolError, match="same layer \\(database_query\\)"):
        validate_hypothesis_set(_payload(hypotheses=same), P50)


def test_reported_probes_must_have_actually_run() -> None:
    """The agent's probe list is a claim; the ProbeLog is the evidence."""
    empty_log = ProbeLog()
    with pytest.raises(ToolError, match="no probe was actually executed"):
        validate_hypothesis_set(_payload(), P50, empty_log)

    ran = ProbeLog()
    ran.record({"page_size": 5}, "p50 500ms")
    assert validate_hypothesis_set(_payload(), P50, ran).measured_total_ms == P50


def test_an_unknown_layer_is_rejected_by_the_schema() -> None:
    with pytest.raises(ToolError, match="layer"):
        validate_hypothesis_set(_payload(hypotheses=[
            _h(1, 100, "gpu"), _h(2, 100, "serialization"), _h(3, 100)]), P50)


# ─────────────────────────────  the probe  ──────────────────────────────

class _FakeTarget:
    """Stands in for MySQLTarget so no database is touched."""
    database = "mini"


def _probe_runner(monkeypatch: pytest.MonkeyPatch, handler, *,
                  endpoint_model=None, corpus=None, digests=None) -> tuple:
    import senapathy.diagnose.probe as probe_mod

    monkeypatch.setattr(probe_mod, "reset_digests", lambda target: None)
    monkeypatch.setattr(probe_mod, "collect_digests", lambda target: digests or [])

    real_client = httpx.Client
    transport = httpx.MockTransport(handler)

    def fake_client(*args: Any, **kw: Any) -> httpx.Client:
        kw.pop("timeout", None)
        return real_client(transport=transport, timeout=5.0)

    monkeypatch.setattr(probe_mod.httpx, "Client", fake_client)
    log = ProbeLog()
    return make_run_probe(
        "http://app", _FakeTarget(),
        endpoint_model=endpoint_model or _paginated_endpoint_model(),
        corpus=corpus or _paginated_corpus(),
        log=log,
    ), log


def test_probe_cannot_be_built_for_an_endpoint_with_no_corpus_requests() -> None:
    """Refuse by name, at construction, rather than falling back to a
    different endpoint -- this is the check that replaces the old hardcoded
    '/api/customers/search' default."""
    empty = RequestCorpus(benchmark=[], hash="x")
    with pytest.raises(ContractViolation, match="no benchmark requests"):
        ProbeCatalog("http://app", _paginated_endpoint_model(), empty)


def test_probe_cannot_be_built_for_a_non_get_corpus_request() -> None:
    non_get = RequestCorpus(benchmark=[
        RequestSpec(id="c1", method="POST", path="/api/customers/search"),
    ], hash="x")
    with pytest.raises(ContractViolation, match="POST /api/customers/search"):
        ProbeCatalog("http://app", _paginated_endpoint_model(), non_get)


def test_probe_clamps_page_size_and_iterations_and_says_so(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[str] = []

    def handle(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        return httpx.Response(200, json={"content": [{"id": 1}]})

    run_probe, _ = _probe_runner(monkeypatch, handle)
    out = run_probe(request_id="dense-req", page_size=9999, iterations=500)

    assert f"iterations clamped from 500 to {MAX_ITERATIONS}" in out
    assert all(f"size={DEFAULT_MAX_PAGE_SIZE}" in url for url in seen)
    # 1 warmup + MAX_ITERATIONS measured
    assert len(seen) == MAX_ITERATIONS + 1


def test_probe_can_only_fire_this_run_s_own_corpus_requests(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, str]] = []

    def handle(request: httpx.Request) -> httpx.Response:
        calls.append((request.method, request.url.path))
        return httpx.Response(200, json={"content": []})

    run_probe, _ = _probe_runner(monkeypatch, handle)
    run_probe(request_id="dense-req", iterations=2)

    assert {c[0] for c in calls} == {"GET"}
    assert {c[1] for c in calls} == {"/api/customers/search"}


def test_an_unknown_request_id_is_refused_by_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_probe, _ = _probe_runner(monkeypatch, lambda r: httpx.Response(200, json={}))
    with pytest.raises(ToolError, match="not one of this run's own corpus"):
        run_probe(request_id="invented-id", iterations=1)


def test_page_size_on_a_non_paginated_endpoint_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_probe, _ = _probe_runner(
        monkeypatch, lambda r: httpx.Response(200, json={}),
        endpoint_model=_unpaginated_endpoint_model(), corpus=_unpaginated_corpus(),
    )
    with pytest.raises(ToolError, match="has no size parameter to vary"):
        run_probe(request_id="dense-id1", page_size=10)


def test_probe_reports_the_signals_the_agent_needs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from senapathy.contracts import SqlDigestStat

    body = {"content": [{"id": i} for i in range(4)]}

    def handle(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=body)

    digests = [
        SqlDigestStat.model_validate(_digest("SELECT SUM(`amount`) FROM `invoice`", 8, 40.0)),
        SqlDigestStat.model_validate(_digest("SET `autocommit` = ?", 100, 9.0)),
    ]
    run_probe, log = _probe_runner(monkeypatch, handle, digests=digests)
    out = run_probe(request_id="dense-req", page_size=4, iterations=2)

    assert "result rows in response      : 4" in out
    # session control excluded: 8 app calls / 2 iterations
    assert "application SQL per request  : 4.0 statements" in out
    assert "server-side SQL per request  : 20.0 ms" in out
    assert "response body size (median)" in out
    assert "latency per result row" in out
    assert len(log) == 1
    assert log.entries[0]["args"] == {"request_id": "dense-req", "iterations": 2,
                                      "page_size": 4}
    assert log.entries[0]["endpoint_called"] == "GET http://app/api/customers/search?name=a&page=0&size=4"


def test_probe_counts_rows_via_the_endpoints_own_unbounded_field(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """UNB-01's shape: the response is not a Page<> — the row count lives in
    whatever field response_shape.unbounded_collections names."""
    body = {"invoiceId": 1, "allocations": [{"id": 1}, {"id": 2}, {"id": 3}]}

    def handle(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=body)

    run_probe, _ = _probe_runner(
        monkeypatch, handle,
        endpoint_model=_unpaginated_endpoint_model(), corpus=_unpaginated_corpus(),
    )
    out = run_probe(request_id="dense-id1", iterations=2)
    assert "result rows in response      : 3" in out


def test_probe_rejects_non_integer_arguments(monkeypatch: pytest.MonkeyPatch) -> None:
    run_probe, _ = _probe_runner(monkeypatch, lambda r: httpx.Response(200, json={}))
    with pytest.raises(ToolError, match="page_size must be an integer"):
        run_probe(request_id="dense-req", page_size="big")


# ──────────────────────────────  loading  ───────────────────────────────

def test_missing_baseline_is_an_actionable_contract_violation(tmp_path: Path) -> None:
    with pytest.raises(ContractViolation, match="Run `senapathy baseline` first"):
        load_baseline(tmp_path / "baseline.json")


def test_a_non_baseline_document_is_rejected(tmp_path: Path) -> None:
    p = tmp_path / "baseline.json"
    p.write_text(json.dumps({"hello": "world"}))
    with pytest.raises(ContractViolation, match="not a baseline document"):
        load_baseline(p)


def test_latest_run_dir_picks_the_newest_and_skips_incomplete(tmp_path: Path) -> None:
    for name in ("20260101T000000Z", "20260301T000000Z"):
        d = tmp_path / name
        d.mkdir()
        (d / "baseline.json").write_text("{}")
    (tmp_path / "20260401T000000Z").mkdir()      # no baseline.json — must be skipped
    assert latest_run_dir(tmp_path).name == "20260301T000000Z"


def test_no_runs_at_all_is_actionable(tmp_path: Path) -> None:
    with pytest.raises(ContractViolation, match="no run with a baseline.json"):
        latest_run_dir(tmp_path)


# ─────────────────  durability of an accepted diagnosis  ────────────────

def _recording_probe_factory(base_url, target, *, endpoint_model=None,
                             corpus=None, log=None, catalog=None):
    """Stand-in for make_run_probe that still writes to the ProbeLog, because
    validation checks the log to prove a reported probe really ran."""
    def run_probe(**kwargs: Any) -> str:
        if log is not None:
            log.record(kwargs, "p50 500ms")
        return "p50 500ms"
    return run_probe


def _write_corpus(run_dir: Path) -> None:
    from senapathy.measure.corpus import save_corpus

    save_corpus(_paginated_corpus(), run_dir / "corpus.json")



def test_an_accepted_diagnosis_survives_a_later_loop_crash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """P01 raises KeyError('choices') on any non-200 from the provider, so a
    rate limit ONE STEP AFTER a successful emit must not discard a diagnosis
    that already passed every check."""
    import senapathy.diagnose.agent as agent_mod

    run = tmp_path / "20260101T000000Z"
    run.mkdir()
    (run / "baseline.json").write_text(json.dumps(_baseline()))
    _write_corpus(run)
    model_path = tmp_path / "endpoint_model.json"
    model_path.write_text(json.dumps(_paginated_endpoint_model()))

    def fake_run_agent(*, handlers, **kw):
        # **kw, not a fixed list: a double that pins every keyword of the
        # function it stands in for fails when the real signature grows one,
        # which says nothing about the behaviour under test.
        handlers["run_probe"](page_size=5, name_filter="a", iterations=2)
        # Not _payload()'s own default: that defaults h1 to database_query at
        # 500 ms, which is exactly the inverse-sql-share contradiction (2c)
        # against this fixture's sql_share_of_latency_pct of 1.05 -- a
        # collision with THIS test's own fixtures, unrelated to what this
        # test is actually about (survival of an accepted diagnosis past a
        # later loop crash).
        handlers["emit_hypotheses"](**_payload(hypotheses=[
            _h(1, 100, "orm_hydration"), _h(2, 300, "serialization"),
            _h(3, 100, "application_logic"),
        ]))
        raise KeyError("choices")          # exactly what a 429 produces

    monkeypatch.setattr(agent_mod, "run_agent", fake_run_agent)
    monkeypatch.setattr(agent_mod, "make_run_probe", _recording_probe_factory)
    monkeypatch.setattr(agent_mod, "_require_api_key", lambda: "test-key")
    monkeypatch.setattr(agent_mod, "make_client", lambda *a, **k: object())

    hset, _log, path = agent_mod.run_diagnosis(
        base_url="http://app", run_dir=run, endpoint_model_path=model_path,
        mysql=_FakeTarget(),
    )

    assert path.exists(), "an accepted diagnosis must be persisted"
    saved = json.loads(path.read_text())
    assert len(saved["hypotheses"]) == 3
    assert "KeyError" in saved["loop_error_after_acceptance"]
    assert hset.measured_total_ms == P50

    # Resume's skip check compares this against the run's CURRENT
    # baseline.json -- a diagnosis attributed against a baseline that has
    # since been re-measured is not evidence for the current one.
    from senapathy.measure.baseline import baseline_hash

    assert saved["baseline_hash"] == baseline_hash(_baseline())


def test_a_crash_before_any_acceptance_is_reported_not_swallowed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import senapathy.diagnose.agent as agent_mod

    run = tmp_path / "20260101T000000Z"
    run.mkdir()
    (run / "baseline.json").write_text(json.dumps(_baseline()))
    _write_corpus(run)
    model_path = tmp_path / "endpoint_model.json"
    model_path.write_text(json.dumps(_paginated_endpoint_model()))

    def fake_run_agent(**kw):
        raise KeyError("choices")

    monkeypatch.setattr(agent_mod, "run_agent", fake_run_agent)
    monkeypatch.setattr(agent_mod, "make_run_probe", _recording_probe_factory)
    monkeypatch.setattr(agent_mod, "_require_api_key", lambda: "test-key")
    monkeypatch.setattr(agent_mod, "make_client", lambda *a, **k: object())

    with pytest.raises(ContractViolation, match="KeyError"):
        agent_mod.run_diagnosis(
            base_url="http://app", run_dir=run, endpoint_model_path=model_path,
            mysql=_FakeTarget(),
        )
    assert not (run / "hypotheses.json").exists()
