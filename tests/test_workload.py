"""P2 tests: the corpus is GENERATED for this run's endpoint, or refused.

No database: the generator takes a sampler object, and these use the static
one. That is deliberate -- it means the request-shaping logic is testable
without seeded MySQL, and the only thing the live sampler adds is where the
values come from.
"""

from __future__ import annotations

from typing import Any

import pytest

from senapathy.contracts.comprehension import EndpointModel
from senapathy.tools.sample import StaticSampler
from senapathy.workload import (CorpusRefusal, generate_corpus,
                                path_parameters, split_route,
                                table_for_parameter)


def _model(route: str, *, tables: list[str] | None = None,
           pagination: dict[str, Any] | None = None,
           query_params: list[dict[str, Any]] | None = None,
           per_row_access_on: str | None = None,
           foreign_keys: dict[str, list[dict[str, str]]] | None = None,
           ) -> EndpointModel:
    citation = {"file": "src/main/java/App.java", "line": 1, "symbol": "x"}
    foreign_keys = foreign_keys or {}
    return EndpointModel.model_validate({
        "route": route,
        "entry": citation,
        "path": [citation],
        "tables": [{"table": t, "columns_read": ["id"],
                    "all_columns": ["id", "name"],
                    "foreign_keys": foreign_keys.get(t, [])}
                   for t in (tables if tables is not None else ["customer"])],
        "data_access": (
            [{"trigger": f"{per_row_access_on}Repository.findNameById",
              "kind": "derived_query", "file": "src/main/java/App.java",
              "line": 1, "tables": [per_row_access_on],
              "executes_per_row": True}]
            if per_row_access_on else []
        ),
        "pagination": pagination if pagination is not None else {
            "mechanism": "offset", "fires_count_query": False,
            "max_page_size": None},
        "request_shape": {
            "parameters": [
                {"location": "query", "citation": citation, **param}
                for param in (query_params or [])
            ],
        },
        "uncertainties": [],
    })


def _sampler(dense: dict[tuple[str, str], list[tuple[Any, int]]] | None = None,
             **by_table: list[Any]) -> StaticSampler:
    return StaticSampler({(t, "id"): v for t, v in by_table.items()}, dense=dense)


# ──────────────────────────────  route parsing  ──────────────────────────


def test_route_and_path_parameters_are_read_from_the_template() -> None:
    assert split_route("GET /api/customers/{id}/invoices") == (
        "GET", "/api/customers/{id}/invoices")
    assert path_parameters("/api/customers/{id}/invoices") == ["id"]
    assert path_parameters("/a/{x}/b/{y}") == ["x", "y"]


def test_a_parameter_resolves_to_the_table_its_path_segment_names() -> None:
    """`{id}` on /customers/{id}/invoices is a CUSTOMER id, not an invoice
    id. Matching on the owning segment is what makes `{id}` resolvable at
    all -- almost every table has a column called `id`."""
    model = _model("GET /api/customers/{id}/invoices",
                   tables=["customer", "invoice"])
    assert table_for_parameter("id", model) == "customer"


# ──────────────────────────  what it generates  ──────────────────────────


def test_requests_are_built_for_this_endpoint_and_no_other() -> None:
    model = _model("GET /api/customers/{id}/invoices",
                   tables=["customer", "invoice"])
    corpus, _ = generate_corpus(model, sampler=_sampler(customer=[1, 2, 3]))

    everything = corpus.benchmark + corpus.held_back
    assert everything
    for spec in everything:
        assert spec.method == "GET"
        assert spec.path.startswith("/api/customers/")
        assert "/invoices" in spec.path
        assert "{id}" not in spec.path, "the template must be substituted"


def test_parameter_values_come_from_the_sampler_not_from_thin_air() -> None:
    model = _model("GET /api/customers/{id}/invoices",
                   tables=["customer", "invoice"])
    corpus, notes = generate_corpus(
        model, sampler=_sampler(customer=[7, 11, 13]))

    used = {spec.path.split("/")[3] for spec in
            corpus.benchmark + corpus.held_back}
    assert used <= {"7", "11", "13"}
    assert any("sampled from customer.id" in n for n in notes)


def test_the_cost_axes_are_varied() -> None:
    """Page depth and page size are what change the cost of a bounded
    endpoint, so the corpus has to span them."""
    model = _model("GET /api/customers/{id}/invoices",
                   tables=["customer", "invoice"])
    corpus, _ = generate_corpus(model, sampler=_sampler(customer=[1]))

    paths = [s.path for s in corpus.benchmark + corpus.held_back]
    sizes = {p.split("size=")[1] for p in paths if "size=" in p}
    depths = {p.split("page=")[1].split("&")[0] for p in paths if "page=" in p}
    assert len(sizes) > 1, f"only one page size generated: {sizes}"
    assert len(depths) > 1, f"only one page depth generated: {depths}"


def test_the_endpoints_own_page_cap_is_respected() -> None:
    """Generating size=500 against a cap of 100 measures a 4xx we did not
    intend to measure."""
    model = _model("GET /api/customers/{id}/invoices",
                   tables=["customer", "invoice"],
                   pagination={"mechanism": "offset",
                               "fires_count_query": False,
                               "max_page_size": 25})
    corpus, notes = generate_corpus(model, sampler=_sampler(customer=[1]))
    for spec in corpus.benchmark + corpus.held_back:
        size = int(spec.path.split("size=")[1])
        assert size <= 25
    assert any("capped" in n for n in notes)


def test_a_held_back_subset_exists_and_is_disjoint() -> None:
    model = _model("GET /api/customers/{id}/invoices",
                   tables=["customer", "invoice"])
    corpus, _ = generate_corpus(model, sampler=_sampler(customer=[1, 2, 3]))

    assert corpus.held_back, "the oracle needs requests measurement never saw"
    benchmark_ids = {s.id for s in corpus.benchmark}
    assert not (benchmark_ids & {s.id for s in corpus.held_back})


def test_generation_is_deterministic_so_the_oracle_stays_valid() -> None:
    """The oracle is keyed by corpus_hash. A generator that varied between
    runs would invalidate the recording every time."""
    model = _model("GET /api/customers/{id}/invoices",
                   tables=["customer", "invoice"])
    first, _ = generate_corpus(model, sampler=_sampler(customer=[1, 2, 3]))
    second, _ = generate_corpus(model, sampler=_sampler(customer=[1, 2, 3]))
    assert first.hash == second.hash


def test_an_endpoint_without_pagination_gets_no_page_parameters() -> None:
    model = _model("GET /api/customers/{id}", tables=["customer"],
                   pagination={"mechanism": "none",
                               "fires_count_query": False})
    corpus, notes = generate_corpus(model, sampler=_sampler(customer=[1, 2]))
    for spec in corpus.benchmark + corpus.held_back:
        assert "page=" not in spec.path and "size=" not in spec.path
    assert any("no page/size" in n for n in notes)


# ─────────────────────────────  what it refuses  ─────────────────────────


def test_an_unresolvable_path_parameter_is_a_loud_refusal() -> None:
    """Never a fallback to another endpoint's corpus."""
    model = _model("GET /api/widgets/{id}/parts", tables=["customer", "invoice"])
    with pytest.raises(CorpusRefusal) as exc:
        generate_corpus(model, sampler=_sampler(customer=[1]))
    assert "could not be resolved" in str(exc.value)
    assert "understand phase" in str(exc.value)


def test_an_empty_table_is_a_refusal_not_a_corpus_of_404s() -> None:
    model = _model("GET /api/customers/{id}", tables=["customer"])
    with pytest.raises(CorpusRefusal) as exc:
        generate_corpus(model, sampler=_sampler(customer=[]))
    assert "no values exist" in str(exc.value)
    assert "measures the error path" in str(exc.value)


def test_a_post_body_that_cannot_be_inferred_is_refused_not_invented() -> None:
    """request_shape records path and query parameters, not a JSON body, so
    nothing records the body's fields. Inventing one measures a 400."""
    model = _model("POST /api/customers/search", tables=["customer"])
    with pytest.raises(CorpusRefusal) as exc:
        generate_corpus(model, sampler=_sampler(customer=[1]))
    assert "cannot infer" in str(exc.value)
    assert "body" in str(exc.value)


def test_a_model_with_no_route_is_refused() -> None:
    model = _model("GET /")
    with pytest.raises(CorpusRefusal) as exc:
        generate_corpus(model, sampler=_sampler(customer=[1]))
    assert "no usable route" in str(exc.value)


# ──────────────────────────────  the sampler  ────────────────────────────


def test_the_sampler_refuses_a_table_off_the_endpoints_path() -> None:
    from senapathy.errors import GuardrailError
    from senapathy.tools.sample import ValueSampler

    sampler = ValueSampler(target=None, allowed_tables=["customer"])
    with pytest.raises(GuardrailError, match="not a table on this endpoint"):
        sampler.sample("users", "id")


def test_the_sampler_refuses_an_identifier_that_is_not_bare() -> None:
    """Table and column names are interpolated into SQL -- MySQL takes no
    placeholder in an identifier position -- so they must be bare words."""
    from senapathy.errors import ToolError
    from senapathy.tools.sample import ValueSampler

    sampler = ValueSampler(target=None, allowed_tables=["customer"])
    with pytest.raises(ToolError, match="not a bare"):
        sampler.sample("customer; DROP TABLE x", "id")
    with pytest.raises(ToolError, match="not a bare"):
        sampler.sample("customer", "id FROM x WHERE 1=1 --")


# ───────────────────  request_shape: required query params  ─────────────


def test_a_required_query_param_with_allowed_values_is_varied() -> None:
    """status is required with no default, but the code constrains it to an
    enum -- those values become a cost axis, the same as page size."""
    model = _model(
        "GET /api/invoices", tables=["invoice"],
        pagination={"mechanism": "none"},
        query_params=[{"name": "status", "type": "string", "required": True,
                       "allowed_values": ["OPEN", "OVERDUE"]}],
    )
    corpus, notes = generate_corpus(model, sampler=_sampler())

    paths = [r.path for r in corpus.benchmark + corpus.held_back]
    assert any("status=OPEN" in p for p in paths)
    assert any("status=OVERDUE" in p for p in paths)
    assert any("varied across its recorded allowed values" in n for n in notes)


def test_a_required_query_param_with_a_default_is_fixed() -> None:
    model = _model(
        "GET /api/invoices", tables=["invoice"],
        pagination={"mechanism": "none"},
        query_params=[{"name": "sort", "type": "string", "required": True,
                       "default": "id"}],
    )
    corpus, notes = generate_corpus(model, sampler=_sampler())

    paths = [r.path for r in corpus.benchmark + corpus.held_back]
    assert all("sort=id" in p for p in paths)
    assert any("fixed at its recorded default" in n for n in notes)


def test_a_required_query_param_with_no_determinable_value_is_refused_by_name() -> None:
    """The exact bug: `status` was required, nothing recorded a value for
    it, and every generated request silently omitted it. This must name the
    parameter instead."""
    model = _model(
        "GET /api/invoices", tables=["invoice"],
        pagination={"mechanism": "none"},
        query_params=[{"name": "status", "type": "string", "required": True}],
    )
    with pytest.raises(CorpusRefusal, match="status"):
        generate_corpus(model, sampler=_sampler())


def test_an_optional_query_param_is_never_required_to_resolve() -> None:
    """Required=False parameters are recorded but not supplied -- omitting
    an optional parameter is valid, not a gap."""
    model = _model(
        "GET /api/invoices", tables=["invoice"],
        pagination={"mechanism": "none"},
        query_params=[{"name": "sort", "type": "string", "required": False}],
    )
    corpus, _ = generate_corpus(model, sampler=_sampler())
    paths = [r.path for r in corpus.benchmark + corpus.held_back]
    assert all("sort=" not in p for p in paths)


def test_pagination_param_names_are_never_sourced_from_request_shape() -> None:
    """page/size come from PaginationInfo, not request_shape -- even if the
    model also (correctly) records them there, they must not be duplicated
    or treated as an unresolvable required parameter."""
    model = _model(
        "GET /api/invoices", tables=["invoice"],
        query_params=[{"name": "page", "type": "integer", "required": True},
                     {"name": "size", "type": "integer", "required": True}],
    )
    corpus, _ = generate_corpus(model, sampler=_sampler())
    for spec in corpus.benchmark + corpus.held_back:
        assert spec.path.count("page=") == 1
        assert spec.path.count("size=") == 1


# ─────────────  required query params: the database as a third source  ──


def _invoice_model(*, query_params: list[dict[str, Any]]) -> EndpointModel:
    """An endpoint over `invoice`, with a real `status` column recorded --
    unlike `_model()`'s tables, which only ever carry id/name."""
    citation = {"file": "src/main/java/App.java", "line": 1, "symbol": "x"}
    return EndpointModel.model_validate({
        "route": "GET /api/invoices",
        "entry": citation,
        "path": [citation],
        "tables": [{"table": "invoice", "columns_read": ["id", "status"],
                    "all_columns": ["id", "status", "amount"]}],
        "pagination": {"mechanism": "none", "fires_count_query": False},
        "request_shape": {
            "parameters": [
                {"location": "query", "citation": citation, **param}
                for param in query_params
            ],
        },
        "uncertainties": [],
    })


_STATUS_PARAM = {"name": "status", "type": "string", "required": True}


def test_a_required_query_param_with_no_code_values_is_sampled_from_the_database() -> None:
    from senapathy.tools.sample import StaticSampler

    model = _invoice_model(query_params=[_STATUS_PARAM])
    sampler = StaticSampler({("invoice", "status"): ["OPEN", "OVERDUE", "PAID"]})

    corpus, notes = generate_corpus(model, sampler=sampler)

    statuses = {spec.path.split("status=")[1] for spec in
               corpus.benchmark + corpus.held_back}
    assert statuses == {"OPEN", "OVERDUE", "PAID"}
    assert any("sampled from the database" in n and "not from the code" in n
              for n in notes)


def test_sampled_query_values_are_capped() -> None:
    from senapathy.tools.sample import StaticSampler
    from senapathy.workload import MAX_SAMPLED_QUERY_VALUES

    model = _invoice_model(query_params=[_STATUS_PARAM])
    many = [f"S{i}" for i in range(MAX_SAMPLED_QUERY_VALUES + 10)]
    sampler = StaticSampler({("invoice", "status"): many})

    corpus, _ = generate_corpus(model, sampler=sampler)
    statuses = {spec.path.split("status=")[1] for spec in
               corpus.benchmark + corpus.held_back}
    assert len(statuses) <= MAX_SAMPLED_QUERY_VALUES


def test_corpus_hash_is_stable_across_two_generations_via_db_sampling() -> None:
    from senapathy.tools.sample import StaticSampler

    model = _invoice_model(query_params=[_STATUS_PARAM])
    sampler = StaticSampler({("invoice", "status"): ["OPEN", "OVERDUE"]})

    first, _ = generate_corpus(model, sampler=sampler)
    second, _ = generate_corpus(model, sampler=sampler)
    assert first.hash == second.hash


def test_a_column_the_model_does_not_recognize_is_not_sampled_as_a_fallback() -> None:
    """Unlike a path parameter, a query parameter has no 'id' to fall back
    to -- sampling must not silently substitute the primary key."""
    from senapathy.tools.sample import StaticSampler

    model = _invoice_model(query_params=[
        {"name": "customerName", "type": "string", "required": True}])
    # Only 'id'/'status'/'amount' are recorded columns; 'customerName' is not.
    sampler = StaticSampler({("invoice", "id"): [1, 2, 3]})

    with pytest.raises(CorpusRefusal, match="customerName"):
        generate_corpus(model, sampler=sampler)


def test_param_override_wins_over_sampling() -> None:
    from senapathy.tools.sample import StaticSampler

    model = _invoice_model(query_params=[_STATUS_PARAM])
    sampler = StaticSampler({("invoice", "status"): ["OPEN", "OVERDUE", "PAID"]})

    corpus, notes = generate_corpus(
        model, sampler=sampler, param_overrides={"status": "OPEN"})

    statuses = {spec.path.split("status=")[1] for spec in
               corpus.benchmark + corpus.held_back}
    assert statuses == {"OPEN"}
    assert any("--param" in n for n in notes)


def test_param_override_also_wins_over_code_recorded_values() -> None:
    """--param is the highest-priority source, ahead of allowed_values and
    default too, not only ahead of sampling."""
    model = _invoice_model(query_params=[
        {"name": "status", "type": "string", "required": True,
         "allowed_values": ["OPEN", "OVERDUE"]}])

    corpus, _ = generate_corpus(
        model, sampler=None, param_overrides={"status": "CLOSED"})

    statuses = {spec.path.split("status=")[1] for spec in
               corpus.benchmark + corpus.held_back}
    assert statuses == {"CLOSED"}


def test_param_for_an_unrecorded_parameter_is_rejected() -> None:
    model = _invoice_model(query_params=[_STATUS_PARAM])

    with pytest.raises(CorpusRefusal, match="nonexistentParam"):
        generate_corpus(model, sampler=None,
                       param_overrides={"nonexistentParam": "x"})


def test_all_three_sources_failing_still_refuses_by_name() -> None:
    from senapathy.tools.sample import StaticSampler

    model = _invoice_model(query_params=[_STATUS_PARAM])
    sampler = StaticSampler({})  # nothing to sample

    with pytest.raises(CorpusRefusal) as exc:
        generate_corpus(model, sampler=sampler)

    message = str(exc.value)
    assert "status" in message
    assert "allowed_values (none recorded)" in message
    assert "default (none recorded)" in message
    assert "sampling the database" in message


# ────────────────────────  probe_corpus: substance checks  ───────────────


def _corpus(n: int = 4) -> "RequestCorpus":
    from senapathy.contracts import RequestCorpus, RequestSpec
    from senapathy.measure.corpus import corpus_hash

    specs = [RequestSpec(id=f"r{i}", method="GET", path=f"/api/invoices?i={i}")
             for i in range(n)]
    benchmark, held_back = specs[:-1], specs[-1:]
    return RequestCorpus(benchmark=benchmark, held_back=held_back,
                         hash=corpus_hash(benchmark, held_back))


def _mock_client(handler) -> "httpx.Client":
    import httpx

    return httpx.Client(transport=httpx.MockTransport(handler))


def test_probe_corpus_records_the_observed_status_on_every_request() -> None:
    import httpx

    from senapathy.workload import probe_corpus

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"ok": True})

    corpus = _corpus()
    probed = probe_corpus(corpus, "http://blue", client=_mock_client(handler))

    everything = probed.benchmark + probed.held_back
    assert everything
    assert all(spec.expected_status == 200 for spec in everything)
    # expected_status is part of what the corpus IS now, so sealing it
    # commits to a different hash than the unprobed version.
    assert probed.hash != corpus.hash


def test_probe_corpus_refuses_when_every_request_fails() -> None:
    """The exact defect this closes: nothing noticed the endpoint never ran.
    A corpus that is entirely 400s must be refused, naming the status counts
    and the first response body, not sealed as if it worked."""
    import httpx

    from senapathy.errors import ContractViolation
    from senapathy.workload import probe_corpus

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"error": "status is required"})

    with pytest.raises(ContractViolation) as exc:
        probe_corpus(_corpus(), "http://blue", client=_mock_client(handler))

    message = str(exc.value)
    assert "400" in message
    assert "status is required" in message


def test_probe_corpus_accepts_a_deliberate_error_path_request() -> None:
    """The rule is not 'no 4xx' -- an intentional error-path request is
    wanted. The rule is that SOMETHING comes back 2xx."""
    import httpx

    from senapathy.workload import probe_corpus

    def handler(request: httpx.Request) -> httpx.Response:
        if "i=0" in str(request.url):
            return httpx.Response(404, json={"error": "not found"})
        return httpx.Response(200, json={"ok": True})

    probed = probe_corpus(_corpus(), "http://blue", client=_mock_client(handler))
    statuses = {spec.expected_status for spec in probed.benchmark + probed.held_back}
    assert statuses == {200, 404}


def test_probe_corpus_names_a_transport_failure() -> None:
    import httpx

    from senapathy.errors import ContractViolation
    from senapathy.workload import probe_corpus

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    with pytest.raises(ContractViolation, match="transport layer"):
        probe_corpus(_corpus(), "http://blue", client=_mock_client(handler))


# ──────────────────  dense benchmark / diverse held-back  ─────────────────
#
# A blind stride across the id space is representative of the table, but a
# Zipf-skewed table's stride mostly lands on sparse rows -- INV-03's baseline
# averaged 8.19 per-row lookups per request against a dense-customer reality
# of 102, so the benchmark measured a tenth of the real per-row cost. These
# tests cover the fix: benchmark path-parameter values biased toward the
# dense end when the model recorded per-row work, held-back left exactly as
# diverse as before.


def _dense_model(*, max_page_size: int | None = 100) -> EndpointModel:
    return _model(
        "GET /api/customers/{id}/invoices",
        tables=["customer", "invoice"],
        pagination={"mechanism": "offset", "fires_count_query": False,
                    "max_page_size": max_page_size},
        per_row_access_on="customer",
        foreign_keys={
            "customer": [{"direction": "in", "constraint": "fk_invoice_customer",
                          "column": "invoice.customer_id", "references": "id"}],
        },
    )


def test_child_table_resolution_finds_invoice_as_customers_child() -> None:
    from senapathy.workload import _child_table_for_parameter

    model = _dense_model()
    assert _child_table_for_parameter("customer", model) == ("invoice", "customer_id")


def test_benchmark_is_biased_toward_ids_with_the_most_child_rows() -> None:
    model = _dense_model()
    sampler = _sampler(
        customer=[1, 8334, 16667],  # the blind stride a 50k-row table gives
        dense={("invoice", "customer_id"): [(1, 52765), (2, 41220), (3, 1439)]},
    )
    corpus, notes = generate_corpus(model, sampler=sampler)

    benchmark_ids = {s.path.split("/")[3] for s in corpus.benchmark}
    assert benchmark_ids == {"1", "2", "3"}, (
        "benchmark should use the densest ids, not the stride-sampled ones")
    held_back_ids = {s.path.split("/")[3] for s in corpus.held_back}
    assert held_back_ids == {"1", "8334", "16667"}, (
        "held-back must keep the diverse, stride-sampled ids unchanged")
    assert any("52,765" in n for n in notes)


def test_benchmark_notes_say_dense_and_why() -> None:
    model = _dense_model()
    sampler = _sampler(
        customer=[1],
        dense={("invoice", "customer_id"): [(1, 1439)]},
    )
    corpus, _ = generate_corpus(model, sampler=sampler)
    assert all(s.note is not None and s.note.startswith("dense: ")
               for s in corpus.benchmark)
    assert any("1,439 child rows" in (s.note or "") for s in corpus.benchmark)


def test_held_back_notes_say_sparse_and_held_for_coverage() -> None:
    model = _dense_model()
    sampler = _sampler(
        customer=[9],
        dense={("invoice", "customer_id"): [(1, 1439)]},  # 9 is not in here
    )
    corpus, _ = generate_corpus(model, sampler=sampler)
    assert all(s.note is not None and s.note.startswith("sparse: ")
               for s in corpus.held_back)
    assert any("(0 child rows), held back for coverage" in (s.note or "")
               for s in corpus.held_back)


def test_benchmark_page_size_is_biased_toward_the_maximum() -> None:
    model = _dense_model(max_page_size=100)
    sampler = _sampler(
        customer=[1],
        dense={("invoice", "customer_id"): [(1, 1439)]},
    )
    corpus, _ = generate_corpus(model, sampler=sampler)

    benchmark_sizes = {int(s.path.split("size=")[1]) for s in corpus.benchmark}
    assert benchmark_sizes == {100}, (
        "a page of 1 measures nothing about per-row cost; benchmark should "
        "only use the endpoint's own maximum")
    held_back_sizes = {int(s.path.split("size=")[1]) for s in corpus.held_back}
    assert len(held_back_sizes) > 1, "held-back must keep the diverse spread"
    assert 1 in held_back_sizes, "held-back must keep the boundary case"


def test_no_per_row_access_means_no_dense_bias() -> None:
    """When the model never recorded per-row work, benchmark and held-back
    both fall back to the same diverse sampling as before -- there is no
    per-row signal to bias toward."""
    model = _model("GET /api/customers/{id}/invoices",
                   tables=["customer", "invoice"])
    sampler = _sampler(
        customer=[1, 2, 3],
        dense={("invoice", "customer_id"): [(9, 999)]},  # should be ignored
    )
    corpus, notes = generate_corpus(model, sampler=sampler)

    benchmark_ids = {s.path.split("/")[3] for s in corpus.benchmark}
    assert benchmark_ids == {"1", "2", "3"}
    assert not any("biased toward the dense end" in n for n in notes)


def test_dense_sampling_is_deterministic_too() -> None:
    model = _dense_model()
    sampler_values = dict(
        customer=[1, 8334, 16667],
        dense={("invoice", "customer_id"): [(1, 52765), (2, 41220), (3, 1439)]},
    )
    first, _ = generate_corpus(model, sampler=_sampler(**sampler_values))
    second, _ = generate_corpus(model, sampler=_sampler(**sampler_values))
    assert first.hash == second.hash
