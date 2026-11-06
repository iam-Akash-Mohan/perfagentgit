"""P2 tests: the corpus is GENERATED for this run's endpoint, or refused.

No database: the generator takes a sampler object, and these use the static
one. That is deliberate -- it means the request-shaping logic is testable
without seeded MySQL, and the only thing the live sampler adds is where the
values come from.
"""

from __future__ import annotations

from typing import Any

import pytest

from perfagent.contracts.comprehension import EndpointModel
from perfagent.tools.sample import StaticSampler
from perfagent.workload import (CorpusRefusal, generate_corpus,
                                path_parameters, split_route,
                                table_for_parameter)


def _model(route: str, *, tables: list[str] | None = None,
           pagination: dict[str, Any] | None = None) -> EndpointModel:
    citation = {"file": "src/main/java/App.java", "line": 1, "symbol": "x"}
    return EndpointModel.model_validate({
        "route": route,
        "entry": citation,
        "path": [citation],
        "tables": [{"table": t, "columns_read": ["id"],
                    "all_columns": ["id", "name"]}
                   for t in (tables if tables is not None else ["customer"])],
        "pagination": pagination if pagination is not None else {
            "mechanism": "offset", "fires_count_query": False,
            "max_page_size": None},
        "uncertainties": [],
    })


def _sampler(**by_table: list[Any]) -> StaticSampler:
    return StaticSampler({(t, "id"): v for t, v in by_table.items()})


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
    """The comprehension model has a ResponseShape and no RequestShape, so
    nothing recorded the body's fields. Inventing one measures a 400."""
    model = _model("POST /api/customers/search", tables=["customer"])
    with pytest.raises(CorpusRefusal) as exc:
        generate_corpus(model, sampler=_sampler(customer=[1]))
    assert "cannot infer" in str(exc.value)
    assert "RequestShape" in str(exc.value)


def test_a_model_with_no_route_is_refused() -> None:
    model = _model("GET /")
    with pytest.raises(CorpusRefusal) as exc:
        generate_corpus(model, sampler=_sampler(customer=[1]))
    assert "no usable route" in str(exc.value)


# ──────────────────────────────  the sampler  ────────────────────────────


def test_the_sampler_refuses_a_table_off_the_endpoints_path() -> None:
    from perfagent.errors import GuardrailError
    from perfagent.tools.sample import ValueSampler

    sampler = ValueSampler(target=None, allowed_tables=["customer"])
    with pytest.raises(GuardrailError, match="not a table on this endpoint"):
        sampler.sample("users", "id")


def test_the_sampler_refuses_an_identifier_that_is_not_bare() -> None:
    """Table and column names are interpolated into SQL -- MySQL takes no
    placeholder in an identifier position -- so they must be bare words."""
    from perfagent.errors import ToolError
    from perfagent.tools.sample import ValueSampler

    sampler = ValueSampler(target=None, allowed_tables=["customer"])
    with pytest.raises(ToolError, match="not a bare"):
        sampler.sample("customer; DROP TABLE x", "id")
    with pytest.raises(ToolError, match="not a bare"):
        sampler.sample("customer", "id FROM x WHERE 1=1 --")
