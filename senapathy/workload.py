"""Phase P2: generate THIS run's request corpus, from evidence in hand.

What this replaces: a 12-request JSON file shipped inside the Python package,
every request a `POST /api/customers/search` against one repo's seed data,
used as the DEFAULT for every run. A run against any other endpoint
silently baselined, measured and gated that one. The corpus is per-run
evidence; it was code.

Where the requests come from:
  * the route, path parameters, required query parameters and pagination
    come from the comprehension model, which read them out of the
    controller — `request_shape` is the symmetric counterpart of
    `response_shape`: no required parameter it records may go unsupplied;
  * path parameter VALUES come from sampling the database — real ids that
    exist, chosen to span cheap and expensive shapes — never invented;
    required query parameter values come from `request_shape`'s own
    `allowed_values` or `default` first, DISTINCT sampled column values
    second, and an operator's `--param NAME=VALUE` ahead of all three —
    never invented, and a required parameter none of those can supply is a
    refusal, not an omission;
  * the axes that change cost are varied deliberately: page depth, page
    size, a selectivity spread across the sampled ids, and any required
    query parameter with recorded allowed_values (e.g. a status filter).

One corpus, two jobs with opposite needs. The ORACLE needs COVERAGE: sparse
rows, empty pages, boundary cases — behaviour must be preserved everywhere,
including where nothing is returned. The BENCHMARK needs COST: the shapes
where the endpoint's work actually lives — a benchmark dominated by empty
responses measures nothing. So path parameters are sampled twice: a diverse,
stride-sampled set for HELD-BACK (unchanged), and, when the model recorded a
per-row data access with a resolvable child table, a set biased toward the
rows with the most matching child rows for BENCHMARK — covering the dense
end of the distribution rather than its middle. Benchmark page sizes are
likewise biased toward the endpoint's own maximum; a page of 1 tells you
nothing about per-row cost. Every generated request's `note` records which
set chose it and why.

Determinism: sampling is ordered and seeded, so the same model against the
same database produces a byte-identical corpus and therefore the same
`corpus_hash`. That matters because the oracle is keyed by that hash: a
generator that shuffled would invalidate the recording on every run.
"""

from __future__ import annotations

import re
from typing import Any

import httpx

from .contracts import RequestCorpus, RequestSpec
from .contracts.comprehension import EndpointModel
from .errors import ContractViolation
from .measure.corpus import corpus_hash

# `{id}` or `:id` in a route template.
_PATH_PARAM = re.compile(r"\{(\w+)\}|:(\w+)\b")

# Page sizes worth measuring: the cheapest shape that still walks the
# mapping loop, a typical page, and the largest the endpoint admits. Cost
# per request scales with this, so it is the primary axis. This spread is
# for the HELD-BACK set, which exists to catch behaviour changes and so
# needs the boundary cases (a page of 1, an empty page) that a benchmark
# built only from the heaviest shape would never exercise.
_PAGE_SIZES = (1, 10, 50)
_PAGE_DEPTHS = (0, 2)

# How many ids/rows to pull when ranking path-parameter values by child-row
# count. Small on purpose: a corpus needs a handful of dense shapes to
# measure, not a census of the distribution's tail.
MAX_DENSE_VALUES = 6

# Generated separately from request_shape, via PaginationInfo's own
# page/size axis below -- a query parameter by either of these names is
# never ALSO sourced from request_shape, which would either duplicate or
# conflict with it.
PAGINATION_PARAM_NAMES = frozenset({"page", "size"})

# How many distinct values to pull for a required query parameter the code
# does not visibly constrain. Capped like path-parameter sampling (limit=6
# there) so a high-cardinality column does not explode the corpus.
MAX_SAMPLED_QUERY_VALUES = 5


class CorpusRefusal(ContractViolation):
    """The corpus could not be generated, and exactly what was missing.

    Never a fallback to another endpoint's corpus: that is the failure this
    phase exists to make impossible.
    """

    def __init__(self, what: str, detail: str, advice: str) -> None:
        super().__init__(
            f"cannot generate a corpus for this endpoint: {what}\n"
            f"  detail: {detail}\n"
            f"  what to do: {advice}"
        )


def split_route(route: str) -> tuple[str, str]:
    """('GET', '/api/customers/{id}/invoices') from any reasonable spelling."""
    parts = str(route or "").strip().split(None, 1)
    if len(parts) == 2:
        method, path = parts
    else:
        method, path = "GET", parts[0] if parts else ""
    return method.strip().upper(), "/" + path.strip().strip("/")


def path_parameters(path: str) -> list[str]:
    """The names of the `{...}` placeholders in a route template."""
    return [a or b for a, b in _PATH_PARAM.findall(path)]


def table_for_parameter(name: str, model: EndpointModel) -> str | None:
    """Which table a path parameter's values should be sampled from.

    `{id}` on `/api/customers/{id}/invoices` means a CUSTOMER id, not an
    invoice id — the owning noun is the path segment before the parameter.
    Matching on that segment rather than on the parameter name is what makes
    `{id}` resolvable at all, since almost every table has one.
    """
    tables = {t.table.lower(): t.table for t in model.tables}
    if not tables:
        return None

    _, path = split_route(model.route)
    segments = [s for s in path.split("/") if s]
    for index, segment in enumerate(segments):
        if not (segment.startswith("{") or segment.startswith(":")):
            continue
        if segment.strip("{}:") != name:
            continue
        if index == 0:
            break
        owner = segments[index - 1].rstrip("s").lower()
        for key, original in tables.items():
            if key == owner or key.rstrip("s") == owner:
                return original
        break

    # Fall back to the single table the endpoint reads, when there is only
    # one — unambiguous, so not a guess.
    if len(tables) == 1:
        return next(iter(tables.values()))
    return None


def _column_for_parameter(name: str, table: str, model: EndpointModel) -> str:
    """The column to sample. A path parameter addresses a row, so: the
    primary key, which for these entities is `id`."""
    for info in model.tables:
        if info.table != table:
            continue
        for candidate in (name, "id"):
            if candidate in (info.all_columns or info.columns_read):
                return candidate
    return "id"


def _query_filter_column(name: str, table: str, model: EndpointModel) -> str | None:
    """The column a QUERY parameter filters, if the model recognizes it as a
    column on this table -- or None if it does not.

    Deliberately NOT `_column_for_parameter`: a path parameter addresses a
    row and falling back to the primary key is correct for it, but a query
    parameter filters by its OWN name's column, and there is no row-identity
    fallback that means anything for it. Falling back to `id` here would
    sample primary keys and submit them as if they were, say, status values.
    """
    for info in model.tables:
        if info.table != table:
            continue
        if name in (info.all_columns or info.columns_read):
            return name
    return None


def _child_table_for_parameter(
    owner_table: str, model: EndpointModel
) -> tuple[str, str] | None:
    """The (table, column) whose row count predicts how much per-row work a
    path parameter value costs -- or None if the model never recorded one.

    `{id}` on `/customers/{id}/invoices` addresses `customer.id`, but
    `customer` itself can never be dense: there is exactly one row per id.
    What scales is `invoice.customer_id` -- the number of invoices a
    customer has is what decides how many times the per-row customer lookup
    actually fires. That relationship is read off `TableInfo.foreign_keys`,
    which `get_schema` already recorded while tracing this endpoint (an
    "in" edge on the owner table's own entry, pointing back at whichever
    child table references it) -- not guessed from path segments, since the
    schema the model already read is more reliable than a naming
    convention.

    Only a table this endpoint's model actually reads is returned: sampling
    one it never touched would be a query against something the allowlist
    was never built to cover.
    """
    known_tables = {t.table for t in model.tables}
    for info in model.tables:
        if info.table != owner_table:
            continue
        for key in info.foreign_keys:
            if key.get("direction") != "in":
                continue
            column_ref = key.get("column", "")
            if "." not in column_ref:
                continue
            child_table, child_column = column_ref.split(".", 1)
            if child_table in known_tables:
                return child_table, child_column
    return None


def _sampled_query_values(
    name: str, model: EndpointModel, sampler: Any
) -> tuple[list[Any] | None, str]:
    """DISTINCT values for a required query parameter the code does not
    visibly constrain, read from the column it filters -- the same
    table_for_parameter() / sampler path path parameters already use, since
    "which table does this parameter's value come from" is the same
    question whether the parameter came from the path or the query string.

    Returns (values_or_None, why) -- `why` explains what was tried even on
    success, since generate_corpus records it in the corpus notes either way.
    """
    table = table_for_parameter(name, model)
    if table is None:
        return None, "no table could be resolved for it"
    column = _query_filter_column(name, table, model)
    if column is None:
        return None, f"'{name}' is not a recorded column on {table}"
    try:
        sampled = list(sampler.sample(table, column, limit=MAX_SAMPLED_QUERY_VALUES))
    except Exception as exc:  # noqa: BLE001 — folded into the refusal, not raised raw
        return None, f"sampling {table}.{column} failed: {type(exc).__name__}: {exc}"
    if not sampled:
        return None, f"{table}.{column} returned no rows"
    # DISTINCT and in the sampler's own stable order (it orders by the
    # column itself -- see ValueSampler) so the corpus hash stays stable
    # across generations against the same data.
    distinct = list(dict.fromkeys(sampled))[:MAX_SAMPLED_QUERY_VALUES]
    return distinct, f"{len(distinct)} distinct value(s) from {table}.{column}"


def _required_query_values(
    model: EndpointModel, *, sampler: Any,
    overrides: dict[str, str] | None = None,
) -> tuple[dict[str, list[Any]], list[str]]:
    """Every query parameter generate_corpus will supply a value for, and
    notes about how each was resolved.

    Pagination's own page/size are excluded here -- they are generated from
    PaginationInfo below, by name, same as before this existed.

    For a REQUIRED parameter, three sources are tried in order, and a
    parameter this cannot determine a value for is never silently omitted
    -- see the CorpusRefusal below, which names all three attempts:
      1. `allowed_values` the code visibly constrains it to (an enum, a
         @Pattern, an explicit check) -- varied as a COST AXIS, like page
         size, since different values are different selectivity.
      2. a bare `default` -- fixed, since there is nothing to vary it
         across without inventing data.
      3. DISTINCT values sampled from the database column it filters --
         still real data, never invented, just not visible in the code.

    `overrides` is the operator's `--param NAME=VALUE` escape hatch. It is
    checked FIRST, ahead of all three sources, for any parameter this run's
    request_shape actually records (required or not) -- but naming a
    parameter request_shape does NOT record is refused rather than silently
    added; see the validation below.
    """
    overrides = overrides or {}
    shape = model.request_shape
    query_param_names = {
        p.name for p in (shape.parameters if shape else [])
        if p.location == "query" and p.name not in PAGINATION_PARAM_NAMES
    }
    unknown = [name for name in overrides if name not in query_param_names]
    if unknown:
        raise CorpusRefusal(
            what=f"--param named {', '.join(sorted(unknown))}, which "
                 f"request_shape does not record as a query parameter",
            detail=(f"this endpoint's recorded query parameters are "
                    f"{sorted(query_param_names) or 'none'}"),
            advice="--param overrides a parameter this run's comprehension "
                   "model already knows about; it is not a way to add one "
                   "the model never recorded. Fix the spelling, or re-run "
                   "the understand phase if the parameter is genuinely "
                   "missing from request_shape.",
        )

    if shape is None:
        return {}, []

    values: dict[str, list[Any]] = {}
    notes: list[str] = []
    for param in shape.parameters:
        if param.location != "query" or param.name in PAGINATION_PARAM_NAMES:
            continue

        if param.name in overrides:
            values[param.name] = [overrides[param.name]]
            notes.append(f"?{param.name} set from --param "
                         f"({overrides[param.name]!r}), overriding whatever "
                         f"the code or the database would have supplied")
            continue

        if not param.required:
            continue

        if param.allowed_values:
            values[param.name] = list(param.allowed_values)
            notes.append(f"?{param.name} varied across its recorded allowed "
                         f"values ({', '.join(str(v) for v in param.allowed_values)})")
            continue

        if param.default is not None:
            values[param.name] = [param.default]
            notes.append(f"?{param.name} fixed at its recorded default "
                         f"({param.default!r})")
            continue

        sampled, sample_detail = _sampled_query_values(param.name, model, sampler)
        if sampled:
            values[param.name] = sampled
            notes.append(f"?{param.name} varied across values sampled from "
                         f"the database ({sample_detail}), not from the code")
            continue

        raise CorpusRefusal(
            what=f"required query parameter '{param.name}' has no "
                 f"determinable value",
            detail=(f"request_shape records '{param.name}' as required; "
                    f"tried: allowed_values (none recorded), default (none "
                    f"recorded), sampling the database ({sample_detail})"),
            advice="Re-run the understand phase so it records the "
                   "parameter's allowed values (an enum, a @Pattern, an "
                   "explicit comparison) or its default, seed the database "
                   "with values for the column it filters, or pass "
                   f"--param {param.name}=VALUE to supply one directly.",
        )
    return values, notes


def generate_corpus(
    model: EndpointModel,
    *,
    sampler: Any,
    requests_per_value: int | None = None,
    param_overrides: dict[str, str] | None = None,
) -> tuple[RequestCorpus, list[str]]:
    """Build the corpus for `model.route`.

    `sampler` is anything with `sample(table, column, limit) -> list[Any]`,
    so this function is testable without a database and the live
    implementation stays in one place (`senapathy/tools/sample.py`).

    `param_overrides` is the operator's `--param NAME=VALUE` escape hatch —
    see `_required_query_values` for how it ranks against the code and the
    database.

    Returns (corpus, notes). `notes` records what could NOT be inferred, so
    the trace says which parts of the corpus are evidence and which are a
    stated limitation.
    """
    method, path = split_route(model.route)
    if not path or path == "/":
        raise CorpusRefusal(
            what="the comprehension model records no usable route",
            detail=f"route={model.route!r}",
            advice="Re-run the understand phase; a corpus cannot be built "
                   "without the endpoint's path.",
        )

    notes: list[str] = []
    path_param_names = path_parameters(path)
    has_per_row_access = bool(model.per_row_accesses())

    # ── resolve each path parameter TWICE: the diverse, stride-sampled
    #    values that serve the HELD-BACK set exactly as before, and --
    #    when the model recorded per-row work and a child table can be
    #    resolved for it -- a dense-biased set of the same parameter's
    #    values for BENCHMARK. One corpus serves two jobs with opposite
    #    needs: the oracle needs COVERAGE (sparse rows, empty pages,
    #    boundaries -- behaviour must hold everywhere); the benchmark needs
    #    COST (the shapes where the endpoint's work actually lives). A
    #    single blind-stride sample across the id space served neither:
    #    stride lands mostly in the middle of a skewed distribution, so a
    #    benchmark built from it measured a tenth of the real per-row cost.
    sparse_values: dict[str, list[Any]] = {}
    dense_values: dict[str, list[Any]] = {}
    dense_counts: dict[str, dict[Any, int]] = {}
    sparse_counts: dict[str, dict[Any, int]] = {}
    for parameter in path_param_names:
        table = table_for_parameter(parameter, model)
        if table is None:
            raise CorpusRefusal(
                what=f"path parameter {{{parameter}}} could not be resolved "
                     f"to a table",
                detail=(f"the endpoint model records tables "
                        f"{[t.table for t in model.tables] or 'none'}, and "
                        f"none matches the path segment before "
                        f"{{{parameter}}}"),
                advice="Re-run the understand phase so the model records the "
                       "tables this endpoint reads, or extend "
                       "table_for_parameter() if the naming is unconventional.",
            )
        column = _column_for_parameter(parameter, table, model)
        sampled = list(sampler.sample(table, column, limit=6))
        if not sampled:
            raise CorpusRefusal(
                what=f"no values exist for {{{parameter}}}",
                detail=f"{table}.{column} returned no rows",
                advice="Seed the database. A corpus of requests that all 404 "
                       "measures the error path, not the endpoint.",
            )
        sparse_values[parameter] = sampled
        notes.append(f"{{{parameter}}} held-back values sampled from "
                     f"{table}.{column} ({len(sampled)} value(s)), spread "
                     f"across the table on purpose -- this is the diversity "
                     f"set, unchanged")

        dense_pairs: list[tuple[Any, int]] = []
        child = _child_table_for_parameter(table, model) if has_per_row_access else None
        if child is not None:
            child_table, child_column = child
            try:
                dense_pairs = list(sampler.sample_dense(
                    child_table, child_column, limit=MAX_DENSE_VALUES))
            except Exception as exc:  # noqa: BLE001 -- folded into a note
                notes.append(
                    f"{{{parameter}}} dense sampling via "
                    f"{child_table}.{child_column} failed "
                    f"({type(exc).__name__}: {exc}); benchmark falls back "
                    f"to the same values as held-back")

        if dense_pairs:
            dense_values[parameter] = [value for value, _ in dense_pairs]
            dense_counts[parameter] = dict(dense_pairs)
            counts_fmt = ", ".join(f"{v}={c:,}" for v, c in dense_pairs)
            notes.append(
                f"{{{parameter}}} benchmark values biased toward the dense "
                f"end via {child_table}.{child_column} ({counts_fmt})")
            try:
                sparse_counts[parameter] = sampler.count_by_value(
                    child_table, child_column, sparse_values[parameter])
            except Exception:  # noqa: BLE001 -- notes just lose the count
                sparse_counts[parameter] = {}
        else:
            dense_values[parameter] = sparse_values[parameter]
            if has_per_row_access:
                notes.append(
                    f"{{{parameter}}} has per-row work recorded but no "
                    f"resolvable child table (no foreign key found pointing "
                    f"at {table}); benchmark uses the same values as "
                    f"held-back")

    # ── resolve every REQUIRED query parameter request_shape records ────
    # (never silently omitted -- see _required_query_values). The same
    # values serve both sets: a status filter isn't a density axis.
    query_values, query_notes = _required_query_values(
        model, sampler=sampler, overrides=param_overrides)
    notes.extend(query_notes)
    combined_names = path_param_names + [n for n in query_values
                                         if n not in path_param_names]
    for values in (sparse_values, dense_values):
        values.update(query_values)

    # ── the cost axes: page size and depth. Held-back keeps the existing
    #    spread (including a page of 1 and an empty page) for boundary
    #    coverage; benchmark is biased toward the endpoint's own maximum --
    #    a page of 1 tells you nothing about per-row cost ─────────────────
    pagination = model.pagination
    held_back_sizes: tuple[int, ...] = _PAGE_SIZES
    benchmark_sizes: tuple[int, ...] = _PAGE_SIZES
    depths: tuple[int, ...] = _PAGE_DEPTHS
    paginates = pagination is not None and pagination.mechanism == "offset"
    if paginates and pagination is not None and pagination.max_page_size:
        cap = pagination.max_page_size
        held_back_sizes = tuple(sorted({min(s, cap) for s in _PAGE_SIZES} | {cap}))
        benchmark_sizes = (cap,)
        notes.append(f"held-back page sizes capped at the endpoint's own "
                     f"maximum ({cap}); benchmark page size fixed at that "
                     f"same maximum")
    elif paginates:
        benchmark_sizes = (max(_PAGE_SIZES),)
    if not paginates:
        held_back_sizes, benchmark_sizes, depths = (0,), (0,), (0,)
        notes.append(
            f"pagination mechanism is "
            f"{pagination.mechanism if pagination else 'unrecorded'}; no "
            f"page/size parameters generated")

    if method != "GET":
        # A request BODY still cannot be inferred: RequestShape records path
        # and query parameters, not the fields of a JSON body, so nothing
        # here describes what to send for a method that needs one. Refusing
        # is the honest answer -- the alternative is inventing a body and
        # measuring a 400.
        raise CorpusRefusal(
            what=f"{method} requests need a body this harness cannot infer",
            detail="request_shape records path and query parameters, not a "
                   "request body, so the body's fields are unknown",
            advice="Generation today covers GET endpoints, whose parameters "
                   "are in the path or query string. A body contract is a "
                   "separate extension; see SENAPATHY.md.",
        )

    # ── build the requests, once per set, from that set's own values ────
    def _build(
        values: dict[str, list[Any]], sizes: tuple[int, ...], *,
        tag: str, counts: dict[str, dict[Any, int]],
    ) -> list[RequestSpec]:
        built: list[RequestSpec] = []
        for parameter_values in _combinations(values, combined_names):
            concrete = path
            label_bits = []
            fixed_query_bits = []
            for name, value in parameter_values.items():
                if name in path_param_names:
                    concrete = concrete.replace("{" + name + "}", str(value))
                    concrete = concrete.replace(":" + name, str(value))
                else:
                    fixed_query_bits.append(f"{name}={value}")
                label_bits.append(f"{name}{value}")

            reason_bits = [
                f"{name}={parameter_values[name]} ({count:,} child rows)"
                for name in path_param_names
                if (count := counts.get(name, {}).get(parameter_values.get(name)))
                is not None
            ]
            if reason_bits:
                reason = f"{tag}: {'; '.join(reason_bits)}"
                if tag == "sparse":
                    reason += ", held back for coverage"
            elif tag == "sparse":
                reason = "sparse: held back for coverage"
            else:
                reason = ("generated: path/query params from the "
                           "comprehension model")

            for size in sizes:
                for depth in depths:
                    query_bits = list(fixed_query_bits)
                    suffix = ""
                    if paginates:
                        query_bits.append(f"page={depth}")
                        query_bits.append(f"size={size}")
                        suffix = f"-p{depth}-s{size}"
                    query = ("?" + "&".join(query_bits)) if query_bits else ""
                    note = (f"{reason}; page depth {depth}, page size {size}"
                            if paginates else reason)
                    built.append(RequestSpec(
                        id=f"{tag}-{'-'.join(label_bits) or 'req'}{suffix}",
                        method=method,
                        path=concrete + query,
                        note=note,
                    ))
        return built

    benchmark = _build(dense_values, benchmark_sizes, tag="dense",
                       counts=dense_counts)
    held_back = _build(sparse_values, held_back_sizes, tag="sparse",
                       counts=sparse_counts)

    if not benchmark and not held_back:
        raise CorpusRefusal(
            what="no requests could be generated",
            detail=f"route={model.route!r}, parameters={combined_names}",
            advice="Check the comprehension model records the route and the "
                   "tables the endpoint reads.",
        )
    if not benchmark:
        benchmark, held_back = held_back, []
        notes.append("dense sampling produced nothing; held-back requests "
                     "promoted to benchmark instead")

    corpus = RequestCorpus(
        benchmark=benchmark, held_back=held_back,
        hash=corpus_hash(benchmark, held_back),
    )
    return corpus, notes


def probe_corpus(
    corpus: RequestCorpus,
    base_url: str,
    *,
    timeout_s: float = 30.0,
    client: httpx.Client | None = None,
) -> RequestCorpus:
    """Fire every generated request once against a live app, BEFORE anything
    is sealed or measured, and record what it actually returned.

    A corpus whose requests do not actually work is not a corpus: this is
    what catches a generated request silently missing a required parameter
    before it burns an oracle recording or a hundred baseline iterations on
    an endpoint that never ran.

    The rule is not "no 4xx" — a deliberate error-path request is wanted,
    and its 4xx becomes its `expected_status`, recorded here, not treated as
    a problem. The rule is narrower: at least one request must come back
    2xx, or nothing here is exercising the endpoint at all, and that is
    refused by status distribution and first response body rather than
    discovered four phases later as "all 100 baseline requests returned
    400".

    Returns a NEW corpus — every spec with its `expected_status` filled in —
    and therefore a new `corpus_hash`, since what a request is expected to
    return is as much a property of that corpus as its path is.
    """
    owns_client = client is None
    http = client or httpx.Client(timeout=timeout_s, follow_redirects=False)
    status_counts: dict[int, int] = {}
    first_body: str | None = None

    def probe_bucket(bucket: list[RequestSpec]) -> list[RequestSpec]:
        nonlocal first_body
        probed: list[RequestSpec] = []
        for spec in bucket:
            url = base_url.rstrip("/") + spec.path
            try:
                response = http.request(
                    spec.method.upper(), url, json=spec.body,
                    headers=spec.headers or None,
                )
            except httpx.HTTPError as exc:
                raise CorpusRefusal(
                    what=f"request {spec.id} ({spec.method} {url}) failed at "
                         f"the transport layer while probing",
                    detail=f"{type(exc).__name__}: {exc}",
                    advice="The app must be up and healthy before a corpus "
                           "can be probed. Fix the environment and generate "
                           "the corpus again; a corpus that was never "
                           "actually fired is not sealed.",
                ) from exc
            status_counts[response.status_code] = (
                status_counts.get(response.status_code, 0) + 1)
            if first_body is None:
                first_body = response.text[:2000]
            probed.append(spec.model_copy(
                update={"expected_status": response.status_code}))
        return probed

    try:
        probed_benchmark = probe_bucket(corpus.benchmark)
        probed_held_back = probe_bucket(corpus.held_back)
    finally:
        if owns_client:
            http.close()

    if not any(200 <= code < 300 for code in status_counts):
        raise CorpusRefusal(
            what="every probed request failed",
            detail=(f"status counts: {dict(sorted(status_counts.items()))}; "
                    f"first response body: {first_body!r}"),
            advice="A corpus that is entirely errors is not exercising the "
                   "endpoint — a deliberate error-path request is fine "
                   "alongside requests that succeed, not instead of them. "
                   "Check the generated requests against the comprehension "
                   "model's request_shape; a wrong or missing required "
                   "parameter value produces exactly this.",
        )

    return RequestCorpus(
        benchmark=probed_benchmark, held_back=probed_held_back,
        hash=corpus_hash(probed_benchmark, probed_held_back),
    )


def _combinations(values: dict[str, list[Any]],
                  order: list[str]) -> list[dict[str, Any]]:
    """One dict per sampled value, walking the parameters in order.

    Deliberately NOT a cartesian product: two parameters with six values
    each would be 36 path shapes times the page axes, which is a corpus
    nobody can run in a warmup budget. Values are zipped instead, so each
    sampled row is used once.
    """
    if not order:
        return [{}]
    longest = max(len(values[name]) for name in order)
    out: list[dict[str, Any]] = []
    for index in range(longest):
        out.append({name: values[name][index % len(values[name])]
                    for name in order})
    return out
