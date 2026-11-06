"""Phase P2: generate THIS run's request corpus, from evidence in hand.

What this replaces: a 12-request JSON file shipped inside the Python package,
every request a `POST /api/customers/search` against one repo's seed data,
used as the DEFAULT for every run. A run against any other endpoint
silently baselined, measured and gated that one. The corpus is per-run
evidence; it was code.

Where the requests come from:
  * the route, path parameters and pagination come from the comprehension
    model, which read them out of the controller;
  * parameter VALUES come from sampling the database — real ids that exist,
    chosen to span cheap and expensive shapes — never invented;
  * the axes that change cost are varied deliberately: page depth, page
    size, and a selectivity spread across the sampled ids.

Determinism: sampling is ordered and seeded, so the same model against the
same database produces a byte-identical corpus and therefore the same
`corpus_hash`. That matters because the oracle is keyed by that hash: a
generator that shuffled would invalidate the recording on every run.
"""

from __future__ import annotations

import re
from typing import Any

from .contracts import RequestCorpus, RequestSpec
from .contracts.comprehension import EndpointModel
from .errors import ContractViolation
from .measure.corpus import corpus_hash

# `{id}` or `:id` in a route template.
_PATH_PARAM = re.compile(r"\{(\w+)\}|:(\w+)\b")

# Page sizes worth measuring: the cheapest shape that still walks the
# mapping loop, a typical page, and the largest the endpoint admits. Cost
# per request scales with this, so it is the primary axis.
_PAGE_SIZES = (1, 10, 50)
_PAGE_DEPTHS = (0, 2)

# One in every N benchmark requests is held back instead. A third is far
# more than a train/test split needs statistically, but the held-back set is
# the only check on a fix tuned to the measured requests, so it is worth the
# requests.
_HELD_BACK_EVERY = 4


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


def generate_corpus(
    model: EndpointModel,
    *,
    sampler: Any,
    requests_per_value: int | None = None,
) -> tuple[RequestCorpus, list[str]]:
    """Build the corpus for `model.route`.

    `sampler` is anything with `sample(table, column, limit) -> list[Any]`,
    so this function is testable without a database and the live
    implementation stays in one place (`perfagent/tools/sample.py`).

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
    parameters = path_parameters(path)

    # ── resolve each path parameter to real values from the database ────
    values: dict[str, list[Any]] = {}
    for parameter in parameters:
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
        values[parameter] = sampled
        notes.append(f"{{{parameter}}} sampled from {table}.{column} "
                     f"({len(sampled)} value(s))")

    # ── the cost axes: page size and depth, bounded by what the endpoint
    #    actually admits ──────────────────────────────────────────────────
    pagination = model.pagination
    sizes: tuple[int, ...] = _PAGE_SIZES
    depths: tuple[int, ...] = _PAGE_DEPTHS
    paginates = pagination is not None and pagination.mechanism == "offset"
    if paginates and pagination is not None and pagination.max_page_size:
        cap = pagination.max_page_size
        sizes = tuple(sorted({min(s, cap) for s in _PAGE_SIZES} | {cap}))
        notes.append(f"page sizes capped at the endpoint's own maximum ({cap})")
    if not paginates:
        sizes, depths = (0,), (0,)
        notes.append(
            f"pagination mechanism is "
            f"{pagination.mechanism if pagination else 'unrecorded'}; no "
            f"page/size parameters generated")

    if method != "GET":
        # A request BODY cannot be inferred: the comprehension model has a
        # ResponseShape and no symmetric RequestShape, so nothing recorded
        # the body's fields. Refusing is the honest answer -- the
        # alternative is inventing a body and measuring a 400.
        raise CorpusRefusal(
            what=f"{method} requests need a body this harness cannot infer",
            detail="the comprehension model records a ResponseShape but no "
                   "request shape, so the body's fields are unknown",
            advice="Generation today covers GET endpoints whose parameters "
                   "are in the path or query string. A RequestShape contract "
                   "(the symmetric counterpart of ResponseShape) is what "
                   "this needs; see PERFAGENT.md.",
        )

    # ── build the requests ──────────────────────────────────────────────
    specs: list[RequestSpec] = []
    index = 0
    for parameter_values in _combinations(values, parameters):
        concrete = path
        label_bits = []
        for name, value in parameter_values.items():
            concrete = concrete.replace("{" + name + "}", str(value))
            concrete = concrete.replace(":" + name, str(value))
            label_bits.append(f"{name}{value}")
        for size in sizes:
            for depth in depths:
                query = ""
                suffix = ""
                if paginates:
                    query = f"?page={depth}&size={size}"
                    suffix = f"-p{depth}-s{size}"
                specs.append(RequestSpec(
                    id=f"{'-'.join(label_bits) or 'req'}{suffix}",
                    method=method,
                    path=concrete + query,
                    note=(f"generated: path params from sampled rows; "
                          f"page depth {depth}, page size {size}"
                          if paginates else "generated: no pagination"),
                ))
                index += 1

    if not specs:
        raise CorpusRefusal(
            what="no requests could be generated",
            detail=f"route={model.route!r}, parameters={parameters}",
            advice="Check the comprehension model records the route and the "
                   "tables the endpoint reads.",
        )

    # ── the held-back split ─────────────────────────────────────────────
    # Deterministic (every Nth), not random: a random split changes the
    # corpus hash between runs and invalidates the oracle.
    benchmark = [s for i, s in enumerate(specs) if i % _HELD_BACK_EVERY]
    held_back = [s for i, s in enumerate(specs) if not i % _HELD_BACK_EVERY]
    if not benchmark:
        benchmark, held_back = specs, []
        notes.append("too few requests to hold any back")

    corpus = RequestCorpus(
        benchmark=benchmark, held_back=held_back,
        hash=corpus_hash(benchmark, held_back),
    )
    return corpus, notes


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
