"""The facts the plan agent is given, rendered from artifacts.

Every number and every path in here is read off `hypotheses.json` or
`endpoint_model.json`. Nothing is computed in the prompt and nothing is left
for the model to work out (invariants 4 and 5) — including the fixability
figures, which arrive with their reasons attached so that a 0.05 is explained
rather than mysterious.
"""

from __future__ import annotations

from ..contracts import EndpointModel, Hypothesis, RankedHypothesis, SqlDigestStat
from ..measure.baseline import is_session_control
from .fixability import unbounded_fields


def render_brief(*, ranked: list[RankedHypothesis],
                 hypotheses: dict[str, Hypothesis],
                 considered: list[str],
                 model: EndpointModel,
                 baseline_digests: list[SqlDigestStat] | None = None) -> str:
    out: list[str] = []
    out.extend(_ranking(ranked, hypotheses, considered))
    out.append("")
    out.extend(_response(model))
    out.append("")
    out.extend(_per_row(model))
    out.append("")
    out.extend(_tables(model))
    out.append("")
    out.extend(_baseline_sql(baseline_digests or []))
    out.append("")
    out.extend(_files(model))
    return "\n".join(out)


def _baseline_sql(digests: list[SqlDigestStat]) -> list[str]:
    """Every application statement this endpoint actually ran during
    baseline measurement, with its own digest hash and call count.

    A `sql_count` candidate must cite one of these BY HASH (`baseline_digest`)
    with the `calls` figure it names (`baseline_calls`) — code checks the
    citation against this same list, not the candidate's prose. A statement
    that is not here was never measured running at all; a candidate that
    claims to collapse calls of one is making that up.
    """
    out = ["=== BASELINE: WHAT ACTUALLY RAN, PER STATEMENT ===",
           "",
           "  A candidate with mechanism 'sql_count' must name ONE of these by",
           "  its digest hash (baseline_digest) and the calls figure below",
           "  (baseline_calls) — copied, not estimated. A digest not listed",
           "  here was not measured running, so a candidate cannot claim to",
           "  reduce its call count.",
           ""]
    app_digests = [d for d in digests if not is_session_control(d)]
    if not app_digests:
        out.append("  none recorded")
        return out
    for digest in app_digests:
        out.append(f"  digest {digest.digest}")
        out.append(f"    sql    {digest.sample_sql.strip()[:160]}")
        out.append(f"    calls  {digest.calls}  (rows examined "
                   f"{digest.rows_examined:,}, rows sent {digest.rows_sent:,})")
    return out


def _ranking(ranked: list[RankedHypothesis], hypotheses: dict[str, Hypothesis],
             considered: list[str]) -> list[str]:
    out = ["=== RANKED BY EXPECTED VALUE (attributed ms x fixability) ===",
           "",
           "Fixability is computed by code from a fixed policy table. It is the",
           "probability that a fix exists which does NOT change the response.",
           "You cannot argue with it; you can only work inside it.",
           ""]
    for position, entry in enumerate(ranked, start=1):
        hypothesis = hypotheses.get(entry.hypothesis_id)
        open_to_you = " <-- OPEN FOR CANDIDATES" if entry.hypothesis_id in considered else ""
        unclassified = "" if entry.fixability.classified else "  [NOT CLASSIFIED]"
        out.append(f"  {position}. [{entry.hypothesis_id}] {entry.claim}"
                   f"{unclassified}{open_to_you}")
        out.append(f"     layer           {entry.layer}")
        out.append(f"     attributed      {entry.attributed_ms:.1f} ms")
        if entry.fixability.classified:
            out.append(f"     fixability      {entry.fixability.fixability:.2f}"
                       f"  ->  expected value {entry.expected_value_ms:.1f} ms")
        else:
            out.append(f"     fixability      NOT CLASSIFIED — no policy row "
                       f"matched, so {entry.fixability.fixability:.2f} is a "
                       f"placeholder and this entry ranks below every "
                       f"classified one")
        out.append(f"     why             {entry.fixability.reason}")
        if entry.fixability.blocked_by:
            out.append(f"     BLOCKED BY      {entry.fixability.blocked_by}")
        if hypothesis is not None:
            out.append(f"     mechanism       {hypothesis.mechanism}")
        out.append("")
    return out


def _response(model: EndpointModel) -> list[str]:
    shape = model.response_shape
    out = ["=== THE RESPONSE IS THE CONTRACT ==="]
    if shape is None:
        out.append("  The comprehension map recorded no response shape. Treat "
                   "every returned field as frozen.")
        return out
    out.append(f"  DTO                     {shape.dto or 'unknown'}")
    out.append(f"  fields returned         {', '.join(shape.fields_returned) or 'unknown'}")
    out.append(f"  computed fields         {', '.join(shape.computed_fields) or 'none'}")
    fields = unbounded_fields(shape)
    out.append(f"  UNBOUNDED COLLECTIONS   {', '.join(fields) or 'none'}")
    if fields:
        out.append("")
        out.append(f"  The size of {', '.join(fields)} scales with the data and is "
                   f"part of the API contract.")
        out.append("  No candidate may change, trim, cap, page or lazily load it.")
    return out


def _per_row(model: EndpointModel) -> list[str]:
    out = ["=== WHAT HAPPENS PER RESULT ROW ==="]
    accesses = model.per_row_accesses()
    out.append(f"  database accesses that run once per row ({len(accesses)}):")
    if not accesses:
        out.append("    none recorded")
    for access in accesses:
        out.append(f"    - {access.trigger}  [{access.kind}]  "
                   f"{access.file}:{access.line}")
        if access.tables:
            out.append(f"      tables: {', '.join(access.tables)}")
        if access.sql:
            out.append(f"      sql: {access.sql.strip()[:160]}")

    out.append("")
    out.append(f"  non-query work that runs once per row ({len(model.per_row_work)}):")
    if not model.per_row_work:
        out.append("    none recorded")
    for work in model.per_row_work:
        flag = "UNBOUNDED" if not work.bounded else f"bounded by {work.bound}"
        out.append(f"    - {work.description}  [{work.kind}]  "
                   f"{work.file}:{work.line}")
        out.append(f"      scales with {work.scales_with} — {flag}")

    if model.pagination is not None:
        page = model.pagination
        cap = page.max_page_size if page.max_page_size is not None else "UNCAPPED"
        out.append("")
        out.append(f"  pagination: {page.mechanism}, count query "
                   f"{'fires' if page.fires_count_query else 'does not fire'}, "
                   f"max page size {cap}")
    return out


def _tables(model: EndpointModel) -> list[str]:
    out = ["=== TABLES AND THE INDEXES THAT ALREADY EXIST ===",
           "",
           "  Read this before proposing an index. An index listed here exists;",
           "  proposing it again will be refused.",
           ""]
    if not model.tables:
        out.append("  no tables recorded")
        return out
    for table in model.tables:
        rows = f"{table.approx_rows:,}" if table.approx_rows is not None else "unknown"
        out.append(f"  {table.table}  (~{rows} rows, entity {table.entity or 'none'})")
        out.append(f"    columns read by this endpoint: "
                   f"{', '.join(table.columns_read) or 'unknown'}")
        out.append(f"    all columns: {', '.join(table.all_columns) or 'unknown'}")
        if table.existing_indexes:
            for index in table.existing_indexes:
                kind = "UNIQUE" if index.unique else "index"
                card = (f", cardinality {index.cardinality:,}"
                        if index.cardinality is not None else "")
                out.append(f"    EXISTING {kind} {index.name} "
                           f"({', '.join(index.columns)}){card}")
        else:
            out.append(f"    EXISTING indexes: NOT MEASURED. Call "
                       f"get_schema(['{table.table}']) before proposing an "
                       f"index here — a proposal on an unmeasured table is "
                       f"refused, because nothing knows what is already there.")
        out.append("")
    return out


def _files(model: EndpointModel) -> list[str]:
    """The files on the execution path, as the starting point for an allowlist."""
    seen: list[str] = []
    for hop in model.call_chain:
        if hop.file not in seen:
            seen.append(hop.file)
    for access in model.data_access:
        if access.file not in seen:
            seen.append(access.file)
    out = ["=== FILES ON THE EXECUTION PATH ===",
           "",
           "  files_to_touch becomes the patch agent's write allowlist. Name only",
           "  files you have read and intend to change.",
           ""]
    out.extend(f"  {path}" for path in seen)
    return out
