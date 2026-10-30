"""The comprehension artifact, rendered for a person.

Every number in here is read off the artifact. Nothing is recomputed into a
nicer story, and there is no sentence a model wrote.

The section order is the point: unbounded per-row work comes before the query
list, because on the endpoint this product was built against the queries were
7% of the latency and the per-row string building was the rest. A human
skimming this should see the real problem in the first five seconds.
"""

from __future__ import annotations

from typing import Any

from ..contracts import EndpointModel

INDENT = "  "
WARN = "⚠ UNBOUNDED"

# How the call chain is annotated. Each data access and each per-row work item
# is attributed to the hop in the SAME FILE with the nearest line — a derived
# association, not something the model stated, so it is used for display only
# and never written back into the artifact.
MAX_ATTRIBUTION_DISTANCE = 60


def render_endpoint_model(model: EndpointModel) -> str:
    out: list[str] = []
    out.extend(_header(model))
    out.append("")
    out.extend(_call_chain(model))
    out.append("")
    out.extend(_per_row_work(model))
    out.append("")
    out.extend(_data_access(model))
    out.append("")
    out.extend(_response(model))
    out.append("")
    out.extend(_pagination(model))
    out.append("")
    out.extend(_tables(model))
    if model.transactions:
        out.append("")
        out.extend(_transactions(model))
    if model.entities:
        out.append("")
        out.extend(_entities(model))
    if model.caching:
        out.append("")
        out.extend(_caching(model))
    if model.hidden_query_triggers:
        out.append("")
        out.extend(_hidden_triggers(model))
    if model.external_calls:
        out.append("")
        out.extend(_external_calls(model))
    if model.tests_covering:
        out.append("")
        out.append("TESTS COVERING THIS ROUTE")
        out.extend(f"{INDENT}{name}" for name in model.tests_covering)
    out.append("")
    out.extend(_uncertainties(model))
    if model.prompt_hash:
        out.append("")
        out.append(f"prompt version    {model.prompt_hash}")
    return "\n".join(out)


# ──────────────────────────────  sections  ───────────────────────────────


def _header(model: EndpointModel) -> list[str]:
    stamp = ""
    if model.base_commit:
        sha, _, dirty = model.base_commit.partition("+")
        stamp = f"base {sha[:7]}" + (f" ({dirty})" if dirty else "")
    return [f"{model.route:<45} {stamp}".rstrip()]


def _call_chain(model: EndpointModel) -> list[str]:
    # "order of execution, not nesting depth" is not a caveat — it is what the
    # artifact records. A tree drawn from a flat ordered list would invent a
    # call depth nobody measured.
    out = ["CALL CHAIN                          "
           "(order of execution, not nesting depth)"]
    if not model.call_chain:
        return out + [f"{INDENT}(none recorded)"]

    for position, hop in enumerate(model.call_chain, start=1):
        joint = "│" if position < len(model.call_chain) else " "
        note = _hop_note(model, position - 1)
        location = f"{_basename(hop.file)}:{hop.line}"
        out.append(f"{INDENT}{position:>2}  {hop.method:<40} "
                   f"{location:<34}{note}".rstrip())
        out.append(f"{INDENT}{joint}     {hop.does}")
    return out


def _per_row_work(model: EndpointModel) -> list[str]:
    unbounded = model.unbounded_per_row_work()
    out = [f"PER-ROW WORK      {len(model.per_row_work)} item(s), "
           f"{len(unbounded)} UNBOUNDED"]
    if not model.per_row_work:
        out.append(f"{INDENT}(the agent reported none — on an endpoint with "
                   f"per-row queries, treat that as suspect)")
        return out

    # Unbounded first. The ordering is itself the finding.
    for work in sorted(model.per_row_work, key=lambda w: w.bounded):
        marker = WARN if not work.bounded else "  bounded  "
        out.append(f"{INDENT}{marker}  {work.description}")
        scale = f"scales with {work.scales_with}"
        if work.bounded and work.bound:
            scale += f" (bounded by {work.bound})"
        out.append(f"{INDENT}{' ' * len(marker)}  "
                   f"{work.file}:{work.line} · {work.kind} · {scale}")
    return out


def _data_access(model: EndpointModel) -> list[str]:
    per_row = model.per_row_accesses()
    out = [f"DATA ACCESS       {len(model.data_access)} total, "
           f"{len(per_row)} per result row"]
    if not model.data_access:
        return out + [f"{INDENT}(none recorded)"]

    for access in model.data_access:
        scope = "per row   " if access.executes_per_row else "1×/request"
        tables = ", ".join(access.tables) or "—"
        out.append(f"{INDENT}{scope}   {access.trigger:<38} "
                   f"{access.kind:<16} {tables}")
        out.append(f"{INDENT}             {access.file}:{access.line}")
        if access.sql:
            out.append(f"{INDENT}             {_one_line(access.sql)}")
        if access.note:
            out.append(f"{INDENT}             note: {_one_line(access.note)}")
    return out


def _response(model: EndpointModel) -> list[str]:
    shape = model.response_shape
    if shape is None:
        return ["RESPONSE          (not recorded)"]

    out = [f"RESPONSE          {shape.dto or '(dto not named)'}"]
    for field in shape.unbounded_collections:
        out.append(f"{INDENT}unbounded       {field:<16}"
                   f"← size is part of the API contract")
    for field in shape.computed_fields:
        out.append(f"{INDENT}computed        {field}")
    plain = [f for f in shape.fields_returned
             if f not in shape.unbounded_collections
             and f not in shape.computed_fields]
    if plain:
        out.append(f"{INDENT}plain           {', '.join(plain)}")
    if not shape.fields_returned and not shape.unbounded_collections:
        out.append(f"{INDENT}(no fields recorded)")
    return out


def _pagination(model: EndpointModel) -> list[str]:
    page = model.pagination
    if page is None:
        return ["PAGINATION        (not recorded)"]
    parts: list[str] = [str(page.mechanism)]
    parts.append("count query fires" if page.fires_count_query
                 else "no count query")
    parts.append(f"page size max {page.max_page_size}"
                 if page.max_page_size is not None else "page size uncapped")
    return [f"PAGINATION        {' · '.join(parts)}"]


def _tables(model: EndpointModel) -> list[str]:
    out = [f"TABLES            {len(model.tables)}"]
    if not model.tables:
        out.append(f"{INDENT}(none recorded — get_schema may have been "
                   f"unavailable)")
        return out

    for table in model.tables:
        count = len(table.existing_indexes)
        if not _was_introspected(table):
            # "NO INDEXES" and "nobody looked" are different facts, and only
            # one of them is a finding. Printing the first when the second is
            # true is exactly the unbacked claim invariant 4 forbids.
            indexes = "not introspected"
        elif count:
            indexes = (f"{count} index{'es' if count != 1 else ''} "
                       f"({', '.join(i.name for i in table.existing_indexes)})")
        else:
            indexes = "NO INDEXES"
        rows = (f"~{table.approx_rows:,} rows"
                if table.approx_rows is not None else "rows unknown")
        reads = (f"reads {len(table.columns_read)} of "
                 f"{len(table.all_columns)} columns"
                 if table.columns_read and table.all_columns
                 else f"reads {', '.join(table.columns_read) or '?'}")
        out.append(f"{INDENT}{table.table:<12}{indexes:<34}"
                   f"{rows:>18}    {reads}")
        for index in table.existing_indexes:
            unique = "UNIQUE " if index.unique else ""
            card = (f"  cardinality {index.cardinality:,}"
                    if index.cardinality is not None else "")
            out.append(f"{INDENT}   {unique}{index.name}"
                       f"({', '.join(index.columns)}){card}")
    return out


def _transactions(model: EndpointModel) -> list[str]:
    out = ["TRANSACTIONS"]
    for transaction in model.transactions:
        flag = "readOnly" if transaction.read_only else "read-write"
        out.append(f"{INDENT}{transaction.method:<40} "
                   f"{transaction.propagation} · {flag}   "
                   f"({transaction.citation.ref()})")
    return out


def _entities(model: EndpointModel) -> list[str]:
    out = ["ENTITIES"]
    for entity in model.entities:
        out.append(f"{INDENT}{entity.name} -> table {entity.table}   "
                   f"({entity.citation.ref()})")
        for relation in entity.relations:
            extra = []
            if relation.batch_size is not None:
                extra.append(f"@BatchSize({relation.batch_size})")
            if relation.fetch_mode is not None:
                extra.append(f"@Fetch({relation.fetch_mode})")
            suffix = f"   {' '.join(extra)}" if extra else ""
            out.append(f"{INDENT}   {relation.field}: {relation.type} "
                       f"{relation.fetch} -> {relation.target}{suffix}")
    return out


def _caching(model: EndpointModel) -> list[str]:
    out = ["CACHING"]
    for cache in model.caching:
        name = f" [{cache.cache_name}]" if cache.cache_name else ""
        out.append(f"{INDENT}{cache.kind} on {cache.on}{name}   "
                   f"({cache.citation.ref()})")
    return out


def _hidden_triggers(model: EndpointModel) -> list[str]:
    out = ["HIDDEN QUERY TRIGGERS"]
    for trigger in model.hidden_query_triggers:
        out.append(f"{INDENT}{trigger.kind} on {trigger.on}: "
                   f"{_one_line(trigger.detail)}   "
                   f"({trigger.citation.ref()})")
    return out


def _external_calls(model: EndpointModel) -> list[str]:
    out = ["EXTERNAL CALLS"]
    for call in model.external_calls:
        out.append(f"{INDENT}{call.method} -> {call.target} [{call.kind}]   "
                   f"({call.citation.ref()})")
    return out


def _uncertainties(model: EndpointModel) -> list[str]:
    out = [f"UNCERTAINTIES     {len(model.uncertainties)}"]
    for item in model.uncertainties:
        out.append(f"{INDENT}· {item}")
    if not model.uncertainties:
        out.append(f"{INDENT}(the agent declared none — treat that as a "
                   f"finding of its own)")
    return out


# ────────────────────────────  attribution  ──────────────────────────────


def _hop_note(model: EndpointModel, index: int) -> str:
    """What happens at this hop, derived from what is attributed to it."""
    accesses = [a for a in model.data_access
                if _nearest_hop(model, a.file, a.line) == index]
    work = [w for w in model.per_row_work
            if _nearest_hop(model, w.file, w.line) == index]

    if any(not w.bounded for w in work):
        return WARN
    if any(a.executes_per_row for a in accesses):
        return "per row"
    if accesses:
        return f"{len(accesses)} query/request"
    if work:
        return "per row"
    return ""


def _nearest_hop(model: EndpointModel, file: str, line: int) -> int | None:
    """Index of the call-chain hop this location most likely belongs to.

    Same file, nearest line, within a window. Display only — a wrong guess
    moves a label, it does not change a fact.
    """
    candidates = [(abs(hop.line - line), i)
                  for i, hop in enumerate(model.call_chain)
                  if hop.file == file]
    if not candidates:
        return None
    distance, index = min(candidates)
    return index if distance <= MAX_ATTRIBUTION_DISTANCE else None


def _was_introspected(table: Any) -> bool:
    """Did get_schema actually read this table?

    The model supplies the table name and `columns_read`; everything else on a
    TableInfo comes from `information_schema`. All of it absent means the
    lookup never happened.
    """
    return bool(table.all_columns) or table.approx_rows is not None


def _basename(path: str) -> str:
    return path.rsplit("/", 1)[-1]


def _one_line(text: str, width: int = 100) -> str:
    collapsed = " ".join(str(text).split())
    return collapsed if len(collapsed) <= width else collapsed[:width - 1] + "…"
