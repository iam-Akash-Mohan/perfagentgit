"""The comprehension artifact, rendered for a person.

Every number in here is read off the artifact. Nothing is recomputed, nothing
is rounded into a nicer story, and there is no sentence a model wrote.
"""

from __future__ import annotations

from ..contracts import EndpointModel

INDENT = "  "


def render_endpoint_model(model: EndpointModel) -> str:
    out: list[str] = []
    out.append(f"ROUTE  {model.route}")
    out.append(f"entry  {model.entry.ref()}"
               + (f"  ({model.entry.symbol})" if model.entry.symbol else ""))
    out.append("")

    out.append(f"CALL CHAIN ({len(model.call_chain)} hops)")
    # A spine, not a nested tree. The artifact records the ORDER methods run
    # in, not how deep each one sits; drawing indentation from list position
    # would invent a nesting depth nobody measured.
    for position, hop in enumerate(model.call_chain, start=1):
        joint = "│" if position < len(model.call_chain) else " "
        out.append(f"{INDENT}{position:>2}  {hop.method}")
        out.append(f"{INDENT}{joint}     {hop.file}:{hop.line}")
        out.append(f"{INDENT}{joint}     {hop.does}")
    if not model.call_chain:
        out.append(f"{INDENT}(none recorded)")
    out.append("")

    per_row = model.per_row_accesses()
    out.append(f"DATA ACCESS ({len(model.data_access)} total, "
               f"{len(per_row)} PER RESULT ROW)")
    for access in model.data_access:
        scope = "PER ROW " if access.executes_per_row else "per req "
        out.append(f"{INDENT}[{scope}] {access.kind:<16} {access.trigger}")
        out.append(f"{INDENT}           {access.file}:{access.line}"
                   + (f"   tables: {', '.join(access.tables)}"
                      if access.tables else ""))
        if access.sql:
            out.append(f"{INDENT}           {_one_line(access.sql)}")
        if access.note:
            out.append(f"{INDENT}           note: {_one_line(access.note)}")
    if not model.data_access:
        out.append(f"{INDENT}(none recorded)")
    out.append("")

    out.append(f"TABLES ({len(model.tables)})")
    for table in model.tables:
        rows = (f"~{table.approx_rows:,} rows"
                if table.approx_rows is not None else "row count unknown")
        label = f" [entity {table.entity}]" if table.entity else ""
        out.append(f"{INDENT}{table.table}{label}   {rows}")
        if table.columns_read:
            out.append(f"{INDENT}   columns read by this endpoint: "
                       f"{', '.join(table.columns_read)} "
                       f"(of {len(table.all_columns)} columns)")
        if table.existing_indexes:
            out.append(f"{INDENT}   existing indexes:")
            for index in table.existing_indexes:
                unique = "UNIQUE " if index.unique else ""
                card = (f"  cardinality {index.cardinality:,}"
                        if index.cardinality is not None else "")
                out.append(f"{INDENT}     {unique}{index.name}"
                           f"({', '.join(index.columns)}){card}")
        else:
            out.append(f"{INDENT}   existing indexes: NONE")
    if not model.tables:
        out.append(f"{INDENT}(none recorded)")
    out.append("")

    if model.transactions:
        out.append("TRANSACTIONS")
        for transaction in model.transactions:
            flag = "readOnly" if transaction.read_only else "read-write"
            out.append(f"{INDENT}{transaction.method}  {transaction.propagation}"
                       f"  {flag}   ({transaction.citation.ref()})")
        out.append("")

    if model.entities:
        out.append("ENTITIES")
        for entity in model.entities:
            out.append(f"{INDENT}{entity.name} -> table {entity.table}   "
                       f"({entity.citation.ref()})")
            for relation in entity.relations:
                out.append(f"{INDENT}   {relation.field}: {relation.type} "
                           f"{relation.fetch} -> {relation.target}")
        out.append("")

    if model.external_calls:
        out.append("EXTERNAL CALLS")
        for call in model.external_calls:
            out.append(f"{INDENT}{call.method} -> {call.target} [{call.kind}]  "
                       f"({call.citation.ref()})")
        out.append("")

    if model.tests_covering:
        out.append("TESTS COVERING THIS ROUTE")
        out.extend(f"{INDENT}{name}" for name in model.tests_covering)
        out.append("")

    out.append(f"UNCERTAINTIES ({len(model.uncertainties)})")
    for item in model.uncertainties:
        out.append(f"{INDENT}- {item}")
    if not model.uncertainties:
        out.append(f"{INDENT}(the agent declared none — treat that as a "
                   f"finding of its own)")

    if model.prompt_hash:
        out.append("")
        out.append(f"prompt version: {model.prompt_hash}")
    return "\n".join(out)


def _one_line(text: str, width: int = 110) -> str:
    collapsed = " ".join(str(text).split())
    return collapsed if len(collapsed) <= width else collapsed[:width - 1] + "…"
