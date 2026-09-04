"""submit_endpoint_analysis — the terminal tool, and it can say no.

Renamed from `emit_endpoint_model`. "Emit" implies the call always lands;
"submit" is honest, because this handler rejects roughly as often as it
accepts on the first attempt and the rejection is the mechanism that makes the
artifact trustworthy.

Termination is THIS TOOL SUCCEEDING. Not `end_turn`, not a step count. That is
what guarantees the phase produces a typed artifact or an honest failure,
never prose someone has to parse.
"""

from __future__ import annotations

from typing import Any, Callable

from pydantic import ValidationError

from ..contracts import EndpointModel, TableInfo
from ..errors import ToolError
from . import SourceRoot, ToolSpec

CALL_HOP_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "method": {"type": "string",
                   "description": "e.g. CustomerService.search"},
        "file": {"type": "string",
                 "description": "Path relative to the repository root."},
        "line": {"type": "integer",
                 "description": "The declaration line. Verified against the "
                                "file — a guessed number is rejected."},
        "does": {"type": "string",
                 "description": "What this method actually does, in one line."},
    },
    "required": ["method", "file", "line", "does"],
}

DATA_ACCESS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "trigger": {"type": "string",
                    "description": "What causes it, e.g. 'repo.findByFilter' "
                                   "or 'lazy load of Customer.invoices'"},
        "kind": {"type": "string",
                 "enum": ["jpql_query", "native_query", "derived_query",
                          "lazy_collection", "lazy_association", "criteria",
                          "jdbc_template", "other"]},
        "file": {"type": "string"},
        "line": {"type": "integer"},
        "sql": {"type": "string",
                "description": "The query, if it is visible in the code."},
        "tables": {"type": "array", "items": {"type": "string"},
                   "description": "Physical tables this access reads or writes."},
        "executes_per_row": {
            "type": "boolean",
            "description": "TRUE if this runs once per RESULT ROW (inside a "
                           "loop, a stream, or a .map over results). FALSE if "
                           "it runs once per request. If you cannot tell from "
                           "the code, say so in uncertainties instead of "
                           "guessing.",
        },
        "note": {"type": "string"},
    },
    "required": ["trigger", "kind", "file", "line", "executes_per_row"],
}

ENTITY_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "table": {"type": "string"},
        "citation": {
            "type": "object",
            "properties": {"file": {"type": "string"},
                           "line": {"type": "integer"},
                           "symbol": {"type": "string"}},
            "required": ["file", "line"],
        },
        "relations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "field": {"type": "string"},
                    "type": {"type": "string",
                             "enum": ["OneToMany", "ManyToOne",
                                      "OneToOne", "ManyToMany"]},
                    "fetch": {"type": "string", "enum": ["LAZY", "EAGER"]},
                    "target": {"type": "string"},
                },
                "required": ["field", "type", "fetch", "target"],
            },
        },
    },
    "required": ["name", "table", "citation"],
}

TRANSACTION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "method": {"type": "string"},
        "propagation": {"type": "string",
                        "description": "e.g. REQUIRED. Use REQUIRED if the "
                                       "annotation does not say."},
        "read_only": {"type": "boolean"},
        "citation": {
            "type": "object",
            "properties": {"file": {"type": "string"},
                           "line": {"type": "integer"},
                           "symbol": {"type": "string"}},
            "required": ["file", "line"],
        },
    },
    "required": ["method", "propagation", "citation"],
}

TABLE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "table": {"type": "string", "description": "Physical table name."},
        "entity": {"type": "string", "description": "The JPA entity, if any."},
        "columns_read": {
            "type": "array", "items": {"type": "string"},
            "description": "The columns this endpoint actually reads. An "
                           "entity with thirty columns where three are read is "
                           "a finding; nothing downstream can see it unless "
                           "you record it here.",
        },
    },
    "required": ["table"],
}

SUBMIT_PARAMETERS: dict[str, Any] = {
    "type": "object",
    "properties": {
        "route": {"type": "string",
                  "description": "The endpoint, e.g. 'POST /api/customers/search'."},
        "call_chain": {
            "type": "array", "items": CALL_HOP_SCHEMA,
            "description": "Every method executed, in order, from the "
                           "controller down.",
        },
        "data_access": {
            "type": "array", "items": DATA_ACCESS_SCHEMA,
            "description": "EVERY database access this endpoint can trigger, "
                           "including lazy collection loads and anything "
                           "inside a loop — not only explicit repository "
                           "queries.",
        },
        "entities": {"type": "array", "items": ENTITY_SCHEMA},
        "transactions": {"type": "array", "items": TRANSACTION_SCHEMA},
        "tables": {
            "type": "array", "items": TABLE_SCHEMA,
            "description": "One entry per table on the path. Column types, "
                           "row counts and existing indexes are filled in "
                           "from the database — you only supply the table "
                           "name, the entity and columns_read.",
        },
        "tests_covering": {"type": "array", "items": {"type": "string"},
                           "description": "Test classes or methods that "
                                          "exercise this route."},
        "uncertainties": {
            "type": "array", "items": {"type": "string"},
            "description": "Anything you could not determine from the code "
                           "alone. An honest uncertainty is worth more than a "
                           "confident guess.",
        },
    },
    "required": ["route", "call_chain", "data_access", "uncertainties"],
}


def _explain(error: ValidationError) -> str:
    """Pydantic's own words, one line per problem, with the path it failed at."""
    lines: list[str] = []
    for problem in error.errors():
        where = ".".join(str(part) for part in problem["loc"]) or "(root)"
        lines.append(f"  {where}: {problem['msg']}")
    return "\n".join(lines)


def make_submit_endpoint_analysis(
    root: SourceRoot,
    sink: dict[str, Any],
    *,
    measured_tables: dict[str, TableInfo] | None = None,
    prompt_hash: str | None = None,
) -> Callable[..., str]:
    """Build the handler.

    `sink` receives the accepted model under the key "model". `measured_tables`
    is what get_schema actually read from the database; the model's own table
    entries keep only the judgements it is qualified to make (which table,
    which entity, which columns are read) and every measured number is
    templated in here (invariant 4).
    """

    def submit_endpoint_analysis(**payload: Any) -> str:
        # Imported here rather than at module scope: validate.py lives in the
        # understand package, which imports this module's siblings.
        from ..understand.validate import render_problems, validate_citations

        try:
            model = EndpointModel.model_validate(payload)
        except ValidationError as exc:
            raise ToolError(
                "the submission did not match the required schema:\n"
                + _explain(exc),
                suggestion="Fix exactly the fields listed above and call "
                           "submit_endpoint_analysis again.",
            ) from exc

        problems = validate_citations(model, root)
        if problems:
            raise ToolError(render_problems(problems))

        model.tables = _merge_tables(model.tables, measured_tables or {})
        model.prompt_hash = prompt_hash
        sink["model"] = model

        per_row = len(model.per_row_accesses())
        return (
            f"accepted: {len(model.call_chain)} call-chain hops, "
            f"{len(model.data_access)} data accesses ({per_row} of them "
            f"per-row), {len(model.tables)} table(s), "
            f"{len(model.uncertainties)} uncertainty(ies). Every citation "
            f"resolved. You are done — stop now."
        )

    return submit_endpoint_analysis


def _merge_tables(claimed: list[TableInfo],
                  measured: dict[str, TableInfo]) -> list[TableInfo]:
    """Model judgement + measured facts. Never the model's version of a fact."""
    out: list[TableInfo] = []
    seen: set[str] = set()

    for entry in claimed:
        seen.add(entry.table)
        fact = measured.get(entry.table)
        if fact is None:
            out.append(entry)
            continue
        out.append(TableInfo(
            table=entry.table,
            entity=entry.entity,
            columns_read=entry.columns_read,
            all_columns=fact.all_columns,
            existing_indexes=fact.existing_indexes,
            approx_rows=fact.approx_rows,
            foreign_keys=fact.foreign_keys,
        ))

    # A table the agent looked up but forgot to declare is still evidence.
    for name, fact in measured.items():
        if name not in seen:
            out.append(fact)
    return out


def submit_tools(root: SourceRoot, sink: dict[str, Any], *,
                 measured_tables: dict[str, TableInfo] | None = None,
                 prompt_hash: str | None = None) -> list[ToolSpec]:
    return [ToolSpec(
        name="submit_endpoint_analysis",
        description=(
            "Submit your finished analysis. This is the ONLY way to finish, "
            "and it can REJECT you: every citation is verified against the "
            "filesystem after you call it. If it rejects, fix exactly what the "
            "error names and submit again."
        ),
        parameters=SUBMIT_PARAMETERS,
        handler=make_submit_endpoint_analysis(
            root, sink, measured_tables=measured_tables,
            prompt_hash=prompt_hash),
        terminal=True,
    )]
