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

import subprocess
from typing import Any, Callable

from pydantic import ValidationError

from ..contracts import EndpointModel, TableInfo
from ..errors import ToolError
from . import SourceRoot, ToolSpec

CITATION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"file": {"type": "string"},
                   "line": {"type": "integer"},
                   "symbol": {"type": "string"}},
    "required": ["file", "line"],
}

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
        "returns": {
            "type": "string",
            "enum": ["entities", "projection", "scalar"],
            "description": "What this access's OWN query returns, not what "
                           "kind of query it is: 'entities' if it produces a "
                           "managed JPA/Hibernate entity (e.g. 'select e "
                           "from Entity e', a derived query returning the "
                           "entity type, a lazy collection/association) — "
                           "hydration happens. 'projection' if it returns a "
                           "constructor expression or DTO (e.g. 'select new "
                           "some.package.SomeDto(...)', a native query "
                           "mapped straight to a record/interface projection) — "
                           "no entity is ever materialised. 'scalar' if it "
                           "returns a count, sum, or other aggregate — "
                           "nothing to hydrate at all. The query SHAPE "
                           "decides this, not the table or the SQL keyword: "
                           "two jpql_query accesses on the same table can "
                           "have different answers here.",
        },
        "note": {"type": "string"},
    },
    "required": ["trigger", "kind", "file", "line", "executes_per_row", "returns"],
}

ENTITY_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "table": {"type": "string"},
        "citation": CITATION_SCHEMA,
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
        "citation": CITATION_SCHEMA,
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

PER_ROW_WORK_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "description": {
            "type": "string",
            "description": "What the work is, concretely. e.g. \"appends one "
                           "'overdue' string per overdue invoice\".",
        },
        "file": {"type": "string"},
        "line": {"type": "integer"},
        "kind": {"type": "string",
                 "enum": ["string_building", "collection_processing",
                          "object_creation", "computation", "serialization",
                          "external_call", "other"]},
        "scales_with": {
            "type": "string",
            "description": "What drives the count, in the domain's own words: "
                           "'invoices per customer', 'page size', 'line items "
                           "per order'.",
        },
        "bounded": {
            "type": "boolean",
            "description": "TRUE only if something LIMITS how much work this "
                           "does — a page size, a hard cap, a fixed-size "
                           "collection. FALSE if it scales with the data and "
                           "nothing caps it. Unbounded per-row work is "
                           "frequently the single largest cost in an endpoint "
                           "and never shows up in the SQL logs.",
        },
        "bound": {
            "type": "string",
            "description": "REQUIRED when bounded is true: name the limit, e.g. "
                           "'page size, max 100'. Claiming a limit without "
                           "naming it is rejected.",
        },
    },
    "required": ["description", "file", "line", "kind", "scales_with", "bounded"],
}

RESPONSE_SHAPE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "dto": {"type": "string", "description": "The returned type, e.g. CustomerDto."},
        "fields_returned": {
            "type": "array", "items": {"type": "string"},
            "description": "Field names only, one per element.",
        },
        "computed_fields": {
            "type": "array", "items": {"type": "string"},
            "description": "Fields produced by real work rather than by "
                           "returning a stored value.",
        },
        "unbounded_collections": {
            "type": "array", "items": {"type": "string"},
            "description": "Returned collections whose SIZE scales with the "
                           "data. This is the important one: if a returned "
                           "collection grows with the data, its size is part "
                           "of the API contract and cannot be trimmed without "
                           "changing behaviour. Give the FIELD NAME ONLY, "
                           "exactly as it appears in fields_returned — 'tags', "
                           "not 'tags (list of strings)'. A later phase matches "
                           "these against the response body and a description "
                           "will not match anything.",
        },
        "citation": CITATION_SCHEMA,
    },
    "required": [],
}

REQUEST_PARAMETER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "location": {"type": "string", "enum": ["path", "query"]},
        "type": {"type": "string",
                 "enum": ["string", "integer", "number", "boolean", "array"]},
        "required": {
            "type": "boolean",
            "description": "TRUE if the request fails without it — no "
                           "default, no handling of a missing value.",
        },
        "default": {
            "description": "The value Spring supplies when the caller omits "
                           "this parameter. Omit this field entirely when "
                           "there is none — a parameter cannot be both "
                           "required and have a default.",
        },
        "allowed_values": {
            "type": "array",
            "description": "Every value the CODE constrains this to: an enum "
                           "type, a @Pattern, an explicit check that rejects "
                           "anything else. Leave empty when nothing in the "
                           "code constrains it — inventing a constraint is "
                           "worse than reporting none, since a corpus "
                           "generator will trust this list literally.",
        },
        "citation": CITATION_SCHEMA,
    },
    "required": ["name", "location", "type", "required", "citation"],
}

REQUEST_SHAPE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "parameters": {
            "type": "array", "items": REQUEST_PARAMETER_SCHEMA,
            "description": "EVERY path and query parameter the endpoint "
                           "reads, including ones with a default — not only "
                           "the required ones. A corpus generator supplies a "
                           "value for every REQUIRED parameter it can "
                           "determine one for (from allowed_values or "
                           "default) and REFUSES by name for one it cannot — "
                           "so 'required, no default, no allowed_values' is "
                           "itself a useful, honest thing to report, not a "
                           "gap to fill with a guess.",
        },
    },
    "required": [],
}

PAGINATION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "mechanism": {"type": "string",
                      "enum": ["offset", "keyset", "none", "unbounded"]},
        "fires_count_query": {
            "type": "boolean",
            "description": "TRUE if a separate COUNT query runs, as it does "
                           "for any Spring Data method returning Page<T>.",
        },
        "max_page_size": {
            "type": "integer",
            "description": "The cap, if there is one. OMIT the field entirely "
                           "when the page size is uncapped — an uncapped page "
                           "size on an endpoint with unbounded per-row work is "
                           "a compounding problem.",
        },
        "citation": CITATION_SCHEMA,
    },
    "required": ["mechanism"],
}

CACHE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "kind": {"type": "string",
                 "enum": ["spring_cacheable", "hibernate_second_level",
                          "query_cache", "manual", "other"]},
        "on": {"type": "string", "description": "The method or entity cached."},
        "cache_name": {"type": "string"},
        "citation": CITATION_SCHEMA,
    },
    "required": ["kind", "on", "citation"],
}

HIDDEN_QUERY_TRIGGER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "kind": {"type": "string",
                 "enum": ["formula", "post_load", "pre_persist",
                          "entity_listener", "hibernate_filter", "audit",
                          "other"]},
        "on": {"type": "string", "description": "The entity or field."},
        "detail": {"type": "string"},
        "citation": CITATION_SCHEMA,
    },
    "required": ["kind", "on", "detail", "citation"],
}

REQUIRED_FIELDS: tuple[str, ...] = (
    "route", "call_chain", "data_access", "per_row_work", "response_shape",
    "request_shape", "uncertainties",
)

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
        "per_row_work": {
            "type": "array", "items": PER_ROW_WORK_SCHEMA,
            "description": "REQUIRED. Work that runs once per RESULT ROW and "
                           "is NOT a database access: building strings, "
                           "iterating collections, creating objects, repeated "
                           "computation. A query per row is one kind of "
                           "per-row cost and often not the expensive one. If "
                           "this endpoint genuinely has none, submit an empty "
                           "array — that is a finding, not an omission.",
        },
        "response_shape": {
            **RESPONSE_SHAPE_SCHEMA,
            "description": "REQUIRED. What the endpoint returns, and which "
                           "returned collections are unbounded in size.",
        },
        "request_shape": {
            **REQUEST_SHAPE_SCHEMA,
            "description": "REQUIRED. Every path and query parameter the "
                           "endpoint reads — the symmetric counterpart of "
                           "response_shape. If the endpoint genuinely takes "
                           "none, submit parameters: [] — that is a finding, "
                           "not an omission.",
        },
        "pagination": {
            **PAGINATION_SCHEMA,
            "description": "How the endpoint pages, if it pages.",
        },
        "caching": {"type": "array", "items": CACHE_SCHEMA,
                    "description": "Caches on the path. Usually empty."},
        "hidden_query_triggers": {
            "type": "array", "items": HIDDEN_QUERY_TRIGGER_SCHEMA,
            "description": "SQL that fires from somewhere other than the call "
                           "site: @Formula, @PostLoad, entity listeners, "
                           "Hibernate filters. Usually empty.",
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
    # per_row_work and response_shape are REQUIRED so that "none found" is a
    # statement the model had to make rather than a field it left out.
    "required": list(REQUIRED_FIELDS),
}

REQUIRED_FIELD_HELP = (
    "`per_row_work`, `response_shape` and `request_shape` are required even "
    "when there is nothing to report: submit `per_row_work: []` to state "
    "that this endpoint does no per-row work outside its queries, or "
    "`request_shape: {\"parameters\": []}` to state it takes none. An empty "
    "array is a finding; a missing field is a blind spot."
)


def git_head(root: SourceRoot) -> str | None:
    """The target repo's commit at analysis time, or None if it is not a repo.

    A dirty worktree is reported as `<sha>+dirty` rather than refused, matching
    what the oracle records: refusing would block legitimate work, and
    returning the clean sha would claim provenance the artifact does not have.

    argv list, never `shell=True`. Invariant 1 governs the AGENT's tool
    surface — this runs in the handler, with no argument the model can reach —
    but harness code has no excuse for shelling out either.

    Duplicated from `oracle/record.py::base_commit` on purpose: importing the
    oracle module into the agent's tool layer would put frozen-recording code
    one import away from a tool schema, and this is twelve lines. A later
    session should hoist one `provenance.py` and have both use it.
    """
    if not (root.path / ".git").exists():
        return None
    try:
        def git(*args: str) -> str:
            done = subprocess.run(["git", "-C", str(root.path), *args],
                                  capture_output=True, text=True, timeout=30)
            if done.returncode != 0:
                raise OSError(done.stderr.strip())
            return done.stdout.strip()

        sha = git("rev-parse", "HEAD")
        dirty = bool(git("status", "--porcelain"))
    except (OSError, subprocess.SubprocessError):
        # Provenance is worth having, not worth failing a phase over.
        return None
    return f"{sha}+dirty" if dirty else sha


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
        from ..understand.validate import (completeness_warnings,
                                           render_problems, validate_citations,
                                           validate_claims)

        # base_commit is provenance, and provenance the model supplies is a
        # claim rather than a fact. Drop whatever arrived and stamp it below.
        claimed_commit = payload.pop("base_commit", None)

        # `required` in the JSON schema is a request, not a guarantee — some
        # providers enforce it, some do not. These two fields have defaults on
        # the Pydantic model, so an omission would validate silently as "none
        # found". Enforce in code, not in a schema the provider may ignore.
        omitted = [name for name in REQUIRED_FIELDS if name not in payload]
        if omitted:
            raise ToolError(
                f"the submission omitted {', '.join(omitted)}. "
                + REQUIRED_FIELD_HELP,
                suggestion="Add the missing field(s) and call "
                           "submit_endpoint_analysis again.",
            )

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
        claims = validate_claims(model)
        if problems or claims:
            raise ToolError(_reject(problems, claims, render_problems))

        model.tables = _merge_tables(model.tables, measured_tables or {})
        model.prompt_hash = prompt_hash
        model.base_commit = git_head(root)

        # A heuristic, never a rejection — it goes where a human will read it.
        model.uncertainties = list(model.uncertainties) + completeness_warnings(model)
        sink["model"] = model

        per_row = len(model.per_row_accesses())
        unbounded = len(model.unbounded_per_row_work())
        params = len(model.request_shape.parameters) if model.request_shape else 0
        note = ""
        if claimed_commit is not None:
            note = (" (the base_commit you sent was discarded; it is read from "
                    "git, not from the submission)")
        return (
            f"accepted: {len(model.call_chain)} call-chain hops, "
            f"{len(model.data_access)} data accesses ({per_row} of them "
            f"per-row), {len(model.per_row_work)} non-query per-row work "
            f"item(s) ({unbounded} UNBOUNDED), {params} request parameter(s), "
            f"{len(model.tables)} table(s), {len(model.uncertainties)} "
            f"uncertainty(ies). Every citation resolved. You are done — stop "
            f"now.{note}"
        )

    return submit_endpoint_analysis


def _reject(problems: list[Any], claims: list[str],
            render: Callable[[Any], str]) -> str:
    """One message carrying every objection, so one retry can fix them all."""
    parts: list[str] = []
    if problems:
        parts.append(render(problems))
    if claims:
        parts.append(f"{len(claims)} claim(s) do not hold up:\n"
                     + "\n".join(claims))
    return "\n\n".join(parts)


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
