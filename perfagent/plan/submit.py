"""submit_plan — the terminal tool, and the place every refusal happens.

The model proposes; this handler disposes. It rejects a plan that attacks a
hypothesis nobody asked about, that proposes an index the database already
has, that names a file it may not write, or that smuggles DDL into
`sql_change`. Every rejection comes back as a tool result the model can act on,
so a bad plan costs one turn rather than a run.

Risk tier, `requires_flag`, expected value and the selection are assigned HERE,
after the model has stopped talking. None of the four is in the tool schema.
"""

from __future__ import annotations

from typing import Any, Callable

from pydantic import ValidationError

from ..contracts import (APPROACHES, CandidateFix, EndpointModel, Plan,
                         RankedHypothesis, Rejection, TableInfo)
from ..errors import ToolError
from ..tools import SourceRoot, ToolSpec
from .fixability import covering_index
from .risk import (MAX_SELECTABLE_TIER, TEST_TREE, assess_risk,
                   forbidden_reason, is_additive_index, parse_index,
                   schema_change_blocked, selection_problem)

MIN_CANDIDATES = 2
MAX_CANDIDATES = 4

# A new file is only ever legitimate in one place: the Flyway migration tree,
# where an additive index lands. Everywhere else, a path that does not exist is
# a path the model invented.
MIGRATION_DIR = "src/main/resources/db/migration"

CANDIDATE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "id": {"type": "string",
               "description": "Short id, e.g. 'c1'. Unique within the plan."},
        "targets_hypothesis": {
            "type": "string",
            "description": "The id of the hypothesis this fix attacks. Must be "
                           "one of the ids offered to you.",
        },
        "approach": {
            "type": "string", "enum": list(APPROACHES),
            "description": "The named technique. Pick the one that describes "
                           "what you will actually write.",
        },
        "description": {
            "type": "string",
            "description": "What you will change, concretely enough that "
                           "someone else could write it.",
        },
        "mechanism_changed": {
            "type": "string",
            "description": "The measurable quantity that moves, named in the "
                           "harness's own terms: 'SQL statements per request', "
                           "'rows examined'. This is checked after the patch "
                           "by the MECHANISM_MATCH gate — if the quantity you "
                           "name does not move, the patch is rejected even if "
                           "it is faster.",
        },
        "predicted_effect": {
            "type": "string",
            "description": "The before and after of that quantity, e.g. "
                           "'queries per request drop from 6 to 2'.",
        },
        "files_to_touch": {
            "type": "array", "items": {"type": "string"},
            "description": "Repository-relative paths. This becomes the patch "
                           "agent's WRITE ALLOWLIST — a file you leave out "
                           "cannot be edited later. Files must already exist, "
                           f"except a new migration under {MIGRATION_DIR}/.",
        },
        "sql_change": {
            "type": "string",
            "description": "The exact SQL, if this candidate adds an index. "
                           "One CREATE INDEX statement, nothing else. Omit it "
                           "entirely for a candidate that changes no schema.",
        },
        "what_could_break": {
            "type": "array", "items": {"type": "string"},
            "description": "Honest failure modes. A candidate with nothing "
                           "here has not been thought about.",
        },
        "preserves_response_bytes": {
            "type": "boolean",
            "description": "TRUE only if the response body is byte-for-byte "
                           "identical afterwards — same fields, same values, "
                           "same order, same collection sizes. This is your "
                           "claim; the frozen oracle checks it later, and a "
                           "false claim ends the run.",
        },
    },
    "required": ["id", "targets_hypothesis", "approach", "description",
                 "mechanism_changed", "predicted_effect", "files_to_touch",
                 "what_could_break", "preserves_response_bytes"],
}

SUBMIT_PARAMETERS: dict[str, Any] = {
    "type": "object",
    "properties": {
        "candidates": {
            "type": "array",
            "minItems": MIN_CANDIDATES,
            "maxItems": MAX_CANDIDATES,
            "items": CANDIDATE_SCHEMA,
            "description": (
                f"{MIN_CANDIDATES} to {MAX_CANDIDATES} candidate fixes. At "
                "least one must target the top-ranked hypothesis, and at least "
                "two must use different approaches — one plan stated twice is "
                "not a choice."
            ),
        },
        "reasoning": {
            "type": "string",
            "description": "Why these candidates and not others. Prose; it is "
                           "recorded, not parsed.",
        },
    },
    "required": ["candidates"],
}


def _explain(error: ValidationError) -> str:
    lines: list[str] = []
    for problem in error.errors():
        where = ".".join(str(part) for part in problem["loc"]) or "(root)"
        lines.append(f"  {where}: {problem['msg']}")
    return "\n".join(lines)


def make_submit_plan(
    *,
    root: SourceRoot,
    model: EndpointModel,
    ranked: list[RankedHypothesis],
    considered: list[str],
    sink: dict[str, Any],
    endpoint: str,
    measured_total_ms: float,
    measured_tables: dict[str, TableInfo] | None = None,
    prompt_hash: str | None = None,
) -> Callable[..., str]:
    """Build the handler. `sink` receives the accepted Plan under "plan".

    `measured_tables` is what `get_schema` read from the live database during
    THIS conversation. It takes precedence over the comprehension map, because
    the map may have been built when the database was unreachable — in which
    case its `existing_indexes` lists are empty and an index that exists looks
    missing.
    """
    expected_by_hypothesis = {r.hypothesis_id: r.expected_value_ms for r in ranked}
    top_id = considered[0] if considered else (ranked[0].hypothesis_id if ranked else "")
    tables = {t.table: t for t in model.tables}

    def submit_plan(**payload: Any) -> str:
        raw = payload.get("candidates")
        if not isinstance(raw, list):
            raise ToolError(
                "submit_plan needs a `candidates` array.",
                suggestion=f"Send {MIN_CANDIDATES} to {MAX_CANDIDATES} "
                           f"candidate objects.",
            )
        if not (MIN_CANDIDATES <= len(raw) <= MAX_CANDIDATES):
            raise ToolError(
                f"you submitted {len(raw)} candidate(s); this phase needs "
                f"between {MIN_CANDIDATES} and {MAX_CANDIDATES}. One plan is "
                f"not a choice, and more than {MAX_CANDIDATES} is a list of "
                f"everything you can think of.",
                suggestion="Submit your strongest candidate plus at least one "
                           "genuinely different approach.",
            )

        candidates: list[CandidateFix] = []
        seen_ids: set[str] = set()
        for index, entry in enumerate(raw):
            if not isinstance(entry, dict):
                raise ToolError(f"candidate {index} is not an object.")
            candidates.append(
                _build(entry, index=index,
                       tables={**tables, **(measured_tables or {})},
                       root=root, expected=expected_by_hypothesis,
                       considered=considered, seen_ids=seen_ids)
            )

        _check_set(candidates, top_id=top_id)

        ordered = sorted(
            candidates,
            key=lambda c: (-c.expected_value_ms, c.risk_tier.value, c.id),
        )
        selected_id, reason, rejections = _select(ordered)

        plan = Plan(
            endpoint=endpoint,
            measured_total_ms=measured_total_ms,
            ranked=ranked,
            considered=considered,
            candidates=ordered,
            selected_id=selected_id,
            selection_reason=reason,
            not_selected=rejections,
            prompt_hash=prompt_hash,
            base_commit=model.base_commit,
        )
        sink["plan"] = plan
        sink["reasoning"] = str(payload.get("reasoning") or "")

        chosen = plan.selected()
        headline = (
            f"selected {chosen.id} ({chosen.approach}, risk tier "
            f"T{chosen.risk_tier.value}, expected value "
            f"{chosen.expected_value_ms:.1f} ms)"
            if chosen is not None else
            "no candidate is selectable — every one was ruled out by code"
        )
        return (
            f"accepted: {len(ordered)} candidate(s) across "
            f"{len({c.targets_hypothesis for c in ordered})} hypothesis(es). "
            f"{headline}. Risk tiers and the selection were assigned by code, "
            f"not by you. You are done — stop now."
        )

    return submit_plan


# ───────────────────────────  one candidate  ─────────────────────────────


def _build(entry: dict[str, Any], *, index: int, tables: dict[str, TableInfo],
           root: SourceRoot, expected: dict[str, float],
           considered: list[str], seen_ids: set[str]) -> CandidateFix:
    """Validate one candidate and stamp the fields code owns."""
    identifier = str(entry.get("id") or f"c{index + 1}")
    target = str(entry.get("targets_hypothesis") or "")
    approach = str(entry.get("approach") or "")
    description = str(entry.get("description") or "")
    sql_change = entry.get("sql_change") or None
    files = [str(f) for f in (entry.get("files_to_touch") or [])]

    if identifier in seen_ids:
        raise ToolError(f"two candidates share the id {identifier!r}. "
                        "Ids must be unique within a plan.")
    seen_ids.add(identifier)

    if target not in considered:
        raise ToolError(
            f"candidate {identifier} targets hypothesis {target!r}, which is "
            f"not one of the hypotheses put to you. Allowed ids: "
            f"{', '.join(considered)}.",
            suggestion="Target the ranked hypotheses you were given. The "
                       "others were ruled out by fixability, and the reason "
                       "is in the brief.",
        )

    forbidden = forbidden_reason(approach=approach, description=description,
                                 sql_change=sql_change, files_to_touch=files)
    if forbidden is not None:
        raise ToolError(
            f"candidate {identifier} is refused at plan time because "
            f"{forbidden}",
            suggestion="Replace it with a candidate that changes application "
                       "code or adds an index, and submit again.",
        )

    _check_files(identifier, files, root)
    if sql_change:
        _check_index(identifier, str(sql_change), tables)

    tier, requires_flag, _why = assess_risk(
        approach=approach, description=description,
        predicted_effect=str(entry.get("predicted_effect") or ""),
    )
    # Checked here, not at measure time: a candidate that changes the schema
    # while the arms share one database could never have been verified, and
    # the patch agent should not spend a build iteration finding that out.
    # Not deleted from the plan either way — "we could not verify this here"
    # is a finding a reviewer should see.
    not_selectable_because = schema_change_blocked(
        approach=approach, sql_change=sql_change, files_to_touch=files,
        repo=root.path,
    )

    try:
        return CandidateFix.model_validate({
            **entry,
            "id": identifier,
            "sql_change": sql_change,
            "files_to_touch": files,
            "risk_tier": tier,
            "requires_flag": requires_flag,
            "expected_value_ms": expected.get(target, 0.0),
            "not_selectable_because": not_selectable_because,
        })
    except ValidationError as exc:
        raise ToolError(
            f"candidate {identifier} did not match the required schema:\n"
            + _explain(exc),
            suggestion="Fix exactly the fields listed and call submit_plan "
                       "again.",
        ) from exc


def _check_files(identifier: str, files: list[str], root: SourceRoot) -> None:
    if not files:
        raise ToolError(
            f"candidate {identifier} lists no files_to_touch. That list is the "
            f"patch agent's write allowlist; an empty one means the fix cannot "
            f"be written."
        )
    for path in files:
        normalised = path.replace("\\", "/")
        if normalised.startswith(TEST_TREE):
            raise ToolError(
                f"candidate {identifier} wants to write {path}. Test files are "
                f"immutable — {TEST_TREE}/** is write-denied."
            )
        # Containment, resolved before checking (a symlink inside the repo can
        # point anywhere). GuardrailError is a ToolError, so the loop reports
        # it as a guardrail rather than as a mistake.
        target = root.resolve(normalised)
        if not target.exists() and MIGRATION_DIR not in normalised:
            raise ToolError(
                f"candidate {identifier} lists {path}, which does not exist in "
                f"the repository. Name files you have read. The only new file "
                f"a plan may create is a migration under {MIGRATION_DIR}/.",
                suggestion="Use read_file or search_code to find the real path.",
            )


def _check_index(identifier: str, sql: str,
                 tables: dict[str, TableInfo]) -> None:
    """An index must be additive, on a table on the path, and not already there."""
    if not is_additive_index(sql):
        raise ToolError(
            f"candidate {identifier} carries a sql_change that is not a single "
            f"additive CREATE INDEX statement: {sql.strip()[:160]!r}."
        )
    parsed = parse_index(sql)
    if parsed is None:
        raise ToolError(
            f"candidate {identifier}: could not read a table and column list "
            f"out of {sql.strip()[:160]!r}. Write it as "
            f"CREATE INDEX idx_name ON table (col_a, col_b);"
        )
    _name, table_name, columns = parsed

    table = tables.get(table_name)
    if table is None:
        raise ToolError(
            f"candidate {identifier} indexes {table_name!r}, which is not a "
            f"table on this endpoint's path. Tables on the path: "
            f"{', '.join(sorted(tables)) or 'none recorded'}."
        )

    # No measured index list means nobody knows what is already there, and
    # "this index does not exist" is a claim about the database. A plan does
    # not get to assume it (invariant 4). The map's list is empty whenever the
    # comprehension phase ran without a reachable database — which is how a
    # duplicate index would otherwise sail through.
    if not table.existing_indexes:
        raise ToolError(
            f"candidate {identifier} proposes an index on {table_name}, but no "
            f"index list has been measured for that table, so there is no way "
            f"to tell whether it already exists.",
            suggestion=f"Call get_schema(['{table_name}']) first and read the "
                       f"existing indexes, then submit again.",
        )

    existing = covering_index(table, columns)
    if existing is not None:
        raise ToolError(
            f"candidate {identifier} proposes an index on {table_name}"
            f"({', '.join(columns)}), but {existing} already covers those "
            f"columns as a prefix. Adding it again costs write throughput and "
            f"buys nothing.",
            suggestion="Either extend the index with a further column that is "
                       "actually filtered on, or propose a different fix.",
        )


# ────────────────────────────  the whole set  ────────────────────────────


def _check_set(candidates: list[CandidateFix], *, top_id: str) -> None:
    if top_id and not any(c.targets_hypothesis == top_id for c in candidates):
        raise ToolError(
            f"no candidate targets {top_id}, the hypothesis with the highest "
            f"expected value. The best-value bottleneck has to be attacked "
            f"before the alternatives are worth reading.",
            suggestion=f"Add a candidate with targets_hypothesis='{top_id}'.",
        )
    approaches = {c.approach for c in candidates}
    if len(approaches) < 2:
        raise ToolError(
            f"every candidate uses the same approach ({approaches.pop()}). "
            f"Two spellings of one idea are not alternatives.",
            suggestion="Add a candidate that attacks the problem a different "
                       "way — a different mechanism, not a different wording.",
        )


def _select(ordered: list[CandidateFix]) -> tuple[str | None, str, list[Rejection]]:
    """Highest expected value, risk tier <= T2, response bytes preserved.

    `ordered` is already sorted by expected value, so the first candidate with
    no selection problem wins. Everything else is recorded with the reason it
    lost — a candidate that was merely out-scored says so, which is what makes
    the alternatives section of the PR body worth reading.
    """
    selected = next((c for c in ordered if selection_problem(c) is None), None)

    rejections: list[Rejection] = []
    for candidate in ordered:
        if selected is not None and candidate.id == selected.id:
            continue
        problem = selection_problem(candidate)
        if problem is None and selected is not None:
            problem = (f"a higher expected value was available: {selected.id} "
                       f"scores {selected.expected_value_ms:.1f} ms against "
                       f"this candidate's {candidate.expected_value_ms:.1f} ms")
        rejections.append(Rejection(candidate_id=candidate.id,
                                    reason=problem or "not selected"))

    if selected is None:
        return (None,
                "no candidate cleared the selection rule: every one was either "
                f"above risk tier T{MAX_SELECTABLE_TIER.value} or did not "
                "claim to preserve the response bytes",
                rejections)

    return (selected.id,
            f"highest expected value ({selected.expected_value_ms:.1f} ms) "
            f"among candidates at risk tier T{MAX_SELECTABLE_TIER.value} or "
            f"below that preserve the response bytes; approach "
            f"{selected.approach}, tier T{selected.risk_tier.value}",
            rejections)


def submit_plan_tools(**kwargs: Any) -> list[ToolSpec]:
    return [ToolSpec(
        name="submit_plan",
        description=(
            "Submit your candidate fixes. This is the ONLY way to finish, and "
            "it can REJECT you: candidates that target the wrong hypothesis, "
            "name a file that does not exist, propose an index that already "
            "exists, or touch anything forbidden are refused with the reason. "
            "Fix exactly what the error names and submit again."
        ),
        parameters=SUBMIT_PARAMETERS,
        handler=make_submit_plan(**kwargs),
        terminal=True,
    )]
