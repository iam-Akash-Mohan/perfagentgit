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

from ..contracts import (APPROACHES, MECHANISMS, CandidateFix, EndpointModel,
                         Plan, RankedHypothesis, Rejection, SqlAudit,
                         SqlDigestStat, TableInfo)
from ..errors import ToolError
from ..gate.rules import INSTRUMENTED_MECHANISMS
from ..tools import SourceRoot, ToolSpec
from ..tools.db import DbToolbox, is_select
from .fixability import covering_index
from .risk import (MAX_SELECTABLE_TIER, SQL_REWRITE_APPROACHES, TEST_TREE,
                   assess_risk, audit_proof_for, derive_scope,
                   forbidden_reason, is_additive_index, parse_index,
                   schema_change_blocked, scope_mismatch_reason,
                   selection_problem, sql_verification_failure,
                   sql_verification_needed, verify_sql_candidate)

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
        "mechanism": {
            "type": "string", "enum": list(MECHANISMS),
            "description": "Which measurable quantity this fix moves. Only "
                           "'sql_count' and 'rows_examined' are instrumented "
                           "— a candidate stating anything else is ruled out "
                           "at plan time, before a patch is written, because "
                           "MECHANISM_MATCH could never confirm it.",
        },
        "mechanism_changed": {
            "type": "string",
            "description": "The same quantity, in words: 'SQL statements per "
                           "request', 'rows examined'. This is checked after "
                           "the patch by the MECHANISM_MATCH gate — if the "
                           "quantity you named in `mechanism` does not move, "
                           "the patch is rejected even if it is faster.",
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
                           "entirely for a candidate that changes no schema. "
                           "NOT for a rewritten query — use sql_before / "
                           "sql_after for that.",
        },
        "sql_before": {
            "type": "string",
            "description": "The query that runs TODAY, as one runnable "
                           "SELECT with %s placeholders — the same "
                           "convention as the explain tool. Required, "
                           "together with sql_after, for any candidate whose "
                           "approach rewrites a query (fetch_join, "
                           "entity_graph, batch_query, dto_projection, "
                           "query_rewrite) and whose mechanism is "
                           "instrumented: this is checked with EXPLAIN "
                           "ANALYZE before the candidate can be selected.",
        },
        "sql_after": {
            "type": "string",
            "description": "The query this candidate replaces sql_before "
                           "with, same %s convention. A claim that rows "
                           "examined drops is measured here, not assumed.",
        },
        "sql_params": {
            "type": "array", "items": {"type": "string"},
            "description": "Values for the %s placeholders shared by "
                           "sql_before and sql_after, in order.",
        },
        "baseline_digest": {
            "type": "string",
            "description": "Required when mechanism is 'sql_count'. The "
                           "exact digest hash (not the SQL text) of ONE "
                           "statement from the brief's 'BASELINE: WHAT "
                           "ACTUALLY RAN' section — the statement whose "
                           "call count this candidate claims to reduce. "
                           "Checked against that same baseline evidence, not "
                           "taken on your word.",
        },
        "baseline_calls": {
            "type": "integer",
            "description": "Required alongside baseline_digest. That "
                           "digest's own 'calls' figure from the brief, "
                           "copied — what you are claiming this statement "
                           "runs TODAY, before your fix.",
        },
        "audit_digest": {
            "type": "string",
            "description": "For approach='add_index' ONLY: the digest hash "
                           "of a MISSING_INDEX statement from the brief's "
                           "SQL AUDIT section, whose own measured "
                           "index_candidate names the SAME table and SAME "
                           "columns as sql_change. A citation that matches "
                           "attaches the audit's own already-measured proof "
                           "(EXPLAIN plan before/after, rows examined, an "
                           "interleaved benchmark) and lets this candidate "
                           "be selected on that statement-level evidence — "
                           "otherwise an add_index candidate cannot be "
                           "selected at all, because blue and green share "
                           "one database and an endpoint-level A/B would be "
                           "comparing two arms that both have the index. "
                           "Omit for any other approach.",
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
                 "mechanism", "mechanism_changed", "predicted_effect",
                 "files_to_touch", "what_could_break",
                 "preserves_response_bytes"],
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
    mysql: Any = None,
    baseline_digests: list[SqlDigestStat] | None = None,
    sql_audit: SqlAudit | None = None,
) -> Callable[..., str]:
    """Build the handler. `sink` receives the accepted Plan under "plan".

    `measured_tables` is what `get_schema` read from the live database during
    THIS conversation. It takes precedence over the comprehension map, because
    the map may have been built when the database was unreachable — in which
    case its `existing_indexes` lists are empty and an index that exists looks
    missing.

    `baseline_digests` is this endpoint's own measured SQL digests, the same
    ones shown in the brief's "BASELINE: WHAT ACTUALLY RAN" section — what a
    `sql_count` candidate's `baseline_digest`/`baseline_calls` claim is
    checked against, rather than trusted. UNB-04 is why: a candidate claimed
    to eliminate "the extra SELECT that loads the Invoice entity" when no
    such statement existed any more (the endpoint had already been reduced to
    one query), and nothing before this checked that the statement being
    reduced ever actually ran.

    `sql_audit` is this run's own sql_audit.json, read for the SAME reason
    `baseline_digests` is: an `add_index` candidate's `audit_digest`
    citation is checked against it, not trusted — see
    `plan/risk.py::audit_proof_for`.

    `mysql` is this run's own target, used ONLY to independently verify a
    SQL-rewrite candidate's `sql_before`/`sql_after` claim with EXPLAIN
    ANALYZE — see `plan/risk.py::verify_sql_candidate`. Built lazily, once,
    on the first candidate that actually needs it, not at every call: most
    submissions never touch it, and a plan with no SQL-rewrite candidate (the
    two instrumentation probes, a Java-side call-structure fix) should not
    pay for a connection it has no use for.
    """
    expected_by_hypothesis = {r.hypothesis_id: r.expected_value_ms for r in ranked}
    attributed_by_hypothesis = {r.hypothesis_id: r.attributed_ms for r in ranked}
    top_id = considered[0] if considered else (ranked[0].hypothesis_id if ranked else "")
    tables = {t.table: t for t in model.tables}

    baseline_by_digest = {d.digest: d for d in (baseline_digests or [])}

    box_cache: dict[str, DbToolbox | None] = {}

    def get_box() -> DbToolbox | None:
        if "box" not in box_cache:
            box_cache["box"] = None
            if mysql is not None:
                try:
                    # No sweep: this box only ever runs EXPLAIN ANALYZE, it
                    # creates no index, so there is nothing of its own to
                    # clean up after — see plan_delta.
                    box_cache["box"] = DbToolbox(
                        mysql, [t.table for t in model.tables],
                        sweep_on_start=False)
                except Exception:  # noqa: BLE001 — unverifiable, not fatal
                    box_cache["box"] = None
        return box_cache["box"]

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
                       root=root, model=model, expected=expected_by_hypothesis,
                       attributed=attributed_by_hypothesis,
                       considered=considered, seen_ids=seen_ids,
                       get_box=get_box, baseline_by_digest=baseline_by_digest,
                       sql_audit=sql_audit)
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


def _normalized_sql(text: str) -> str:
    """Collapse whitespace and case for an identity comparison only — this is
    not the SQL-aware comparison `is_select`/EXPLAIN do, just enough to tell
    a genuine rewrite from the same statement retyped with different
    spacing."""
    return " ".join(text.split()).lower()


# How far a candidate's claimed call count may sit from what baseline
# actually measured before "approximately the count claimed" stops being
# true. Loose on purpose: the model is reading a number off a brief, not
# re-deriving it, and a corpus re-run can shift totals a little even for an
# unchanged statement.
_BASELINE_CALLS_TOLERANCE = 0.25


def _baseline_citation_failure(
    *, identifier: str, mechanism: str | None, baseline_digest: str | None,
    baseline_calls: Any, baseline_by_digest: dict[str, SqlDigestStat],
) -> str | None:
    """Why a `sql_count` candidate's baseline citation doesn't hold up, or
    None. Only `sql_count` claims are checked — a `rows_examined` claim has
    its own, stronger check in `sql_verification_failure` (EXPLAIN ANALYZE,
    not a citation)."""
    if mechanism != "sql_count":
        return None
    if not baseline_digest or baseline_calls is None:
        return (
            f"candidate {identifier} has mechanism 'sql_count' but names no "
            f"baseline_digest/baseline_calls — a claim about how many times "
            f"a statement runs must cite the statement it is about, from "
            f"this endpoint's own measured baseline, not assert a count with "
            f"nothing behind it."
        )
    cited = baseline_by_digest.get(str(baseline_digest))
    if cited is None:
        available = ", ".join(sorted(baseline_by_digest)) or "none recorded"
        return (
            f"candidate {identifier} names baseline_digest "
            f"{baseline_digest!r}, which does not appear in this "
            f"endpoint's own measured baseline — a statement that was never "
            f"recorded running cannot be claimed to run too often. Digests "
            f"actually measured: {available}."
        )
    try:
        claimed = int(baseline_calls)
    except (TypeError, ValueError):
        return (
            f"candidate {identifier}'s baseline_calls ({baseline_calls!r}) "
            f"is not an integer."
        )
    actual = cited.calls
    tolerance = max(1, round(actual * _BASELINE_CALLS_TOLERANCE))
    if abs(claimed - actual) > tolerance:
        return (
            f"candidate {identifier} claims baseline_digest "
            f"{baseline_digest!r} runs {claimed} time(s), but baseline "
            f"measured {actual} call(s) for it — not even approximately "
            f"the count claimed."
        )
    return None


def _build(entry: dict[str, Any], *, index: int, tables: dict[str, TableInfo],
           root: SourceRoot, model: EndpointModel, expected: dict[str, float],
           attributed: dict[str, float], considered: list[str],
           seen_ids: set[str], get_box: Callable[[], DbToolbox | None],
           baseline_by_digest: dict[str, SqlDigestStat],
           sql_audit: SqlAudit | None = None,
           ) -> CandidateFix:
    """Validate one candidate and stamp the fields code owns."""
    identifier = str(entry.get("id") or f"c{index + 1}")
    target = str(entry.get("targets_hypothesis") or "")
    approach = str(entry.get("approach") or "")
    description = str(entry.get("description") or "")
    mechanism_changed = str(entry.get("mechanism_changed") or "")
    sql_change = entry.get("sql_change") or None
    sql_before = entry.get("sql_before") or None
    sql_after = entry.get("sql_after") or None
    sql_params = [str(p) for p in (entry.get("sql_params") or [])]
    files = [str(f) for f in (entry.get("files_to_touch") or [])]
    baseline_digest = entry.get("baseline_digest") or None
    baseline_calls = entry.get("baseline_calls")

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

    mechanism = entry.get("mechanism")
    if mechanism not in MECHANISMS:
        raise ToolError(
            f"candidate {identifier} must state `mechanism` as one of "
            f"{', '.join(MECHANISMS)}. Got {mechanism!r}.",
            suggestion="Set mechanism to the measurable quantity this fix "
                       "actually moves, then submit again.",
        )
    # A valid-but-unmeasured mechanism is not a schema violation — it is a
    # planning error: MECHANISM_MATCH can never confirm it, so this candidate
    # cannot pass the gate no matter what the patch does. Recorded as
    # not_selectable_because, below, rather than letting it through to spend
    # a patch, a verify and a measurement finding that out.
    mechanism_unmeasured = None if mechanism in INSTRUMENTED_MECHANISMS else (
        f"its mechanism is {mechanism!r}, which this harness does not "
        f"instrument — only {', '.join(sorted(INSTRUMENTED_MECHANISMS))} "
        f"carry before/after values, so MECHANISM_MATCH could never confirm "
        f"this candidate regardless of what the patch does."
    )

    # A query_rewrite, fetch_join, entity_graph, batch_query or
    # dto_projection candidate IS a different SQL statement — its claim is
    # checkable with EXPLAIN ANALYZE before anything is patched, and BATCH-01
    # is what happens when that check is skipped: two candidates targeting
    # one hypothesis score identically (expected_value_ms is attributed_ms x
    # the HYPOTHESIS's fixability, not anything about the candidate), so
    # selection fell through to candidate id and shipped a correlated
    # subquery MySQL re-executes once per outer row. Required unconditionally
    # — not only when a live database happens to be reachable right now —
    # the same way files_to_touch or what_could_break are required: the
    # ARTIFACT must carry this, whether or not code can also verify it today.
    verification = None
    no_op_rewrite = None
    if sql_verification_needed(approach=approach, mechanism=mechanism):
        if not sql_before or not sql_after:
            raise ToolError(
                f"candidate {identifier} proposes {approach!r} with "
                f"mechanism {mechanism!r}; EXPLAIN ANALYZE needs a statement "
                f"on each side to check that claim before anything is "
                f"patched.",
                suggestion="Supply sql_before (what runs today) and "
                           "sql_after (what you will replace it with) as "
                           "runnable SELECTs with %s placeholders, then "
                           "submit again.",
            )
        for label, text in (("sql_before", sql_before), ("sql_after", sql_after)):
            if not is_select(str(text)):
                raise ToolError(
                    f"candidate {identifier}'s {label} is not a single "
                    f"SELECT statement: {str(text).strip()[:160]!r}.",
                    suggestion="Pass the literal query EXPLAIN ANALYZE can "
                               "run — no DDL, no DML, one statement.",
                )
        # UNB-04: a candidate claimed to remove "the extra SELECT that loads
        # the Invoice entity" by rewriting a query into ITSELF, byte for
        # byte once whitespace is collapsed. EXPLAIN ANALYZE "verified" it
        # fine -- the same statement plans identically to itself -- which is
        # exactly why this is checked before verification is trusted, not
        # instead of it: a rewrite that does not change the statement cannot
        # change anything it claims to.
        no_op_rewrite = (
            f"candidate {identifier}'s sql_before and sql_after are the same "
            f"statement once whitespace is collapsed — a rewrite that does "
            f"not change the statement cannot reduce its call count or its "
            f"rows examined, whatever the description claims."
            if _normalized_sql(str(sql_before)) == _normalized_sql(str(sql_after))
            else None
        )
        verification = verify_sql_candidate(
            get_box(), sql_before=str(sql_before), sql_after=str(sql_after),
            params=sql_params or None, mechanism=mechanism,
        )

    # files_to_touch is the model's OWN declared scope; it is routinely
    # narrower than the approach it describes — a query_rewrite that changes
    # a repository method's signature obviously touches whatever calls it
    # (COL-06). Expand it to the rest of THIS endpoint's own traced call
    # path before anything downstream judges whether the scope fits the
    # approach, so a legitimate caller is added here rather than discovered
    # by the patch agent after a build iteration it can only abandon.
    files, scope_reasons = derive_scope(files, model)

    forbidden = forbidden_reason(approach=approach, description=description,
                                 sql_change=sql_change, files_to_touch=files)
    if forbidden is not None:
        raise ToolError(
            f"candidate {identifier} is refused at plan time because "
            f"{forbidden}",
            suggestion="Replace it with a candidate that changes application "
                       "code or adds an index, and submit again.",
        )

    # Checked AFTER the expansion above, and no longer a hard rejection: a
    # caller added by derive_scope can rescue a candidate whose only problem
    # was a missing file on its own call path, but it cannot invent an
    # entity file for an approach the call path never visits (COL-04). When
    # the mismatch survives the expansion, the approach genuinely cannot be
    # realised within this candidate's own scope — that is a planning
    # failure, recorded as not_selectable_because rather than handed to the
    # patch agent as a dead end.
    scope_mismatch = scope_mismatch_reason(
        description=description, mechanism_changed=mechanism_changed,
        sql_change=sql_change, files_to_touch=files,
    )

    _check_files(identifier, files, root)
    index_table_columns = (
        _check_index(identifier, str(sql_change), tables) if sql_change else None
    )

    # An add_index candidate that CITES a matching SQL-audit finding takes
    # the schema-candidate gate path instead of schema_change_blocked's
    # unconditional refusal -- see audit_proof_for and CandidateFix.
    # schema_proof. audit_digest not set at all is not an attempt to claim
    # this path, so it is not an error; it falls through to
    # schema_change_blocked exactly as before this existed.
    audit_digest = entry.get("audit_digest") or None
    schema_proof: Any = None
    audit_citation_problem: str | None = None
    if audit_digest is not None:
        if index_table_columns is None:
            audit_citation_problem = (
                f"candidate {identifier} sets audit_digest but has no "
                f"sql_change naming a table and columns to look up a "
                f"matching SQL-audit proof for."
            )
        else:
            schema_proof, audit_citation_problem = audit_proof_for(
                table=index_table_columns[0], columns=index_table_columns[1],
                audit_digest=str(audit_digest), sql_audit=sql_audit,
            )

    tier, requires_flag, _why = assess_risk(
        approach=approach, description=description,
        predicted_effect=str(entry.get("predicted_effect") or ""),
    )
    # A sql_count candidate is a claim about how many times some statement
    # runs TODAY -- a claim that is checkable against this endpoint's own
    # measured baseline, the same way a rows_examined claim is checkable
    # with EXPLAIN ANALYZE. UNB-04 is why this exists: the candidate named a
    # statement ("the extra SELECT that loads the Invoice entity") that had
    # already been designed out, and nothing caught that it had never run at
    # all. Soft, like every other evidence check here -- a candidate that
    # cannot cite real evidence is unsupported, not malformed.
    baseline_unsupported = _baseline_citation_failure(
        identifier=identifier, mechanism=mechanism,
        baseline_digest=baseline_digest, baseline_calls=baseline_calls,
        baseline_by_digest=baseline_by_digest,
    )

    # Checked here, not at measure time: a candidate that changes the schema
    # while the arms share one database could never have been verified, and
    # the patch agent should not spend a build iteration finding that out.
    # Not deleted from the plan either way — "we could not verify this here"
    # is a finding a reviewer should see. Skipped entirely when schema_proof
    # resolved: that candidate is proven at the statement level instead, and
    # needs no per-arm isolation because it never compares the two arms.
    schema_problem = (
        audit_citation_problem if schema_proof is None else None
    ) or (
        schema_change_blocked(
            approach=approach, sql_change=sql_change, files_to_touch=files,
            repo=root.path,
        ) if schema_proof is None else None
    )
    not_selectable_because = (
        schema_problem or scope_mismatch or mechanism_unmeasured
        or no_op_rewrite or baseline_unsupported
        or (sql_verification_failure(verification) if verification else None)
    )

    # The hypothesis-level expected value is the only number that exists
    # before a candidate is measured. Once EXPLAIN ANALYZE has actually run
    # both sides of a rows_examined candidate, that measurement is strictly
    # better evidence than the hypothesis's fixability heuristic (a proxy for
    # "will a fix exist", not "does THIS one work") — see
    # verify_sql_candidate for why sql_count stays on the hypothesis-level
    # number instead.
    expected_value_ms = expected.get(target, 0.0)
    if verification is not None and verification.verified and verification.scored:
        fraction_reduced = max(0.0, min(1.0, -verification.rows_delta_pct / 100.0))
        expected_value_ms = round(
            attributed.get(target, 0.0) * fraction_reduced, 3)

    try:
        return CandidateFix.model_validate({
            **entry,
            "id": identifier,
            "sql_change": sql_change,
            "sql_before": sql_before,
            "sql_after": sql_after,
            "sql_params": sql_params,
            "sql_verification": verification,
            "files_to_touch": files,
            "risk_tier": tier,
            "requires_flag": requires_flag,
            "expected_value_ms": expected_value_ms,
            "not_selectable_because": not_selectable_because,
            "schema_proof": schema_proof,
            "scope_reasons": scope_reasons,
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
                 tables: dict[str, TableInfo]) -> tuple[str, list[str]]:
    """An index must be additive, on a table on the path, and not already
    there. Returns (table, columns) so the caller can look up whether an
    audit_digest citation names a matching statement-level proof."""
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

    return table_name, columns


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


def _coin_flip_tie(a: CandidateFix, b: CandidateFix) -> bool:
    """Same expected value AND same risk tier — the sort has nothing left to
    decide on but lexical candidate id, a property of which one the model
    happened to write first, not a judgement.

    expected_value_ms is attributed_ms x the HYPOTHESIS's fixability (or, for
    a scored SQL-rewrite candidate, attributed_ms x its OWN measured rows
    delta) — either way, two candidates can land on the identical number.
    BATCH-01 is what happens when that silent tiebreak picks the worse one
    and nobody can see it was a coin flip.
    """
    return (a.expected_value_ms == b.expected_value_ms
            and a.risk_tier == b.risk_tier)


def _select(ordered: list[CandidateFix]) -> tuple[str | None, str, list[Rejection]]:
    """Highest expected value, risk tier <= T2, response bytes preserved.

    `ordered` is already sorted by expected value, so the first candidate with
    no selection problem wins. Everything else is recorded with the reason it
    lost — a candidate that was merely out-scored says so, which is what makes
    the alternatives section of the PR body worth reading. A candidate whose
    expected value was NOT lower must not be told it was: that claim is false
    whenever the two numbers are equal, whether the sort then fell back to the
    (real, reasoned) lower-risk-tier preference or, with that tied too, to
    (arbitrary) candidate id — see `_coin_flip_tie`.
    """
    selected = next((c for c in ordered if selection_problem(c) is None), None)

    rejections: list[Rejection] = []
    coin_flips: list[str] = []
    for candidate in ordered:
        if selected is not None and candidate.id == selected.id:
            continue
        problem = selection_problem(candidate)
        if problem is None and selected is not None:
            if candidate.expected_value_ms != selected.expected_value_ms:
                problem = (
                    f"a higher expected value was available: {selected.id} "
                    f"scores {selected.expected_value_ms:.1f} ms against "
                    f"this candidate's {candidate.expected_value_ms:.1f} ms"
                )
            elif _coin_flip_tie(candidate, selected):
                coin_flips.append(candidate.id)
                problem = (
                    f"tied with {selected.id} at expected value "
                    f"{selected.expected_value_ms:.1f} ms and risk tier "
                    f"T{selected.risk_tier.value} — code found no measurable "
                    f"difference between them; {selected.id} was chosen "
                    f"only because it sorted first by candidate id, not "
                    f"because it scored higher."
                )
            else:
                problem = (
                    f"tied with {selected.id} at expected value "
                    f"{selected.expected_value_ms:.1f} ms; {selected.id} was "
                    f"preferred for its lower risk tier "
                    f"(T{selected.risk_tier.value} vs "
                    f"T{candidate.risk_tier.value}), not a higher score."
                )
        rejections.append(Rejection(candidate_id=candidate.id,
                                    reason=problem or "not selected"))

    if selected is None:
        return (None,
                "no candidate cleared the selection rule: every one was either "
                f"above risk tier T{MAX_SELECTABLE_TIER.value} or did not "
                "claim to preserve the response bytes",
                rejections)

    reason = (
        f"highest expected value ({selected.expected_value_ms:.1f} ms) "
        f"among candidates at risk tier T{MAX_SELECTABLE_TIER.value} or "
        f"below that preserve the response bytes; approach "
        f"{selected.approach}, tier T{selected.risk_tier.value}"
    )
    if coin_flips:
        reason += (
            f". TIE: {', '.join(coin_flips)} scored identically and "
            f"lost only to candidate-id order, not to a measured difference "
            f"— a human should read both before merging."
        )
    return (selected.id, reason, rejections)


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
