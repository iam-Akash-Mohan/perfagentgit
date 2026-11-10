"""Risk tiers and selection. Both are code reading a fixed table.

A model that could set its own risk tier would set it to LOW, and a model that
could pick the winning candidate would pick the one it wrote most confidently.
Neither field is in the tool schema; both are computed here from what the
candidate says it will do.

T4 is not a tier a candidate can be submitted with — it is a rejection at plan
time, before a worktree exists. T3 is allowed into the artifact, flagged, and
cannot be selected.
"""

from __future__ import annotations

import re
from pathlib import Path

from ..contracts import (CandidateFix, EndpointModel, IndexCandidate,
                         RiskTier, SqlAudit, SqlVerification)
from ..gate.rules import MECHANISM_MOVE_PCT
from ..tools.db import DbToolbox

# The tier table, by approach. Everything in T1 either adds something the old
# code path did not have (an index, a fetch plan) or narrows a transaction to
# read-only; nothing in it changes what a query returns.
TIER_BY_APPROACH: dict[str, RiskTier] = {
    "add_index": RiskTier.LOW,
    "entity_graph": RiskTier.LOW,
    "fetch_join": RiskTier.LOW,
    "read_only_transaction": RiskTier.LOW,
    "dto_projection": RiskTier.LOW,
    "batch_query": RiskTier.MEDIUM,
    "query_rewrite": RiskTier.MEDIUM,
    "remove_redundant_call": RiskTier.MEDIUM,
    "move_work_out_of_loop": RiskTier.MEDIUM,
    # An approach the table does not name cannot be certified as low risk.
    # HIGH rather than MEDIUM so `other` is flagged and never auto-selected.
    "other": RiskTier.HIGH,
}

# Approaches whose fix IS a different SQL statement — as opposed to
# remove_redundant_call / move_work_out_of_loop / read_only_transaction,
# which change how many times (or in what transaction) the SAME query runs,
# with no new statement for EXPLAIN to look at. `add_index` is deliberately
# excluded too: its `sql_change` is a CREATE INDEX, not a SELECT, and
# `schema_change_blocked` already disqualifies it before this would run.
SQL_REWRITE_APPROACHES = frozenset({
    "fetch_join", "entity_graph", "batch_query", "dto_projection",
    "query_rewrite",
})

# T3: correct-looking changes whose failure mode is invisible in a diff.
CACHING_WORDS = (
    "cache", "caching", "cacheable", "@cacheable", "ehcache", "caffeine",
    "redis", "memoize", "memoise", "second-level cache",
)
TX_BOUNDARY_WORDS = (
    "propagation", "requires_new", "transaction boundary",
    "transactional boundary", "new transaction", "separate transaction",
    "split the transaction", "commit boundary", "nested transaction",
)

# T4: forbidden outright.
AUTH_WORDS = (
    "spring security", "authentication", "authorization", "@preauthorize",
    "@secured", "authenticat", "principal check",
)
DDL_FORBIDDEN = re.compile(
    r"\b(create\s+table|alter\s+table|drop\s+(table|column|index|view)|"
    r"add\s+column|modify\s+column|change\s+column|rename\s+(table|column)|"
    r"truncate\s+table|create\s+view)\b", re.I,
)
# The one schema change the design permits.
CREATE_INDEX = re.compile(
    r"^\s*create\s+(unique\s+)?index\s+[`\"]?[\w.]+[`\"]?\s+on\s+"
    r"[`\"]?[\w.]+[`\"]?\s*\([^)]+\)\s*;?\s*$", re.I,
)
INDEX_PARTS = re.compile(
    r"create\s+(?:unique\s+)?index\s+[`\"]?(?P<name>[\w.]+)[`\"]?\s+on\s+"
    r"[`\"]?(?P<table>\w+)[`\"]?\s*\((?P<columns>[^)]+)\)", re.I,
)

TEST_TREE = "src/test"


def parse_index(sql: str) -> tuple[str, str, list[str]] | None:
    """(index name, table, columns) from a CREATE INDEX statement, or None."""
    match = INDEX_PARTS.search(sql or "")
    if match is None:
        return None
    columns = [
        part.strip().strip("`\"").split()[0]
        for part in match.group("columns").split(",")
        if part.strip()
    ]
    return match.group("name"), match.group("table"), columns


def is_additive_index(sql: str) -> bool:
    """One CREATE INDEX statement and nothing else.

    Checked with a regex over the whole string rather than by looking for the
    words `create index` somewhere in it: `CREATE INDEX ...; DROP TABLE x;`
    contains both.
    """
    text = (sql or "").strip()
    if not text:
        return False
    if DDL_FORBIDDEN.search(text):
        return False
    return CREATE_INDEX.match(text) is not None


# Techniques JPA/Spring confine to a PARTICULAR kind of file -- not a style
# preference, a fact about where the framework permits the annotation to
# live. c2 (COL-04) described adding @SqlResultSetMapping while scoping the
# patch to one repository interface; that annotation can only go on an
# entity class, so the candidate was unimplementable the moment it was
# written, and the patch agent spent a full green-worktree build and
# baseline measurement discovering that for itself.
_REQUIRES_ENTITY_FILE = (
    "@sqlresultsetmapping", "@constructorresult", "@namednativequery",
)


def _looks_like_entity_file(path: str) -> bool:
    normalised = path.replace("\\", "/").lower()
    name = normalised.rsplit("/", 1)[-1]
    if name.endswith(("repository.java", "controller.java", "service.java",
                      "dto.java", "mapper.java")):
        return False
    return "/domain/" in normalised or "/entity/" in normalised or "/model/" in normalised


def call_path_files(model: EndpointModel) -> list[str]:
    """Every file on the endpoint's traced call path, in call order.

    `call_chain` is recorded by comprehension from the controller down to
    wherever the trace stopped; this is the one list in the whole artifact
    that already says which files call which, so scope is derived from it
    rather than re-guessed from prose.
    """
    seen: list[str] = []
    for hop in model.call_chain:
        if hop.file not in seen:
            seen.append(hop.file)
    return seen


def derive_scope(
    files_to_touch: list[str], model: EndpointModel,
) -> tuple[list[str], dict[str, str]]:
    """Expand files_to_touch to the rest of the endpoint's own call path.

    COL-06: a query_rewrite that changed a repository method's parameter
    type broke the service calling it, and the service was never in scope —
    the candidate had no way to fix what its own change broke. The file it
    needed was not invented; it was one hop up the SAME traced call chain
    the candidate's own target file sits on.

    Only ever walks toward the entry point from the deepest file the
    candidate already names, and only along a path this endpoint itself was
    traced on — never a file unrelated to the candidate's own target.
    Returns the (possibly unchanged) file list and a reason for each file
    added, keyed by path, for `CandidateFix.scope_reasons`.
    """
    path = call_path_files(model)
    on_path = [i for i, f in enumerate(path) if f in files_to_touch]
    out = list(files_to_touch)
    reasons: dict[str, str] = {}
    if on_path:
        deepest = path[max(on_path)]
        for file in path[: max(on_path)]:
            if file not in out:
                out.append(file)
                reasons[file] = (
                    f"on this endpoint's own traced call path to {deepest}, "
                    f"which this candidate already touches — included so a "
                    f"change to {deepest} has its caller in scope"
                )

    # A method this endpoint's own trace shows being called from more than
    # one site: STM-01's h4 (CustomerRepository.findProfile, independently
    # called by StatementService, OutstandingCalculator AND
    # StatementNarrator) is incoherent the same way COL-04's
    # @SqlResultSetMapping candidate was if a plan only lists SOME of those
    # callers — the backward walk above only ever reaches ONE of several
    # sibling call sites, whichever the model happened to name, because
    # siblings are not ancestors of each other in the traced path. A fix
    # that removes the repetition has to touch every site that makes the
    # call, not whichever one is already in scope.
    for method, caller_files in repeated_callers(model).items():
        if any(f in out for f in caller_files):
            for file in caller_files:
                if file not in out:
                    out.append(file)
                    reasons[file] = (
                        f"{method} is called from here too, on this "
                        f"endpoint's own traced call path — a candidate "
                        f"touching one of its callers needs every one of "
                        f"them, or the fix only removes the repetition at "
                        f"the site it happened to name"
                    )
    return out, reasons


def repeated_callers(model: EndpointModel) -> dict[str, list[str]]:
    """method -> the deduplicated files of every site that calls it, for
    every method this endpoint's own traced call path invokes from more
    than one place.

    Read straight off `call_chain`'s own structure: in a DFS-ordered trace,
    the hop immediately before any call IS that call's caller, so no
    parsing of a hypothesis's prose is needed to find "every site a
    repeated call happens from" — it is already there, in the data.
    """
    by_method: dict[str, list[int]] = {}
    for i, hop in enumerate(model.call_chain):
        by_method.setdefault(hop.method, []).append(i)

    callers: dict[str, list[str]] = {}
    for method, indices in by_method.items():
        if len(indices) < 2:
            continue
        files: list[str] = []
        for idx in indices:
            if idx == 0:
                continue
            caller_file = model.call_chain[idx - 1].file
            if caller_file not in files:
                files.append(caller_file)
        if len(files) > 1:
            callers[method] = files
    return callers


def scope_mismatch_reason(*, description: str, mechanism_changed: str,
                          sql_change: str | None,
                          files_to_touch: list[str]) -> str | None:
    """Does this candidate's OWN description name a technique its OWN
    files_to_touch cannot support? None if the scope and the approach agree.

    Checked here, before a worktree exists, so a candidate that could never
    be implemented within its own declared scope is replaced in the same
    plan-submission conversation instead of reaching the patch agent, which
    has no choice at that point but to abandon after spending a build
    iteration finding out.
    """
    text = f"{description} {mechanism_changed} {sql_change or ''}".lower()
    if not any(marker in text for marker in _REQUIRES_ENTITY_FILE):
        return None
    if any(_looks_like_entity_file(f) for f in files_to_touch):
        return None
    return (
        "it describes @SqlResultSetMapping, @ConstructorResult or "
        "@NamedNativeQuery, which JPA only allows on an entity class, but "
        f"files_to_touch names no entity file: {', '.join(files_to_touch)}. "
        "A JPQL constructor expression (`SELECT NEW` the DTO's fully "
        "qualified name) needs none of this and stays inside the "
        "repository interface — prefer it for a DTO projection."
    )


def forbidden_reason(*, approach: str, description: str, sql_change: str | None,
                     files_to_touch: list[str]) -> str | None:
    """Why this candidate is T4, or None. T4 is refused, never recorded."""
    tests = [f for f in files_to_touch if f.replace("\\", "/").startswith(TEST_TREE)]
    if tests:
        return (f"it writes to {', '.join(tests)}. Test files are immutable: "
                f"{TEST_TREE}/** is write-denied for every phase.")

    if sql_change and not is_additive_index(sql_change):
        return ("its sql_change is not a single additive CREATE INDEX "
                f"statement: {sql_change.strip()[:160]!r}. An additive index "
                "is the only schema change permitted.")

    text = f"{approach} {description}".lower()
    if DDL_FORBIDDEN.search(text):
        return ("it describes a schema change beyond an additive index. "
                "Tables, columns and views are fixed.")
    if any(word in text for word in AUTH_WORDS):
        return "it touches authentication or authorization, which is out of scope."
    return None


def assess_risk(*, approach: str, description: str,
                predicted_effect: str = "") -> tuple[RiskTier, bool, str]:
    """(tier, requires_flag, why) for a candidate that is not forbidden.

    Caching and transaction-boundary changes are detected from the prose
    rather than from the approach, because both arrive described as something
    else — "keep the result between requests" is a cache.
    """
    text = f"{approach} {description} {predicted_effect}".lower()

    cached = [w for w in CACHING_WORDS if w in text]
    if cached:
        return (RiskTier.HIGH, True,
                f"it adds caching ({cached[0]}), which changes what the "
                f"endpoint returns over time rather than how fast it returns it")

    boundary = [w for w in TX_BOUNDARY_WORDS if w in text]
    if boundary:
        return (RiskTier.HIGH, True,
                f"it changes transaction boundaries ({boundary[0]}), whose "
                f"failure mode does not show up in a diff")

    tier = TIER_BY_APPROACH.get(approach, RiskTier.HIGH)
    if tier is RiskTier.HIGH:
        return (tier, True,
                f"approach '{approach}' is not in the risk table and cannot be "
                f"certified low-risk")
    return (tier, False, f"approach '{approach}' is tier {tier.value}")


# ──────────────────────────  schema-change feasibility  ──────────────────


def schema_change_blocked(*, approach: str, sql_change: str | None,
                          files_to_touch: list[str], repo: Path) -> str | None:
    """Would THIS candidate's schema change be unmeasurable, right now?

    Asked at SELECTION time, before a green worktree exists — before the patch
    agent spends a single build iteration on a candidate that could never have
    been verified. The isolation guard in `measure/isolation.py` used to be
    the only place this was checked, and by then the plan had already picked
    the candidate: the knowledge arrived too late to change anything.

    Blue is compared against ITSELF here, because there is no green yet and
    nothing has diverged — the only question this early is whether the
    harness has ANY per-arm isolation mechanism at all. It does not today, so
    this fires for every schema-changing candidate. That is correct: the
    measure-time guard exists as a BACKSTOP for when the environment changes
    between plan and measure, not as the primary check — this is.

    Two signals, matching `touches_schema`'s definition plus the literal case
    named in the brief: the approach is `add_index`, the migration directory
    appears among the files to touch, OR `sql_change` is set at all (a
    candidate can in principle carry DDL without declaring `add_index`).

    NOT called at all for an `add_index` candidate whose `audit_digest`
    resolves to a proof via `audit_proof_for` — that candidate takes the
    schema-candidate gate path instead, proven at the STATEMENT level
    (plan shape, rows examined, an interleaved direct-statement benchmark),
    which needs no per-arm isolation because it never compares the two
    arms at all. This function is still exactly correct for every OTHER
    schema-touching candidate: a hand-written index with no audit
    citation, or any other DDL this design does not have a statement-level
    proof mechanism for yet.
    """
    from ..measure.isolation import arms_share_one_schema, touches_schema

    changes_schema = bool(sql_change) or touches_schema(
        approach=approach, files_changed=files_to_touch)
    if not changes_schema:
        return None
    return arms_share_one_schema(repo, repo)


def audit_proof_for(
    *, table: str, columns: list[str], audit_digest: str | None,
    sql_audit: SqlAudit | None,
) -> tuple[IndexCandidate | None, str | None]:
    """(proof, problem) for an `add_index` candidate's `audit_digest`
    citation.

    `proof` is the SQL audit's OWN already-measured `IndexCandidate` —
    EXPLAIN plan before/after, rows examined, an interleaved statement-
    level benchmark, redundancy, write cost, size — copied onto this
    candidate rather than re-measured, exactly the "reuse what exists"
    discipline the audit phase itself was built on. `problem` names why
    the citation does not resolve, so a mismatched or invented digest is
    refused by name rather than silently falling back to the (currently
    unmeasurable) ordinary schema path.

    Returns `(None, None)` — not an error — when `audit_digest` is not
    set at all: citing the audit is optional, and a candidate that does
    not attempt it is not making a claim this function can confirm or
    refuse; it falls through to `schema_change_blocked` as before.
    """
    if audit_digest is None:
        return None, None
    if sql_audit is None:
        return None, (
            f"cites audit digest {audit_digest!r}, but no SQL audit "
            f"evidence (sql_audit.json) is available for this run."
        )
    found = sql_audit.by_digest(audit_digest)
    if found is None:
        return None, (
            f"cites audit digest {audit_digest!r}, which is not among the "
            f"statements this run's SQL audit analysed."
        )
    if found.classification != "MISSING_INDEX" or found.index_candidate is None:
        return None, (
            f"cites audit digest {audit_digest!r}, which the SQL audit "
            f"classified {found.classification!r}, not MISSING_INDEX with "
            f"a measured candidate."
        )
    proof = found.index_candidate
    if proof.already_covered_by is not None:
        return None, (
            f"the SQL audit's own candidate for digest {audit_digest!r} is "
            f"already covered by {proof.already_covered_by} — this "
            f"statement is not actually missing an index."
        )
    if proof.table.lower() != table.lower():
        return None, (
            f"cites audit digest {audit_digest!r}, whose own measured "
            f"candidate indexes {proof.table!r}, not {table!r}."
        )
    if [c.lower() for c in columns] != [c.lower() for c in proof.columns]:
        return None, (
            f"cites audit digest {audit_digest!r}, whose own measured "
            f"candidate indexes ({', '.join(proof.columns)}), not "
            f"({', '.join(columns)}) — the statement-level proof is for "
            f"the exact column list the audit measured, not a different one."
        )
    return proof, None


# ────────────────────────────────  selection  ────────────────────────────

MAX_SELECTABLE_TIER = RiskTier.MEDIUM


def selection_problem(candidate: CandidateFix) -> str | None:
    """Why this candidate cannot be the selected one, or None."""
    if candidate.not_selectable_because is not None:
        return candidate.not_selectable_because
    if candidate.risk_tier > MAX_SELECTABLE_TIER:
        return (f"risk tier {candidate.risk_tier.value} is above the "
                f"T{MAX_SELECTABLE_TIER.value} ceiling for an unattended fix")
    if not candidate.preserves_response_bytes:
        return ("it does not claim to preserve the response bytes, and the "
                "frozen oracle compares them byte for byte")
    return None


# ─────────────────────  SQL-rewrite candidates: verify, don't guess  ─────
#
# BATCH-01: a plan offered two candidates targeting one hypothesis — a
# correlated subquery (`query_rewrite`) and a batched `IN (...) GROUP BY`
# (`batch_query`) — both scored identically (expected_value_ms is
# attributed_ms x the HYPOTHESIS's fixability, the same number for any
# candidate attacking it), so selection fell through to lexical candidate id
# and happened to ship the one MySQL re-executes once per outer row. Its own
# rows_examined never moved. Nobody had to write a rule about correlated
# subqueries — EXPLAIN ANALYZE already knew.


def sql_verification_needed(*, approach: str, mechanism: str | None) -> bool:
    """Whether this candidate must supply sql_before/sql_after."""
    from ..gate.rules import INSTRUMENTED_MECHANISMS
    return (approach in SQL_REWRITE_APPROACHES
            and mechanism in INSTRUMENTED_MECHANISMS)


def verify_sql_candidate(
    box: DbToolbox | None,
    *, sql_before: str, sql_after: str, params: list[str] | None,
    mechanism: str | None,
) -> SqlVerification:
    """Run `sql_before`/`sql_after` through EXPLAIN ANALYZE and compare.

    `scored=True` only for `rows_examined`: that is a per-call quantity, and
    one EXPLAIN ANALYZE of each statement is a fair before/after comparison.
    A `sql_count` fix changes how many times the SAME shape of statement
    runs, which one EXPLAIN of each side cannot see — a batched `IN (...)`
    naturally examines more rows in its single call than one row's worth of
    the per-row original, and scoring on that delta would wrongly penalise
    exactly the fix this harness is supposed to find. It is still verified —
    both statements must at least run — the evidence just is not used to set
    expected_value_ms or not_selectable_because.
    """
    if box is None:
        return SqlVerification(
            verified=False,
            attempted=False,
            note="no live database connection at plan time; this candidate's "
                 "SQL was not independently checked against EXPLAIN ANALYZE.",
        )
    try:
        before, after = box.plan_delta(sql_before, sql_after, params)
    except Exception as exc:  # noqa: BLE001 — a bad statement is evidence too
        return SqlVerification(
            verified=False,
            attempted=True,
            note=f"EXPLAIN ANALYZE could not run both statements: {exc}",
        )

    delta_pct = (0.0 if before.rows_examined == 0 else
                round(100.0 * (after.rows_examined - before.rows_examined)
                      / before.rows_examined, 1))
    scored = mechanism == "rows_examined"
    note = (f"EXPLAIN ANALYZE: rows examined {before.rows_examined:,} -> "
            f"{after.rows_examined:,} ({delta_pct:+.1f}%), actual time "
            f"{before.actual_ms:,.1f} ms -> {after.actual_ms:,.1f} ms")
    if not scored:
        note += (". Mechanism is sql_count: a single EXPLAIN ANALYZE of each "
                 "statement does not capture a call-count fix, so this "
                 "delta is reported but not used to score or disqualify.")
    return SqlVerification(
        verified=True,
        rows_examined_before=before.rows_examined,
        rows_examined_after=after.rows_examined,
        rows_delta_pct=delta_pct,
        actual_ms_before=before.actual_ms,
        actual_ms_after=after.actual_ms,
        scored=scored,
        note=note,
    )


def sql_verification_failure(verification: SqlVerification) -> str | None:
    """Why this candidate is disqualified before a patch exists, or None.

    Three states, not two:

    - could not verify at all (`attempted=True`, `verified=False`) — EXPLAIN
      ANALYZE was run and failed to even complete against this candidate's
      own statements (bad syntax, a reference it rejects, a parameter
      mismatch). This is an UNKNOWN, not a disproof — but a candidate
      claiming an instrumented mechanism has not earned selection without
      evidence, so it disqualifies all the same. Distinct from
      `attempted=False` (no live database at plan time at all), which is a
      run-wide planning limitation, not something this candidate did wrong,
      and never disqualifies on its own.
    - verified and scored, quantity flat — disproven: the same bar
      MECHANISM_MATCH uses after a patch says this candidate would fail it,
      so it is refused here instead of after a patch, a verify and a
      measurement have already been paid for.
    - verified and scored, quantity moved — not disqualified.
    - verified but not scored (`sql_count`, a single EXPLAIN cannot see a
      call-count fix) — not disqualified; the evidence is recorded, not used
      to gate.
    """
    if not verification.verified:
        if not verification.attempted:
            return None
        return (f"EXPLAIN ANALYZE could not verify this candidate's own "
                f"sql_before/sql_after before a patch exists ({verification.note}). "
                f"An instrumented mechanism has not earned selection without "
                f"evidence — this is unverifiable, not disproven, but it "
                f"still disqualifies until it can be checked.")
    if not verification.scored:
        return None
    if verification.rows_delta_pct <= -MECHANISM_MOVE_PCT:
        return None
    return (f"EXPLAIN ANALYZE shows rows examined do not clear the "
            f"{MECHANISM_MOVE_PCT:.0f}% mechanism-match bar before any patch "
            f"exists: {verification.rows_examined_before:,} -> "
            f"{verification.rows_examined_after:,} "
            f"({verification.rows_delta_pct:+.1f}%). This candidate would "
            f"fail MECHANISM_MATCH after a patch, a verify and a measurement "
            f"had already been paid for.")
