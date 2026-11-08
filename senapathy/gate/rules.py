"""The six gates. Every threshold here is a constant, on purpose.

Thresholds are module constants and not function parameters. A gate whose bar
can be passed in is a gate that will eventually be called with a lower bar —
probably by a caller trying to get a run to finish. `evaluate_gates` therefore
takes evidence and nothing else, and computes `passed` as the last statement
from the collected results. There is no early return and no override keyword.

Each gate reports its threshold and its actual value as strings, so the report
can print the comparison a reader would otherwise have to trust.
"""

from __future__ import annotations

import re
from typing import Any

from ..contracts import (CandidatePlan, EquivalenceVerdict, GateDecision,
                         GateName, GateResult, Patch, PerfReport,
                         TerminalState)

# ─────────────────────────────  the thresholds  ──────────────────────────

# p95 must improve by at least this much. Negative delta means faster.
P95_IMPROVEMENT_PCT = 15.0

# Paired test, not the unpaired one — see STATISTICAL_SIGNIFICANCE below.
MAX_P_VALUE = 0.01

# A fix for one bottleneck that rewrites 200 lines is not a fix for one
# bottleneck.
MAX_CHANGED_LINES = 200

# How much a predicted quantity must actually move to count as confirmed.
MECHANISM_MOVE_PCT = 5.0

TEST_TREE = "src/test"


# The measurable quantities PerfReport actually carries before/after values
# for. A candidate's `mechanism` field is validated against this set at plan
# time (senapathy/plan/submit.py); anything else reaches this gate only from
# a pre-existing artifact recorded before the typed field existed.
INSTRUMENTED_MECHANISMS = frozenset({"sql_count", "rows_examined"})

# Predicted mechanisms, classified by a FIXED keyword table rather than by
# asking a model what its own prose meant. Only the first two are instrumented;
# see the MECHANISM_MATCH gate for what happens to the others. USED ONLY AS A
# FALLBACK: a candidate with a typed `mechanism` field never reaches this
# table — see classify_mechanism.
_MECHANISM_KEYWORDS: dict[str, tuple[str, ...]] = {
    "sql_count": (
        "quer", "n+1", "n + 1", "statement", "round trip", "roundtrip",
        "fetch join", "entitygraph", "entity graph", "batch", "join fetch",
    ),
    "rows_examined": (
        "index", "rows examined", "full scan", "table scan", "projection",
        "only the columns", "select only",
    ),
    "serialization": (
        "serial", "jackson", "objectmapper", "json writer", "reflection",
        "afterburner",
    ),
    "allocation": (
        "allocat", "stringbuilder", "string builder", "concat", "boxing",
        "garbage",
    ),
}

# Additive index DDL is the one schema change the design permits.
_ALLOWED_DDL = re.compile(
    r"\b(create\s+(unique\s+)?index|add\s+(unique\s+)?(index|key))\b", re.I
)
_FORBIDDEN_DDL = re.compile(
    r"\b(create\s+table|alter\s+table|drop\s+(table|column|index)|"
    r"add\s+column|modify\s+column|change\s+column|rename\s+(table|column)|"
    r"truncate\s+table)\b",
    re.I,
)
_MAPPING = re.compile(
    r'@(Request|Get|Post|Put|Delete|Patch)Mapping\s*\(\s*"?([^")]*)', re.I
)
_RECORD = re.compile(r"public\s+record\s+(\w+)\s*\(([^)]*)\)")


# ────────────────────────────  diff inspection  ──────────────────────────


def touched_files(diff: str) -> list[str]:
    """Files the diff actually changes, read from its headers.

    Deliberately taken from the diff rather than from what the patch claims:
    the point of DIFF_MINIMALITY is to check the claim.
    """
    found: list[str] = []
    for line in diff.splitlines():
        if line.startswith("+++ b/"):
            path = line[len("+++ b/"):].strip()
            if path != "/dev/null" and path not in found:
                found.append(path)
    return found


def changed_lines(diff: str) -> tuple[list[str], list[str]]:
    """(added, removed) content lines, excluding the +++/--- headers."""
    added = [
        l[1:] for l in diff.splitlines()
        if l.startswith("+") and not l.startswith("+++")
    ]
    removed = [
        l[1:] for l in diff.splitlines()
        if l.startswith("-") and not l.startswith("---")
    ]
    return added, removed


def mapping_paths(lines: list[str]) -> set[str]:
    return {match.group(2).strip() for line in lines for match in _MAPPING.finditer(line)}


def record_signatures(lines: list[str]) -> dict[str, str]:
    return {
        match.group(1): " ".join(match.group(2).split())
        for line in lines for match in _RECORD.finditer(line)
    }


def classify_mechanism(plan: CandidatePlan) -> str:
    """Which measurable quantity the plan predicted would move.

    `mechanism` is a typed field the plan agent states directly, validated
    at plan time — this function returns it unchanged whenever it is set.
    The keyword scan below only runs for a plan recorded before that field
    existed, and reads `mechanism_changed` ALONE: COL-06's plan stated
    mechanism_changed="rows examined" but was classified sql_count because
    its `description` also mentioned a query, which is true of nearly every
    candidate's description and tells you nothing about which quantity was
    actually predicted to move.
    """
    if plan.mechanism is not None:
        return plan.mechanism

    text = plan.mechanism_changed.lower()
    matched = [
        kind for kind in ("sql_count", "rows_examined", "serialization", "allocation")
        if any(keyword in text for keyword in _MECHANISM_KEYWORDS[kind])
    ]
    if len(matched) > 1:
        # Ambiguity is not a result — picking the first in some fixed order
        # is exactly the bug this replaces. Name every kind it matched.
        return "ambiguous:" + ",".join(matched)
    if matched:
        return matched[0]
    return "unknown"


def _pct_change(before: float, after: float) -> float:
    if before == 0:
        return 0.0
    return 100.0 * (after - before) / before


# ──────────────────────────────  the gates  ──────────────────────────────


def _behaviour_preserved(verdict: EquivalenceVerdict) -> GateResult:
    passed = verdict.passed and verdict.held_back_passed
    return GateResult(
        name=GateName.BEHAVIOUR_PRESERVED,
        passed=passed,
        threshold="oracle verdict passed AND held-back set passed",
        actual=(
            f"passed={verdict.passed}, held_back_passed={verdict.held_back_passed}, "
            f"{verdict.identical}/{verdict.compared} identical, "
            f"{len(verdict.diffs)} differing path(s)"
        ),
        detail=(
            "Behaviour is unchanged on every recorded request, including the "
            "held-back set the measurement never touched."
            if passed else
            "The patch changed behaviour. Nothing downstream of this matters "
            "until it does not."
        ),
    )


def _improvement_threshold(perf: PerfReport) -> GateResult:
    passed = perf.p95_delta_pct <= -P95_IMPROVEMENT_PCT
    direction = "faster" if perf.p95_delta_pct < 0 else "slower"
    return GateResult(
        name=GateName.IMPROVEMENT_THRESHOLD,
        passed=passed,
        threshold=f"p95 delta <= -{P95_IMPROVEMENT_PCT:.1f}% (at least "
                  f"{P95_IMPROVEMENT_PCT:.0f}% faster)",
        actual=f"p95 {perf.before.p95_ms:.1f} ms -> {perf.after.p95_ms:.1f} ms "
               f"({perf.p95_delta_pct:+.2f}%, {direction})",
        detail=(
            f"p95 improved by {abs(perf.p95_delta_pct):.2f}%."
            if passed else
            f"p95 is {abs(perf.p95_delta_pct):.2f}% {direction}, which does not "
            f"clear the {P95_IMPROVEMENT_PCT:.0f}% bar. A change this size is "
            "not worth the risk of shipping it."
        ),
    )


def _statistical_significance(paired_p: float | None) -> GateResult:
    """Wilcoxon signed-rank, NOT Mann-Whitney.

    Each request is measured on both arms back to back, so the samples are
    paired by design. On this corpus the per-request spread (a 1.8 ms error
    path next to a 1500 ms search) is two orders of magnitude larger than the
    between-arm difference, so an unpaired rank test cannot see the arm effect
    at all and reports p=0.36 where the paired test reports p=0.0015. Gating on
    the unpaired value would pass regressions.

    Note this gate is direction-BLIND: a significant regression passes it. That
    is correct division of labour — IMPROVEMENT_THRESHOLD carries direction,
    this one carries confidence — but it means a passing row here is not on its
    own good news.
    """
    if paired_p is None:
        return GateResult(
            name=GateName.STATISTICAL_SIGNIFICANCE,
            passed=False,
            threshold=f"Wilcoxon signed-rank p < {MAX_P_VALUE}",
            actual="not computed",
            detail="No paired p-value was recorded, so significance is unknown. "
                   "Unknown is not a pass.",
        )
    passed = paired_p < MAX_P_VALUE
    return GateResult(
        name=GateName.STATISTICAL_SIGNIFICANCE,
        passed=passed,
        threshold=f"Wilcoxon signed-rank p < {MAX_P_VALUE}",
        actual=f"p = {paired_p:.6f}",
        detail=(
            "The difference between the arms is unlikely to be noise. This gate "
            "measures confidence, not direction — see IMPROVEMENT_THRESHOLD for "
            "which way it went."
            if passed else
            f"p = {paired_p:.6f} does not clear {MAX_P_VALUE}, so the measured "
            "difference is indistinguishable from noise."
        ),
    )


def _diff_minimality(patch: Patch, plan: CandidatePlan) -> GateResult:
    actually_touched = touched_files(patch.diff) or list(patch.files_changed)
    allowed = set(plan.files_to_touch)
    outside = sorted(set(actually_touched) - allowed)
    tests = sorted(f for f in actually_touched if f.startswith(TEST_TREE))
    total_lines = patch.lines_added + patch.lines_removed

    passed = not outside and not tests and total_lines < MAX_CHANGED_LINES
    problems: list[str] = []
    if outside:
        problems.append(f"touched files outside the plan: {', '.join(outside)}")
    if tests:
        problems.append(f"touched test files: {', '.join(tests)}")
    if total_lines >= MAX_CHANGED_LINES:
        problems.append(f"{total_lines} changed lines is not a minimal fix")

    return GateResult(
        name=GateName.DIFF_MINIMALITY,
        passed=passed,
        threshold=f"only planned files, no {TEST_TREE}/**, "
                  f"< {MAX_CHANGED_LINES} changed lines",
        actual=f"{len(actually_touched)} file(s), +{patch.lines_added} "
               f"-{patch.lines_removed} = {total_lines} line(s)",
        detail="; ".join(problems) if problems else
               "The diff stays inside the planned files and is small enough to "
               "review in one sitting.",
    )


def _mechanism_match(perf: PerfReport, plan: CandidatePlan) -> GateResult:
    """Did the quantity the plan predicted would move actually move?

    "Right answer, wrong reason" has to be detectable. A latency win whose
    predicted mechanism did not budge means the diagnosis was wrong even though
    the graph looks good, and shipping it teaches the wrong lesson to whoever
    reads the PR.

    Mechanisms we do not instrument FAIL. That is fail-closed on purpose: an
    unconfirmable mechanism is not a confirmed one. It also means a genuine
    serialization or allocation win cannot pass this gate until P07/P08 grow an
    instrument for it — a real limitation, and the honest place to put it.
    """
    kind = classify_mechanism(plan)
    sql_delta = _pct_change(perf.sql_count_before, perf.sql_count_after)
    rows_delta = _pct_change(perf.rows_examined_before, perf.rows_examined_after)

    if kind == "sql_count":
        passed = sql_delta <= -MECHANISM_MOVE_PCT
        threshold = (f"predicted fewer queries: SQL statements must drop by "
                     f">= {MECHANISM_MOVE_PCT:.0f}%")
        actual = (f"SQL statements {perf.sql_count_before} -> "
                  f"{perf.sql_count_after} ({sql_delta:+.1f}%)")
        detail = ("The predicted reduction in database round trips happened."
                  if passed else
                  "The plan predicted fewer queries and the query count did not "
                  "drop. Any latency change came from somewhere else, so the "
                  "stated mechanism is not what improved.")
    elif kind == "rows_examined":
        passed = rows_delta <= -MECHANISM_MOVE_PCT
        threshold = (f"predicted less scanning: rows examined must drop by "
                     f">= {MECHANISM_MOVE_PCT:.0f}%")
        actual = (f"rows examined {perf.rows_examined_before} -> "
                  f"{perf.rows_examined_after} ({rows_delta:+.1f}%)")
        detail = ("The predicted reduction in rows examined happened."
                  if passed else
                  "The plan predicted less scanning and rows examined did not "
                  "drop.")
    elif kind.startswith("ambiguous:"):
        matched = kind.split(":", 1)[1]
        passed = False
        threshold = "the predicted mechanism must be unambiguous"
        actual = (f"mechanism_changed text matched more than one kind "
                  f"({matched}); measured quantities moved SQL "
                  f"{sql_delta:+.1f}%, rows {rows_delta:+.1f}%")
        detail = (
            f"This plan predates the typed `mechanism` field and its "
            f"mechanism_changed text matches more than one measurable kind "
            f"({matched}), with nothing to disambiguate which was actually "
            f"predicted. Matching several kinds is ambiguity, not a result, "
            f"so this gate fails closed rather than guessing."
        )
    else:
        passed = False
        threshold = "the predicted mechanism must be one this harness measures"
        actual = (f"predicted mechanism classified as '{kind}'; measured "
                  f"quantities moved SQL {sql_delta:+.1f}%, rows "
                  f"{rows_delta:+.1f}%")
        detail = (
            f"The plan predicts a '{kind}' improvement, and nothing in this "
            "harness measures that. SQL statement counts and rows examined are "
            "the only instrumented mechanisms, and neither moved. An "
            "unconfirmable mechanism cannot be a confirmed one, so this gate "
            "fails closed."
        )

    return GateResult(
        name=GateName.MECHANISM_MATCH, passed=passed,
        threshold=threshold, actual=actual, detail=detail,
    )


def _no_forbidden_change(patch: Patch) -> GateResult:
    added, removed = changed_lines(patch.diff)
    files = touched_files(patch.diff) or list(patch.files_changed)
    problems: list[str] = []

    tests = [f for f in files if f.startswith(TEST_TREE)]
    if tests:
        problems.append(f"test files modified: {', '.join(sorted(tests))}")

    for line in added + removed:
        if _FORBIDDEN_DDL.search(line) and not _ALLOWED_DDL.search(line):
            problems.append(f"schema change beyond an added index: {line.strip()[:80]}")
            break

    before_paths, after_paths = mapping_paths(removed), mapping_paths(added)
    if before_paths != after_paths:
        problems.append(
            f"request mapping changed: {sorted(before_paths)} -> "
            f"{sorted(after_paths)}"
        )

    before_records, after_records = record_signatures(removed), record_signatures(added)
    for name in sorted(set(before_records) & set(after_records)):
        if before_records[name] != after_records[name]:
            problems.append(
                f"response shape changed: record {name}"
                f"({before_records[name]}) -> ({after_records[name]})"
            )

    passed = not problems
    return GateResult(
        name=GateName.NO_FORBIDDEN_CHANGE, passed=passed,
        threshold="no schema change beyond an added index, no API contract "
                  "change, no test modification",
        actual=f"{len(files)} file(s) inspected, {len(added)} added and "
               f"{len(removed)} removed line(s)",
        detail="; ".join(problems) if problems else
               "No schema DDL, no mapping change, no altered response record, "
               "no test file touched.",
    )


# ──────────────────────────────  the decision  ───────────────────────────


def evaluate_gates(
    verdict: EquivalenceVerdict,
    perf_report: PerfReport,
    patch: Patch,
    plan: CandidatePlan,
    *,
    paired_p_value: float | None = None,
) -> GateDecision:
    """Run all six gates and compute the decision.

    The paired p-value is now a FIELD on `PerfReport`, read from the artifact.
    The keyword survives for callers holding a report recorded before the
    field existed, and for those it is an INPUT rather than an override: it
    cannot make a failing gate pass, only supply the evidence one gate reads.
    When both are present the artifact wins, because the artifact is what a
    reader can check.

    Every gate runs even after one has failed — a report that stops at the
    first failure hides the other five, and the reader needs all six to know
    whether the patch is close or hopeless.
    """
    results = [
        _behaviour_preserved(verdict),
        _improvement_threshold(perf_report),
        _statistical_significance(
            perf_report.p_value_paired
            if perf_report.p_value_paired is not None else paired_p_value),
        _diff_minimality(patch, plan),
        _mechanism_match(perf_report, plan),
        _no_forbidden_change(patch),
    ]
    # Last statement, from the collected results. No branch above may shortcut
    # this, and no argument to this function can change it.
    passed = all(result.passed for result in results)
    return GateDecision(results=results, passed=passed)


def terminal_state_for(
    decision: GateDecision, *, shipped: bool = False
) -> TerminalState:
    """The honest terminal state for a gate decision.

    Order matters and encodes what a reader most needs to know first: a
    behaviour change makes every other verdict irrelevant, and "we measured it
    and it is not better" is a different and more useful answer than "the patch
    was unacceptable for some other reason".
    """
    failed = {r.name for r in decision.results if not r.passed}

    if GateName.BEHAVIOUR_PRESERVED in failed:
        return TerminalState.BEHAVIOUR_NOT_PRESERVED
    if failed & {GateName.IMPROVEMENT_THRESHOLD, GateName.STATISTICAL_SIGNIFICANCE}:
        return TerminalState.NO_SIGNIFICANT_IMPROVEMENT
    if failed:
        # Minimality, mechanism, or a forbidden change: the bottleneck may be
        # real and the measurement good, but this patch is not shippable.
        return TerminalState.BOTTLENECK_NOT_FIXABLE
    return TerminalState.PR_OPENED if shipped else TerminalState.READY_TO_SHIP


# ─────────────────────────────  confidence band  ─────────────────────────
#
# A SEPARATE signal from the gate's pass/fail. The gate already decided
# whether the evidence clears every threshold; this says how much of that
# "passed" is still resting on something nobody actually classified -- an
# unclassified fixability score, a test suite that never executed. Neither
# of those fails a gate, and both matter to a reviewer deciding whether to
# trust the result. Every input here is a fact already recorded on another
# artifact, read off a FIXED table -- a model is never asked for this.

# 3+ stacked unknowns is treated the same as a failed gate: at that point
# "the evidence passed" is carrying enough asterisks that it is no longer a
# confidence question, it is a coverage gap.
_LOW_UNKNOWN_CEILING = 3


def confidence_band_for(
    decision: GateDecision, *, verdict: EquivalenceVerdict, perf: PerfReport,
    risk_tier: int, diff_lines: int, held_back_count: int,
    selected_fixability_classified: bool, tests_executed: int,
) -> dict[str, Any]:
    """LOW / MEDIUM / HIGH, with the reason, the inputs, and what would raise
    it -- all computed, none of it narrated.
    """
    failed = [r.name.value for r in decision.results if not r.passed]

    unknowns: list[str] = []
    if not selected_fixability_classified:
        unknowns.append("the selected hypothesis's fixability was a "
                         "placeholder, not a policy-table classification")
    if tests_executed == 0:
        unknowns.append("the repository's own test suite executed zero tests")

    p95_margin = abs(perf.p95_delta_pct) - P95_IMPROVEMENT_PCT
    p = perf.p_value_paired

    if failed:
        level, reason = "LOW", f"a gate did not pass: {', '.join(failed)}"
    elif risk_tier >= 3:
        level = "LOW"
        reason = (f"risk tier T{risk_tier} carries more behaviour risk than "
                  f"an unattended fix should ship with")
    elif len(unknowns) >= _LOW_UNKNOWN_CEILING:
        level, reason = "LOW", "; ".join(unknowns)
    elif unknowns:
        level, reason = "MEDIUM", " and ".join(unknowns)
    else:
        level = "HIGH"
        reason = (f"every gate passed with margin to spare (p95 "
                  f"{p95_margin:+.1f} points past threshold, paired p "
                  f"{p if p is not None else 'not computed'}), risk tier "
                  f"T{risk_tier}, a {diff_lines}-line diff")

    would_raise_bits = []
    if not selected_fixability_classified:
        would_raise_bits.append(
            "a classified fixability assessment for the selected hypothesis")
    if tests_executed == 0:
        would_raise_bits.append(
            "a test suite that actually executes against this repository")
    if level == "HIGH":
        would_raise = "Nothing listed here is withholding this from HIGH."
    elif would_raise_bits:
        would_raise = " and ".join(would_raise_bits) + " would raise this."
    else:
        would_raise = ("A lower risk tier or a cleared gate would raise "
                       "this; it is not withheld by an unknown.")

    inputs = [
        f"behaviour: {verdict.identical}/{verdict.compared} identical, "
        f"{held_back_count} held back, held_back_passed={verdict.held_back_passed}",
        f"paired p: {p if p is not None else 'not computed'} "
        f"(threshold {MAX_P_VALUE})",
        f"mechanism: {'CONFIRMED' if perf.mechanism_confirmed else 'NOT CONFIRMED'}",
        f"p95 margin past threshold: {p95_margin:+.1f} points "
        f"({abs(perf.p95_delta_pct):.1f}% vs a {P95_IMPROVEMENT_PCT:.0f}% floor)",
        f"risk tier: T{risk_tier}",
        f"diff size: {diff_lines} line(s)",
    ]
    if unknowns:
        inputs.append("unknowns: " + "; ".join(unknowns))

    return {"level": level, "reason": reason, "inputs": inputs,
            "would_raise": would_raise}


def render_gate_table(decision: GateDecision, terminal: TerminalState) -> str:
    out: list[str] = ["", "GATE DECISION", f"  decided by     : {decision.decided_by}",
                      f"  overall        : "
                      f"{'PASSED' if decision.passed else 'FAILED'}",
                      f"  terminal state : {terminal.value}", ""]
    for result in decision.results:
        mark = "PASS" if result.passed else "FAIL"
        out.append(f"  [{mark}] {result.name.value}")
        out.append(f"         threshold  {result.threshold}")
        out.append(f"         actual     {result.actual}")
        for line in _wrap(result.detail, 66):
            out.append(f"         {line}")
        out.append("")
    return "\n".join(out)


def _wrap(text: str, width: int) -> list[str]:
    import textwrap

    return textwrap.wrap(text, width=width) or [""]
