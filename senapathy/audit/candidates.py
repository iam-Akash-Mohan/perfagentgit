"""Proposing a fix for a non-optimal statement, never asserting one.

Every number here comes from the database: `index_benefit` for
MISSING_INDEX, `query_equivalent` + an interleaved `benchmark_interleaved`
for NON_SARGABLE. A statement this module cannot confidently propose a
PROVEN candidate for keeps its classification and marker — recorded as a
finding, not forced into a candidate that was never actually measured.
"""

from __future__ import annotations

import re
import statistics
from typing import Any

from ..contracts import AuditedStatement, IndexCandidate, RewriteCandidate
from .reconstruct import (extract_equality_columns, table_alias_map,
                          table_reference_end)

# Live-confirmed this matters, not a round number picked for its own sake:
# Wilcoxon signed-rank's exact two-sided p-value over n PAIRED, non-tied
# samples has a FLOOR of 2 * 0.5**n, regardless of how large or consistent
# the real effect is -- at n=5 that floor is 0.0625, which can never clear
# a p < 0.05 gate (gate.rules.MAX_INDEX_P_VALUE) even when every single
# round improved. mini-spring's own IDX-01 scenario hit this exactly: 5/5
# rounds faster, p=0.0625, STATEMENT_FASTER failed on a false negative the
# statistics could never have passed. 8 rounds (matching measure/ab.py's
# own `ab_iterations` default for the endpoint-level gate) floors at
# 2 * 0.5**8 ≈ 0.0078, comfortably below the bar.
BENCHMARK_ROUNDS = 8


def _paired_p_value(before: list[float], after: list[float]) -> float | None:
    """Wilcoxon signed-rank over paired (before, after) samples — the SAME
    paired-significance discipline the endpoint-level gate already uses
    (gate.rules.STATISTICAL_SIGNIFICANCE / measure.ab.wilcoxon_p), applied
    here to a statement-level interleaved benchmark instead of an
    endpoint-level one."""
    from scipy.stats import wilcoxon

    deltas = [a - b for a, b in zip(after, before)]
    if len(deltas) < 3 or all(d == 0 for d in deltas):
        return None
    try:
        return float(wilcoxon(deltas, alternative="two-sided").pvalue)
    except ValueError:
        return None


def _median_improvement_pct(before: list[float], after: list[float]) -> float | None:
    if not before or not after:
        return None
    before_median = statistics.median(before)
    after_median = statistics.median(after)
    if not before_median or after_median >= before_median:
        return None
    return round(100.0 * (before_median - after_median) / before_median, 1)

# `DATE(col) >= %s AND DATE(col) < %s` (or `<=`/`>`, either side, any
# order) -- the ONE non-sargable shape this module rewrites, because it is
# the one PROVABLY correct for any bound values, not just contiguous ones:
# comparing a DATETIME/TIMESTAMP column against a DATE literal with a
# half-open range (>=/<) is exactly equivalent whether or not the DATE()
# wrapper is there (MySQL already treats the literal as that day's
# midnight); dropping the wrapper changes nothing about which rows match.
# A bare equality (`DATE(col) = %s`) is NOT handled -- that one actually
# needs a day-wide range to stay equivalent, a different, more invasive
# rewrite this module does not attempt.
#
# Whitespace- and backtick-tolerant on purpose: `executable_sql` comes from
# a digest's own text (`DATE ( \`cp1_0\` . \`paid_on\` ) >= %s`, spaces and
# backticks exactly as MySQL's digest renders it), not hand-written SQL --
# confirmed live against COL-06's real query, which is this pattern.
_DATE_RANGE = re.compile(
    r"DATE\s*\(\s*(?P<col>(?:`?\w+`?\s*\.\s*)?`?\w+`?)\s*\)\s*"
    r"(?P<op>>=|<=|>|<)\s*%s",
    re.I,
)


def propose_index_candidate(
    box: Any, statement: AuditedStatement,
) -> IndexCandidate | None:
    """A MISSING_INDEX finding's candidate column set, measured by
    `index_benefit` — the statement-level proof an `add_index` plan
    candidate ships on. The column set is every equality-filtered column
    this statement's own digest text names, in WHERE-clause order — the
    natural leading-column order for a composite index (an equality seek
    can only use a composite index's LEADING columns, so a range or sort
    column has to come after them, never before). Returns None when no
    column could be identified at all (nothing to propose)."""
    if statement.executable_sql is None:
        return None
    by_table = extract_equality_columns(statement.digest_text)
    if not by_table:
        return None
    # `access_types`/`keys_used` are keyed by whatever alias EXPLAIN's own
    # "table" column reports, NOT necessarily the real table name — a query
    # aliased `p1_0` reports EXPLAIN rows keyed "p1_0", confirmed live.
    # Resolved back to the real name via this statement's own FROM clause so
    # the unindexed-table check below actually fires, rather than silently
    # falling through to the single-table fallback every time.
    aliases = table_alias_map(statement.digest_text)
    target_table = next(
        (aliases.get(t.lower(), t) for t in statement.access_types
         if statement.keys_used.get(t) is None
         and aliases.get(t.lower(), t) in by_table),
        next(iter(by_table), None),
    )
    if target_table is None:
        return None
    columns = by_table[target_table]
    # The SAME alias EXPLAIN reported for target_table, to look its own
    # access type back up in access_types/keys_used (both keyed by alias).
    explain_key = next(
        (t for t in statement.access_types
         if aliases.get(t.lower(), t) == target_table),
        target_table,
    )
    rationale = (
        f"{', '.join(columns)} lead{'s' if len(columns) == 1 else ''} the "
        f"index because {'it is' if len(columns) == 1 else 'they are'} "
        f"filtered by equality in this statement's own WHERE clause, in "
        f"the order the predicate names them — a composite index can only "
        f"use its leading columns for an equality seek, so a range or "
        f"sort column has to come after them, never before."
    )

    hint_position = table_reference_end(statement.executable_sql, target_table)
    result = box.index_benefit_measured(
        target_table, columns, statement.executable_sql,
        statement.resolved_params or None,
        benchmark_hint_position=hint_position, benchmark_rounds=BENCHMARK_ROUNDS,
    )
    writes = box.table_write_volume(target_table)
    if result.covered_by is not None:
        return IndexCandidate(
            table=target_table, columns=columns,
            rows_examined_before=statement.rows_examined,
            rows_examined_after=statement.rows_examined,
            actual_ms_before=statement.actual_ms,
            actual_ms_after=statement.actual_ms,
            already_covered_by=result.covered_by,
            table_writes_since_reset=writes or {},
        )
    if result.timed_out or result.after is None:
        return None
    return IndexCandidate(
        table=target_table, columns=columns, column_rationale=rationale,
        plan_before=f"{target_table}: {result.before.summary()}",
        plan_after=f"{target_table}: {result.after.summary()}",
        access_type_before=result.before.access_types.get(explain_key, ""),
        access_type_after=result.after.access_types.get(explain_key, ""),
        rows_examined_before=result.before.rows_examined,
        rows_examined_after=result.after.rows_examined,
        actual_ms_before=result.before.actual_ms,
        actual_ms_after=result.after.actual_ms,
        size_bytes=result.size_bytes,
        table_writes_since_reset=writes or {},
        benchmark_rounds=len(result.without_index_ms_samples),
        without_index_ms_samples=result.without_index_ms_samples,
        with_index_ms_samples=result.with_index_ms_samples,
        benchmark_p_value=_paired_p_value(
            result.without_index_ms_samples, result.with_index_ms_samples),
        improvement_pct=_median_improvement_pct(
            result.without_index_ms_samples, result.with_index_ms_samples),
    )


def propose_rewrite_candidate(
    box: Any, statement: AuditedStatement,
) -> RewriteCandidate | None:
    """A NON_SARGABLE finding's candidate, for the one pattern this module
    can rewrite with proof: a DATE()-wrapped half-open range. Equivalence
    is PROVEN with `query_equivalent`, not assumed from the rewrite's own
    logic being correct in theory — a rewrite `query_equivalent` cannot
    confirm returns None here, the same discipline as every other
    unproven-candidate refusal in this project. The improvement itself is
    measured by an INTERLEAVED benchmark, never a sequential before/after."""
    if statement.executable_sql is None:
        return None
    matches = list(_DATE_RANGE.finditer(statement.executable_sql))
    if not matches:
        return None

    rewritten = _DATE_RANGE.sub(r"\g<col> \g<op> %s", statement.executable_sql)
    if rewritten == statement.executable_sql:
        return None

    params = statement.resolved_params or None
    equivalence = box.query_equivalent(
        statement.executable_sql, rewritten, params)
    verified = equivalence.split("\n", 1)[0].strip() == "EQUIVALENT"
    if not verified:
        return None

    before_plan, after_plan = box.plan_delta(
        statement.executable_sql, rewritten, params)

    a_times, b_times = box.benchmark_interleaved(
        statement.executable_sql, rewritten, params, rounds=BENCHMARK_ROUNDS)
    original_median = statistics.median(a_times)
    candidate_median = statistics.median(b_times)
    improvement_pct = (
        round(100.0 * (original_median - candidate_median) / original_median, 1)
        if original_median and candidate_median < original_median
        else None
    )

    return RewriteCandidate(
        sql_before=statement.executable_sql, sql_after=rewritten,
        equivalence_verified=verified, equivalence_note=equivalence,
        rows_examined_before=before_plan.rows_examined,
        rows_examined_after=after_plan.rows_examined,
        benchmark_rounds=BENCHMARK_ROUNDS,
        original_ms_samples=a_times, candidate_ms_samples=b_times,
        improvement_pct=improvement_pct,
    )
