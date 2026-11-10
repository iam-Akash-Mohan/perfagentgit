"""The SQL audit artifact: a systematic, code-driven analysis of every
application statement captured at baseline, independent of what the
diagnosis agent happens to ask about.

COL-04 is the proof this exists for: a full scan of 1.5M rows sat in the
baseline's own digest evidence while diagnosis theorised about ORM
hydration. The diagnosis agent CAN call `explain` on a busy digest, but
often does not — this phase analyses every captured statement, every time,
as ordinary code, not an agent decision.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

AuditClassification = Literal[
    "OPTIMAL", "MISSING_INDEX", "NON_SARGABLE", "BAD_JOIN_ORDER",
    "OVER_FETCHING", "UNANALYSABLE",
]


class IndexCandidate(BaseModel):
    """A MISSING_INDEX finding's proposed fix, measured by `index_benefit`
    — never asserted, always the tool's own before/after numbers.

    This IS the statement-level proof an `add_index` plan candidate can
    ship on: an index is proven by the database, not by a stopwatch. Plan
    shape and rows examined are deterministic and repeatable on ANY schema
    state, including one blue and green share — only an ENDPOINT-level
    wall-clock claim needs isolated arms, and this makes no such claim.
    """

    table: str
    columns: list[str]
    column_rationale: str = Field(
        default="",
        description="Why this column order — equality-filtered columns "
                    "lead a range or sort column, never the reverse, "
                    "because a composite index can only use its leading "
                    "columns for an equality seek. Empty for the "
                    "already-covered refusal case, where no order was "
                    "ever chosen.")
    plan_before: str = Field(
        default="",
        description="The access type and key EXPLAIN reported before the "
                    "candidate index existed — e.g. 'invoice: type=ALL, "
                    "key=no index'. Empty for the already-covered refusal "
                    "case.")
    plan_after: str = Field(
        default="",
        description="The access type and key EXPLAIN reported WITH the "
                    "candidate index built — e.g. 'invoice: type=ref, "
                    "key=senapathy_tmp_x'. The change this names (a scan "
                    "becoming a seek) is what plan_improved checks for; "
                    "empty for the already-covered refusal case.")
    access_type_before: str = Field(
        default="",
        description="Just the EXPLAIN access type before the candidate "
                    "index existed (e.g. 'ALL') — structured, for "
                    "gate.rules.PLAN_IMPROVED to compare against a fixed "
                    "ranking rather than parsing plan_before's prose.")
    access_type_after: str = Field(
        default="",
        description="The EXPLAIN access type WITH the candidate index "
                    "built (e.g. 'ref'). PLAN_IMPROVED passes when this "
                    "ranks better than access_type_before on EXPLAIN's own "
                    "ALL < index < range < ref < eq_ref < const ordering.")
    rows_examined_before: int
    rows_examined_after: int
    actual_ms_before: float
    actual_ms_after: float
    already_covered_by: str | None = Field(
        default=None,
        description="Set, with before/after left at the pre-candidate "
                    "values, when index_benefit refused to build this "
                    "candidate because an existing index already covers "
                    "it — the statement is NOT actually MISSING an index; "
                    "something else explains the rows-examined ratio. "
                    "index_not_redundant reads this field directly.")
    size_bytes: int | None = Field(
        default=None,
        description="The candidate index's own size on disk, from InnoDB's "
                    "statistics — None when the account cannot read "
                    "mysql.innodb_index_stats, reported honestly rather "
                    "than guessed at, same as index_benefit's own size "
                    "reporting.")
    table_writes_since_reset: dict[str, int] = Field(
        default_factory=dict,
        description="{'insert': n, 'update': n, 'delete': n, 'fetch': n} "
                    "from performance_schema.table_io_waits_summary_by_"
                    "table — cumulative since the server's statistics were "
                    "last reset, not a rate. Every index this candidate "
                    "adds slows every one of the insert/update rows here; "
                    "EXPLAIN never shows that cost, so the report states "
                    "it next to the reads this index serves. Empty when "
                    "the account cannot read performance_schema or the "
                    "already-covered case had nothing to measure.")
    benchmark_rounds: int = 0
    without_index_ms_samples: list[float] = Field(
        default_factory=list,
        description="The statement's own raw execution time, repeated, "
                    "with the optimizer forced OFF the candidate index "
                    "(IGNORE INDEX) — interleaved with with_index_ms_"
                    "samples (A,B,A,B...), never measured sequentially, so "
                    "neither side is credited with the other's buffer-pool "
                    "warming.")
    with_index_ms_samples: list[float] = Field(
        default_factory=list,
        description="The SAME statement, same round, with the candidate "
                    "index available to the optimizer — interleaved "
                    "against without_index_ms_samples at the SAME index "
                    "position (round i of each list was measured back to "
                    "back).")
    benchmark_p_value: float | None = Field(
        default=None,
        description="Wilcoxon signed-rank over the paired (without, with) "
                    "samples — the SAME paired-significance discipline the "
                    "endpoint-level gate already uses "
                    "(gate.rules.STATISTICAL_SIGNIFICANCE), applied here at "
                    "the statement level instead. None when fewer than 3 "
                    "rounds or every delta is zero.")
    improvement_pct: float | None = Field(
        default=None,
        description="Median with-index vs. median without-index wall-clock "
                    "time from the interleaved benchmark. None if the "
                    "index did not make the statement faster.")


class RewriteCandidate(BaseModel):
    """A NON_SARGABLE finding's proposed fix. Only ever populated for a
    pattern this audit can rewrite WITH PROOF of equivalence
    (`query_equivalent`) and a measured, interleaved benchmark — an
    unproven rewrite is recorded as a marker on the statement, never as a
    candidate here. See `audit/candidates.py::propose_rewrite` for exactly
    which patterns qualify."""

    sql_before: str
    sql_after: str
    equivalence_verified: bool
    equivalence_note: str
    rows_examined_before: int
    rows_examined_after: int
    benchmark_rounds: int
    original_ms_samples: list[float]
    candidate_ms_samples: list[float]
    improvement_pct: float | None = Field(
        default=None,
        description="Median candidate vs. median original, from the "
                    "INTERLEAVED benchmark — None if the rewrite made "
                    "things worse or the samples are too noisy to call.")


class AuditedStatement(BaseModel):
    """One application statement captured at baseline, fully analysed.

    Every digest baseline captured gets an entry here — `classification`
    is `UNANALYSABLE` with `unanalysable_reason` set, never silent
    omission, when a statement could not be reconstructed or measured.
    """

    digest: str
    digest_text: str = Field(
        description="The statement as MySQL's own digest normalises it — "
                    "literals collapsed to `?`. What the baseline itself "
                    "already recorded.")
    executable_sql: str | None = Field(
        default=None,
        description="The digest text reconstructed into something EXPLAIN "
                    "can run, with real bound values — None when "
                    "`unanalysable_reason` explains why it could not be.")
    resolved_params: list[str] = []
    calls: int
    total_ms: float

    limit_offset_source: Literal["digest", "corpus", "fallback"] | None = Field(
        default=None,
        description="Where this statement's LIMIT/OFFSET values came from, "
                    "when it has one — None for a statement with no LIMIT "
                    "clause at all. 'digest': the whole statement came from "
                    "a live connection's own SQL_TEXT history, real values "
                    "throughout, nothing synthesised. 'corpus': spliced "
                    "from a matching benchmark request's own page/size "
                    "query parameters (see `limit_offset_request_id`) -- "
                    "the real page depth this run actually measured at. "
                    "'fallback': a fixed synthetic pair, used only when no "
                    "corpus request supplied real values -- page depth "
                    "changes the plan, so a number built this way is a "
                    "weaker claim than one built from 'corpus' or 'digest' "
                    "and is marked as such rather than presented the same.")
    limit_offset_request_id: str | None = Field(
        default=None,
        description="The corpus benchmark request whose page/size supplied "
                    "this statement's LIMIT/OFFSET, set only when "
                    "limit_offset_source is 'corpus'.")

    classification: AuditClassification
    unanalysable_reason: str | None = None
    non_sargable_marker: str | None = Field(
        default=None,
        description="The literal EXPLAIN-plan syntax that defeats an "
                    "index, when classification is NON_SARGABLE — a "
                    "function-wrapped column, a leading wildcard, or an "
                    "implicit type conversion. Same detector the plan "
                    "phase's fixability policy already uses "
                    "(plan/fixability.py::non_sargable_marker), applied "
                    "to the statement's own text rather than a model's "
                    "prose about it.")

    access_types: dict[str, str] = {}
    keys_used: dict[str, str | None] = {}
    rows_examined: int = 0
    rows_returned: int = 0
    actual_ms: float = 0.0
    uses_filesort: bool = False
    uses_temporary: bool = False

    index_candidate: IndexCandidate | None = None
    rewrite_candidate: RewriteCandidate | None = None


class SqlAudit(BaseModel):
    """One endpoint's complete SQL audit. Written to `sql_audit.json`,
    read by DIAGNOSE (reconciliation, see `diagnose/validate.py`) and
    PLAN (the brief's baseline section)."""

    endpoint: str
    measured_total_ms: float
    baseline_hash: str | None = Field(
        default=None,
        description="This run's baseline.json hash at the time of the "
                    "sweep, so a resume's staleness check "
                    "(orchestrator._stale_inputs) can tell an audit "
                    "derived from the current baseline from one left over "
                    "from an earlier measurement, the same way every other "
                    "baseline-derived artifact already is.")
    statements: list[AuditedStatement] = []

    def non_optimal(self) -> list[AuditedStatement]:
        return [s for s in self.statements if s.classification != "OPTIMAL"]

    def by_digest(self, digest: str) -> AuditedStatement | None:
        return next((s for s in self.statements if s.digest == digest), None)
