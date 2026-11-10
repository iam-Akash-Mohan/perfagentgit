"""The database toolkit: four tools that measure instead of guessing.

This is what makes the product a database optimizer rather than a Java
refactorer. `get_schema` already says which indexes exist; none of it says what
a MISSING index would be worth, and "add an index on (customer_id, status)" is
a guess until something has timed it.

`index_benefit` is the one that matters. It builds the index, measures the same
statement again, and drops it — so the number in the plan is a measurement, not
a belief. That also makes it the most dangerous thing in this codebase: it
issues DDL against a live schema. Every guardrail below is in code, and none of
them is a parameter a caller can relax:

  * additive indexes only — no DROP, no ALTER, no column change;
  * only tables the endpoint model named, passed into the factory;
  * every created index is named `senapathy_tmp_<hash>`, so a leaked one is
    identifiable at a glance;
  * a sweep at construction drops anything `senapathy_tmp_*` a crashed run left;
  * the index is dropped in a `finally`, including on timeout;
  * and the whole tool refuses unless the connection is the configured
    measurement database.

Read-only by default: `explain`, `list_indexes` and `query_equivalent` cannot
write anything, and `explain`/`query_equivalent` refuse any statement that is
not a SELECT.
"""

from __future__ import annotations

import hashlib
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from ..contracts import IndexInfo
from ..errors import GuardrailError, ToolError
from ..measure.sql import MySQLTarget, connect
from . import ToolSpec

# Every index this module creates carries it. A leaked index is then both
# identifiable and sweepable; an index called `idx_invoice_status` left behind
# by a crash is indistinguishable from one a human meant to add.
TEMP_PREFIX = "senapathy_tmp_"

# Exists ONLY so the sweep below can still find and drop an index a crashed
# run left behind under the product's FORMER name. Never used to name a new
# index -- `_temp_index_name()` only ever stamps `TEMP_PREFIX` -- so this
# constant has exactly one reader. Safe to delete once enough time has
# passed that no `perfagent_tmp_*` index can still be sitting on anyone's
# table; until then, removing it silently strands that index forever.
LEGACY_TEMP_PREFIX = "perfagent_tmp_"

# SELECTs are capped hard. A plan agent that pastes a cross join should get a
# tool error, not a stalled phase.
DEFAULT_TIMEOUT_MS = 10_000

# Index creation gets its own, larger budget: the spec's own worked example
# takes 11 seconds on a 1.2M-row table, and killing that at 10s would make the
# tool useless on exactly the tables it exists for.
DEFAULT_INDEX_TIMEOUT_S = 120.0

MAX_INDEX_COLUMNS = 8
MAX_DIFF_ROWS_SCANNED = 200_000

# `actual time=0.1..1642 rows=1120858 loops=1` out of an EXPLAIN ANALYZE tree.
_ACTUAL = re.compile(
    r"actual time=(?P<first>[\d.e+-]+)\.\.(?P<last>[\d.e+-]+)\s+"
    r"rows=(?P<rows>[\d.e+-]+)\s+loops=(?P<loops>[\d.e+-]+)")

# Comments and strings are stripped before the statement is classified, so
# `/* SELECT */ DROP TABLE x` is seen for what it is.
_LINE_COMMENT = re.compile(r"(--|#)[^\n]*")
_BLOCK_COMMENT = re.compile(r"/\*.*?\*/", re.S)


# ──────────────────────────────  statements  ─────────────────────────────


def strip_comments(sql: str) -> str:
    return _BLOCK_COMMENT.sub(" ", _LINE_COMMENT.sub(" ", sql or "")).strip()


def is_select(sql: str) -> bool:
    """One SELECT and nothing else.

    A CTE (`WITH ... SELECT`) is refused too. It is read-only and would be
    safe, but the spec says SELECT and a liberalisation here is the kind of
    thing that should be asked for rather than assumed.
    """
    body = strip_comments(sql)
    if not body:
        return False
    if ";" in body.rstrip().rstrip(";"):
        return False                      # more than one statement
    return body.lstrip("(").lstrip().lower().startswith("select")


def require_select(sql: str, *, what: str) -> str:
    body = strip_comments(sql)
    if not is_select(sql):
        raise ToolError(
            f"{what} accepts a single SELECT statement and nothing else. "
            f"Received: {body[:160]!r}.",
            suggestion="Pass the read query you want measured. DDL, DML and "
                       "multi-statement input are refused by this tool.",
        )
    return sql.strip().rstrip(";")


_INTEGER_PARAM = re.compile(r"^-?\d+$")


def bind_params(params: Sequence[str] | None) -> tuple[Any, ...]:
    """Params the way MySQL's own grammar needs them, not the way the tool
    schema types them.

    Every placeholder value arrives as a `str` -- `EXPLAIN_PARAMETERS` and
    `INDEX_BENEFIT_PARAMETERS` type `params` as `array of string`, because
    that is the only shape the model can supply. Handed straight to the
    connector, a Python `str` is rendered as a QUOTED literal regardless of
    where it lands -- correct for a WHERE-clause comparison ('ACH',
    '2023-06-01'), a syntax error for LIMIT/OFFSET, which MySQL's grammar
    accepts only as a bare integer, never a quoted one ("You have an error
    in your SQL syntax ... near ''50' OFFSET '0''" is this bug, not a
    malformed statement). A parameter that is purely digits is unambiguous
    in every query shape this tool sees; converting just those to `int`
    leaves every genuine string comparison exactly as quoted as before.
    """
    if not params:
        return ()
    return tuple(int(p) if _INTEGER_PARAM.match(p) else p for p in params)


def covering_index(indexes: Sequence[IndexInfo],
                   columns: Sequence[str]) -> str | None:
    """The existing index that already covers `columns` as a PREFIX, if any.

    Prefix, not set-equality: an index on (customer_id, status) serves a filter
    on customer_id alone, so proposing (customer_id) when that composite exists
    buys nothing. The reverse is not true, which is why the comparison is
    ordered — (customer_id) does NOT cover (customer_id, status).

    One definition, used by both this module and the plan phase's submit
    validator. Two copies of this rule would eventually disagree, and the
    disagreement would show up as a duplicate index in production.
    """
    wanted = [c.strip().lower() for c in columns if c and c.strip()]
    if not wanted:
        return None
    for index in indexes:
        have = [c.lower() for c in index.columns]
        if have[:len(wanted)] == wanted:
            return index.name
    return None


class LiveIndex(IndexInfo):
    """An index as the server currently has it, with its size on disk.

    `size_bytes` is not on `IndexInfo` because `IndexInfo` is an artifact
    shape and this is tool output — read by a model, never persisted. Adding a
    field to the contract for a number that never reaches an artifact would be
    inventing one.
    """

    size_bytes: int | None = None


@dataclass
class PlanReading:
    """One EXPLAIN ANALYZE, reduced to the numbers a decision needs."""

    tree: str
    access_types: dict[str, str] = field(default_factory=dict)
    keys_used: dict[str, str | None] = field(default_factory=dict)
    rows_examined: int = 0
    rows_returned: int = 0
    actual_ms: float = 0.0

    @property
    def any_index_used(self) -> bool:
        return any(key for key in self.keys_used.values())

    @property
    def uses_filesort(self) -> bool:
        """EXPLAIN ANALYZE's tree format says `-> Sort: ...`, never the word
        'filesort' — that spelling is classic EXPLAIN's "Extra" column, a
        different report this tool does not read. Verified directly against
        a real server: `ORDER BY amount LIMIT 10` with no serving index
        produces a `-> Sort: customer_payment.amount, limit input to ...`
        node, not the word 'filesort' anywhere in the tree."""
        return "-> sort:" in self.tree.lower()

    @property
    def uses_temporary(self) -> bool:
        """Two real spellings seen on a live server for a GROUP BY/ORDER BY
        combination neither an index nor the aggregation path alone can
        serve: `Table scan on <temporary>` and `Temporary table with
        deduplication`. Checking for '<temporary>' covers both — the second
        phrase always accompanies a temp table that also gets scanned."""
        return "<temporary>" in self.tree.lower()

    def summary(self) -> str:
        parts = []
        for table, access in self.access_types.items():
            key = self.keys_used.get(table) or "no index"
            parts.append(f"{table}: type={access}, key={key}")
        return " · ".join(parts) or "no table access reported"


@dataclass
class IndexBenefitResult:
    """`index_benefit`'s own measurement, structured rather than rendered.

    Exactly one of two shapes: `covered_by` set means the call was REFUSED
    (an index already serves this column set) and `before`/`after` are both
    None — there was nothing to measure. Otherwise `covered_by` is None and
    `before`/`after` carry the comparison (unless `timed_out`, in which case
    `after` stays None too — the index was killed mid-build).
    """

    table: str
    columns: list[str]
    index_name: str | None
    covered_by: str | None = None
    covered_by_columns: list[str] = field(default_factory=list)
    before: PlanReading | None = None
    after: PlanReading | None = None
    created_ms: float = 0.0
    size_bytes: int | None = None
    dropped: bool = True
    swept: list[str] = field(default_factory=list)
    timed_out: bool = False
    without_index_ms_samples: list[float] = field(default_factory=list)
    with_index_ms_samples: list[float] = field(default_factory=list)


def parse_tree(tree: str) -> tuple[int, int, float]:
    """(rows examined, rows returned, elapsed ms) out of an EXPLAIN ANALYZE tree.

    `rows=` on a node is rows PER LOOP, so the work a node did is rows × loops
    — that product is what distinguishes "42 rows, 50,000 times" from "42
    rows". Rows examined is the largest such product in the tree, which is the
    scan the query actually paid for; rows returned is the root node's own
    output. Both are read off the server's own report, never estimated here.
    """
    rows_examined = 0
    rows_returned = 0
    elapsed = 0.0
    for position, match in enumerate(_ACTUAL.finditer(tree)):
        try:
            rows = float(match.group("rows"))
            loops = float(match.group("loops"))
            last = float(match.group("last"))
        except ValueError:          # pragma: no cover - server formatting drift
            continue
        rows_examined = max(rows_examined, int(round(rows * loops)))
        if position == 0:           # the root node is printed first
            rows_returned = int(round(rows * loops))
            elapsed = last
    return rows_examined, rows_returned, elapsed


# ────────────────────────────────  the tools  ────────────────────────────


class DbToolbox:
    """The four database tools, bound to one measurement database.

    `allowed_tables` comes from `endpoint_model.tables` and is the containment
    rule for everything here: a tool that can index any table can index a
    table this run was never asked about.
    """

    def __init__(
        self,
        target: MySQLTarget,
        allowed_tables: Iterable[str],
        *,
        timeout_ms: int = DEFAULT_TIMEOUT_MS,
        index_timeout_s: float = DEFAULT_INDEX_TIMEOUT_S,
        sweep_on_start: bool = True,
    ) -> None:
        self.target = target
        self.allowed = [t for t in allowed_tables if t]
        self.timeout_ms = timeout_ms
        self.index_timeout_s = index_timeout_s
        self.swept: list[str] = []
        if sweep_on_start:
            # A crashed run leaves its temporary index behind. Sweeping at
            # construction means the next run cleans up after the last one
            # without anyone having to notice.
            self.swept = self.sweep()

    # ── containment ──────────────────────────────────────────────────────

    def _check_table(self, table: str) -> str:
        name = (table or "").strip().strip("`")
        if not re.fullmatch(r"\w+", name):
            raise ToolError(f"{table!r} is not a table name.")
        if name not in self.allowed:
            raise GuardrailError(
                f"{name!r} is not a table on this endpoint's path and was "
                f"refused. Tables this run may touch: "
                f"{', '.join(sorted(self.allowed)) or 'none'}.",
                suggestion="Work on the tables the endpoint model recorded.",
            )
        return name

    def _assert_measurement_database(self) -> None:
        """Refuse if an explicit SENAPATHY_MYSQL_* override names a different
        database than the one this toolbox was built for.

        This tool creates indexes. `from_env()` is an explicit override, not
        a default (see MySQLTarget) — when an operator has NOT set
        SENAPATHY_MYSQL_*, there is nothing to check `self.target` against,
        and it is trusted as built (from this run's own profile). When they
        HAVE set it, pointing this toolbox anywhere else is refused.
        """
        try:
            configured = MySQLTarget.from_env()
        except ToolError:
            return
        if (self.target.host, self.target.port, self.target.database) != (
                configured.host, configured.port, configured.database):
            raise GuardrailError(
                f"this toolbox points at {self.target.dsn()}, which is not the "
                f"database SENAPATHY_MYSQL_* names ({configured.dsn()}). "
                f"Creating an index anywhere else is refused.",
                suggestion="Unset SENAPATHY_MYSQL_* to use the run's own "
                           "detected database, or point it at this one.",
            )

    # ── shared plumbing ──────────────────────────────────────────────────

    def _cursor(self, conn: Any) -> Any:
        cursor = conn.cursor()
        cursor.execute(f"SET SESSION max_execution_time = {int(self.timeout_ms)}")
        cursor.execute(f"USE `{self.target.database}`")
        return cursor

    def _read_plan(self, cursor: Any, sql: str,
                   params: Sequence[str] | None) -> PlanReading:
        """Classic EXPLAIN for access types, EXPLAIN ANALYZE for real numbers.

        Both, because neither alone is enough: the tree carries actual rows and
        actual time but not the access type per table, and classic EXPLAIN
        carries the access type but only an optimizer ESTIMATE of rows.
        """
        arguments = bind_params(params)

        reading = PlanReading(tree="")
        cursor.execute(f"EXPLAIN {sql}", arguments)
        columns = [c[0].lower() for c in (cursor.description or [])]
        for row in cursor.fetchall():
            record = dict(zip(columns, row))
            table = str(record.get("table") or "?")
            reading.access_types[table] = str(record.get("type") or "?")
            key = record.get("key")
            reading.keys_used[table] = str(key) if key else None

        cursor.execute(f"EXPLAIN ANALYZE {sql}", arguments)
        reading.tree = "\n".join(str(row[0]) for row in cursor.fetchall())
        (reading.rows_examined, reading.rows_returned,
         reading.actual_ms) = parse_tree(reading.tree)
        return reading

    def _indexes(self, cursor: Any, table: str) -> list[LiveIndex]:
        cursor.execute(
            "SELECT INDEX_NAME, SEQ_IN_INDEX, COLUMN_NAME, NON_UNIQUE, "
            "CARDINALITY FROM information_schema.STATISTICS "
            "WHERE TABLE_SCHEMA = %s AND TABLE_NAME = %s "
            "ORDER BY INDEX_NAME, SEQ_IN_INDEX",
            (self.target.database, table),
        )
        found: dict[str, LiveIndex] = {}
        for name, _seq, column, non_unique, cardinality in cursor.fetchall():
            entry = found.get(str(name))
            if entry is None:
                entry = LiveIndex(
                    name=str(name), columns=[],
                    unique=not bool(int(non_unique or 0)),
                    cardinality=(int(cardinality)
                                 if cardinality is not None else None),
                )
                found[str(name)] = entry
            entry.columns.append(str(column))

        for name, entry in found.items():
            entry.size_bytes = self._index_size(cursor, table, name)
        return list(found.values())

    def _index_size(self, cursor: Any, table: str, index: str) -> int | None:
        """Pages × page size, from InnoDB's own statistics.

        Degrades to None rather than raising: the size is the cost side of a
        trade-off, and a user whose account cannot read `mysql.*` should still
        get the benefit side.
        """
        try:
            cursor.execute(
                "SELECT stat_value * @@innodb_page_size "
                "FROM mysql.innodb_index_stats "
                "WHERE database_name = %s AND table_name = %s "
                "AND index_name = %s AND stat_name = 'size'",
                (self.target.database, table, index),
            )
            row = cursor.fetchone()
        except Exception:  # noqa: BLE001 — a missing privilege is not a failure
            return None
        return int(row[0]) if row and row[0] is not None else None

    # ── 1. explain ───────────────────────────────────────────────────────

    def explain(self, sql: str, params: list[str] | None = None) -> str:
        """EXPLAIN ANALYZE one SELECT and report what it actually did."""
        reading = self.explain_measured(sql, params)
        return "\n".join([
            "PLAN (EXPLAIN ANALYZE — actual rows and actual time)",
            reading.tree,
            "",
            "SUMMARY",
            f"  access        {reading.summary()}",
            f"  rows examined {reading.rows_examined:,}"
            f"   ->  rows returned {reading.rows_returned:,}"
            + (f"   ({reading.rows_examined / reading.rows_returned:,.0f}x "
               f"read per row returned)"
               if reading.rows_returned else ""),
            f"  actual time   {reading.actual_ms:,.1f} ms",
            f"  index used    {'yes' if reading.any_index_used else 'NO'}",
        ])

    # ── verify a candidate's own before/after claim, read-only ───────────

    def explain_measured(self, sql: str,
                        params: list[str] | None = None) -> PlanReading:
        """The structured `PlanReading` behind `explain()`'s rendered text —
        for code that needs the numbers themselves (the SQL audit's
        classifier), not a model-facing report. Same SELECT-only guardrail,
        same single EXPLAIN ANALYZE; `explain()` is now a thin renderer over
        this."""
        statement = require_select(sql, what="explain_measured")
        with connect(self.target) as conn:
            cursor = self._cursor(conn)
            try:
                return self._read_plan(cursor, statement, params)
            except Exception as exc:  # noqa: BLE001 — the caller must read this
                raise _explain_error(exc, statement) from exc
            finally:
                cursor.close()

    def table_write_volume(self, table: str) -> dict[str, int] | None:
        """{'insert': n, 'update': n, 'delete': n, 'fetch': n} for `table`,
        from `performance_schema.table_io_waits_summary_by_table` —
        CUMULATIVE since the server's statistics were last reset, not a
        rate. An index is not free: it slows every INSERT and UPDATE on the
        table it sits on, and EXPLAIN never shows that cost. This is the
        write side of that trade-off, read the same way `_index_size`
        reads the size side — both degrade to None rather than raising,
        because an account that cannot read performance_schema should
        still get whatever evidence IS available, not a hard failure.
        """
        name = self._check_table(table)
        try:
            with connect(self.target) as conn:
                cursor = conn.cursor()
                try:
                    cursor.execute(
                        "SELECT COUNT_INSERT, COUNT_UPDATE, COUNT_DELETE, "
                        "COUNT_FETCH FROM "
                        "performance_schema.table_io_waits_summary_by_table "
                        "WHERE OBJECT_SCHEMA = %s AND OBJECT_NAME = %s",
                        (self.target.database, name),
                    )
                    row = cursor.fetchone()
                finally:
                    cursor.close()
        except Exception:  # noqa: BLE001 — a missing privilege is not a failure
            return None
        if row is None:
            return None
        insert, update, delete, fetch = row
        return {"insert": int(insert or 0), "update": int(update or 0),
               "delete": int(delete or 0), "fetch": int(fetch or 0)}

    def plan_delta(self, sql_before: str, sql_after: str,
                   params: list[str] | None = None) -> tuple[PlanReading, PlanReading]:
        """EXPLAIN ANALYZE both statements on one connection. Nothing created,
        nothing dropped — unlike `index_benefit`, there is no index here to
        leave behind, so the only safety properties that apply are the ones
        `explain` already has: SELECT-only, and the session timeout set in
        `_cursor`.

        Called from the plan phase's candidate validation (`plan/submit.py`),
        not exposed to the model as a tool — this is code independently
        checking a candidate's claim about its own `sql_after`, not the agent
        exploring.
        """
        before_sql = require_select(sql_before, what="plan_delta (sql_before)")
        after_sql = require_select(sql_after, what="plan_delta (sql_after)")
        with connect(self.target) as conn:
            cursor = self._cursor(conn)
            try:
                before = self._read_plan(cursor, before_sql, params)
                after = self._read_plan(cursor, after_sql, params)
            except Exception as exc:  # noqa: BLE001 — the caller must read this
                raise _explain_error(exc, f"{before_sql}  /  {after_sql}") from exc
            finally:
                cursor.close()
        return before, after

    def benchmark_interleaved(
        self, sql_a: str, sql_b: str, params: list[str] | None = None,
        *, rounds: int = 5,
    ) -> tuple[list[float], list[float]]:
        """Wall-clock timings for two statements, interleaved (A, B, A, B,
        ...), never sequentially. A sequential A-then-B comparison credits
        B with whatever buffer-pool warming A already paid for; alternating
        gives both statements the same cold/warm mix across the run.
        Read-only, nothing created or dropped — same safety properties as
        `explain`/`plan_delta`: SELECT-only, one connection, the session
        timeout `_cursor` already sets.

        Returns `(a_timings_ms, b_timings_ms)`, `rounds` entries each, in
        the order they were measured. The caller decides what to do with
        the distribution (median is what `audit/candidates.py` uses) —
        this method measures, it does not summarise.
        """
        a_sql = require_select(sql_a, what="benchmark_interleaved (sql_a)")
        b_sql = require_select(sql_b, what="benchmark_interleaved (sql_b)")
        bound = bind_params(params)
        with connect(self.target) as conn:
            cursor = self._cursor(conn)
            try:
                return self._interleave(cursor, a_sql, b_sql, bound, rounds)
            except Exception as exc:  # noqa: BLE001 — the caller must read this
                raise _explain_error(exc, f"{a_sql}  /  {b_sql}") from exc
            finally:
                cursor.close()

    def _interleave(
        self, cursor: Any, a_sql: str, b_sql: str, bound: tuple[Any, ...],
        rounds: int,
    ) -> tuple[list[float], list[float]]:
        """The actual A,B,A,B,... loop, on a cursor the caller already
        opened — shared by `benchmark_interleaved` (opens its own
        connection) and `index_benefit_measured` (reuses the connection
        that just built the candidate index, so the benchmark runs WHILE
        it exists rather than needing a second create/drop cycle)."""
        a_timings: list[float] = []
        b_timings: list[float] = []
        for _ in range(max(1, rounds)):
            started = time.perf_counter()
            cursor.execute(a_sql, bound)
            cursor.fetchall()
            a_timings.append((time.perf_counter() - started) * 1000.0)

            started = time.perf_counter()
            cursor.execute(b_sql, bound)
            cursor.fetchall()
            b_timings.append((time.perf_counter() - started) * 1000.0)
        return a_timings, b_timings

    # ── 2. list_indexes ──────────────────────────────────────────────────

    def list_indexes(self, table: str) -> str:
        """Every index on one table, with its columns IN ORDER."""
        name = self._check_table(table)
        with connect(self.target) as conn:
            cursor = self._cursor(conn)
            try:
                indexes = self._indexes(cursor, name)
            finally:
                cursor.close()

        if not indexes:
            return f"{name}: no indexes at all, not even a primary key."

        out = [f"INDEXES ON {name}", ""]
        for index in sorted(indexes, key=lambda i: i.name):
            kind = "UNIQUE" if index.unique else "index"
            out.append(f"  {kind} {index.name}({', '.join(index.columns)})")
            out.append(
                f"    column order is significant: this index serves a filter "
                f"on {index.columns[0]}"
                + (f", or on {', '.join(index.columns[:2])} together"
                   if len(index.columns) > 1 else "")
                + ", and not one on a later column alone."
            )
            out.append(f"    cardinality {index.cardinality:,}"
                       if index.cardinality is not None
                       else "    cardinality unknown")
            if index.size_bytes is not None:
                out.append(f"    size {_megabytes(index.size_bytes)}")
            out.append("")
        return "\n".join(out)

    # ── 3. index_benefit ─────────────────────────────────────────────────

    def index_benefit(self, table: str, columns: list[str], sql: str,
                      params: list[str] | None = None) -> str:
        """Measure what an index WOULD be worth, then leave no trace.

        The index is created, the statement re-measured, and the index dropped
        in a `finally` — on success, on failure, and on timeout. Thin wrapper
        over `index_benefit_measured`, which does the actual work and hands
        back the numbers rather than rendered text — the SQL audit phase
        calls that one directly rather than parsing this string back apart.
        """
        result = self.index_benefit_measured(table, columns, sql, params)
        if result.covered_by is not None:
            return (
                f"REFUSED: {result.table}({', '.join(result.columns)}) is "
                f"already covered by the existing index {result.covered_by}"
                f"({', '.join(result.covered_by_columns)}). Creating it "
                f"again costs write throughput and buys nothing. Stop here "
                f"and measure something else."
            )
        if result.timed_out:
            raise ToolError(
                f"creating {result.index_name} on {result.table}"
                f"({', '.join(result.columns)}) exceeded the "
                f"{self.index_timeout_s:.0f}s ceiling and was killed. The "
                f"index was dropped"
                f"{'' if result.dropped else ' — VERIFY MANUALLY, the drop also failed'}.",
                suggestion="Measure a narrower index, or accept that an index "
                           "this expensive to build is itself a cost.",
            )
        assert result.before is not None and result.after is not None
        return _render_benefit(
            table=result.table, columns=result.columns,
            index_name=result.index_name, before=result.before,
            after=result.after, created_ms=result.created_ms,
            size_bytes=result.size_bytes, dropped=result.dropped,
            swept=result.swept,
        )

    def index_benefit_measured(
        self, table: str, columns: list[str], sql: str,
        params: list[str] | None = None, *,
        benchmark_hint_position: int | None = None,
        benchmark_rounds: int = 0,
    ) -> "IndexBenefitResult":
        """`index_benefit`'s own measurement, as numbers rather than text.

        Not exposed to the model as a tool — called from code (the SQL audit
        phase) the same way `plan_delta` is: a caller that already has its
        own reason to run this and needs the structured result, not a model
        exploring. Same safety properties as `index_benefit`: additive only,
        `senapathy_tmp_*` naming, dropped in `finally`, swept at
        construction, KILL QUERY watchdog on timeout.

        `benchmark_hint_position` (an offset into `sql`, from
        `audit.reconstruct.table_reference_end`) turns on an INTERLEAVED
        real-execution benchmark, run WHILE the candidate index still
        exists — the only way to measure "with it" against "without it"
        without a second create/drop cycle. `sql` with an `IGNORE INDEX
        (...)` clause spliced in at that position is the "without" side;
        `sql` unchanged is the "with" side, since the optimizer picks the
        index it was just given automatically. Skipped (both samples lists
        stay empty) when no position is given, when `benchmark_rounds` is
        0, or when the index could not be built at all.
        """
        self._assert_measurement_database()
        name = self._check_table(table)
        wanted = _clean_columns(columns)
        statement = require_select(sql, what="index_benefit")

        swept = self.sweep()
        with connect(self.target) as conn:
            cursor = self._cursor(conn)
            try:
                existing = self._indexes(cursor, name)
                covered = covering_index(existing, wanted)
                if covered is not None:
                    return IndexBenefitResult(
                        table=name, columns=wanted, index_name=None,
                        covered_by=covered,
                        covered_by_columns=_columns_of(existing, covered),
                        swept=swept,
                    )
                _check_columns_exist(cursor, self.target.database, name, wanted)

                before = self._read_plan(cursor, statement, params)
                index_name = _temp_index_name(name, wanted)

                created_ms = 0.0
                size_bytes: int | None = None
                after: PlanReading | None = None
                timed_out = False
                without_samples: list[float] = []
                with_samples: list[float] = []
                try:
                    started = time.perf_counter()
                    timed_out = self._create_index(conn, cursor, index_name,
                                                   name, wanted)
                    created_ms = (time.perf_counter() - started) * 1000.0
                    if not timed_out:
                        after = self._read_plan(cursor, statement, params)
                        size_bytes = self._index_size(cursor, name, index_name)
                        if benchmark_hint_position is not None and benchmark_rounds > 0:
                            without_sql = require_select(
                                statement[:benchmark_hint_position]
                                + f" IGNORE INDEX (`{index_name}`)"
                                + statement[benchmark_hint_position:],
                                what="index_benefit (benchmark without-index variant)",
                            )
                            without_samples, with_samples = self._interleave(
                                cursor, without_sql, statement,
                                bind_params(params), benchmark_rounds,
                            )
                finally:
                    # The one line this whole tool is organised around.
                    dropped = self._drop_index(cursor, name, index_name)
            finally:
                cursor.close()

        return IndexBenefitResult(
            table=name, columns=wanted, index_name=index_name,
            before=before, after=after, created_ms=created_ms,
            size_bytes=size_bytes, dropped=dropped, swept=swept,
            timed_out=timed_out, without_index_ms_samples=without_samples,
            with_index_ms_samples=with_samples,
        )

    def _create_index(self, conn: Any, cursor: Any, index_name: str,
                      table: str, columns: Sequence[str]) -> bool:
        """CREATE INDEX with a watchdog. True if it had to be killed.

        `max_execution_time` governs SELECTs only — it will not stop a DDL
        statement — so the ceiling is enforced by a timer that KILLs the query
        from a second connection. Without it "hard timeout" would mean "we
        hope it finishes".
        """
        statement = (f"CREATE INDEX `{index_name}` ON `{table}` "
                     f"({', '.join('`' + c + '`' for c in columns)})")
        killer = _Watchdog(self.target, _connection_id(cursor),
                           self.index_timeout_s)
        killer.start()
        try:
            cursor.execute(statement)
        except Exception as exc:  # noqa: BLE001
            if killer.fired:
                return True
            raise ToolError(
                f"could not create {index_name} on {table}"
                f"({', '.join(columns)}): {exc}"
            ) from exc
        finally:
            killer.cancel()
        return killer.fired

    def _drop_index(self, cursor: Any, table: str, index_name: str) -> bool:
        """Drop the temporary index. Only ever one with our prefix."""
        if not index_name.startswith(TEMP_PREFIX):   # unreachable by design
            raise GuardrailError(
                f"refusing to drop {index_name!r}: this tool only ever drops "
                f"indexes it created, which all begin {TEMP_PREFIX!r}."
            )
        try:
            cursor.execute(f"DROP INDEX `{index_name}` ON `{table}`")
            return True
        except Exception:  # noqa: BLE001 — nothing may mask the measurement
            return False

    # ── 4. query_equivalent ──────────────────────────────────────────────

    def query_equivalent(self, original_sql: str, candidate_sql: str,
                         params: list[str] | None = None) -> str:
        """Do two SELECTs return the same rows? Code decides, not the model."""
        original = require_select(original_sql, what="query_equivalent")
        candidate = require_select(candidate_sql, what="query_equivalent")
        ordered = _has_order_by(original)

        with connect(self.target) as conn:
            cursor = self._cursor(conn)
            try:
                left, left_ms = _run(cursor, original, params)
                right, right_ms = _run(cursor, candidate, params)
            finally:
                cursor.close()

        difference = _first_difference(left, right, ordered=ordered)
        verdict = "EQUIVALENT" if difference is None else "NOT EQUIVALENT"
        out = [
            f"{verdict}",
            f"  original   {len(left):,} row(s) in {left_ms:,.1f} ms",
            f"  candidate  {len(right):,} row(s) in {right_ms:,.1f} ms",
            f"  compared   {'in order' if ordered else 'as unordered sets'} "
            f"({_why_ordered(ordered)})",
        ]
        if difference is not None:
            out.append(f"  DIFFERENCE {difference}")
            out.append("  A rewrite that returns different rows is not a "
                       "rewrite. The oracle would reject this.")
        else:
            out.append(f"  speedup    {left_ms / right_ms:,.2f}x"
                       if right_ms else "  speedup    not measurable")
        return "\n".join(out)

    # ── the sweep ────────────────────────────────────────────────────────

    def sweep(self) -> list[str]:
        """Drop every `senapathy_tmp_*` OR `perfagent_tmp_*` index left on an
        allowed table.

        A run killed between CREATE and DROP leaves one behind. Sweeping here
        means the next call cleans it up, and the prefix is what makes that
        safe: nothing without either prefix is ever touched. Both prefixes
        are checked so that an index a crashed run leaked under the
        product's FORMER name (`perfagent_tmp_*`) is not orphaned forever by
        the rename — a sweep that only recognised the new name would never
        find it again.
        """
        dropped: list[str] = []
        if not self.allowed:
            return dropped
        marks = ", ".join(["%s"] * len(self.allowed))
        with connect(self.target) as conn:
            cursor = conn.cursor()
            try:
                cursor.execute(f"USE `{self.target.database}`")
                cursor.execute(
                    f"SELECT DISTINCT TABLE_NAME, INDEX_NAME "
                    f"FROM information_schema.STATISTICS "
                    f"WHERE TABLE_SCHEMA = %s AND TABLE_NAME IN ({marks}) "
                    f"AND (INDEX_NAME LIKE %s OR INDEX_NAME LIKE %s)",
                    (self.target.database, *self.allowed,
                     TEMP_PREFIX + "%", LEGACY_TEMP_PREFIX + "%"),
                )
                for table, index in cursor.fetchall():
                    try:
                        cursor.execute(
                            f"DROP INDEX `{index}` ON `{table}`")
                        dropped.append(f"{table}.{index}")
                    except Exception:  # noqa: BLE001 — report what we managed
                        continue
            finally:
                cursor.close()
        return dropped


# ─────────────────────────────  small helpers  ───────────────────────────


class _Watchdog:
    """KILL QUERY from a second connection after `seconds`."""

    def __init__(self, target: MySQLTarget, connection_id: int | None,
                 seconds: float) -> None:
        self.target = target
        self.connection_id = connection_id
        self.fired = False
        self._timer = threading.Timer(seconds, self._kill)
        self._timer.daemon = True

    def start(self) -> None:
        if self.connection_id is not None:
            self._timer.start()

    def cancel(self) -> None:
        self._timer.cancel()

    def _kill(self) -> None:
        if self.connection_id is None:   # `start` never arms the timer then
            return
        self.fired = True
        try:
            with connect(self.target) as conn:
                cursor = conn.cursor()
                try:
                    cursor.execute(f"KILL QUERY {int(self.connection_id)}")
                finally:
                    cursor.close()
        except Exception:  # noqa: BLE001 — the drop in `finally` is the backstop
            pass


def _connection_id(cursor: Any) -> int | None:
    try:
        cursor.execute("SELECT CONNECTION_ID()")
        row = cursor.fetchone()
        return int(row[0]) if row else None
    except Exception:  # noqa: BLE001
        return None


def _temp_index_name(table: str, columns: Sequence[str]) -> str:
    digest = hashlib.sha1(
        f"{table}:{','.join(columns)}".encode()).hexdigest()[:10]
    return f"{TEMP_PREFIX}{digest}"


def _clean_columns(columns: Sequence[str]) -> list[str]:
    wanted = [str(c).strip().strip("`") for c in (columns or []) if str(c).strip()]
    if not wanted:
        raise ToolError("index_benefit needs at least one column.")
    if len(wanted) > MAX_INDEX_COLUMNS:
        raise ToolError(
            f"{len(wanted)} columns is not an index, it is a copy of the "
            f"table. The ceiling is {MAX_INDEX_COLUMNS}.")
    for column in wanted:
        if not re.fullmatch(r"\w+", column):
            raise ToolError(f"{column!r} is not a column name.")
    if len(set(wanted)) != len(wanted):
        raise ToolError(f"duplicate column in {wanted}.")
    return wanted


def _check_columns_exist(cursor: Any, database: str, table: str,
                         columns: Sequence[str]) -> None:
    cursor.execute(
        "SELECT COLUMN_NAME FROM information_schema.COLUMNS "
        "WHERE TABLE_SCHEMA = %s AND TABLE_NAME = %s",
        (database, table),
    )
    present = {str(row[0]).lower() for row in cursor.fetchall()}
    missing = [c for c in columns if c.lower() not in present]
    if missing:
        raise ToolError(
            f"{table} has no column(s) {', '.join(missing)}. Columns: "
            f"{', '.join(sorted(present))}.")


def _columns_of(indexes: Sequence[IndexInfo], name: str) -> list[str]:
    for index in indexes:
        if index.name == name:
            return list(index.columns)
    return []


def _megabytes(size_bytes: int) -> str:
    return f"{size_bytes / (1024 * 1024):,.1f} MB"


def _explain_error(exc: Exception, statement: str) -> ToolError:
    text = str(exc)
    if "max_execution_time" in text.lower() or "3024" in text:
        return ToolError(
            f"the statement exceeded the {DEFAULT_TIMEOUT_MS / 1000:.0f}s "
            f"ceiling and was killed: {text}",
            suggestion="Measure a narrower query, or add a WHERE that matches "
                       "what the endpoint actually sends.")
    return ToolError(f"EXPLAIN failed: {text}\nStatement: {statement[:300]}")


def _run(cursor: Any, sql: str,
         params: Sequence[str] | None) -> tuple[list[tuple[Any, ...]], float]:
    started = time.perf_counter()
    cursor.execute(sql, bind_params(params))
    rows = [tuple(row) for row in cursor.fetchall()]
    return rows, (time.perf_counter() - started) * 1000.0


def _why_ordered(ordered: bool) -> str:
    return ("the original has an ORDER BY, so row order is part of what it "
            "promises" if ordered else
            "the original has no ORDER BY, so row order is not part of its "
            "contract and is not compared")


def _has_order_by(sql: str) -> bool:
    return re.search(r"\border\s+by\b", strip_comments(sql), re.I) is not None


def _first_difference(left: list[tuple[Any, ...]], right: list[tuple[Any, ...]],
                      *, ordered: bool) -> str | None:
    """The first way these two result sets differ, in words. Computed, never
    asked of a model (invariant 5)."""
    if len(left) != len(right):
        return (f"row count: the original returns {len(left):,} row(s), the "
                f"candidate {len(right):,}")

    if ordered:
        for position, (a, b) in enumerate(zip(left, right)):
            if a != b:
                return (f"row {position + 1} differs: original {a!r}, "
                        f"candidate {b!r}")
        return None

    counted_left = _multiset(left)
    counted_right = _multiset(right)
    for row, count in counted_left.items():
        other = counted_right.get(row, 0)
        if other != count:
            return (f"row {row!r} appears {count}x in the original and "
                    f"{other}x in the candidate")
    for row, count in counted_right.items():
        if row not in counted_left:
            return f"the candidate returns {row!r}, which the original does not"
    return None


def _multiset(rows: list[tuple[Any, ...]]) -> dict[tuple[str, ...], int]:
    """Rows as comparable keys. `Decimal('1.0')` and `1.0` are not the same
    value to the oracle, so they are not the same value here either."""
    counted: dict[tuple[str, ...], int] = {}
    for row in rows:
        key = tuple(f"{type(v).__name__}:{v!r}" for v in row)
        counted[key] = counted.get(key, 0) + 1
    return counted


def _render_benefit(*, table: str, columns: Sequence[str], index_name: str,
                    before: PlanReading, after: PlanReading, created_ms: float,
                    size_bytes: int | None, dropped: bool,
                    swept: Sequence[str]) -> str:
    reduction = (
        100.0 * (before.rows_examined - after.rows_examined) / before.rows_examined
        if before.rows_examined else 0.0
    )
    speedup = (before.actual_ms / after.actual_ms) if after.actual_ms else 0.0

    out = [f"index_benefit('{table}', {list(columns)})", ""]
    if swept:
        out.append(f"  SWEPT    {len(swept)} orphaned index(es) from an earlier "
                   f"run: {', '.join(swept)}")
    out.extend([
        f"  BEFORE   {before.summary()}",
        f"           rows_examined={before.rows_examined:,}   "
        f"actual={before.actual_ms:,.1f} ms",
        f"  AFTER    {after.summary()}",
        f"           rows_examined={after.rows_examined:,}   "
        f"actual={after.actual_ms:,.1f} ms",
        f"  COST     creation {created_ms / 1000:,.1f} s"
        + (f" · estimated size {_megabytes(size_bytes)}"
           if size_bytes is not None else " · size unmeasured"),
        f"  VERDICT  rows examined {'reduced' if reduction >= 0 else 'INCREASED'} "
        f"{abs(reduction):.3f}%"
        + (f", {speedup:,.1f}x faster" if speedup > 1 else ""),
        "",
        f"  The index was temporary ({index_name}) and has been "
        + ("dropped. Nothing was left on the schema."
           if dropped else
           "LEFT BEHIND — the drop failed. The next call will sweep it."),
    ])
    return "\n".join(out)


# ──────────────────────────────  the factory  ────────────────────────────

EXPLAIN_PARAMETERS: dict[str, Any] = {
    "type": "object",
    "properties": {
        "sql": {"type": "string",
                "description": "One SELECT statement. Use %s for parameters."},
        "params": {"type": "array", "items": {"type": "string"},
                   "description": "Values for the %s placeholders, in order."},
    },
    "required": ["sql"],
}

LIST_INDEXES_PARAMETERS: dict[str, Any] = {
    "type": "object",
    "properties": {
        "table": {"type": "string", "description": "Physical table name."},
    },
    "required": ["table"],
}

INDEX_BENEFIT_PARAMETERS: dict[str, Any] = {
    "type": "object",
    "properties": {
        "table": {"type": "string", "description": "Physical table name."},
        "columns": {
            "type": "array", "items": {"type": "string"},
            "description": "Columns IN INDEX ORDER. Order decides what the "
                           "index can serve: (a, b) helps a filter on a, or on "
                           "a and b, and never one on b alone.",
        },
        "sql": {"type": "string",
                "description": "The SELECT to measure, with %s parameters."},
        "params": {"type": "array", "items": {"type": "string"},
                   "description": "Values for the %s placeholders, in order."},
    },
    "required": ["table", "columns", "sql"],
}

QUERY_EQUIVALENT_PARAMETERS: dict[str, Any] = {
    "type": "object",
    "properties": {
        "original_sql": {"type": "string",
                         "description": "The query as it is today."},
        "candidate_sql": {"type": "string",
                          "description": "Your rewrite of it."},
        "params": {"type": "array", "items": {"type": "string"},
                   "description": "Values for the %s placeholders. Both "
                                  "statements are run with the same ones."},
    },
    "required": ["original_sql", "candidate_sql"],
}

# Which phase gets what. Diagnosis explains what IS; it does not try fixes, so
# it has no way to create an index. Planning may measure one before proposing
# it. Only the patch agent, which is already writing code, gets all four.
DIAGNOSE_TOOLS: tuple[str, ...] = ("explain", "list_indexes")
PLAN_TOOLS: tuple[str, ...] = ("explain", "list_indexes", "index_benefit")
PATCH_TOOLS: tuple[str, ...] = ("explain", "list_indexes", "index_benefit",
                                "query_equivalent")


def db_tools(target: MySQLTarget, allowed_tables: Iterable[str], *,
             names: Iterable[str] = PATCH_TOOLS,
             toolbox: DbToolbox | None = None) -> list[ToolSpec]:
    """The database tools a phase is allowed to have.

    `names` is explicit at every call site rather than defaulted per phase:
    which phase may create an index is a design decision, and it should be
    readable where the registry is built.
    """
    box = toolbox or DbToolbox(target, allowed_tables)
    available = {
        "explain": ToolSpec(
            name="explain",
            description=(
                "EXPLAIN ANALYZE a SELECT against the real database: the plan "
                "with ACTUAL rows and ACTUAL timings, plus the access type per "
                "table, rows examined versus rows returned, and whether any "
                "index was used. This runs the query — it is a measurement, "
                "not an estimate."
            ),
            parameters=EXPLAIN_PARAMETERS,
            handler=box.explain,
        ),
        "list_indexes": ToolSpec(
            name="list_indexes",
            description=(
                "Every index on a table, with its columns IN ORDER, whether it "
                "is unique, its cardinality and its size on disk. Column order "
                "decides which filters an index can serve, so read it before "
                "proposing anything."
            ),
            parameters=LIST_INDEXES_PARAMETERS,
            handler=box.list_indexes,
        ),
        "index_benefit": ToolSpec(
            name="index_benefit",
            description=(
                "MEASURE what an index would be worth before proposing it. "
                "Creates the index temporarily, re-runs EXPLAIN ANALYZE, drops "
                "it again, and reports rows examined and actual time before "
                "and after, with the creation time and index size as the cost. "
                "Refuses if an existing index already covers those columns, "
                "and names it. The index never survives the call."
            ),
            parameters=INDEX_BENEFIT_PARAMETERS,
            handler=box.index_benefit,
        ),
        "query_equivalent": ToolSpec(
            name="query_equivalent",
            description=(
                "Run two SELECTs and compare their result sets exactly — same "
                "rows, same values. Order-insensitive unless the original has "
                "an ORDER BY. Reports the first difference. A rewrite that "
                "returns different rows is not a rewrite, and this is how you "
                "find that out before the behaviour oracle does."
            ),
            parameters=QUERY_EQUIVALENT_PARAMETERS,
            handler=box.query_equivalent,
        ),
    }

    wanted = list(names)
    unknown = [n for n in wanted if n not in available]
    if unknown:
        raise ValueError(f"unknown database tool(s): {', '.join(unknown)}")
    return [available[name] for name in wanted]


def optional_db_tools(target: MySQLTarget | None, allowed_tables: Iterable[str],
                      *, names: Iterable[str],
                      on_error: Any = None) -> list[ToolSpec]:
    """`db_tools`, or an empty list and a note when the database is unreachable.

    Constructing a toolbox runs the orphan sweep, which is a round trip. A
    phase that can still do most of its job from source must not die because
    MySQL is down — so the failure is reported through `on_error` and the
    phase runs with a smaller tool surface.
    """
    if target is None:
        return []
    try:
        return db_tools(target, allowed_tables, names=names)
    except Exception as exc:  # noqa: BLE001 — a missing database is degraded, not dead
        if on_error is not None:
            on_error(f"the database tools are unavailable "
                     f"({type(exc).__name__}: {exc}); running without "
                     f"{', '.join(names)}.")
        return []
