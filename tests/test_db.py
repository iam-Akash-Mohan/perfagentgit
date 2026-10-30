"""The database toolkit. No database: `connect` is replaced by a fake server.

The tests that matter most are the ones about the index that gets created.
`index_benefit` is the only thing in this product that issues DDL, so the
drop-in-`finally`, the name prefix, the table allowlist, the sweep and the
measurement-database check each get a test that fails if the guardrail is
removed — including the case where everything after the CREATE blows up.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

import pytest

from perfagent.contracts import IndexInfo
from perfagent.errors import GuardrailError, ToolError
from perfagent.measure.sql import MySQLTarget
from perfagent.plan.migration import (existing_migration_for,
                                      find_migration_dir, index_name,
                                      next_version, uses_flyway,
                                      write_index_migration)
from perfagent.tools import db as dbmod
from perfagent.tools.db import (DIAGNOSE_TOOLS, PATCH_TOOLS, PLAN_TOOLS,
                                TEMP_PREFIX, DbToolbox, covering_index,
                                db_tools, is_select, optional_db_tools,
                                parse_tree, require_select)

TREE = """-> Aggregate: sum(i.amount)  (cost=2446 rows=1) (actual time=176..176 rows=1 loops=1)
    -> Filter: (i.`status` = 'OPEN')  (cost=2420 rows=265) (actual time=11.4..176 rows=571 loops=1)
        -> Index lookup on i using idx_invoice_customer_id  (cost=2420 rows=2645) (actual time=11.4..176 rows=2645 loops=1)
"""

FAST_TREE = """-> Aggregate: sum(i.amount)  (cost=10 rows=1) (actual time=0.9..0.9 rows=1 loops=1)
    -> Index lookup on i using perfagent_tmp_x  (cost=8 rows=571) (actual time=0.1..0.8 rows=571 loops=1)
"""

SQL = "select sum(amount) from invoice where customer_id = %s and status = 'OPEN'"

EXPLAIN_COLUMNS = ("id", "select_type", "table", "type", "possible_keys",
                   "key", "rows", "filtered", "Extra")


# ─────────────────────────────  the fake server  ─────────────────────────


class FakeCursor:
    """Answers by first matching substring, in the order they were declared."""

    def __init__(self, server: "FakeServer") -> None:
        self.server = server
        self.description: list[tuple[str, ...]] | None = None
        self._rows: list[tuple[Any, ...]] = []

    def execute(self, sql: str, params: Any = ()) -> None:
        self.server.executed.append(sql)
        for pattern, error in self.server.errors.items():
            if pattern.lower() in sql.lower():
                raise error
        self._rows = []
        self.description = None
        for pattern, rows in self.server.rows.items():
            if pattern.lower() in sql.lower():
                self._rows = list(rows)
                if pattern == "EXPLAIN " and rows:
                    self.description = [(c,) for c in EXPLAIN_COLUMNS]
                break

    def fetchall(self) -> list[tuple[Any, ...]]:
        return self._rows

    def fetchone(self) -> tuple[Any, ...] | None:
        return self._rows[0] if self._rows else None

    def close(self) -> None:
        pass


class FakeServer:
    def __init__(self, rows: dict[str, list[tuple[Any, ...]]] | None = None,
                 errors: dict[str, Exception] | None = None) -> None:
        # Longest-first so "EXPLAIN ANALYZE" wins over "EXPLAIN ".
        self.rows = rows or {}
        self.errors = errors or {}
        self.executed: list[str] = []

    def cursor(self) -> FakeCursor:
        return FakeCursor(self)

    def ran(self, fragment: str) -> list[str]:
        return [s for s in self.executed if fragment.lower() in s.lower()]


def script(*, indexes: list[tuple[str, int, str, int, int]] | None = None,
           trees: list[str] | None = None,
           orphans: list[tuple[str, str]] | None = None,
           columns: list[str] | None = None,
           rows: list[tuple[Any, ...]] | None = None) -> FakeServer:
    """A server with just enough answers for one toolbox call."""
    trees = trees or [TREE]
    # Order matters: the fake answers the FIRST pattern that matches, and the
    # sweep's query contains both "INDEX_NAME LIKE" and
    # "information_schema.STATISTICS". Most specific first.
    server = FakeServer({
        "INDEX_NAME LIKE": orphans or [],
        "EXPLAIN ANALYZE": [(trees[0],)],
        "EXPLAIN ": [(1, "SIMPLE", "i", "ref", "idx_invoice_customer_id",
                      "idx_invoice_customer_id", 2645, 10.0, "")],
        "information_schema.STATISTICS": indexes if indexes is not None else [],
        "information_schema.COLUMNS": [(c,) for c in
                                       (columns or ["id", "customer_id",
                                                    "amount", "status"])],
        "innodb_index_stats": [(38_000_000,)],
        "CONNECTION_ID": [(7,)],
        "select": rows if rows is not None else [(1,)],
    })
    server._trees = list(trees)          # type: ignore[attr-defined]
    return server


@pytest.fixture
def connected(monkeypatch):
    """Replace `connect` with one that yields a scripted server."""

    def install(server: FakeServer):
        from contextlib import contextmanager

        @contextmanager
        def fake_connect(target: MySQLTarget) -> Iterator[FakeServer]:
            yield server

        monkeypatch.setattr(dbmod, "connect", fake_connect)
        return server

    return install


def toolbox(server: FakeServer, connected, **kwargs: Any) -> DbToolbox:
    connected(server)
    kwargs.setdefault("sweep_on_start", False)
    return DbToolbox(MySQLTarget.from_env(), ["customer", "invoice"], **kwargs)


# ───────────────────────────  statement guards  ──────────────────────────


@pytest.mark.parametrize("sql", [
    "select 1",
    "  SELECT a FROM t WHERE x = %s  ",
    "(select 1)",
    "-- a comment\nselect 1",
    "select 1;",
])
def test_a_select_is_accepted(sql: str) -> None:
    assert is_select(sql)


@pytest.mark.parametrize("sql", [
    "drop table invoice",
    "update invoice set amount = 0",
    "insert into invoice values (1)",
    "with x as (select 1) select * from x",     # read-only, still refused
    "select 1; drop table invoice",             # the classic
    "/* select */ drop table invoice",          # a comment is not a statement
    "",
])
def test_anything_else_is_refused(sql: str) -> None:
    assert not is_select(sql)
    with pytest.raises(ToolError, match="single SELECT"):
        require_select(sql, what="explain")


def test_explain_refuses_ddl_before_it_reaches_the_server(connected) -> None:
    server = script()
    box = toolbox(server, connected)
    with pytest.raises(ToolError, match="single SELECT"):
        box.explain("DROP INDEX idx_invoice_customer_id ON invoice")
    assert server.executed == [], "nothing may reach the server"


# ──────────────────────────────  plan reading  ───────────────────────────


def test_rows_examined_is_rows_times_loops() -> None:
    """42 rows fetched 50,000 times is not 42 rows."""
    tree = ("-> Nested loop  (actual time=1..900 rows=50000 loops=1)\n"
            "    -> Index lookup on i  (actual time=0.01..0.02 rows=42 loops=50000)")
    examined, returned, elapsed = parse_tree(tree)
    assert examined == 42 * 50_000
    assert returned == 50_000
    assert elapsed == 900.0


def test_the_real_tree_is_read_correctly() -> None:
    examined, returned, elapsed = parse_tree(TREE)
    assert (examined, returned, elapsed) == (2645, 1, 176.0)


def test_explain_reports_what_the_server_said(connected) -> None:
    box = toolbox(script(), connected)
    rendered = box.explain(SQL, ["42"])
    assert "rows examined 2,645" in rendered
    assert "rows returned 1" in rendered
    assert "actual time   176.0 ms" in rendered
    assert "index used    yes" in rendered
    assert "type=ref" in rendered


# ────────────────────────────  covering index  ───────────────────────────


def index(name: str, *columns: str) -> IndexInfo:
    return IndexInfo(name=name, columns=list(columns), unique=False)


def test_a_composite_covers_its_own_prefix() -> None:
    indexes = [index("idx_a_b", "customer_id", "status")]
    assert covering_index(indexes, ["customer_id"]) == "idx_a_b"
    assert covering_index(indexes, ["customer_id", "status"]) == "idx_a_b"
    # The reverse is NOT true, and getting this backwards means proposing
    # nothing useful ever again.
    assert covering_index([index("idx_a", "customer_id")],
                          ["customer_id", "status"]) is None
    assert covering_index(indexes, ["status"]) is None


def test_the_plan_phase_and_the_toolkit_share_one_definition() -> None:
    """Two copies of this rule would eventually disagree, and the
    disagreement would be a duplicate index in production."""
    from perfagent.plan import fixability

    assert fixability.covering_index.__module__.endswith("fixability")
    assert covering_index is dbmod.covering_index
    table = type("T", (), {"existing_indexes": [index("idx_a_b", "a", "b")]})()
    assert fixability.covering_index(table, ["a"]) == "idx_a_b"


# ───────────────────────────  index_benefit  ─────────────────────────────


def test_index_benefit_measures_before_and_after_then_drops(connected) -> None:
    server = script(indexes=[("idx_invoice_customer_id", 1, "customer_id", 1, 28691)])
    box = toolbox(server, connected)

    # The second EXPLAIN ANALYZE must report the faster tree.
    calls = {"n": 0}
    original = FakeCursor.execute

    def counting(self, sql, params=()):
        if "EXPLAIN ANALYZE" in sql:
            calls["n"] += 1
            if calls["n"] == 2:
                self.server.rows["EXPLAIN ANALYZE"] = [(FAST_TREE,)]
        return original(self, sql, params)

    FakeCursor.execute = counting                      # type: ignore[assignment]
    try:
        rendered = box.index_benefit("invoice", ["customer_id", "status"], SQL,
                                     ["42"])
    finally:
        FakeCursor.execute = original                  # type: ignore[assignment]

    assert "rows_examined=2,645" in rendered
    assert "rows_examined=571" in rendered
    assert "VERDICT  rows examined reduced" in rendered
    assert "has been dropped" in rendered

    created = server.ran("CREATE INDEX")
    dropped = server.ran("DROP INDEX")
    assert len(created) == 1 and len(dropped) == 1
    assert TEMP_PREFIX in created[0] and TEMP_PREFIX in dropped[0]


def test_the_index_is_dropped_even_when_everything_after_it_fails(
    connected
) -> None:
    """The `finally` is the whole design. If the second measurement throws,
    the index must still go."""
    server = script()
    server.errors["EXPLAIN ANALYZE"] = RuntimeError("server went away")
    box = toolbox(server, connected)

    with pytest.raises(Exception):
        box.index_benefit("invoice", ["status"], SQL, ["42"])

    # It failed before CREATE here; now fail AFTER it.
    server = script()
    box = toolbox(server, connected)
    calls = {"n": 0}
    original = FakeCursor.execute

    def exploding(self, sql, params=()):
        if "EXPLAIN ANALYZE" in sql:
            calls["n"] += 1
            if calls["n"] == 2:
                raise RuntimeError("killed mid-measurement")
        return original(self, sql, params)

    FakeCursor.execute = exploding                     # type: ignore[assignment]
    try:
        with pytest.raises(Exception):
            box.index_benefit("invoice", ["status"], SQL, ["42"])
    finally:
        FakeCursor.execute = original                  # type: ignore[assignment]

    assert server.ran("CREATE INDEX"), "the index was created"
    assert server.ran("DROP INDEX"), "and dropped anyway"


def test_an_existing_index_is_refused_and_named(connected) -> None:
    server = script(indexes=[
        ("idx_invoice_customer_id", 1, "customer_id", 1, 28691)])
    box = toolbox(server, connected)
    rendered = box.index_benefit("invoice", ["customer_id"], SQL, ["42"])

    assert rendered.startswith("REFUSED")
    assert "idx_invoice_customer_id" in rendered
    assert not server.ran("CREATE INDEX"), "nothing may be built"


def test_a_table_outside_the_endpoint_model_is_refused(connected) -> None:
    server = script()
    box = toolbox(server, connected)
    with pytest.raises(GuardrailError, match="not a table on this endpoint"):
        box.index_benefit("users", ["email"], SQL)
    with pytest.raises(GuardrailError):
        box.list_indexes("users")
    assert server.executed == []


def test_a_column_the_table_does_not_have_is_refused(connected) -> None:
    box = toolbox(script(columns=["id", "customer_id"]), connected)
    with pytest.raises(ToolError, match="no column"):
        box.index_benefit("invoice", ["tenant_id"], SQL)


@pytest.mark.parametrize("columns,fragment", [
    ([], "at least one column"),
    (["a"] * 9, "not an index"),
    (["status", "status"], "duplicate"),
    (["status; DROP TABLE invoice"], "not a column name"),
])
def test_column_lists_are_validated(connected, columns, fragment) -> None:
    box = toolbox(script(), connected)
    with pytest.raises(ToolError, match=fragment):
        box.index_benefit("invoice", columns, SQL)


def test_index_benefit_refuses_a_database_that_is_not_the_configured_one(
    connected, monkeypatch
) -> None:
    """This tool creates indexes. Pointing it anywhere else is impossible,
    not merely discouraged."""
    connected(script())
    elsewhere = MySQLTarget(host="prod.internal", port=3306, database="live")
    box = DbToolbox(elsewhere, ["invoice"], sweep_on_start=False)

    with pytest.raises(GuardrailError, match="configured measurement database"):
        box.index_benefit("invoice", ["status"], SQL)


def test_only_an_index_this_tool_created_can_be_dropped(connected) -> None:
    box = toolbox(script(), connected)
    with pytest.raises(GuardrailError, match="only ever drops"):
        box._drop_index(FakeCursor(script()), "invoice", "idx_invoice_customer_id")


def test_the_temporary_name_always_carries_the_prefix() -> None:
    name = dbmod._temp_index_name("invoice", ["customer_id", "status"])
    assert name.startswith(TEMP_PREFIX)
    assert name == dbmod._temp_index_name("invoice", ["customer_id", "status"])
    assert name != dbmod._temp_index_name("invoice", ["status", "customer_id"])


# ──────────────────────────────  the sweep  ──────────────────────────────


def test_construction_sweeps_what_a_crashed_run_left(connected) -> None:
    server = script(orphans=[("invoice", f"{TEMP_PREFIX}abc123")])
    connected(server)
    box = DbToolbox(MySQLTarget.from_env(), ["customer", "invoice"])

    assert box.swept == [f"invoice.{TEMP_PREFIX}abc123"]
    assert server.ran("DROP INDEX")
    assert TEMP_PREFIX in server.ran("DROP INDEX")[0]


def test_the_sweep_only_ever_looks_for_the_prefix(connected) -> None:
    server = script(orphans=[])
    connected(server)
    DbToolbox(MySQLTarget.from_env(), ["invoice"])
    lookup = server.ran("INDEX_NAME LIKE")
    assert lookup, "the sweep must query for orphans"
    assert not server.ran("DROP INDEX"), "nothing to drop, nothing dropped"


# ───────────────────────────  query_equivalent  ──────────────────────────


def rows_server(left: list[tuple], right: list[tuple]) -> FakeServer:
    server = script()
    server._left, server._right = left, right          # type: ignore[attr-defined]
    return server


def run_equivalent(server: FakeServer, connected, original: str,
                   candidate: str) -> str:
    box = toolbox(server, connected)
    calls = {"n": 0}
    original_execute = FakeCursor.execute

    def alternating(self, sql, params=()):
        if sql.lower().lstrip().startswith("select") and "information_schema" not in sql:
            calls["n"] += 1
            self.server.rows["select"] = (self.server._left if calls["n"] == 1
                                          else self.server._right)
        return original_execute(self, sql, params)

    FakeCursor.execute = alternating                   # type: ignore[assignment]
    try:
        return box.query_equivalent(original, candidate)
    finally:
        FakeCursor.execute = original_execute          # type: ignore[assignment]


def test_a_rewrite_returning_different_rows_is_named(connected) -> None:
    server = rows_server([(1, "a"), (2, "b")], [(1, "a")])
    rendered = run_equivalent(server, connected,
                              "select id, name from customer order by id",
                              "select id, name from customer where id = 1 order by id")
    assert rendered.startswith("NOT EQUIVALENT")
    assert "row count" in rendered
    assert "2" in rendered and "1" in rendered


def test_row_order_matters_only_when_the_original_orders(connected) -> None:
    shuffled = ([(1, "a"), (2, "b")], [(2, "b"), (1, "a")])

    unordered = run_equivalent(rows_server(*shuffled), connected,
                               "select id, name from customer",
                               "select id, name from customer")
    assert unordered.startswith("EQUIVALENT")
    assert "unordered sets" in unordered

    ordered = run_equivalent(rows_server(*shuffled), connected,
                             "select id, name from customer order by id",
                             "select id, name from customer order by name")
    assert ordered.startswith("NOT EQUIVALENT")
    assert "row 1 differs" in ordered


def test_a_value_that_merely_looks_equal_is_not_equal(connected) -> None:
    """The oracle compares values, not their printed forms."""
    from decimal import Decimal

    rendered = run_equivalent(
        rows_server([(Decimal("1.00"),)], [(1.0,)]), connected,
        "select amount from invoice", "select amount from invoice")
    assert rendered.startswith("NOT EQUIVALENT")


def test_query_equivalent_refuses_anything_that_is_not_a_select(connected) -> None:
    box = toolbox(script(), connected)
    with pytest.raises(ToolError, match="single SELECT"):
        box.query_equivalent("select 1", "delete from invoice")


# ────────────────────────────  the tool surface  ─────────────────────────


def test_each_phase_gets_the_tools_its_job_needs(connected) -> None:
    connected(script())
    target = MySQLTarget.from_env()
    box = DbToolbox(target, ["invoice"], sweep_on_start=False)

    names = lambda wanted: [s.name for s in db_tools(   # noqa: E731
        target, ["invoice"], names=wanted, toolbox=box)]

    # Diagnosis explains what IS. It cannot create an index.
    assert names(DIAGNOSE_TOOLS) == ["explain", "list_indexes"]
    assert "index_benefit" not in names(DIAGNOSE_TOOLS)
    assert "query_equivalent" not in names(DIAGNOSE_TOOLS)

    # Planning may measure one before proposing it.
    assert names(PLAN_TOOLS) == ["explain", "list_indexes", "index_benefit"]

    # The patch agent, already writing code, gets all four.
    assert names(PATCH_TOOLS) == ["explain", "list_indexes", "index_benefit",
                                  "query_equivalent"]


def test_an_unreachable_database_degrades_instead_of_killing_the_phase(
    monkeypatch
) -> None:
    def explode(target: MySQLTarget) -> Any:
        raise ToolError("cannot connect to MySQL")

    monkeypatch.setattr(dbmod, "connect", explode)
    notes: list[str] = []
    assert optional_db_tools(MySQLTarget.from_env(), ["invoice"],
                             names=PLAN_TOOLS, on_error=notes.append) == []
    assert notes and "unavailable" in notes[0]
    assert optional_db_tools(None, ["invoice"], names=PLAN_TOOLS) == []


def test_diagnosis_is_wired_without_index_benefit() -> None:
    """Wiring, not prose: the subset is named in code."""
    import inspect

    from perfagent.diagnose import agent as diagnose

    source = inspect.getsource(diagnose.run_diagnosis)
    assert "DIAGNOSE_TOOLS" in source
    assert "PATCH_TOOLS" not in source and "PLAN_TOOLS" not in source


# ───────────────────────────────  migrations  ────────────────────────────


@pytest.fixture
def flyway_repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    (root / "src/main/resources/db/migration").mkdir(parents=True)
    (root / "pom.xml").write_text("<dependency>flyway-core</dependency>")
    (root / "src/main/resources/db/migration/V1__baseline.sql").write_text(
        "CREATE TABLE x (id BIGINT);\n")
    return root


def test_a_migration_is_written_with_the_next_version(flyway_repo: Path) -> None:
    path = write_index_migration(
        flyway_repo, "invoice", ["customer_id", "status"],
        "the aggregate filters on both columns",
        evidence="rows examined 2,645 -> 571", run_id="run-1")

    assert path.name == "V2__add_invoice_customer_id_status_index.sql"
    body = path.read_text()
    assert "-- PerfAgent: the aggregate filters on both columns" in body
    assert "-- Evidence: rows examined 2,645 -> 571 (see run run-1)" in body
    assert ("CREATE INDEX idx_invoice_customer_id_status ON invoice "
            "(customer_id, status);") in body


def test_writing_the_same_index_twice_does_not_make_two_migrations(
    flyway_repo: Path
) -> None:
    """Flyway would apply both and MySQL would reject the duplicate key name."""
    first = write_index_migration(flyway_repo, "invoice", ["status"], "why")
    second = write_index_migration(flyway_repo, "invoice", ["status"], "why again")

    assert first == second
    assert first.read_text().count("CREATE INDEX") == 1
    migrations = sorted(p.name for p in first.parent.glob("*.sql"))
    assert migrations == ["V1__baseline.sql", "V2__add_invoice_status_index.sql"]


def test_a_different_index_gets_the_next_version(flyway_repo: Path) -> None:
    write_index_migration(flyway_repo, "invoice", ["status"], "a")
    second = write_index_migration(flyway_repo, "customer", ["name"], "b")
    assert second.name == "V3__add_customer_name_index.sql"


def test_a_repository_without_flyway_is_refused(tmp_path: Path) -> None:
    """A migration nothing applies is worse than no migration: the patch then
    claims an index the schema does not have."""
    root = tmp_path / "docker-owned"
    (root / "docker/mysql/init").mkdir(parents=True)
    (root / "pom.xml").write_text("<project/>")
    (root / "docker/mysql/init/01-schema.sql").write_text("CREATE TABLE x(id INT);")

    with pytest.raises(ToolError, match="no Flyway migration directory"):
        write_index_migration(root, "invoice", ["status"], "why")
    assert not uses_flyway(root)


def test_the_version_is_read_from_the_files_that_are_there(tmp_path: Path) -> None:
    directory = tmp_path / "m"
    directory.mkdir()
    assert next_version(directory) == 1
    for name in ("V1__a.sql", "V9__b.sql", "V10__c.sql", "notes.txt",
                 "R__repeatable.sql"):
        (directory / name).write_text("-- x")
    assert next_version(directory) == 11


def test_the_index_name_is_conventional_and_fits() -> None:
    assert (index_name("invoice", ["customer_id", "status"])
            == "idx_invoice_customer_id_status")
    assert len(index_name("invoice", ["x" * 80])) <= 64


def test_the_migration_directory_is_discovered_not_assumed(tmp_path: Path) -> None:
    root = tmp_path / "r"
    (root / "migrations").mkdir(parents=True)
    (root / "migrations/V4__x.sql").write_text("-- x")
    assert find_migration_dir(root) == root / "migrations"
    assert existing_migration_for(root / "migrations", "idx_nothing") is None


@pytest.mark.parametrize("bad", ["in voice", "invoice;DROP", ""])
def test_a_migration_refuses_an_identifier_it_cannot_trust(
    flyway_repo: Path, bad: str
) -> None:
    with pytest.raises(ToolError):
        write_index_migration(flyway_repo, bad or "invoice",
                              [bad] if bad else [], "why")
