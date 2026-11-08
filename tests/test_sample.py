"""ValueSampler against the digest connection: schema-less by design.

`MySQLTarget`'s own connection selects no default schema (see its docstring
in measure/sql.py) -- the digest reads depend on that. `ValueSampler` used to
issue bare `` `table` `` references on that same connection, which fails with
"No database selected" the instant it runs for real. These tests replace
`connect` with a fake that behaves exactly like that: no USE, no default
schema, every unqualified reference is an error.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator

import pytest

from senapathy.errors import ToolError
from senapathy.measure.sql import MySQLTarget
from senapathy.tools.sample import ValueSampler


def _target() -> MySQLTarget:
    return MySQLTarget(host="127.0.0.1", port=3307, user="root",
                       password="root", database="mini")


class SchemaLessCursor:
    """Answers COUNT(*) then a row SELECT -- refusing anything not qualified
    with the schema, exactly like a real connection with no default database."""

    def __init__(self, total: int, values: list[Any],
                 fail_with: Exception | None = None) -> None:
        self.total = total
        self.values = values
        self.fail_with = fail_with
        self.executed: list[str] = []
        self._rows: list[tuple[Any, ...]] = []

    def execute(self, sql: str, params: Any = ()) -> None:
        self.executed.append(sql)
        if self.fail_with is not None:
            raise self.fail_with
        if "max_execution_time" in sql:
            return
        if "." not in sql.split("FROM", 1)[1].split("WHERE", 1)[0]:
            # No `schema`.`table` -- this is exactly what a real schema-less
            # connection does with a bare identifier.
            from mysql.connector.errors import ProgrammingError

            raise ProgrammingError("1046 (3D000): No database selected")
        if "COUNT(*)" in sql:
            self._rows = [(self.total,)]
        else:
            self._rows = [(v,) for v in self.values]

    def fetchone(self) -> tuple[Any, ...] | None:
        return self._rows[0] if self._rows else None

    def fetchall(self) -> list[tuple[Any, ...]]:
        return self._rows

    def close(self) -> None:
        pass


class RowsCursor:
    """Returns a fixed set of rows for whatever non-SET statement runs,
    refusing anything not qualified with the schema -- same rule as
    SchemaLessCursor, for the dense/count queries that don't fit its
    COUNT(*)-then-SELECT shape."""

    def __init__(self, rows: list[tuple[Any, ...]],
                 fail_with: Exception | None = None) -> None:
        self.rows = rows
        self.fail_with = fail_with
        self.executed: list[str] = []

    def execute(self, sql: str, params: Any = ()) -> None:
        self.executed.append(sql)
        if self.fail_with is not None:
            raise self.fail_with
        if "max_execution_time" in sql:
            return
        if "." not in sql.split("FROM", 1)[1].split("WHERE", 1)[0]:
            from mysql.connector.errors import ProgrammingError

            raise ProgrammingError("1046 (3D000): No database selected")

    def fetchall(self) -> list[tuple[Any, ...]]:
        return self.rows

    def close(self) -> None:
        pass


def _install(monkeypatch: pytest.MonkeyPatch,
             cursor: SchemaLessCursor | RowsCursor) -> None:
    @contextmanager
    def fake_connect(target: MySQLTarget) -> Iterator[Any]:
        yield type("Conn", (), {"cursor": lambda self: cursor})()

    import senapathy.measure.sql as sql_mod

    monkeypatch.setattr(sql_mod, "connect", fake_connect)


def test_sample_works_against_a_schema_less_connection(
    monkeypatch: pytest.MonkeyPatch
) -> None:
    cursor = SchemaLessCursor(total=4, values=[1, 2, 3, 4])
    _install(monkeypatch, cursor)

    sampler = ValueSampler(_target(), allowed_tables=["invoice"])
    values = sampler.sample("invoice", "id", limit=4)

    assert values == [1, 2, 3, 4]
    assert cursor.executed, "the sampler must actually issue queries"


def test_every_table_reference_is_qualified_with_the_schema(
    monkeypatch: pytest.MonkeyPatch
) -> None:
    cursor = SchemaLessCursor(total=1, values=[1])
    _install(monkeypatch, cursor)

    sampler = ValueSampler(_target(), allowed_tables=["invoice"])
    sampler.sample("invoice", "id")

    statements = [s for s in cursor.executed if "max_execution_time" not in s]
    assert statements, "COUNT and SELECT must have run"
    for statement in statements:
        assert "`mini`.`invoice`" in statement, (
            f"table reference not qualified with the schema: {statement!r}")


def test_a_driver_error_while_sampling_is_a_named_tool_error(
    monkeypatch: pytest.MonkeyPatch
) -> None:
    """A ProgrammingError must never reach the caller as itself -- it has to
    become a ToolError naming the target and what was being sampled, the
    same standard as every other refusal in this product."""
    from mysql.connector.errors import ProgrammingError

    cursor = SchemaLessCursor(
        total=0, values=[],
        fail_with=ProgrammingError("1146 (42S02): Table 'mini.ghost' doesn't exist"))
    _install(monkeypatch, cursor)

    sampler = ValueSampler(_target(), allowed_tables=["ghost"])
    with pytest.raises(ToolError, match="cannot sample ghost.id"):
        sampler.sample("ghost", "id")


# ───────────  the failure a phase actually sees, not a traceback  ────────


def test_a_driver_error_inside_the_workload_phase_is_a_named_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """Before this fix, a ProgrammingError from the digest connection would
    have escaped `_run_workload` -- and therefore `run_pipeline` -- as
    itself. `run_pipeline` catches ToolError/ContractViolation/
    BudgetExhausted, nothing from mysql.connector: the guarantee has to be
    that no raw driver exception ever reaches that boundary in the first
    place."""
    from mysql.connector.errors import ProgrammingError

    from senapathy.contracts.comprehension import EndpointModel
    from senapathy.orchestrator import Budget, Context
    from senapathy.orchestrator import _run_workload
    from senapathy.trace import Trace

    citation = {"file": "src/main/java/App.java", "line": 1, "symbol": "x"}
    model = EndpointModel.model_validate({
        "route": "GET /api/customers/{id}",
        "entry": citation,
        "path": [citation],
        "tables": [{"table": "customer", "columns_read": ["id"],
                    "all_columns": ["id", "name"]}],
        "pagination": {"mechanism": "offset", "fires_count_query": False,
                       "max_page_size": None},
        "uncertainties": [],
    })

    import senapathy.onboard as onboard_mod
    import senapathy.understand.agent as understand_mod

    monkeypatch.setattr(understand_mod, "load_model", lambda path: model)
    monkeypatch.setattr(onboard_mod, "load_profile",
                        lambda path: object())  # unused: a sampler is supplied

    cursor = SchemaLessCursor(
        total=0, values=[],
        fail_with=ProgrammingError("1045 (28000): Access denied for user"))
    _install(monkeypatch, cursor)
    failing_sampler = ValueSampler(_target(), allowed_tables=["customer"])

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    ctx = Context(
        run_id="r1", run_dir=run_dir, repo=None, endpoint="GET /api/customers/{id}",
        budget=Budget(), trace=Trace(run_dir, "r1"),
        options={"sampler": failing_sampler},
    )

    with pytest.raises(ToolError, match="cannot sample customer.id"):
        _run_workload(ctx)


# ──────────────────────  dense / count_by_value queries  ──────────────────


def test_sample_dense_ranks_densest_first_and_qualifies_the_table(
    monkeypatch: pytest.MonkeyPatch
) -> None:
    cursor = RowsCursor(rows=[(1, 52765), (2, 41220), (3, 1439)])
    _install(monkeypatch, cursor)

    sampler = ValueSampler(_target(), allowed_tables=["invoice"])
    ranked = sampler.sample_dense("invoice", "customer_id", limit=3)

    assert ranked == [(1, 52765), (2, 41220), (3, 1439)]
    statements = [s for s in cursor.executed if "max_execution_time" not in s]
    assert statements
    for statement in statements:
        assert "`mini`.`invoice`" in statement
        assert "GROUP BY" in statement
        assert "RAND(" not in statement


def test_sample_dense_errors_become_a_named_tool_error(
    monkeypatch: pytest.MonkeyPatch
) -> None:
    from mysql.connector.errors import ProgrammingError

    cursor = RowsCursor(
        rows=[],
        fail_with=ProgrammingError("1146 (42S02): Table 'mini.ghost' doesn't exist"))
    _install(monkeypatch, cursor)

    sampler = ValueSampler(_target(), allowed_tables=["ghost"])
    with pytest.raises(ToolError, match="cannot rank ghost.id"):
        sampler.sample_dense("ghost", "id")


def test_count_by_value_fills_zero_for_values_with_no_matching_rows(
    monkeypatch: pytest.MonkeyPatch
) -> None:
    cursor = RowsCursor(rows=[(1, 1439)])
    _install(monkeypatch, cursor)

    sampler = ValueSampler(_target(), allowed_tables=["invoice"])
    counts = sampler.count_by_value("invoice", "customer_id", [1, 9])

    assert counts == {1: 1439, 9: 0}


def test_count_by_value_with_no_values_skips_the_query() -> None:
    sampler = ValueSampler(_target(), allowed_tables=["invoice"])
    assert sampler.count_by_value("invoice", "customer_id", []) == {}
