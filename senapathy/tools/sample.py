"""Real parameter values, sampled from the database. Read-only, by shape.

The corpus generator needs ids that actually exist. Inventing them produces
a corpus of 404s that measures the error path; taking them from a model
produces numbers nobody can check. So they come from the database, through
a query this module BUILDS rather than accepts: there is no caller-supplied
SQL here, only a table and a column that must both already appear in the
endpoint model.

Determinism matters more than statistical spread. The oracle is keyed by
`corpus_hash`, so a sampler that returned a different set on each call
would invalidate the recording every run. Ordering is therefore always
`ORDER BY <column>` and never `RAND()`.
"""

from __future__ import annotations

import re
from typing import Any, Iterable, Sequence

from ..errors import GuardrailError, ToolError

# A bare identifier. Anything else cannot reach the SQL string, which is the
# only reason it is safe to interpolate a table and column name at all --
# MySQL does not take placeholders for identifiers.
_IDENTIFIER = re.compile(r"\w+")

DEFAULT_LIMIT = 6
MAX_LIMIT = 50
STATEMENT_TIMEOUT_MS = 5_000


class ValueSampler:
    """Samples distinct column values from the endpoint's own tables."""

    def __init__(self, target: Any, allowed_tables: Iterable[str]) -> None:
        self.target = target
        self.allowed = {str(t) for t in allowed_tables}

    def _check(self, table: str, column: str) -> tuple[str, str]:
        name = (table or "").strip().strip("`")
        field = (column or "").strip().strip("`")
        for value, what in ((name, "table"), (field, "column")):
            if not _IDENTIFIER.fullmatch(value):
                raise ToolError(
                    f"{value!r} is not a bare {what} name, so it cannot be "
                    f"used in an identifier position.")
        if name not in self.allowed:
            raise GuardrailError(
                f"{name!r} is not a table on this endpoint's path and was "
                f"refused. Tables this run may sample: "
                f"{', '.join(sorted(self.allowed)) or 'none'}.",
                suggestion="Sample from the tables the endpoint model "
                           "recorded reading.",
            )
        return name, field

    def sample(self, table: str, column: str, *,
               limit: int = DEFAULT_LIMIT) -> list[Any]:
        """Up to `limit` distinct non-null values, in a stable order.

        Spread across the table rather than taken off the top: the first N
        ids of a Zipf-skewed table are the heaviest rows, and a corpus built
        only from those measures the tail of the distribution as if it were
        the middle.
        """
        name, field = self._check(table, column)
        bounded = max(1, min(int(limit), MAX_LIMIT))

        from mysql.connector import Error as MySQLError

        from ..measure.sql import connect

        # The digest connection is deliberately schema-less (see
        # MySQLTarget's docstring), so every table reference here must be
        # qualified -- an earlier version left these bare and every sample
        # call failed with "No database selected".
        qualified = f"`{self.target.database}`.`{name}`"
        try:
            # `connect` is a @contextmanager -- an earlier version of this
            # called it like a factory and then called .cursor() on the
            # generator wrapper. mypy caught it; it would have been an
            # AttributeError on the first real sample.
            with connect(self.target) as connection:
                cursor = connection.cursor()
                try:
                    cursor.execute(
                        f"SET SESSION max_execution_time = {STATEMENT_TIMEOUT_MS}")
                    # Total first, so the stride can span the table
                    # deterministically.
                    cursor.execute(
                        f"SELECT COUNT(*) FROM {qualified} "
                        f"WHERE `{field}` IS NOT NULL")
                    row = cursor.fetchone()
                    total = int(row[0]) if row else 0
                    if total == 0:
                        return []

                    stride = max(1, total // bounded)
                    cursor.execute(
                        f"SELECT `{field}` FROM {qualified} "
                        f"WHERE `{field}` IS NOT NULL "
                        f"ORDER BY `{field}` LIMIT %s",
                        (stride * bounded,),
                    )
                    rows: Sequence[Sequence[Any]] = cursor.fetchall()
                    picked = [rows[i][0] for i in range(0, len(rows), stride)]
                    return picked[:bounded]
                finally:
                    cursor.close()
        except MySQLError as exc:
            raise ToolError(
                f"cannot sample {name}.{field} from {self.target.dsn()}: {exc}",
                suggestion="Confirm the table and column exist in that "
                "schema and that the connecting account has SELECT on it.",
            ) from exc

    def sample_dense(self, table: str, column: str, *,
                      limit: int = DEFAULT_LIMIT) -> list[tuple[Any, int]]:
        """Up to `limit` values of `table.column`, densest first, paired with
        how many rows carry each one.

        For a path parameter that addresses a row with per-row work fanning
        out from it (e.g. `{id}` on `/customers/{id}/invoices`), `table` and
        `column` here are the CHILD side of the relationship — `invoice` and
        `customer_id`, not `customer` and `id`. The parent table can never
        be dense: there is exactly one row per id. What scales is how many
        child rows reference it, which is exactly this count.
        """
        name, field = self._check(table, column)
        bounded = max(1, min(int(limit), MAX_LIMIT))

        from mysql.connector import Error as MySQLError

        from ..measure.sql import connect

        qualified = f"`{self.target.database}`.`{name}`"
        try:
            with connect(self.target) as connection:
                cursor = connection.cursor()
                try:
                    cursor.execute(
                        f"SET SESSION max_execution_time = {STATEMENT_TIMEOUT_MS}")
                    # GROUP BY with no WHERE beyond IS NOT NULL -- see
                    # docs/schema-changes.md (or the caller's note) for what
                    # this costs on a large table; this call does not bound
                    # it beyond `limit` on the OUTPUT, not the scan.
                    cursor.execute(
                        f"SELECT `{field}`, COUNT(*) AS cnt FROM {qualified} "
                        f"WHERE `{field}` IS NOT NULL "
                        f"GROUP BY `{field}` "
                        f"ORDER BY cnt DESC, `{field}` ASC "
                        f"LIMIT %s",
                        (bounded,),
                    )
                    return [(row[0], int(row[1])) for row in cursor.fetchall()]
                finally:
                    cursor.close()
        except MySQLError as exc:
            raise ToolError(
                f"cannot rank {name}.{field} by row count from "
                f"{self.target.dsn()}: {exc}",
                suggestion="Confirm the table and column exist in that "
                "schema and that the connecting account has SELECT on it.",
            ) from exc

    def count_by_value(self, table: str, column: str,
                        values: Sequence[Any]) -> dict[Any, int]:
        """COUNT(*) in `table` for each of `values`, grouped by `column`.

        Used to annotate a request's note with how many child rows a
        sampled id actually has, regardless of which strategy picked it —
        so a held-back request can say "0 child rows" just as plainly as a
        benchmark request says "1,439 child rows".
        """
        name, field = self._check(table, column)
        wanted = list(dict.fromkeys(values))
        if not wanted:
            return {}

        from mysql.connector import Error as MySQLError

        from ..measure.sql import connect

        qualified = f"`{self.target.database}`.`{name}`"
        marks = ", ".join(["%s"] * len(wanted))
        try:
            with connect(self.target) as connection:
                cursor = connection.cursor()
                try:
                    cursor.execute(
                        f"SET SESSION max_execution_time = {STATEMENT_TIMEOUT_MS}")
                    cursor.execute(
                        f"SELECT `{field}`, COUNT(*) FROM {qualified} "
                        f"WHERE `{field}` IN ({marks}) "
                        f"GROUP BY `{field}`",
                        tuple(wanted),
                    )
                    counts = {row[0]: int(row[1]) for row in cursor.fetchall()}
                finally:
                    cursor.close()
        except MySQLError as exc:
            raise ToolError(
                f"cannot count {name}.{field} from {self.target.dsn()}: {exc}",
                suggestion="Confirm the table and column exist in that "
                "schema and that the connecting account has SELECT on it.",
            ) from exc
        return {value: counts.get(value, 0) for value in wanted}


class StaticSampler:
    """A sampler backed by a dict, for tests and offline rehearsals.

    Exists so the generator can be exercised with no database at all —
    `tests/test_workload.py` uses it, and so does the offline rehearsal.
    It is NOT a fallback: nothing constructs one implicitly, because a
    corpus built from invented ids would look exactly like a real one.
    """

    def __init__(
        self, values: dict[tuple[str, str], list[Any]],
        dense: dict[tuple[str, str], list[tuple[Any, int]]] | None = None,
    ) -> None:
        self.values = values
        self.dense = dense or {}

    def sample(self, table: str, column: str, *,
               limit: int = DEFAULT_LIMIT) -> list[Any]:
        return list(self.values.get((table, column), []))[:limit]

    def sample_dense(self, table: str, column: str, *,
                      limit: int = DEFAULT_LIMIT) -> list[tuple[Any, int]]:
        return list(self.dense.get((table, column), []))[:limit]

    def count_by_value(self, table: str, column: str,
                        values: Sequence[Any]) -> dict[Any, int]:
        pairs = dict(self.dense.get((table, column), []))
        return {value: pairs.get(value, 0) for value in dict.fromkeys(values)}
