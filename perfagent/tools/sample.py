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

        from ..measure.sql import connect

        # `connect` is a @contextmanager -- an earlier version of this
        # called it like a factory and then called .cursor() on the
        # generator wrapper. mypy caught it; it would have been an
        # AttributeError on the first real sample.
        with connect(self.target) as connection:
            cursor = connection.cursor()
            cursor.execute(
                f"SET SESSION max_execution_time = {STATEMENT_TIMEOUT_MS}")
            # Total first, so the stride can span the table deterministically.
            cursor.execute(
                f"SELECT COUNT(*) FROM `{name}` WHERE `{field}` IS NOT NULL")
            row = cursor.fetchone()
            total = int(row[0]) if row else 0
            if total == 0:
                return []

            stride = max(1, total // bounded)
            cursor.execute(
                f"SELECT `{field}` FROM `{name}` WHERE `{field}` IS NOT NULL "
                f"ORDER BY `{field}` LIMIT %s",
                (stride * bounded,),
            )
            rows: Sequence[Sequence[Any]] = cursor.fetchall()
            picked = [rows[i][0] for i in range(0, len(rows), stride)]
            return picked[:bounded]


class StaticSampler:
    """A sampler backed by a dict, for tests and offline rehearsals.

    Exists so the generator can be exercised with no database at all —
    `tests/test_workload.py` uses it, and so does the offline rehearsal.
    It is NOT a fallback: nothing constructs one implicitly, because a
    corpus built from invented ids would look exactly like a real one.
    """

    def __init__(self, values: dict[tuple[str, str], list[Any]]) -> None:
        self.values = values

    def sample(self, table: str, column: str, *,
               limit: int = DEFAULT_LIMIT) -> list[Any]:
        return list(self.values.get((table, column), []))[:limit]
