"""get_schema — the live database, including the indexes that already exist.

Returning existing indexes is the whole point. A later phase has to tell a
MISSING index from one that is already there, and must never propose a
duplicate; neither is possible from entity annotations, which describe what
Hibernate would create and not what the running database has.

If the database is unreachable this raises. It does not fall back to guessing
from `@Table` annotations: a guessed index list is worse than no index list,
because everything downstream would treat it as measured.
"""

from __future__ import annotations

from difflib import get_close_matches
from typing import Any, Callable

from ..contracts import IndexInfo, TableInfo
from ..errors import ToolError
from ..measure.sql import MySQLTarget, connect
from . import ToolSpec

MAX_TABLES_PER_CALL = 20


def _placeholders(count: int) -> str:
    return ", ".join(["%s"] * count)


def _known_tables(cursor: Any, database: str) -> list[str]:
    cursor.execute(
        "SELECT TABLE_NAME FROM information_schema.TABLES "
        "WHERE TABLE_SCHEMA = %s ORDER BY TABLE_NAME",
        (database,),
    )
    return [str(row[0]) for row in cursor.fetchall()]


def introspect(target: MySQLTarget, tables: list[str]) -> dict[str, TableInfo]:
    """Read columns, indexes, row counts and foreign keys for named tables.

    Every number here comes from `information_schema`. None of it is ever
    supplied by a model (invariant 4).
    """
    wanted = [t.strip() for t in tables if t and t.strip()]
    if not wanted:
        raise ToolError("get_schema needs at least one table name.")
    if len(wanted) > MAX_TABLES_PER_CALL:
        raise ToolError(
            f"get_schema was asked for {len(wanted)} tables; the cap is "
            f"{MAX_TABLES_PER_CALL}.",
            suggestion="Ask for the tables this endpoint actually touches.",
        )

    with connect(target) as conn:
        cursor = conn.cursor()
        try:
            available = _known_tables(cursor, target.database)
            missing = [t for t in wanted if t not in available]
            if missing:
                hints = []
                for name in missing:
                    close = get_close_matches(name, available, n=3)
                    hints.append(f"  '{name}'"
                                 + (f" — did you mean {', '.join(close)}?"
                                    if close else ""))
                raise ToolError(
                    f"schema '{target.database}' has no such table(s):\n"
                    + "\n".join(hints)
                    + f"\nTables in this schema: {', '.join(available)}",
                )

            found: dict[str, TableInfo] = {
                name: TableInfo(table=name) for name in wanted
            }
            marks = _placeholders(len(wanted))
            args = (target.database, *wanted)

            cursor.execute(
                f"SELECT TABLE_NAME, COLUMN_NAME, COLUMN_TYPE, IS_NULLABLE "
                f"FROM information_schema.COLUMNS "
                f"WHERE TABLE_SCHEMA = %s AND TABLE_NAME IN ({marks}) "
                f"ORDER BY TABLE_NAME, ORDINAL_POSITION",
                args,
            )
            for table, column, column_type, nullable in cursor.fetchall():
                null = "NULL" if str(nullable).upper() == "YES" else "NOT NULL"
                found[str(table)].all_columns.append(
                    f"{column} {column_type} {null}")

            cursor.execute(
                f"SELECT TABLE_NAME, INDEX_NAME, SEQ_IN_INDEX, COLUMN_NAME, "
                f"NON_UNIQUE, CARDINALITY "
                f"FROM information_schema.STATISTICS "
                f"WHERE TABLE_SCHEMA = %s AND TABLE_NAME IN ({marks}) "
                f"ORDER BY TABLE_NAME, INDEX_NAME, SEQ_IN_INDEX",
                args,
            )
            builders: dict[tuple[str, str], IndexInfo] = {}
            for table, index, _seq, column, non_unique, cardinality in cursor.fetchall():
                key = (str(table), str(index))
                entry = builders.get(key)
                if entry is None:
                    entry = IndexInfo(
                        name=str(index), columns=[],
                        unique=not bool(int(non_unique or 0)),
                        cardinality=int(cardinality) if cardinality is not None else None,
                    )
                    builders[key] = entry
                    found[str(table)].existing_indexes.append(entry)
                entry.columns.append(str(column))

            cursor.execute(
                f"SELECT TABLE_NAME, TABLE_ROWS FROM information_schema.TABLES "
                f"WHERE TABLE_SCHEMA = %s AND TABLE_NAME IN ({marks})",
                args,
            )
            for table, rows in cursor.fetchall():
                found[str(table)].approx_rows = int(rows) if rows is not None else None

            cursor.execute(
                f"SELECT TABLE_NAME, COLUMN_NAME, CONSTRAINT_NAME, "
                f"REFERENCED_TABLE_NAME, REFERENCED_COLUMN_NAME "
                f"FROM information_schema.KEY_COLUMN_USAGE "
                f"WHERE TABLE_SCHEMA = %s AND REFERENCED_TABLE_NAME IS NOT NULL "
                f"AND (TABLE_NAME IN ({marks}) OR REFERENCED_TABLE_NAME IN ({marks}))",
                (target.database, *wanted, *wanted),
            )
            for table, column, constraint, ref_table, ref_column in cursor.fetchall():
                edge_out = {
                    "direction": "out", "constraint": str(constraint),
                    "column": str(column),
                    "references": f"{ref_table}.{ref_column}",
                }
                edge_in = {
                    "direction": "in", "constraint": str(constraint),
                    "column": f"{table}.{column}",
                    "references": str(ref_column),
                }
                if str(table) in found:
                    found[str(table)].foreign_keys.append(edge_out)
                if str(ref_table) in found:
                    found[str(ref_table)].foreign_keys.append(edge_in)
        finally:
            cursor.close()

    return found


def render_tables(tables: dict[str, TableInfo]) -> str:
    """What the model reads. Indexes first, because that is the point."""
    out: list[str] = []
    for name, info in tables.items():
        out.append(f"TABLE {name}"
                   + (f"   (~{info.approx_rows:,} rows, approximate)"
                      if info.approx_rows is not None else ""))
        out.append("  columns:")
        for column in info.all_columns:
            out.append(f"    {column}")

        out.append("  EXISTING INDEXES:")
        if not info.existing_indexes:
            out.append("    (none — not even a primary key)")
        for index in info.existing_indexes:
            unique = "UNIQUE " if index.unique else ""
            cardinality = (f", cardinality {index.cardinality:,}"
                           if index.cardinality is not None else "")
            out.append(f"    {unique}{index.name}({', '.join(index.columns)})"
                       f"{cardinality}")

        if info.foreign_keys:
            out.append("  foreign keys:")
            for key in info.foreign_keys:
                arrow = "->" if key.get("direction") == "out" else "<-"
                out.append(f"    {key.get('column')} {arrow} "
                           f"{key.get('references')}  [{key.get('constraint')}]")
        out.append("")

    out.append("These indexes ALREADY EXIST. Do not report a missing index "
               "that is listed above.")
    return "\n".join(out)


def make_get_schema(target: MySQLTarget,
                    cache: dict[str, TableInfo] | None = None
                    ) -> Callable[..., str]:
    """Build the handler. `cache` accumulates what was actually measured.

    The phase templates `EndpointModel.tables` from this cache rather than
    from whatever the model retypes into its submission, so a cardinality in
    the artifact is always a number the database reported.
    """

    def get_schema(tables: list[str] | str) -> str:
        names = [tables] if isinstance(tables, str) else list(tables or [])
        found = introspect(target, names)
        if cache is not None:
            cache.update(found)
        return render_tables(found)

    return get_schema


GET_SCHEMA_PARAMETERS: dict[str, Any] = {
    "type": "object",
    "properties": {
        "tables": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Physical table names, e.g. ['customer', 'invoice'].",
        },
    },
    "required": ["tables"],
}


def schema_tools(target: MySQLTarget,
                 cache: dict[str, TableInfo] | None = None) -> list[ToolSpec]:
    return [ToolSpec(
        name="get_schema",
        description=(
            "Columns, types, approximate row counts, foreign keys and — most "
            "importantly — the indexes that ALREADY EXIST on the given tables, "
            "read from the live database. Call it for every table this "
            "endpoint touches. Knowing which indexes exist matters as much as "
            "knowing which are missing."
        ),
        parameters=GET_SCHEMA_PARAMETERS,
        handler=make_get_schema(target, cache),
    )]


def database_target(trace: Any = None, *, phase: str = "agent") -> MySQLTarget | None:
    """The MySQL target from the environment, or None if one cannot be built.

    Constructed, not connected. An UNREACHABLE database has to fail inside
    `get_schema`, with a message the model can read and work around, rather
    than at registry-build time where it would abort a phase that can still do
    most of its job from source alone. `from_env` validates, so a garbled
    PERFAGENT_MYSQL_PORT raises here — which is why this swallows and notes it.
    """
    try:
        return MySQLTarget.from_env()
    except Exception as exc:  # noqa: BLE001 — no database is degraded, not dead
        if trace is not None:
            from ..trace import EventType

            trace.emit(
                EventType.NOTE, phase=phase,
                note=f"no database target could be built ({type(exc).__name__}: "
                     f"{exc}); running without get_schema.",
            )
        return None
