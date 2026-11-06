from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Any, Iterator

from pydantic import BaseModel, Field

from ..contracts import SqlDigestStat
from ..errors import ToolError

# performance_schema timers are in picoseconds.
PICOSECONDS_PER_MS = 1_000_000_000

DIGEST_TABLE = "performance_schema.events_statements_summary_by_digest"

_ENV_PREFIX = "PERFAGENT_MYSQL_"


class MySQLTarget(BaseModel):
    """Where the app's database lives, and who to log in as.

    performance_schema is not readable by the application user (the app user
    has rights on its own schema only), and TRUNCATE on a performance_schema
    table needs DROP. So evidence collection uses an admin login that is
    deliberately separate from whatever the app is configured with — this
    connection never touches application data.
    """

    host: str = "127.0.0.1"
    port: int = 3307
    user: str = "root"
    password: str = "root"
    database: str = Field(
        default="mini",
        description="The APPLICATION schema. Used to filter digests, not to "
        "select a default database — the connection stays schema-less so that "
        "our own bookkeeping statements do not show up in the evidence.",
    )

    @classmethod
    def from_env(cls) -> "MySQLTarget":
        """PERFAGENT_MYSQL_HOST / _PORT / _USER / _PASSWORD / _DATABASE."""
        values: dict[str, Any] = {}
        for field in ("host", "port", "user", "password", "database"):
            raw = os.environ.get(_ENV_PREFIX + field.upper())
            if raw is not None:
                values[field] = raw
        return cls.model_validate(values)

    def dsn(self) -> str:
        """Printable identity, never the password."""
        return f"{self.user}@{self.host}:{self.port}/{self.database}"


@contextmanager
def connect(target: MySQLTarget) -> Iterator[Any]:
    """Open a short-lived connection, no default schema selected."""
    try:
        import mysql.connector
        from mysql.connector import Error as MySQLError
    except ImportError as exc:  # pragma: no cover - dependency is declared
        raise ToolError(
            "mysql-connector-python is not installed",
            suggestion="pip install -e '.[dev]'",
        ) from exc

    try:
        conn = mysql.connector.connect(
            host=target.host,
            port=target.port,
            user=target.user,
            password=target.password,
            autocommit=True,
            connection_timeout=10,
        )
    except MySQLError as exc:
        raise ToolError(
            f"cannot connect to MySQL at {target.host}:{target.port} as "
            f"{target.user}: {exc}",
            suggestion="Start the database (docker compose up -d mysql in the "
            "target repo) and set PERFAGENT_MYSQL_USER / _PASSWORD to an "
            "account with SELECT and DROP on performance_schema.",
        ) from exc
    try:
        yield conn
    finally:
        try:
            conn.close()
        except Exception:  # noqa: BLE001 - closing must never mask the real error
            pass


def reset_digests(target: MySQLTarget) -> None:
    """Empty the digest table so the next collection covers one run exactly.

    Called after warmup and before the measured iterations. Without the reset
    the digests describe every statement since the server started, and a
    per-request statement count computed from them is meaningless.
    """
    from mysql.connector import Error as MySQLError

    with connect(target) as conn:
        cursor = conn.cursor()
        try:
            cursor.execute(f"TRUNCATE TABLE {DIGEST_TABLE}")
        except MySQLError as exc:
            raise ToolError(
                f"cannot truncate {DIGEST_TABLE} as {target.user}: {exc}",
                suggestion="performance_schema must be ON and the user needs "
                "DROP on it. Check the server was started with "
                "--performance-schema=ON.",
            ) from exc
        finally:
            cursor.close()


def collect_digests(target: MySQLTarget, limit: int | None = None) -> list[SqlDigestStat]:
    """Read the statement digests for the application schema, slowest first.

    Only statements whose SCHEMA_NAME matches the app's database are returned,
    which is also what keeps this function's own SELECT out of the result: the
    connection has no default schema, so our statements land under a NULL
    SCHEMA_NAME and are filtered away.
    """
    from mysql.connector import Error as MySQLError

    query = f"""
        SELECT DIGEST,
               DIGEST_TEXT,
               COUNT_STAR,
               SUM_TIMER_WAIT,
               SUM_ROWS_EXAMINED,
               SUM_ROWS_SENT,
               SUM_NO_INDEX_USED
        FROM {DIGEST_TABLE}
        WHERE SCHEMA_NAME = %s
          AND DIGEST_TEXT IS NOT NULL
          AND COUNT_STAR > 0
        ORDER BY SUM_TIMER_WAIT DESC
    """

    with connect(target) as conn:
        cursor = conn.cursor()
        try:
            cursor.execute(query, (target.database,))
            rows = cursor.fetchall()
        except MySQLError as exc:
            raise ToolError(
                f"cannot read {DIGEST_TABLE} as {target.user}: {exc}",
                suggestion="The application user usually cannot read "
                "performance_schema. Set PERFAGENT_MYSQL_USER / _PASSWORD to an "
                "admin account.",
            ) from exc
        finally:
            cursor.close()

    stats: list[SqlDigestStat] = []
    for digest, digest_text, count_star, timer_wait, rows_examined, rows_sent, no_index in rows:
        calls = int(count_star or 0)
        total_ms = float(timer_wait or 0) / PICOSECONDS_PER_MS
        stats.append(
            SqlDigestStat(
                digest=str(digest),
                sample_sql=str(digest_text),
                calls=calls,
                total_ms=round(total_ms, 3),
                mean_ms=round(total_ms / calls, 4) if calls else 0.0,
                rows_examined=int(rows_examined or 0),
                rows_sent=int(rows_sent or 0),
                no_index_used=int(no_index or 0) > 0,
            )
        )

    return stats[:limit] if limit is not None else stats


def require_base_url(base_url: str | None) -> str:
    """The arm's URL, or a refusal naming why there is no default.

    Ports are chosen at runtime by the provision phase, so a module-level
    `http://localhost:18080` could only ever name one repo's configured port
    and be wrong everywhere else. Callers that have no URL are a bug, not a
    case to paper over with a constant.
    """
    if base_url:
        return base_url
    raise ToolError(
        "no base URL was supplied. The arms are provisioned onto free ports "
        "chosen at runtime, so there is no default to fall back to — pass "
        "--url, or the base_url the provision phase recorded."
    )
