from __future__ import annotations

import os
import re
from contextlib import contextmanager
from typing import Any, Iterator

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from ..contracts import SqlDigestStat
from ..contracts.repo import DatabaseKind, DatabaseProfile
from ..errors import ToolError

# performance_schema timers are in picoseconds.
PICOSECONDS_PER_MS = 1_000_000_000

DIGEST_TABLE = "performance_schema.events_statements_summary_by_digest"

_ENV_PREFIX = "SENAPATHY_MYSQL_"
# The admin login is sourced separately from the app's own datasource
# credentials — see the class docstring — so it gets its own prefix rather
# than overloading _ENV_PREFIX's USER/PASSWORD.
_ADMIN_USER_VAR = "SENAPATHY_MYSQL_ADMIN_USER"
_ADMIN_PASSWORD_VAR = "SENAPATHY_MYSQL_ADMIN_PASSWORD"

# jdbc:mysql://host[:port]/database — the same shape onboarding's own
# detector matches, kept independent of it so this module has no dependency
# on `onboard.py`.
_JDBC = re.compile(
    r"jdbc:\w+://(?P<host>[\w.\-]+)(?::(?P<port>\d+))?/(?P<db>[\w\-]+)"
)

# MySQL's own protocol default — not a value guessed for any one repo, the
# same way a URL with no :port means :80 for HTTP.
MYSQL_DEFAULT_PORT = 3306


class MySQLTarget(BaseModel):
    """Where the app's database lives, and who to log in as.

    performance_schema is not readable by the application user (the app user
    has rights on its own schema only), and TRUNCATE on a performance_schema
    table needs DROP. So evidence collection uses an admin login that is
    deliberately separate from whatever the app is configured with — this
    connection never touches application data.

    No field has a default. A default that happens to be right for one repo
    is exactly how a hardcoded `127.0.0.1:3307/mini` survived a sweep that
    banned the string "mini-spring" but not the database named "mini" — a
    target for a real run must come from that run's own detected profile
    (`from_profile`) or an explicit environment override (`from_env`), and
    anything else is a bug in whoever constructed it, not a value worth
    guessing.
    """

    host: str
    port: int
    user: str
    password: str
    database: str = Field(
        description="The APPLICATION schema. Used to filter digests, not to "
        "select a default database — the connection stays schema-less so that "
        "our own bookkeeping statements do not show up in the evidence.",
    )

    @classmethod
    def from_profile(cls, database: DatabaseProfile) -> "MySQLTarget":
        """The run's own target: schema identity from onboarding, admin login
        from the environment.

        This is the source of truth for an orchestrated run — `from_env` is
        an explicit override for standalone CLI use, not a default. Host,
        port and schema come from what onboarding detected in the repo's own
        config (falling back to parsing `jdbc_url` if a field was not split
        out); the admin login is NEVER the app's own detected credentials —
        performance_schema is routinely unreadable by the app user — so it is
        sourced from SENAPATHY_MYSQL_ADMIN_USER / _ADMIN_PASSWORD and its
        absence is refused by name rather than defaulting to root/root.
        """
        if database.kind is not DatabaseKind.MYSQL:
            raise ToolError(
                f"the detected datasource is {database.kind.value}, not "
                f"mysql; there is no MySQL target to build."
                + (f" {database.unsupported_reason}"
                   if database.unsupported_reason else "")
            )

        host, port, schema = database.host, database.port, database.database
        if database.jdbc_url and (host is None or port is None or schema is None):
            match = _JDBC.search(database.jdbc_url)
            if match:
                host = host or match.group("host")
                port = port or (int(match.group("port"))
                                if match.group("port") else None)
                schema = schema or match.group("db")
                # A jdbc: URL with no :port segment means the protocol
                # default, same as a browser treating a bare host as :80 --
                # not a guess, MySQL's own convention.
                if port is None:
                    port = MYSQL_DEFAULT_PORT

        missing = [name for name, value in
                  (("host", host), ("port", port), ("database", schema))
                  if value is None]
        if missing:
            raise ToolError(
                f"the detected database profile is missing {', '.join(missing)} "
                f"(jdbc_url={database.jdbc_url!r}); cannot build a MySQL "
                f"connection target from it.",
                suggestion="Onboarding reads the jdbc: URL from the repo's own "
                "config; check it names a host, port and database explicitly.",
            )

        load_dotenv()
        admin_user = os.environ.get(_ADMIN_USER_VAR)
        admin_password = os.environ.get(_ADMIN_PASSWORD_VAR)
        missing_admin = [name for name, value in
                         ((_ADMIN_USER_VAR, admin_user),
                          (_ADMIN_PASSWORD_VAR, admin_password))
                         if value is None]
        if missing_admin:
            raise ToolError(
                f"performance_schema needs an admin login, kept separate from "
                f"the application's own detected credentials; set "
                f"{' and '.join(missing_admin)}.",
                suggestion="Point it at an account with SELECT and DROP on "
                "performance_schema — not the app's datasource user, which "
                "usually cannot read it.",
            )
        assert admin_user is not None and admin_password is not None
        assert host is not None and port is not None and schema is not None

        return cls(host=str(host), port=int(port), user=admin_user,
                   password=admin_password, database=str(schema))

    @classmethod
    def from_env(cls) -> "MySQLTarget":
        """An explicit override for standalone CLI use — not this product's
        default. Every one of SENAPATHY_MYSQL_HOST / _PORT / _USER /
        _PASSWORD / _DATABASE is required; there is no value filled in for a
        variable left unset, because a value that fills itself in silently is
        what made this connect to the wrong database before.
        """
        load_dotenv()
        values: dict[str, Any] = {}
        missing: list[str] = []
        for field in ("host", "port", "user", "password", "database"):
            env_name = _ENV_PREFIX + field.upper()
            raw = os.environ.get(env_name)
            if raw is None:
                missing.append(env_name)
            else:
                values[field] = raw
        if missing:
            raise ToolError(
                f"missing {', '.join(missing)}. Set every SENAPATHY_MYSQL_* "
                f"variable to connect explicitly, or build the target from "
                f"this run's own detected profile with "
                f"MySQLTarget.from_profile() instead.",
            )
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
            "target repo) and point the connecting account (SENAPATHY_MYSQL_* "
            "or SENAPATHY_MYSQL_ADMIN_USER / _ADMIN_PASSWORD) at one with "
            "SELECT and DROP on performance_schema.",
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
                "performance_schema. Use an admin account — "
                "SENAPATHY_MYSQL_ADMIN_USER / _ADMIN_PASSWORD for a "
                "profile-built target, or SENAPATHY_MYSQL_USER / _PASSWORD "
                "for an explicit override.",
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
