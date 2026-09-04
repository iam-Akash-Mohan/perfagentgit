"""Record what the endpoint returns TODAY, on unmodified code.

This module is the plumbing half of P13: fire the corpus, capture responses and
database state, stamp the recording with the target repo's commit, content-hash
the bundle, and refuse to overwrite it. The verdict logic lives in
`compare.py` and `canonicalise.py`.

Two properties matter more than anything else here:

* **Record and replay share one code path.** `_capture()` is used by both
  `record_oracle()` and `replay_live()`. If recording and replaying used
  separate request code, a difference in how *we* send the request would show
  up as a behaviour change in the *application*, and you would spend a day
  chasing a diff that was ours.

* **The recording is immutable.** `.perfagent/oracle.json` is never truncated.
  `--force` archives the existing bundle under its own content hash first, so a
  "frozen" artifact that was in fact replaced is always recoverable.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any, Iterable

import httpx

from ..contracts import (BehaviourOracle, RecordedResponse, RequestCorpus,
                         RequestSpec, TolerancePolicy)
from ..errors import ContractViolation, ToolError
from ..ids import content_hash
from ..measure.sql import MySQLTarget, connect
from .canonicalise import load_policy

ORACLE_PATH = Path(".perfagent/oracle.json")
DEFAULT_TIMEOUT_S = 120.0

# P20's RepoProfile should eventually own this; today it is duplicated with
# perfagent/tools.py:REPO_ROOT.
DEFAULT_TARGET_REPO = Path("/Users/akashm/Documents/mini-spring")

# Response headers that carry behaviour. Deliberately an allowlist: `Date`
# changes every request and `Content-Length` changes with JSON whitespace, so
# recording either would guarantee a false failure on every check. The body is
# compared directly, so Content-Length adds nothing anyway.
HEADERS_OF_INTEREST = (
    "content-type",
    "location",
    "etag",
    "cache-control",
    "www-authenticate",
    "allow",
    "retry-after",
)

# CAST(... AS CHAR) on binary data can produce invalid UTF-8; HEX() is safe.
_BINARY_TYPES = frozenset(
    {"blob", "tinyblob", "mediumblob", "longblob", "binary", "varbinary", "bit"}
)


# ──────────────────────────────  provenance  ─────────────────────────────


def base_commit(repo_path: str | Path | None = None) -> str:
    """The commit the recording was made against.

    A dirty worktree is reported as `<sha>+dirty` rather than refused. Refusing
    would block legitimate work; silently returning the clean sha would make the
    oracle claim provenance it does not have. `+dirty` in the artifact is the
    honest answer, and it is visible in the bundle hash.

    argv list, never `shell=True` — invariant 1 governs the agent's tool
    surface, but harness code has no excuse for shelling out either.
    """
    repo = Path(
        repo_path
        if repo_path is not None
        else os.environ.get("PERFAGENT_TARGET_REPO", DEFAULT_TARGET_REPO)
    )
    if not (repo / ".git").exists():
        raise ContractViolation(
            f"{repo} is not a git repository, so the recording cannot be "
            "stamped with a base commit. Set PERFAGENT_TARGET_REPO."
        )

    def git(*args: str) -> str:
        result = subprocess.run(
            ["git", "-C", str(repo), *args],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            raise ContractViolation(
                f"git {' '.join(args)} failed in {repo}: {result.stderr.strip()}"
            )
        return result.stdout.strip()

    sha = git("rev-parse", "HEAD")
    dirty = bool(git("status", "--porcelain"))
    return f"{sha}+dirty" if dirty else sha


# ─────────────────────────────  database state  ──────────────────────────


def _row_expression(column: str, data_type: str) -> str:
    if "`" in column:
        raise ContractViolation(f"refusing to build SQL for column name {column!r}")
    quoted = f"`{column}`"
    rendered = f"HEX({quoted})" if data_type.lower() in _BINARY_TYPES else f"CAST({quoted} AS CHAR)"
    # CHAR(0) marks NULL. Without it, CONCAT_WS skips NULLs and ('a', NULL, 'b')
    # collides with ('a', 'b', NULL) — a column swap would go undetected.
    return f"IFNULL({rendered}, CHAR(0))"


def db_table_snapshot(target: MySQLTarget) -> dict[str, Any]:
    """Per-table row count and two order-insensitive row checksums.

    The point is to catch a patch that writes something it should not. A search
    endpoint that starts UPDATEing a counter, an optimisation that "warms" a
    table, an accidental cascade — none of those show up in a response body.

    Two aggregates rather than one, because each misses something the other
    catches: BIT_XOR cancels a pair of identical rows (a XOR a = 0), and SUM
    cannot distinguish some rearrangements of numeric content. Both are
    order-insensitive, which matters because row order is not part of the
    application's behaviour.

    The column list is recorded alongside, so a schema change is visible rather
    than silently changing what the checksum means.
    """
    from mysql.connector import Error as MySQLError

    snapshot: dict[str, Any] = {}
    with connect(target) as conn:
        cursor = conn.cursor()
        try:
            cursor.execute(
                """
                SELECT TABLE_NAME
                FROM information_schema.TABLES
                WHERE TABLE_SCHEMA = %s AND TABLE_TYPE = 'BASE TABLE'
                ORDER BY TABLE_NAME
                """,
                (target.database,),
            )
            tables = [str(row[0]) for row in cursor.fetchall()]

            for table in tables:
                if "`" in table:
                    raise ContractViolation(f"refusing to build SQL for table {table!r}")
                cursor.execute(
                    """
                    SELECT COLUMN_NAME, DATA_TYPE
                    FROM information_schema.COLUMNS
                    WHERE TABLE_SCHEMA = %s AND TABLE_NAME = %s
                    ORDER BY ORDINAL_POSITION
                    """,
                    (target.database, table),
                )
                columns = [(str(c), str(t)) for c, t in cursor.fetchall()]
                if not columns:
                    snapshot[table] = {"count": 0, "sum_crc32": 0, "xor_crc32": 0, "columns": []}
                    continue

                row_text = "CONCAT_WS(CHAR(31), " + ", ".join(
                    _row_expression(name, dtype) for name, dtype in columns
                ) + ")"
                cursor.execute(
                    f"SELECT COUNT(*), "  # noqa: S608 - identifiers are validated above
                    f"COALESCE(SUM(CRC32(rowtext)), 0), "
                    f"COALESCE(BIT_XOR(CRC32(rowtext)), 0) "
                    f"FROM (SELECT {row_text} AS rowtext "
                    f"FROM `{target.database}`.`{table}`) AS snapshot_rows"
                )
                count, crc_sum, crc_xor = cursor.fetchone()
                snapshot[table] = {
                    "count": int(count or 0),
                    "sum_crc32": int(crc_sum or 0),
                    "xor_crc32": int(crc_xor or 0),
                    "columns": [name for name, _ in columns],
                }
        except MySQLError as exc:
            raise ToolError(
                f"cannot snapshot database state as {target.dsn()}: {exc}",
                suggestion="The oracle needs SELECT on the application schema "
                "and on information_schema. Set PERFAGENT_MYSQL_USER / "
                "_PASSWORD to an account that has it.",
            ) from exc
        finally:
            cursor.close()

    return snapshot


def db_state_hash(snapshot: dict[str, Any]) -> str:
    """Content hash of a table snapshot. This is what the verdict compares."""
    return content_hash(snapshot)


# ──────────────────────────────  firing  ─────────────────────────────────


def _headers_of_interest(response: httpx.Response) -> dict[str, str]:
    return {
        name: response.headers[name]
        for name in HEADERS_OF_INTEREST
        if name in response.headers
    }


def _decode_body(response: httpx.Response) -> Any:
    """Parsed JSON when the response says JSON, raw text otherwise.

    A JSON content-type whose body will not parse is recorded as its raw text,
    not raised on — that IS the endpoint's behaviour today, and the oracle's job
    is to record behaviour, not to like it.
    """
    if not response.content:
        return None
    content_type = response.headers.get("content-type", "")
    if "json" in content_type.lower():
        try:
            return response.json()
        except ValueError:
            return response.text
    return response.text


def _capture(
    base_url: str,
    specs: Iterable[RequestSpec],
    *,
    target: MySQLTarget | None,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    client: httpx.Client | None = None,
) -> tuple[list[RecordedResponse], dict[str, dict[str, Any]]]:
    """Fire every spec once, sequentially, in the order given.

    Sequential and once-each, both on purpose. Concurrency would make the
    per-request database snapshots meaningless (you could not say which request
    a change followed), and firing a request twice would hide any endpoint whose
    second call differs from its first — which is itself behaviour worth pinning.

    Order matters and is part of the recording: the database snapshot after
    request N reflects every request up to N. `replay_live()` therefore has to
    walk the corpus in the same order, which is why both go through here.

    Returns the recorded responses and, separately, the per-request table
    detail. The detail is diagnostics only — never verdict input.
    """
    owns_client = client is None
    http = client or httpx.Client(timeout=timeout_s, follow_redirects=False)

    recorded: list[RecordedResponse] = []
    detail: dict[str, dict[str, Any]] = {}
    try:
        for spec in specs:
            url = base_url.rstrip("/") + spec.path
            try:
                response = http.request(
                    spec.method.upper(),
                    url,
                    json=spec.body,
                    headers=spec.headers or None,
                )
                _ = response.content
            except httpx.HTTPError as exc:
                # An oracle with a hole in it is worse than no oracle: the hole
                # is exactly where a regression would hide.
                raise ToolError(
                    f"request {spec.id} ({spec.method} {url}) failed at the "
                    f"transport layer: {type(exc).__name__}: {exc}",
                    suggestion="The app must be up and healthy for the whole "
                    "recording. Fix the environment and record again; a "
                    "partial oracle is not recorded.",
                ) from exc

            table_state = db_table_snapshot(target) if target is not None else None
            if table_state is not None:
                detail[spec.id] = table_state

            recorded.append(
                RecordedResponse(
                    request_id=spec.id,
                    status=response.status_code,
                    body=_decode_body(response),
                    headers_of_interest=_headers_of_interest(response),
                    db_state_hash=db_state_hash(table_state) if table_state is not None else None,
                    # WireMock is not stood up until P19; until then no external
                    # call journal exists and this stays empty by construction.
                    external_calls_made=[],
                )
            )
    finally:
        if owns_client:
            http.close()

    return recorded, detail


# ──────────────────────────────  bundling  ───────────────────────────────


def bundle_hash(oracle_fields: dict[str, Any]) -> str:
    """Content hash of everything in the bundle except the hash itself.

    Both the writer and the reader go through this one function, so a bundle can
    never be written with a hash that its own reader would reject.
    """
    payload = {k: v for k, v in oracle_fields.items() if k != "hash"}
    return content_hash(payload)


def dbstate_path(oracle_path: str | Path = ORACLE_PATH) -> Path:
    """Where the advisory per-table detail goes, next to the bundle."""
    path = Path(oracle_path)
    return path.with_name(path.stem + ".dbstate.json")


def load_oracle(path: str | Path = ORACLE_PATH) -> BehaviourOracle:
    """Read the bundle and re-verify its content hash.

    The stated hash is never trusted. A bundle whose content no longer hashes to
    its own stamp has been edited, and an edited oracle is not an oracle.
    """
    oracle_path = Path(path)
    if not oracle_path.exists():
        raise ContractViolation(
            f"no oracle at {oracle_path}. Run `perfagent oracle record` against "
            "unmodified code first."
        )
    try:
        payload = json.loads(oracle_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ContractViolation(f"{oracle_path} is not valid JSON: {exc}") from exc

    oracle = BehaviourOracle.model_validate(payload)
    recomputed = bundle_hash(oracle.model_dump(mode="json"))
    if recomputed != oracle.hash:
        raise ContractViolation(
            f"{oracle_path} states hash {oracle.hash} but its content hashes to "
            f"{recomputed}. The recording has been modified since it was made; "
            "re-record against unmodified code rather than trusting it."
        )
    return oracle


def write_oracle(
    oracle: BehaviourOracle,
    detail: dict[str, Any],
    path: str | Path = ORACLE_PATH,
    *,
    force: bool = False,
) -> Path:
    """Write the bundle, refusing to clobber an existing one.

    `force` archives rather than deletes. "Immutable, with a destructive escape
    hatch" is just mutable with extra steps.
    """
    oracle_path = Path(path)
    oracle_path.parent.mkdir(parents=True, exist_ok=True)

    if oracle_path.exists():
        if not force:
            existing = json.loads(oracle_path.read_text(encoding="utf-8")).get("hash", "unknown")
            raise ContractViolation(
                f"{oracle_path} already exists (hash {existing}). The oracle is "
                "meant to be immutable: it is the record of behaviour BEFORE any "
                "patch, and re-recording after a patch would make it agree with "
                "whatever the patch did. Pass --force only if you are certain "
                "the code under test is unmodified."
            )
        archive = oracle_path.with_name(
            f"{oracle_path.stem}.{json.loads(oracle_path.read_text(encoding='utf-8')).get('hash', 'unknown')}{oracle_path.suffix}"
        )
        oracle_path.replace(archive)

    oracle_path.write_text(
        json.dumps(oracle.model_dump(mode="json"), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    dbstate_path(oracle_path).write_text(
        json.dumps(
            {"oracle_hash": oracle.hash, "tables": detail},
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return oracle_path


# ────────────────────────────────  API  ──────────────────────────────────


def record_oracle(
    base_url: str,
    corpus: RequestCorpus,
    *,
    policy: TolerancePolicy | None = None,
    repo_path: str | Path | None = None,
    mysql: MySQLTarget | None = None,
    capture_db: bool = True,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    path: str | Path = ORACLE_PATH,
    force: bool = False,
    client: httpx.Client | None = None,
) -> dict[str, Any]:
    """Capture behaviour on unmodified code and freeze it.

    Fires the benchmark set and then the held-back set, once each, sequentially.
    Both go into one `recorded` list; which set a response belonged to is
    recovered at check time from the corpus, whose hash is stamped here. That
    keeps `RecordedResponse` as the contract defines it — no invented field.

    Returns the written bundle as a dict (the exact JSON on disk), so the caller
    sees what was frozen and not a re-serialisation of it.
    """
    if not corpus.benchmark:
        raise ContractViolation("cannot record an oracle from an empty benchmark set")

    resolved_policy = policy if policy is not None else load_policy()
    target = (mysql or MySQLTarget.from_env()) if capture_db else None

    # Pre-flight state, before anything is fired. `check` compares its own
    # pre-flight against this and refuses to run on a database that is not in
    # the state the oracle was recorded against — otherwise every cumulative
    # per-request hash diverges for reasons that have nothing to do with the
    # patch, and the oracle fails on itself.
    initial = db_table_snapshot(target) if target is not None else None

    recorded, detail = _capture(
        base_url,
        [*corpus.benchmark, *corpus.held_back],
        target=target,
        timeout_s=timeout_s,
        client=client,
    )

    fields: dict[str, Any] = {
        "corpus_hash": corpus.hash,
        "recorded": [r.model_dump(mode="json") for r in recorded],
        "policy": resolved_policy.model_dump(mode="json"),
        "base_commit": base_commit(repo_path),
    }
    oracle = BehaviourOracle.model_validate({**fields, "hash": bundle_hash(fields)})

    write_oracle(
        oracle,
        {"initial": initial, "after_request": detail},
        path=path,
        force=force,
    )
    return oracle.model_dump(mode="json")


def replay_live(
    base_url: str,
    corpus: RequestCorpus,
    *,
    mysql: MySQLTarget | None = None,
    capture_db: bool = True,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    client: httpx.Client | None = None,
) -> list[RecordedResponse]:
    """Fire the corpus against a live app and return what came back.

    Identical code path to `record_oracle`, same order, same body decoding, same
    header allowlist. Nothing here reads the oracle, so a replay cannot be
    influenced by what was recorded.
    """
    target = (mysql or MySQLTarget.from_env()) if capture_db else None
    recorded, _ = _capture(
        base_url,
        [*corpus.benchmark, *corpus.held_back],
        target=target,
        timeout_s=timeout_s,
        client=client,
    )
    return recorded


def recorded_initial_db_state(path: str | Path = ORACLE_PATH) -> dict[str, Any] | None:
    """The pre-flight table detail from the advisory side file, if present.

    Advisory: used to fail fast on a mis-seeded database and to say WHICH table
    drifted. Never an input to the verdict — the verdict reads only the hashes
    inside the content-hashed bundle.
    """
    side = dbstate_path(path)
    if not side.exists():
        return None
    try:
        payload = json.loads(side.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    initial = payload.get("initial")
    return initial if isinstance(initial, dict) else None


def db_state_drift(
    recorded: dict[str, Any] | None, live: dict[str, Any] | None
) -> list[str]:
    """Name the tables whose snapshot differs. Diagnostics, not a verdict.

    `check` uses this as a pre-flight guard. Every per-request database hash is
    CUMULATIVE — the snapshot after request N reflects requests 1..N — so a
    database that does not start where the recording started makes every hash
    after the first mutation diverge for reasons that have nothing to do with
    the patch. Failing here with the table name beats failing later with
    `db_state_matched: false` and no lead.
    """
    if recorded is None or live is None:
        return []
    drifted: list[str] = []
    for table in sorted(set(recorded) | set(live)):
        before, after = recorded.get(table), live.get(table)
        if before != after:
            if before is None:
                drifted.append(f"{table} (absent when recorded)")
            elif after is None:
                drifted.append(f"{table} (missing now)")
            else:
                drifted.append(
                    f"{table} (count {before.get('count')} -> {after.get('count')})"
                )
    return drifted
