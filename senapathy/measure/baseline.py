from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
import typer
from pydantic import BaseModel

from ..contracts import RequestCorpus, SqlDigestStat
from ..errors import ContractViolation, SenapathyError, ToolError
from ..ids import content_hash
from .bench import DEFAULT_ITERATIONS, DEFAULT_TIMEOUT_S, DEFAULT_WARMUP, run_benchmark
from .corpus import DEFAULT_CORPUS_PATH, load_corpus
from .sql import require_base_url, MySQLTarget, collect_digests, reset_digests

# No default port. The arms are provisioned onto free ports chosen at
# runtime, so a module constant here could only ever name one repo's
# configured port and be wrong everywhere else. Callers pass the URL
# the provisioning phase actually bound.
DEFAULT_BASE_URL: str | None = None
HEALTH_PATH = "/actuator/health"
TOP_DIGESTS = 10


# ────────────────────────────  health check  ────────────────────────────

def health_check(base_url: str, timeout_s: float = 10.0) -> dict[str, Any]:
    """Confirm the app answers before spending ten minutes measuring nothing.

    Raises ToolError with an actionable message if it does not.
    """
    url = base_url.rstrip("/") + HEALTH_PATH
    try:
        with httpx.Client(timeout=timeout_s) as client:
            response = client.get(url)
    except httpx.HTTPError as exc:
        raise ToolError(
            f"app is not reachable at {base_url}: {type(exc).__name__}: {exc}",
            suggestion="Start the target app with the `senapathy` profile and "
            f"confirm it is listening, e.g. curl {url}",
        ) from exc

    if response.status_code != 200:
        raise ToolError(
            f"health check {url} returned HTTP {response.status_code}",
            suggestion="The app is listening but not healthy. Check its logs "
            "and its database connection.",
        )

    try:
        payload = response.json()
    except ValueError:
        payload = {}
    status = payload.get("status")
    if status is not None and status != "UP":
        raise ToolError(
            f"health check {url} reports status {status!r}",
            suggestion=f"Component detail: {json.dumps(payload.get('components', {}))[:400]}",
        )
    return {"url": url, "http_status": response.status_code, "status": status or "UP",
            "body": payload}


# ──────────────────────────  derived arithmetic  ─────────────────────────
# Invariant 5: nothing here is ever handed to a model to compute. These are the
# numbers a report is allowed to quote, and they are all computed in Python
# from recorded samples and recorded digest rows.

# JDBC opens every transaction with a handful of session-control statements.
# They are real rows in the digest table and stay in the artifact, but counting
# them as application SQL would put `SET autocommit = ?` at the top of a
# per-request call ranking and bury the actual N+1.
_SESSION_CONTROL_PREFIXES = (
    "SET ",
    "COMMIT",
    "ROLLBACK",
    "START TRANSACTION",
    "BEGIN",
    "SHOW ",
    "SELECT @@",
)


def is_session_control(digest: SqlDigestStat) -> bool:
    """True for connection/transaction bookkeeping rather than application SQL."""
    sql = " ".join(digest.sample_sql.split()).upper()
    return sql.startswith(_SESSION_CONTROL_PREFIXES)


def derive_facts(bench: dict[str, Any], digests: list[SqlDigestStat]) -> dict[str, Any]:
    """Turn raw samples and digest rows into the facts a report may quote."""
    answered = [s for s in bench["samples"] if s["status"] is not None]
    measured_requests = len(answered)
    measured_latency_ms = sum(s["latency_ms"] for s in answered)

    app_digests = [d for d in digests if not is_session_control(d)]
    session_digests = [d for d in digests if is_session_control(d)]

    total_statements = sum(d.calls for d in digests)
    app_statements = sum(d.calls for d in app_digests)
    total_sql_ms = sum(d.total_ms for d in digests)
    total_rows_examined = sum(d.rows_examined for d in digests)
    total_rows_sent = sum(d.rows_sent for d in digests)

    # The N+1 candidate: the application digest called most often relative to
    # the number of requests. A statement that runs once per request is
    # request-scoped; one that runs several times per request is row-scoped,
    # and that gap is the N+1. This records the ratio; calling it a defect is a
    # later phase's job.
    busiest = max(app_digests, key=lambda d: d.calls, default=None)

    return {
        "measured_requests": measured_requests,
        "measured_latency_ms": round(measured_latency_ms, 3),
        "total_sql_statements": total_statements,
        "statements_per_request": (
            round(total_statements / measured_requests, 2) if measured_requests else 0.0
        ),
        "app_sql_statements": app_statements,
        "app_statements_per_request": (
            round(app_statements / measured_requests, 2) if measured_requests else 0.0
        ),
        "session_control_statements": total_statements - app_statements,
        "total_sql_ms": round(total_sql_ms, 3),
        "sql_share_of_latency_pct": (
            round(100.0 * total_sql_ms / measured_latency_ms, 2)
            if measured_latency_ms > 0
            else 0.0
        ),
        "total_rows_examined": total_rows_examined,
        "total_rows_sent": total_rows_sent,
        "distinct_digests": len(digests),
        "distinct_app_digests": len(app_digests),
        "distinct_session_digests": len(session_digests),
        "digests_without_index": sum(1 for d in digests if d.no_index_used),
        "busiest_digest": (
            {
                "digest": busiest.digest,
                "sample_sql": busiest.sample_sql,
                "calls": busiest.calls,
                "calls_per_request": (
                    round(busiest.calls / measured_requests, 2) if measured_requests else 0.0
                ),
                "total_ms": busiest.total_ms,
                "rows_examined": busiest.rows_examined,
            }
            if busiest is not None
            else None
        ),
    }


def baseline_hash(document: dict[str, Any]) -> str:
    """The identity a downstream artifact stamps as `baseline_hash` to record
    which baseline it was derived from, and that a resume's skip check
    recomputes from THIS run's current baseline.json to tell a stale
    downstream artifact from a current one.

    Content-hashed like `corpus_hash`/`oracle_hash`, so the same baseline
    document always hashes identically and a changed one never does.
    """
    return content_hash(document)


# ────────────────────────────  the measurement  ──────────────────────────

def run_baseline(
    base_url: str | None = DEFAULT_BASE_URL,
    corpus_path: str | Path | None = None,
    warmup: int = DEFAULT_WARMUP,
    iterations: int = DEFAULT_ITERATIONS,
    *,
    mysql: MySQLTarget | None = None,
    runs_dir: str | Path = Path(".senapathy/runs"),
    timeout_s: float = DEFAULT_TIMEOUT_S,
    collect_sql: bool = True,
    expects_sql: bool = False,
) -> tuple[dict[str, Any], Path]:
    """Health-check, warm up, reset digests, measure, collect digests, write JSON.

    Returns (document, path_to_baseline_json).

    The ordering matters and is the whole reason this function exists: the
    digest reset happens after the last warmup request and before the first
    measured one, via run_benchmark's on_measure_start hook. Reset any earlier
    and warmup statements pollute the counts; any later and the first measured
    requests are missing from them.

    `expects_sql` is the comprehension model's own claim that this endpoint
    touches the database (it reads `tables` or `data_access`) — the caller's
    job, since this function has no EndpointModel of its own. "I measured
    something" is not "I measured the endpoint": a baseline with no 2xx
    response, or with zero SQL statements for an endpoint the model says
    reads the database, is a baseline of a request that never ran, and is
    refused rather than written.
    """
    # Built only when needed: --no-sql means no database is required at all,
    # and MySQLTarget no longer carries defaults to fall back on silently.
    target: MySQLTarget | None = (mysql or MySQLTarget.from_env()) if collect_sql else None
    corpus: RequestCorpus = load_corpus(corpus_path)

    resolved_url = require_base_url(base_url)
    health = health_check(resolved_url)
    digests: list[SqlDigestStat]
    if collect_sql:
        assert target is not None
        digest_target: MySQLTarget = target
        # Fail fast on a privilege problem now rather than after the run.
        reset_digests(digest_target)
        bench = run_benchmark(
            resolved_url, corpus, warmup=warmup, iterations=iterations,
            timeout_s=timeout_s,
            on_measure_start=lambda: reset_digests(digest_target),
        )
        digests = collect_digests(digest_target)
    else:
        bench = run_benchmark(
            resolved_url, corpus, warmup=warmup, iterations=iterations,
            timeout_s=timeout_s, on_measure_start=None,
        )
        digests = []

    facts = derive_facts(bench, digests)

    if not any(code.startswith("2") for code in bench["status_counts"]):
        failing = bench.get("first_non_2xx") or {}
        raise ToolError(
            "refusing to write a baseline in which no measured request "
            f"succeeded: status counts: {bench['status_counts']}; first "
            f"response ({failing.get('request_id', '?')}, status "
            f"{failing.get('status')}): {failing.get('body')!r}",
            suggestion="A baseline measures the endpoint under test, not its "
            "error path. Check the corpus against the comprehension model's "
            "request_shape -- a wrong or missing required parameter "
            "produces exactly this.",
        )

    if collect_sql and expects_sql and facts["total_sql_statements"] == 0:
        failing = bench.get("first_non_2xx") or {}
        raise ToolError(
            "refusing to write a baseline that recorded zero SQL statements "
            "for an endpoint the comprehension model says touches the "
            f"database: status counts: {bench['status_counts']}"
            + (f"; first non-2xx response ({failing.get('request_id')}, "
               f"status {failing.get('status')}): {failing.get('body')!r}"
               if failing else ""),
            suggestion="Measuring something that never ran is not a "
            "baseline. Confirm the requests reach the endpoint's actual "
            "route and that performance_schema is capturing this "
            "connection, not a different one.",
        )

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = Path(runs_dir) / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    out_path = run_dir / "baseline.json"

    document: dict[str, Any] = {
        "schema_version": 1,
        "kind": "baseline",
        "arm": "blue",
        "captured_at": timestamp,
        # Top-level, not just nested under "corpus": this is what a resume's
        # skip check reads to tell a baseline measured against the corpus
        # that is on disk now from one measured against a corpus that has
        # since changed.
        "corpus_hash": corpus.hash,
        "target": {
            "base_url": base_url,
            "health": health,
            "mysql": target.dsn() if target is not None else None,
        },
        "corpus": {
            "path": str(Path(corpus_path) if corpus_path else DEFAULT_CORPUS_PATH),
            "hash": corpus.hash,
            "requests": [r.model_dump(mode="json") for r in corpus.benchmark],
        },
        "benchmark": bench,
        "sql_digests": [d.model_dump(mode="json") for d in digests],
        "derived": facts,
    }

    out_path.write_text(
        json.dumps(document, indent=2, ensure_ascii=False, default=_json_default) + "\n",
        encoding="utf-8",
    )
    return document, out_path


def _json_default(obj: Any) -> Any:
    if isinstance(obj, BaseModel):
        return obj.model_dump(mode="json")
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"not JSON serialisable: {type(obj).__name__}")


# ──────────────────────────────  rendering  ──────────────────────────────
# Pure string building. The command below is the only thing that prints.

def _truncate(text: str, width: int) -> str:
    collapsed = " ".join(text.split())
    return collapsed if len(collapsed) <= width else collapsed[: width - 1] + "…"


def render_summary(document: dict[str, Any], top: int = TOP_DIGESTS) -> str:
    bench = document["benchmark"]
    facts = document["derived"]
    latency = bench["latency"]
    if isinstance(latency, BaseModel):
        latency = latency.model_dump(mode="json")

    lines: list[str] = []
    lines.append("")
    lines.append(f"baseline  {document['captured_at']}   arm={document['arm']}")
    lines.append(f"  app     {document['target']['base_url']}  (health: {document['target']['health']['status']})")
    lines.append(f"  mysql   {document['target']['mysql']}")
    lines.append(f"  corpus  {document['corpus']['hash']}  ({len(document['corpus']['requests'])} requests)")
    lines.append(
        f"  run     {bench['warmup']} warmup discarded, "
        f"{bench['iterations']} measured, {bench['wall_ms'] / 1000:.1f}s wall"
    )

    lines.append("")
    lines.append("LATENCY (ms, from raw samples)")
    lines.append(f"  {'n':>8} {'p50':>10} {'p95':>10} {'p99':>10} {'mean':>10} {'stdev':>10}")
    lines.append(
        f"  {latency['n']:>8} {latency['p50_ms']:>10.1f} {latency['p95_ms']:>10.1f} "
        f"{latency['p99_ms']:>10.1f} {latency['mean_ms']:>10.1f} {latency['stdev_ms']:>10.1f}"
    )
    statuses = ", ".join(f"{k}×{v}" for k, v in bench["status_counts"].items())
    lines.append(f"  status: {statuses}")
    if bench["transport_failures"]:
        lines.append(f"  transport failures (excluded from stats): {bench['transport_failures']}")

    lines.append("")
    lines.append("PER REQUEST (ms)")
    lines.append(f"  {'request':<40} {'n':>4} {'p50':>9} {'p95':>9} {'status':>12}")
    per_request = bench["per_request"]
    def _p50(item: tuple[str, Any]) -> float:
        stats = item[1]["latency"]
        stats = stats.model_dump(mode="json") if isinstance(stats, BaseModel) else stats
        return stats["p50_ms"]
    for request_id, entry in sorted(per_request.items(), key=_p50, reverse=True):
        stats = entry["latency"]
        stats = stats.model_dump(mode="json") if isinstance(stats, BaseModel) else stats
        codes = ",".join(f"{k}×{v}" for k, v in entry["statuses"].items())
        lines.append(
            f"  {_truncate(request_id, 40):<40} {entry['n']:>4} "
            f"{stats['p50_ms']:>9.1f} {stats['p95_ms']:>9.1f} {codes:>12}"
        )

    digests = document["sql_digests"]
    lines.append("")
    lines.append(f"TOP {min(top, len(digests))} SQL DIGESTS (of {facts['distinct_digests']}, by total time)")
    lines.append(
        f"  {'calls':>7} {'calls/req':>10} {'total ms':>10} {'mean ms':>9} "
        f"{'rows exam':>11} {'no-idx':>7}  sql"
    )
    requests = facts["measured_requests"] or 1
    for digest in digests[:top]:
        marker = "  " if is_session_control(SqlDigestStat.model_validate(digest)) else "* "
        lines.append(
            f"  {digest['calls']:>7} {digest['calls'] / requests:>10.2f} "
            f"{digest['total_ms']:>10.1f} {digest['mean_ms']:>9.3f} "
            f"{digest['rows_examined']:>11} {'YES' if digest['no_index_used'] else '-':>7}  "
            f"{marker}{_truncate(digest['sample_sql'], 94)}"
        )
    lines.append("  (* = application SQL; unmarked rows are JDBC session control)")

    lines.append("")
    lines.append("DERIVED")
    lines.append(f"  measured requests           {facts['measured_requests']}")
    lines.append(
        f"  total SQL statements        {facts['total_sql_statements']} "
        f"({facts['app_sql_statements']} application, "
        f"{facts['session_control_statements']} session control)"
    )
    lines.append(f"  statements per request      {facts['statements_per_request']}")
    lines.append(f"  app statements per request  {facts['app_statements_per_request']}")
    lines.append(f"  total SQL time              {facts['total_sql_ms']:.1f} ms")
    lines.append(f"  measured latency (summed)   {facts['measured_latency_ms']:.1f} ms")
    lines.append(f"  SQL share of latency        {facts['sql_share_of_latency_pct']:.2f} %")
    lines.append(f"  rows examined / rows sent   {facts['total_rows_examined']} / {facts['total_rows_sent']}")
    lines.append(f"  digests not using an index  {facts['digests_without_index']}")
    busiest = facts["busiest_digest"]
    if busiest:
        lines.append(
            f"  busiest app digest          {busiest['calls']} calls "
            f"({busiest['calls_per_request']} per request), "
            f"{busiest['total_ms']:.1f} ms, {busiest['rows_examined']} rows examined"
        )
        lines.append(f"                              {_truncate(busiest['sample_sql'], 90)}")
    lines.append("")
    return "\n".join(lines)


# ────────────────────────────────  command  ──────────────────────────────

def baseline_command(
    url: str = typer.Option(DEFAULT_BASE_URL, "--url", help="Base URL of the running app."),
    corpus: Path | None = typer.Option(
        None, "--corpus", help="Request corpus JSON. Defaults to the shipped corpus."
    ),
    warmup: int = typer.Option(DEFAULT_WARMUP, "--warmup", help="Discarded warmup requests."),
    iterations: int = typer.Option(
        DEFAULT_ITERATIONS, "--iterations", "-n", help="Measured requests."
    ),
    runs_dir: Path = typer.Option(
        Path(".senapathy/runs"), "--runs-dir", help="Where the run directory is created."
    ),
    timeout: float = typer.Option(DEFAULT_TIMEOUT_S, "--timeout", help="Per-request timeout (s)."),
    top: int = typer.Option(TOP_DIGESTS, "--top", help="SQL digests to show."),
    no_sql: bool = typer.Option(False, "--no-sql", help="Skip performance_schema capture."),
) -> None:
    """Measure the blue-arm baseline and capture SQL evidence."""
    try:
        document, path = run_baseline(
            base_url=url,
            corpus_path=corpus,
            warmup=warmup,
            iterations=iterations,
            runs_dir=runs_dir,
            timeout_s=timeout,
            collect_sql=not no_sql,
        )
    except (ToolError, ContractViolation) as exc:
        typer.secho(f"baseline failed: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc
    except SenapathyError as exc:
        typer.secho(f"baseline failed: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc

    typer.echo(render_summary(document, top=top))
    typer.echo(f"written: {path}")
