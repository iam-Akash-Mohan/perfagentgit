from __future__ import annotations

from pathlib import Path
from typing import Any

from ..contracts import SqlDigestStat
from ..errors import ContractViolation
from ..measure.baseline import is_session_control

TOP_DIGESTS = 8
SQL_WIDTH = 88


# ─────────────────────────────  loading  ────────────────────────────────
# Inputs are read from disk exactly as the measurement phases wrote them.
# Nothing here re-runs a benchmark or recomputes a latency: if a number is not
# already in an artifact, this module does not have it.

def load_baseline(path: str | Path) -> dict[str, Any]:
    import json

    p = Path(path)
    if not p.exists():
        raise ContractViolation(
            f"baseline not found at {p}. Run `senapathy baseline` first."
        )
    doc = json.loads(p.read_text(encoding="utf-8"))
    for key in ("benchmark", "sql_digests", "derived"):
        if key not in doc:
            raise ContractViolation(f"{p} is not a baseline document (no '{key}')")
    return doc


def load_endpoint_model(path: str | Path) -> dict[str, Any]:
    import json

    p = Path(path)
    if not p.exists():
        raise ContractViolation(
            f"endpoint model not found at {p}. Run the comprehension agent first."
        )
    return json.loads(p.read_text(encoding="utf-8"))


def latest_run_dir(runs_dir: str | Path = Path(".senapathy/runs")) -> Path:
    """Most recent run directory. Names are UTC timestamps, so sorting works."""
    root = Path(runs_dir)
    candidates = sorted(d for d in root.glob("*") if (d / "baseline.json").exists())
    if not candidates:
        raise ContractViolation(
            f"no run with a baseline.json under {root}. Run `senapathy baseline` first."
        )
    return candidates[-1]


# ────────────────────────  computed, not inferred  ───────────────────────
# Invariant 5: the model is never asked to count, sum, divide or compare.
# Every derived number below is produced here and handed over as a finished
# fact, so the model's only job is to explain them.

def app_digests(baseline: dict[str, Any]) -> list[SqlDigestStat]:
    """Application SQL only, slowest first. JDBC session bookkeeping is
    dropped: `SET autocommit` running twice per request is not a bottleneck,
    and leaving it in makes a per-request call ranking useless."""
    rows = [SqlDigestStat.model_validate(d) for d in baseline["sql_digests"]]
    return [d for d in rows if not is_session_control(d)]


def repeated_digests(
    baseline: dict[str, Any], threshold: float = 1.5
) -> list[tuple[SqlDigestStat, float]]:
    """Digests executing more than `threshold` times per request, with the ratio.

    A statement running once per request is request-scoped. One running several
    times per request is row-scoped, and that gap is the N+1 signature. This
    reports the ratio; deciding whether it is the bottleneck is the point of
    the whole diagnosis and is NOT settled here.
    """
    requests = baseline["derived"]["measured_requests"] or 1
    out: list[tuple[SqlDigestStat, float]] = []
    for d in app_digests(baseline):
        ratio = d.calls / requests
        if ratio > threshold:
            out.append((d, ratio))
    return sorted(out, key=lambda pair: pair[1], reverse=True)


def estimated_result_rows(baseline: dict[str, Any]) -> int | None:
    """Total result rows the measured run produced.

    Derived, and the derivation matters: a digest that fires once per result
    row has a call count equal to the number of rows. So the busiest row-scoped
    digest's call count IS the row total. If nothing is row-scoped there is no
    per-row cost to attribute and this returns None rather than a guess.
    """
    repeated = repeated_digests(baseline)
    if not repeated:
        return None
    return max(d.calls for d, _ in repeated)


def ms_per_result_row(baseline: dict[str, Any]) -> float | None:
    rows = estimated_result_rows(baseline)
    if not rows:
        return None
    return baseline["derived"]["measured_latency_ms"] / rows


def unexplained_share_pct(baseline: dict[str, Any]) -> float:
    """Percentage of measured latency NOT accounted for by server-side SQL.

    This is the number the diagnosis exists to explain.
    """
    return 100.0 - baseline["derived"]["sql_share_of_latency_pct"]


# ─────────────────────────────  rendering  ──────────────────────────────

def _truncate(text: str, width: int) -> str:
    collapsed = " ".join(str(text).split())
    return collapsed if len(collapsed) <= width else collapsed[: width - 1] + "…"


def _call_chain_lines(endpoint_model: dict[str, Any]) -> list[str]:
    """One line per hop. Handles both the emit_endpoint_model shape and the
    EndpointModel contract shape, because P04's scratch driver writes the
    former and the contract describes the latter."""
    lines: list[str] = []
    chain = endpoint_model.get("call_chain") or []
    for i, hop in enumerate(chain):
        method = hop.get("method", "?")
        file = hop.get("file", "?")
        line = hop.get("line", "?")
        does = _truncate(hop.get("does", ""), 60)
        lines.append(f"  {i + 1}. {method}  ({file}:{line})  — {does}")
    if not lines:
        lines.append("  (no call chain in the endpoint model)")
    return lines


def _data_access_lines(endpoint_model: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    for access in endpoint_model.get("data_access") or []:
        scope = "PER ROW " if access.get("executes_per_row") else "per req "
        lines.append(
            f"  [{scope}] {access.get('kind', '?')}  {access.get('trigger', '?')}  "
            f"({access.get('file', '?')}:{access.get('line', '?')})"
        )
    return lines


def render_evidence(baseline: dict[str, Any], endpoint_model: dict[str, Any]) -> str:
    """The ONLY view of the measurements a model ever receives.

    Deliberately small. Raw per-request samples and full SQL logs are excluded:
    a hundred latency values invite the model to do arithmetic on them, and the
    arithmetic is already done. What it gets is the distribution, the top
    digests, the finished derived facts, and the call chain.
    """
    bench = baseline["benchmark"]
    latency = bench["latency"]
    facts = baseline["derived"]
    requests = facts["measured_requests"] or 1

    out: list[str] = []
    out.append("=== MEASURED EVIDENCE (facts — do not contradict these) ===")
    out.append("")
    out.append(f"Endpoint: {endpoint_model.get('route', '?')}")
    out.append(
        f"Run: {bench['iterations']} measured requests, "
        f"{bench['warmup']} warmup discarded, corpus {bench['corpus_hash']}"
    )
    out.append("")

    out.append("LATENCY (ms, percentiles from raw samples)")
    out.append(f"  {'n':>6} {'p50':>9} {'p95':>9} {'p99':>9} {'mean':>9} {'stdev':>9}")
    out.append(
        f"  {latency['n']:>6} {latency['p50_ms']:>9.1f} {latency['p95_ms']:>9.1f} "
        f"{latency['p99_ms']:>9.1f} {latency['mean_ms']:>9.1f} {latency['stdev_ms']:>9.1f}"
    )
    out.append(f"  response statuses: {bench['status_counts']}")
    out.append("")

    digests = app_digests(baseline)
    out.append(f"TOP {min(TOP_DIGESTS, len(digests))} APPLICATION SQL DIGESTS (by total time)")
    out.append(
        f"  {'calls':>6} {'calls/req':>10} {'total ms':>9} {'mean ms':>8} "
        f"{'rows exam':>10} {'no idx':>7}  sql"
    )
    for d in digests[:TOP_DIGESTS]:
        out.append(
            f"  {d.calls:>6} {d.calls / requests:>10.2f} {d.total_ms:>9.1f} "
            f"{d.mean_ms:>8.3f} {d.rows_examined:>10} "
            f"{'YES' if d.no_index_used else '-':>7}  {_truncate(d.sample_sql, SQL_WIDTH)}"
        )
    out.append("")

    out.append("DERIVED FACTS (computed in Python from the recorded run)")
    out.append(f"  application SQL statements per request : {facts['app_statements_per_request']}")
    out.append(f"  total application SQL statements       : {facts['app_sql_statements']}")
    out.append(
        f"  server-side SQL time                   : {facts['total_sql_ms']:.1f} ms"
        f"  of {facts['measured_latency_ms']:.1f} ms measured"
    )
    out.append(f"  SQL share of measured latency          : {facts['sql_share_of_latency_pct']:.2f} %")
    out.append(f"  NOT explained by SQL                   : {unexplained_share_pct(baseline):.2f} %")

    rows = estimated_result_rows(baseline)
    per_row = ms_per_result_row(baseline)
    if rows and per_row:
        out.append(
            f"  result rows returned (all requests)    : {rows}"
            f"  ({rows / requests:.2f} per request)"
        )
        out.append(f"  measured latency per result row        : {per_row:.1f} ms")
    out.append(f"  rows examined / rows sent by MySQL     : {facts['total_rows_examined']} / {facts['total_rows_sent']}")
    out.append(f"  digests reporting no index used        : {facts['digests_without_index']}")
    out.append("")

    repeated = repeated_digests(baseline)
    out.append("CANDIDATE N+1 (digests executing more than once per request)")
    if repeated:
        for d, ratio in repeated:
            out.append(
                f"  {ratio:.2f}x per request, {d.calls} calls, {d.total_ms:.1f} ms total, "
                f"{d.mean_ms:.3f} ms each"
            )
            out.append(f"      {_truncate(d.sample_sql, SQL_WIDTH)}")
    else:
        out.append("  none — no digest executes more than once per request")
    out.append("")

    out.append("CALL CHAIN (from the comprehension model)")
    out.extend(_call_chain_lines(endpoint_model))
    access_lines = _data_access_lines(endpoint_model)
    if access_lines:
        out.append("")
        out.append("DECLARED DATA ACCESS")
        out.extend(access_lines)

    uncertainties = endpoint_model.get("uncertainties") or []
    if uncertainties:
        out.append("")
        out.append("COMPREHENSION UNCERTAINTIES")
        for u in uncertainties:
            out.append(f"  - {_truncate(u, 110)}")

    return "\n".join(out)
