"""Interleaved A/B measurement, and the mechanism check.

TWO passes, deliberately, because the two quantities have opposite requirements.

* LATENCY is measured INTERLEAVED — blue, green, blue, green — because it is
  the noise-sensitive quantity. Anything that drifts over the wall-clock of a
  run (CPU thermal throttling, a background indexer, the buffer pool warming)
  lands on both arms roughly equally when they alternate, and lands entirely on
  the second arm when they do not. Sequential runs are how a machine that got
  slower during the run reports an improvement.

* SQL STATEMENT COUNTS are measured SEQUENTIALLY, one arm at a time. They have
  to be: `performance_schema.events_statements_summary_by_digest` is global to
  the MySQL server and both arms query the same `mini` schema, so during
  interleaved firing there is no way to attribute a digest to an arm. Statement
  counts are not timing-sensitive — 847 queries is 847 queries on a busy
  machine — so sequencing costs nothing here, whereas guessing the attribution
  would cost the mechanism check its meaning.

The mechanism check is the point of the second pass. A patch can make latency
fall for a reason unrelated to its stated mechanism, and "right answer, wrong
reason" has to be detectable: if the plan said "two queries per row become one"
and the statement count did not move, the diagnosis was wrong even though the
latency graph looks like a win.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable

import httpx

from ..contracts import (LatencyStats, PerfReport, RequestCorpus, RequestSpec,
                         SqlDigestStat)
from ..errors import ContractViolation, ToolError
from ..measure.bench import latency_stats
from ..measure.sql import MySQLTarget, collect_digests, reset_digests

DEFAULT_WARMUP = 10
DEFAULT_ITERATIONS = 10
DEFAULT_TIMEOUT_S = 300.0

# Below this the p-value is reported but should not be leaned on. Recorded in
# the artifact rather than enforced here; the gate (P15) decides.
MIN_PAIRS_FOR_SIGNIFICANCE = 20


@dataclass
class Sample:
    seq: int
    arm: str
    request_id: str
    status: int | None
    latency_ms: float
    error: str | None = None


@dataclass
class Pair:
    """One request measured on both arms, adjacent in time."""

    iteration: int
    request_id: str
    blue_ms: float
    green_ms: float

    @property
    def delta_ms(self) -> float:
        return self.green_ms - self.blue_ms


@dataclass
class InterleavedRun:
    samples: list[Sample] = field(default_factory=list)
    pairs: list[Pair] = field(default_factory=list)
    status_counts: dict[str, dict[str, int]] = field(default_factory=dict)
    discarded_warmup: int = 0

    def order(self, limit: int | None = None) -> list[str]:
        """The firing order as `seq arm request_id`, so interleaving is
        auditable rather than asserted."""
        rendered = [f"{s.seq:>4} {s.arm:<5} {s.request_id}" for s in self.samples]
        return rendered if limit is None else rendered[:limit]

    def arm_samples(self, arm: str) -> list[float]:
        return [p.blue_ms if arm == "blue" else p.green_ms for p in self.pairs]


def _fire(client: httpx.Client, base_url: str, spec: RequestSpec) -> tuple[int | None, float, str | None]:
    """One request, wall-clock timed with the body fully read.

    Same shape as measure/bench.py::_fire on purpose: a streamed response that
    was not drained would look faster than it is.
    """
    url = base_url.rstrip("/") + spec.path
    started = time.perf_counter()
    try:
        response = client.request(
            spec.method.upper(), url, json=spec.body, headers=spec.headers or None
        )
        _ = response.content
        return response.status_code, (time.perf_counter() - started) * 1000.0, None
    except httpx.HTTPError as exc:
        return None, (time.perf_counter() - started) * 1000.0, f"{type(exc).__name__}: {exc}"


def run_interleaved(
    blue_url: str,
    green_url: str,
    corpus: RequestCorpus,
    *,
    warmup: int = DEFAULT_WARMUP,
    iterations: int = DEFAULT_ITERATIONS,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    on_sample: Callable[[Sample], None] | None = None,
    client: httpx.Client | None = None,
) -> InterleavedRun:
    """Fire blue, green, blue, green ... across the benchmark set.

    Only `corpus.benchmark` is measured. The held-back set exists so that
    behaviour can be checked on requests no measurement ever touched; measuring
    them here would spend exactly the property that makes them useful.

    Warmup pairs are fired and DISCARDED. On a JVM the first requests through a
    path are interpreted bytecode with a cold pool, and measuring that compares
    two JIT compilers rather than two implementations.
    """
    if iterations < 1:
        raise ValueError("iterations must be >= 1")
    if not corpus.benchmark:
        raise ContractViolation("corpus has no benchmark requests")

    owns = client is None
    http = client or httpx.Client(timeout=timeout_s, follow_redirects=False)
    run = InterleavedRun()
    run.status_counts = {"blue": {}, "green": {}}
    seq = 0

    try:
        for _ in range(warmup):
            for spec in corpus.benchmark:
                _fire(http, blue_url, spec)
                _fire(http, green_url, spec)
                run.discarded_warmup += 2

        for iteration in range(iterations):
            for spec in corpus.benchmark:
                measured: dict[str, float] = {}
                # Strict alternation, blue first, adjacent in time: the two
                # halves of a pair see the same machine.
                for arm, url in (("blue", blue_url), ("green", green_url)):
                    status, elapsed_ms, error = _fire(http, url, spec)
                    sample = Sample(seq, arm, spec.id, status, elapsed_ms, error)
                    run.samples.append(sample)
                    if on_sample is not None:
                        on_sample(sample)
                    seq += 1

                    key = str(status) if status is not None else "transport_error"
                    run.status_counts[arm][key] = run.status_counts[arm].get(key, 0) + 1
                    if error is not None:
                        raise ToolError(
                            f"the {arm} arm failed on request {spec.id}: {error}. "
                            "A comparison with a hole in it is not a comparison."
                        )
                    measured[arm] = elapsed_ms

                run.pairs.append(
                    Pair(iteration, spec.id, measured["blue"], measured["green"])
                )
    finally:
        if owns:
            http.close()

    _assert_status_parity(run)
    return run


def _assert_status_parity(run: InterleavedRun) -> None:
    """Same status mix on both arms.

    Not a behaviour verdict — that is the oracle's job — but a 500 appearing on
    one arm would make its latency meaningless, and a fast error path is the
    easiest way to accidentally "improve" a benchmark.
    """
    if run.status_counts.get("blue") != run.status_counts.get("green"):
        raise ContractViolation(
            "the two arms returned different status mixes during measurement — "
            f"blue {run.status_counts.get('blue')} vs green "
            f"{run.status_counts.get('green')}. Latency across different "
            "response classes is not comparable."
        )


# ─────────────────────────────  significance  ────────────────────────────


def mann_whitney_p(blue: list[float], green: list[float]) -> float:
    """Two-sided Mann-Whitney U on the per-request latencies.

    Note on the choice: these samples are PAIRED (each request measured on both
    arms back to back), and the paired-sample test is Wilcoxon signed-rank,
    which is more powerful here because it uses the pairing. Mann-Whitney is
    what the phase specifies, so it is the headline number; `wilcoxon_p` is
    computed alongside and both go into the artifact.
    """
    from scipy.stats import mannwhitneyu

    if len(blue) < 3 or len(green) < 3:
        return 1.0
    if len(set(blue)) == 1 and len(set(green)) == 1 and blue[0] == green[0]:
        return 1.0
    return float(mannwhitneyu(blue, green, alternative="two-sided").pvalue)


def wilcoxon_p(pairs: list[Pair]) -> float | None:
    """Two-sided Wilcoxon signed-rank on the paired differences."""
    from scipy.stats import wilcoxon

    deltas = [p.delta_ms for p in pairs]
    if len(deltas) < 3 or all(d == 0 for d in deltas):
        return None
    try:
        return float(wilcoxon(deltas, alternative="two-sided").pvalue)
    except ValueError:
        return None


def delta_pct(before: float, after: float) -> float:
    """Signed percentage change. Negative means green got faster."""
    if before == 0:
        return 0.0
    return round(100.0 * (after - before) / before, 3)


# ────────────────────────────  mechanism check  ──────────────────────────


@dataclass
class ArmSql:
    statements: int
    rows_examined: int
    digests: list[SqlDigestStat]

    @property
    def application_statements(self) -> int:
        from ..measure.baseline import is_session_control

        return sum(d.calls for d in self.digests if not is_session_control(d))


def measure_sql_for_arm(
    base_url: str,
    corpus: RequestCorpus,
    target: MySQLTarget,
    *,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    client: httpx.Client | None = None,
) -> ArmSql:
    """Reset the digest table, fire the benchmark set once, read the digests.

    Sequential by necessity — see the module docstring. The reset is what makes
    the counts describe this pass rather than every statement since the server
    started.
    """
    reset_digests(target)
    owns = client is None
    http = client or httpx.Client(timeout=timeout_s, follow_redirects=False)
    try:
        for spec in corpus.benchmark:
            status, _, error = _fire(http, base_url, spec)
            if error is not None:
                raise ToolError(f"SQL pass failed on {spec.id} at {base_url}: {error}")
    finally:
        if owns:
            http.close()

    digests = collect_digests(target)
    return ArmSql(
        statements=sum(d.calls for d in digests),
        rows_examined=sum(d.rows_examined for d in digests),
        digests=digests,
    )


def mechanism_verdict(
    blue_sql: ArmSql, green_sql: ArmSql, requests: int
) -> tuple[bool, str]:
    """Did the SQL work actually change? Computed, never asserted by prose.

    `mechanism_confirmed` is True only when the statement count or the rows
    examined moved by more than 5%. A latency win with unmoved SQL is the
    "right answer, wrong reason" case, and it is reported as such rather than
    being quietly folded into a good-looking result.
    """
    blue_app = blue_sql.application_statements
    green_app = green_sql.application_statements
    statements_delta = delta_pct(blue_app, green_app)
    rows_delta = delta_pct(blue_sql.rows_examined, green_sql.rows_examined)

    per_request_before = blue_app / requests if requests else 0.0
    per_request_after = green_app / requests if requests else 0.0

    changed = abs(statements_delta) > 5.0 or abs(rows_delta) > 5.0
    if changed:
        note = (
            f"application SQL statements per request {per_request_before:.2f} -> "
            f"{per_request_after:.2f} ({statements_delta:+.1f}%); rows examined "
            f"{blue_sql.rows_examined} -> {green_sql.rows_examined} "
            f"({rows_delta:+.1f}%). The database work changed."
        )
    else:
        note = (
            f"application SQL statements per request {per_request_before:.2f} -> "
            f"{per_request_after:.2f} ({statements_delta:+.1f}%); rows examined "
            f"{blue_sql.rows_examined} -> {green_sql.rows_examined} "
            f"({rows_delta:+.1f}%). The database work did NOT change materially, "
            f"so any latency difference comes from somewhere other than SQL — "
            f"check that against the mechanism the patch claimed."
        )
    return changed, note


# ──────────────────────────────  the report  ─────────────────────────────


def build_report(
    run: InterleavedRun, blue_sql: ArmSql, green_sql: ArmSql
) -> PerfReport:
    """Assemble a PerfReport. Every number here is computed from samples."""
    before = latency_stats(run.arm_samples("blue"))
    after = latency_stats(run.arm_samples("green"))
    requests = len({p.request_id for p in run.pairs}) or 1
    confirmed, note = mechanism_verdict(blue_sql, green_sql, requests)

    return PerfReport(
        before=before,
        after=after,
        p50_delta_pct=delta_pct(before.p50_ms, after.p50_ms),
        p95_delta_pct=delta_pct(before.p95_ms, after.p95_ms),
        sql_count_before=blue_sql.application_statements,
        sql_count_after=green_sql.application_statements,
        rows_examined_before=blue_sql.rows_examined,
        rows_examined_after=green_sql.rows_examined,
        p_value=round(mann_whitney_p(run.arm_samples("blue"), run.arm_samples("green")), 6),
        mechanism_confirmed=confirmed,
        mechanism_note=note,
    )


def render_report(
    report: PerfReport, run: InterleavedRun, document: dict[str, Any],
    order_lines: int = 24,
) -> str:
    before, after = report.before, report.after
    requests = len({p.request_id for p in run.pairs}) or 1

    def row(label: str, b: Any, g: Any, delta: str = "") -> str:
        return f"  {label:<26} {str(b):>14} {str(g):>14}   {delta}"

    out: list[str] = ["", "A/B MEASUREMENT  (interleaved)"]
    out.append(f"  arms            blue {document['blue_url']}   "
               f"green {document['green_url']}")
    out.append(f"  pairs           {len(run.pairs)} "
               f"({requests} request(s) x {document['iterations']} iteration(s))")
    out.append(f"  warmup          {run.discarded_warmup} request(s), discarded")
    out.append("")
    out.append(row("", "BLUE", "GREEN", "DELTA"))
    out.append("  " + "-" * 72)
    out.append(row("p50 ms", f"{before.p50_ms:.1f}", f"{after.p50_ms:.1f}",
                   f"{report.p50_delta_pct:+.1f} %"))
    out.append(row("p95 ms", f"{before.p95_ms:.1f}", f"{after.p95_ms:.1f}",
                   f"{report.p95_delta_pct:+.1f} %"))
    out.append(row("p99 ms", f"{before.p99_ms:.1f}", f"{after.p99_ms:.1f}"))
    out.append(row("mean ms", f"{before.mean_ms:.1f}", f"{after.mean_ms:.1f}"))
    out.append(row("stdev ms", f"{before.stdev_ms:.1f}", f"{after.stdev_ms:.1f}"))
    out.append(row("SQL statements/request",
                   f"{report.sql_count_before / requests:.2f}",
                   f"{report.sql_count_after / requests:.2f}",
                   f"{delta_pct(report.sql_count_before, report.sql_count_after):+.1f} %"))
    out.append(row("rows examined", report.rows_examined_before,
                   report.rows_examined_after,
                   f"{delta_pct(report.rows_examined_before, report.rows_examined_after):+.1f} %"))
    out.append("")
    out.append(f"  Mann-Whitney U p-value    {report.p_value:.6f}"
               f"   (n={len(run.pairs)} per arm)")
    if document.get("wilcoxon_p") is not None:
        out.append(f"  Wilcoxon signed-rank p    {document['wilcoxon_p']:.6f}"
                   f"   (paired test; see ab.py)")
    if len(run.pairs) < MIN_PAIRS_FOR_SIGNIFICANCE:
        out.append(f"  WARNING: only {len(run.pairs)} pairs — below the "
                   f"{MIN_PAIRS_FOR_SIGNIFICANCE} this reporter treats as enough "
                   f"to lean on a p-value.")
    out.append("")
    out.append("MECHANISM CHECK")
    out.append(f"  confirmed       {report.mechanism_confirmed}")
    for line in _wrap(report.mechanism_note, 74):
        out.append(f"  {line}")
    out.append("")
    out.append(f"FIRING ORDER (first {order_lines} of {len(run.samples)} measured "
               f"requests — interleaving is visible here, not asserted)")
    for line in run.order(order_lines):
        out.append(f"  {line}")
    if len(run.samples) > order_lines:
        out.append(f"  ... {len(run.samples) - order_lines} more; full order in "
                   f"perf_report.json")
    return "\n".join(out)


def _wrap(text: str, width: int) -> list[str]:
    import textwrap

    return textwrap.wrap(text, width=width) or [""]
