from __future__ import annotations

import json
import time
from typing import Any, Callable

import httpx

from ..contracts import SqlDigestStat
from ..errors import ToolError
from ..measure.baseline import is_session_control
from ..measure.bench import latency_stats
from ..measure.sql import MySQLTarget, collect_digests, reset_digests

# Hard caps. The agent picks the numbers, so the numbers have to be bounded:
# page_size=100000 against a 50k-row table with a per-row query is a denial of
# service against the thing we are trying to measure.
MAX_ITERATIONS = 20
MAX_PAGE_SIZE = 100
PROBE_WARMUP = 1
PROBE_TIMEOUT_S = 180.0
MAX_NAME_FILTER = 200


class ProbeLog:
    """Record of probes that were ACTUALLY executed.

    The agent also reports the probes it ran, but a self-reported probe list is
    a claim, not evidence. Validation checks the agent's story against this.
    """

    def __init__(self) -> None:
        self.entries: list[dict[str, Any]] = []

    def record(self, args: dict[str, Any], summary: str) -> None:
        self.entries.append({"tool": "run_probe", "args": args, "result": summary})

    def __len__(self) -> int:
        return len(self.entries)


def _app_sql(digests: list[SqlDigestStat]) -> tuple[int, float]:
    """(statement count, total ms) for application SQL only."""
    app = [d for d in digests if not is_session_control(d)]
    return sum(d.calls for d in app), sum(d.total_ms for d in app)


def make_run_probe(
    base_url: str,
    target: MySQLTarget,
    *,
    path: str = "/api/customers/search",
    log: ProbeLog | None = None,
) -> Callable[..., str]:
    """Build the `run_probe` handler.

    The endpoint, the HTTP method and the request shape are fixed HERE, by
    construction. The agent cannot choose a URL, a verb, a header or a body
    field — only a page size, a name filter and an iteration count, all
    bounded. That is what makes this tool safe to hand to a model under
    invariant 1: it is not a shell, it is one parameterised read.

    The only write it performs is TRUNCATE on
    performance_schema.events_statements_summary_by_digest, which is a
    diagnostic counter table, never application data.
    """
    probe_log = log if log is not None else ProbeLog()

    def run_probe(page_size: int, name_filter: str, iterations: int = 10) -> str:
        # Clamp rather than reject: a refused probe costs the agent a turn and
        # teaches it nothing. A clamped probe still produces evidence, and the
        # summary says plainly what was changed.
        try:
            page_size = int(page_size)
            iterations = int(iterations)
        except (TypeError, ValueError) as exc:
            raise ToolError(
                f"page_size and iterations must be integers: {exc}",
                suggestion="Example: run_probe(page_size=5, name_filter='a', iterations=10)",
            ) from exc

        if not isinstance(name_filter, str):
            raise ToolError("name_filter must be a string")
        if len(name_filter) > MAX_NAME_FILTER:
            raise ToolError(f"name_filter must be at most {MAX_NAME_FILTER} characters")

        notes: list[str] = []
        if page_size > MAX_PAGE_SIZE:
            notes.append(f"page_size clamped from {page_size} to {MAX_PAGE_SIZE}")
            page_size = MAX_PAGE_SIZE
        if page_size < 1:
            notes.append(f"page_size raised from {page_size} to 1")
            page_size = 1
        if iterations > MAX_ITERATIONS:
            notes.append(f"iterations clamped from {iterations} to {MAX_ITERATIONS}")
            iterations = MAX_ITERATIONS
        if iterations < 1:
            notes.append(f"iterations raised from {iterations} to 1")
            iterations = 1

        body = {"name": name_filter, "page": 0, "size": page_size}
        url = base_url.rstrip("/") + path

        samples_ms: list[float] = []
        body_sizes: list[int] = []
        statuses: list[int] = []
        last_body: bytes = b""

        try:
            with httpx.Client(timeout=httpx.Timeout(PROBE_TIMEOUT_S)) as client:
                for _ in range(PROBE_WARMUP):
                    client.post(url, json=body).read()

                # Reset AFTER warmup, immediately before the measured requests,
                # so the digests describe exactly these `iterations` calls.
                reset_digests(target)

                for _ in range(iterations):
                    started = time.perf_counter()
                    response = client.post(url, json=body)
                    content = response.content
                    samples_ms.append((time.perf_counter() - started) * 1000.0)
                    body_sizes.append(len(content))
                    statuses.append(response.status_code)
                    last_body = content

                digests = collect_digests(target)
        except httpx.HTTPError as exc:
            raise ToolError(
                f"probe could not reach {url}: {type(exc).__name__}: {exc}",
                suggestion="The app may have stopped. Do not retry more than once.",
            ) from exc

        statements, sql_ms = _app_sql(digests)
        stats = latency_stats(samples_ms)
        median_body = sorted(body_sizes)[len(body_sizes) // 2]

        # Rows actually returned — parsed ONCE, after the timed loop, so
        # parsing never lands inside a latency sample.
        rows: int | None = None
        try:
            parsed = json.loads(last_body)
            if isinstance(parsed, dict) and isinstance(parsed.get("content"), list):
                rows = len(parsed["content"])
        except (ValueError, TypeError):
            rows = None

        status_counts: dict[str, int] = {}
        for code in statuses:
            status_counts[str(code)] = status_counts.get(str(code), 0) + 1

        sql_ms_per_request = sql_ms / iterations
        sql_share = (100.0 * sql_ms_per_request / stats.p50_ms) if stats.p50_ms else 0.0

        out: list[str] = []
        out.append(
            f"probe: page_size={page_size} name_filter={name_filter!r} "
            f"iterations={iterations} ({PROBE_WARMUP} warmup discarded)"
        )
        for note in notes:
            out.append(f"  NOTE: {note}")
        out.append(f"  statuses                     : {status_counts}")
        out.append(f"  p50 latency                  : {stats.p50_ms:.1f} ms")
        out.append(f"  p95 latency                  : {stats.p95_ms:.1f} ms")
        out.append(
            f"  result rows in response      : {rows if rows is not None else 'unparsed'}"
        )
        out.append(f"  response body size (median)  : {median_body} bytes")
        out.append(f"  application SQL per request  : {statements / iterations:.1f} statements")
        out.append(
            f"  server-side SQL per request  : {sql_ms_per_request:.1f} ms "
            f"({sql_share:.1f} % of p50)"
        )
        if rows:
            out.append(f"  latency per result row       : {stats.p50_ms / rows:.1f} ms")
            out.append(f"  body bytes per result row    : {median_body // rows}")
        summary = "\n".join(out)

        probe_log.record(
            {"page_size": page_size, "name_filter": name_filter, "iterations": iterations},
            summary,
        )
        return summary

    return run_probe


RUN_PROBE_TOOL = {
    "type": "function",
    "function": {
        "name": "run_probe",
        "description": (
            "Run a live experiment against the endpoint and measure what happens. "
            "Fires the search request `iterations` times, resets and re-reads "
            "performance_schema around the measured calls, and reports p50 latency, "
            "application SQL statements per request, server-side SQL ms per request, "
            "result row count and response body size.\n\n"
            "This is READ-ONLY: it can only call the search endpoint with a different "
            "page size and name filter. It cannot change code, data or configuration.\n\n"
            f"Caps: page_size at most {MAX_PAGE_SIZE}, iterations at most {MAX_ITERATIONS} "
            "(values above the cap are clamped, not rejected).\n\n"
            "COST WARNING: a probe costs page_size x per-row latency x iterations. "
            "For page_size 20 or more, use iterations=3 to 5 — the point is the shape "
            "of the curve, not a tight confidence interval."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "page_size": {
                    "type": "integer",
                    "description": (
                        "Rows requested per page. This is the main experimental knob: "
                        "sweep it to find out whether cost scales with the number of "
                        "result rows."
                    ),
                },
                "name_filter": {
                    "type": "string",
                    "description": (
                        "The `name` filter sent to the endpoint. 'a' matches broadly; "
                        "'zzzz-no-such-customer' matches nothing, which isolates the "
                        "filter query from all per-row work."
                    ),
                },
                "iterations": {
                    "type": "integer",
                    "description": "Measured requests, default 10, maximum 20.",
                },
            },
            "required": ["page_size", "name_filter"],
        },
    },
}
