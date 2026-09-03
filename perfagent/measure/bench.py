from __future__ import annotations

import statistics
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from typing import Any, Callable

import httpx

from ..contracts import LatencyStats, RequestCorpus, RequestSpec
from ..errors import ToolError

DEFAULT_WARMUP = 20
DEFAULT_ITERATIONS = 100
DEFAULT_TIMEOUT_S = 120.0


def percentile(sorted_samples: list[float], q: float) -> float:
    """Linear-interpolated percentile over RAW samples.

    `sorted_samples` must already be sorted ascending. `q` is in [0, 100].

    This exists so that nothing anywhere is tempted to average a set of
    per-request means: a p95 of means is not a p95, and the whole point of the
    baseline is that the tail is real.
    """
    if not sorted_samples:
        raise ValueError("percentile of an empty sample set")
    if len(sorted_samples) == 1:
        return float(sorted_samples[0])
    rank = (q / 100.0) * (len(sorted_samples) - 1)
    low = int(rank)
    high = min(low + 1, len(sorted_samples) - 1)
    frac = rank - low
    return float(sorted_samples[low] + (sorted_samples[high] - sorted_samples[low]) * frac)


def latency_stats(samples_ms: list[float]) -> LatencyStats:
    """Build a LatencyStats from raw per-request latencies."""
    if not samples_ms:
        raise ValueError("cannot compute latency stats from zero samples")
    ordered = sorted(samples_ms)
    return LatencyStats(
        n=len(ordered),
        p50_ms=round(percentile(ordered, 50), 3),
        p95_ms=round(percentile(ordered, 95), 3),
        p99_ms=round(percentile(ordered, 99), 3),
        mean_ms=round(statistics.fmean(ordered), 3),
        stdev_ms=round(statistics.stdev(ordered), 3) if len(ordered) > 1 else 0.0,
    )


def _fire(
    client: httpx.Client, base_url: str, spec: RequestSpec
) -> tuple[int | None, float, str | None]:
    """Send one request. Returns (status, wall_clock_ms, transport_error).

    Wall clock is measured with perf_counter around the send, so it includes
    everything the caller would experience: serialisation, network, server work,
    and reading the body off the socket. The body is fully read before the clock
    stops — otherwise a streamed response would look faster than it is.
    """
    url = base_url.rstrip("/") + spec.path
    started = time.perf_counter()
    try:
        response = client.request(
            spec.method.upper(),
            url,
            json=spec.body,
            headers=spec.headers or None,
        )
        _ = response.content  # force the full body before stopping the clock
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        return response.status_code, elapsed_ms, None
    except httpx.HTTPError as exc:
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        return None, elapsed_ms, f"{type(exc).__name__}: {exc}"


def run_benchmark(
    base_url: str,
    corpus: RequestCorpus,
    warmup: int = DEFAULT_WARMUP,
    iterations: int = DEFAULT_ITERATIONS,
    *,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    on_measure_start: Callable[[], None] | None = None,
    on_sample: Callable[[dict[str, Any]], None] | None = None,
    client: httpx.Client | None = None,
) -> dict[str, Any]:
    """Fire `warmup` throwaway requests, then `iterations` measured ones.

    The warmup requests are DISCARDED. On a JVM the first few hundred
    milliseconds of any code path are interpreted bytecode, the connection pool
    is cold and the buffer pool is empty; measuring that produces a p99 that
    describes the JIT rather than the endpoint. Warmup is not optional.

    Requests cycle through `corpus.benchmark` in order, so every request in the
    corpus is hit the same number of times give or take one, and no single
    expensive shape can dominate by luck of the draw.

    `on_measure_start` fires exactly once — after the last warmup request and
    before the first measured one. That is the only correct moment to reset
    server-side counters, and the hook exists so this module never has to know
    that MySQL is involved.

    Returns a plain dict; `latency` and the per-request entries hold
    `LatencyStats` models from contracts.py.
    """
    if warmup < 0:
        raise ValueError("warmup must be >= 0")
    if iterations < 1:
        raise ValueError("iterations must be >= 1")
    specs = corpus.benchmark
    if not specs:
        raise ValueError("corpus has no benchmark requests")

    owns_client = client is None
    http = client or httpx.Client(
        timeout=httpx.Timeout(timeout_s),
        follow_redirects=False,
        headers={"User-Agent": "perfagent-bench/1"},
    )

    warmup_ms: list[float] = []
    warmup_failures = 0
    samples: list[dict[str, Any]] = []

    try:
        for i in range(warmup):
            spec = specs[i % len(specs)]
            status, elapsed_ms, error = _fire(http, base_url, spec)
            warmup_ms.append(elapsed_ms)
            if error is not None:
                warmup_failures += 1

        if on_measure_start is not None:
            on_measure_start()

        run_started = time.perf_counter()
        started_at = datetime.now(timezone.utc).isoformat()
        for i in range(iterations):
            spec = specs[i % len(specs)]
            status, elapsed_ms, error = _fire(http, base_url, spec)
            sample = {
                "i": i,
                "request_id": spec.id,
                "status": status,
                "latency_ms": round(elapsed_ms, 3),
                "error": error,
            }
            samples.append(sample)
            if on_sample is not None:
                on_sample(sample)
        wall_ms = (time.perf_counter() - run_started) * 1000.0
        finished_at = datetime.now(timezone.utc).isoformat()
    finally:
        if owns_client:
            http.close()

    # A transport failure has a duration but no response; folding a 120s
    # timeout into the latency distribution would quietly invent a tail. Such
    # samples are counted and listed instead. 4xx/5xx ARE real responses and
    # stay in the distribution.
    answered = [s for s in samples if s["status"] is not None]
    failed = [s for s in samples if s["status"] is None]
    if not answered:
        raise ToolError(
            f"all {iterations} measured requests failed against {base_url}",
            suggestion="Check the app is up and the corpus paths are correct; "
            f"first error was: {failed[0]['error'] if failed else 'unknown'}",
        )

    per_request: dict[str, Any] = {}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for s in answered:
        grouped[s["request_id"]].append(s)
    for request_id, group in grouped.items():
        per_request[request_id] = {
            "n": len(group),
            "statuses": dict(Counter(str(s["status"]) for s in group)),
            "latency": latency_stats([s["latency_ms"] for s in group]),
        }

    return {
        "base_url": base_url,
        "corpus_hash": corpus.hash,
        "requests_in_corpus": len(specs),
        "warmup": warmup,
        "iterations": iterations,
        "started_at": started_at,
        "finished_at": finished_at,
        "wall_ms": round(wall_ms, 3),
        "latency": latency_stats([s["latency_ms"] for s in answered]),
        "status_counts": dict(sorted(Counter(str(s["status"]) for s in answered).items())),
        "transport_failures": len(failed),
        "warmup_discarded": {
            "n": len(warmup_ms),
            "failures": warmup_failures,
            "mean_ms": round(statistics.fmean(warmup_ms), 3) if warmup_ms else 0.0,
        },
        "per_request": per_request,
        "samples": samples,
        "errors": [
            {"i": s["i"], "request_id": s["request_id"], "error": s["error"]}
            for s in failed
        ],
    }
