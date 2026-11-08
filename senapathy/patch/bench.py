"""The in-loop benchmark: the agent's own delta, before it emits.

The patch agent used to find out whether its change helped only after the run
ended — through the measure phase, the gate, and a report nobody was reading
yet. It could emit a patch that made the endpoint SLOWER and have no way to
know. Twenty requests against the green tree costs one app boot and closes
that loop.

This is a HEURISTIC and the prompt says so in those words. It runs one arm
with no interleaving, so it carries every bias P14's design exists to remove:
the machine may be busy, the buffer pool is in whatever state the last run
left it, and there is no paired comparison. It is good enough to catch "this
is three times worse", which is the question the agent needs answered before
it commits, and it is not evidence of an improvement.

Every failure in here returns a note instead of raising. A benchmark that
cannot run must not take the patch phase down with it.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..measure.arms import (Arm, arm_argv, await_health, build_arm, start_arm,
                            stop_arm)
from ..measure.bench import run_benchmark
from ..measure.corpus import load_corpus

# Twenty measured requests after three throwaway ones. The warmup is not
# optional even here: the first JVM request on a path is interpreted bytecode
# and would make every patch look like a regression.
WARMUP = 3
ITERATIONS = 20

# Its own port, so an in-loop benchmark can never collide with the blue arm a
# measurement phase is holding open.
DEFAULT_PORT = 18099

LOG_DIR = Path(".senapathy/arms/logs")


@dataclass
class QuickResult:
    """What the agent is shown. Every number measured, none estimated."""

    p50_ms: float
    p95_ms: float
    sql_per_request: float | None
    requests: int
    baseline_p50_ms: float | None
    baseline_sql_per_request: float | None

    @property
    def delta_pct(self) -> float | None:
        if not self.baseline_p50_ms:
            return None
        return 100.0 * (self.p50_ms - self.baseline_p50_ms) / self.baseline_p50_ms

    @property
    def regressed(self) -> bool:
        delta = self.delta_pct
        return delta is not None and delta > 0

    def render(self) -> str:
        lines = ["", "YOUR CHANGE, MEASURED (heuristic — see below):"]
        if self.baseline_p50_ms is not None:
            direction = "SLOWER" if self.regressed else "faster"
            lines.append(
                f"  p50 {self.baseline_p50_ms:,.0f} ms -> {self.p50_ms:,.0f} ms "
                f"({self.delta_pct:+.0f}%, {direction})")
        else:
            lines.append(f"  p50 {self.p50_ms:,.0f} ms (no pre-patch figure to "
                         f"compare against)")
        lines.append(f"  p95 {self.p95_ms:,.0f} ms over {self.requests} requests")

        if self.sql_per_request is not None:
            before = (f"{self.baseline_sql_per_request:,.1f} -> "
                      if self.baseline_sql_per_request is not None else "")
            lines.append(f"  SQL statements per request {before}"
                         f"{self.sql_per_request:,.1f}")

        lines.append("")
        lines.append(
            "  This is {n} requests on one arm, not the real measurement. The "
            "run's verdict comes from an INTERLEAVED A/B against the unpatched "
            "tree with a paired significance test, and that is the number that "
            "decides whether this ships.".format(n=self.requests))
        if self.regressed:
            lines.append(
                "  IT GOT SLOWER. Do not emit this patch as it stands — work "
                "out why, fix it, and re-run build and run_tests. If you "
                "believe the measurement rather than the change is at fault, "
                "say so in the emit_patch risks.")
        return "\n".join(lines)


def quick_benchmark(
    green: Path,
    *,
    java_home: str | None = None,
    corpus_path: Path | None = None,
    baseline_p50_ms: float | None = None,
    baseline_sql_per_request: float | None = None,
    port: int = DEFAULT_PORT,
    mysql: Any = None,
    log_dir: Path = LOG_DIR,
) -> QuickResult | str:
    """Boot the patched tree, fire {ITERATIONS} requests, stop it again.

    Returns a `QuickResult`, or a one-line note explaining why there is no
    number. The note is given to the agent verbatim: "the benchmark could not
    run" is honest, and a silently missing delta is not.
    """
    arm: Arm | None = None
    try:
        corpus = load_corpus(corpus_path)
        build_arm(Path(green), java_home=java_home)
        arm = Arm(name="patch-quick", path=Path(green), port=port,
                  argv=arm_argv(Path(green), port, java_home=java_home))
        start_arm(arm, log_dir)
        await_health(arm)

        sql_counter = _SqlCounter(mysql)
        outcome = run_benchmark(
            arm.base_url, corpus, warmup=WARMUP, iterations=ITERATIONS,
            on_measure_start=sql_counter.start,
        )
        latency = outcome["latency"]
        return QuickResult(
            p50_ms=float(latency.p50_ms),
            p95_ms=float(latency.p95_ms),
            sql_per_request=sql_counter.per_request(ITERATIONS),
            requests=ITERATIONS,
            baseline_p50_ms=baseline_p50_ms,
            baseline_sql_per_request=baseline_sql_per_request,
        )
    except Exception as exc:  # noqa: BLE001 — never take the phase down
        tail = f" App output:\n{arm.tail_log(12)}" if arm is not None else ""
        return (f"(the in-loop benchmark could not run: "
                f"{type(exc).__name__}: {exc}.{tail})")
    finally:
        if arm is not None:
            try:
                stop_arm(arm)
            except Exception:  # noqa: BLE001
                pass


class _SqlCounter:
    """Statement counts across the measured window, or nothing at all.

    Uses the same digest reset the real measurement uses, so the number means
    the same thing in both places. No database configured is not an error: the
    latency delta is still worth having.
    """

    def __init__(self, target: Any) -> None:
        self.target = target
        self.ok = False

    def start(self) -> None:
        if self.target is None:
            return
        try:
            from ..measure.sql import reset_digests

            reset_digests(self.target)
            self.ok = True
        except Exception:  # noqa: BLE001
            self.ok = False

    def per_request(self, requests: int) -> float | None:
        if not self.ok or not requests:
            return None
        try:
            from ..measure.baseline import is_session_control
            from ..measure.sql import collect_digests

            statements = sum(d.calls for d in collect_digests(self.target)
                             if not is_session_control(d))
            return round(statements / requests, 2)
        except Exception:  # noqa: BLE001
            return None
