"""Render a run's JSONL trace as the narrative of what happened.

The trace is the spine — it carries the order phases ran in, how long each
took, and what each cost. The detail lines come from the artifacts themselves,
because that is where the numbers live and invariant 4 says a number in any
rendered output is templated from a recorded measurement rather than from
prose.

A skipped phase is drawn differently from one that ran. On a resumed run most
of the narrative is skips, and a reader needs to see at a glance which work was
actually done this time.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .contracts import Phase, RunState, TerminalState
from .errors import ContractViolation
from .orchestrator import BY_NAME, PIPELINE, RUNS_DIR, state_path
from .trace import Trace

ARROW = "↓"


def resolve_run(run_id: str | None, runs_dir: Path = RUNS_DIR) -> Path:
    """A run directory by id, or the most recent one that has a run state."""
    directory = Path(runs_dir)
    if run_id:
        candidate = directory / run_id
        if not candidate.exists():
            raise ContractViolation(f"no run directory at {candidate}")
        return candidate
    with_state = sorted(
        (d for d in directory.glob("*") if state_path(d).exists()), reverse=True
    )
    if not with_state:
        raise ContractViolation(
            f"no run under {directory} has a {state_path(Path('.')).name}. "
            "Run `senapathy run` first."
        )
    return with_state[0]


def _read(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _dig(payload: Any, *path: Any, default: Any = None) -> Any:
    """Walk a nested payload, returning `default` at the first missing step.

    Total on purpose. `senapathy trace` is most useful on a run that was
    interrupted, which is precisely when artifacts are half-written or absent,
    so a KeyError here would take away the one tool for diagnosing the run.
    """
    node = payload
    for key in path:
        try:
            node = node[key]
        except (KeyError, IndexError, TypeError):
            return default
    return node


def phase_detail(name: str, run_dir: Path) -> str:
    """One line of detail per phase, read from that phase's own artifact.

    Read from artifacts rather than from the trace summary so that
    `senapathy trace` on an old run still says something true: a summary string
    recorded at the time can drift from the artifact, and the artifact is what
    the report cites.
    """
    if name == "understand":
        model = _read(run_dir / "endpoint_model.json")
        if not model:
            return "endpoint model absent"
        # `call_chain`/`data_access` is the shape diagnose/evidence.py reads;
        # `path`/`sql_sources` is the EndpointModel contract's naming. Accept
        # either, because P04's driver writes one and the contract says the other.
        hops = len(
            _dig(model, "call_chain", default=None) or _dig(model, "path", default=[])
            or []
        )
        accesses = len(
            _dig(model, "data_access", default=None)
            or _dig(model, "sql_sources", default=[]) or []
        )
        file = _dig(model, "entry", "file", default="?")
        line = _dig(model, "entry", "line", default="?")
        return (f"inspected {file}:{line}; EndpointModel with {hops} hop(s), "
                f"{accesses} data access(es)")

    if name == "workload":
        corpus = _read(run_dir / "corpus.json")
        if not corpus:
            return "corpus absent"
        return (f"corpus of {len(_dig(corpus, 'benchmark', default=[]))} benchmark "
                f"and {len(_dig(corpus, 'held_back', default=[]))} held-back "
                f"request(s)")

    if name == "oracle":
        oracle = _read(run_dir / "oracle.json")
        if not oracle:
            return "oracle absent"
        return (f"sealed oracle ({len(_dig(oracle, 'recorded', default=[]))} "
                f"requests, hash {_dig(oracle, 'hash', default='?')})")

    if name == "baseline":
        baseline = _read(run_dir / "baseline.json")
        if not baseline:
            return "baseline absent"
        p50 = _dig(baseline, "benchmark", "latency", "p50_ms")
        p95 = _dig(baseline, "benchmark", "latency", "p95_ms")
        if p50 is None or p95 is None:
            return "baseline recorded (no latency block)"
        per_request = _dig(
            baseline, "derived", "app_statements_per_request", default=0.0
        )
        return (f"baseline p50 {p50:.0f} ms, p95 {p95:.0f} ms, "
                f"{per_request:.2f} app SQL statements per request")

    if name == "diagnose":
        hypotheses = _read(run_dir / "hypotheses.json")
        if not hypotheses:
            return "diagnosis absent"
        listed = _dig(hypotheses, "hypotheses", default=[])
        ranked = _dig(hypotheses, "ranked_ids", default=[])
        if not listed:
            return "diagnosis recorded (no hypotheses)"
        top = next((h for h in listed if h.get("id") == (ranked or [None])[0]),
                   listed[0])
        return (f"{len(listed)} hypotheses, "
                f"{len(_dig(hypotheses, 'probes_executed', default=[]))} probes; "
                f"top: {str(top.get('claim', ''))[:60]}")

    if name == "patch":
        patch = _read(run_dir / "patch.json")
        if not patch:
            return "patch absent"
        if patch.get("outcome") and patch["outcome"] != "PATCHED":
            return (f"ABANDONED after {patch.get('iterations_used')} of "
                    f"{patch.get('iterations_allowed')} iterations")
        files = _dig(patch, "patch", "files_changed", default=[])
        if not patch.get("patch"):
            return "patch recorded"
        return (f"patched {len(files)} file(s) "
                f"(+{_dig(patch, 'patch', 'lines_added', default=0)}/"
                f"-{_dig(patch, 'patch', 'lines_removed', default=0)}), "
                f"{_dig(patch, 'patch', 'build_iterations', default=0)} build "
                f"iterations")

    if name == "verify":
        verification = _read(run_dir / "verification.json")
        if not verification:
            return "verification absent"
        verdict = _dig(verification, "verdict", default={})
        if not verdict:
            return "verification recorded"
        if verdict.get("passed"):
            return (f"verified {verdict.get('identical')}/"
                    f"{verdict.get('compared')} identical, held-back set passed")
        shapes = _dig(verification, "diff_shapes", default=[])
        return (f"behaviour CHANGED: {verification.get('diffs_total', 0)} leaf "
                f"difference(s) in {len(shapes)} shape(s)")

    if name == "measure":
        perf = _read(run_dir / "perf_report.json")
        if not perf:
            return "measurement absent"
        delta = _dig(perf, "report", "p95_delta_pct")
        if delta is None:
            return "measurement recorded"
        direction = "faster" if delta < 0 else "slower"
        paired = perf.get("wilcoxon_p")
        line = f"measured p95 {delta:+.1f}% ({direction})"
        return f"{line}, paired p = {paired:.6f}" if paired is not None else line

    if name == "gate":
        gate = _read(run_dir / "gate.json") or _read(run_dir / "gate_preview.json")
        if not gate:
            return "gate absent"
        decision = gate.get("gate", gate)
        results = _dig(decision, "results", default=[])
        if not results:
            return "gate recorded"
        failed = [r.get("name") for r in results if not r.get("passed")]
        return "gate PASSED" if not failed else f"gate FAILED: {', '.join(failed)}"

    if name == "ship":
        gate = _read(run_dir / "gate.json")
        if not gate:
            return "ship absent"
        if "pr" not in gate:
            return "ship recorded"
        pr = gate.get("pr")
        if pr is None:
            return "no pull request opened"
        if pr.get("dry_run"):
            return "pull request prepared (dry run, nothing pushed)"
        return f"pull request opened: {pr.get('pr_url')}"

    return ""


def render_narrative(run_dir: Path) -> str:
    state = RunState.model_validate(
        json.loads(state_path(run_dir).read_text(encoding="utf-8"))
    )
    trace = Trace(run_dir, state.run_id)
    events = trace.read_all()

    ends = {e["phase"]: e for e in events if e.get("type") == "phase_end"}
    guardrails = [e for e in events if e.get("type") == "guardrail_triggered"]
    terminal_event = next(
        (e for e in reversed(events) if e.get("type") == "terminal_state"), None
    )

    out: list[str] = ["", f"Agent started · {state.endpoint}",
                      f"  run {state.run_id}"]
    for spec in PIPELINE:
        end = ends.get(spec.name)
        detail = phase_detail(spec.name, run_dir)
        if end is None:
            if spec.phase in state.completed_phases:
                out.append(f"  {ARROW} {spec.name}: {detail}")
            continue
        marker = "(skipped, already done)" if end.get("skipped") else ""
        timing = (f" [{end['duration_ms'] / 1000:.1f}s]"
                  if end.get("duration_ms") else "")
        out.append(f"  {ARROW} {detail}{timing} {marker}".rstrip())

    for event in guardrails:
        out.append(f"  {ARROW} BLOCKED at {event['phase']}: {event['reason']}")

    if terminal_event is not None:
        out.append(f"  {ARROW} {terminal_event['terminal']}")
        if terminal_event.get("reason"):
            out.append(f"      {terminal_event['reason']}")
    elif state.terminal:
        out.append(f"  {ARROW} {state.terminal.value}")
    else:
        out.append(f"  {ARROW} (run did not reach a terminal state — still "
                   f"in progress or interrupted)")
    return "\n".join(out)


def render_cost(run_dir: Path) -> str:
    """Tokens and dollars per phase.

    Tokens are measured. Dollars are DERIVED from a configured rate, which is
    why the rate is printed next to them: a dollar figure computed from a made
    up price is exactly the sort of unbacked number invariant 4 exists to stop.
    """
    cost = _read(run_dir / "cost.json")
    if not cost:
        return "\nCOST\n  no cost.json — the run did not reach a terminal state."

    out = ["", "COST",
           f"  {'phase':<12} {'calls':>6} {'tokens in':>11} {'tokens out':>11} "
           f"{'usd':>10}"]
    out.append("  " + "-" * 54)
    for phase, entry in sorted(cost.get("by_phase", {}).items()):
        out.append(
            f"  {phase:<12} {int(entry['calls']):>6} "
            f"{int(entry['tokens_in']):>11} {int(entry['tokens_out']):>11} "
            f"{entry['usd']:>10.4f}"
        )
    out.append("  " + "-" * 54)
    out.append(f"  {'total':<12} {'':>6} {cost['tokens_in']:>11} "
               f"{cost['tokens_out']:>11} {cost['usd']:>10.4f}")
    out.append(f"  wall clock   {cost['elapsed_s']:.1f}s")

    rate_in = cost.get("usd_per_million_input", 0.0)
    rate_out = cost.get("usd_per_million_output", 0.0)
    if not rate_in and not rate_out:
        out.append("  NOTE: no token price is configured, so the dollar column "
                   "is zero by construction.")
        out.append("        Tokens are measured; dollars are not. Set "
                   "USD_PER_MILLION_INPUT/OUTPUT in orchestrator.py.")
    else:
        out.append(f"  priced at ${rate_in}/M input, ${rate_out}/M output "
                   f"(configuration, not a measurement)")
    return "\n".join(out)


def render_trace(run_id: str | None = None, runs_dir: Path = RUNS_DIR) -> str:
    run_dir = resolve_run(run_id, runs_dir)
    return render_narrative(run_dir) + "\n" + render_cost(run_dir)
