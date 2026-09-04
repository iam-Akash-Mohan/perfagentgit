"""The pipeline, as a state machine in plain Python.

The phase ORDER is a module constant. No model is asked what to do next, no
model can skip a phase, and there is no branch in this file that a model's
output can steer — invariant 6 applied to control flow rather than to verdicts.

Two transitions carry guards that exist because a prompt asking nicely is not a
guarantee:

  * IMPLEMENT cannot be reached without a recorded, hash-verified oracle. A
    patch written before behaviour was frozen has nothing to be checked
    against, and recording the oracle afterwards would record the patch's own
    behaviour as the baseline.
  * MEASURE cannot be reached without a PASSING verification for the patch
    currently in the green tree. A performance number for a patch that changed
    behaviour is worse than no number.

Both are checked in `_preconditions`, before the runner is called, and neither
can be disabled by an argument to `run_pipeline`.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator

from .contracts import (ArtifactRef, Phase, RunState, TerminalState)
from .errors import BudgetExhausted, ContractViolation, PerfAgentError, ToolError
from .ids import content_hash, new_run_id
from .trace import EventType, Trace

RUNS_DIR = Path(".perfagent/runs")
STATE_FILE = "run_state.json"

DEFAULT_BLUE_PORT = 18080
DEFAULT_GREEN_PORT = 8081

# Token pricing is NOT a measurement. The provider's rate is configuration, so
# the dollar figure here is derived from a rate you set rather than something
# observed — tokens are the ground truth and are always reported. Set the rate
# to your actual contract before quoting the dollars anywhere.
USD_PER_MILLION_INPUT = 0.0
USD_PER_MILLION_OUTPUT = 0.0

DEFAULT_MAX_USD = 5.0
DEFAULT_MAX_WALL_S = 3600.0


# ──────────────────────────────  the budget  ─────────────────────────────


@dataclass
class Budget:
    """A run-level ceiling on spend and on wall clock.

    Exceeding either is a terminal STATE, not an exception that escapes: the
    run stops, keeps whatever it has proven so far, and says why. A crash at
    minute fifty-nine throws away fifty-nine minutes of evidence.
    """

    # None means no ceiling. ZERO means zero, which is not the same thing —
    # `if self.max_wall_s` treated a 0 budget as unlimited, so a run given no
    # time at all would have run forever.
    max_usd: float | None = DEFAULT_MAX_USD
    max_wall_s: float | None = DEFAULT_MAX_WALL_S
    usd_per_million_input: float = USD_PER_MILLION_INPUT
    usd_per_million_output: float = USD_PER_MILLION_OUTPUT

    started_at: float = field(default_factory=time.monotonic)
    tokens_in: int = 0
    tokens_out: int = 0
    usd: float = 0.0
    by_phase: dict[str, dict[str, float]] = field(default_factory=dict)

    def elapsed_s(self) -> float:
        return time.monotonic() - self.started_at

    def remaining_s(self) -> float:
        if self.max_wall_s is None:
            return float("inf")
        return max(0.0, self.max_wall_s - self.elapsed_s())

    def charge(self, phase: str, tokens_in: int, tokens_out: int) -> float:
        usd = (
            tokens_in / 1_000_000 * self.usd_per_million_input
            + tokens_out / 1_000_000 * self.usd_per_million_output
        )
        self.tokens_in += tokens_in
        self.tokens_out += tokens_out
        self.usd += usd
        entry = self.by_phase.setdefault(
            phase, {"tokens_in": 0.0, "tokens_out": 0.0, "usd": 0.0, "calls": 0.0}
        )
        entry["tokens_in"] += tokens_in
        entry["tokens_out"] += tokens_out
        entry["usd"] += usd
        entry["calls"] += 1
        return usd

    def check(self) -> str | None:
        """The reason the run must stop, or None."""
        if self.max_usd is not None and self.usd >= self.max_usd:
            return (
                f"spend reached ${self.usd:.4f} of the ${self.max_usd:.2f} budget"
            )
        if self.max_wall_s is not None and self.elapsed_s() >= self.max_wall_s:
            return (
                f"wall clock reached {self.elapsed_s():.0f}s of the "
                f"{self.max_wall_s:.0f}s budget"
            )
        return None

    def assert_within(self) -> None:
        reason = self.check()
        if reason is not None:
            raise BudgetExhausted(reason)


class MeteredClient:
    """Wraps an LLM client so every call is charged to the run budget.

    Delegates everything else, because `run_agent` reads `last_raw_message` off
    the client between calls and a proxy that hid it would break the loop.
    """

    def __init__(self, inner: Any, budget: Budget, phase: str) -> None:
        self._inner = inner
        self._budget = budget
        self._phase = phase

    def complete(self, **kwargs: Any) -> Any:
        self._budget.assert_within()
        response = self._inner.complete(**kwargs)
        usage = getattr(response, "usage", None)
        if usage is not None:
            self._budget.charge(
                self._phase,
                getattr(usage, "input_tokens", 0) or 0,
                getattr(usage, "output_tokens", 0) or 0,
            )
        self._budget.assert_within()
        return response

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


# ────────────────────────────  the phase table  ──────────────────────────


@dataclass
class PhaseOutcome:
    artifact: Path | None = None
    terminal: TerminalState | None = None
    summary: str = ""
    detail: dict[str, Any] = field(default_factory=dict)


@dataclass
class Context:
    run_id: str
    run_dir: Path
    repo: Path | None
    endpoint: str
    budget: Budget
    trace: Trace
    blue_port: int = DEFAULT_BLUE_PORT
    green_port: int = DEFAULT_GREEN_PORT
    java_home: str | None = None
    options: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PhaseSpec:
    phase: Phase
    name: str
    artifact: Callable[[Context], Path]
    runner: Callable[[Context], PhaseOutcome] | None
    validate: Callable[[Context, Path], str | None] | None = None
    uses_llm: bool = False
    describe: str = ""


def _run_artifact(name: str) -> Callable[[Context], Path]:
    return lambda ctx: ctx.run_dir / name


def _fixed(path: str) -> Callable[[Context], Path]:
    return lambda ctx: Path(path)


# ─────────────────────────────  phase runners  ───────────────────────────


def _run_oracle(ctx: Context) -> PhaseOutcome:
    from .measure.arms import single_arm
    from .measure.corpus import load_corpus
    from .oracle.canonicalise import load_policy
    from .oracle.record import ORACLE_PATH, record_oracle

    corpus = load_corpus(ctx.options.get("corpus"))
    with single_arm(
        ctx.run_id, which="blue", repo=ctx.repo, port=ctx.blue_port,
        java_home=ctx.java_home, build=ctx.options.get("build", True),
    ) as arm:
        bundle = record_oracle(
            arm.base_url, corpus, policy=load_policy(),
            repo_path=ctx.repo, path=ORACLE_PATH, force=False,
        )
    return PhaseOutcome(
        artifact=ORACLE_PATH,
        summary=f"sealed oracle ({len(bundle['recorded'])} requests, "
                f"hash {bundle['hash']})",
        detail={"hash": bundle["hash"], "recorded": len(bundle["recorded"])},
    )


def _run_baseline(ctx: Context) -> PhaseOutcome:
    """Measure the blue arm, then adopt the artifact into THIS run's directory.

    `run_baseline` creates its own timestamped run directory, which predates the
    orchestrator. Left alone, the pipeline's baseline would land in a sibling
    directory and every downstream phase — which reads `run_dir/baseline.json` —
    would not find it. So the produced file is copied in, and the directory it
    came from is recorded rather than hidden.
    """
    import shutil

    from .measure.arms import single_arm
    from .measure.baseline import run_baseline

    with single_arm(
        ctx.run_id, which="blue", repo=ctx.repo, port=ctx.blue_port,
        java_home=ctx.java_home, build=ctx.options.get("build", True),
    ) as arm:
        document, produced = run_baseline(
            base_url=arm.base_url,
            corpus_path=ctx.options.get("corpus"),
            warmup=ctx.options.get("warmup", 20),
            iterations=ctx.options.get("iterations", 100),
            runs_dir=ctx.run_dir / "baseline-runs",
        )

    adopted = ctx.run_dir / "baseline.json"
    shutil.copy2(produced, adopted)

    # `benchmark.latency` is a LatencyStats MODEL in the returned document, not
    # a dict — it is only JSON on disk. Reading it as a mapping raised
    # "'LatencyStats' object is not subscriptable" on the first real run.
    latency = document["benchmark"]["latency"]
    return PhaseOutcome(
        artifact=adopted,
        summary=f"baseline p95 {latency.p95_ms:.0f} ms",
        detail={"p50_ms": latency.p50_ms, "p95_ms": latency.p95_ms,
                "produced_in": str(Path(produced).parent)},
    )


def _run_diagnose(ctx: Context) -> PhaseOutcome:
    from .diagnose.agent import run_diagnosis
    from .measure.arms import single_arm

    client = _metered_llm(ctx, "diagnose")
    with single_arm(
        ctx.run_id, which="blue", repo=ctx.repo, port=ctx.blue_port,
        java_home=ctx.java_home, build=ctx.options.get("build", True),
    ) as arm:
        hypothesis_set, probe_log, path = run_diagnosis(
            base_url=arm.base_url, run_dir=ctx.run_dir, client=client
        )
    if not hypothesis_set.hypotheses:
        return PhaseOutcome(
            artifact=Path(path), terminal=TerminalState.NO_BOTTLENECK_FOUND,
            summary="no hypothesis cleared the bar",
        )
    return PhaseOutcome(
        artifact=Path(path),
        summary=f"{len(hypothesis_set.hypotheses)} hypotheses, "
                f"{len(probe_log)} probes",
        detail={"hypotheses": len(hypothesis_set.hypotheses),
                "probes": len(probe_log),
                "top": hypothesis_set.ranked_ids[0]},
    )


def _run_patch(ctx: Context) -> PhaseOutcome:
    """Write the patch on a green worktree belonging to THIS run.

    Green is derived state: the patch phase regenerates it from the base commit
    every time. A worktree left by an earlier run therefore has to go, or
    `GreenWorktree.create` refuses and the pipeline cannot be run twice in a
    row. Removing it is safe precisely because nothing upstream depends on it —
    blue is the source of truth, and the oracle is already frozen.
    """
    from .measure.arms import GREEN_DIR
    from .patch.agent import run_patch
    from .patch.worktree import GreenWorktree, existing_green_branch

    if Path(GREEN_DIR).exists():
        stale = existing_green_branch(GREEN_DIR, repo=ctx.repo)
        ctx.trace.emit(
            EventType.NOTE, phase="patch",
            note=f"removing a green worktree left by an earlier run ({stale})",
        )
        GreenWorktree(
            path=Path(GREEN_DIR).resolve(), branch=stale, base_commit="",
            repo=_repo_root(ctx),
        ).remove()

    document = run_patch(
        ctx.run_dir, run_id=ctx.run_id, repo=ctx.repo,
        java_home=ctx.java_home, client=_metered_llm(ctx, "patch"),
    )
    patch = document["patch"]
    if document["outcome"] != "PATCHED":
        return PhaseOutcome(
            artifact=ctx.run_dir / "patch.json",
            terminal=TerminalState.BOTTLENECK_NOT_FIXABLE,
            summary=f"ABANDONED after {document['iterations_used']} iterations",
            detail={"last_error": str(document.get("last_error", ""))[:200]},
        )
    return PhaseOutcome(
        artifact=ctx.run_dir / "patch.json",
        summary=f"patched {len(patch['files_changed'])} file(s) "
                f"(+{patch['lines_added']}/-{patch['lines_removed']}), "
                f"{patch['build_iterations']} build iterations",
        detail={"files": patch["files_changed"],
                "iterations": patch["build_iterations"]},
    )


def _repo_root(ctx: Context):
    from .patch.worktree import target_repo

    return target_repo(ctx.repo)


def _run_verify(ctx: Context) -> PhaseOutcome:
    from .measure.arms import both_arms, existing_worktree
    from .measure.verify import run_verification

    green = existing_worktree(".perfagent/green", repo=ctx.repo)
    with both_arms(
        f"verify-{ctx.run_id}", repo=ctx.repo, blue_port=ctx.blue_port,
        green_port=ctx.green_port, java_home=ctx.java_home,
        build=ctx.options.get("build", True),
    ) as pair:
        verdict, path = run_verification(
            pair.green.base_url, run_dir=ctx.run_dir,
            green_diff=green.source_diff(),
        )
    if not verdict.passed:
        return PhaseOutcome(
            artifact=path, terminal=TerminalState.BEHAVIOUR_NOT_PRESERVED,
            summary=f"behaviour CHANGED: {len(verdict.diffs)} differing path(s)",
            detail={"diffs": len(verdict.diffs)},
        )
    return PhaseOutcome(
        artifact=path,
        summary=f"verified {verdict.identical}/{verdict.compared} identical",
        detail={"identical": verdict.identical, "compared": verdict.compared},
    )


def _run_measure(ctx: Context) -> PhaseOutcome:
    from .measure.ab import (ArmSql, build_report, measure_sql_for_arm,
                             run_interleaved, wilcoxon_p)
    from .measure.arms import both_arms
    from .measure.corpus import load_corpus
    from .measure.sql import MySQLTarget

    corpus = load_corpus(ctx.options.get("corpus"))
    with both_arms(
        f"measure-{ctx.run_id}", repo=ctx.repo, blue_port=ctx.blue_port,
        green_port=ctx.green_port, java_home=ctx.java_home,
        build=ctx.options.get("build", True),
    ) as pair:
        run = run_interleaved(
            pair.blue.base_url, pair.green.base_url, corpus,
            warmup=ctx.options.get("ab_warmup", 3),
            iterations=ctx.options.get("ab_iterations", 8),
        )
        target = MySQLTarget.from_env()
        blue_sql = measure_sql_for_arm(pair.blue.base_url, corpus, target)
        green_sql = measure_sql_for_arm(pair.green.base_url, corpus, target)
        report = build_report(run, blue_sql, green_sql)
        argv = pair.argv_record()

    document = {
        "report": report.model_dump(mode="json"),
        "blue_url": f"http://localhost:{ctx.blue_port}",
        "green_url": f"http://localhost:{ctx.green_port}",
        "iterations": ctx.options.get("ab_iterations", 8),
        "warmup_rounds": ctx.options.get("ab_warmup", 3),
        "wilcoxon_p": wilcoxon_p(run.pairs),
        "arm_argv": argv,
        "jvm_flags_identical": True,
        "firing_order": [
            {"seq": s.seq, "arm": s.arm, "request_id": s.request_id,
             "status": s.status, "latency_ms": round(s.latency_ms, 3)}
            for s in run.samples
        ],
        "pairs": [
            {"iteration": p.iteration, "request_id": p.request_id,
             "blue_ms": round(p.blue_ms, 3), "green_ms": round(p.green_ms, 3),
             "delta_ms": round(p.delta_ms, 3)} for p in run.pairs
        ],
        "status_counts": run.status_counts,
        "sql_digests": {
            "blue": [d.model_dump(mode="json") for d in blue_sql.digests[:20]],
            "green": [d.model_dump(mode="json") for d in green_sql.digests[:20]],
        },
    }
    path = ctx.run_dir / "perf_report.json"
    path.write_text(json.dumps(document, indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8")
    direction = "faster" if report.p95_delta_pct < 0 else "slower"
    return PhaseOutcome(
        artifact=path,
        summary=f"measured p95 {report.p95_delta_pct:+.1f}% ({direction})",
        detail={"p95_delta_pct": report.p95_delta_pct},
    )


def _run_gate(ctx: Context) -> PhaseOutcome:
    from .ship import run_gate

    gated = run_gate(ctx.run_dir)
    failed = [r.name.value for r in gated["decision"].results if not r.passed]
    path = ctx.run_dir / "gate_preview.json"
    path.write_text(
        json.dumps(gated["decision"].model_dump(mode="json"), indent=2) + "\n",
        encoding="utf-8",
    )
    summary = ("gate PASSED" if gated["decision"].passed
               else f"gate FAILED: {', '.join(failed)}")
    return PhaseOutcome(artifact=path, summary=summary,
                        detail={"failed": failed})


def _run_ship(ctx: Context) -> PhaseOutcome:
    from .ship import run_ship

    result = run_ship(
        ctx.run_dir, client=_metered_llm(ctx, "ship"),
        dry_run=ctx.options.get("dry_run", True),
    )
    return PhaseOutcome(
        artifact=result["gate_path"], terminal=result["terminal"],
        summary=result["terminal"].value,
        detail={"report": str(result["report_path"])},
    )


def _metered_llm(ctx: Context, phase: str) -> MeteredClient:
    from .patch.agent import _require_api_key
    from .llm import GroqClient

    return MeteredClient(GroqClient(api_key=_require_api_key()), ctx.budget, phase)


# ────────────────────────────  validators  ───────────────────────────────


def _valid_json(ctx: Context, path: Path) -> str | None:
    try:
        json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return f"{path} is not readable JSON: {exc}"
    return None


def _valid_oracle(ctx: Context, path: Path) -> str | None:
    from .oracle.record import load_oracle

    try:
        load_oracle(path)
    except ContractViolation as exc:
        return str(exc)
    return None


def _valid_endpoint_model(ctx: Context, path: Path) -> str | None:
    """The comprehension model must actually describe an endpoint.

    `_valid_json` was too weak here and it mattered: a stub file containing
    `{"phase": "understand"}` is valid JSON, so a destroyed artifact passed
    validation and the phase was skipped as though it had been done. A
    validator that only checks parseability certifies syntax, not evidence.
    """
    problem = _valid_json(ctx, path)
    if problem:
        return problem
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return f"{path} does not contain an object"
    missing = [
        field for field in ("route", "call_chain", "data_access")
        if not payload.get(field)
    ]
    if missing:
        return (
            f"{path} is not an endpoint model — missing or empty: "
            f"{', '.join(missing)}. It parses as JSON but describes nothing."
        )
    return None


def _valid_corpus(ctx: Context, path: Path) -> str | None:
    from .measure.corpus import load_corpus

    try:
        load_corpus(path)
    except ContractViolation as exc:
        return str(exc)
    return None


def _valid_patch(ctx: Context, path: Path) -> str | None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("outcome") != "PATCHED":
        return f"patch.json records outcome={payload.get('outcome')}, not PATCHED"
    return None


# ──────────────────────────────  the pipeline  ───────────────────────────
#
# Order is a module constant. Nothing reorders it, nothing inserts into it, and
# no argument to run_pipeline changes it.

PIPELINE: tuple[PhaseSpec, ...] = (
    PhaseSpec(Phase.COMPREHEND, "understand",
              _fixed(".perfagent/endpoint_model.json"), None,
              _valid_endpoint_model,
              describe="endpoint comprehension model"),
    PhaseSpec(Phase.WORKLOAD, "workload",
              _fixed("perfagent/measure/corpus.json"), None, _valid_corpus,
              describe="request corpus"),
    PhaseSpec(Phase.ORACLE, "oracle",
              _fixed(".perfagent/oracle.json"), _run_oracle, _valid_oracle,
              describe="frozen behaviour recording"),
    PhaseSpec(Phase.BASELINE, "baseline",
              _run_artifact("baseline.json"), _run_baseline, _valid_json,
              describe="blue-arm latency and SQL evidence"),
    PhaseSpec(Phase.DIAGNOSE, "diagnose",
              _run_artifact("hypotheses.json"), _run_diagnose, _valid_json,
              uses_llm=True, describe="ranked bottleneck hypotheses"),
    PhaseSpec(Phase.IMPLEMENT, "patch",
              _run_artifact("patch.json"), _run_patch, _valid_patch,
              uses_llm=True, describe="the code change"),
    PhaseSpec(Phase.VERIFY, "verify",
              _run_artifact("verification.json"), _run_verify, _valid_json,
              describe="behaviour equivalence verdict"),
    PhaseSpec(Phase.MEASURE, "measure",
              _run_artifact("perf_report.json"), _run_measure, _valid_json,
              describe="interleaved A/B measurement"),
    PhaseSpec(Phase.GATE, "gate",
              _run_artifact("gate_preview.json"), _run_gate, _valid_json,
              describe="the six gates"),
    PhaseSpec(Phase.SHIP, "ship",
              _run_artifact("gate.json"), _run_ship, _valid_json,
              uses_llm=True, describe="report and PR decision"),
)

BY_NAME = {spec.name: spec for spec in PIPELINE}
BY_PHASE = {spec.phase: spec for spec in PIPELINE}


# ──────────────────────────  transition guards  ──────────────────────────


def _preconditions(spec: PhaseSpec, ctx: Context, state: RunState) -> str | None:
    """Why this phase may NOT be entered, or None.

    These are transition guards, not advice. They run before the runner, they
    read artifacts rather than trusting `state.completed_phases`, and there is
    no argument that turns them off.
    """
    if spec.phase is Phase.IMPLEMENT:
        from .oracle.record import ORACLE_PATH, load_oracle

        try:
            load_oracle(ORACLE_PATH)
        except ContractViolation as exc:
            return (
                "cannot enter the patch phase without a recorded oracle: "
                f"{exc} A patch written before behaviour is frozen has nothing "
                "to be checked against, and recording the oracle afterwards "
                "would freeze the patch's own behaviour as the baseline."
            )
        if not (ctx.run_dir / "hypotheses.json").exists():
            return ("cannot enter the patch phase without a diagnosis: "
                    f"{ctx.run_dir / 'hypotheses.json'} is missing.")

    if spec.phase is Phase.MEASURE:
        from .measure.arms import existing_worktree
        from .measure.verify import require_verified

        try:
            green = existing_worktree(".perfagent/green", repo=ctx.repo)
            require_verified(ctx.run_dir, green_diff=green.source_diff())
        except ContractViolation as exc:
            return f"cannot enter the measure phase: {exc}"

    if spec.phase in (Phase.GATE, Phase.SHIP):
        for needed in ("verification.json", "perf_report.json", "patch.json"):
            if not (ctx.run_dir / needed).exists():
                return (f"cannot enter the {spec.name} phase: "
                        f"{needed} has not been produced.")
    return None


# ───────────────────────────────  run state  ─────────────────────────────


def state_path(run_dir: Path) -> Path:
    return Path(run_dir) / STATE_FILE


def load_state(run_dir: Path) -> RunState:
    path = state_path(run_dir)
    if not path.exists():
        raise ContractViolation(
            f"no run state at {path}. There is nothing to resume; start a new "
            "run without --resume."
        )
    return RunState.model_validate(json.loads(path.read_text(encoding="utf-8")))


def save_state(state: RunState, run_dir: Path) -> Path:
    path = state_path(run_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(state.model_dump(mode="json"), indent=2, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )
    return path


def _artifact_ref(kind: str, path: Path) -> ArtifactRef:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return ArtifactRef(kind=kind, hash=content_hash(payload), path=str(path))


# ─────────────────────────────  the state machine  ───────────────────────


def run_pipeline(
    *,
    run_id: str | None = None,
    repo: str | Path | None = None,
    endpoint: str = "POST /api/customers/search",
    runs_dir: Path = RUNS_DIR,
    resume: str | None = None,
    max_usd: float | None = DEFAULT_MAX_USD,
    max_wall_s: float | None = DEFAULT_MAX_WALL_S,
    blue_port: int = DEFAULT_BLUE_PORT,
    green_port: int = DEFAULT_GREEN_PORT,
    java_home: str | None = None,
    options: dict[str, Any] | None = None,
    on_event: Callable[[dict[str, Any]], None] | None = None,
) -> RunState:
    """Walk the pipeline in order. Resume picks up after the last completed phase."""
    if resume:
        run_dir = Path(runs_dir) / resume
        state = load_state(run_dir)
        identifier = state.run_id
    else:
        identifier = run_id or new_run_id()
        run_dir = Path(runs_dir) / identifier
        run_dir.mkdir(parents=True, exist_ok=True)
        state = RunState(
            run_id=identifier, repo=str(repo or ""), endpoint=endpoint,
            phase=PIPELINE[0].phase,
        )
        save_state(state, run_dir)

    budget = Budget(max_usd=max_usd, max_wall_s=max_wall_s)
    trace = Trace(run_dir, state.run_id)
    ctx = Context(
        run_id=state.run_id, run_dir=run_dir,
        repo=Path(repo) if repo else (Path(state.repo) if state.repo else None),
        endpoint=state.endpoint, budget=budget, trace=trace,
        blue_port=blue_port, green_port=green_port, java_home=java_home,
        options=options or {},
    )

    def emit(event: dict[str, Any]) -> None:
        if on_event is not None:
            on_event(event)

    trace.emit(EventType.PHASE_START, phase="run", endpoint=state.endpoint,
               resumed=bool(resume), run_id=state.run_id)
    emit({"kind": "run_start", "run_id": state.run_id,
          "endpoint": state.endpoint, "resumed": bool(resume)})

    for spec in PIPELINE:
        # ── already done? ────────────────────────────────────────────────
        artifact = spec.artifact(ctx)
        already = spec.phase in state.completed_phases
        valid = artifact.exists() and (
            spec.validate is None or spec.validate(ctx, artifact) is None
        )
        if already or valid:
            reason = "completed" if already else "artifact present"
            state = _complete(state, spec, artifact, run_dir, skipped=True)
            trace.emit(EventType.PHASE_END, phase=spec.name, skipped=True,
                       reason=reason, artifact=str(artifact))
            emit({"kind": "phase_skipped", "phase": spec.name,
                  "reason": reason, "artifact": str(artifact)})
            continue

        # ── may we enter? ────────────────────────────────────────────────
        blocked = _preconditions(spec, ctx, state)
        if blocked is not None:
            trace.emit(EventType.GUARDRAIL_TRIGGERED, phase=spec.name,
                       reason=blocked)
            emit({"kind": "blocked", "phase": spec.name, "reason": blocked})
            return _terminate(
                state, run_dir, trace, TerminalState.INTERNAL_ERROR, blocked,
                budget, emit,
            )

        if spec.runner is None:
            message = (
                f"the {spec.name} phase has no runner in this build and its "
                f"artifact ({artifact}) is absent. Produce it first."
            )
            trace.emit(EventType.GUARDRAIL_TRIGGERED, phase=spec.name,
                       reason=message)
            emit({"kind": "blocked", "phase": spec.name, "reason": message})
            return _terminate(
                state, run_dir, trace, TerminalState.INTERNAL_ERROR, message,
                budget, emit,
            )

        # ── budget before the work, not after ────────────────────────────
        exceeded = budget.check()
        if exceeded is not None:
            return _terminate(
                state, run_dir, trace, TerminalState.BUDGET_EXHAUSTED,
                exceeded, budget, emit,
            )

        # ── run it ───────────────────────────────────────────────────────
        state.phase = spec.phase
        save_state(state, run_dir)
        trace.emit(EventType.PHASE_START, phase=spec.name)
        emit({"kind": "phase_start", "phase": spec.name,
              "describe": spec.describe})
        started = time.monotonic()

        try:
            outcome = spec.runner(ctx)
        except BudgetExhausted as exc:
            return _terminate(
                state, run_dir, trace, TerminalState.BUDGET_EXHAUSTED,
                str(exc), budget, emit,
            )
        except (ToolError, ContractViolation) as exc:
            trace.emit(EventType.NOTE, phase=spec.name, error=str(exc))
            emit({"kind": "phase_failed", "phase": spec.name, "error": str(exc)})
            return _terminate(
                state, run_dir, trace, _failure_state(spec), str(exc),
                budget, emit,
            )
        except PerfAgentError as exc:
            return _terminate(
                state, run_dir, trace, TerminalState.INTERNAL_ERROR, str(exc),
                budget, emit,
            )

        elapsed_ms = (time.monotonic() - started) * 1000.0
        if outcome.artifact is not None:
            state = _complete(state, spec, outcome.artifact, run_dir)
        state.tokens_in, state.tokens_out = budget.tokens_in, budget.tokens_out
        state.usd_spent = round(budget.usd, 6)
        save_state(state, run_dir)

        trace.emit(EventType.PHASE_END, phase=spec.name, skipped=False,
                   summary=outcome.summary, duration_ms=round(elapsed_ms, 1),
                   artifact=str(outcome.artifact) if outcome.artifact else None,
                   **outcome.detail if isinstance(outcome.detail, dict) else {})
        emit({"kind": "phase_end", "phase": spec.name,
              "summary": outcome.summary, "duration_ms": elapsed_ms})

        if outcome.terminal is not None:
            return _terminate(
                state, run_dir, trace, outcome.terminal, outcome.summary,
                budget, emit,
            )

    # Reaching the end without a terminal state should be impossible: SHIP
    # always sets one. Say so rather than reporting success by default.
    return _terminate(
        state, run_dir, trace, TerminalState.INTERNAL_ERROR,
        "the pipeline finished without any phase declaring a terminal state",
        budget, emit,
    )


def _failure_state(spec: PhaseSpec) -> TerminalState:
    """The honest terminal state for a phase that raised."""
    if spec.phase in (Phase.ORACLE, Phase.BASELINE):
        # Both need a healthy app and database; a failure there is the
        # environment, not the endpoint.
        return TerminalState.ENVIRONMENT_INFEASIBLE
    if spec.phase is Phase.VERIFY:
        return TerminalState.BEHAVIOUR_NOT_PRESERVED
    if spec.phase is Phase.IMPLEMENT:
        return TerminalState.BOTTLENECK_NOT_FIXABLE
    return TerminalState.INTERNAL_ERROR


def _complete(
    state: RunState, spec: PhaseSpec, artifact: Path, run_dir: Path,
    *, skipped: bool = False,
) -> RunState:
    if spec.phase not in state.completed_phases:
        state.completed_phases.append(spec.phase)
    if artifact.exists():
        try:
            state.artifacts[spec.name] = _artifact_ref(spec.name, artifact)
        except (OSError, json.JSONDecodeError):
            pass
    state.phase = spec.phase
    save_state(state, run_dir)
    return state


def _terminate(
    state: RunState, run_dir: Path, trace: Trace, terminal: TerminalState,
    reason: str, budget: Budget, emit: Callable[[dict[str, Any]], None],
) -> RunState:
    """Stop cleanly: record the state, the reason, and whatever was proven."""
    state.terminal = terminal
    state.tokens_in, state.tokens_out = budget.tokens_in, budget.tokens_out
    state.usd_spent = round(budget.usd, 6)
    save_state(state, run_dir)

    trace.emit(EventType.TERMINAL_STATE, phase="run", terminal=terminal.value,
               reason=reason, tokens_in=budget.tokens_in,
               tokens_out=budget.tokens_out, usd=round(budget.usd, 6),
               elapsed_s=round(budget.elapsed_s(), 1))
    (run_dir / "cost.json").write_text(
        json.dumps({
            "tokens_in": budget.tokens_in, "tokens_out": budget.tokens_out,
            "usd": round(budget.usd, 6), "elapsed_s": round(budget.elapsed_s(), 1),
            "by_phase": budget.by_phase,
            "usd_per_million_input": budget.usd_per_million_input,
            "usd_per_million_output": budget.usd_per_million_output,
        }, indent=2) + "\n",
        encoding="utf-8",
    )
    emit({"kind": "terminal", "terminal": terminal.value, "reason": reason,
          "completed": [p.value for p in state.completed_phases]})
    return state
