"""The pipeline, as a state machine in plain Python.

The phase ORDER is a module constant. No model is asked what to do next, no
model can skip a phase, and there is no branch in this file that a model's
output can steer — invariant 6 applied to control flow rather than to verdicts.

Three transitions carry guards that exist because a prompt asking nicely is not
a guarantee:

  * IMPLEMENT cannot be reached without a recorded, hash-verified oracle. A
    patch written before behaviour was frozen has nothing to be checked
    against, and recording the oracle afterwards would record the patch's own
    behaviour as the baseline.
  * IMPLEMENT also cannot be reached without a PLAN. Without one the patch
    agent takes the hypothesis with the largest attributed milliseconds, which
    on the endpoint this was built against is the response body itself — three
    runs, three oracle rejections, zero PRs.
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

from .contracts import (ArtifactRef, GateName, Phase, RunState,
                        TerminalState, Usage)
from .errors import BudgetExhausted, ContractViolation, PerfAgentError, ToolError
from .ids import content_hash, new_run_id
from .trace import EventType, Trace

RUNS_DIR = Path(".perfagent/runs")
STATE_FILE = "run_state.json"

# No default ports. 18080/8081 were lifted from one repo's
# `application-perfagent.yml`; on a machine where something else holds 8081
# the green arm answered 401 and the run measured a stranger's service.
# `measure.arms.free_port_pair()` asks the kernel instead, at provision time.
DEFAULT_BLUE_PORT: int | None = None
DEFAULT_GREEN_PORT: int | None = None

# Token pricing is NOT a measurement. The provider's rate is configuration, so
# the dollar figure is derived from a rate someone set rather than something
# observed — tokens are the ground truth and are always reported.
#
# These flat rates are the FALLBACK, used only for a response that does not
# say which model produced it. The real figure comes from `llm/pricing.py`,
# per model, because a run can use a different model per phase and one flat
# rate would quietly average them. Leaving these at zero is what made
# `perfagent trace` print $0.00 for every phase of every run.
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

    def charge(self, phase: str, tokens_in: int, tokens_out: int,
               model: str | None = None) -> float:
        usd = self._price(tokens_in, tokens_out, model)
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

    def _price(self, tokens_in: int, tokens_out: int,
               model: str | None) -> float:
        """What the provider's published rate says this call cost.

        An unknown model falls back to the flat rates and `pricing.usd_for`
        warns once, naming the model — a zero that announces itself is
        recoverable; a silent one is what this fixed.
        """
        if model:
            from .llm.pricing import is_known, usd_for

            if is_known(model):
                return usd_for(model, Usage(input_tokens=tokens_in,
                                            output_tokens=tokens_out))
        return (tokens_in / 1_000_000 * self.usd_per_million_input
                + tokens_out / 1_000_000 * self.usd_per_million_output)

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

    Delegates everything else, because the loop asks the client to format tool
    results (`tool_result_messages`) and a proxy that hid provider-specific
    methods would silently fall back to the wrong wire shape.
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
                getattr(response, "model", None) or getattr(self._inner, "model", None),
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
    blue_port: int | None = DEFAULT_BLUE_PORT
    green_port: int | None = DEFAULT_GREEN_PORT
    java_home: str | None = None
    options: dict[str, Any] = field(default_factory=dict)

    def profile(self) -> Any:
        """The detected RepoProfile for this run, read from its artifact.

        A method rather than a field so that a resumed run picks the profile
        off disk exactly like every other artifact, instead of depending on
        whatever the process that resumed it happened to hold in memory.
        """
        from .onboard import load_profile

        return load_profile(self.run_dir / "repo_profile.json")

    def provisioning(self) -> Any:
        from .provision import load_provisioning

        return load_provisioning(self.run_dir / "provisioning.json")


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



def _run_onboard(ctx: Context) -> PhaseOutcome:
    """Read the repository and write down what it is.

    First phase on purpose: a repo this harness cannot onboard must fail
    here, before a single token is spent.
    """
    from .onboard import assert_instrumentable, onboard, save_profile

    if ctx.repo is None:
        return PhaseOutcome(
            terminal=TerminalState.ENVIRONMENT_INFEASIBLE,
            summary="no target repository was given",
            detail={"advice": "pass --repo <path>"},
        )
    profile = onboard(ctx.repo)
    assert_instrumentable(profile)
    artifact = save_profile(profile, ctx.run_dir / "repo_profile.json")
    return PhaseOutcome(
        artifact=artifact,
        summary=(f"{profile.build.tool.value} · {profile.entrypoint} · "
                 f"{profile.database.kind.value}"),
        detail={
            "source_root": profile.layout.source_root,
            "write_allowlist": profile.write_allowlist,
            "migrations": profile.migrations.tool.value,
        },
    )


def _run_provision(ctx: Context) -> PhaseOutcome:
    """Free ports, a reachable database, a current schema."""
    from .provision import provision, save_provisioning

    profile = ctx.profile()
    record = provision(
        profile, repo=ctx.repo,
        blue_port=ctx.blue_port, green_port=ctx.green_port,
        java_home=ctx.java_home,
        require_database=ctx.options.get("require_database", True),
        run_migrations=ctx.options.get("run_migrations", True),
    )
    # The rest of the pipeline reads the ports off the context, so bind them
    # here rather than making every phase re-open the artifact.
    ctx.blue_port, ctx.green_port = record.blue.port, record.green.port
    artifact = save_provisioning(record, ctx.run_dir / "provisioning.json")
    return PhaseOutcome(
        artifact=artifact,
        summary=f"blue :{record.blue.port}  green :{record.green.port}",
        detail={"database_ready": record.database_ready,
                "migrations": record.migrations_applied},
    )


def _run_understand(ctx: Context) -> PhaseOutcome:
    """Trace the endpoint the run was actually asked about.

    `--endpoint` used to be decorative: the CLI accepted it, this file carried
    it, and the phase never received it — it only checked whether an artifact
    file existed and skipped. The endpoint is now passed in, and the skip check
    (`_valid_endpoint_model`) compares the artifact's route against it.
    """
    from .agent import LoopBudget
    from .understand.agent import understand

    if ctx.repo is None:
        raise ContractViolation(
            "the understand phase needs a target repository. Pass --repo or "
            "set PERFAGENT_TARGET_REPO."
        )

    model = understand(
        repo=ctx.repo,
        endpoint=ctx.endpoint,
        client=_metered_llm(ctx, "understand"),
        trace=ctx.trace,
        budget=LoopBudget(
            max_steps=ctx.options.get("understand_max_steps", 30),
            max_usd=ctx.budget.max_usd,
            max_wall_s=ctx.budget.remaining_s() if ctx.budget.max_wall_s else None,
        ),
        run_dir=ctx.run_dir,
    )
    per_row = len(model.per_row_accesses())
    return PhaseOutcome(
        artifact=ctx.run_dir / "endpoint_model.json",
        summary=f"{model.route}: {len(model.call_chain)} hops, "
                f"{len(model.data_access)} data accesses ({per_row} per-row)",
        detail={"route": model.route, "per_row_accesses": per_row,
                "tables": len(model.tables),
                "prompt_hash": model.prompt_hash},
    )



def _run_workload(ctx: Context) -> PhaseOutcome:
    """Generate THIS run's corpus from the comprehension model and the DB."""
    from .measure.corpus import save_corpus
    from .onboard import load_profile
    from .tools.sample import ValueSampler
    from .understand.agent import load_model
    from .workload import generate_corpus

    model = load_model(ctx.run_dir / "endpoint_model.json")
    profile = load_profile(ctx.run_dir / "repo_profile.json")

    sampler = ctx.options.get("sampler")
    if sampler is None:
        from .measure.sql import MySQLTarget
        sampler = ValueSampler(
            MySQLTarget.from_env(),
            allowed_tables=[t.table for t in model.tables],
        )
    corpus, notes = generate_corpus(model, sampler=sampler)
    artifact = save_corpus(corpus, ctx.run_dir / "corpus.json")
    for note in notes:
        ctx.trace.emit(EventType.NOTE, phase="workload", note=note)
    return PhaseOutcome(
        artifact=artifact,
        summary=(f"{len(corpus.benchmark)} benchmark + "
                 f"{len(corpus.held_back)} held-back request(s), "
                 f"hash {corpus.hash}"),
        detail={"corpus_hash": corpus.hash, "notes": notes,
                "endpoint": ctx.endpoint},
    )


def _run_oracle(ctx: Context) -> PhaseOutcome:
    from .measure.arms import single_arm
    from .measure.corpus import load_corpus
    from .oracle.canonicalise import load_policy
    from .oracle.record import ORACLE_PATH, record_oracle

    corpus = load_corpus(ctx.run_dir / "corpus.json")
    with single_arm(
        ctx.run_id, which="blue", repo=ctx.repo, port=ctx.blue_port,
        java_home=ctx.java_home, profile=ctx.profile(), build=ctx.options.get("build", True),
    ) as arm:
        bundle = record_oracle(
            arm.base_url, corpus, policy=load_policy(),
            repo_path=ctx.repo, path=ctx.run_dir / "oracle.json", force=False,
            endpoint=ctx.endpoint,
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
        java_home=ctx.java_home, profile=ctx.profile(), build=ctx.options.get("build", True),
    ) as arm:
        document, produced = run_baseline(
            base_url=arm.base_url,
            corpus_path=ctx.run_dir / "corpus.json",
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
        java_home=ctx.java_home, profile=ctx.profile(), build=ctx.options.get("build", True),
    ) as arm:
        hypothesis_set, probe_log, path = run_diagnosis(
            base_url=arm.base_url, run_dir=ctx.run_dir, client=client,
            trace=ctx.trace,
            # This run's comprehension artifact, not a global file that may
            # describe a different endpoint entirely.
            endpoint_model_path=ctx.run_dir / "endpoint_model.json",
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


def _run_plan(ctx: Context) -> PhaseOutcome:
    """Rank the diagnosis by expected value, then plan candidate fixes.

    The ranking is code and happens whether or not the agent produces
    anything, so a run that ends here still records WHY each hypothesis was
    or was not worth attacking.
    """
    from .agent import LoopBudget
    from .plan.agent import load_inputs, plan

    if ctx.repo is None:
        raise ContractViolation(
            "the plan phase needs a target repository. Pass --repo or set "
            "PERFAGENT_TARGET_REPO."
        )

    hypotheses, model = load_inputs(ctx.run_dir)
    result = plan(
        repo=ctx.repo,
        endpoint=ctx.endpoint or model.route,
        hypotheses=hypotheses,
        model=model,
        client=_metered_llm(ctx, "plan"),
        trace=ctx.trace,
        budget=LoopBudget(
            max_steps=ctx.options.get("plan_max_steps", 24),
            max_usd=ctx.budget.max_usd,
            max_wall_s=ctx.budget.remaining_s() if ctx.budget.max_wall_s else None,
        ),
        run_dir=ctx.run_dir,
    )

    artifact = ctx.run_dir / "plan.json"
    selected = result.selected()
    if selected is None:
        return PhaseOutcome(
            artifact=artifact, terminal=TerminalState.BOTTLENECK_NOT_FIXABLE,
            summary=f"{len(result.candidates)} candidate(s), none selectable",
            detail={"reason": result.selection_reason},
        )
    return PhaseOutcome(
        artifact=artifact,
        summary=f"selected {selected.id} ({selected.approach}, "
                f"T{selected.risk_tier.value}) targeting "
                f"{selected.targets_hypothesis}",
        detail={"selected": selected.id, "approach": selected.approach,
                "risk_tier": selected.risk_tier.value,
                "expected_value_ms": selected.expected_value_ms,
                "candidates": len(result.candidates)},
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
        java_home=ctx.java_home, profile=ctx.profile(), client=_metered_llm(ctx, "patch"),
        candidate_index=int(ctx.options.get("candidate_index", 0)),
        oracle_feedback=str(ctx.options.get("oracle_feedback", "")),
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


def _repo_root(ctx: Context) -> Path:
    from .patch.worktree import target_repo

    return target_repo(ctx.repo)


def _run_verify(ctx: Context) -> PhaseOutcome:
    from .measure.arms import both_arms, existing_worktree
    from .measure.verify import run_verification

    green = existing_worktree(".perfagent/green", repo=ctx.repo)
    with both_arms(
        f"verify-{ctx.run_id}", repo=ctx.repo, blue_port=ctx.blue_port,
        green_port=ctx.green_port, java_home=ctx.java_home, profile=ctx.profile(),
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

    corpus = load_corpus(ctx.run_dir / "corpus.json")
    with both_arms(
        f"measure-{ctx.run_id}", repo=ctx.repo, blue_port=ctx.blue_port,
        green_port=ctx.green_port, java_home=ctx.java_home, profile=ctx.profile(),
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
    """One client per phase, built from config.

    The provider and model are configuration (perfagent.yml / env), not
    literals in this file — switching provider must not require a code edit.
    """
    from .llm import make_client

    return MeteredClient(make_client(), ctx.budget, phase)


# ────────────────────────────  validators  ───────────────────────────────


def _isolation_problem(ctx: Context) -> str | None:
    """Whether this run's patch can be measured at all, or None if it can."""
    from .measure.arms import existing_worktree
    from .measure.isolation import isolation_problem

    patch_path = ctx.run_dir / "patch.json"
    plan_path = ctx.run_dir / "plan.json"
    approach, files = "", []
    if plan_path.exists():
        try:
            from .plan.agent import load_plan

            selected = load_plan(plan_path).selected()
            approach = selected.approach if selected else ""
        except Exception:  # noqa: BLE001 — a missing plan is caught elsewhere
            approach = ""
    if patch_path.exists():
        try:
            payload = json.loads(patch_path.read_text(encoding="utf-8"))
            files = list(payload.get("patch", {}).get("files_changed") or [])
        except (OSError, json.JSONDecodeError):
            files = []

    try:
        green = existing_worktree(".perfagent/green", repo=ctx.repo)
        diff = green.source_diff()
        green_path = green.path
    except Exception:  # noqa: BLE001
        return None
    blue = _repo_root(ctx)
    return isolation_problem(blue, green_path, approach=approach, diff=diff,
                             files_changed=files)


def _valid_json(ctx: Context, path: Path) -> str | None:
    try:
        json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return f"{path} is not readable JSON: {exc}"
    return None


def _valid_oracle(ctx: Context, path: Path) -> str | None:
    """Self-consistent AND recorded against THIS run's endpoint.

    The hash check alone let a recording of another route satisfy "an oracle
    exists". Behaviour equivalence would then be asserted against responses
    from a different endpoint -- the single most load-bearing comparison in
    the product, made against the wrong evidence.
    """
    from .oracle.record import load_oracle

    try:
        oracle = load_oracle(path)
    except ContractViolation as exc:
        return str(exc)

    if ctx.endpoint and oracle.endpoint:
        found, wanted = (_normalise_route(oracle.endpoint),
                         _normalise_route(ctx.endpoint))
        if found != wanted:
            return (f"{path} was recorded against {oracle.endpoint!r}, but "
                    f"this run is about {ctx.endpoint!r}.")
    return None


def _normalise_route(route: str) -> tuple[str, str]:
    """('POST', '/api/customers/search') from any reasonable spelling."""
    parts = str(route or "").strip().split(None, 1)
    if len(parts) == 2:
        method, path = parts
    else:
        method, path = "", parts[0] if parts else ""
    path = "/" + path.strip().strip("/")
    return method.strip().upper(), path


def _valid_endpoint_model(ctx: Context, path: Path) -> str | None:
    """The comprehension model must describe THIS RUN'S endpoint.

    `_valid_json` was too weak here and it mattered: a stub file containing
    `{"phase": "understand"}` is valid JSON, so a destroyed artifact passed
    validation and the phase was skipped as though it had been done. A
    validator that only checks parseability certifies syntax, not evidence.

    Route matching is the second half of the same lesson. An artifact for
    `GET /api/customers/{id}` satisfies "a comprehension model exists" and
    tells the rest of the pipeline nothing about the endpoint under test, so
    a cached model for a different route must NOT let this phase skip.
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
    if ctx.endpoint:
        found = _normalise_route(str(payload.get("route", "")))
        wanted = _normalise_route(ctx.endpoint)
        if found != wanted:
            return (
                f"{path} describes {payload.get('route')!r}, but this run is "
                f"about {ctx.endpoint!r}. A cached model for a different route "
                f"does not satisfy this phase."
            )
    return None


def _valid_plan(ctx: Context, path: Path) -> str | None:
    """A plan artifact must carry a SELECTED candidate for THIS endpoint.

    Same lesson as `_valid_endpoint_model`: a file that parses is not evidence.
    A plan with no selection means the phase ran and found nothing shippable,
    and skipping into `patch` on the strength of it would hand the patch agent
    an empty allowlist.
    """
    problem = _valid_json(ctx, path)
    if problem:
        return problem
    from .plan.agent import load_plan

    try:
        plan = load_plan(path)
    except Exception as exc:  # noqa: BLE001 — a bad artifact is a reason, not a crash
        return f"{path} is not a plan artifact: {type(exc).__name__}: {exc}"

    if plan.selected() is None:
        return (f"{path} records no selected candidate "
                f"({plan.selection_reason or 'no reason given'}).")
    if ctx.endpoint:
        found, wanted = _normalise_route(plan.endpoint), _normalise_route(ctx.endpoint)
        if found != wanted:
            return (f"{path} plans {plan.endpoint!r}, but this run is about "
                    f"{ctx.endpoint!r}.")
    return None



def _valid_repo_profile(ctx: Context, path: Path) -> str | None:
    """A profile must parse AND describe the repo this run was pointed at.

    A profile left over from another repository satisfies "a profile
    exists" and tells the pipeline nothing true -- the same trap
    `_valid_endpoint_model` exists to close for the comprehension model.
    """
    problem = _valid_json(ctx, path)
    if problem:
        return problem
    from .onboard import load_profile

    try:
        profile = load_profile(path)
    except Exception as exc:  # noqa: BLE001
        return f"{path} is not a repo profile: {type(exc).__name__}: {exc}"
    if ctx.repo is not None:
        wanted = Path(ctx.repo).expanduser().resolve()
        if Path(profile.repo) != wanted:
            return (f"{path} profiles {profile.repo}, but this run is about "
                    f"{wanted}.")
    return None


def _valid_provisioning(ctx: Context, path: Path) -> str | None:
    """Ports recorded by a PREVIOUS run are not evidence that they are free
    now. Re-provisioning is cheap; measuring a stranger's service is not."""
    problem = _valid_json(ctx, path)
    if problem:
        return problem
    from .measure.arms import port_is_free
    from .provision import load_provisioning

    try:
        record = load_provisioning(path)
    except Exception as exc:  # noqa: BLE001
        return f"{path} is not a provisioning record: {exc}"
    for arm in (record.blue, record.green):
        if not port_is_free(arm.port):
            return (f"{path} records the {arm.name} arm on port {arm.port}, "
                    f"which is in use now. Re-provisioning.")
    return None


def _valid_corpus(ctx: Context, path: Path) -> str | None:
    """A corpus must parse, be non-empty, and contain ONLY this endpoint.

    It used to check only that the file parsed. A corpus full of
    `POST /api/customers/search` therefore validated cleanly for a run about
    `GET /api/invoices/{id}`, the workload phase was skipped as "artifact
    present", and the oracle was recorded against the wrong endpoint
    entirely -- surfacing much later, if at all, as a corpus-hash mismatch.
    """
    from .measure.corpus import load_corpus

    try:
        corpus = load_corpus(path)
    except ContractViolation as exc:
        return str(exc)

    if not corpus.benchmark:
        return f"{path} has an empty benchmark set; there is nothing to measure."

    if not ctx.endpoint:
        return None
    wanted_method, wanted_path = _normalise_route(ctx.endpoint)
    template = [segment for segment in wanted_path.split("/") if segment]
    offenders: list[str] = []
    for spec in corpus.benchmark + corpus.held_back:
        found_method = (spec.method or "").strip().upper()
        found_path = "/" + (spec.path or "").split("?")[0].strip().strip("/")
        if found_method != wanted_method or not _matches_template(
                found_path, template):
            offenders.append(f"{spec.id} ({found_method} {found_path})")
    if offenders:
        shown = ", ".join(offenders[:5])
        more = f" and {len(offenders) - 5} more" if len(offenders) > 5 else ""
        return (f"{path} contains request(s) that are not this run's "
                f"endpoint ({ctx.endpoint}): {shown}{more}.")
    return None


def _matches_template(concrete: str, template: list[str]) -> bool:
    """Does a concrete path fit the route template it claims to?

    `/api/customers/7/invoices` matches `/api/customers/{id}/invoices`: the
    literal segments must be equal and a `{...}` segment matches anything
    non-empty. Comparing the strings directly would reject every generated
    request, since generation is what substitutes the parameters.
    """
    segments = [s for s in concrete.split("/") if s]
    if len(segments) != len(template):
        return False
    for found, wanted in zip(segments, template):
        if wanted.startswith("{") or wanted.startswith(":"):
            if not found:
                return False
            continue
        if found != wanted:
            return False
    return True


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
    PhaseSpec(Phase.ONBOARD, "onboard",
              _run_artifact("repo_profile.json"), _run_onboard,
              _valid_repo_profile,
              describe="what kind of repository this is"),
    PhaseSpec(Phase.PROVISION, "provision",
              _run_artifact("provisioning.json"), _run_provision,
              _valid_provisioning,
              describe="free ports, reachable database, current schema"),
    PhaseSpec(Phase.COMPREHEND, "understand",
              _run_artifact("endpoint_model.json"), _run_understand,
              _valid_endpoint_model, uses_llm=True,
              describe="endpoint comprehension model"),
    PhaseSpec(Phase.WORKLOAD, "workload",
              _run_artifact("corpus.json"), _run_workload, _valid_corpus,
              describe="request corpus, generated for this endpoint"),
    PhaseSpec(Phase.ORACLE, "oracle",
              _run_artifact("oracle.json"), _run_oracle, _valid_oracle,
              describe="frozen behaviour recording"),
    PhaseSpec(Phase.BASELINE, "baseline",
              _run_artifact("baseline.json"), _run_baseline, _valid_json,
              describe="blue-arm latency and SQL evidence"),
    PhaseSpec(Phase.DIAGNOSE, "diagnose",
              _run_artifact("hypotheses.json"), _run_diagnose, _valid_json,
              uses_llm=True, describe="ranked bottleneck hypotheses"),
    PhaseSpec(Phase.PLAN, "plan",
              _run_artifact("plan.json"), _run_plan, _valid_plan,
              uses_llm=True, describe="ranked candidate fixes"),
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
    if spec.phase is Phase.PLAN:
        if not (ctx.run_dir / "hypotheses.json").exists():
            return ("cannot enter the plan phase without a diagnosis: "
                    f"{ctx.run_dir / 'hypotheses.json'} is missing. A plan "
                    "ranks measured hypotheses; there is nothing to rank.")

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
        # The patch agent implements a NAMED CANDIDATE from a plan. Without
        # one it falls back to the top hypothesis by attributed ms, which is
        # how three runs in a row attacked a bottleneck the oracle could never
        # accept. No argument to this function turns the check off.
        if not (ctx.run_dir / "plan.json").exists():
            return ("cannot enter the patch phase without a plan: "
                    f"{ctx.run_dir / 'plan.json'} is missing. Run "
                    "`perfagent plan` — ranking by raw milliseconds picks the "
                    "biggest bottleneck, not the fixable one.")

    if spec.phase is Phase.MEASURE:
        from .measure.arms import existing_worktree
        from .measure.verify import require_verified

        try:
            green = existing_worktree(".perfagent/green", repo=ctx.repo)
            require_verified(ctx.run_dir, green_diff=green.source_diff())
        except ContractViolation as exc:
            return f"cannot enter the measure phase: {exc}"

        # A schema change cannot be A/B measured while both arms share one
        # database. Checked BEFORE measuring, because the failure mode is a
        # believable number rather than an error — see measure/isolation.py.
        shared = _isolation_problem(ctx)
        if shared is not None:
            return f"ISOLATION: {shared}"

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
    blue_port: int | None = DEFAULT_BLUE_PORT,
    green_port: int | None = DEFAULT_GREEN_PORT,
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

    position = 0
    while position < len(PIPELINE):
        spec = PIPELINE[position]
        position += 1

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
            # "We could not measure this" is not an internal error and must
            # not be filed as one.
            terminal = (TerminalState.MEASUREMENT_NOT_ISOLATED
                        if blocked.startswith("ISOLATION:")
                        else TerminalState.INTERNAL_ERROR)
            return _terminate(
                state, run_dir, trace, terminal, blocked, budget, emit,
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
                state, run_dir, trace, _failure_state(spec, str(exc)),
                str(exc), budget, emit,
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
            retry = _retry_decision(spec, outcome, ctx, state)
            if retry is not None:
                state = _rewind_to_patch(state, ctx, run_dir, retry)
                position = _index_of(Phase.IMPLEMENT)
                trace.emit(EventType.NOTE, phase=spec.name, retry=retry.kind,
                           attempt=ctx.options.get("patch_attempt", 1),
                           candidate=ctx.options.get("candidate_index", 0),
                           note=retry.why)
                emit({"kind": "retry", "phase": spec.name, "reason": retry.why})
                continue
            return _terminate(
                state, run_dir, trace, outcome.terminal, outcome.summary,
                budget, emit,
            )

        # The gate does not set a terminal state — ship does — so a gate that
        # failed ONLY on the improvement threshold is read here, where there
        # is still a candidate left to try.
        if spec.phase is Phase.GATE:
            retry = _improvement_retry(outcome, ctx, state)
            if retry is not None:
                state = _rewind_to_patch(state, ctx, run_dir, retry)
                position = _index_of(Phase.IMPLEMENT)
                trace.emit(EventType.NOTE, phase=spec.name, retry=retry.kind,
                           candidate=ctx.options.get("candidate_index", 0),
                           note=retry.why)
                emit({"kind": "retry", "phase": spec.name, "reason": retry.why})
                continue

    # Reaching the end without a terminal state should be impossible: SHIP
    # always sets one. Say so rather than reporting success by default.
    return _terminate(
        state, run_dir, trace, TerminalState.INTERNAL_ERROR,
        "the pipeline finished without any phase declaring a terminal state",
        budget, emit,
    )


# ───────────────────────────────  retries  ───────────────────────────────
#
# Two failures are worth another attempt, and they are different failures.
#
#   * The oracle REJECTED the patch. The agent changed behaviour, usually in a
#     way it can see and fix once the diff is put in front of it. Worth up to
#     MAX_PATCH_ATTEMPTS tries on the SAME candidate, because the second
#     attempt is informed by evidence the first did not have.
#   * The gate failed ONLY on the improvement threshold. The patch is correct
#     and too small. Another attempt at the same candidate would produce the
#     same patch, so this moves to the NEXT candidate — which the plan already
#     wrote down, with its own approach and its own predicted effect.
#
# Everything else terminates. A retry loop that retries everything is a loop
# that spends a budget discovering the same thing repeatedly.

MAX_PATCH_ATTEMPTS = 2


@dataclass(frozen=True)
class Retry:
    kind: str
    why: str


def _index_of(phase: Phase) -> int:
    for index, spec in enumerate(PIPELINE):
        if spec.phase is phase:
            return index
    raise ContractViolation(f"{phase} is not in the pipeline")


def _selectable_candidates(ctx: Context) -> list[Any]:
    """The plan's candidates this harness is allowed to implement, in order.

    Ordered by expected value already. Candidates the plan ruled out for cause
    — above the risk ceiling, or not claiming to preserve the response bytes —
    are not retried into: they were rejected by code, and a retry is not an
    appeal.
    """
    from .plan.agent import load_plan
    from .plan.risk import selection_problem

    path = ctx.run_dir / "plan.json"
    if not path.exists():
        return []
    try:
        plan = load_plan(path)
    except Exception:  # noqa: BLE001
        return []
    return [c for c in plan.candidates if selection_problem(c) is None]


def _retry_decision(spec: PhaseSpec, outcome: PhaseOutcome, ctx: Context,
                    state: RunState) -> "Retry | None":
    """Whether a failed phase is worth another attempt, and which kind."""
    if outcome.terminal is not TerminalState.BEHAVIOUR_NOT_PRESERVED:
        return None
    if spec.phase not in (Phase.VERIFY, Phase.IMPLEMENT):
        return None

    attempt = int(ctx.options.get("patch_attempt", 1))
    candidates = _selectable_candidates(ctx)
    index = int(ctx.options.get("candidate_index", 0))

    if attempt < MAX_PATCH_ATTEMPTS:
        ctx.options["patch_attempt"] = attempt + 1
        ctx.options["oracle_feedback"] = _oracle_feedback(ctx)
        return Retry(
            kind="behaviour",
            why=(f"the oracle rejected attempt {attempt} of candidate "
                 f"{index + 1}; retrying with the recorded differences as "
                 f"context ({attempt + 1}/{MAX_PATCH_ATTEMPTS})"),
        )

    if index + 1 < len(candidates):
        ctx.options["candidate_index"] = index + 1
        ctx.options["patch_attempt"] = 1
        ctx.options["oracle_feedback"] = ""
        return Retry(
            kind="next_candidate",
            why=(f"candidate {index + 1} changed behaviour on "
                 f"{MAX_PATCH_ATTEMPTS} attempts; moving to candidate "
                 f"{index + 2} of {len(candidates)}"),
        )
    return None


def _improvement_retry(outcome: PhaseOutcome, ctx: Context,
                       state: RunState) -> "Retry | None":
    """A gate that failed ONLY on the improvement threshold gets the next
    candidate. Any other failing gate does not: a behaviour change, a
    mechanism that did not move or a forbidden edit are not fixed by trying a
    different fix for the same bottleneck."""
    failed = list(outcome.detail.get("failed") or [])
    if failed != [GateName.IMPROVEMENT_THRESHOLD.value]:
        return None

    candidates = _selectable_candidates(ctx)
    index = int(ctx.options.get("candidate_index", 0))
    if index + 1 >= len(candidates):
        return None

    ctx.options["candidate_index"] = index + 1
    ctx.options["patch_attempt"] = 1
    ctx.options["oracle_feedback"] = ""
    return Retry(
        kind="next_candidate",
        why=(f"candidate {index + 1} preserved behaviour but did not clear "
             f"the improvement threshold; trying candidate {index + 2} of "
             f"{len(candidates)}, which costs one build"),
    )


def _oracle_feedback(ctx: Context) -> str:
    """The differences the oracle actually recorded, for the next attempt.

    Read off verification.json rather than regenerated: the patch agent is
    being told what the frozen recording said, and anything else would be a
    paraphrase of the only evidence that matters.
    """
    path = ctx.run_dir / "verification.json"
    if not path.exists():
        return ""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return ""
    verdict = payload.get("verdict") or {}
    diffs = verdict.get("diffs") or []
    if not diffs:
        return ""
    lines = [
        f"  request {d.get('request_id')}  path {d.get('path')}\n"
        f"    recorded: {str(d.get('before'))[:160]}\n"
        f"    yours:    {str(d.get('after'))[:160]}"
        for d in diffs[:8]
    ]
    more = (f"\n  ... and {len(diffs) - 8} more differing path(s)"
            if len(diffs) > 8 else "")
    return (f"{len(diffs)} path(s) differed from the frozen recording:\n"
            + "\n".join(lines) + more)


def _rewind_to_patch(state: RunState, ctx: Context, run_dir: Path,
                     retry: "Retry") -> RunState:
    """Delete what the failed attempt produced, so the phases really re-run.

    The skip check reads artifacts from disk, so leaving `patch.json` in place
    would make the retry a no-op that skipped straight back to the failure.
    """
    redo = (Phase.IMPLEMENT, Phase.VERIFY, Phase.MEASURE, Phase.GATE)
    for spec in PIPELINE:
        if spec.phase not in redo:
            continue
        # Ask the spec where its artifact is rather than hardcoding the name.
        # Hardcoding it meant the rewind deleted `verification.json` while the
        # phase had actually written `verify.json`, so the skip check sailed
        # straight past the phase being retried and the run ended with no
        # terminal state at all.
        artifact = spec.artifact(ctx)
        if artifact.exists() and artifact.is_relative_to(Path(run_dir)):
            artifact.unlink()
        if spec.phase in state.completed_phases:
            state.completed_phases.remove(spec.phase)
        state.artifacts.pop(spec.name, None)
    state.phase = Phase.IMPLEMENT
    save_state(state, run_dir)
    return state


# A verify failure whose message is about the ARMS rather than the responses.
# These are the words `arms.py` uses when it refuses to compare two things that
# are not comparable.
_ENVIRONMENT_MARKS = (
    "DIFFERENT flags", "has never been built", "is missing", "did not boot",
    "exited during startup", "is not a git repository", "port",
)


def _failure_state(spec: PhaseSpec, error: str = "") -> TerminalState:
    """The honest terminal state for a phase that raised."""
    # A repo that cannot be read or cannot be started is the environment,
    # by definition. These two phases have no other kind of failure, so the
    # mapping is on the PHASE rather than on words in the message — the
    # string-matching below is a workaround for VERIFY, not a pattern worth
    # copying.
    if spec.phase in (Phase.ONBOARD, Phase.PROVISION):
        return TerminalState.ENVIRONMENT_INFEASIBLE
    if spec.phase is Phase.VERIFY and any(mark in error
                                          for mark in _ENVIRONMENT_MARKS):
        return TerminalState.ENVIRONMENT_INFEASIBLE
    if spec.phase in (Phase.ORACLE, Phase.BASELINE):
        # Both need a healthy app and database; a failure there is the
        # environment, not the endpoint.
        return TerminalState.ENVIRONMENT_INFEASIBLE
    if spec.phase is Phase.VERIFY:
        # Only a real DIFFERENCE is a behaviour change. The verify phase also
        # fails when the two arms could not be made comparable at all — a
        # classpath mismatch, an arm that would not boot — and calling that
        # "behaviour not preserved" blames the patch for the environment.
        # Same principle as the isolation guard: could-not-measure is not
        # did-not-work.
        return TerminalState.BEHAVIOUR_NOT_PRESERVED
    if spec.phase in (Phase.PLAN, Phase.IMPLEMENT):
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
