"""P16 tests: the state machine, the transition guards, the budget, the trace.

No network, no JVM, no LLM. Phase runners are replaced with stubs, because what
is under test is the ORDER, the guards, the resume behaviour and the budget —
not the phases, which have their own tests.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from senapathy.contracts import Phase, RunState, TerminalState
from senapathy.errors import BudgetExhausted, ContractViolation, ToolError
from senapathy import orchestrator as orch
from senapathy.orchestrator import (BY_NAME, PIPELINE, Budget, Context,
                                    MeteredClient, PhaseOutcome, load_state,
                                    run_pipeline, save_state, state_path)
from senapathy.trace import Trace


# ──────────────────────────────  fixtures  ───────────────────────────────


@pytest.fixture
def runs(tmp_path: Path) -> Path:
    return tmp_path / "runs"


def stub_pipeline(monkeypatch, *, fails: dict[str, Exception] | None = None,
                  terminals: dict[str, TerminalState] | None = None,
                  order: list[str] | None = None):
    """Replace every runner with one that writes its artifact and records order.

    The runner is handed the PATCHED artifact resolver, not the original spec's.
    An earlier version closed over the original, so the stubs wrote to the real
    `.senapathy/oracle.json`, `.senapathy/endpoint_model.json` and
    `senapathy/measure/corpus.json` and destroyed them. `_assert_under_run_dir`
    is the belt to that braces.
    """
    fails = fails or {}
    terminals = terminals or {}
    recorded = order if order is not None else []

    def artifact_for(name: str):
        return lambda ctx: ctx.run_dir / f"{name}.json"

    def make(name: str, resolve, terminal):
        def runner(ctx: Context) -> PhaseOutcome:
            recorded.append(name)
            if name in fails:
                raise fails[name]
            path = resolve(ctx)
            _assert_under_run_dir(path, ctx.run_dir)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"phase": name}))
            return PhaseOutcome(artifact=path, terminal=terminal,
                                summary=f"{name} ok")
        return runner

    patched = []
    for spec in PIPELINE:
        resolve = artifact_for(spec.name)
        patched.append(orch.PhaseSpec(
            phase=spec.phase, name=spec.name, artifact=resolve,
            runner=make(spec.name, resolve, terminals.get(spec.name)),
            validate=orch._valid_json, uses_llm=spec.uses_llm,
            describe=spec.describe,
        ))
    patched = tuple(patched)

    monkeypatch.setattr(orch, "PIPELINE", patched)
    monkeypatch.setattr(orch, "BY_NAME", {s.name: s for s in patched})
    # Guards read real artifacts on disk; neutralise them for order tests.
    monkeypatch.setattr(orch, "_preconditions", lambda spec, ctx, state: None)
    return recorded


def _assert_under_run_dir(path: Path, run_dir: Path) -> None:
    """A test must never write outside its own run directory."""
    resolved, root = Path(path).resolve(), Path(run_dir).resolve()
    assert str(resolved).startswith(str(root)), (
        f"a stubbed phase tried to write {resolved}, which is outside the test "
        f"run directory {root}. That is how the real project artifacts got "
        f"destroyed once."
    )


# ────────────────────────────  the fixed order  ──────────────────────────


def test_the_phase_order_is_a_constant_and_matches_the_pipeline() -> None:
    assert [s.name for s in PIPELINE] == [
        # ONBOARD and PROVISION lead: a repository this harness cannot read
        # or cannot start must fail before a single token is spent.
        "onboard", "provision",
        "understand", "workload", "oracle", "baseline", "diagnose",
        "plan", "patch", "verify", "measure", "gate", "ship",
    ]
    # Phase enum values, in ascending order. PLAN sits between DIAGNOSE and
    # IMPLEMENT: ranking by fixability has to happen before anything is
    # written, or the patch agent attacks the biggest number rather than the
    # fixable one.
    assert [s.phase for s in PIPELINE] == [
        Phase.ONBOARD, Phase.PROVISION,
        Phase.COMPREHEND, Phase.WORKLOAD, Phase.ORACLE, Phase.BASELINE,
        Phase.DIAGNOSE, Phase.PLAN, Phase.IMPLEMENT, Phase.VERIFY,
        Phase.MEASURE, Phase.GATE, Phase.SHIP,
    ]


def test_phases_execute_in_order(monkeypatch, runs: Path) -> None:
    order = stub_pipeline(
        monkeypatch, terminals={"ship": TerminalState.NO_SIGNIFICANT_IMPROVEMENT}
    )
    state = run_pipeline(run_id="r1", runs_dir=runs, endpoint="POST /x")

    assert order == [s.name for s in PIPELINE]
    assert state.terminal == TerminalState.NO_SIGNIFICANT_IMPROVEMENT
    assert state.completed_phases == [s.phase for s in PIPELINE]


def test_no_argument_reorders_or_skips_the_pipeline() -> None:
    """A model cannot steer control flow because control flow takes no input."""
    import inspect

    parameters = set(inspect.signature(run_pipeline).parameters)
    for forbidden in ("phases", "order", "skip", "only", "start_at", "until"):
        assert forbidden not in parameters


# ─────────────────────────────  resume / skip  ───────────────────────────


def test_a_phase_with_a_valid_artifact_is_skipped(monkeypatch, runs: Path) -> None:
    order = stub_pipeline(
        monkeypatch, terminals={"ship": TerminalState.NO_SIGNIFICANT_IMPROVEMENT}
    )
    run_dir = runs / "r2"
    run_dir.mkdir(parents=True)
    (run_dir / "oracle.json").write_text('{"already": true}')

    run_pipeline(run_id="r2", runs_dir=runs)

    assert "oracle" not in order
    assert "baseline" in order


def test_an_artifact_whose_input_hash_no_longer_matches_is_not_skipped(
    monkeypatch, runs: Path
) -> None:
    """Artifacts built on a broken corpus must not be silently reusable: a
    resume that only checks 'does the file exist and parse' would skip an
    oracle recorded against a corpus this run no longer has."""
    order = stub_pipeline(
        monkeypatch, terminals={"ship": TerminalState.NO_SIGNIFICANT_IMPROVEMENT}
    )
    run_dir = runs / "r-stale"
    run_dir.mkdir(parents=True)
    (run_dir / "corpus.json").write_text(json.dumps({"hash": "current-corpus"}))
    (run_dir / "oracle.json").write_text(
        json.dumps({"corpus_hash": "an-old-corpus-this-run-no-longer-has"}))

    run_pipeline(run_id="r-stale", runs_dir=runs)

    # Not skipped: the stub re-ran it and overwrote the stale content.
    assert "oracle" in order
    assert json.loads((run_dir / "oracle.json").read_text()) == {"phase": "oracle"}


def test_an_artifact_whose_input_hash_still_matches_is_skipped(
    monkeypatch, runs: Path
) -> None:
    """The matching case, so the mismatch test above is proof of a check and
    not a check that always fires."""
    order = stub_pipeline(
        monkeypatch, terminals={"ship": TerminalState.NO_SIGNIFICANT_IMPROVEMENT}
    )
    run_dir = runs / "r-fresh"
    run_dir.mkdir(parents=True)
    (run_dir / "corpus.json").write_text(json.dumps({"hash": "current-corpus"}))
    (run_dir / "oracle.json").write_text(
        json.dumps({"corpus_hash": "current-corpus"}))

    run_pipeline(run_id="r-fresh", runs_dir=runs)

    assert "oracle" not in order


def test_resume_restarts_after_the_last_completed_phase(
    monkeypatch, runs: Path
) -> None:
    """Killing a run and resuming must not redo earlier work."""
    first = stub_pipeline(
        monkeypatch,
        fails={"diagnose": ToolError("provider fell over")},
    )
    state = run_pipeline(run_id="r3", runs_dir=runs)
    assert first == ["onboard", "provision", "understand", "workload",
                     "oracle", "baseline", "diagnose"]
    assert state.terminal == TerminalState.INTERNAL_ERROR
    assert Phase.BASELINE in state.completed_phases
    assert Phase.DIAGNOSE not in state.completed_phases

    second: list[str] = []
    stub_pipeline(
        monkeypatch, order=second,
        terminals={"ship": TerminalState.NO_SIGNIFICANT_IMPROVEMENT},
    )
    resumed = run_pipeline(runs_dir=runs, resume="r3")

    # The four phases that already produced artifacts are not re-run.
    assert "understand" not in second and "baseline" not in second
    assert second[0] == "diagnose"
    assert resumed.terminal == TerminalState.NO_SIGNIFICANT_IMPROVEMENT


def test_a_completed_phase_whose_artifact_was_deleted_re_runs(
    monkeypatch, runs: Path
) -> None:
    """`completed_phases` is a record of what happened, not evidence the
    output still exists. Deleting an artifact after the phase that produced
    it ran must be enough to force a re-run on resume -- the old condition
    (`already or valid`) let `completed_phases` skip a phase forever
    regardless of whether its artifact was still on disk."""
    first = stub_pipeline(
        monkeypatch,
        fails={"diagnose": ToolError("provider fell over")},
    )
    state = run_pipeline(run_id="r-deleted", runs_dir=runs)
    assert "oracle" in first
    assert Phase.ORACLE in state.completed_phases

    run_dir = runs / "r-deleted"
    oracle_path = run_dir / "oracle.json"
    assert oracle_path.exists()
    oracle_path.unlink()

    second: list[str] = []
    stub_pipeline(
        monkeypatch, order=second,
        terminals={"ship": TerminalState.NO_SIGNIFICANT_IMPROVEMENT},
    )
    events: list[dict] = []
    resumed = run_pipeline(runs_dir=runs, resume="r-deleted",
                           on_event=events.append)

    # Re-run, not skipped -- and the artifact exists again afterwards.
    assert "oracle" in second
    assert oracle_path.exists()
    assert resumed.terminal == TerminalState.NO_SIGNIFICANT_IMPROVEMENT

    missing_notes = [e for e in events if e["kind"] == "artifact_missing"]
    assert any(e["phase"] == "oracle" for e in missing_notes), (
        "the operator must be told WHY a 'completed' phase re-ran"
    )


def test_resume_keeps_the_original_run_id_and_endpoint(
    monkeypatch, runs: Path
) -> None:
    stub_pipeline(monkeypatch, fails={"oracle": ToolError("nope")})
    run_pipeline(run_id="r4", runs_dir=runs, endpoint="POST /special")

    stub_pipeline(
        monkeypatch, terminals={"ship": TerminalState.NO_SIGNIFICANT_IMPROVEMENT}
    )
    resumed = run_pipeline(runs_dir=runs, resume="r4")
    assert resumed.run_id == "r4"
    assert resumed.endpoint == "POST /special"


def test_resuming_a_run_that_does_not_exist_is_refused(runs: Path) -> None:
    with pytest.raises(ContractViolation, match="nothing to resume"):
        run_pipeline(runs_dir=runs, resume="never-existed")


def test_run_state_is_persisted_after_every_phase(monkeypatch, runs: Path) -> None:
    seen: list[list[str]] = []

    original = orch.save_state

    def spy(state: RunState, run_dir: Path):
        seen.append([p.value for p in state.completed_phases])
        return original(state, run_dir)

    stub_pipeline(monkeypatch, terminals={"ship": TerminalState.PR_OPENED})
    monkeypatch.setattr(orch, "save_state", spy)
    run_pipeline(run_id="r5", runs_dir=runs)

    # Monotonically growing, and written more often than once.
    assert len(seen) > len(PIPELINE)
    assert seen[-1] == [s.phase.value for s in PIPELINE]
    on_disk = load_state(runs / "r5")
    assert on_disk.terminal == TerminalState.PR_OPENED


# ───────────────────────────  transition guards  ─────────────────────────


def context_for(tmp_path: Path) -> Context:
    return Context(
        run_id="g", run_dir=tmp_path, repo=None, endpoint="e",
        budget=Budget(), trace=Trace(tmp_path, "g"),
    )


def write_oracle(run_dir: Path, *, endpoint: str = "") -> Path:
    """A minimal, self-consistent oracle.json for THIS run_dir -- never the
    global default, which no longer exists."""
    from senapathy.contracts import BehaviourOracle, TolerancePolicy
    from senapathy.oracle.record import oracle_hash, write_oracle as _write

    policy = TolerancePolicy()
    computed = oracle_hash(corpus_hash="c", recorded=[], policy=policy,
                           base_commit="a" * 40, excluded_tables=[])
    oracle = BehaviourOracle.model_validate({
        "endpoint": endpoint, "corpus_hash": "c", "recorded": [],
        "policy": policy.model_dump(mode="json"), "base_commit": "a" * 40,
        "excluded_tables": [], "hash": computed,
    })
    return _write(oracle, {}, path=run_dir / "oracle.json")


def test_patch_is_refused_without_a_recorded_oracle(tmp_path: Path) -> None:
    """The guard that matters most: a patch written before behaviour is frozen
    has nothing to be checked against. No oracle.json in THIS run's directory
    -- not a monkeypatched global -- is what must trigger the refusal."""
    (tmp_path / "hypotheses.json").write_text("{}")
    reason = orch._preconditions(BY_NAME["patch"], context_for(tmp_path), None)
    assert reason is not None
    assert "without a recorded oracle" in reason
    assert "freeze the patch's own behaviour" in reason


def test_patch_is_refused_without_a_diagnosis(tmp_path: Path) -> None:
    write_oracle(tmp_path, endpoint="e")
    reason = orch._preconditions(BY_NAME["patch"], context_for(tmp_path), None)
    assert reason is not None and "without a diagnosis" in reason


def test_patch_is_refused_when_the_oracle_is_for_a_different_endpoint(
    tmp_path: Path
) -> None:
    """The real defect this phase guard exists to close: a recording of
    POST /api/customers/search guarding a patch for a different endpoint.
    Scoping the read to this run's own oracle.json is necessary but not
    sufficient on its own -- the file in THIS run's directory could still
    describe a different route, so the endpoint itself must be checked."""
    (tmp_path / "hypotheses.json").write_text("{}")
    write_oracle(tmp_path, endpoint="POST /api/customers/search")

    ctx = Context(run_id="g", run_dir=tmp_path, repo=None,
                 endpoint="GET /api/customers/{id}/invoices",
                 budget=Budget(), trace=Trace(tmp_path, "g"))

    reason = orch._preconditions(BY_NAME["patch"], ctx, None)

    assert reason is not None
    assert "POST /api/customers/search" in reason
    assert "GET /api/customers/{id}/invoices" in reason


def test_measure_is_refused_without_a_passing_verification(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(
        "senapathy.measure.arms.existing_worktree",
        lambda path, repo=None: type("W", (), {"source_diff": lambda self: "d"})(),
    )
    reason = orch._preconditions(BY_NAME["measure"], context_for(tmp_path), None)
    assert reason is not None
    assert "cannot enter the measure phase" in reason


@pytest.mark.parametrize("phase_name", ["gate", "ship"])
def test_gate_and_ship_need_the_upstream_artifacts(
    tmp_path: Path, phase_name: str
) -> None:
    reason = orch._preconditions(BY_NAME[phase_name], context_for(tmp_path), None)
    assert reason is not None
    assert "has not been produced" in reason


def test_a_blocked_transition_stops_the_run_cleanly(monkeypatch, runs: Path) -> None:
    order = stub_pipeline(monkeypatch)
    monkeypatch.setattr(
        orch, "_preconditions",
        lambda spec, ctx, state: "no oracle" if spec.name == "patch" else None,
    )
    state = run_pipeline(run_id="r6", runs_dir=runs)

    assert "patch" not in order
    assert state.terminal == TerminalState.INTERNAL_ERROR
    # Everything before the block is still recorded, not thrown away.
    assert Phase.DIAGNOSE in state.completed_phases


def test_a_phase_with_no_runner_and_no_artifact_fails_honestly(
    monkeypatch, runs: Path
) -> None:
    """understand/workload have no runner in this build. If their artifact is
    absent the run must stop and say so, not silently continue."""
    order: list[str] = []
    stub_pipeline(monkeypatch, order=order)
    patched = tuple(
        orch.PhaseSpec(
            phase=s.phase, name=s.name, artifact=s.artifact,
            runner=None if s.name == "oracle" else s.runner,
            validate=s.validate, uses_llm=s.uses_llm, describe=s.describe,
        )
        for s in orch.PIPELINE
    )
    monkeypatch.setattr(orch, "PIPELINE", patched)

    state = run_pipeline(run_id="norunner", runs_dir=runs)

    assert state.terminal == TerminalState.INTERNAL_ERROR
    assert "oracle" not in order
    trace_events = Trace(runs / "norunner", "norunner").read_all()
    blocked = [e for e in trace_events if e.get("type") == "guardrail_triggered"]
    assert blocked and "no runner in this build" in blocked[0]["reason"]


# ──────────────────────────────  the budget  ─────────────────────────────


class FakeUsage:
    def __init__(self, i: int, o: int) -> None:
        self.input_tokens, self.output_tokens = i, o


class FakeResponse:
    def __init__(self, i: int, o: int) -> None:
        self.usage = FakeUsage(i, o)


class FakeClient:
    def __init__(self) -> None:
        self.calls = 0
        self.last_raw_message = {"content": "x"}

    def complete(self, **kwargs):
        self.calls += 1
        return FakeResponse(1000, 500)


def test_the_metered_client_charges_every_call() -> None:
    budget = Budget(max_usd=1.0, usd_per_million_input=1.0, usd_per_million_output=2.0)
    client = MeteredClient(FakeClient(), budget, "diagnose")

    client.complete(messages=[])
    client.complete(messages=[])

    assert budget.tokens_in == 2000 and budget.tokens_out == 1000
    assert budget.usd == pytest.approx(2 * (0.001 + 0.001))
    assert budget.by_phase["diagnose"]["calls"] == 2


def test_the_metered_client_delegates_everything_else() -> None:
    """run_agent reads last_raw_message off the client between calls."""
    inner = FakeClient()
    client = MeteredClient(inner, Budget(), "patch")
    assert client.last_raw_message == {"content": "x"}


def test_exceeding_the_spend_ceiling_raises_budget_exhausted() -> None:
    budget = Budget(max_usd=0.001, usd_per_million_input=1000.0)
    client = MeteredClient(FakeClient(), budget, "diagnose")
    with pytest.raises(BudgetExhausted, match="spend reached"):
        client.complete(messages=[])


def test_exceeding_the_wall_clock_is_detected() -> None:
    budget = Budget(max_wall_s=0.0)
    budget.started_at = time.monotonic() - 10
    assert "wall clock reached" in (budget.check() or "")


def test_a_budget_overrun_is_a_terminal_state_not_a_crash(
    monkeypatch, runs: Path
) -> None:
    """A crash at minute fifty-nine throws away fifty-nine minutes of evidence."""
    stub_pipeline(
        monkeypatch, fails={"diagnose": BudgetExhausted("spend reached the cap")}
    )
    state = run_pipeline(run_id="r8", runs_dir=runs)

    assert state.terminal == TerminalState.BUDGET_EXHAUSTED
    # Partial findings survive.
    assert Phase.BASELINE in state.completed_phases
    assert (runs / "r8" / "baseline.json").exists()
    assert (runs / "r8" / "cost.json").exists()


def test_the_budget_is_checked_before_a_phase_runs(monkeypatch, runs: Path) -> None:
    order = stub_pipeline(monkeypatch)

    real_check = Budget.check
    calls = {"n": 0}

    def check(self):
        calls["n"] += 1
        return "spend reached the cap" if calls["n"] > 2 else None

    monkeypatch.setattr(Budget, "check", check)
    state = run_pipeline(run_id="r9", runs_dir=runs)

    assert state.terminal == TerminalState.BUDGET_EXHAUSTED
    assert len(order) < len(PIPELINE)


def test_cost_json_records_tokens_and_the_price_it_used(
    monkeypatch, runs: Path
) -> None:
    stub_pipeline(monkeypatch, terminals={"ship": TerminalState.PR_OPENED})
    run_pipeline(run_id="r10", runs_dir=runs)
    cost = json.loads((runs / "r10" / "cost.json").read_text())
    # Tokens are measured; the rate is configuration and is recorded alongside
    # so a dollar figure can never be read without knowing where it came from.
    assert "tokens_in" in cost and "usd_per_million_input" in cost


# ──────────────────────  early terminal states  ──────────────────────────


@pytest.mark.parametrize(
    "phase_name,terminal",
    [
        ("diagnose", TerminalState.NO_BOTTLENECK_FOUND),
        ("patch", TerminalState.BOTTLENECK_NOT_FIXABLE),
        ("verify", TerminalState.BEHAVIOUR_NOT_PRESERVED),
    ],
)
def test_any_phase_may_terminate_the_run_early(
    monkeypatch, runs: Path, phase_name: str, terminal: TerminalState
) -> None:
    order = stub_pipeline(monkeypatch, terminals={phase_name: terminal})
    state = run_pipeline(run_id=f"early-{phase_name}", runs_dir=runs)

    assert state.terminal == terminal
    assert order[-1] == phase_name
    later = [s.name for s in PIPELINE][
        [s.name for s in PIPELINE].index(phase_name) + 1:
    ]
    assert not set(order) & set(later), "phases after the terminal one still ran"


def test_a_failing_environment_phase_reports_environment_infeasible(
    monkeypatch, runs: Path
) -> None:
    stub_pipeline(monkeypatch, fails={"oracle": ToolError("app never came up")})
    state = run_pipeline(run_id="r11", runs_dir=runs)
    assert state.terminal == TerminalState.ENVIRONMENT_INFEASIBLE


def test_every_terminal_state_is_reachable_from_the_orchestrator() -> None:
    """Every state the contract defines has a code path that produces it."""
    source = Path(orch.__file__).read_text()
    ship_source = (Path(orch.__file__).parent / "ship.py").read_text()
    gate_source = (Path(orch.__file__).parent / "gate" / "rules.py").read_text()
    combined = source + ship_source + gate_source
    for state in TerminalState:
        assert state.value in combined, f"{state.value} is unreachable"


# ───────────────────────────────  the trace  ─────────────────────────────


def test_the_trace_renders_a_narrative_and_a_cost_table(
    monkeypatch, runs: Path
) -> None:
    from senapathy.trace_view import render_trace

    stub_pipeline(
        monkeypatch, terminals={"ship": TerminalState.NO_SIGNIFICANT_IMPROVEMENT}
    )
    run_pipeline(run_id="r12", runs_dir=runs, endpoint="POST /api/customers/search")

    rendered = render_trace("r12", runs)
    assert "Agent started · POST /api/customers/search" in rendered
    assert "NO_SIGNIFICANT_IMPROVEMENT" in rendered
    assert "COST" in rendered
    assert "↓" in rendered


def test_the_trace_marks_skipped_phases(monkeypatch, runs: Path) -> None:
    from senapathy.trace_view import render_trace

    stub_pipeline(monkeypatch, fails={"diagnose": ToolError("stop")})
    run_pipeline(run_id="r13", runs_dir=runs)
    stub_pipeline(
        monkeypatch, terminals={"ship": TerminalState.NO_SIGNIFICANT_IMPROVEMENT}
    )
    run_pipeline(runs_dir=runs, resume="r13")

    rendered = render_trace("r13", runs)
    # A reader of a resumed run needs to see which work was actually redone.
    assert "skipped, already done" in rendered


def test_the_trace_says_when_dollars_are_not_priced(monkeypatch, runs: Path) -> None:
    from senapathy.trace_view import render_trace

    stub_pipeline(monkeypatch, terminals={"ship": TerminalState.PR_OPENED})
    run_pipeline(run_id="r14", runs_dir=runs)
    rendered = render_trace("r14", runs)
    assert "no token price is configured" in rendered
    assert "Tokens are measured; dollars are not" in rendered


def test_trace_of_a_missing_run_is_refused(runs: Path) -> None:
    from senapathy.trace_view import render_trace

    runs.mkdir(parents=True)
    with pytest.raises(ContractViolation):
        render_trace("nope", runs)
