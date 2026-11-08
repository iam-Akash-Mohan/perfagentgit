"""F4: the retry loops, the isolation guard, the paired p-value, the cost.

No network, no JVM, no LLM. What is under test is the ORDER a failed phase
produces and the terminal state it lands on — not the phases themselves,
which have their own tests.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from senapathy import orchestrator as orch
from senapathy.contracts import (GateName, LatencyStats, PerfReport, Phase,
                                 RunState, TerminalState)
from senapathy.measure.isolation import (isolation_problem, resolve_datasource,
                                         touches_schema)
from senapathy.orchestrator import Budget, Context, PhaseOutcome
from senapathy.trace import Trace


# ─────────────────────────  the shared-schema guard  ─────────────────────


def repo_with_url(path: Path, url: str) -> Path:
    resources = path / "src/main/resources"
    resources.mkdir(parents=True, exist_ok=True)
    (resources / "application-perfagent.yml").write_text(
        f"spring:\n  datasource:\n    url: {url}\n", encoding="utf-8")
    return path


MINI = ("jdbc:mysql://localhost:3307/mini?useSSL=false&"
        "allowPublicKeyRetrieval=true")


def test_a_jdbc_url_resolves_to_a_comparable_identity(tmp_path: Path) -> None:
    source = resolve_datasource(repo_with_url(tmp_path / "blue", MINI))
    assert source.identity == ("mysql", "localhost", "3307", "mini")
    assert source.render() == "mysql://localhost:3307/mini"


@pytest.mark.parametrize("kwargs", [
    {"approach": "add_index"},
    {"diff": "+++ b/src/main/resources/db/migration/V2__idx.sql"},
    {"files_changed": ["src/main/resources/db/migration/V2__idx.sql"]},
])
def test_every_route_to_a_schema_change_is_recognised(kwargs) -> None:
    assert touches_schema(**kwargs) is True


def test_a_java_only_patch_is_not_a_schema_change() -> None:
    assert touches_schema(
        approach="fetch_join",
        diff="+++ b/src/main/java/com/ex/service/CustomerService.java",
        files_changed=["src/main/java/com/ex/service/CustomerService.java"],
    ) is False


def test_an_index_on_two_arms_sharing_one_schema_is_refused(tmp_path: Path) -> None:
    """The false negative this exists to prevent: both arms get the index, the
    A/B reports no improvement, and nothing looks broken."""
    blue = repo_with_url(tmp_path / "blue", MINI)
    green = repo_with_url(tmp_path / "green", MINI)

    problem = isolation_problem(blue, green, approach="add_index")
    assert problem is not None
    assert "SAME schema: mysql://localhost:3307/mini" in problem
    assert "false negative" in problem
    assert "was NOT run" in problem
    # It must not read as a verdict on the patch.
    assert "does not help" in problem and "is a statement that this harness" in problem


def test_isolated_arms_are_allowed_to_measure_an_index(tmp_path: Path) -> None:
    blue = repo_with_url(tmp_path / "blue", MINI)
    green = repo_with_url(tmp_path / "green",
                          MINI.replace("/mini?", "/mini_green?"))
    assert isolation_problem(blue, green, approach="add_index") is None


def test_a_java_patch_is_never_blocked_by_the_schema_guard(tmp_path: Path) -> None:
    blue = repo_with_url(tmp_path / "blue", MINI)
    green = repo_with_url(tmp_path / "green", MINI)
    assert isolation_problem(blue, green, approach="fetch_join",
                             diff="+++ b/src/main/java/A.java") is None


def test_unknown_isolation_is_not_isolation(tmp_path: Path) -> None:
    """An arm whose config names no JDBC URL is an unknown, and an unknown
    must not be read as a pass."""
    blue = (tmp_path / "blue"); blue.mkdir()
    green = (tmp_path / "green"); green.mkdir()
    problem = isolation_problem(blue, green, approach="add_index")
    assert problem is not None and "unknown is not an isolation" in problem


def test_the_guard_terminates_with_its_own_state(monkeypatch, tmp_path: Path) -> None:
    """Not NO_SIGNIFICANT_IMPROVEMENT, and not INTERNAL_ERROR."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    ctx = Context(run_id="r1", run_dir=run_dir, repo=tmp_path,
                  endpoint="POST /x", budget=Budget(), trace=Trace(run_dir, "r1"))

    monkeypatch.setattr(orch, "_isolation_problem",
                        lambda c: "this patch changes the database schema…")
    monkeypatch.setattr(orch, "_valid_patch", lambda c, p: None)
    for name in ("verification.json", "perf_report.json", "patch.json"):
        (run_dir / name).write_text("{}")
    monkeypatch.setattr("senapathy.measure.verify.require_verified",
                        lambda *a, **k: None)
    monkeypatch.setattr("senapathy.measure.arms.existing_worktree",
                        lambda *a, **k: type("W", (), {
                            "source_diff": lambda self: "",
                            "path": tmp_path})())

    state = RunState(run_id="r1", repo=str(tmp_path), endpoint="POST /x",
                     phase=Phase.VERIFY)
    blocked = orch._preconditions(orch.BY_NAME["measure"], ctx, state)
    assert blocked is not None and blocked.startswith("ISOLATION:")


# ────────────────────────────  the retry loops  ──────────────────────────


def plan_on_disk(run_dir: Path, candidates: int = 2) -> None:
    """A plan with `candidates` selectable candidates, newest-first by value."""
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "plan.json").write_text(json.dumps({
        "endpoint": "POST /x", "measured_total_ms": 1000.0, "ranked": [],
        "candidates": [
            {"id": f"c{n}", "targets_hypothesis": "h2", "approach": "fetch_join",
             "description": f"candidate {n}", "mechanism_changed": "queries",
             "predicted_effect": "fewer", "files_to_touch": ["src/main/java/A.java"],
             "what_could_break": [], "preserves_response_bytes": True,
             "risk_tier": 1, "expected_value_ms": 100.0 - n}
            for n in range(candidates)
        ],
        "selected_id": "c0", "selection_reason": "highest expected value",
        "not_selected": [], "decided_by": "code",
    }))


def run_with(monkeypatch, runs: Path, *, terminals=None, gate_failed=None,
             run_id="r", candidates: int = 2
             ) -> tuple[list[str], RunState, list[dict]]:
    from tests.test_orchestrator import stub_pipeline

    order = stub_pipeline(monkeypatch, terminals=terminals or {})
    if gate_failed is not None:
        patched = []
        for spec in orch.PIPELINE:
            if spec.phase is Phase.GATE:
                def gate_runner(ctx, _spec=spec):
                    path = _spec.artifact(ctx)
                    path.write_text(json.dumps({"phase": "gate"}))
                    return PhaseOutcome(artifact=path, summary="gate FAILED",
                                        detail={"failed": gate_failed})
                spec = orch.PhaseSpec(
                    phase=spec.phase, name=spec.name, artifact=spec.artifact,
                    runner=gate_runner, validate=spec.validate,
                    uses_llm=spec.uses_llm, describe=spec.describe)
            patched.append(spec)
        monkeypatch.setattr(orch, "PIPELINE", tuple(patched))
        monkeypatch.setattr(orch, "BY_NAME", {s.name: s for s in patched})

    plan_on_disk(runs / run_id, candidates=candidates)
    events: list[dict] = []
    state = orch.run_pipeline(run_id=run_id, runs_dir=runs,
                              on_event=events.append)
    return order, state, events


def test_a_rejected_patch_is_retried_once_with_the_oracle_diff(
    monkeypatch, tmp_path: Path
) -> None:
    runs = tmp_path / "runs"
    order, state, events = run_with(
        monkeypatch, runs, candidates=1,
        terminals={"verify": TerminalState.BEHAVIOUR_NOT_PRESERVED})

    assert order.count("patch") == 2, "the patch phase ran again"
    assert order.count("verify") == 2
    retries = [e for e in events if e["kind"] == "retry"]
    assert len(retries) == 1, "one candidate: one retry, then stop"
    assert "retrying with the recorded differences" in retries[0]["reason"]


def test_two_rejections_move_to_the_next_candidate(
    monkeypatch, tmp_path: Path
) -> None:
    runs = tmp_path / "runs"
    order, state, events = run_with(
        monkeypatch, runs,
        terminals={"verify": TerminalState.BEHAVIOUR_NOT_PRESERVED})

    reasons = [e["reason"] for e in events if e["kind"] == "retry"]
    # Attempt 2 on candidate 1, then candidate 2, then attempt 2 on it.
    assert any("moving to candidate 2" in r for r in reasons)
    assert state.terminal is TerminalState.BEHAVIOUR_NOT_PRESERVED
    assert order.count("patch") == 4, "two attempts on each of two candidates"


def test_exhausted_candidates_terminate_honestly(
    monkeypatch, tmp_path: Path
) -> None:
    runs = tmp_path / "runs"
    order, state, events = run_with(
        monkeypatch, runs, run_id="one", candidates=1,
        terminals={"verify": TerminalState.BEHAVIOUR_NOT_PRESERVED})
    assert state.terminal is TerminalState.BEHAVIOUR_NOT_PRESERVED
    assert order.count("patch") == 2, "two attempts, one candidate, then stop"


def test_a_gate_failing_only_on_improvement_tries_the_next_candidate(
    monkeypatch, tmp_path: Path
) -> None:
    runs = tmp_path / "runs"
    order, state, events = run_with(
        monkeypatch, runs,
        gate_failed=[GateName.IMPROVEMENT_THRESHOLD.value])

    reasons = [e["reason"] for e in events if e["kind"] == "retry"]
    assert any("did not clear the improvement threshold" in r for r in reasons)
    assert order.count("patch") == 2


def test_a_gate_failing_on_behaviour_does_not_retry(
    monkeypatch, tmp_path: Path
) -> None:
    """A behaviour failure, a mechanism that did not move, or a forbidden edit
    are not fixed by trying a different fix for the same bottleneck."""
    runs = tmp_path / "runs"
    order, state, events = run_with(
        monkeypatch, runs,
        gate_failed=[GateName.IMPROVEMENT_THRESHOLD.value,
                     GateName.MECHANISM_MATCH.value])

    assert [e for e in events if e["kind"] == "retry"] == []
    assert order.count("patch") == 1


# ───────────────────────────  the paired p-value  ────────────────────────


def stats(p50: float) -> LatencyStats:
    return LatencyStats(n=20, p50_ms=p50, p95_ms=p50 * 1.1, p99_ms=p50 * 1.2,
                        mean_ms=p50, stdev_ms=1.0)


def test_the_gate_reads_the_paired_p_value_off_the_artifact() -> None:
    from senapathy.gate.rules import evaluate_gates
    from tests.test_gate import patch, plan, verdict

    report = PerfReport(
        before=stats(1000), after=stats(600), p50_delta_pct=-40.0,
        p95_delta_pct=-40.0, sql_count_before=6, sql_count_after=2,
        rows_examined_before=100, rows_examined_after=10,
        p_value=0.36,                 # unpaired: sees nothing
        p_value_paired=0.0015,        # paired: sees the arm effect
        mechanism_confirmed=True, mechanism_note="queries dropped",
    )
    decision = evaluate_gates(verdict(), report, patch(), plan())
    significance = next(r for r in decision.results
                        if r.name is GateName.STATISTICAL_SIGNIFICANCE)
    assert significance.passed, "the field must be read without a kwarg"
    assert "0.001500" in (significance.actual or "")


def test_the_artifact_wins_over_the_loose_keyword() -> None:
    from senapathy.gate.rules import evaluate_gates
    from tests.test_gate import patch, plan, verdict

    report = PerfReport(
        before=stats(1000), after=stats(600), p50_delta_pct=-40.0,
        p95_delta_pct=-40.0, sql_count_before=6, sql_count_after=2,
        rows_examined_before=100, rows_examined_after=10,
        p_value=0.36, p_value_paired=0.9,      # the artifact says: not significant
        mechanism_confirmed=True, mechanism_note="",
    )
    decision = evaluate_gates(verdict(), report, patch(), plan(),
                              paired_p_value=0.0001)
    significance = next(r for r in decision.results
                        if r.name is GateName.STATISTICAL_SIGNIFICANCE)
    assert not significance.passed, "a kwarg may not overrule the artifact"


# ──────────────────────────────  real costs  ─────────────────────────────


def test_a_run_is_priced_from_the_model_that_served_it() -> None:
    budget = Budget()
    budget.charge("plan", 1_000_000, 1_000_000, "llama-3.3-70b-versatile")
    assert budget.usd == pytest.approx(0.59 + 0.79)
    assert budget.by_phase["plan"]["usd"] == pytest.approx(0.59 + 0.79)


def test_an_unknown_model_prices_at_zero_but_keeps_the_tokens() -> None:
    budget = Budget()
    budget.charge("plan", 1_000, 1_000, "a-model-nobody-configured")
    assert budget.usd == 0.0
    assert budget.tokens_in == 1_000 and budget.tokens_out == 1_000
