"""Step ceilings: raised defaults, reachable from `run`, named in the stop
message.

Steps are a crude backstop — --max-usd and --max-wall are the real brakes —
so these tests only check plumbing: does a --{phase}-max-steps flag actually
reach that phase's LoopBudget, and does a budget_exhausted message name the
flag that raises it. They do not re-test run_agent's budget mechanics
(test_agent_loop.py already does) or the phases' own logic.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from senapathy import orchestrator as orch
from senapathy.agent import LoopBudget
from senapathy.contracts.repo import DatabaseKind, DatabaseProfile
from senapathy.orchestrator import Budget, Context
from senapathy.trace import Trace


def ctx_for(tmp_path: Path, **options) -> Context:
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    return Context(
        run_id="r1", run_dir=run_dir, repo=tmp_path, endpoint="GET /x",
        budget=Budget(max_usd=5.0, max_wall_s=100.0),
        trace=Trace(run_dir, "r1"), options=options,
    )


class FakeProfile:
    """Enough of a RepoProfile for `_mysql_target`/`profile=` plumbing. A
    real MySQL database, not NONE: diagnose's `run_probe` needs an actual
    target to build (it is not an optional lookup tool the way `get_schema`
    is), so these tests give every phase one rather than special-casing it."""
    database = DatabaseProfile(
        kind=DatabaseKind.MYSQL, host="127.0.0.1", port=3307, database="mini",
    )


@pytest.fixture(autouse=True)
def no_real_llm(monkeypatch):
    """Every phase runner builds a client via _metered_llm; none of these
    tests make a real call, so the client itself is irrelevant."""
    monkeypatch.setattr(orch, "_metered_llm", lambda ctx, phase: "client")
    # diagnose and patch also read ctx.profile() off disk; what it contains
    # is irrelevant here since the phase function itself is stubbed out.
    monkeypatch.setattr(Context, "profile", lambda self: FakeProfile())
    monkeypatch.setenv("SENAPATHY_MYSQL_ADMIN_USER", "admin")
    monkeypatch.setenv("SENAPATHY_MYSQL_ADMIN_PASSWORD", "adminpw")


class FakeComprehensionModel:
    route = "GET /x"
    call_chain = []
    data_access = []
    tables = []
    prompt_hash = "h"

    def per_row_accesses(self):
        return []


def test_understand_flag_reaches_its_loop_budget(monkeypatch, tmp_path):
    import senapathy.understand.agent as understand_mod

    captured = {}

    def fake_understand(**kwargs):
        captured.update(kwargs)
        return FakeComprehensionModel()

    monkeypatch.setattr(understand_mod, "understand", fake_understand)

    ctx = ctx_for(tmp_path, understand_max_steps=7)
    orch._run_understand(ctx)

    budget = captured["budget"]
    assert isinstance(budget, LoopBudget)
    assert budget.max_steps == 7
    assert budget.step_flag == "--understand-max-steps"
    assert budget.max_usd == 5.0
    assert budget.max_wall_s is not None


def test_understand_default_is_120_when_run_sets_no_flag(monkeypatch, tmp_path):
    import senapathy.understand.agent as understand_mod

    captured = {}
    monkeypatch.setattr(understand_mod, "understand",
                        lambda **k: (captured.update(k), FakeComprehensionModel())[1])

    orch._run_understand(ctx_for(tmp_path))
    assert captured["budget"].max_steps == 120


class FakeHypothesisSet:
    hypotheses = [object()]
    ranked_ids = ["h1"]


def test_diagnose_flag_reaches_its_loop_budget(monkeypatch, tmp_path):
    import senapathy.diagnose.agent as diagnose_mod
    import senapathy.measure.arms as arms_mod

    class FakeArm:
        base_url = "http://blue"

    class FakeArmCtx:
        def __enter__(self):
            return FakeArm()

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(arms_mod, "single_arm", lambda *a, **k: FakeArmCtx())

    captured = {}

    def fake_run_diagnosis(**kwargs):
        captured.update(kwargs)
        return FakeHypothesisSet(), [object()], Path("hypotheses.json")

    monkeypatch.setattr(diagnose_mod, "run_diagnosis", fake_run_diagnosis)

    ctx = ctx_for(tmp_path, diagnose_max_steps=9)
    orch._run_diagnose(ctx)

    budget = captured["budget"]
    assert isinstance(budget, LoopBudget)
    assert budget.max_steps == 9
    assert budget.step_flag == "--diagnose-max-steps"
    assert budget.max_usd == 5.0


def test_diagnose_default_is_80_when_run_sets_no_flag(monkeypatch, tmp_path):
    import senapathy.diagnose.agent as diagnose_mod
    import senapathy.measure.arms as arms_mod

    class FakeArmCtx:
        def __enter__(self):
            return type("Arm", (), {"base_url": "http://blue"})()

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(arms_mod, "single_arm", lambda *a, **k: FakeArmCtx())

    captured = {}

    def fake_run_diagnosis(**kwargs):
        captured.update(kwargs)
        return FakeHypothesisSet(), [], Path("hypotheses.json")

    monkeypatch.setattr(diagnose_mod, "run_diagnosis", fake_run_diagnosis)

    orch._run_diagnose(ctx_for(tmp_path))
    assert captured["budget"].max_steps == 80


class FakePlanResult:
    candidates = [object()]
    selection_reason = "no candidate scored above zero"

    def selected(self):
        return None


def test_plan_flag_reaches_its_loop_budget(monkeypatch, tmp_path):
    import senapathy.plan.agent as plan_mod

    captured = {}

    def fake_plan(**kwargs):
        captured.update(kwargs)
        return FakePlanResult()

    monkeypatch.setattr(plan_mod, "plan", fake_plan)
    monkeypatch.setattr(
        plan_mod, "load_inputs",
        lambda run_dir: (FakeHypothesisSet(), FakeComprehensionModel()),
    )

    ctx = ctx_for(tmp_path, plan_max_steps=11)
    orch._run_plan(ctx)

    budget = captured["budget"]
    assert isinstance(budget, LoopBudget)
    assert budget.max_steps == 11
    assert budget.step_flag == "--plan-max-steps"
    assert budget.max_usd == 5.0


def test_plan_default_is_60_when_run_sets_no_flag(monkeypatch, tmp_path):
    import senapathy.plan.agent as plan_mod

    captured = {}
    monkeypatch.setattr(plan_mod, "plan",
                        lambda **k: (captured.update(k), FakePlanResult())[1])
    monkeypatch.setattr(
        plan_mod, "load_inputs",
        lambda run_dir: (FakeHypothesisSet(), FakeComprehensionModel()),
    )

    orch._run_plan(ctx_for(tmp_path))
    assert captured["budget"].max_steps == 60


def test_patch_flag_reaches_its_loop_budget(monkeypatch, tmp_path):
    import senapathy.patch.agent as patch_mod

    captured = {}

    def fake_run_patch(*args, **kwargs):
        captured.update(kwargs)
        return {
            "outcome": "ABANDONED", "iterations_used": 1, "last_error": "x",
            "patch": {"files_changed": [], "lines_added": 0, "lines_removed": 0,
                      "build_iterations": 0},
        }

    monkeypatch.setattr(patch_mod, "run_patch", fake_run_patch)

    ctx = ctx_for(tmp_path, patch_max_steps=13)
    orch._run_patch(ctx)

    budget = captured["budget"]
    assert isinstance(budget, LoopBudget)
    assert budget.max_steps == 13
    assert budget.step_flag == "--patch-max-steps"
    assert budget.max_usd == 5.0


def test_patch_default_is_80_when_run_sets_no_flag(monkeypatch, tmp_path):
    import senapathy.patch.agent as patch_mod

    captured = {}

    def fake_run_patch(*args, **kwargs):
        captured.update(kwargs)
        return {
            "outcome": "ABANDONED", "iterations_used": 1, "last_error": "x",
            "patch": {"files_changed": [], "lines_added": 0, "lines_removed": 0,
                      "build_iterations": 0},
        }

    monkeypatch.setattr(patch_mod, "run_patch", fake_run_patch)

    orch._run_patch(ctx_for(tmp_path))
    assert captured["budget"].max_steps == 80


# ──────────────────  the exhaustion message names the flag  ──────────────


@pytest.mark.parametrize("flag", [
    "--understand-max-steps", "--diagnose-max-steps",
    "--plan-max-steps", "--patch-max-steps", "--max-steps",
])
def test_budget_exhausted_message_names_the_flag(flag):
    budget = LoopBudget(max_steps=5, step_flag=flag)
    reason = budget.exceeded(steps=5, usd=0.0)
    assert reason is not None
    assert f"raise it with {flag}" in reason


def test_a_budget_with_no_flag_still_reports_a_plain_reason():
    budget = LoopBudget(max_steps=5)
    reason = budget.exceeded(steps=5, usd=0.0)
    assert reason == "reached the step ceiling of 5"


# ────────────  `run` exposes all four flags at the right defaults  ────────


def test_run_command_exposes_all_four_step_flags_with_raised_defaults():
    import inspect

    from senapathy.cli import run

    sig = inspect.signature(run)
    expected = {
        "understand_max_steps": ("--understand-max-steps", 120),
        "diagnose_max_steps": ("--diagnose-max-steps", 80),
        "plan_max_steps": ("--plan-max-steps", 60),
        "patch_max_steps": ("--patch-max-steps", 80),
    }
    for name, (flag, default) in expected.items():
        option = sig.parameters[name].default
        assert option.param_decls == (flag,)
        assert option.default == default


# ───────────────────────  module-level ceilings  ──────────────────────────


def test_the_four_module_level_ceilings_were_raised():
    from senapathy.diagnose.agent import MAX_STEPS as diagnose_max
    from senapathy.patch.agent import MAX_STEPS as patch_max

    assert diagnose_max == 80
    assert patch_max == 80


def test_the_patch_build_test_iteration_limit_is_unchanged():
    """patch/tools.py:MAX_ITERATIONS bounds build/test CYCLES, not LLM steps —
    explicitly out of scope for this change."""
    from senapathy.patch.tools import MAX_ITERATIONS

    assert MAX_ITERATIONS == 8
