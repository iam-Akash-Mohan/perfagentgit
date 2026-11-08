"""The orchestrator's half of the endpoint flag.

`senapathy run --endpoint "..."` accepted the value, the orchestrator carried
it, and the comprehension phase never received it — it only checked whether an
artifact file existed and skipped. These tests are the guard on that.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

import senapathy.orchestrator as orch
from senapathy.contracts import Phase
from senapathy.orchestrator import Budget, Context, _valid_endpoint_model
from senapathy.trace import Trace

MODEL = {
    "route": "POST /api/customers/search",
    "entry": {"file": "C.java", "line": 18},
    "path": [{"file": "C.java", "line": 18}],
    "call_chain": [{"method": "C.search", "file": "C.java", "line": 18,
                    "does": "delegates"}],
    "data_access": [{"trigger": "repo.find", "kind": "jpql_query",
                     "file": "R.java", "line": 9, "executes_per_row": False}],
}


def context(tmp_path: Path, endpoint: str) -> Context:
    return Context(run_id="r1", run_dir=tmp_path, repo=None, endpoint=endpoint,
                   budget=Budget(), trace=Trace(tmp_path, "r1"))


def artifact(tmp_path: Path, payload: dict[str, Any]) -> Path:
    path = tmp_path / "endpoint_model.json"
    path.write_text(json.dumps(payload))
    return path


def test_a_model_for_the_requested_endpoint_satisfies_the_phase(
    tmp_path: Path
) -> None:
    ctx = context(tmp_path, "POST /api/customers/search")
    assert _valid_endpoint_model(ctx, artifact(tmp_path, MODEL)) is None


def test_a_cached_model_for_a_different_route_does_not(tmp_path: Path) -> None:
    ctx = context(tmp_path, "GET /api/customers/{id}")
    problem = _valid_endpoint_model(ctx, artifact(tmp_path, MODEL))
    assert problem is not None
    assert "POST /api/customers/search" in problem
    assert "GET /api/customers/{id}" in problem


def test_route_matching_tolerates_spelling_but_not_meaning(
    tmp_path: Path
) -> None:
    for spelling in ("POST /api/customers/search",
                     "post /api/customers/search",
                     "POST  /api/customers/search/"):
        ctx = context(tmp_path, spelling)
        assert _valid_endpoint_model(ctx, artifact(tmp_path, MODEL)) is None

    for different in ("GET /api/customers/search", "POST /api/customers"):
        ctx = context(tmp_path, different)
        assert _valid_endpoint_model(ctx, artifact(tmp_path, MODEL)) is not None


def test_a_stub_that_merely_parses_is_still_rejected(tmp_path: Path) -> None:
    """`{"phase": "understand"}` is valid JSON, and a destroyed artifact once
    passed validation because of it."""
    ctx = context(tmp_path, "POST /api/customers/search")
    problem = _valid_endpoint_model(ctx, artifact(tmp_path,
                                                  {"phase": "understand"}))
    assert problem is not None and "describes nothing" in problem


def test_the_comprehension_phase_has_a_runner_and_uses_the_llm() -> None:
    spec = orch.BY_PHASE[Phase.COMPREHEND]
    assert spec.runner is not None, "the phase must not be a no-op any more"
    assert spec.uses_llm is True


def test_the_artifact_lives_in_the_run_directory_not_a_global_path(
    tmp_path: Path
) -> None:
    """A single global endpoint_model.json is how a model for one route came
    to satisfy a run about another."""
    ctx = context(tmp_path, "POST /x")
    resolved = orch.BY_PHASE[Phase.COMPREHEND].artifact(ctx)
    assert resolved == tmp_path / "endpoint_model.json"


def test_the_runner_passes_the_run_s_endpoint_to_the_phase(
    monkeypatch, tmp_path: Path
) -> None:
    seen: dict[str, Any] = {}

    def fake_understand(**kwargs: Any):
        seen.update(kwargs)

        class Model:
            route = kwargs["endpoint"]
            call_chain: list = []
            data_access: list = []
            tables: list = []
            prompt_hash = "p"

            def per_row_accesses(self):
                return []

        (kwargs["run_dir"] / "endpoint_model.json").write_text(
            json.dumps({**MODEL, "route": kwargs["endpoint"]}))
        return Model()

    import senapathy.understand.agent as understand_mod

    monkeypatch.setattr(understand_mod, "understand", fake_understand)
    monkeypatch.setattr(orch, "_metered_llm", lambda ctx, phase: object())
    monkeypatch.setattr(orch, "_mysql_target", lambda ctx, phase: None)

    ctx = context(tmp_path, "GET /api/customers/{id}")
    ctx.repo = tmp_path
    outcome = orch._run_understand(ctx)

    assert seen["endpoint"] == "GET /api/customers/{id}"
    assert seen["repo"] == tmp_path
    assert outcome.detail["route"] == "GET /api/customers/{id}"


def test_the_phase_refuses_without_a_repository(tmp_path: Path) -> None:
    from senapathy.errors import ContractViolation

    ctx = context(tmp_path, "POST /x")
    with pytest.raises(ContractViolation, match="target repository"):
        orch._run_understand(ctx)


def test_diagnose_reads_this_run_s_comprehension_artifact() -> None:
    """Not a global file that may describe a different endpoint entirely."""
    source = (Path(__file__).resolve().parent.parent
              / "senapathy" / "orchestrator.py").read_text()
    assert 'endpoint_model_path=ctx.run_dir / "endpoint_model.json"' in source
