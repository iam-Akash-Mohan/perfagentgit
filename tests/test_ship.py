"""The `open_pull_request` precondition: the worktree must actually BE what
the patch artifact claims, or shipping pushes someone else's state under this
patch's name.

No network: every test uses dry_run=True, so the only git commands that run
are `add`, `status`, the new `rev-parse`/`diff` checks, and the local
`commit` -- `push` and `gh pr create` are never reached.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from perfagent.contracts import (CandidateFix, EquivalenceVerdict,
                                 LatencyStats, Patch, PerfReport, Plan,
                                 RiskTier, TerminalState)
from perfagent.errors import ToolError
from perfagent.ship import open_pull_request, run_ship


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args], capture_output=True, text=True, check=True
    )
    return result.stdout


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """A real git repo, one commit, on the branch the patch artifact claims."""
    r = tmp_path / "green"
    r.mkdir()
    (r / "README.md").write_text("hello\n")
    _git(r, "init", "-q", "-b", "perfagent/fixture")
    _git(r, "config", "user.email", "t@t")
    _git(r, "config", "user.name", "t")
    _git(r, "add", "-A")
    _git(r, "commit", "-qm", "base")
    return r


def _base_commit(repo: Path) -> str:
    return _git(repo, "rev-parse", "HEAD").strip()


def test_refuses_when_worktree_is_on_the_wrong_branch(repo: Path) -> None:
    base = _base_commit(repo)
    _git(repo, "checkout", "-q", "-b", "some-other-branch")
    (repo / "V2.sql").write_text("CREATE INDEX x ON y (z);\n")

    with pytest.raises(ToolError) as exc:
        open_pull_request(
            repo, "perfagent/fixture", title="t", body="b",
            base_commit=base, files_changed=["V2.sql"], dry_run=True,
        )

    message = str(exc.value)
    assert "perfagent/fixture" in message  # what the patch artifact claims
    assert "some-other-branch" in message  # what the worktree actually is


def test_refuses_when_worktree_is_missing_a_claimed_file(repo: Path) -> None:
    base = _base_commit(repo)
    (repo / "V2.sql").write_text("CREATE INDEX x ON y (z);\n")
    # Claim two files; only one was actually touched.

    with pytest.raises(ToolError) as exc:
        open_pull_request(
            repo, "perfagent/fixture", title="t", body="b",
            base_commit=base, files_changed=["V2.sql", "NeverTouched.java"],
            dry_run=True,
        )

    message = str(exc.value)
    assert "V2.sql" in message
    assert "NeverTouched.java" in message
    assert "claimed by patch.files_changed but not actually touched" in message


def test_refuses_when_worktree_touches_an_unclaimed_file(repo: Path) -> None:
    base = _base_commit(repo)
    (repo / "V2.sql").write_text("CREATE INDEX x ON y (z);\n")
    (repo / "Extra.java").write_text("class Extra {}\n")
    # Claim only V2.sql; Extra.java was ALSO touched, uncommitted.

    with pytest.raises(ToolError) as exc:
        open_pull_request(
            repo, "perfagent/fixture", title="t", body="b",
            base_commit=base, files_changed=["V2.sql"], dry_run=True,
        )

    message = str(exc.value)
    assert "Extra.java" in message
    assert "actually touched but not in patch.files_changed" in message


def test_passes_when_branch_and_files_match_the_patch(repo: Path) -> None:
    base = _base_commit(repo)
    (repo / "V2.sql").write_text("CREATE INDEX x ON y (z);\n")

    result = open_pull_request(
        repo, "perfagent/fixture", title="t", body="b",
        base_commit=base, files_changed=["V2.sql"], dry_run=True,
    )

    assert result["dry_run"] is True
    assert result["branch"] == "perfagent/fixture"
    # dry_run commits locally (there was something to commit) but never pushes.
    assert _git(repo, "status", "--porcelain").strip() == ""
    assert any("git push" in cmd for cmd in result["would_run"])
    assert any("gh pr create" in cmd for cmd in result["would_run"])


# ───────────────────  run_ship renders the REAL terminal state  ──────────


class _NoNetworkClient:
    """Guarantees write_narrative cannot reach a network, by construction.

    write_narrative(client=None) builds the real default client itself --
    None there means "use the real one", not "skip it". This object is the
    only way to make the call-site's `llm = client or _default_client()` pick
    something that can never open a socket.
    """

    def complete(self, **kwargs):
        raise AssertionError(
            "a model call was attempted; this test must never make one")


_DIFF = """diff --git a/V2.sql b/V2.sql
new file mode 100644
--- /dev/null
+++ b/V2.sql
@@ -0,0 +1,1 @@
+CREATE INDEX idx_x ON invoice (customer_id, status);
"""


def _write_all_passing_run(run_dir: Path, *, base_commit: str, branch: str) -> None:
    """A minimal, internally-coherent run dir where every gate passes.

    Built from real Pydantic objects (not hand-typed JSON) wherever the
    loader actually validates the shape -- verification, perf_report, patch,
    plan -- so this fixture cannot silently drift from the contracts it is
    supposed to satisfy.
    """
    verdict = EquivalenceVerdict(
        passed=True, compared=5, identical=5, diffs=[],
        db_state_matched=True, external_calls_matched=True,
        held_back_passed=True, oracle_hash="orc",
    )
    (run_dir / "verification.json").write_text(json.dumps({
        "verdict": verdict.model_dump(mode="json"),
        "diffs_total": 0, "oracle_hash": "orc", "corpus_hash": "corp",
        "held_back_requests": [],
    }))

    before = LatencyStats(n=10, p50_ms=1000.0, p95_ms=1400.0, p99_ms=1500.0,
                          mean_ms=1000.0, stdev_ms=100.0)
    after = LatencyStats(n=10, p50_ms=700.0, p95_ms=1000.0, p99_ms=1050.0,
                         mean_ms=700.0, stdev_ms=60.0)
    perf = PerfReport(
        before=before, after=after,
        p50_delta_pct=-30.0, p95_delta_pct=-28.57,
        sql_count_before=10, sql_count_after=2,
        rows_examined_before=1000, rows_examined_after=1000,
        p_value=0.5, mechanism_confirmed=True, mechanism_note="n",
    )
    (run_dir / "perf_report.json").write_text(json.dumps({
        "report": perf.model_dump(mode="json"),
        "wilcoxon_p": 0.0009, "pairs": [],
    }))

    patch = Patch(
        plan_id="c1", branch=branch, diff=_DIFF,
        files_changed=["V2.sql"], lines_added=1, lines_removed=0,
        build_iterations=1, tests_passed=1, tests_before=1,
    )
    (run_dir / "patch.json").write_text(json.dumps({
        "outcome": "PATCHED", "patch": patch.model_dump(mode="json"),
        "targets_hypothesis": "h1",
        "worktree": {"base_commit": base_commit},
        "emitted": {"summary": "s", "mechanism_changed": "SQL statements per request",
                   "files_touched": ["V2.sql"], "expected_effect": "e", "risks": []},
    }))

    candidate = CandidateFix(
        id="c1", targets_hypothesis="h1", approach="fetch_join",
        description="fetch join replaces the per-row query",
        mechanism_changed="SQL statements per request",
        predicted_effect="queries per request drop",
        files_to_touch=["V2.sql"], what_could_break=[],
        preserves_response_bytes=True, risk_tier=RiskTier.LOW,
        expected_value_ms=100.0,
    )
    plan = Plan(
        endpoint="GET /x", measured_total_ms=1000.0, ranked=[],
        considered=["h1"], candidates=[candidate], selected_id="c1",
        selection_reason="r", base_commit=base_commit,
    )
    (run_dir / "plan.json").write_text(
        json.dumps(plan.model_dump(mode="json")))

    (run_dir / "hypotheses.json").write_text(json.dumps({
        "measured_total_ms": 1000.0, "ranked_ids": ["h1"],
        "hypotheses": [{
            "id": "h1", "claim": "c", "mechanism": "m", "layer": "serialization",
            "evidence": [], "attributed_ms": 500.0, "prediction": "p",
            "confidence": 0.8,
        }],
    }))
    (run_dir / "baseline.json").write_text(json.dumps({
        "corpus": {"hash": "x", "requests": [{"id": "r1", "path": "GET /x"}]},
        "benchmark": {"latency": {"p50_ms": 500.0}},
    }))
    (run_dir / "endpoint_model.json").write_text(json.dumps({"uncertainties": []}))


def test_dry_run_ship_renders_ready_to_ship_not_bottleneck_not_fixable(
    tmp_path: Path, repo: Path
) -> None:
    """The actual regression: `run_gate()` always computes its terminal state
    with shipped=False, and the report used to read THAT value straight off
    `gated["terminal"]` -- so a dry run with every gate passing rendered
    BOTTLENECK_NOT_FIXABLE, a FAILURE state, into its own report body."""
    base = _base_commit(repo)
    branch = "perfagent/fixture"
    (repo / "V2.sql").write_text("CREATE INDEX idx_x ON invoice (customer_id, status);\n")

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_all_passing_run(run_dir, base_commit=base, branch=branch)

    result = run_ship(
        run_dir, green_path=repo, client=_NoNetworkClient(), dry_run=True
    )

    assert result["decision"].passed is True
    assert result["terminal"] == TerminalState.READY_TO_SHIP
    assert result["terminal"] != TerminalState.BOTTLENECK_NOT_FIXABLE

    report_text = (run_dir / "report.md").read_text()
    assert "READY_TO_SHIP" in report_text
    assert "BOTTLENECK_NOT_FIXABLE" not in report_text

    gate_doc = json.loads((run_dir / "gate.json").read_text())
    assert gate_doc["terminal_state"] == "READY_TO_SHIP"
    assert gate_doc["pr"]["dry_run"] is True
    assert "gate_decision" in gate_doc["artifacts"]
