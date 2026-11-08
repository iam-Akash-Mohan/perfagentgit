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

from senapathy.contracts import (CandidateFix, EquivalenceVerdict,
                                 LatencyStats, Patch, PerfReport, Plan,
                                 RiskTier, TerminalState)
from senapathy.contracts.plan import FixabilityAssessment, RankedHypothesis
from senapathy.errors import ToolError
from senapathy.gate.rules import terminal_state_for
from senapathy.ship import open_pull_request, run_gate, run_ship, write_report


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
    _git(r, "init", "-q", "-b", "senapathy/fixture")
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
            repo, "senapathy/fixture", title="t", body="b",
            base_commit=base, files_changed=["V2.sql"], dry_run=True,
        )

    message = str(exc.value)
    assert "senapathy/fixture" in message  # what the patch artifact claims
    assert "some-other-branch" in message  # what the worktree actually is


def test_refuses_when_worktree_is_missing_a_claimed_file(repo: Path) -> None:
    base = _base_commit(repo)
    (repo / "V2.sql").write_text("CREATE INDEX x ON y (z);\n")
    # Claim two files; only one was actually touched.

    with pytest.raises(ToolError) as exc:
        open_pull_request(
            repo, "senapathy/fixture", title="t", body="b",
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
            repo, "senapathy/fixture", title="t", body="b",
            base_commit=base, files_changed=["V2.sql"], dry_run=True,
        )

    message = str(exc.value)
    assert "Extra.java" in message
    assert "actually touched but not in patch.files_changed" in message


def test_passes_when_branch_and_files_match_the_patch(repo: Path) -> None:
    base = _base_commit(repo)
    (repo / "V2.sql").write_text("CREATE INDEX x ON y (z);\n")

    result = open_pull_request(
        repo, "senapathy/fixture", title="t", body="b",
        base_commit=base, files_changed=["V2.sql"], dry_run=True,
    )

    assert result["dry_run"] is True
    assert result["branch"] == "senapathy/fixture"
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
    branch = "senapathy/fixture"
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


# ─────────────────────  the reworked report body  ─────────────────────────


def _write_reconciliation_run(run_dir: Path, *, base_commit: str, branch: str) -> None:
    """Three ranked causes, one unclassified and selected, zero tests
    executed -- a fixture that exercises every new section at once rather
    than one bare-minimum field at a time: the human-check callout, the
    reconciliation arithmetic, and the confidence band's two unknowns.
    """
    verdict = EquivalenceVerdict(
        passed=True, compared=12, identical=12, diffs=[],
        db_state_matched=True, external_calls_matched=True,
        held_back_passed=True, oracle_hash="orc",
    )
    (run_dir / "verification.json").write_text(json.dumps({
        "verdict": verdict.model_dump(mode="json"),
        "diffs_total": 0, "oracle_hash": "orc", "corpus_hash": "corp",
        "held_back_requests": ["r1", "r2", "r3", "r4"],
    }))

    before = LatencyStats(n=10, p50_ms=30.0, p95_ms=44.0, p99_ms=49.0,
                          mean_ms=30.0, stdev_ms=3.0)
    after = LatencyStats(n=10, p50_ms=6.0, p95_ms=11.0, p99_ms=12.0,
                         mean_ms=6.0, stdev_ms=1.0)
    perf = PerfReport(
        before=before, after=after,
        p50_delta_pct=-80.0, p95_delta_pct=-75.0,
        sql_count_before=100, sql_count_after=2,
        rows_examined_before=1000, rows_examined_after=1000,
        p_value=0.4, p_value_paired=0.0001,
        mechanism_confirmed=True, mechanism_note="n",
    )
    (run_dir / "perf_report.json").write_text(json.dumps({
        "report": perf.model_dump(mode="json"),
        "wilcoxon_p": 0.0001, "pairs": [],
    }))

    patch = Patch(
        plan_id="c1", branch=branch, diff=_DIFF,
        files_changed=["V2.sql"], lines_added=1, lines_removed=0,
        build_iterations=1, tests_passed=0, tests_before=0,
    )
    (run_dir / "patch.json").write_text(json.dumps({
        "outcome": "PATCHED", "patch": patch.model_dump(mode="json"),
        "targets_hypothesis": "h1",
        "worktree": {"base_commit": base_commit},
        "baseline_tests": "tests executed 0, passed 0, failures 0, errors 0, "
                          "skipped 0; test files on disk 2",
        "emitted": {"summary": "s", "mechanism_changed": "SQL statements per request",
                   "files_touched": ["V2.sql"], "expected_effect": "e", "risks": []},
    }))

    candidate = CandidateFix(
        id="c1", targets_hypothesis="h1", approach="fetch_join",
        description="fetch join replaces the per-row query",
        mechanism_changed="SQL statements per request",
        predicted_effect="queries per request drop",
        files_to_touch=["V2.sql"], what_could_break=[],
        preserves_response_bytes=True, risk_tier=RiskTier.MEDIUM,
        expected_value_ms=4.0,
    )
    ranked = [
        RankedHypothesis(
            hypothesis_id="h1", claim="per-row N+1 dominates", layer="application_logic",
            attributed_ms=10.0,
            fixability=FixabilityAssessment(
                hypothesis_id="h1", fixability=0.4,
                reason="UNCLASSIFIED: no policy row matched this hypothesis",
                classified=False,
            ),
            expected_value_ms=4.0,
        ),
        RankedHypothesis(
            hypothesis_id="h2", claim="serialization dominates", layer="serialization",
            attributed_ms=8.0,
            fixability=FixabilityAssessment(
                hypothesis_id="h2", fixability=0.3, reason="library-level",
                classified=True,
            ),
            expected_value_ms=2.4,
        ),
        RankedHypothesis(
            hypothesis_id="h3", claim="network transfer dominates", layer="network",
            attributed_ms=6.0,
            fixability=FixabilityAssessment(
                hypothesis_id="h3", fixability=0.15, reason="out of scope",
                blocked_by="transfer time is not changed by a code patch",
                classified=True,
            ),
            expected_value_ms=0.9,
        ),
    ]
    plan = Plan(
        endpoint="GET /x", measured_total_ms=24.0, ranked=ranked,
        considered=["h1"], candidates=[candidate], selected_id="c1",
        selection_reason="r", base_commit=base_commit,
    )
    (run_dir / "plan.json").write_text(json.dumps(plan.model_dump(mode="json")))

    (run_dir / "hypotheses.json").write_text(json.dumps({
        "measured_total_ms": 24.0, "ranked_ids": ["h1", "h2", "h3"],
        "hypotheses": [
            {"id": "h1", "claim": "per-row N+1 dominates", "mechanism": "m",
             "layer": "application_logic", "evidence": [], "attributed_ms": 10.0,
             "prediction": "p", "confidence": 0.8},
            {"id": "h2", "claim": "serialization dominates", "mechanism": "m",
             "layer": "serialization", "evidence": [], "attributed_ms": 8.0,
             "prediction": "p", "confidence": 0.6},
            {"id": "h3", "claim": "network transfer dominates", "mechanism": "m",
             "layer": "network", "evidence": [], "attributed_ms": 6.0,
             "prediction": "p", "confidence": 0.5},
        ],
    }))
    (run_dir / "baseline.json").write_text(json.dumps({
        "corpus": {"hash": "x", "requests": [{"id": "r1", "path": "GET /x"}]},
        "benchmark": {"latency": {"p50_ms": 28.0}},
    }))
    (run_dir / "endpoint_model.json").write_text(json.dumps({"uncertainties": []}))


def _render_reconciliation_report(tmp_path: Path, repo: Path) -> str:
    base = _base_commit(repo)
    branch = "senapathy/recon"
    (repo / "V2.sql").write_text("CREATE INDEX idx_x ON invoice (customer_id, status);\n")

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_reconciliation_run(run_dir, base_commit=base, branch=branch)

    gated = run_gate(run_dir)
    terminal = terminal_state_for(gated["decision"], shipped=False)
    body, _, _ = write_report(gated, terminal=terminal, client=_NoNetworkClient())
    return body


def test_provenance_collapses_and_drops_duplicated_blocks(
    tmp_path: Path, repo: Path
) -> None:
    body = _render_reconciliation_report(tmp_path, repo)

    assert "<details>\n<summary>Provenance" in body
    # Dropped: these duplicate a block already rendered verbatim elsewhere,
    # so they must not reappear as a second full copy in the footer.
    for key in ("gate_rows", "ranked_alternatives", "narrative", "file_list",
               "pre_existing_risks"):
        assert f"- `{key}` = " not in body
        # but the artifact backing it is still named, hash and all.
        assert f"`{key}` ->" in body
    # Everything else is still individually listed.
    assert "- `attributed_ms` = 10.0" in body


def test_reconciliation_computes_predicted_vs_delivered_from_artifacts(
    tmp_path: Path, repo: Path
) -> None:
    body = _render_reconciliation_report(tmp_path, repo)

    assert "## Reconciliation: predicted vs. delivered" in body
    assert "attributed **10.0 ms**" in body
    assert "delivered **24.0 ms**" in body
    assert "2.40x the predicted amount" in body
    # The other two ranked causes, named with their own attribution.
    assert "serialization dominates" in body
    assert "network transfer dominates" in body
    assert "attributed 8.0 ms" in body
    assert "attributed 6.0 ms" in body
    # 10 + 8 + 6 == 24, an exact match to what was delivered.
    assert "24.0 ms total" in body
    assert "within 0.0% of the 24.0 ms" in body


def test_human_check_names_the_selected_hypotheses_unclassified_fixability(
    tmp_path: Path, repo: Path
) -> None:
    body = _render_reconciliation_report(tmp_path, repo)

    assert "## What a human must check before merging" in body
    assert "never classified" in body
    assert "**Risk tier:** **MEDIUM** (`T2`)" in body
    assert "test suite did not meaningfully run" in body
    assert "What the oracle did not cover" in body
    # Confidence band surfaces the same two unknowns, up in the header.
    assert "**Confidence:** **MEDIUM**" in body
    assert "placeholder" in body
    assert "test suite executed zero tests" in body
