"""Ship it, or refuse honestly.

Two paths, and the refusal is the common one. If the gate passed, the green
branch is committed, pushed, and a PR is opened with the rendered body. If it
did not, no PR is created and the report is written anyway with the terminal
state the artifacts actually support — because "we measured it and it is not
better" is a result, and hiding it would make the next run repeat the work.

Nothing here can flip a gate. `evaluate_gates` has already returned a
`GateDecision` whose `passed` is computed, and this module only reads it.
"""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .artifacts import ArtifactStore
from .contracts import (ArtifactRef, CandidatePlan, EquivalenceVerdict,
                        GateDecision, Hypothesis, Patch, PerfReport,
                        TerminalState)
from .errors import ContractViolation, ToolError
from .gate.rules import evaluate_gates, render_gate_table, terminal_state_for
from .report.narrative import write_narrative
from .report.render import Backed, EvidenceLedger, render_report

GIT_TIMEOUT_S = 300


# ─────────────────────────  loading the evidence  ────────────────────────


def _read(path: Path, label: str) -> dict[str, Any]:
    if not path.exists():
        raise ContractViolation(
            f"no {label} at {path}. The gate needs every artifact; it does not "
            "guess at a missing one."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def load_evidence(run_dir: Path) -> dict[str, Any]:
    """Every artifact the gate reads, plus a content-addressed ref for each.

    Refs come from `ArtifactStore`, so each one is verified against the file it
    points at before the ledger will quote a number out of it.
    """
    directory = Path(run_dir)
    payloads = {
        "verification": _read(directory / "verification.json", "verification"),
        "perf_report": _read(directory / "perf_report.json", "performance report"),
        "patch": _read(directory / "patch.json", "patch"),
        "hypotheses": _read(directory / "hypotheses.json", "diagnosis"),
        "baseline": _read(directory / "baseline.json", "baseline"),
    }

    store = ArtifactStore(directory)
    refs: dict[str, ArtifactRef] = {}
    for kind, payload in payloads.items():
        target = store.artifacts_dir / f"{kind}.json"
        target.write_text(
            json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        from .ids import content_hash

        refs[kind] = ArtifactRef(
            kind=kind, hash=content_hash(payload), path=str(target)
        )

    return {"payloads": payloads, "refs": refs, "run_dir": directory}


def selected_plan(run_dir: Path) -> CandidatePlan:
    """The candidate the plan phase SELECTED, as the shape the gates read.

    This replaces `plan_from_patch`, which rebuilt a plan out of the patch's
    own self-description because the plan phase did not exist. That hack could
    only ever agree with the patch — it was derived from it — so
    DIFF_MINIMALITY was checking the diff against a file list copied out of the
    diff. The allowlist now comes from a decision made BEFORE the patch was
    written, which is the only way that gate means anything.

    `risk_tier` and `requires_flag` come across unchanged; both were assigned
    by code in the plan phase, never by a model.
    """
    from .plan.agent import load_plan

    path = Path(run_dir) / "plan.json"
    if not path.exists():
        raise ContractViolation(
            f"no plan at {path}. The gate needs the plan the patch was written "
            f"against — run `perfagent plan`. It does not reconstruct one from "
            f"the patch, because a plan derived from the patch cannot "
            f"contradict it."
        )
    plan = load_plan(path)
    candidate = plan.selected()
    if candidate is None:
        raise ContractViolation(
            f"{path} records no selected candidate "
            f"({plan.selection_reason or 'no reason given'}), so there was "
            f"nothing for the patch phase to implement."
        )
    return CandidatePlan(
        id=candidate.id,
        targets_hypothesis=candidate.targets_hypothesis,
        description=candidate.description,
        mechanism_changed=candidate.mechanism_changed,
        predicted_effect=candidate.predicted_effect,
        files_to_touch=list(candidate.files_to_touch),
        risk_tier=candidate.risk_tier,
        what_could_break=list(candidate.what_could_break),
        requires_flag=candidate.requires_flag,
    )


def top_hypothesis(hypotheses_document: dict[str, Any]) -> Hypothesis:
    ranked = hypotheses_document["ranked_ids"]
    by_id = {h["id"]: h for h in hypotheses_document["hypotheses"]}
    return Hypothesis.model_validate(by_id[ranked[0]])


# ────────────────────────────────  the gate  ──────────────────────────────


def run_gate(run_dir: Path) -> dict[str, Any]:
    """Load the artifacts, evaluate the six gates, pick the terminal state."""
    directory = Path(run_dir)
    evidence = load_evidence(directory)
    payloads = evidence["payloads"]

    verdict = EquivalenceVerdict.model_validate(payloads["verification"]["verdict"])
    perf = PerfReport.model_validate(payloads["perf_report"]["report"])
    patch = Patch.model_validate(payloads["patch"]["patch"])
    plan = selected_plan(directory)

    decision = evaluate_gates(
        verdict, perf, patch, plan,
        paired_p_value=payloads["perf_report"].get("wilcoxon_p"),
    )
    terminal = terminal_state_for(decision, shipped=False)

    return {
        "evidence": evidence,
        "verdict": verdict,
        "perf": perf,
        "patch": patch,
        "plan": plan,
        "decision": decision,
        "terminal": terminal,
    }


# ──────────────────────────────  the report  ─────────────────────────────


def build_ledger(gated: dict[str, Any]) -> tuple[EvidenceLedger, dict[str, Backed]]:
    """Put every quantity on the ledger, each with the artifact behind it."""
    refs = gated["evidence"]["refs"]
    payloads = gated["evidence"]["payloads"]
    ledger = EvidenceLedger()

    ledger.record_from(refs["perf_report"], payloads["perf_report"], {
        "blue_p50": "report.before.p50_ms",
        "blue_p95": "report.before.p95_ms",
        "blue_p99": "report.before.p99_ms",
        "green_p50": "report.after.p50_ms",
        "green_p95": "report.after.p95_ms",
        "green_p99": "report.after.p99_ms",

        "blue_sql": "report.sql_count_before",
        "green_sql": "report.sql_count_after",
        "blue_rows": "report.rows_examined_before",
        "green_rows": "report.rows_examined_after",
    })
    # Deltas are recorded as SIGNED, worded strings derived from the artifact
    # value. An unsigned "4.513 %" in a table reads as an improvement when it
    # is a regression, and the sign is the whole message.
    report_payload = payloads["perf_report"]["report"]
    for key, source in (("p50_delta", "p50_delta_pct"), ("p95_delta", "p95_delta_pct")):
        value = report_payload[source]
        ledger.record(
            key,
            f"{value:+.2f} % ({'faster' if value < 0 else 'slower'})",
            refs["perf_report"],
        )
    for key, source in (("paired_p", "wilcoxon_p"),):
        raw_p = payloads["perf_report"][source]
        ledger.record(key, f"{raw_p:.6f}" if raw_p is not None else "not computed",
                      refs["perf_report"])
    ledger.record("unpaired_p", f"{report_payload['p_value']:.6f}",
                  refs["perf_report"])
    ledger.record(
        "pair_count", len(payloads["perf_report"]["pairs"]), refs["perf_report"]
    )

    ledger.record_from(refs["verification"], payloads["verification"], {
        "oracle_hash": "oracle_hash",
        "corpus_hash": "corpus_hash",
        "compared": "verdict.compared",
        "identical": "verdict.identical",
    })
    ledger.record(
        "diff_count", payloads["verification"].get("diffs_total", 0),
        refs["verification"],
    )
    ledger.record(
        "held_back_count", len(payloads["verification"]["held_back_requests"]),
        refs["verification"],
    )

    ledger.record_from(refs["patch"], payloads["patch"], {
        "branch": "patch.branch",
        "lines_added": "patch.lines_added",
        "lines_removed": "patch.lines_removed",
        "base_commit": "worktree.base_commit",
        "patch_summary": "emitted.summary",
        "mechanism_changed": "emitted.mechanism_changed",
        "expected_effect": "emitted.expected_effect",
    })

    hypothesis = top_hypothesis(payloads["hypotheses"])
    ledger.record("bottleneck_claim", hypothesis.claim, refs["hypotheses"])
    ledger.record("attributed_ms", hypothesis.attributed_ms, refs["hypotheses"])
    ledger.record_from(refs["baseline"], payloads["baseline"], {
        "baseline_p50": "benchmark.latency.p50_ms",
        "endpoint": "corpus.requests.0.path",
    })

    file_list = "\n".join(f"- `{f}`" for f in gated["patch"].files_changed)
    ledger.record("file_list", file_list, refs["patch"])

    gate_rows = "\n".join(
        f"| `{r.name.value}` | {'PASS' if r.passed else 'FAIL'} | "
        f"{r.threshold} | {r.actual} |"
        for r in gated["decision"].results
    )
    decision_ref = refs["perf_report"]
    ledger.record("gate_rows", gate_rows, decision_ref)
    ledger.record(
        "gate_summary",
        "all gates passed" if gated["decision"].passed
        else f"{sum(1 for r in gated['decision'].results if not r.passed)} of "
             f"{len(gated['decision'].results)} gates failed",
        decision_ref,
    )
    ledger.record("terminal_state", gated["terminal"].value, decision_ref)

    return ledger, {}


def write_report(
    gated: dict[str, Any], *, client: Any = None
) -> tuple[str, Path, dict[str, Any]]:
    """Render report.md. One LLM call, for the root-cause paragraph only."""
    ledger, _ = build_ledger(gated)

    narrative, narrative_trace = write_narrative(
        top_hypothesis(gated["evidence"]["payloads"]["hypotheses"]),
        gated["plan"],
        client=client,
    )
    # The paragraph is prose, so its "artifact" is the diagnosis it explains.
    ledger.record(
        "narrative", narrative, gated["evidence"]["refs"]["hypotheses"]
    )

    provenance_rows = "\n".join(
        f"- `{entry['claim']}` = {entry['value']} "
        f"(`{entry['artifact']}`, hash `{entry['hash']}`)"
        for entry in ledger.provenance()
    )
    ledger.record(
        "provenance_rows", provenance_rows,
        gated["evidence"]["refs"]["perf_report"],
    )

    body = render_report(ledger)
    out_path = Path(gated["evidence"]["run_dir"]) / "report.md"
    out_path.write_text(body, encoding="utf-8")
    return body, out_path, narrative_trace


# ────────────────────────────────  shipping  ─────────────────────────────


def _git(path: Path, *args: str, check: bool = True) -> str:
    result = subprocess.run(
        ["git", "-C", str(path), *args], capture_output=True, text=True,
        timeout=GIT_TIMEOUT_S,
    )
    if check and result.returncode != 0:
        raise ToolError(
            f"git {' '.join(args)} failed in {path}: "
            f"{result.stderr.strip() or result.stdout.strip()}"
        )
    return result.stdout


def open_pull_request(
    green_path: Path, branch: str, title: str, body: str, *, dry_run: bool = False
) -> dict[str, Any]:
    """Commit, push, `gh pr create`. Only reached when the gate passed."""
    _git(green_path, "add", "-A")
    status = _git(green_path, "status", "--porcelain")
    if status.strip():
        _git(
            green_path,
            "-c", "user.email=perfagent@local", "-c", "user.name=perfagent",
            "commit", "-q", "-m", title,
        )
    head = _git(green_path, "rev-parse", "HEAD").strip()

    if dry_run:
        return {"dry_run": True, "branch": branch, "head": head,
                "would_run": ["git push -u origin " + branch,
                              "gh pr create --head " + branch]}

    _git(green_path, "push", "-u", "origin", branch)
    created = subprocess.run(
        ["gh", "pr", "create", "--head", branch, "--title", title, "--body", body],
        cwd=str(green_path), capture_output=True, text=True, timeout=GIT_TIMEOUT_S,
    )
    if created.returncode != 0:
        raise ToolError(
            "gh pr create failed: "
            f"{created.stderr.strip() or created.stdout.strip()}"
        )
    return {"dry_run": False, "branch": branch, "head": head,
            "pr_url": created.stdout.strip()}


def run_ship(
    run_dir: Path,
    *,
    green_path: Path = Path(".perfagent/green"),
    client: Any = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Gate, report, and either open a PR or refuse — honestly, either way."""
    gated = run_gate(run_dir)
    decision: GateDecision = gated["decision"]

    body, report_path, narrative_trace = write_report(gated, client=client)

    shipped: dict[str, Any] | None = None
    if decision.passed:
        shipped = open_pull_request(
            Path(green_path), gated["patch"].branch,
            title=f"perf: {gated['plan'].description[:60]}",
            body=body, dry_run=dry_run,
        )
        terminal = terminal_state_for(decision, shipped=not dry_run)
    else:
        # No PR. The report is still written, with the state the evidence
        # supports — a refusal nobody records is a refusal that gets repeated.
        terminal = gated["terminal"]

    document = {
        "terminal_state": terminal.value,
        "gate": decision.model_dump(mode="json"),
        "pr": shipped,
        "report_path": str(report_path),
        "narrative": narrative_trace,
        "artifacts": {
            kind: ref.model_dump(mode="json")
            for kind, ref in gated["evidence"]["refs"].items()
        },
        "shipped_at": datetime.now(timezone.utc).isoformat(),
    }
    out_path = Path(run_dir) / "gate.json"
    out_path.write_text(
        json.dumps(document, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    return {
        "document": document, "decision": decision, "terminal": terminal,
        "report": body, "report_path": report_path, "gate_path": out_path,
        "narrative": narrative_trace,
    }


def render_ship_summary(result: dict[str, Any]) -> str:
    decision, terminal = result["decision"], result["terminal"]
    out = [render_gate_table(decision, terminal)]
    pr = result["document"]["pr"]
    if pr is None:
        out.append("NO PULL REQUEST")
        out.append("  The gate did not pass, so nothing was pushed and no PR was "
                   "opened.")
        out.append(f"  The report records the honest outcome: {terminal.value}")
    elif pr.get("dry_run"):
        out.append("PULL REQUEST (dry run — nothing was pushed)")
        for command in pr["would_run"]:
            out.append(f"  would run: {command}")
    else:
        out.append("PULL REQUEST OPENED")
        out.append(f"  {pr['pr_url']}")
    out.append("")
    trace = result["narrative"]
    out.append(f"NARRATIVE  source={trace.get('source')} "
               f"violations={len(trace.get('violations', []))}")
    if trace.get("reason"):
        out.append(f"  reason: {trace['reason']}")
    return "\n".join(out)
