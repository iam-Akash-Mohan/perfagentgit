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
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .artifacts import ArtifactStore
from .contracts import (ArtifactRef, CandidatePlan, EquivalenceVerdict,
                        GateDecision, Hypothesis, Patch, PerfReport,
                        TerminalState)
from .errors import ContractViolation, ToolError
from .gate.rules import (confidence_band_for, evaluate_gates,
                         render_gate_table, terminal_state_for)
from .report.narrative import write_narrative
from .report.render import Backed, EvidenceLedger, render_report

# Entries dropped from the printed provenance list because they already
# duplicate a block rendered verbatim elsewhere on the page (a table, a
# bulleted ranking, a file list) -- repeating them as "`key` = <the whole
# block again>" is how the footer grew to ~60 lines restating the document.
# Their artifact hashes are still listed, in `_artifacts_rendered_elsewhere`.
_PROVENANCE_DUPLICATES = frozenset({
    "gate_rows", "ranked_alternatives", "narrative", "file_list",
    "pre_existing_risks",
})

_RISK_TIER_WORDS: dict[int, tuple[str, str]] = {
    1: ("LOW", "an index, @EntityGraph, a read-only transaction, or view narrowing"),
    2: ("MEDIUM", "a query rewrite, batching, pushing a predicate into SQL, "
                  "or a projection"),
    3: ("HIGH", "flow restructuring or a transaction-boundary change -- "
                "flag-gated, not something an unattended fix should ship"),
    4: ("FORBIDDEN", "a schema change beyond an additive index, an API "
                     "contract change, or touching auth/tests"),
}

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
        # The plan is evidence too: "why not the bigger number" is the part
        # of this report that makes it reviewable, and it is quoted from here.
        "plan": _read(directory / "plan.json", "plan"),
        # The comprehension map carries pre-existing risks the understand
        # phase noticed while tracing the endpoint — a missing ORDER BY on an
        # offset-paginated query, for one — which have nothing to do with this
        # patch and everything to do with whether a reviewer should trust the
        # endpoint. Found looking for a bottleneck; still worth reporting.
        "endpoint_model": _read(directory / "endpoint_model.json",
                                "comprehension model"),
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
            f"against — run `senapathy plan`. It does not reconstruct one from "
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
        mechanism=candidate.mechanism,
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


def _gate_decision_ref(gated: dict[str, Any]) -> ArtifactRef:
    """Write the GateDecision itself to the artifact store and ref it.

    `evaluate_gates` PRODUCES the decision; nothing upstream of it (the perf
    report, the patch, ...) did. `terminal_state` and the pass/fail of each
    gate are facts about THIS decision, so they need to be backed by it, not
    by whichever input artifact happened to be lying around.
    """
    from .ids import content_hash

    run_dir = gated["evidence"]["run_dir"]
    store = ArtifactStore(run_dir)
    payload = gated["decision"].model_dump(mode="json")
    target = store.artifacts_dir / "gate_decision.json"
    target.write_text(
        json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    ref = ArtifactRef(kind="gate_decision", hash=content_hash(payload), path=str(target))
    # Visible in gate.json's own "artifacts" block alongside the inputs.
    gated["evidence"]["refs"]["gate_decision"] = ref
    return ref


def build_ledger(
    gated: dict[str, Any], terminal: TerminalState
) -> tuple[EvidenceLedger, dict[str, Backed]]:
    """Put every quantity on the ledger, each with the artifact behind it.

    `terminal` is the state THIS run actually reached — computed by the
    caller, who knows whether this is a dry run or a real push — not
    recomputed here from a pre-ship default.
    """
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
    held_back_count = len(payloads["verification"]["held_back_requests"])
    ledger.record("held_back_count", held_back_count, refs["verification"])

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

    # Planned vs. granted vs. final, not just the final diff: a file
    # reaches the patch either because the plan named it or because
    # request_file_access granted it mid-patch, and a reviewer should see
    # which happened, not just the union the diff ends up touching.
    planned = set(gated["plan"].files_to_touch)
    granted = set(gated["patch"].granted_files)
    file_list_lines = [f"- `{f}`" for f in gated["patch"].files_changed]
    if granted:
        file_list_lines.append("")
        file_list_lines.append(
            f"Planned scope: {len(planned)} file(s). "
            f"Granted during patching: {len(granted)} "
            f"({', '.join(f'`{f}`' for f in sorted(granted))}). "
            f"Final scope: {len(planned | granted)} file(s)."
        )
    file_list = "\n".join(file_list_lines)
    ledger.record("file_list", file_list, refs["patch"])

    gate_rows = "\n".join(
        f"| `{r.name.value}` | {'PASS' if r.passed else 'FAIL'} | "
        f"{r.threshold} | {r.actual} |"
        for r in gated["decision"].results
    )
    decision_ref = _gate_decision_ref(gated)
    ledger.record("gate_rows", gate_rows, decision_ref)
    ledger.record(
        "gate_summary",
        "all gates passed" if gated["decision"].passed
        else f"{sum(1 for r in gated['decision'].results if not r.passed)} of "
             f"{len(gated['decision'].results)} gates failed",
        decision_ref,
    )
    ledger.record("terminal_state", terminal.value, decision_ref)

    # ── the plan: the ranking, and why each alternative lost ────────────
    plan_ref = refs["plan"]
    plan_payload = payloads["plan"]
    ledger.record("ranked_alternatives", _ranked_rows(plan_payload), plan_ref)
    ledger.record("rejected_candidates", _rejected_rows(plan_payload), plan_ref)
    ledger.record_from(plan_ref, plan_payload, {"endpoint": "endpoint"})
    selected = _selected_candidate(plan_payload)
    ledger.record("approach", selected.get("approach", "unknown"), plan_ref)
    ledger.record("risk_tier", f"T{selected.get('risk_tier', '?')}", plan_ref)
    ledger.record("candidates_considered",
                  len(plan_payload.get("candidates") or []), plan_ref)
    ledger.record("index_evidence", _index_evidence(gated, selected), plan_ref)

    # ── what a human must check: computed, not left for a reviewer to dig
    #    out of a ranking list or a field nothing else surfaces ──────────
    risk_tier_int = int(selected.get("risk_tier") or 0)
    ledger.record("fixability_check",
                  _fixability_check(plan_payload, hypothesis.id), plan_ref)
    ledger.record("risk_tier_check", _risk_tier_check(risk_tier_int), plan_ref)
    ledger.record("test_suite_check", _test_suite_check(payloads["patch"]),
                  refs["patch"])

    # ── reconciliation: predicted vs. delivered, computed from artifacts ──
    before_p50 = payloads["perf_report"]["report"]["before"]["p50_ms"]
    after_p50 = payloads["perf_report"]["report"]["after"]["p50_ms"]
    delivered_ms = before_p50 - after_p50
    ledger.record("delivered_ms", f"{delivered_ms:.1f}", refs["perf_report"])
    attributed = hypothesis.attributed_ms
    ledger.record(
        "delivery_ratio",
        (f"{delivered_ms / attributed:.2f}x the predicted amount"
         if attributed else "undefined (predicted 0 ms)"),
        refs["perf_report"],
    )
    ledger.record(
        "reconciliation_note",
        _reconciliation_note(payloads["hypotheses"], hypothesis.id, delivered_ms),
        refs["hypotheses"],
    )

    # ── confidence band: a fixed rule table, never a model's guess ──────
    selected_entry = _ranked_entry_for(plan_payload, hypothesis.id)
    selected_classified = bool(
        ((selected_entry or {}).get("fixability") or {}).get("classified", True))
    tests_executed = payloads["patch"]["patch"].get("tests_before", 0)
    diff_lines = gated["patch"].lines_added + gated["patch"].lines_removed
    band = confidence_band_for(
        gated["decision"], verdict=gated["verdict"], perf=gated["perf"],
        risk_tier=risk_tier_int, diff_lines=diff_lines,
        held_back_count=held_back_count,
        selected_fixability_classified=selected_classified,
        tests_executed=tests_executed,
    )
    ledger.record("confidence_level", band["level"], decision_ref)
    ledger.record("confidence_reason", band["reason"], decision_ref)
    ledger.record("confidence_inputs",
                  "\n".join(f"- {i}" for i in band["inputs"]), decision_ref)
    ledger.record("confidence_would_raise", band["would_raise"], decision_ref)

    # ── the mechanism check: did the predicted quantity actually move? ──
    report_body = payloads["perf_report"]["report"]
    ledger.record(
        "mechanism_verdict",
        ("CONFIRMED — the quantity the plan predicted would move did move."
         if report_body.get("mechanism_confirmed")
         else "NOT CONFIRMED — the predicted quantity did not move. A latency "
              "change without its stated mechanism means the diagnosis was "
              "wrong even if the graph looks good."),
        refs["perf_report"])
    ledger.record("mechanism_note", report_body.get("mechanism_note", ""),
                  refs["perf_report"])

    # ── pre-existing risks: found while diagnosing, not caused by the patch ──
    model_ref = refs["endpoint_model"]
    uncertainties = payloads["endpoint_model"].get("uncertainties") or []
    ledger.record("pre_existing_risks", _pre_existing_rows(uncertainties),
                  model_ref)

    # ── bookkeeping tables excluded from the behaviour comparison ───────
    excluded = payloads["verification"].get("excluded_tables") or []
    ledger.record(
        "excluded_tables",
        ", ".join(f"`{t}`" for t in excluded) if excluded else "none",
        refs["verification"],
    )

    return ledger, {}


def _ranked_entry_for(plan_payload: dict[str, Any], hypothesis_id: str) -> dict[str, Any] | None:
    for entry in plan_payload.get("ranked") or []:
        if entry.get("hypothesis_id") == hypothesis_id:
            return entry
    return None


def _fixability_check(plan_payload: dict[str, Any], hypothesis_id: str) -> str:
    """Whether the diagnosed cause's fixability was a real classification or
    the policy table's 0.40 placeholder -- stated outright, not buried in a
    ranking list a reviewer has to read closely to notice.
    """
    entry = _ranked_entry_for(plan_payload, hypothesis_id)
    if entry is None:
        return ("The diagnosed cause does not appear in the plan's ranked "
                "list, so its fixability was never assessed here.")
    fix = entry.get("fixability") or {}
    if fix.get("classified", True):
        return (f"The diagnosed cause's fixability was a policy-table "
                f"classification ({fix.get('fixability')}: "
                f"\"{fix.get('reason')}\"), not a placeholder.")
    return (
        f"**The diagnosed cause's fixability was never classified.** No "
        f"policy-table row matched it, so the ranking used "
        f"{fix.get('fixability')} as a placeholder rather than an "
        f"assessment — the ranking that produced this fix rested on that "
        f"placeholder."
    )


def _risk_tier_check(risk_tier_int: int) -> str:
    word, meaning = _RISK_TIER_WORDS.get(
        risk_tier_int, ("UNKNOWN", "an unrecognised tier"))
    return f"**{word}** (`T{risk_tier_int}`) — {meaning}."


def _test_suite_check(patch_payload: dict[str, Any]) -> str:
    """What the repository's OWN test suite did, read from the string the
    patch phase already computed (`TestCounts.render()`) -- never re-derived
    from the model's narrative, since this is exactly the fact a reviewer
    asks for first and the report carried nowhere before this.

    `baseline_tests` sits at the TOP level of patch.json, a sibling of the
    `patch` block itself, not nested inside it.
    """
    rendered = patch_payload.get("baseline_tests", "not recorded")
    match = re.search(r"tests executed (\d+)", rendered)
    executed = int(match.group(1)) if match else None
    if executed:
        return f"The suite ran before this patch: {rendered}."
    return (
        f"**The repository's own test suite did not meaningfully run**: "
        f"{rendered}. Zero tests executed means this run cannot say whether "
        f"the patch broke anything the suite would have caught — only that "
        f"nothing it covers ran, before or after."
    )


def _reconciliation_note(
    hypotheses_doc: dict[str, Any], selected_id: str, delivered_ms: float,
) -> str:
    """Which other ranked causes' attributed cost also disappeared, and
    whether the total accounts for what was actually delivered -- computed
    from the hypothesis set, never narrated.
    """
    hypotheses = hypotheses_doc.get("hypotheses") or []
    others = [h for h in hypotheses if h.get("id") != selected_id]
    if not others:
        return ("The diagnosis ranked only one cause; there is nothing else "
                "to reconcile it against.")

    rows = "\n".join(
        f"- **{h.get('claim')}** (`{h.get('layer')}`, attributed "
        f"{h.get('attributed_ms')} ms)"
        for h in others
    )
    all_total = sum(h.get("attributed_ms", 0.0) for h in hypotheses)
    gap_pct = (abs(delivered_ms - all_total) / delivered_ms * 100.0
              if delivered_ms else 0.0)
    if gap_pct <= 15.0:
        verdict = (
            f"Summed across all {len(hypotheses)} ranked causes, the "
            f"diagnosis attributed {all_total:.1f} ms total — within "
            f"{gap_pct:.1f}% of the {delivered_ms:.1f} ms actually "
            f"delivered by a patch that only targeted one of them. That "
            f"close a match is the signal that the ranking attributed one "
            f"underlying mechanism's downstream effects to separate "
            f"hypotheses, rather than finding three independent costs."
        )
    else:
        verdict = (
            f"Summed across all {len(hypotheses)} ranked causes, the "
            f"diagnosis attributed {all_total:.1f} ms total, "
            f"{gap_pct:.1f}% away from the {delivered_ms:.1f} ms actually "
            f"delivered — not close enough to credit the other hypotheses' "
            f"cost to this fix; some of the delivered improvement is "
            f"unexplained by the diagnosis."
        )
    return (f"The diagnosis ranked {len(others)} other cause(s) as separate "
            f"from the one this patch targeted:\n\n{rows}\n\n{verdict}")


def _pre_existing_rows(uncertainties: list[str]) -> str:
    """What the comprehension phase was unsure about, or flagged outright.

    Everything here came from `completeness_warnings()` or the model's own
    stated uncertainty — never invented for the report. A correctness hazard
    found while diagnosing performance (a missing ORDER BY, for one) belongs
    here: it is a property of the endpoint today, not of this patch.
    """
    if not uncertainties:
        return "- (none recorded)"
    return "\n".join(f"- {item}" for item in uncertainties)


def _selected_candidate(plan_payload: dict[str, Any]) -> dict[str, Any]:
    chosen = plan_payload.get("selected_id")
    for candidate in plan_payload.get("candidates") or []:
        if candidate.get("id") == chosen:
            return candidate
    return {}


def _ranked_rows(plan_payload: dict[str, Any]) -> str:
    """The expected-value table, with the fixability reason for every row.

    This is the section that makes the PR reviewable: a reader can see that
    the largest measured cost was not attacked, and exactly why not.
    """
    rows = []
    for entry in plan_payload.get("ranked") or []:
        fix = entry.get("fixability") or {}
        classified = "" if fix.get("classified", True) else " *(not classified)*"
        rows.append(
            f"- **{entry.get('claim')}**{classified}  \n"
            f"  `{entry.get('layer')}` · attributed {entry.get('attributed_ms')} ms "
            f"× fixability {fix.get('fixability')} = "
            f"expected value {entry.get('expected_value_ms')} ms  \n"
            f"  _{fix.get('reason')}_"
            + (f"  \n  **Blocked by:** {fix.get('blocked_by')}"
               if fix.get("blocked_by") else "")
        )
    return "\n".join(rows) or "- (no ranking recorded)"


def _rejected_rows(plan_payload: dict[str, Any]) -> str:
    rows = [f"- `{r.get('candidate_id')}` — {r.get('reason')}"
            for r in plan_payload.get("not_selected") or []]
    return "\n".join(rows) or "- (the plan produced a single candidate)"


def _index_evidence(gated: dict[str, Any], selected: dict[str, Any]) -> str:
    """The migration in the diff, verbatim, or a statement that there is none.

    The migration file carries its own `-- Evidence:` line, written from the
    `index_benefit` measurement, so quoting the file quotes the measurement.
    """
    diff = gated["patch"].diff
    migrations = [line[len("+++ b/"):].strip() for line in diff.splitlines()
                  if line.startswith("+++ b/")
                  and "db/migration" in line]
    if not migrations:
        return ("No schema change: this patch is application code only, so "
                "there is no `EXPLAIN` plan to compare.")

    added = [line[1:] for line in diff.splitlines()
             if line.startswith("+") and not line.startswith("+++")
             and ("CREATE INDEX" in line.upper() or line.strip().startswith("--"))]
    body = "\n".join(added)
    return (f"This patch adds a migration: `{', '.join(migrations)}`\n\n"
            f"```sql\n{body}\n```\n\n"
            f"The `Evidence:` comment above is the `index_benefit` "
            f"measurement — rows examined before and after, taken by building "
            f"the index, re-running `EXPLAIN ANALYZE`, and dropping it again.")


def write_report(
    gated: dict[str, Any], *, terminal: TerminalState, client: Any = None
) -> tuple[str, Path, dict[str, Any]]:
    """Render report.md. One LLM call, for the root-cause paragraph only.

    `terminal` must be the state this run actually reached (the caller
    computes it, since only the caller knows dry-run vs. a real push) — it is
    not re-derived here, so the rendered body can never drift from it.
    """
    ledger, _ = build_ledger(gated, terminal)

    narrative, narrative_trace = write_narrative(
        top_hypothesis(gated["evidence"]["payloads"]["hypotheses"]),
        gated["plan"],
        client=client,
    )
    # The paragraph is prose, so its "artifact" is the diagnosis it explains.
    ledger.record(
        "narrative", narrative, gated["evidence"]["refs"]["hypotheses"]
    )

    kept_rows = []
    duplicate_rows = []
    for entry in ledger.provenance():
        line = (f"- `{entry['claim']}` = {entry['value']} "
                f"(`{entry['artifact']}`, hash `{entry['hash']}`)")
        if entry["claim"] in _PROVENANCE_DUPLICATES:
            duplicate_rows.append(
                f"- `{entry['claim']}` -> `{entry['artifact']}`, "
                f"hash `{entry['hash']}`")
        else:
            kept_rows.append(line)

    provenance_block = "\n".join([
        "<details>",
        "<summary>Provenance (every number, with its artifact and hash)</summary>",
        "",
        "Every number above is templated from a recorded artifact; the "
        "renderer refuses a slot with no `ArtifactRef` behind it and "
        "refuses a template containing a bare numeral.",
        "",
        *kept_rows,
        "",
        "**Rendered elsewhere above as a table or list, not repeated here "
        "— hash kept for provenance:**",
        "",
        *duplicate_rows,
        "",
        "</details>",
    ])
    ledger.record(
        "provenance_rows", provenance_block,
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


def _check_worktree_matches_patch(
    green_path: Path, branch: str, base_commit: str, files_changed: list[str]
) -> None:
    """Refuse to ship if the worktree is not what the patch artifact claims.

    The gate verified the PATCH; it never looked at the WORKTREE. Those can
    drift apart — a stale checkout, a `--green` pointed at the wrong path, a
    patch artifact left over from a different run — and if they have, pushing
    whatever happens to be sitting in the worktree under the patch's claimed
    branch name ships something nobody gated. This is an environment mismatch,
    not a broken artifact, so it is a ToolError, not a ContractViolation.
    """
    actual_branch = _git(green_path, "rev-parse", "--abbrev-ref", "HEAD").strip()
    if actual_branch != branch:
        raise ToolError(
            f"worktree branch mismatch at {green_path}: patch.branch claims "
            f"'{branch}', but the worktree is actually on '{actual_branch}'. "
            f"Check out the right branch, or point --green at the right "
            f"worktree, before shipping."
        )

    # `git diff` alone misses a brand-new file that was never `git add`ed --
    # exactly the state a freshly-written migration is in before ship commits
    # it. Union tracked changes against base with untracked-but-not-ignored
    # files so a new file the patch added is actually visible to this check.
    diff_output = _git(green_path, "diff", "--name-only", base_commit)
    tracked_changed = {line for line in diff_output.splitlines() if line.strip()}
    untracked_output = _git(green_path, "ls-files", "--others", "--exclude-standard")
    untracked = {line for line in untracked_output.splitlines() if line.strip()}
    actual_files = tracked_changed | untracked
    claimed_files = set(files_changed)
    if actual_files != claimed_files:
        missing = sorted(claimed_files - actual_files)
        extra = sorted(actual_files - claimed_files)
        problems = []
        if missing:
            problems.append(f"claimed by patch.files_changed but not actually "
                            f"touched: {', '.join(missing)}")
        if extra:
            problems.append(f"actually touched but not in patch.files_changed: "
                            f"{', '.join(extra)}")
        raise ToolError(
            f"worktree content mismatch at {green_path}: patch.files_changed "
            f"claims {sorted(claimed_files)} against base {base_commit}, but "
            f"the worktree's actual diff touches {sorted(actual_files)}. "
            + "; ".join(problems) + "."
        )


def open_pull_request(
    green_path: Path, branch: str, title: str, body: str, *,
    base_commit: str, files_changed: list[str], dry_run: bool = False,
) -> dict[str, Any]:
    """Commit, push, `gh pr create`. Only reached when the gate passed."""
    _check_worktree_matches_patch(green_path, branch, base_commit, files_changed)

    _git(green_path, "add", "-A")
    status = _git(green_path, "status", "--porcelain")
    if status.strip():
        _git(
            green_path,
            "-c", "user.email=senapathy@local", "-c", "user.name=senapathy",
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
    green_path: Path = Path(".senapathy/green"),
    client: Any = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Gate, report, and either open a PR or refuse — honestly, either way."""
    gated = run_gate(run_dir)
    decision: GateDecision = gated["decision"]

    # Computed ONCE, before the report is built, so the rendered body carries
    # the state this run actually reached: READY_TO_SHIP for a dry run,
    # PR_OPENED for a real push, the failure state otherwise. `shipped` here
    # only matters when decision.passed (terminal_state_for ignores it
    # otherwise) — true exactly when this is not a dry run, since a passing,
    # non-dry-run decision is shipped by the open_pull_request call below.
    terminal = terminal_state_for(decision, shipped=not dry_run)

    body, report_path, narrative_trace = write_report(
        gated, client=client, terminal=terminal
    )

    shipped: dict[str, Any] | None = None
    if decision.passed:
        base_commit = gated["evidence"]["payloads"]["patch"]["worktree"]["base_commit"]
        shipped = open_pull_request(
            Path(green_path), gated["patch"].branch,
            title=f"perf: {gated['plan'].description[:60]}",
            body=body,
            base_commit=base_commit,
            files_changed=gated["patch"].files_changed,
            dry_run=dry_run,
        )
    # else: no PR. The report is still written, with the state the evidence
    # supports — a refusal nobody records is a refusal that gets repeated.

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
