"""Tier 1: reasoning replay. Fast, offline, no database.

For each scenario under eval/scenarios/<ID>/, replays the diagnose and plan
phases against that scenario's frozen baseline/endpoint_model, with its own
cassette directory standing in for the LLM (offline=True -- a cassette miss
raises, it never falls through to a live call) and its own tool_responses.json
standing in for anything that needed a live app or database the first time
(run_probe, explain, list_indexes, index_benefit, query_equivalent,
get_schema). A requested args set not in that frozen table also raises --
replay never falls through to a live call there either. Everything else --
emit_hypotheses's validation, submit_plan's validation, read_file/
search_code/list_dir/find_route -- is the real, current code, run against
mini-spring checked out at the scenario's own git tag (read-only, extracted
once into eval/_checkouts/ and cached; nothing here writes to it).

This is the tier that would have caught: a mechanism classified from
description text instead of the typed field (COL-06), a diagnosis that
ignores the measured SQL share of latency, and a hypothesis that falls
through the fixability table unclassified on a slow query. None of those
need a running app or a seeded database to catch -- they are bugs in how
this project's OWN code reasons about already-recorded evidence.

Usage:
    .venv/bin/python eval/tier1.py                 # every scenario
    .venv/bin/python eval/tier1.py INV-04 PAY-03    # just these
"""

from __future__ import annotations

import json
import sys
import time
from datetime import date
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from senapathy.agent import LoopBudget, run_agent  # noqa: E402
from senapathy.contracts import EndpointModel, HypothesisSet  # noqa: E402
from senapathy.errors import ToolError  # noqa: E402

from _checkout import CheckoutError, ensure_checkout  # noqa: E402

SCENARIOS_DIR = Path(__file__).resolve().parent / "scenarios"
MINI_SPRING_REPO = Path("/Users/akashm/Documents/mini-spring")
LIVE_TOOLS = ("explain", "list_indexes", "index_benefit", "query_equivalent",
             "get_schema")


class ScenarioFailure(Exception):
    """A named, specific reason one scenario failed -- never a bare traceback."""


# ────────────────────────────  frozen tool calls  ─────────────────────────


def load_tool_responses(scenario_dir: Path) -> dict[str, list[dict[str, Any]]]:
    return json.loads((scenario_dir / "tool_responses.json").read_text())


def make_frozen_stub(entries: list[dict[str, Any]], tool_name: str):
    """Exact (tool, args) -> recorded result. Never falls through to a live
    call: an args set not seen during freezing is a replay drift, not a
    reason to guess."""
    table = {
        json.dumps(e["args"], sort_keys=True): e["result"]
        for e in entries if e["tool"] == tool_name
    }

    def handler(**kwargs: Any) -> str:
        key = json.dumps(kwargs, sort_keys=True)
        if key not in table:
            raise ToolError(
                f"no frozen {tool_name} response for args {kwargs} -- replay "
                f"has drifted from what this scenario froze. Recorded arg "
                f"sets: {sorted(table)}"
            )
        return table[key]

    return handler


def tools_from_cassettes(cassette_dir: Path) -> list[dict[str, Any]]:
    """The exact tool schema list a conversation was offered, read straight
    out of its own smallest (first-turn) cassette file."""
    smallest: dict[str, Any] | None = None
    smallest_n = None
    for path in cassette_dir.glob("*.json"):
        doc = json.loads(path.read_text(encoding="utf-8"))
        n = len(doc["request"]["messages"])
        if smallest_n is None or n < smallest_n:
            smallest, smallest_n = doc, n
    if smallest is None:
        raise ScenarioFailure(f"no cassette files under {cassette_dir}")
    return smallest["request"].get("tools") or []


# ──────────────────────────────  the replay  ──────────────────────────────


def replay_diagnose(scenario_dir: Path) -> HypothesisSet:
    from senapathy.contracts import RequestCorpus
    from senapathy.diagnose.agent import build_task
    from senapathy.diagnose.probe import ProbeCatalog, ProbeLog
    from senapathy.diagnose.validate import validate_hypothesis_set
    from senapathy.llm import make_client

    baseline = json.loads((scenario_dir / "baseline.json").read_text())
    endpoint_model_raw = json.loads((scenario_dir / "endpoint_model.json").read_text())
    corpus = RequestCorpus.model_validate(
        json.loads((scenario_dir / "corpus.json").read_text()))
    p50 = baseline["benchmark"]["latency"]["p50_ms"]
    sql_share = (baseline.get("derived") or {}).get("sql_share_of_latency_pct")

    # Must match diagnose/agent.py::run_diagnosis exactly (same base_url, same
    # endpoint_model/corpus) -- the catalog's own description text goes into
    # the prompt build_task renders, so a mismatch here is a cassette miss on
    # the very first request, not a subtle one.
    catalog = ProbeCatalog("http://app", endpoint_model_raw, corpus)

    tool_responses = load_tool_responses(scenario_dir)
    frozen_probe = make_frozen_stub(tool_responses["diagnose"], "run_probe")
    probe_log = ProbeLog()

    def run_probe(**kwargs: Any) -> str:
        result = frozen_probe(**kwargs)
        probe_log.record(kwargs, result)
        return result

    accepted: dict[str, HypothesisSet] = {}
    rejections: list[str] = []

    def emit_hypotheses(**payload: Any) -> str:
        try:
            result = validate_hypothesis_set(
                payload, p50, probe_log, sql_share_of_latency_pct=sql_share)
        except ToolError as exc:
            rejections.append(str(exc))
            raise
        accepted["set"] = result
        return (f"accepted: {len(result.hypotheses)} hypotheses across "
               f"{len({h.layer for h in result.hypotheses})} layers, "
               f"{len(result.probes_run)} probes. You are done — stop now.")

    cassette_dir = scenario_dir / "cassettes" / "diagnose"
    client = make_client(cassette_dir=cassette_dir, offline=True)
    outcome = run_agent(
        client=client,
        task=build_task(baseline, endpoint_model_raw, catalog=catalog),
        tools=tools_from_cassettes(cassette_dir),
        handlers={"run_probe": run_probe, "emit_hypotheses": emit_hypotheses},
        budget=LoopBudget(max_steps=30),
    )
    if "set" not in accepted:
        detail = f"; last rejection: {rejections[-1]}" if rejections else ""
        raise ScenarioFailure(
            f"diagnose replay never reached an accepted diagnosis "
            f"(stopped: {outcome.stopped_because}){detail}"
        )
    return accepted["set"]


def _verification_key(*, sql_before: str, sql_after: str,
                      params: list[str] | None, mechanism: str | None) -> str:
    return json.dumps({"sql_before": sql_before, "sql_after": sql_after,
                       "params": params or [], "mechanism": mechanism},
                      sort_keys=True)


def frozen_sql_verification_table(scenario_dir: Path) -> dict[str, dict[str, Any]]:
    """Every candidate's sql_verification this scenario's own `plan.json`
    already carries -- computed with a LIVE database when live_freeze.py ran,
    the one time this scenario's EXPLAIN ANALYZE claim could actually be
    checked. Tier 1 has no database (fast, offline, free) and so can never
    run that check itself; replaying it against this frozen record, instead
    of silently reporting "not checked" on every run, is what gives
    min_rows_examined_improvement_pct real teeth without needing one."""
    path = scenario_dir / "plan.json"
    if not path.exists():
        return {}
    plan = json.loads(path.read_text())
    table: dict[str, dict[str, Any]] = {}
    for candidate in plan.get("candidates") or []:
        verification = candidate.get("sql_verification")
        if not verification:
            continue
        key = _verification_key(
            sql_before=candidate.get("sql_before") or "",
            sql_after=candidate.get("sql_after") or "",
            # submit.py calls verify_sql_candidate with `sql_params or None`
            # (collapsing an empty list) -- match that exactly, or an empty
            # `sql_params: []` here (the field's own default) never matches
            # the `params=None` the real call actually made at freeze time.
            params=candidate.get("sql_params") or None,
            mechanism=candidate.get("mechanism"),
        )
        table[key] = verification
    return table


def make_frozen_verify_sql_candidate(scenario_dir: Path):
    """A drop-in for plan/risk.py::verify_sql_candidate that answers from
    this scenario's own frozen plan.json instead of a live EXPLAIN ANALYZE.

    Falls through to the real function (which degrades to verified=False
    with no live `box`) when replay drifted to a sql_before/sql_after this
    scenario never recorded -- a replay drift is reported as "not checked",
    same as always, not invented."""
    from senapathy.contracts import SqlVerification
    from senapathy.plan.risk import verify_sql_candidate as real_verify

    table = frozen_sql_verification_table(scenario_dir)

    def stub(box: Any, *, sql_before: str, sql_after: str,
            params: list[str] | None, mechanism: str | None) -> Any:
        key = _verification_key(sql_before=sql_before, sql_after=sql_after,
                                params=params, mechanism=mechanism)
        frozen = table.get(key)
        if frozen is None:
            return real_verify(box, sql_before=sql_before, sql_after=sql_after,
                              params=params, mechanism=mechanism)
        return SqlVerification.model_validate(frozen)

    return stub


def replay_plan(scenario_dir: Path, repo: Path, hypotheses: HypothesisSet) -> Any:
    from unittest.mock import patch

    from senapathy.contracts import SqlDigestStat
    from senapathy.ids import content_hash
    from senapathy.llm import make_client
    from senapathy.plan.agent import DEFAULT_STACK, TOP_N, _unbounded_summary
    from senapathy.plan.brief import render_brief
    from senapathy.plan.fixability import rank_hypotheses
    from senapathy.plan.submit import make_submit_plan
    from senapathy.prompts import load_prompt
    from senapathy.tools import SourceRoot
    from senapathy.tools.routes import route_tools
    from senapathy.tools.source import source_tools

    endpoint_model = EndpointModel.model_validate(
        json.loads((scenario_dir / "endpoint_model.json").read_text()))
    root = SourceRoot.of(repo)

    ranked = rank_hypotheses(hypotheses, endpoint_model)
    considered = [r.hypothesis_id for r in ranked[:TOP_N]]
    by_id = {h.id: h for h in hypotheses.hypotheses}

    # Must match plan.py::_load_baseline_digests exactly, or the brief this
    # replay builds is byte-different from the one live_freeze.py recorded
    # and every cassette request hash misses -- LOOP-01 is why this is here,
    # not a copy for its own sake.
    baseline_path = scenario_dir / "baseline.json"
    baseline_digests = (
        [SqlDigestStat.model_validate(d)
         for d in json.loads(baseline_path.read_text()).get("sql_digests") or []]
        if baseline_path.exists() else []
    )

    system = load_prompt(
        "plan.system", unbounded_collections=_unbounded_summary(endpoint_model))
    user = load_prompt(
        "plan.user", repo_root=str(root), stack=DEFAULT_STACK,
        endpoint=endpoint_model.route,
        measured_total_ms=f"{hypotheses.measured_total_ms:.1f}",
        brief=render_brief(ranked=ranked, hypotheses=by_id,
                           considered=considered, model=endpoint_model,
                           baseline_digests=baseline_digests),
    )
    prompt_hash = content_hash({"system": system.hash, "user": user.hash})

    sink: dict[str, Any] = {}
    submit_plan = make_submit_plan(
        root=root, model=endpoint_model, ranked=ranked, considered=considered,
        sink=sink, endpoint=endpoint_model.route,
        measured_total_ms=hypotheses.measured_total_ms, prompt_hash=prompt_hash,
        baseline_digests=baseline_digests,
    )

    tool_responses = load_tool_responses(scenario_dir)
    handlers = {spec.name: spec.handler
               for spec in [*route_tools(root), *source_tools(root)]}
    for name in LIVE_TOOLS:
        handlers[name] = make_frozen_stub(tool_responses["plan"], name)
    handlers["submit_plan"] = submit_plan

    cassette_dir = scenario_dir / "cassettes" / "plan"
    client = make_client(cassette_dir=cassette_dir, offline=True)
    with patch("senapathy.plan.submit.verify_sql_candidate",
              make_frozen_verify_sql_candidate(scenario_dir)):
        outcome = run_agent(
            client=client, task=user.text, system=system.text,
            tools=tools_from_cassettes(cassette_dir), handlers=handlers,
            terminal_tool="submit_plan", budget=LoopBudget(max_steps=30),
        )
    if "plan" not in sink:
        raise ScenarioFailure(
            f"plan replay never reached an accepted plan "
            f"(stopped: {outcome.stopped_because})"
        )
    return sink["plan"]


# ───────────────────────────  the assertions  ─────────────────────────────
#
# Outcomes are gated; methods are recorded. `approach` is HOW a candidate gets
# there — four different approaches (dto_projection, fetch_join, entity_graph,
# batch_query) are all correct fixes for INV-04's N+1, and re-sampling the
# live conversation during a re-freeze drew three different ones across three
# tries. Asserting a specific approach meant widening `approach_any_of` every
# time the model drew a new (still correct) one — which is how an eval ends
# up asserting nothing. `mechanism` is WHAT moves, which is the actual claim
# a scenario makes about the fix, and it does not vary with which approach
# was used to get there. Where the plan phase's own EXPLAIN ANALYZE
# verification (plan/risk.py::verify_sql_candidate) succeeded and applies
# (`scored=True`, i.e. the rows_examined mechanism), the MEASURED delta is
# asserted too — not just that a verification object exists, but that it
# shows the improvement this scenario claims. That measurement can fail to
# run for reasons unrelated to the scenario (a parameter-binding quirk in a
# native LIMIT/OFFSET rewrite, say) — when that happens it is reported, not
# silently treated as a pass, but it does not fail the scenario either: an
# unmeasured claim is a gap in coverage, not evidence the fix broke.


def check_expectations(expected: dict[str, Any], hypotheses: HypothesisSet,
                       plan: Any) -> tuple[list[str], list[str]]:
    """Returns (problems, context) — context is informational, never gates."""
    problems: list[str] = []
    context: list[str] = []
    layers_present = {h.layer.value for h in hypotheses.hypotheses}

    for required in expected.get("required_layers") or []:
        if required not in layers_present:
            problems.append(
                f"required layer {required!r} is not present (got "
                f"{sorted(layers_present)})")
    for forbidden in expected.get("forbidden_layers") or []:
        if forbidden in layers_present:
            problems.append(f"forbidden layer {forbidden!r} is present")

    dominant_any_of = expected.get("dominant_layer_any_of")
    if dominant_any_of is not None:
        dominant = max(hypotheses.hypotheses, key=lambda h: h.attributed_ms)
        if dominant.layer.value not in dominant_any_of:
            problems.append(
                f"dominant hypothesis is {dominant.id} ({dominant.layer.value}, "
                f"{dominant.attributed_ms} ms); expected one of {dominant_any_of}")

    plan_expected = expected.get("plan") or {}
    if plan_expected.get("selectable"):
        if plan.selected_id is None:
            problems.append(
                f"expected a selectable candidate; none was selected "
                f"({plan.selection_reason})")
        else:
            selected = plan.selected()
            context.append(f"approach={selected.approach}")

            mechanism_any_of = plan_expected.get("mechanism_any_of")
            if mechanism_any_of and selected.mechanism not in mechanism_any_of:
                problems.append(
                    f"selected mechanism {selected.mechanism!r} not in "
                    f"{mechanism_any_of}")
            else:
                context.append(f"mechanism={selected.mechanism}")

            # approach_context is recorded for a human reading the scenario,
            # never checked: see the module note above.
            approach_context = plan_expected.get("approach_context")
            if approach_context and selected.approach not in approach_context:
                context.append(
                    f"NOTE: approach {selected.approach!r} is not in this "
                    f"scenario's recorded family {approach_context} — still "
                    f"not a failure, but worth a human's eyes if it keeps "
                    f"happening")

            # The actual outcome some scenarios test for is SCOPE, not
            # approach — STM-01 exists to prove derive_scope's
            # repeated_callers() expansion reaches every sibling call site,
            # regardless of which approach label ends up on the winning
            # candidate.
            must_touch = plan_expected.get("files_to_touch_contains") or []
            missing = [f for f in must_touch if f not in selected.files_to_touch]
            if missing:
                problems.append(
                    f"selected candidate's files_to_touch is missing "
                    f"{missing} (got {selected.files_to_touch})")

            min_pct = plan_expected.get("min_rows_examined_improvement_pct")
            verification = selected.sql_verification
            if min_pct is not None:
                if verification is None:
                    problems.append(
                        "min_rows_examined_improvement_pct is set but the "
                        "selected candidate carries no sql_verification at "
                        "all (approach is not SQL-rewrite-shaped, or "
                        "mechanism is not instrumented)")
                elif not verification.verified or not verification.scored:
                    context.append(
                        f"measured effect NOT checked this run: {verification.note}")
                elif verification.rows_delta_pct > -min_pct:
                    problems.append(
                        f"EXPLAIN ANALYZE measured only "
                        f"{verification.rows_delta_pct:+.1f}% on rows "
                        f"examined; this scenario claims at least "
                        f"{min_pct:.0f}% ({verification.note})")
                else:
                    context.append(
                        f"measured rows_examined {verification.rows_delta_pct:+.1f}% "
                        f"(claim: at least {min_pct:.0f}%)")
    else:
        if plan.selected_id is not None:
            problems.append(
                f"expected NO selectable candidate; {plan.selected_id} was "
                f"selected")
        contains = plan_expected.get("not_selectable_because_contains")
        if contains:
            reasons = [c.not_selectable_because or "" for c in plan.candidates]
            if not any(contains in r for r in reasons):
                problems.append(
                    f"no candidate's not_selectable_because contains "
                    f"{contains!r} (got {reasons})")
    return problems, context


# ───────────────────────────────  quarantine  ──────────────────────────────
#
# A scenario that stays red forever is not a signal, it's wallpaper -- "4/5"
# gets read as "basically green" and a real regression hides in the one
# permanently-red row. Quarantining makes the exclusion EXPLICIT: a reason, a
# date, out of the pass/fail denominator -- and it cannot go stale silently,
# because the scenario is still replayed every run. If it starts passing, the
# table says so loudly instead of staying quarantined by inertia.


def load_quarantine(scenario_dir: Path) -> dict[str, Any] | None:
    path = scenario_dir / "quarantine.json"
    if not path.exists():
        return None
    doc = json.loads(path.read_text())
    if "since" not in doc or "reason" not in doc:
        raise SystemExit(
            f"{path} is missing 'since' or 'reason' -- a quarantine with no "
            f"date or no reason is exactly the silent-permanent case this "
            f"exists to prevent.")
    return doc


def days_since(since: str) -> int:
    return (date.today() - date.fromisoformat(since)).days


# ───────────────────────────────  the table  ──────────────────────────────


def run_scenario(scenario_id: str) -> tuple[bool, str, float]:
    scenario_dir = SCENARIOS_DIR / scenario_id
    started = time.monotonic()
    try:
        meta = json.loads((scenario_dir / "meta.json").read_text())
        expected = json.loads((scenario_dir / "expected.json").read_text())
        try:
            repo = ensure_checkout(
                MINI_SPRING_REPO, meta["mini_spring_tag"], meta["mini_spring_commit"])
        except CheckoutError as exc:
            raise ScenarioFailure(str(exc)) from exc
        hypotheses = replay_diagnose(scenario_dir)
        plan = replay_plan(scenario_dir, repo, hypotheses)
        problems, context = check_expectations(expected, hypotheses, plan)
        elapsed = time.monotonic() - started
        if problems:
            detail = "; ".join(problems)
            if context:
                detail += " [" + ", ".join(context) + "]"
            return False, detail, elapsed
        detail = "ok" + (" (" + ", ".join(context) + ")" if context else "")
        return True, detail, elapsed
    except ScenarioFailure as exc:
        return False, str(exc), time.monotonic() - started
    except Exception as exc:  # noqa: BLE001 -- report in the table, don't crash it
        return False, f"{type(exc).__name__}: {exc}", time.monotonic() - started


def main() -> None:
    ids = sys.argv[1:] or sorted(p.name for p in SCENARIOS_DIR.iterdir() if p.is_dir())
    if not ids:
        raise SystemExit(f"no scenarios under {SCENARIOS_DIR}")

    results = []
    for scenario_id in ids:
        print(f"running {scenario_id}...", file=sys.stderr)
        quarantine = load_quarantine(SCENARIOS_DIR / scenario_id)
        passed, detail, elapsed = run_scenario(scenario_id)
        results.append((scenario_id, passed, detail, elapsed, quarantine))

    width = max(len(r[0]) for r in results)
    print(f"\n{'SCENARIO':<{width}}  STATUS       TIME    DETAIL")
    for scenario_id, passed, detail, elapsed, quarantine in results:
        if quarantine is None:
            mark = "PASS" if passed else "FAIL"
        elif passed:
            mark = "Q-PASSING"   # quarantined but replaying clean -- lift it
        else:
            mark = "QUARANTINED"
        print(f"{scenario_id:<{width}}  {mark:<11}  {elapsed:5.1f}s  {detail}")
        if quarantine is not None:
            age = days_since(quarantine["since"])
            flag = (" <-- now passing; remove quarantine.json"
                   if passed else "")
            print(f"{'':<{width}}    since {quarantine['since']} "
                 f"({age} day{'s' if age != 1 else ''} ago): "
                 f"{quarantine['reason']}{flag}")

    core = [r for r in results if r[4] is None]
    quarantined = [r for r in results if r[4] is not None]
    core_failed = [r for r in core if not r[1]]
    total = sum(r[3] for r in results)

    summary = f"\n{len(core) - len(core_failed)}/{len(core)} passed"
    if quarantined:
        summary += f", {len(quarantined)} quarantined"
    summary += f" in {total:.1f}s total."
    print(summary)

    newly_passing = [r for r in quarantined if r[1]]
    if newly_passing:
        print("QUARANTINED SCENARIO(S) NOW PASSING -- "
             "remove their quarantine.json: "
             + ", ".join(r[0] for r in newly_passing))

    # A quarantine excuses a scenario from the pass/fail count; it does not
    # excuse it from ever being looked at again. Exit non-zero here too, so
    # "it's just quarantined" cannot be confused with "nothing to see" --
    # the fix is to lift the quarantine, not to keep re-running past it.
    if core_failed or newly_passing:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
