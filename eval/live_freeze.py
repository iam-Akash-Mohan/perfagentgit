"""Freeze one scenario by running diagnose+plan live, once, against current code.

Why live and not cassette archaeology: this project's tool schemas and
prompt-building code (fixability.py, submit.py's mechanism field, brief
rendering) keep changing, so a historical conversation recorded months or
even hours ago does not replay against today's code -- the request bodies no
longer hash the same. Re-running once, for real, against the frozen FACTS
(endpoint_model.json / baseline.json -- properties of the measured system,
not of any one conversation) produces a cassette set that is guaranteed
self-consistent with the code that just created it. That is what tier1.py
replays afterwards, offline, forever.

Needs, ONE TIME, for this script only: mini-spring checked out at the
scenario's own eval/<ID> tag and running on :18080, and MySQL reachable (the
SENAPATHY_MYSQL_* vars). Tier 1 itself needs neither.

Usage:
    .venv/bin/python eval/live_freeze.py PAY-03 --run-dir .senapathy/runs/PAY-03 \
        --endpoint "GET /api/payments/recent"
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from senapathy.agent import LoopBudget  # noqa: E402
from senapathy.contracts import EndpointModel  # noqa: E402
from senapathy.diagnose.agent import run_diagnosis  # noqa: E402
from senapathy.llm import make_client  # noqa: E402
from senapathy.measure.sql import MySQLTarget  # noqa: E402
from senapathy.plan.agent import plan as run_plan  # noqa: E402
from senapathy.plan.agent import write_plan  # noqa: E402
from senapathy.trace import Trace  # noqa: E402

from _extract import extract_tool_calls  # noqa: E402
from _checkout import ensure_checkout  # noqa: E402

SCENARIOS_DIR = Path(__file__).resolve().parent / "scenarios"
SCRATCH_DIR = Path(__file__).resolve().parent / "scratch"
LIVE_TOOLS = {"run_probe", "explain", "list_indexes", "index_benefit",
             "query_equivalent", "get_schema"}


def _longest_cassette(cassette_dir: Path) -> dict[str, Any]:
    best: dict[str, Any] | None = None
    best_len = -1
    for path in cassette_dir.glob("*.json"):
        doc = json.loads(path.read_text(encoding="utf-8"))
        n = len(doc["request"]["messages"])
        if n > best_len:
            best, best_len = doc, n
    if best is None:
        raise SystemExit(f"no cassette was written to {cassette_dir}")
    return best


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scenario_id")
    parser.add_argument("--run-dir", required=True, type=Path,
                        help="The already-completed run this scenario freezes.")
    parser.add_argument("--repo", default=Path("/Users/akashm/Documents/mini-spring"),
                        type=Path)
    parser.add_argument("--base-url", default="http://localhost:18080")
    args = parser.parse_args()

    scenario_dir = SCENARIOS_DIR / args.scenario_id
    scratch_dir = SCRATCH_DIR / args.scenario_id
    if scratch_dir.exists():
        shutil.rmtree(scratch_dir)
    scratch_dir.mkdir(parents=True)
    # sql_audit.json is OPTIONAL -- a run frozen before the audit phase
    # existed has none, and diagnose/plan both already treat its absence as
    # "nothing to reconcile against" rather than an error. Carried the same
    # way as the other three evidence files: a property of the measured
    # system, re-used as-is rather than re-measured by this script.
    names = ["baseline.json", "endpoint_model.json", "corpus.json"]
    if (args.run_dir / "sql_audit.json").exists():
        names.append("sql_audit.json")
    for name in names:
        shutil.copy2(args.run_dir / name, scratch_dir / name)

    diagnose_cassettes = scenario_dir / "cassettes" / "diagnose"
    plan_cassettes = scenario_dir / "cassettes" / "plan"
    for d in (diagnose_cassettes, plan_cassettes):
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)

    trace = Trace(scratch_dir, args.scenario_id)
    mysql = MySQLTarget.from_env()

    print("running diagnose, live...")
    diag_client = make_client(cassette_dir=diagnose_cassettes, offline=False)
    hypothesis_set, probe_log, hyp_path = run_diagnosis(
        base_url=args.base_url,
        run_dir=scratch_dir,
        endpoint_model_path=scratch_dir / "endpoint_model.json",
        mysql=mysql,
        budget=LoopBudget(max_steps=30),
        client=diag_client,
        trace=trace,
    )
    print(f"  diagnose done: {len(hypothesis_set.hypotheses)} hypotheses, "
         f"{len(probe_log)} probe(s) -> {hyp_path}")
    for h in hypothesis_set.hypotheses:
        print(f"    {h.id:4} {h.layer:20} {h.attributed_ms:7.1f} ms  {h.claim[:70]}")

    endpoint_model = EndpointModel.model_validate(
        json.loads((scratch_dir / "endpoint_model.json").read_text()))

    # The SAME stable path tier1.py will read the plan prompt's `repo_root:`
    # line from later -- not the live working copy's own path, which moves
    # between branches and would make this scenario replay against a
    # different hash than the one just recorded here.
    pinned_repo = ensure_checkout(
        args.repo, f"eval/{args.scenario_id}", endpoint_model.base_commit)

    print("running plan, live...")
    plan_client = make_client(cassette_dir=plan_cassettes, offline=False)
    plan_result = run_plan(
        repo=pinned_repo,
        endpoint=endpoint_model.route,
        hypotheses=hypothesis_set,
        model=endpoint_model,
        client=plan_client,
        trace=trace,
        budget=LoopBudget(max_steps=30),
        run_dir=scratch_dir,
        mysql=mysql,
    )
    plan_path = write_plan(plan_result, scratch_dir)
    print(f"  plan done: selected={plan_result.selected_id} "
         f"({plan_result.selection_reason}) -> {plan_path}")
    for c in plan_result.candidates:
        print(f"    {c.id:4} {c.approach:20} mechanism={c.mechanism} "
             f"not_selectable={c.not_selectable_because}")

    tool_responses = {
        "diagnose": [c for c in extract_tool_calls(
            _longest_cassette(diagnose_cassettes)["request"]["messages"])
            if c["tool"] in LIVE_TOOLS],
        "plan": [c for c in extract_tool_calls(
            _longest_cassette(plan_cassettes)["request"]["messages"])
            if c["tool"] in LIVE_TOOLS],
    }
    (scenario_dir / "tool_responses.json").write_text(
        json.dumps(tool_responses, indent=2, ensure_ascii=False, sort_keys=True) + "\n")
    print(f"froze {len(tool_responses['diagnose'])} diagnose + "
         f"{len(tool_responses['plan'])} plan live-tool response(s)")

    for name in names:   # whatever was actually carried into scratch_dir above
        shutil.copy2(args.run_dir / name, scenario_dir / name)
    shutil.copy2(hyp_path, scenario_dir / "hypotheses.json")
    shutil.copy2(plan_path, scenario_dir / "plan.json")

    meta = {
        "scenario": args.scenario_id,
        "endpoint": endpoint_model.route,
        "mini_spring_commit": endpoint_model.base_commit,
        "mini_spring_tag": f"eval/{args.scenario_id}",
    }
    (scenario_dir / "meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False) + "\n")
    print(f"wrote meta.json: {meta}")
    print("\nNOTE: write expected.json by hand from the hypotheses.json/"
         "plan.json just written above, after checking they say what you "
         "expect this scenario to say.")


if __name__ == "__main__":
    main()
