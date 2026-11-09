"""Re-run only the plan phase, live, reusing a scenario's existing hypotheses.

Needed once: live_freeze.py originally passed the live working-copy path as
`repo`, so `repo_root:` in the recorded prompt doesn't match the pinned
checkout path tier1.py reconstructs. This redoes just the plan conversation
against the pinned checkout instead -- no live app needed, only MySQL (for
scenarios whose plan phase uses explain/list_indexes/index_benefit).

Usage:
    .venv/bin/python eval/redo_plan.py COL-06
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from senapathy.agent import LoopBudget  # noqa: E402
from senapathy.contracts import EndpointModel, HypothesisSet  # noqa: E402
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


def _longest_cassette(cassette_dir: Path) -> dict:
    best, best_len = None, -1
    for path in cassette_dir.glob("*.json"):
        doc = json.loads(path.read_text())
        n = len(doc["request"]["messages"])
        if n > best_len:
            best, best_len = doc, n
    return best


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scenario_id")
    parser.add_argument("--repo", default=Path("/Users/akashm/Documents/mini-spring"),
                        type=Path)
    args = parser.parse_args()

    scenario_dir = SCENARIOS_DIR / args.scenario_id
    scratch_dir = SCRATCH_DIR / args.scenario_id
    hypotheses = HypothesisSet.model_validate(
        json.loads((scenario_dir / "hypotheses.json").read_text()))
    endpoint_model = EndpointModel.model_validate(
        json.loads((scenario_dir / "endpoint_model.json").read_text()))

    pinned_repo = ensure_checkout(
        args.repo, f"eval/{args.scenario_id}", endpoint_model.base_commit)

    plan_cassettes = scenario_dir / "cassettes" / "plan"
    if plan_cassettes.exists():
        shutil.rmtree(plan_cassettes)
    plan_cassettes.mkdir(parents=True)

    trace = Trace(scratch_dir, args.scenario_id)
    plan_client = make_client(cassette_dir=plan_cassettes, offline=False)
    plan_result = run_plan(
        repo=pinned_repo,
        endpoint=endpoint_model.route,
        hypotheses=hypotheses,
        model=endpoint_model,
        client=plan_client,
        trace=trace,
        budget=LoopBudget(max_steps=30),
        run_dir=scratch_dir,
        mysql=MySQLTarget.from_env(),
    )
    plan_path = write_plan(plan_result, scratch_dir)
    shutil.copy2(plan_path, scenario_dir / "plan.json")
    print(f"plan done: selected={plan_result.selected_id} "
         f"({plan_result.selection_reason})")
    for c in plan_result.candidates:
        print(f"  {c.id:4} {c.approach:20} mechanism={c.mechanism} "
             f"not_selectable={c.not_selectable_because}")

    tool_responses = json.loads((scenario_dir / "tool_responses.json").read_text())
    tool_responses["plan"] = [
        c for c in extract_tool_calls(
            _longest_cassette(plan_cassettes)["request"]["messages"])
        if c["tool"] in LIVE_TOOLS
    ]
    (scenario_dir / "tool_responses.json").write_text(
        json.dumps(tool_responses, indent=2, ensure_ascii=False, sort_keys=True) + "\n")
    print(f"froze {len(tool_responses['plan'])} plan live-tool response(s)")


if __name__ == "__main__":
    main()
