"""The plan phase: a diagnosis in, a ranked set of candidate fixes out.

The ranking happens BEFORE the model is called and does not involve it. By the
time the agent reads anything, code has already decided which hypotheses are
worth attacking and said why; the agent's job is to propose concrete fixes for
the ones that survived, not to re-litigate the ordering.

Like the comprehension phase, the transcript does not leave here. The `Plan`
is the entire output.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..agent import LoopBudget, LoopStatus, run_agent
from ..contracts import (EndpointModel, Hypothesis, HypothesisSet, Plan,
                         RankedHypothesis, SqlDigestStat, TableInfo)
from ..errors import ContractViolation
from ..ids import content_hash
from ..measure.baseline import baseline_hash
from ..prompts import load_prompt
from ..tools import SourceRoot, ToolRegistry
from ..tools.routes import route_tools
from ..tools.db import PLAN_TOOLS, optional_db_tools
from ..tools.schema import database_target, schema_tools
from ..tools.source import source_tools
from ..trace import EventType, Trace
from .brief import render_brief
from .fixability import rank_hypotheses
from .submit import submit_plan_tools

PHASE = "plan"
ARTIFACT = "plan.json"
DEFAULT_STACK = "Spring Boot / Maven / JPA-Hibernate / MySQL"

# How many hypotheses are put to the agent. Two, because the point of the
# expected-value ranking is to narrow the field: handing over all of them would
# reintroduce the choice the ranking just made. The rest stay in the artifact
# with their fixability reasons, which is what makes the PR body reviewable.
TOP_N = 2

PROMPT_ORDER: tuple[str, ...] = (
    "find_route", "read_file", "get_schema", "list_indexes", "index_benefit",
    "explain", "search_code", "list_dir", "submit_plan",
)


def build_registry(
    root: SourceRoot,
    sink: dict[str, Any],
    *,
    model: EndpointModel,
    ranked: list[RankedHypothesis],
    considered: list[str],
    endpoint: str,
    measured_total_ms: float,
    mysql: Any = None,
    measured_tables: dict[str, TableInfo] | None = None,
    prompt_hash: str | None = None,
    on_tool_error: Any = None,
    baseline_digests: list[SqlDigestStat] | None = None,
) -> ToolRegistry:
    """The tools this phase offers: the read surface, plus one terminal call.

    No write tool and no probe. A plan is a claim about what should change,
    and nothing here can change anything.
    """
    specs = [
        *route_tools(root),
        *source_tools(root),
        *(schema_tools(mysql, measured_tables) if mysql is not None else []),
        # A proposed index is MEASURED here, before it is selected — which is
        # the difference between "this should help" and "this examined 571
        # rows instead of 2,645". No write tool: planning cannot change code.
        *optional_db_tools(mysql, [t.table for t in model.tables],
                           names=PLAN_TOOLS, on_error=on_tool_error),
        *submit_plan_tools(root=root, model=model, ranked=ranked,
                           considered=considered, sink=sink, endpoint=endpoint,
                           measured_total_ms=measured_total_ms,
                           measured_tables=measured_tables,
                           prompt_hash=prompt_hash, mysql=mysql,
                           baseline_digests=baseline_digests),
    ]
    rank = {name: i for i, name in enumerate(PROMPT_ORDER)}
    specs.sort(key=lambda spec: rank.get(spec.name, len(rank)))
    return ToolRegistry().extend(specs)


def plan(
    *,
    repo: Path,
    endpoint: str,
    hypotheses: HypothesisSet,
    model: EndpointModel,
    client: Any,
    trace: Trace,
    budget: LoopBudget,
    run_dir: Path,
    mysql: Any = None,
    stack: str = DEFAULT_STACK,
    prompts_directory: Path | None = None,
) -> Plan:
    """Rank by expected value, then let the agent propose fixes for the top two."""
    root = SourceRoot.of(repo)
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    if mysql is None:
        # get_schema is in this phase's surface for one reason: a candidate
        # must never propose an index the database already has. The map
        # carries what `understand` measured; this lets the agent confirm it
        # against the live schema before committing to an index.
        mysql = database_target(trace, phase=PHASE)

    ranked = rank_hypotheses(hypotheses, model)
    if not ranked:
        raise ContractViolation(
            "the diagnosis contains no hypotheses, so there is nothing to plan."
        )
    considered = [entry.hypothesis_id for entry in ranked[:TOP_N]]
    by_id: dict[str, Hypothesis] = {h.id: h for h in hypotheses.hypotheses}

    trace.emit(EventType.PHASE_START, phase=PHASE, endpoint=endpoint,
               repo=str(root),
               ranked=[{"id": r.hypothesis_id, "attributed_ms": r.attributed_ms,
                        "fixability": r.fixability.fixability,
                        "expected_value_ms": r.expected_value_ms,
                        "reason": r.fixability.reason,
                        "blocked_by": r.fixability.blocked_by}
                       for r in ranked],
               considered=considered)

    baseline_digests = _load_baseline_digests(run_dir)

    system = load_prompt(
        "plan.system", directory=prompts_directory, trace=trace, phase=PHASE,
        unbounded_collections=_unbounded_summary(model),
    )
    user = load_prompt(
        "plan.user", directory=prompts_directory, trace=trace, phase=PHASE,
        repo_root=str(root), stack=stack, endpoint=endpoint,
        measured_total_ms=f"{hypotheses.measured_total_ms:.1f}",
        brief=render_brief(ranked=ranked, hypotheses=by_id,
                           considered=considered, model=model,
                           baseline_digests=baseline_digests),
    )
    prompt_hash = content_hash({"system": system.hash, "user": user.hash})

    sink: dict[str, Any] = {}
    # What get_schema actually reads during this conversation. An index list
    # measured here beats the one in the map, which is empty whenever the
    # comprehension phase ran without a database.
    measured: dict[str, TableInfo] = {}
    registry = build_registry(
        root, sink, model=model, ranked=ranked, considered=considered,
        endpoint=endpoint, measured_total_ms=hypotheses.measured_total_ms,
        mysql=mysql, measured_tables=measured, prompt_hash=prompt_hash,
        on_tool_error=lambda note: trace.emit(EventType.NOTE, phase=PHASE,
                                              note=note),
        baseline_digests=baseline_digests,
    )

    outcome = run_agent(
        client=client,
        task=user.text,
        system=system.text,
        tools=registry.schemas(),
        handlers=registry.handlers(),
        terminal_tool=registry.terminal_name(),
        budget=budget,
        trace=trace,
        phase=PHASE,
    )

    reference = (f"trace: {trace.path} (run {trace.run_id}), "
                 f"{outcome.steps} step(s), ${outcome.usage.usd:.4f}, "
                 f"{outcome.usage.input_tokens}/{outcome.usage.output_tokens} tokens")

    if outcome.status is not LoopStatus.COMPLETED:
        raise ContractViolation(
            f"the plan agent stopped with status {outcome.status.value}: "
            f"{outcome.stopped_because}. {reference}"
        )

    result: Plan | None = sink.get("plan")
    if result is None:
        raise ContractViolation(
            f"the plan agent finished without submitting any candidates. "
            f"{reference}\nIts final words were:\n"
            f"{outcome.final_text.strip() or '(nothing)'}"
        )

    path = write_plan(result, run_dir)
    selected = result.selected()
    trace.emit(EventType.ARTIFACT_WRITTEN, phase=PHASE, artifact=str(path),
               candidates=len(result.candidates),
               selected=result.selected_id,
               approach=selected.approach if selected else None,
               risk_tier=selected.risk_tier.value if selected else None,
               prompt_hash=prompt_hash, steps=outcome.steps,
               usd=round(outcome.usage.usd, 6))
    trace.emit(EventType.PHASE_END, phase=PHASE, artifact=str(path),
               summary=_summary(result))
    return result


def _summary(result: Plan) -> str:
    selected = result.selected()
    if selected is None:
        return f"{len(result.candidates)} candidate(s), none selectable"
    return (f"{len(result.candidates)} candidate(s), selected {selected.id} "
            f"({selected.approach}, T{selected.risk_tier.value}) targeting "
            f"{selected.targets_hypothesis}")


def _unbounded_summary(model: EndpointModel) -> str:
    """What goes into the system prompt's byte-identical paragraph."""
    from .fixability import unbounded_fields

    fields = unbounded_fields(model.response_shape)
    if not fields:
        return ("none recorded — treat every returned field as frozen anyway, "
                "because the oracle does")
    return ", ".join(fields)


def _load_baseline_digests(run_dir: Path) -> list[SqlDigestStat]:
    """This endpoint's own measured SQL digests, if baseline.json exists.

    A `sql_count` candidate must cite one of these by hash — see
    `brief.py::_baseline_sql` and `submit.py`'s check against it. Missing
    entirely (a caller that never ran baseline, or a test fixture) degrades
    to an empty list rather than raising: the brief then shows "none
    recorded", and every `sql_count` candidate is refused as unsupported,
    which is the honest answer when there is no baseline to cite at all.
    """
    path = Path(run_dir) / "baseline.json"
    if not path.exists():
        return []
    document = json.loads(path.read_text(encoding="utf-8"))
    return [SqlDigestStat.model_validate(d) for d in document.get("sql_digests") or []]


# ──────────────────────────────  artifacts  ──────────────────────────────


def write_plan(result: Plan, run_dir: Path) -> Path:
    path = Path(run_dir) / ARTIFACT
    document = result.model_dump(mode="json")
    # What a resume's skip check compares against THIS run's current
    # baseline.json -- a plan ranked against a baseline that has since been
    # re-measured is not evidence for the current one. Absent file, no stamp:
    # nothing to compare against is not the same as a mismatch.
    baseline_path = Path(run_dir) / "baseline.json"
    if baseline_path.exists():
        try:
            document["baseline_hash"] = baseline_hash(
                json.loads(baseline_path.read_text(encoding="utf-8")))
        except json.JSONDecodeError:
            pass
    path.write_text(
        json.dumps(document, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return path


def load_plan(path: Path) -> Plan:
    payload: Any = json.loads(Path(path).read_text(encoding="utf-8"))
    return Plan.model_validate(payload)


def resolve_run_dir(runs_dir: Path = Path(".senapathy/runs")) -> Path:
    """Newest run directory that holds a diagnosis.

    Not simply the newest directory: a baseline-only run is newer than the one
    that produced hypotheses.json, and defaulting to it would report a missing
    file rather than the run the user meant.
    """
    candidates = sorted(
        (d for d in Path(runs_dir).glob("*") if (d / "hypotheses.json").exists()),
        reverse=True,
    )
    if not candidates:
        raise ContractViolation(
            f"no run under {runs_dir} contains hypotheses.json. Run "
            "`senapathy diagnose` first — a plan ranks a measured diagnosis, "
            "not a guess."
        )
    return candidates[0]


def load_inputs(run_dir: Path,
                endpoint_model_path: Path | None = None
                ) -> tuple[HypothesisSet, EndpointModel]:
    """The two artifacts this phase reads, with errors a person can act on."""
    directory = Path(run_dir)
    diagnosis = directory / "hypotheses.json"
    if not diagnosis.exists():
        raise ContractViolation(
            f"no diagnosis at {diagnosis}. Run `senapathy diagnose` first, or "
            f"point --run-dir at the run that produced hypotheses.json."
        )
    comprehension = Path(endpoint_model_path or (directory / "endpoint_model.json"))
    if not comprehension.exists():
        # A map from ANOTHER run is not substituted silently. It may describe a
        # different endpoint or a different commit, and fixability read off the
        # wrong map is worse than no fixability — so the newest one is named as
        # a suggestion and the caller decides.
        elsewhere = _newest_endpoint_model(directory.parent)
        hint = (f" The newest comprehension artifact on disk is {elsewhere}; "
                f"pass --endpoint-model {elsewhere} if it describes this "
                f"endpoint at this commit."
                if elsewhere is not None else "")
        raise ContractViolation(
            f"no comprehension model at {comprehension}. Run `senapathy "
            f"understand` first — without the map there is no "
            f"response_shape.unbounded_collections, and fixability cannot be "
            f"assessed.{hint}"
        )

    hypotheses = HypothesisSet.model_validate(
        json.loads(diagnosis.read_text(encoding="utf-8")))
    model = EndpointModel.model_validate(
        json.loads(comprehension.read_text(encoding="utf-8")))
    return hypotheses, model


def _newest_endpoint_model(runs_dir: Path) -> Path | None:
    """The most recently modified endpoint_model.json under `runs_dir`."""
    found = sorted(Path(runs_dir).glob("*/endpoint_model.json"),
                   key=lambda p: p.stat().st_mtime, reverse=True)
    return found[0] if found else None
