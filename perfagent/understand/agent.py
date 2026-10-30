"""The comprehension phase: one route in, one validated EndpointModel out.

The boundary this function draws is the important part. The transcript does
not leave here. Ten phases each handing their conversation to the next is how
a run runs out of context window at phase six; ten phases each handing over a
typed artifact is how it does not.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..agent import LoopBudget, LoopStatus, run_agent
from ..contracts import EndpointModel, TableInfo
from ..errors import ContractViolation
from ..ids import content_hash
from ..prompts import load_prompt
from ..tools import SourceRoot, ToolRegistry
from ..tools.routes import route_tools
from ..tools.schema import schema_tools
from ..tools.source import source_tools
from ..tools.submit import submit_tools
from ..trace import EventType, Trace

PHASE = "understand"
ARTIFACT = "endpoint_model.json"
DEFAULT_STACK = "Spring Boot / Maven / JPA-Hibernate / MySQL"


# The order the system prompt presents the tools in. The order a model reads
# tool definitions in is part of the prompt whether you intended it or not, so
# the registry is sorted to match rather than assembled in whatever order the
# factories happen to be called. `list_dir` is not in the prompt at all and
# lands after the tools that are.
PROMPT_ORDER: tuple[str, ...] = (
    "find_route", "read_file", "get_schema", "search_code", "list_dir",
    "submit_endpoint_analysis",
)


def build_registry(
    root: SourceRoot,
    sink: dict[str, Any],
    *,
    mysql: Any = None,
    measured_tables: dict[str, TableInfo] | None = None,
    prompt_hash: str | None = None,
) -> ToolRegistry:
    """The tools this phase offers, bound to one repository.

    Ordered to match `PROMPT_ORDER`, which is the order the system prompt
    presents them in. The previous version claimed that and did not do it —
    `search_code` arrived before `get_schema` — and a docstring making a claim
    the code does not keep is worse than no docstring.
    """
    specs = [
        *route_tools(root),
        *source_tools(root),
        *(schema_tools(mysql, measured_tables) if mysql is not None else []),
        *submit_tools(root, sink, measured_tables=measured_tables,
                      prompt_hash=prompt_hash),
    ]
    rank = {name: i for i, name in enumerate(PROMPT_ORDER)}
    specs.sort(key=lambda spec: rank.get(spec.name, len(rank)))
    return ToolRegistry().extend(specs)


def understand(
    *,
    repo: Path,
    endpoint: str,
    client: Any,
    trace: Trace,
    budget: LoopBudget,
    run_dir: Path,
    mysql: Any = None,
    stack: str = DEFAULT_STACK,
    prompts_directory: Path | None = None,
) -> EndpointModel:
    """Trace one endpoint and return its validated model.

    Raises ContractViolation when the agent finishes without an accepted
    submission — a phase that cannot produce its artifact says so, rather than
    returning something partial that the next phase would trust.
    """
    root = SourceRoot.of(repo)
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    if mysql is None:
        mysql = _database_target(trace)

    system = load_prompt("understand.system", directory=prompts_directory,
                         trace=trace, phase=PHASE)
    user = load_prompt("understand.user", directory=prompts_directory,
                       trace=trace, phase=PHASE,
                       repo_root=str(root), stack=stack, endpoint=endpoint)
    prompt_hash = content_hash({"system": system.hash, "user": user.hash})

    sink: dict[str, Any] = {}
    measured: dict[str, TableInfo] = {}
    registry = build_registry(root, sink, mysql=mysql, measured_tables=measured,
                              prompt_hash=prompt_hash)

    trace.emit(EventType.PHASE_START, phase=PHASE, endpoint=endpoint,
               repo=str(root), tools=registry.names(), prompt_hash=prompt_hash)

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
                 f"{outcome.steps} step(s), "
                 f"${outcome.usage.usd:.4f}, "
                 f"{outcome.usage.input_tokens}/{outcome.usage.output_tokens} tokens")

    if outcome.status is not LoopStatus.COMPLETED:
        raise ContractViolation(
            f"the comprehension agent stopped with status "
            f"{outcome.status.value}: {outcome.stopped_because}. {reference}"
        )

    model: EndpointModel | None = sink.get("model")
    if model is None:
        raise ContractViolation(
            f"the comprehension agent finished without submitting an analysis. "
            f"{reference}\nIts final words were:\n{outcome.final_text.strip() or '(nothing)'}"
        )

    path = write_model(model, run_dir)
    trace.emit(EventType.ARTIFACT_WRITTEN, phase=PHASE, artifact=str(path),
               route=model.route, hops=len(model.call_chain),
               data_accesses=len(model.data_access),
               per_row=len(model.per_row_accesses()),
               tables=len(model.tables), prompt_hash=prompt_hash,
               usd=round(outcome.usage.usd, 6), steps=outcome.steps)
    trace.emit(EventType.PHASE_END, phase=PHASE, artifact=str(path),
               summary=f"{model.route}: {len(model.call_chain)} hops, "
                       f"{len(model.data_access)} data accesses")
    return model


def _database_target(trace: Trace) -> Any:
    """The MySQL target, or None if one cannot even be described.

    Constructed, not connected: an UNREACHABLE database has to fail inside
    `get_schema`, with a message the model can read and work around, not here
    where it would abort a phase that can still do 90% of its job from source
    alone.

    `from_env()` reads environment variables and validates them, so a garbled
    `PERFAGENT_MYSQL_PORT` raises right here — which is exactly the failure at
    the door this comment used to claim was impossible. Swallow it, note it in
    the trace, and run without the tool.
    """
    try:
        from ..measure.sql import MySQLTarget

        return MySQLTarget.from_env()
    except Exception as exc:  # noqa: BLE001 — no database is a degraded run, not a dead one
        trace.emit(
            EventType.NOTE, phase=PHASE,
            note=f"no database target could be built ({type(exc).__name__}: "
                 f"{exc}); running without get_schema. `tables` will carry no "
                 f"measured facts.",
        )
        return None


def write_model(model: EndpointModel, run_dir: Path) -> Path:
    path = Path(run_dir) / ARTIFACT
    path.write_text(
        json.dumps(model.model_dump(mode="json"), indent=2, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )
    return path


def load_model(path: Path) -> EndpointModel:
    """Read an artifact back. Used by the orchestrator's skip check."""
    payload: Any = json.loads(Path(path).read_text(encoding="utf-8"))
    return EndpointModel.model_validate(payload)
