from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from ..agent import run_agent
from ..contracts import HypothesisSet, Layer
from ..errors import ContractViolation, ToolError
from ..llm import GroqClient
from ..measure.sql import MySQLTarget
from .evidence import (
    load_baseline,
    load_endpoint_model,
    ms_per_result_row,
    render_evidence,
    unexplained_share_pct,
)
from .probe import RUN_PROBE_TOOL, ProbeLog, make_run_probe
from .validate import ATTRIBUTION_TOLERANCE, validate_hypothesis_set

# Tool surface is deliberately two tools. Every request re-sends every schema,
# and read_file/search_code cost tokens on a provider tier that limits tokens
# per minute — measured across two runs, the agent spent its read calls on
# malformed arguments and established mechanism from probe numbers anyway.
DEFAULT_BASE_URL = "http://localhost:18080"
DEFAULT_ENDPOINT_MODEL = Path(".perfagent/endpoint_model.json")
MAX_STEPS = 20

_LAYERS = [l.value for l in Layer]


# ────────────────────────────  terminal tool  ────────────────────────────

EMIT_HYPOTHESES_TOOL = {
    "type": "function",
    "function": {
        "name": "emit_hypotheses",
        "description": (
            "Call this ONCE, after running probes, to deliver your diagnosis. "
            "This is the only way to finish. Your submission is checked by code: "
            "if a check fails you will be told exactly what is wrong and must fix "
            "it and call this again."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "hypotheses": {
                    "type": "array",
                    "minItems": 3,
                    "description": (
                        "At least 3 COMPETING explanations, spanning at least 2 "
                        "different layers. Not one theory restated three ways."
                    ),
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string", "description": "Short id, e.g. 'h1'"},
                            "claim": {
                                "type": "string",
                                "description": "What is slow, in one sentence.",
                            },
                            "mechanism": {
                                "type": "string",
                                "description": (
                                    "WHY that costs time — the specific chain of cause "
                                    "and effect, citing code where you read it."
                                ),
                            },
                            "layer": {
                                "type": "string",
                                "enum": _LAYERS,
                                "description": "Which layer the time is spent in.",
                            },
                            "evidence": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": (
                                    "Specific numbers from the measured evidence or from "
                                    "your probe results, quoted. Not general reasoning."
                                ),
                            },
                            "attributed_ms": {
                                "type": "number",
                                "description": (
                                    "How many ms of the measured p50 this explains. The "
                                    "sum across all hypotheses must fit inside p50."
                                ),
                            },
                            "prediction": {
                                "type": "string",
                                "description": (
                                    "A testable statement that would be FALSE if this "
                                    "hypothesis is wrong."
                                ),
                            },
                            "confidence": {
                                "type": "number",
                                "description": "0..1",
                            },
                        },
                        "required": [
                            "id", "claim", "mechanism", "layer", "evidence",
                            "attributed_ms", "prediction", "confidence",
                        ],
                    },
                },
                "probes_run": {
                    "type": "array",
                    "description": "Every probe you actually ran, and what it settled.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "tool": {"type": "string", "description": "e.g. 'run_probe'"},
                            "args": {
                                "type": "object",
                                "description": "The arguments you passed.",
                                "additionalProperties": True,
                            },
                            "discriminates_between": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": (
                                    "Hypothesis ids this probe separates. A probe that "
                                    "separates nothing is not an experiment."
                                ),
                            },
                            "result": {
                                "type": "string",
                                "description": "What it measured and what it ruled in or out.",
                            },
                        },
                        "required": ["tool", "args", "discriminates_between", "result"],
                    },
                },
                "ranked_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "All hypothesis ids, most significant first. Rank by attributed "
                        "ms supported by evidence, not by how familiar the cause is."
                    ),
                },
            },
            "required": ["hypotheses", "probes_run", "ranked_ids"],
        },
    },
}


# ──────────────────────────────  the prompt  ─────────────────────────────
# Note: run_agent (P02) has no system-role support — it opens the conversation
# with a single user message — so the instructions and the evidence travel
# together in the task string. Every number interpolated here comes from a
# recorded artifact, never from prose.

def build_task(baseline: dict[str, Any], endpoint_model: dict[str, Any]) -> str:
    facts = baseline["derived"]
    p50 = baseline["benchmark"]["latency"]["p50_ms"]
    per_row = ms_per_result_row(baseline)
    unexplained = unexplained_share_pct(baseline)
    sql_share = facts["sql_share_of_latency_pct"]
    ceiling = p50 * (1.0 + ATTRIBUTION_TOLERANCE)

    per_row_line = (
        f"{per_row:.0f} ms of measured latency per result row"
        if per_row
        else "an unknown cost per result row"
    )

    return f"""You are a performance diagnostician. You are given real measurements from a
running Spring Boot endpoint. Your job is to work out WHERE THE TIME ACTUALLY GOES
and deliver ranked, falsifiable hypotheses.

RULES — these are not negotiable:

1. The numbers below are MEASURED FACTS from a recorded benchmark run. You must not
   contradict them, restate them differently, or explain them away. If your theory
   disagrees with a number, your theory is wrong.
2. You must produce AT LEAST 3 COMPETING hypotheses spanning AT LEAST 2 different
   layers. Three restatements of one idea will be rejected by code.
3. You must RUN PROBES BEFORE RANKING. Ranking without experiments is guessing.
   Each probe you report must declare which hypotheses it separates.
4. Do not do arithmetic on raw samples — the derived facts below are already computed.
   Quote them.

{render_evidence(baseline, endpoint_model)}

=== THE OPEN QUESTION — THIS IS THE WHOLE TASK ===

Server-side SQL accounts for only {sql_share:.2f}% of measured latency.
That means {unexplained:.2f}% of the time — the large majority — is spent SOMEWHERE ELSE,
and nothing in the SQL evidence tells you where.

The cost is {per_row_line}. A request returning zero rows is fast; a request returning
a few rows is slow. So something expensive happens PER RESULT ROW, and it is mostly
not the database.

Your central task: determine what that {unexplained:.2f}% is.

Do not assume it is the N+1 queries. The N+1 is real and visible above, but the SQL
timings say those queries account for a small fraction of the total. If you rank the
N+1 first, you must justify it against the {sql_share:.2f}% figure with evidence.

Candidate directions worth separating (do not just accept one):
  - work done per row in application code (loops, allocation, collection building)
  - ORM hydration: how many entity objects are materialised per row
  - serialization: how large the response is and what it costs to build the JSON
  - the database queries themselves

=== HOW TO EXPERIMENT ===

run_probe fires the real endpoint with a page size and name filter you choose, and
reports p50 latency, application SQL statements per request, server-side SQL ms per
request, result row count and RESPONSE BODY SIZE IN BYTES.

Start by SWEEPING page_size: 1, 5, 20, 50. That tells you whether cost scales
linearly with the number of result rows, and which quantities scale with it —
latency, SQL statements, SQL ms, or response bytes. Quantities that scale together
are candidates for the same cause; quantities that stay flat are ruled out.
Use iterations=3 for page_size 20 and above; large pages are slow.

A probe with name_filter='zzzz-no-such-customer' returns zero rows and isolates the
filter query from all per-row work. That is a useful control.

For MECHANISM, use the call chain and the declared data access above — they tell you
which method runs per row and what it touches. State the mechanism in terms of what
that code does to each row, and tie it to a quantity your probes measured.

=== WHEN YOU ARE DONE ===

Call emit_hypotheses. Requirements enforced by code:
  - at least 3 hypotheses, spanning at least 2 different layers
  - layer must be one of: {', '.join(_LAYERS)}
  - sum of attributed_ms must not exceed {ceiling:.0f} ms
    (measured p50 is {p50:.1f} ms, {int(ATTRIBUTION_TOLERANCE * 100)}% tolerance)
  - ranked_ids must list every hypothesis id, most significant first
  - probes_run must contain at least one real probe with discriminates_between set

Do not write a prose report. Call emit_hypotheses."""


# ───────────────────────────  orchestration  ─────────────────────────────

def run_diagnosis(
    base_url: str = DEFAULT_BASE_URL,
    run_dir: Path | None = None,
    *,
    endpoint_model_path: Path = DEFAULT_ENDPOINT_MODEL,
    mysql: MySQLTarget | None = None,
    max_steps: int = MAX_STEPS,
    client: Any = None,
) -> tuple[HypothesisSet, ProbeLog, Path]:
    """Load evidence, let the agent probe, validate, persist.

    Reuses run_agent (P02) unmodified. run_agent hard-codes a short circuit for
    `emit_endpoint_model` only, so `emit_hypotheses` flows through the normal
    handler path — which is what we want: the handler validates before
    accepting, and a rejection comes back to the model as a tool result it can
    act on. The accepted set is captured in a holder rather than returned,
    because run_agent's return value is whatever the model said last.
    """
    from .evidence import latest_run_dir

    target = mysql or MySQLTarget.from_env()
    directory = Path(run_dir) if run_dir is not None else latest_run_dir()
    baseline = load_baseline(directory / "baseline.json")
    endpoint_model = load_endpoint_model(endpoint_model_path)
    p50 = baseline["benchmark"]["latency"]["p50_ms"]

    probe_log = ProbeLog()
    accepted: dict[str, HypothesisSet] = {}
    rejections: list[str] = []

    def emit_hypotheses(**payload: Any) -> str:
        """Terminal tool. Code decides whether this is acceptable."""
        try:
            result = validate_hypothesis_set(payload, p50, probe_log)
        except ToolError as exc:
            rejections.append(str(exc))
            raise
        accepted["set"] = result
        return (
            f"accepted: {len(result.hypotheses)} hypotheses across "
            f"{len({h.layer for h in result.hypotheses})} layers, "
            f"{len(result.probes_run)} probes. You are done — stop now."
        )

    llm = client or GroqClient(api_key=_require_api_key())

    # Once validation has ACCEPTED a diagnosis, that artifact is durable. The
    # loop may still die afterwards — run_agent keeps going after the terminal
    # tool returns, and P01 raises KeyError('choices') on any non-200 from the
    # provider, so a rate limit one step after a successful emit would
    # otherwise throw away work that already passed every check.
    loop_error: Exception | None = None
    try:
        run_agent(
            client=llm,
            task=build_task(baseline, endpoint_model),
            tools=[RUN_PROBE_TOOL, EMIT_HYPOTHESES_TOOL],
            handlers={
                "run_probe": make_run_probe(base_url, target, log=probe_log),
                "emit_hypotheses": emit_hypotheses,
            },
            max_steps=max_steps,
        )
    except Exception as exc:  # noqa: BLE001 - see comment above
        loop_error = exc

    if "set" not in accepted:
        detail = f"\nLast rejection was:\n{rejections[-1]}" if rejections else ""
        if loop_error is not None:
            detail += (
                f"\nThe agent loop then failed with "
                f"{type(loop_error).__name__}: {loop_error}"
            )
        raise ContractViolation(
            f"the agent finished without an accepted diagnosis after {max_steps} steps "
            f"({len(probe_log)} probes executed, {len(rejections)} submissions "
            f"rejected).{detail}"
        )

    hypothesis_set = accepted["set"]
    out_path = directory / "hypotheses.json"
    document = hypothesis_set.model_dump(mode="json")
    document["probes_executed"] = probe_log.entries
    document["baseline"] = str(directory / "baseline.json")
    document["endpoint_model"] = str(endpoint_model_path)
    document["submissions_rejected"] = len(rejections)
    if loop_error is not None:
        # Recorded, not hidden: the diagnosis is valid, but the run did not end
        # cleanly and the reader should know why.
        document["loop_error_after_acceptance"] = (
            f"{type(loop_error).__name__}: {loop_error}"
        )
    out_path.write_text(
        json.dumps(document, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return hypothesis_set, probe_log, out_path


def _require_api_key() -> str:
    from dotenv import load_dotenv

    load_dotenv()
    key = os.environ.get("GROQ_API_KEY")
    if not key:
        raise ContractViolation("GROQ_API_KEY is not set (put it in .env)")
    return key


# ─────────────────────────────  rendering  ──────────────────────────────

def render_diagnosis(
    hypothesis_set: HypothesisSet, probe_log: ProbeLog, loop_error: str | None = None
) -> str:
    by_id = {h.id: h for h in hypothesis_set.hypotheses}
    attributed = sum(h.attributed_ms for h in hypothesis_set.hypotheses)

    out: list[str] = []
    out.append("")
    out.append("RANKED HYPOTHESES")
    out.append(f"  measured p50 {hypothesis_set.measured_total_ms:.1f} ms  ·  "
               f"{attributed:.1f} ms attributed across "
               f"{len(hypothesis_set.hypotheses)} hypotheses")
    out.append("")

    for rank, hid in enumerate(hypothesis_set.ranked_ids, start=1):
        h = by_id.get(hid)
        if h is None:
            continue
        share = (100.0 * h.attributed_ms / hypothesis_set.measured_total_ms
                 if hypothesis_set.measured_total_ms else 0.0)
        out.append(f"  {rank}. [{h.id}] {h.claim}")
        out.append(f"     layer       {h.layer.value}")
        out.append(f"     attributed  {h.attributed_ms:.1f} ms of p50 ({share:.1f} %)"
                   f"   confidence {h.confidence:.2f}")
        out.append(f"     mechanism   {h.mechanism}")
        out.append(f"     prediction  {h.prediction}")
        for e in h.evidence:
            out.append(f"     evidence    - {e}")
        out.append("")

    out.append(f"PROBES REPORTED ({len(hypothesis_set.probes_run)})")
    if not hypothesis_set.probes_run:
        out.append("  none")
    for p in hypothesis_set.probes_run:
        args = ", ".join(f"{k}={v!r}" for k, v in p.args.items())
        out.append(f"  {p.tool}({args})")
        out.append(f"     separates   {', '.join(p.discriminates_between) or '(nothing)'}")
        out.append(f"     settled     {p.result}")
    out.append("")

    out.append(f"PROBES ACTUALLY EXECUTED ({len(probe_log)})")
    for entry in probe_log.entries:
        args = ", ".join(f"{k}={v!r}" for k, v in entry["args"].items())
        out.append(f"  {entry['tool']}({args})")
        first = entry["result"].splitlines()
        for line in first[1:6]:
            out.append(f"   {line}")
    if loop_error:
        out.append(f"NOTE: the diagnosis was accepted, but the agent loop then failed "
                   f"with {loop_error}")
        out.append("")
    return "\n".join(out)
