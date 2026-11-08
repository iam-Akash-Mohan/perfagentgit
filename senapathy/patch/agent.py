"""The patch agent: read the diagnosis, write a real fix, prove it builds.

Reuses make_client (S1) and run_agent (P02) unmodified. `emit_patch` is
deliberately NOT passed to run_agent as `terminal_tool` -- its handler
refuses a submission that has not built and passed tests, and the refusal
comes back as a tool result the agent can act on, not a reason to end the
conversation. `accepted` is set the moment emit_patch succeeds and is
durable from then on, regardless of what happens afterward.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Sequence

from ..agent import LoopBudget, run_agent
from ..contracts import CandidateFix, EndpointModel, Patch
from ..errors import ContractViolation
from ..llm import make_client
from ..measure.baseline import baseline_hash
from ..measure.sql import MySQLTarget
from ..plan.risk import call_path_files
from ..tools import SourceRoot
from ..tools.source import READ_FILE_PARAMETERS, make_read_file
from ..trace import Trace
from .standards import lint_diff, load_standards, render_violations
from .tools import (ADD_MIGRATION_TOOL, APPLY_PATCH_TOOL, BUILD_TOOL,
                    MAX_ITERATIONS, REQUEST_FILE_ACCESS_TOOL, RUN_TESTS_TOOL,
                    PatchTools, TestCounts, count_test_files,
                    parse_test_counts)
from .worktree import GREEN_DIR, GreenWorktree, build_env

# Read-only, same containment rule as the comprehension registry's own
# read_file (SourceRoot.resolve rejects anything outside the worktree). Not
# restricted to files_to_touch: reading is not writing, and the write
# allowlist is the guardrail that matters -- COL-05 died reproducing
# old_text from memory after its first edit, and COL-03 hallucinated
# print_tree reaching for exactly this, on a registry that had never had a
# way to read anything back.
READ_FILE_TOOL = {
    "type": "function",
    "function": {
        "name": "read_file",
        "description": (
            "Read a source file from the green worktree, with line numbers. "
            "Use it to get the CURRENT, exact content before calling "
            "apply_patch -- old_text must match byte for byte, and after "
            "your first edit the file no longer matches what you were shown "
            "at the start of this task."
        ),
        "parameters": READ_FILE_PARAMETERS,
    },
}

# Sized for a real codebase, not mini-spring: enough LLM turns to read
# context, apply a patch, build, fix a failed build, and emit, with room for
# a rejected submission to be corrected. Distinct from
# patch/tools.py:MAX_ITERATIONS, which caps build/test CYCLES, not steps.
# Steps are a backstop; the dollar and wall-clock ceilings are the real
# brakes.
MAX_STEPS = 80

# NOTE: a nine-entry SOURCE_CONTEXT tuple of one repo's own classes used
# to live here. It was already dead -- `context_files()` derives the list
# from the plan and the endpoint model -- and it named `com/ex` paths, so
# it is gone rather than merely unused.


# ────────────────────────────  terminal tool  ────────────────────────────

EMIT_PATCH_TOOL = {
    "type": "function",
    "function": {
        "name": "emit_patch",
        "description": (
            "Call this ONCE, after build and run_tests have both succeeded, to "
            "deliver your patch. This is the only way to finish. Code checks "
            "that the tree actually builds and tests pass before accepting; if "
            "it does not, you will be told and must fix it and call again."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "summary": {
                    "type": "string",
                    "description": "What you changed, in two or three sentences.",
                },
                "mechanism_changed": {
                    "type": "string",
                    "description": (
                        "The specific mechanism that now costs less — e.g. 'two "
                        "queries per result row replaced by one join'. Not 'made "
                        "it faster'."
                    ),
                },
                "files_touched": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Every file you edited, repo-relative.",
                },
                "expected_effect": {
                    "type": "string",
                    "description": (
                        "What should measurably change, and in which direction. "
                        "Name the quantity: SQL statements per request, rows "
                        "examined, latency."
                    ),
                },
                "risks": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "What could break, specifically. 'None' is not an answer "
                        "— if you changed how a value is computed, say where it "
                        "could differ."
                    ),
                },
            },
            "required": [
                "summary", "mechanism_changed", "files_touched",
                "expected_effect", "risks",
            ],
        },
    },
}

# The other way this phase can end. "ABANDONED" with every diagnostic field
# empty is not an answer a reviewer can act on -- `reason` is required so
# that calling this without one is a schema violation, not a silent gap.
ABANDON_PATCH_TOOL = {
    "type": "function",
    "function": {
        "name": "abandon_patch",
        "description": (
            "Call this if you have concluded this candidate cannot be "
            "patched -- request_file_access was refused for a file the fix "
            "genuinely needs, the approach does not actually work, or some "
            "other reason that means continuing would not produce a real "
            "patch. State why. This is not a failure to avoid; an honest, "
            "explained abandonment is a better outcome than silence or a "
            "patch that does not build."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "reason": {
                    "type": "string",
                    "description": (
                        "Specifically why this candidate cannot be patched. "
                        "Not 'it didn't work' -- name the file, the "
                        "constraint, or the error that makes it impossible."
                    ),
                },
            },
            "required": ["reason"],
        },
    },
}


# ──────────────────────────────  the prompt  ─────────────────────────────
# run_agent has no system-role support, so instructions and evidence travel
# together in the opening user message. Every number here is read from
# hypotheses.json; none is written by hand.

def _render_source(green: Path, files: Sequence[str]) -> str:
    """The files the agent may edit, verbatim, plus the ones on the path.

    `apply_patch` needs byte-exact `old_text` and there is no read_file tool
    here, so whatever is not in this block cannot be edited correctly. The list
    used to be a hard-coded tuple of mini-spring's own classes; it is now
    derived from the plan and the endpoint model, because a tool that only
    works on one repository is not a tool.
    """
    blocks: list[str] = []
    for rel in files:
        path = green / rel
        if not path.exists() or path.is_dir():
            continue
        blocks.append(f"----- {rel}\n{path.read_text(encoding='utf-8').rstrip()}")
    return "\n\n".join(blocks)


def context_files(candidate: CandidateFix, model: EndpointModel) -> list[str]:
    """Files to show: everything writable, then everything on the call path."""
    out: list[str] = [f for f in candidate.files_to_touch
                      if not f.endswith(".sql")]
    for hop in model.call_chain:
        if hop.file not in out:
            out.append(hop.file)
    for access in model.data_access:
        if access.file not in out:
            out.append(access.file)
    return out


def build_task(*, candidate: CandidateFix, model: EndpointModel, green: Path,
               fixability_reason: str = "", evidence: str = "",
               standards: str = "", measured_p50_ms: float | None = None,
               max_iterations: int = MAX_ITERATIONS,
               oracle_feedback: str = "") -> str:
    """The patch prompt, built from ONE SELECTED CANDIDATE.

    It used to be built from `hypotheses.json` and `ranked_ids[0]` — the
    hypothesis with the largest attributed milliseconds. On the endpoint this
    product was built against that is the response body itself, so the agent
    was asked three times to make the response cheaper without making it
    smaller, which is not possible. The plan phase now decides what to attack
    and why; this phase implements that decision.
    """
    unbounded = []
    if model.response_shape is not None:
        unbounded = [str(f).split("(")[0].strip()
                     for f in model.response_shape.unbounded_collections]

    allowlist = "\n".join(f"  {f}" for f in candidate.files_to_touch)
    risks = "\n".join(f"  - {r}" for r in candidate.what_could_break) or "  (none stated)"
    latency = (f"Measured p50 latency: {measured_p50_ms:.1f} ms\n"
               if measured_p50_ms else "")
    frozen = (", ".join(unbounded) if unbounded
              else "(none recorded — treat every returned field as frozen)")

    rejected = ""
    if oracle_feedback:
        rejected = f"""=== YOUR PREVIOUS ATTEMPT WAS REJECTED BY THE ORACLE ===

The tree has been reset. A previous patch for this same candidate changed the
response, and these are the differences the frozen recording actually found:

{oracle_feedback}

Read them literally. Each line is one request and one JSON path where your
output differed from what the unmodified code produced. Fix the cause — do not
work around the comparison, which you cannot see or change.

"""

    return f"""You are a performance engineer implementing ONE PLANNED CHANGE. The analysis
is done: a diagnosis measured where the time goes, and a plan phase chose this
candidate over the alternatives and said why. Your job is to write it, prove it
compiles, and prove the existing tests still pass.

=== THE CANDIDATE YOU ARE IMPLEMENTING ===

  id                 {candidate.id}
  approach           {candidate.approach}
  targets            {candidate.targets_hypothesis}
  what to change     {candidate.description}
  mechanism          {candidate.mechanism_changed}
  predicted effect   {candidate.predicted_effect}
  risk tier          T{candidate.risk_tier.value}
{latency}
Why this one and not the bigger number:
  {fixability_reason or "(no fixability reason recorded)"}

The plan's own list of what could break — these are the things to be careful of:
{risks}

{evidence}{rejected}=== THE HARD CONSTRAINT — READ THIS TWICE ===

THE HTTP RESPONSE MUST STAY BYTE-IDENTICAL.

A frozen recording of this endpoint's responses was made before you were asked
anything. Every recorded request is replayed afterwards and compared field by
field, including a held-back set. One differing byte ends the run. That oracle
is not in your tool surface and you cannot see, widen or regenerate it.

UNBOUNDED COLLECTIONS IN THE RESPONSE: {frozen}

Those fields grow with the data and their size is part of the contract. You may
not trim, cap, page or lazily load them, and you may not propose a slimmer DTO.
Produce EXACTLY THE SAME BYTES WITH LESS WORK.

Also forbidden, and enforced in code rather than by this paragraph:
  - caching of any kind
  - changed transaction boundaries
  - any API change: paths, request shape, response shape, status codes
  - any schema change other than an added index, via add_migration
  - any edit to src/test/** — test files are the evidence that behaviour held

=== THE FILES YOU MAY WRITE ===

The plan named these, before any code was written. Writing anywhere else is
refused by code:

{allowlist}

A new database migration is the one exception, and only through add_migration,
which parses your SQL and refuses anything that is not a single CREATE INDEX.

=== JAVA STANDARDS ===

{standards or "(no standards file found)"}

The rules marked ENFORCED are checked against your diff by a lint pass when you
call emit_patch. A violation is a rejection, not a warning.

=== THE CODE, EXACTLY AS IT IS ON DISK ===

apply_patch needs old_text that matches BYTE FOR BYTE. Copy your snippets from
below rather than retyping them — a retyped snippet with different indentation
is the single most common way this fails.

{_render_source(green, context_files(candidate, model))}

=== HOW TO WORK ===

0. The content above can go stale the moment you make your first edit, and it
   never included every file in the repository. Call read_file whenever you
   need a file's CURRENT content — before an apply_patch whose old_text must
   match byte for byte, or to look at a file (a DTO, a sibling class) that
   was not shown above. It is read-only; it does not change what you may
   write to.
1. Make the change the candidate describes. apply_patch, or add_migration for
   an index.
2. build. If it fails you get the compiler output verbatim — read it literally,
   it names the file and the line. Fix and build again.
3. run_tests. It requires a successful build first. When it passes you will
   also be given a QUICK BENCHMARK of your own change: twenty requests against
   the patched tree, with the pre-patch figure beside it. That number is a
   heuristic, not the verdict — the real measurement is an interleaved A/B
   against the unpatched tree with a paired significance test, run later. Use
   it to catch a regression before you commit to one.
4. Only when build AND run_tests have both succeeded, call emit_patch.

You have {max_iterations} build/test iterations in total, shared between build
and run_tests. If you run out, the run is abandoned and no patch is emitted.

5. If the fix needs a file outside the allowlist above — e.g. changing a
   method's signature means its caller must change too — call
   request_file_access(file) before giving up. It is granted automatically
   when the file is on this endpoint's own traced call path, and refused by
   name otherwise. Only if it is refused, or the approach does not work for
   some other real reason, call abandon_patch with a specific reason. That is
   a better outcome than running out the clock in silence.

Do not write a prose report. Make the change, prove it, then call emit_patch."""


def build_continuation_task(diff: str, said: str, missing: list[str]) -> str:
    """Ask again for whatever step the model stopped short of.

    Both observed failures were the same shape: the model did the work and then
    ended its turn instead of finishing the protocol. Once it produced a
    perfectly formed emit_patch payload as PROSE (`stop_reason=end_turn`, no
    tool call) over a diff that built and passed tests; once it stopped after a
    green build without running tests.

    Accepting that prose would let a model's text decide the phase outcome,
    which is what invariant 6 forbids. So this asks for the missing CALLS.
    run_agent opens a fresh conversation, so the verified state is restated.
    """
    steps = "\n".join(f"  {i}. {step}" for i, step in enumerate(missing, start=1))
    return f"""You were asked to patch a Spring Boot endpoint. You did the work but you did
not finish the protocol — you ended your turn instead of making the remaining
tool calls.

Text is not a submission. Only tool CALLS count: the outcome of this task is
recorded from the calls and nothing you write as prose is read.

Your changes are still in place, exactly as you left them. This is the diff:

{diff or "(no changes)"}

What you said last time:

{said.strip()[:1500]}

DO EXACTLY THIS, as tool calls, in order, and nothing else:
{steps}

Do not edit any file again unless a build error tells you to. Do not restate
your answer as text. Make the calls."""


# ───────────────────────────  orchestration  ─────────────────────────────

def resolve_run_dir(runs_dir: Path = Path(".senapathy/runs")) -> Path:
    """Newest run directory that actually holds a diagnosis.

    Not simply the newest directory: a baseline-only run is newer than the run
    that produced hypotheses.json, and defaulting to it produced a bare
    FileNotFoundError instead of a usable message.
    """
    candidates = sorted(
        (d for d in runs_dir.glob("*") if (d / "hypotheses.json").exists()),
        reverse=True,
    )
    if not candidates:
        raise ContractViolation(
            f"no run under {runs_dir} contains hypotheses.json. Run "
            "`senapathy diagnose` first — the patch agent fixes a measured "
            "cause, not a guessed one."
        )
    return candidates[0]


def load_hypotheses(run_dir: Path, hypotheses_path: Path | None = None) -> dict[str, Any]:
    """Read the diagnosis, with an actionable error rather than a traceback."""
    path = hypotheses_path or (Path(run_dir) / "hypotheses.json")
    if not path.exists():
        raise ContractViolation(
            f"no diagnosis at {path}. Run `senapathy diagnose` first, or point "
            "--run-dir at the run that produced hypotheses.json."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def _database_target(profile: Any) -> MySQLTarget:
    """This run's target: from the profile when there is one, else an
    explicit environment override for standalone `senapathy patch` use."""
    if profile is not None:
        return MySQLTarget.from_profile(profile.database)
    return MySQLTarget.from_env()


def _database_tools(run_dir: Path, profile: Any = None) -> list[Any]:
    """The four database tools, bound to this run's tables.

    The allowlist comes from the comprehension artifact, so the patch agent
    can only touch tables the endpoint actually reads. No artifact, or no
    reachable database, means no database tools — never a wider allowlist.
    """
    from ..tools.db import PATCH_TOOLS, optional_db_tools

    model_path = Path(run_dir) / "endpoint_model.json"
    if not model_path.exists():
        return []
    try:
        payload = json.loads(model_path.read_text(encoding="utf-8"))
        tables = [str(t.get("table")) for t in (payload.get("tables") or [])
                  if t.get("table")]
        target = _database_target(profile)
    except Exception:  # noqa: BLE001 — a degraded surface, not a dead phase
        return []
    return optional_db_tools(target, tables, names=PATCH_TOOLS)


def run_patch(
    run_dir: Path,
    *,
    run_id: str,
    repo: str | Path | None = None,
    green_dir: str | Path = GREEN_DIR,
    hypotheses_path: Path | None = None,
    max_steps: int = MAX_STEPS,
    budget: LoopBudget | None = None,
    max_iterations: int = MAX_ITERATIONS,
    max_terminal_retries: int = 2,
    java_home: str | None = None,
    profile: Any = None,
    client: Any = None,
    keep_worktree: bool = True,
    candidate_index: int = 0,
    oracle_feedback: str = "",
    trace: Trace | None = None,
) -> dict[str, Any]:
    """Create green, measure the pre-patch baseline, let the agent patch, persist.

    Returns a document with `outcome` either "PATCHED" or "ABANDONED". An
    abandoned run resets the worktree and reports the last error rather than
    emitting a patch that does not build.

    `trace`, tagged phase="patch", is the same handle `understand`/`plan`/
    `diagnose` take -- without it, nothing from this phase's LLM calls or
    build/test iterations reaches trace.jsonl, which is how two ABANDONED
    runs left no record of what the agent did or why it stopped.

    `max_steps` is a convenience for the standalone `senapathy patch`
    command, which has no --max-usd/--max-wall of its own; it is folded into
    a `LoopBudget(max_steps=max_steps, step_flag="--max-steps")` when no
    `budget` is given. The `run` pipeline, which knows the run-level dollar
    and wall-clock ceilings, passes a full `budget` directly, and it wins.
    `max_iterations` is a SEPARATE build/test-cycle budget
    (patch/tools.py:MAX_ITERATIONS), not an LLM step count.
    """
    if budget is None:
        budget = LoopBudget(max_steps=max_steps, step_flag="--max-steps")

    directory = Path(run_dir)
    hypotheses = load_hypotheses(directory, hypotheses_path)
    candidate, model, plan = load_plan_inputs(directory, index=candidate_index)
    reason = _fixability_reason(plan, candidate)

    blue_status_before = None
    green = GreenWorktree.create(run_id, repo=repo, green=green_dir)
    blue_status_before = green.blue_status()

    try:
        baseline = _measure_baseline(green, java_home, profile)
        tools = PatchTools(
            green=green,
            baseline=baseline,
            java_home=java_home,
            max_iterations=max_iterations,
            allowed_files=tuple(candidate.files_to_touch),
            allowed_tables=tuple(t.table for t in model.tables),
            call_path_files=tuple(call_path_files(model)),
            benchmark=_benchmark_for(green, hypotheses, java_home, profile),
            # The detected layout. Without it the write guardrail defends
            # `src/main/java` on every repo, which on a Gradle/Kotlin tree
            # means it defends a directory that does not exist.
            profile=profile,
            trace=trace,
        )

        accepted: dict[str, Any] = {}
        rejections: list[str] = []
        abandonment: dict[str, Any] = {}

        def abandon_patch(**payload: Any) -> str:
            """The model's own stated reason for giving up. Recorded, never
            silently dropped."""
            abandonment["reason"] = payload.get("reason", "")
            return ("Understood. This run will be recorded as abandoned "
                    "with the reason you gave. Nothing further will be "
                    "accepted — stop now.")

        def emit_patch(**payload: Any) -> str:
            """Terminal tool. Code decides whether this is acceptable."""
            if tools.guardrail_violations:
                message = (
                    "REJECTED: a guardrail was violated earlier in this run "
                    f"({tools.guardrail_violations[-1]}). No patch can be accepted."
                )
                rejections.append(message)
                return message
            if not tools.last_build_ok:
                message = (
                    "REJECTED: the tree does not currently build, so this patch "
                    "cannot be accepted. Call build, fix what it reports, then "
                    "run_tests, then call emit_patch again."
                )
                rejections.append(message)
                return message
            if not tools.last_tests_ok:
                message = (
                    "REJECTED: run_tests has not passed since your last edit. "
                    "Call run_tests and make it pass before emitting."
                )
                rejections.append(message)
                return message
            if tools.edits_since_verify:
                message = (
                    f"REJECTED: you have made {tools.edits_since_verify} edit(s) "
                    "since the last successful build and test. Re-verify before "
                    "emitting."
                )
                rejections.append(message)
                return message

            # The standards lint, over the diff, at the last possible moment —
            # which is the only moment the whole diff exists. A rejection here
            # is retryable; an accepted patch that breaks a stated rule is not.
            violations = lint_diff(green.diff(), repo=green.path)
            if violations:
                message = "REJECTED: " + render_violations(violations)
                rejections.append(message)
                tools.standards_violations.extend(v.rule for v in violations)
                return message

            accepted["payload"] = payload
            note = ("" if not tools.benchmark_regressed else
                    " NOTE: your own benchmark said this is slower; that is "
                    "recorded with the patch.")
            return ("accepted: the patch builds and the tests pass. You are "
                    "done — stop now." + note)

        llm = client or make_client()

        # As in P09: once code has ACCEPTED a submission the work is durable.
        # run_agent keeps going after a terminal tool returns, and P01 raises on
        # any non-200 from the provider, so a rate limit one step after a good
        # emit would otherwise throw away a patch that already passed every check.
        # The patch agent is the only phase with all four database tools: it
        # is already writing code, so measuring an index and checking a
        # rewrite returns the same rows are things it needs before it commits
        # to either. `query_equivalent` especially — a rewrite that returns
        # different rows is found here, one build cycle early, instead of by
        # the oracle at the end of the run.
        db_specs = _database_tools(directory, profile)
        handlers = {
            "read_file": make_read_file(SourceRoot.of(green.path)),
            "apply_patch": tools.apply_patch,
            "build": tools.build,
            "run_tests": tools.run_tests,
            "emit_patch": emit_patch,
            "abandon_patch": abandon_patch,
            "add_migration": tools.add_migration,
            "request_file_access": tools.request_file_access,
            **{spec.name: spec.handler for spec in db_specs},
        }
        tool_set = [READ_FILE_TOOL, APPLY_PATCH_TOOL, BUILD_TOOL, RUN_TESTS_TOOL,
                    ADD_MIGRATION_TOOL, REQUEST_FILE_ACCESS_TOOL, EMIT_PATCH_TOOL,
                    ABANDON_PATCH_TOOL, *(spec.schema() for spec in db_specs)]

        loop_error: Exception | None = None
        loop_stopped_because = ""
        loop_final_text = ""
        task = build_task(
            candidate=candidate, model=model, green=green.path,
            fixability_reason=reason, standards=load_standards(),
            evidence=_evidence_block(hypotheses, candidate),
            measured_p50_ms=hypotheses.get("measured_total_ms"),
            max_iterations=max_iterations,
            oracle_feedback=oracle_feedback,
        )
        terminal_retries = 0
        while True:
            try:
                said = run_agent(
                    client=llm,
                    task=task,
                    tools=tool_set,
                    handlers=handlers,
                    budget=budget,
                    trace=trace,
                    phase="patch",
                )
            except Exception as exc:  # noqa: BLE001 - see comment above
                loop_error = exc
                break

            loop_stopped_because = said.stopped_because
            loop_final_text = said.final_text

            if accepted or tools.guardrail_violations or abandonment:
                break

            # What did it stop short of? Nudging is only honest when there is a
            # concrete next call, and only possible while budget remains.
            missing: list[str] = []
            if tools.edits_since_verify or not tools.last_build_ok:
                missing.append("call build")
            if not tools.last_tests_ok or tools.edits_since_verify:
                missing.append("call run_tests")
            missing.append(
                "call emit_patch with summary, mechanism_changed, files_touched, "
                "expected_effect and risks"
            )
            needs_iterations = sum(
                1 for step in missing if step.startswith("call build")
                or step.startswith("call run_tests")
            )
            budget_left = tools.max_iterations - tools.iterations

            nothing_done = not green.diff().strip()
            if (
                nothing_done
                or terminal_retries >= max_terminal_retries
                or budget_left < needs_iterations
            ):
                break
            terminal_retries += 1
            task = build_continuation_task(
                green.diff(), said if isinstance(said, str) else str(said), missing
            )

        document = _build_document(
            green=green,
            tools=tools,
            candidate=candidate,
            baseline=baseline,
            accepted=accepted,
            rejections=rejections,
            loop_error=loop_error,
            abandonment=abandonment,
            loop_stopped_because=loop_stopped_because,
            loop_final_text=loop_final_text,
            blue_status_before=blue_status_before,
        )
        document["terminal_call_retries"] = terminal_retries

        if document["outcome"] == "ABANDONED":
            # A bad state is a reset, not a debugging session.
            document["reset_to"] = green.reset_to_checkpoint()

        _persist(directory, document, green)
        return document
    finally:
        if not keep_worktree:
            green.remove()


def _measure_baseline(green: GreenWorktree, java_home: str | None,
                      profile: Any = None) -> TestCounts:
    """Build and test the pristine worktree, before the agent touches anything.

    Without this there is no reference for "the test count may not decrease",
    and a build that was already broken would be blamed on the patch.
    """
    import subprocess

    probe = PatchTools(
        green=green, baseline=TestCounts(), java_home=java_home,
        max_iterations=2, profile=profile,
    )
    build_result = probe.build()
    if not probe.last_build_ok:
        raise ContractViolation(
            "the pristine green worktree does not build, so nothing the agent "
            f"does could be evaluated against it.\n{build_result}"
        )

    # NOT `-q`: quiet mode suppresses surefire's "Tests run: N, ..." summary
    # lines -- the exact text `parse_test_counts` greps for. With `-q` this
    # always measured zero tests executed, for every repo, regardless of how
    # many actually ran; see the matching note on PatchTools.run_tests.
    completed = subprocess.run(
        ["mvn", "test"],
        cwd=str(green.path), capture_output=True, text=True, timeout=900,
        env=build_env(java_home),
    )
    counts = parse_test_counts(
        (completed.stdout or "") + (completed.stderr or ""), green.path
    )
    counts.files = count_test_files(green.path)
    return counts


#: The exact string run_agent returns as `stopped_because` when the model's
#: last turn had no tool call in it. Matched literally rather than guessed
#: at, since it is the one case where `final_text` is the model's actual
#: last word rather than stale content from an earlier turn.
_MODEL_STOPPED_TALKING = "the model stopped calling tools"


def _abandoned_reason(
    *, abandonment: dict[str, Any], loop_error: Exception | None,
    tools: PatchTools, loop_stopped_because: str, loop_final_text: str = "",
) -> str:
    """One human-readable sentence for WHY this run has no patch.

    Checked in the order a reader would actually want the answer: did the
    model say why (through abandon_patch, or in its own last words when it
    simply stopped talking instead), did the loop itself blow up, did a
    guardrail fire, did the last build/test attempt fail, and only then --
    the case this existed for -- did the loop simply stop (budget
    exhausted, no progress) with none of the above ever triggering.

    COL-04 is the case `loop_final_text` exists for: the model wrote a
    coherent, specific explanation ("the repository file is the only file
    permitted... the necessary mapping cannot be done within this single
    file") as plain text instead of calling abandon_patch, stop_reason
    end_turn. Before this, that explanation lived only in trace.jsonl --
    patch.json recorded the generic "the model stopped calling tools" and
    nothing else, which is invisible to the person the report is for.
    """
    if abandonment.get("reason"):
        return f"the agent called abandon_patch: {abandonment['reason']}"
    if loop_error is not None:
        return f"the agent loop raised {type(loop_error).__name__}: {loop_error}"
    if tools.guardrail_violations:
        return f"guardrail violation: {tools.guardrail_violations[-1]}"
    if loop_stopped_because == _MODEL_STOPPED_TALKING and loop_final_text.strip():
        return f"the model stopped without submitting; its last words: {loop_final_text.strip()[:600]}"
    if tools.last_error:
        # The first non-empty lines, not just the first line: for a test
        # failure `last_error` now leads with a count header ("3 failing
        # test(s)...") followed by the actual failing test and its
        # assertion message -- line 0 alone named nothing a reviewer could
        # act on, which is exactly what this field was found doing.
        lines = [l for l in tools.last_error.splitlines() if l.strip()]
        return f"last build/test failure: {' | '.join(lines[:3])[:400]}"
    if loop_stopped_because:
        return f"the agent loop stopped on its own: {loop_stopped_because}"
    return "no reason was recorded for this abandonment"


def _build_document(
    *,
    green: GreenWorktree,
    tools: PatchTools,
    candidate: CandidateFix,
    baseline: TestCounts,
    accepted: dict[str, Any],
    rejections: list[str],
    loop_error: Exception | None,
    blue_status_before: str,
    abandonment: dict[str, Any] | None = None,
    loop_stopped_because: str = "",
    loop_final_text: str = "",
) -> dict[str, Any]:
    """Assemble the artifact. Every number is computed here, not reported by
    the model.

    `plan_id` and `targets_hypothesis` come from the SELECTED CANDIDATE, not
    from `hypotheses["ranked_ids"][0]` — that is the diagnosis's own ranking
    by attributed milliseconds, the exact thing `build_task`'s docstring
    says this phase stopped trusting. Reading it here anyway is how a run
    that planned and implemented candidate c1 (approach dto_projection,
    targeting h3) persisted a patch.json claiming it targeted h1 instead.
    """
    files, added, removed = green.diff_stat()
    diff = green.diff()
    patched = bool(accepted) and not tools.guardrail_violations

    # Patch (contracts.py) carries the computed facts. The model's narrative has
    # no field on it, so it travels alongside in the document rather than being
    # bolted onto the contract — see the note in the phase report.
    patch = Patch(
        plan_id=candidate.id,
        branch=green.branch,
        diff=diff,
        files_changed=files,
        lines_added=added,
        lines_removed=removed,
        build_iterations=tools.iterations,
        tests_passed=0,
        tests_before=baseline.executed,
    )

    last_tests = [e for e in tools.log if e["tool"] == "run_tests" and e["ok"]]
    if last_tests:
        patch.tests_passed = baseline.executed

    document: dict[str, Any] = {
        "outcome": "PATCHED" if patched else "ABANDONED",
        "patch": patch.model_dump(mode="json"),
        "targets_hypothesis": candidate.targets_hypothesis,
        "worktree": {
            "path": str(green.path),
            "branch": green.branch,
            "base_commit": green.base_commit,
            "carried_over_from_blue": green.carried_over,
            "skipped_from_blue": green.skipped,
            "checkpoints": green.checkpoints,
        },
        "baseline_tests": baseline.render(),
        "iterations_used": tools.iterations,
        "iterations_allowed": tools.max_iterations,
        "tool_log": tools.log,
        "refusals": tools.refusals,
        "guardrail_violations": tools.guardrail_violations,
        "submissions_rejected": rejections,
        "blue_status_at_start": blue_status_before,
        "blue_status_at_end": green.blue_status(),
    }
    if patched:
        document["emitted"] = accepted["payload"]
    else:
        document["last_error"] = tools.last_error or "no build error recorded"
        document["abandoned_reason"] = _abandoned_reason(
            abandonment=abandonment or {}, loop_error=loop_error, tools=tools,
            loop_stopped_because=loop_stopped_because,
            loop_final_text=loop_final_text,
        )
    if loop_error is not None:
        document["loop_error"] = f"{type(loop_error).__name__}: {loop_error}"
    return document


def _persist(directory: Path, document: dict[str, Any], green: GreenWorktree) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    # What a resume's skip check compares against THIS run's current
    # baseline.json -- a patch built against a baseline that has since been
    # re-measured is not evidence for the current one.
    baseline_path = directory / "baseline.json"
    if baseline_path.exists():
        try:
            document["baseline_hash"] = baseline_hash(
                json.loads(baseline_path.read_text(encoding="utf-8")))
        except json.JSONDecodeError:
            pass
    (directory / "patch.json").write_text(
        json.dumps(document, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (directory / "patch.diff").write_text(
        document["patch"]["diff"] or "", encoding="utf-8"
    )


def _require_api_key() -> str:
    """DEPRECATED. `make_client()` resolves the key for the CONFIGURED
    provider; this function only ever knew about one.

    It used to prefer CEREBRAS_API_KEY because the misleadingly-named
    `GroqClient` posts to Cerebras. Both it and the diagnose phase's copy
    (which had the mirror-image bug, reading GROQ_API_KEY only) now go
    through `make_client`, so the provider comes from `senapathy.yml` and a
    missing key names every variable that would have worked.
    """
    from ..config import Settings
    from ..llm import resolve_key

    from dotenv import load_dotenv

    load_dotenv()
    provider = Settings.load().llm.provider
    key = resolve_key(provider)
    if key is None:
        raise ContractViolation(
            f"the {provider} provider needs no API key, so there is none to "
            f"return; call make_client() instead."
        )
    return key


# ─────────────────────────────  rendering  ──────────────────────────────

def render_patch(document: dict[str, Any], max_diff_lines: int = 120) -> str:
    out: list[str] = ["", "PATCH RUN"]
    worktree = document["worktree"]
    out.append(f"  worktree     {worktree['path']}")
    out.append(f"  branch       {worktree['branch']}")
    out.append(f"  base commit  {worktree['base_commit']}")
    if worktree["carried_over_from_blue"]:
        out.append(
            f"  carried over {len(worktree['carried_over_from_blue'])} uncommitted "
            f"file(s) from the original working copy:"
        )
        for rel in worktree["carried_over_from_blue"]:
            out.append(f"                 {rel}")
    for rel in worktree.get("skipped_from_blue", []):
        out.append(f"  skipped      {rel}")
    out.append(f"  baseline     {document['baseline_tests']}")
    out.append("")

    out.append(f"ITERATIONS ({document['iterations_used']} of "
               f"{document['iterations_allowed']})")
    for entry in document["tool_log"]:
        mark = "ok  " if entry["ok"] else "FAIL"
        args = entry["args"]
        detail = args.get("file") or args.get("counts") or ""
        out.append(f"  [{entry['iteration']}] {mark} {entry['tool']:12} {detail}")
        first = entry["result"].splitlines()
        if not entry["ok"]:
            for line in first[:6]:
                out.append(f"        {line}")
    out.append("")

    if document["refusals"]:
        out.append(f"REFUSALS ({len(document['refusals'])})")
        for refusal in document["refusals"]:
            out.append(f"  {refusal.splitlines()[0]}")
        out.append("")

    if document["guardrail_violations"]:
        out.append("GUARDRAIL VIOLATIONS")
        for violation in document["guardrail_violations"]:
            out.append(f"  {violation}")
        out.append("")

    patch = document["patch"]
    out.append("DIFF")
    out.append(f"  {len(patch['files_changed'])} file(s), "
               f"+{patch['lines_added']} -{patch['lines_removed']}")
    diff_lines = (patch["diff"] or "").splitlines()
    for line in diff_lines[:max_diff_lines]:
        out.append(f"  {line}")
    if len(diff_lines) > max_diff_lines:
        out.append(f"  [... {len(diff_lines) - max_diff_lines} more diff lines; "
                   f"full text in patch.diff ...]")
    out.append("")

    if document["outcome"] == "PATCHED":
        emitted = document["emitted"]
        out.append("EMITTED PATCH")
        out.append(f"  summary            {emitted['summary']}")
        out.append(f"  mechanism changed  {emitted['mechanism_changed']}")
        out.append(f"  expected effect    {emitted['expected_effect']}")
        out.append(f"  files touched      {', '.join(emitted['files_touched'])}")
        for risk in emitted["risks"]:
            out.append(f"  risk               {risk}")
    else:
        out.append("ABANDONED")
        out.append("  no patch was emitted. The worktree has been reset to its "
                   "baseline checkpoint.")
        out.append(f"  reason: {document.get('abandoned_reason', '(not recorded)')}")
        out.append("  last error:")
        for line in str(document.get("last_error", "")).splitlines()[:20]:
            out.append(f"    {line}")
    if document.get("loop_error"):
        out.append("")
        out.append(f"NOTE: the agent loop ended with {document['loop_error']}")
    return "\n".join(out)


# ──────────────────────  the plan this phase implements  ─────────────────


def load_plan_inputs(run_dir: Path, *, index: int = 0
                     ) -> tuple[CandidateFix, EndpointModel, Any]:
    """The selected candidate, the map, and the plan it came from.

    Required, not optional. Without a plan this phase would fall back to the
    hypothesis with the largest attributed milliseconds, which is how three
    runs in a row attacked a bottleneck the oracle could never accept.
    """
    from ..plan.agent import load_plan

    directory = Path(run_dir)
    plan_path = directory / "plan.json"
    if not plan_path.exists():
        raise ContractViolation(
            f"no plan at {plan_path}. Run `senapathy plan` first — this phase "
            f"implements a NAMED CANDIDATE chosen by expected value, not the "
            f"biggest number in the diagnosis."
        )
    plan = load_plan(plan_path)
    # `index` is the retry's doing: candidate 0 is the selection, and a later
    # one is reached only after the one before it failed for a reason the plan
    # could not have known. Candidates the plan ruled out for cause are not
    # in this list at all.
    from ..plan.risk import selection_problem

    candidates = [c for c in plan.candidates if selection_problem(c) is None]
    candidate = (candidates[index] if 0 <= index < len(candidates)
                 else plan.selected())
    if candidate is None:
        raise ContractViolation(
            f"{plan_path} records no selected candidate "
            f"({plan.selection_reason or 'no reason given'}), so there is "
            f"nothing to implement."
        )

    model_path = directory / "endpoint_model.json"
    if not model_path.exists():
        raise ContractViolation(
            f"no comprehension model at {model_path}. The patch prompt names "
            f"the unbounded response collections from it, and writing a patch "
            f"without knowing which fields are frozen is how behaviour breaks."
        )
    model = EndpointModel.model_validate(
        json.loads(model_path.read_text(encoding="utf-8")))
    return candidate, model, plan


def _fixability_reason(plan: Any, candidate: CandidateFix) -> str:
    assessment = plan.assessment_for(candidate.targets_hypothesis)
    return assessment.reason if assessment is not None else ""


def _evidence_block(hypotheses: dict[str, Any], candidate: CandidateFix) -> str:
    """The measured evidence for the hypothesis this candidate attacks."""
    by_id = {h["id"]: h for h in hypotheses.get("hypotheses", [])}
    target = by_id.get(candidate.targets_hypothesis)
    if target is None:
        return ""
    lines = "\n".join(f"     - {e}" for e in target.get("evidence", []))
    return (f"=== THE MEASURED EVIDENCE FOR {target['id']} ===\n\n"
            f"  {target['claim']}\n"
            f"     layer        {target['layer']}\n"
            f"     attributed   {target['attributed_ms']:.0f} ms\n"
            f"     mechanism    {target['mechanism']}\n"
            f"     evidence:\n{lines}\n\n")


def _benchmark_for(green: GreenWorktree, hypotheses: dict[str, Any],
                   java_home: str | None, profile: Any = None) -> Any:
    """The in-loop benchmark, bound to this worktree and this baseline."""
    from .bench import quick_benchmark

    baseline_p50 = hypotheses.get("measured_total_ms")

    def run() -> Any:
        try:
            target = _database_target(profile)
        except Exception:  # noqa: BLE001 — latency alone is still worth having
            target = None
        return quick_benchmark(
            green.path, java_home=java_home, baseline_p50_ms=baseline_p50,
            mysql=target)

    return run
