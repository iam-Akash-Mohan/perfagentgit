"""The patch agent: read the diagnosis, write a real fix, prove it builds.

Reuses GroqClient (P01) and run_agent (P02) unmodified. run_agent short-circuits
only on `emit_endpoint_model`, so `emit_patch` travels the normal handler path —
which is what we want: the handler refuses a submission that has not built and
passed tests, and the refusal comes back as a tool result the agent can act on.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from ..agent import run_agent
from ..contracts import Patch
from ..errors import ContractViolation
from ..llm import GroqClient
from .tools import (APPLY_PATCH_TOOL, BUILD_TOOL, MAX_ITERATIONS,
                    RUN_TESTS_TOOL, PatchTools, TestCounts, count_test_files,
                    parse_test_counts)
from .worktree import GREEN_DIR, GreenWorktree, build_env

MAX_STEPS = 24

# Shown to the agent in full, because apply_patch needs byte-exact old_text and
# the agent has no read_file tool. ReportService is excluded on purpose: 200
# dead one-line methods that would crowd out everything that matters.
SOURCE_CONTEXT = (
    "src/main/java/com/ex/web/CustomerController.java",
    "src/main/java/com/ex/service/CustomerService.java",
    "src/main/java/com/ex/service/InvoiceService.java",
    "src/main/java/com/ex/repo/CustomerRepository.java",
    "src/main/java/com/ex/repo/InvoiceRepository.java",
    "src/main/java/com/ex/domain/Customer.java",
    "src/main/java/com/ex/domain/Invoice.java",
    "src/main/java/com/ex/dto/CustomerDto.java",
    "src/main/java/com/ex/dto/SearchRequest.java",
)


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


# ──────────────────────────────  the prompt  ─────────────────────────────
# run_agent has no system-role support, so instructions and evidence travel
# together in the opening user message. Every number here is read from
# hypotheses.json; none is written by hand.

def _render_source(green: Path) -> str:
    blocks: list[str] = []
    for rel in SOURCE_CONTEXT:
        path = green / rel
        if not path.exists():
            continue
        blocks.append(f"----- {rel}\n{path.read_text(encoding='utf-8').rstrip()}")
    return "\n\n".join(blocks)


def build_task(hypotheses: dict[str, Any], green: Path) -> str:
    ranked = hypotheses["ranked_ids"]
    by_id = {h["id"]: h for h in hypotheses["hypotheses"]}
    top = by_id[ranked[0]]
    p50 = hypotheses["measured_total_ms"]
    others = [by_id[i] for i in ranked[1:]]

    evidence = "\n".join(f"     - {e}" for e in top["evidence"])
    runners_up = "\n".join(
        f"  [{h['id']}] {h['claim']}  ({h['attributed_ms']:.0f} ms attributed, "
        f"layer {h['layer']})"
        for h in others
    )

    return f"""You are a performance engineer. A diagnosis phase has already measured this
endpoint and ranked the causes. Your job is to write the code change that removes
the top-ranked cause — and to prove it compiles and passes the existing tests.

=== THE MEASURED DIAGNOSIS ===

Measured p50 latency: {p50:.1f} ms

TOP-RANKED CAUSE — this is what you are fixing:
  [{top['id']}] {top['claim']}
     layer        {top['layer']}
     attributed   {top['attributed_ms']:.0f} ms of the {p50:.1f} ms p50
     confidence   {top['confidence']:.2f}
     mechanism    {top['mechanism']}
     prediction   {top['prediction']}
     evidence:
{evidence}

Also diagnosed, lower ranked:
{runners_up}

=== THE HARD CONSTRAINT — READ THIS TWICE ===

THE HTTP RESPONSE MUST STAY BYTE-IDENTICAL.

The response body is large by design. Making it smaller is a BEHAVIOUR CHANGE and
will be rejected by a frozen oracle that compares every byte of every response
against a recording made before your patch. That oracle is not in your tool
surface and you cannot see, widen or regenerate it.

Concretely, all of these will FAIL:
  - removing or renaming a field in the response
  - returning fewer rows, or a different page shape
  - changing a number's formatting or scale — BigDecimal("0") and
    BigDecimal("0.00") serialise differently and that counts as a change
  - reordering the elements of the tags array
  - changing a 400 response into anything else

Note that the top-ranked hypothesis PREDICTS that a slimmer response would be
faster. That prediction is correct and it is off-limits. Your job is the harder
one: PRODUCE EXACTLY THE SAME BYTES WITH LESS WORK.

=== WHAT YOU MAY AND MAY NOT TOUCH ===

You may edit ONLY files under src/main/java. Anything else — the build file,
configuration, resources, and above all src/test/** — is refused in code, not
by convention. Test files are immutable: they are the evidence that behaviour
did not change.

You must NOT:
  - add caching of any kind
  - change transaction boundaries
  - alter the API contract (paths, request shape, response shape, status codes)
  - modify the database schema, beyond adding an index

=== DIRECTIONS WORTH CONSIDERING ===

These are directions, not a checklist. Pick what the evidence supports; you may
do something not listed here if you can justify it from the measurements.

  - eliminating repeated queries, e.g. a fetch join or @EntityGraph so a
    collection arrives with its parent instead of one query per row
  - projecting only the columns actually used, instead of hydrating whole
    entities you then throw away
  - avoiding repeated work inside per-row loops — anything computed once
    outside the loop is not computed N times inside it
  - more efficient string construction

=== THE CODE, EXACTLY AS IT IS ON DISK ===

apply_patch needs old_text that matches BYTE FOR BYTE. Copy your snippets from
below rather than retyping them — a retyped snippet with different indentation
is the single most common way this fails.

{_render_source(green)}

=== HOW TO WORK ===

1. Decide on one change that attacks the top-ranked cause without changing bytes.
2. apply_patch it.
3. build. If it fails you get the compiler output verbatim — read it literally,
   it names the file and the line. Fix and build again.
4. run_tests. It requires a successful build first.
5. Only when build AND run_tests have both succeeded, call emit_patch.

You have {MAX_ITERATIONS} build/test iterations in total, shared between build and
run_tests. Spend them on the problem, not on retyping snippets. If you run out,
the run is abandoned and no patch is emitted — an honest abandonment is a better
outcome than a patch that does not build.

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

def resolve_run_dir(runs_dir: Path = Path(".perfagent/runs")) -> Path:
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
            "`perfagent diagnose` first — the patch agent fixes a measured "
            "cause, not a guessed one."
        )
    return candidates[0]


def load_hypotheses(run_dir: Path, hypotheses_path: Path | None = None) -> dict[str, Any]:
    """Read the diagnosis, with an actionable error rather than a traceback."""
    path = hypotheses_path or (Path(run_dir) / "hypotheses.json")
    if not path.exists():
        raise ContractViolation(
            f"no diagnosis at {path}. Run `perfagent diagnose` first, or point "
            "--run-dir at the run that produced hypotheses.json."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def run_patch(
    run_dir: Path,
    *,
    run_id: str,
    repo: str | Path | None = None,
    green_dir: str | Path = GREEN_DIR,
    hypotheses_path: Path | None = None,
    max_steps: int = MAX_STEPS,
    max_iterations: int = MAX_ITERATIONS,
    max_terminal_retries: int = 2,
    java_home: str | None = None,
    client: Any = None,
    keep_worktree: bool = True,
) -> dict[str, Any]:
    """Create green, measure the pre-patch baseline, let the agent patch, persist.

    Returns a document with `outcome` either "PATCHED" or "ABANDONED". An
    abandoned run resets the worktree and reports the last error rather than
    emitting a patch that does not build.
    """
    directory = Path(run_dir)
    hypotheses = load_hypotheses(directory, hypotheses_path)

    blue_status_before = None
    green = GreenWorktree.create(run_id, repo=repo, green=green_dir)
    blue_status_before = green.blue_status()

    try:
        baseline = _measure_baseline(green, java_home)
        tools = PatchTools(
            green=green,
            baseline=baseline,
            java_home=java_home,
            max_iterations=max_iterations,
        )

        accepted: dict[str, Any] = {}
        rejections: list[str] = []

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

            accepted["payload"] = payload
            return "accepted: the patch builds and the tests pass. You are done — stop now."

        llm = client or GroqClient(api_key=_require_api_key())

        # As in P09: once code has ACCEPTED a submission the work is durable.
        # run_agent keeps going after a terminal tool returns, and P01 raises on
        # any non-200 from the provider, so a rate limit one step after a good
        # emit would otherwise throw away a patch that already passed every check.
        handlers = {
            "apply_patch": tools.apply_patch,
            "build": tools.build,
            "run_tests": tools.run_tests,
            "emit_patch": emit_patch,
        }
        tool_set = [APPLY_PATCH_TOOL, BUILD_TOOL, RUN_TESTS_TOOL, EMIT_PATCH_TOOL]

        loop_error: Exception | None = None
        task = build_task(hypotheses, green.path)
        terminal_retries = 0
        while True:
            try:
                said = run_agent(
                    client=llm,
                    task=task,
                    tools=tool_set,
                    handlers=handlers,
                    max_steps=max_steps,
                )
            except Exception as exc:  # noqa: BLE001 - see comment above
                loop_error = exc
                break

            if accepted or tools.guardrail_violations:
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
            hypotheses=hypotheses,
            baseline=baseline,
            accepted=accepted,
            rejections=rejections,
            loop_error=loop_error,
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


def _measure_baseline(green: GreenWorktree, java_home: str | None) -> TestCounts:
    """Build and test the pristine worktree, before the agent touches anything.

    Without this there is no reference for "the test count may not decrease",
    and a build that was already broken would be blamed on the patch.
    """
    import subprocess

    probe = PatchTools(
        green=green, baseline=TestCounts(), java_home=java_home, max_iterations=2
    )
    build_result = probe.build()
    if not probe.last_build_ok:
        raise ContractViolation(
            "the pristine green worktree does not build, so nothing the agent "
            f"does could be evaluated against it.\n{build_result}"
        )

    completed = subprocess.run(
        ["mvn", "-q", "test"],
        cwd=str(green.path), capture_output=True, text=True, timeout=900,
        env=build_env(java_home),
    )
    counts = parse_test_counts(
        (completed.stdout or "") + (completed.stderr or ""), green.path
    )
    counts.files = count_test_files(green.path)
    return counts


def _build_document(
    *,
    green: GreenWorktree,
    tools: PatchTools,
    hypotheses: dict[str, Any],
    baseline: TestCounts,
    accepted: dict[str, Any],
    rejections: list[str],
    loop_error: Exception | None,
    blue_status_before: str,
) -> dict[str, Any]:
    """Assemble the artifact. Every number is computed here, not reported by
    the model."""
    files, added, removed = green.diff_stat()
    diff = green.diff()
    patched = bool(accepted) and not tools.guardrail_violations

    # Patch (contracts.py) carries the computed facts. The model's narrative has
    # no field on it, so it travels alongside in the document rather than being
    # bolted onto the contract — see the note in the phase report.
    patch = Patch(
        plan_id=hypotheses["ranked_ids"][0],
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
        "targets_hypothesis": hypotheses["ranked_ids"][0],
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
    if loop_error is not None:
        document["loop_error"] = f"{type(loop_error).__name__}: {loop_error}"
    return document


def _persist(directory: Path, document: dict[str, Any], green: GreenWorktree) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "patch.json").write_text(
        json.dumps(document, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (directory / "patch.diff").write_text(
        document["patch"]["diff"] or "", encoding="utf-8"
    )


def _require_api_key() -> str:
    """The key for the endpoint GroqClient actually posts to.

    llm.py posts to https://api.cerebras.ai, so a Cerebras key (`csk-...`) is
    what that endpoint accepts; a Groq key (`gsk_...`) comes back as HTTP 401
    "Wrong API Key". CEREBRAS_API_KEY is therefore preferred, with GROQ_API_KEY
    kept as a fallback for whenever the client is pointed back at Groq.

    NOTE: perfagent/diagnose/agent.py::_require_api_key has the same bug and
    reads GROQ_API_KEY only. It is left alone here because P09 is not this
    phase; it needs the same three-line fix.
    """
    from dotenv import load_dotenv

    load_dotenv()
    for name in ("CEREBRAS_API_KEY", "GROQ_API_KEY"):
        key = os.environ.get(name)
        if key:
            return key
    raise ContractViolation(
        "neither CEREBRAS_API_KEY nor GROQ_API_KEY is set (put one in .env). "
        "llm.py posts to api.cerebras.ai, so it needs the csk-... key."
    )


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
        out.append("  last error:")
        for line in str(document.get("last_error", "")).splitlines()[:20]:
            out.append(f"    {line}")
    if document.get("loop_error"):
        out.append("")
        out.append(f"NOTE: the agent loop ended with {document['loop_error']}")
    return "\n".join(out)
