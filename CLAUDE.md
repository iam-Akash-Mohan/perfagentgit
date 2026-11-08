# Senapathy — invariants for every Claude session in this repo

Read this before writing any code. These rules are not suggestions; they are the
product. `SENAPATHY.md` is the full design doc — read it if you need context beyond
the current phase.

## What this project is

An autonomous agent that takes one API endpoint from a Spring Boot repo, finds the
real performance bottleneck from runtime evidence, fixes it, proves behaviour didn't
change, measures the improvement, and opens a PR.

## Non-negotiable invariants

1. **No shell tool.** The agent never gets `bash`, `subprocess` access, or anything
   equivalent. Every capability is a named tool with a JSON schema and an allowlist.
   The only exception is `run_probe`, which is sandboxed and diagnosis-only.

2. **The oracle is frozen.** Behaviour recordings are captured against unmodified
   code, content-hashed, and are not in any agent's tool surface. No code path may
   let an agent read, widen, or regenerate them.

3. **Test files are immutable.** `src/test/**` in a target repo is write-denied.
   Test count may never decrease. Enforce in code, not in a prompt.

4. **No number without an artifact.** Every latency, count, percentage and file path
   in any report or PR body is templated from a recorded measurement. The model
   writes prose only. If a renderer can emit an unbacked claim, that's a bug.

5. **Never ask the model to count, sum, or compare.** Compute it in Python and hand
   the model the answer. Response equality is decided by code, always.

6. **Verdicts are code.** Behaviour equivalence, statistical significance, and gate
   decisions are computed. The model has no vote.

7. **Blue worktree is read-only.** All writes go to the green worktree.

8. **No framework.** No LangChain, no LlamaIndex, no vector DB. Direct Anthropic
   Messages API. This is deliberate.

## Code conventions

- Python 3.12, `pydantic` v2 for every artifact, `typer` for CLI, `httpx` for HTTP.
- Every artifact is a Pydantic model in `senapathy/contracts.py`. **Never redefine a
  shape inline** — import it. If a phase needs a new field, say so; don't invent one.
- Type hints everywhere. `from __future__ import annotations` at the top of modules.
- No `print()` in library code. Emit trace events; the CLI renders them.
- Tools raise `ToolError` for expected failures; the loop converts those into
  `is_error` tool results. Never let a tool exception kill a run.
- Tests never hit the network. Use the recorded-cassette mechanism from P01.

## Phase discipline

Work is organised into numbered phases in `phases/`. When executing a phase:

- Build **only** what its DELIVERABLES section lists.
- Do not create files belonging to a later phase, even if it seems helpful.
- The phase is done when its ACCEPTANCE TEST passes and no other test regressed.
- If the spec seems wrong, say so and stop. Don't silently redesign.

## The learning constraint

Some phases are marked **HAND-WRITE**. For those, do not produce the implementation.
Explain the mechanics, answer questions, review code that the human wrote, and point
out what's wrong with it. The human is learning AI engineering by building this; a
finished implementation handed over defeats the purpose.

Phases marked HAND-WRITE: P01, P02, P03 (done by hand), P08, P15 (gate
rules). P13 was generated — review compare.py before trusting a verdict.
