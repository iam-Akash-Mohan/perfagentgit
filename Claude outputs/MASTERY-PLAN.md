# PerfAgent — Component Mastery & Production Plan

From "a pipeline that runs" to "a product that ships the best optimization it can find,
and that you can defend line by line."

Ten sessions. Each one makes you the owner of a component and makes that component
production-grade.

---

## 1. The two goals, stated precisely

**Product goal.** Given an endpoint in a Spring Boot repo, PerfAgent finds the best
available optimization — service-layer *and* database-layer, including query rewrites,
index additions, and EXPLAIN-driven plan analysis — implements it to your team's Java
standards, proves behaviour is unchanged, measures the gain, and opens a PR a reviewer
can merge without redoing the work.

**Learning goal.** You can explain every component, run it standalone, debug it when it
misbehaves, and defend why it is shaped the way it is.

Both are served by the same work: you cannot harden what you don't understand, and you
don't truly understand what you haven't changed.

---

## 2. Session protocol

Every session runs the same four beats.

**Beat 1 — I explain, you read.**
I read the actual files first, then explain: what the component does, a worked example
with real values from your runs, how it does it, and where it is weak. I name the exact
files and line ranges to read, in the order that makes sense. You read them and ask
until nothing is fuzzy.

**Beat 2 — We decide what "best" means.**
Concrete changes: a new tool, a changed prompt, a different validation, a new artifact
field, a refactor. We argue about it. You bring what you know about Growfin's codebase —
that's the input I don't have.

**Beat 3 — One prompt for Claude Code.**
Covers everything from Beat 2, plus the standing quality bar (§4). You paste it, it
executes, you review the result with me.

**Beat 4 — You learn to drive it.**
The exact command, what each flag does, what it reads, what it writes, what a healthy
run looks like, and the three ways it most commonly fails. You run it yourself.

A session ends when you can explain the component to someone else without notes.

---

## 3. What changes structurally

Three things become first-class that aren't today.

### 3.1 Database optimization as a capability

Today the patch agent only edits Java. Real endpoint optimization is often *not* Java —
it's a missing composite index, a query fetching columns nobody reads, a JOIN the
planner is getting wrong.

New component (Session 4): a **DB optimization toolkit** the agent can use to *test*
ideas rather than assert them.

```
list_indexes(table)              what already exists — never propose a duplicate
explain(sql, params)             EXPLAIN ANALYZE: real rows, real timings, real plan
index_benefit(table, columns)    create the index in a shadow schema, re-EXPLAIN,
                                 report the delta, drop it. An empirical answer.
query_equivalent(old, new)       run both, compare result sets exactly. A rewrite
                                 that returns different rows is not a rewrite.
```

`index_benefit` is the one that matters. It converts "an index here would probably
help" into "this index takes the plan from a 1.2M-row scan to a 40-row lookup, measured."
That is the difference between a suggestion and an engineering recommendation.

Index changes ship as a proper migration file (Flyway), not raw DDL.

### 3.2 Java code standards, as data

A file at the repo root — `standards/java.md` — stating your team's rules:

```markdown
# Java standards for generated patches

## Required
- Constructor injection only. No field @Autowired.
- Repository methods return DTOs or projections, never entities, above the service layer.
- @Transactional(readOnly = true) on read paths.
- Every new query method has a Javadoc line stating what it returns.

## Forbidden
- No new static mutable state.
- No `new` on a Spring-managed type.
- No raw JDBC outside the repository layer.

## Style
- Prefer `Optional<T>` over null returns.
- Streams over index loops when it reads more clearly.
```

Loaded into the patch agent's prompt at runtime, and — where the rule is mechanically
checkable — enforced by a linter pass on the diff, so it is a gate rather than a hope.

You edit that file to change agent behaviour. No Python involved. That property is what
makes it usable by a team rather than by you alone.

### 3.3 Candidate search instead of single-shot

Today: one hypothesis → one fix → verify → measure → gate.

After Session 8: several candidates → implement each → verify each → measure each →
keep the winner → attempt to stack the winner with a compatible second fix → repeat
until nothing improves.

That is the difference between "an optimization" and "the best optimization I could
find", and it is the honest answer to what you asked for.

---

## 4. Standing quality bar

Every session's Claude Code prompt carries these. Not restated each time.

- **Modularity.** One module, one responsibility. No file over ~400 lines. This
  component's CLI moves into its own module — `cli.py` is 30KB today and will be
  decomposed session by session.
- **Types.** Full annotations. `mypy --strict` clean. Artifacts are Pydantic models
  from `contracts.py`, never shapes invented inline.
- **Errors.** `ToolError` for expected failures; tool handlers never raise into the
  loop. Error text is written for the model to act on.
- **Tests.** Every session adds tests and keeps the existing suite green. No network,
  no DB in unit tests. The whole suite runs in under 30 seconds.
- **No print in library code.** Trace events; the CLI renders.
- **Docs.** Each component gets a `docs/<component>.md`: what it does, its inputs and
  outputs, how to run it, its failure modes. This doubles as your pitch material.
- **Invariants.** The eight in `CLAUDE.md` hold. Any session that would weaken one
  stops and raises it instead.

---

## 5. The ten sessions

### Session 1 — Comprehension · `perfagent understand`

**State:** exists as a script, never packaged. `.perfagent/endpoint_model.json` was
hand-rebuilt after a test destroyed it. The pipeline currently cannot run on a new
endpoint.

**What we build:**
- A real phase runner: route in, validated `EndpointModel` out
- **Schema awareness** — the model must carry, per entity: table name, columns actually
  read, existing indexes, row-count estimate. Session 3 cannot reason about indexes if
  Session 1 never looked at the schema.
- Stronger citation validation: the symbol must be *on* the cited line (± a 2-line
  window for annotations and wrapped signatures), not merely a line that exists
- Explicit uncertainty capture for interfaces with multiple implementations,
  conditional beans, and AOP

**Deliverable:** `perfagent understand --repo X --endpoint "POST /y"` works on any
endpoint, offline via cassettes in tests.

**After this you can:** explain how an agent explores an unfamiliar codebase, why
forced structured output is a better stop condition than a step limit, and how model
claims get checked mechanically.

---

### Session 2 — Evidence · `perfagent baseline`

**State:** works. Latency, SQL digests, warmup, variance. Missing everything the
database needs.

**What we build:**
- **`EXPLAIN ANALYZE` per top digest**, captured into the evidence bundle
- Index-usage facts: `no_index_used`, rows examined ÷ rows returned, full-scan flags
- A **CPU/allocation profile** (JFR or async-profiler) so non-SQL mechanisms become
  provable. Today `MECHANISM_MATCH` fails closed on anything that isn't SQL, which
  structurally forbids serialization and allocation fixes from ever passing the gate.
- `EvidenceBundle.render_for_model()` extracted properly from P08 — capped, digested,
  zero raw logs
- Fix: `LatencyStats` returned as a model where a dict was expected; baseline written
  to a sibling directory; USD rate config so cost reporting stops printing zero

**Deliverable:** evidence rich enough that a DBA would recognise it as a real
performance report.

**After this you can:** explain why measurement precedes diagnosis, why the model is
told the count instead of asked to count, and what makes a benchmark trustworthy —
warmup, interleaving, variance gating, pinned resources.

---

### Session 3 — Diagnosis & Plan · `perfagent diagnose`, `perfagent plan`

**State:** diagnosis works and ranks badly. Plan (P11) does not exist. This is why
three consecutive runs all attacked the serializer and all lost.

**What we build:**
- **Expected-value ranking.** Not `attributed_ms` alone but `attributed_ms × P(safe fix
  exists)`. Under this, an 11% N+1 with a clean fix outranks a 78% response-size problem
  that the byte-identical constraint forbids touching.
- **DB-aware hypothesis layers:** `missing_index`, `suboptimal_plan`, `over_fetching`,
  `n_plus_one`, `unbounded_per_row_work` — each with its own evidence signature
- **The plan phase.** 3–4 concrete candidates per run, each with: mechanism, files, the
  measurable prediction, risk tier from a fixed table, and what could break
- **Risk tiers in code**, not model judgement:
  - T1 index, `@EntityGraph`, `readOnly`, projection
  - T2 query rewrite, batching, predicate pushdown
  - T3 flow restructuring, transaction boundaries, caching → feature-flagged
  - T4 forbidden: schema beyond additive index, API contract, auth, tests
- **Fixability assessment** — a candidate that cannot preserve bytes is marked
  `unfixable_under_constraint` and stops consuming runs

**Deliverable:** `perfagent plan` emits tiered candidates; the top one is fixable by
construction.

**After this you can:** explain grounding, forced hypothesis competition, why probes
must declare what they discriminate between, and the "right answer, wrong reason"
failure that a 12% win can hide.

---

### Session 4 — DB optimization toolkit *(new component)*

**State:** does not exist.

**What we build:** the four tools in §3.1, plus:
- A **shadow schema** — a clone of the production schema with statistics but no data
  movement, where candidate indexes can be created and dropped safely
- Flyway migration generation for any index that survives
- Guardrails: additive indexes only, never on a table not on the endpoint's path, never
  a duplicate of an existing index, size estimate reported before proposing

**Deliverable:** the agent can say *"this composite index changes the plan from a
1,120,858-row scan to a 42-row index lookup"* and have measured it.

**After this you can:** explain how an agent runs experiments rather than retrieving
facts, and where a controlled exception to "no shell" is justified and what constraints
make it safe.

---

### Session 5 — Patch agent & Java standards · `perfagent patch`

**State:** works, single-shot, no standards awareness, `mvn test` runs zero tests so the
test guardrail is vacuous.

**What we build:**
- `standards/java.md` loaded into the prompt; mechanically checkable rules enforced as a
  post-diff lint pass
- **Multi-candidate implementation** — implement candidate N in its own worktree, so
  Session 8's search has something to search over
- **Measure inside the loop** — the agent sees its own latency delta before emitting,
  instead of finding out three phases later
- DB tools from Session 4 wired in
- Fix surefire so tests actually run; make the test-count guardrail real
- Continuation handling when the model stops one call short (already patched — make it
  principled)

**Deliverable:** a patch agent that writes standards-compliant Java *and* SQL/migrations,
and knows whether its own change helped.

**After this you can:** explain autonomous code modification under capability
restriction, why errors are the highest-value context an agent receives, and why
`apply_patch` is exact-string replacement rather than whole-file rewriting.

---

### Session 6 — The oracle · `perfagent oracle`, `perfagent verify`

**State:** the strongest component in the codebase. Deep dive is mostly comprehension.

**What we cover and harden:**
- Read `record.py`, `canonicalise.py`, `compare.py` end to end
- The hole it closed: caller-supplied tolerance was a comparison the caller could win
- DB-state hashing — why `COUNT(*)` plus two aggregates, and why XOR alone is not enough
- Index changes must leave data identical — assert that explicitly
- Tolerance governance: every entry requires a written rationale; a test pins the policy
  so it cannot widen quietly
- Held-back coverage: is 3-of-12 enough, and what does it actually catch

**Deliverable:** you can defend the trust boundary to a skeptical reviewer.

**After this you can:** explain verification versus evaluation, why a grader the agent
can touch is not a grader, and what golden-master testing buys and costs.

---

### Session 7 — Gate, report, ship · **first green PR**

**State:** gate works and has never passed. `ship.py` has never run.

**What we build:**
- Wire `MECHANISM_MATCH` to Session 2's profiler so non-SQL fixes become provable
- `p_value_paired` as a real contract field; the gate reads Wilcoxon
- PR body a reviewer merges without redoing the work: bottleneck, root cause,
  alternatives rejected and why, before/after with methodology, mechanism confirmation,
  the EXPLAIN plan diff for DB changes, the full trace link
- Evidence ledger hardening — the hardcoded-numeral check was a real catch; make sure
  it covers tables, headings, and the narrative
- **Run it to a real PR on mini-spring**

**Deliverable:** an opened PR you would merge.

**After this you can:** explain provenance as an architectural property, and why the
gate is code with fixed thresholds and the model has no vote.

---

### Session 8 — Orchestrator & candidate search · `perfagent run`

**State:** works, resumable, single-candidate.

**What we build:**
- The **search loop** from §3.3 — implement, verify, measure, rank, stack, repeat
- A **leaderboard artifact**: every candidate tried, its verdict, its delta, why it lost
- Retry on oracle rejection with the diff as new evidence
- Budget-aware search: stop when the remaining budget can't fund another candidate
- Split `orchestrator.py` (34KB) into phase specs, transition guards, and the runner

**Deliverable:** `perfagent run` returns the best optimization it could find, with the
alternatives it beat recorded.

**After this you can:** explain workflow versus agent, why the FSM was built last and
not first, and how to design an optimization search that terminates.

---

### Session 9 — Evaluation · `perfagent eval`

**State:** does not exist. This is the gap between "it worked once" and "I know how
often it works."

**What we build:**
- **Ten scenarios** on mini-spring with a ground-truth answer key the agent cannot
  reach: N+1, missing index, in-memory filtering, redundant call, over-fetching, bad
  pagination, suboptimal join, one 80/20 two-problem case, and **two negative controls**
  where the correct answer is "nothing to do"
- Each scenario run **three times** — agents are stochastic and one run proves nothing
- Metrics: diagnosis accuracy, ranking accuracy, evidence validity, fix validity,
  minimality, negative-control precision, stop discipline, cost, wall clock, **variance**
- A scorecard artifact

**Deliverable:** a number you can say out loud. *"8 of 10 correct diagnoses, both
negative controls clean, zero behaviour regressions, mean $0.40 and 9 minutes per run."*

**After this you can:** explain how to build reliable evaluations for AI systems — the
single skill that most separates AI engineers from people who call APIs. Skipping this
session means you have opinions where you could have evidence.

---

### Session 10 — Production & pitch

**What we build:**
- **`AnthropicClient`** implementing `LLMClient` — retries, prompt caching, cost
  accounting, cassettes, extended thinking for diagnosis. Run the eval suite on both
  models and compare accuracy and cost. That comparison is itself a strong slide.
- **Model routing** — expensive model for diagnosis and planning, cheap for mechanical
  summarisation
- **Code index (tree-sitter)** — retrofit, then measure tokens/steps/dollars against the
  numbers from Session 1. That measured delta is your proof that tool design beats prompt
  engineering.
- **Repo profile + feasibility gate** — AI infers how to build and run a repo, code
  verifies by executing it. This is what lets it onboard a Growfin service.
- Final decomposition of `cli.py`; delete `test.py` and `Untitled`
- **Pitch pack:** one-page summary, the successful-run trace, the *declined*-run trace,
  the eval scorecard, cost per run, and an honest limits section

**Deliverable:** ready to point at a Growfin service, and ready to present.

**After this you can:** explain the whole system, its measured performance, and its
limits — which is what makes a pitch credible rather than enthusiastic.

---

## 6. Sequencing and why

```
1 comprehension ──► 2 evidence ──► 3 diagnosis+plan ──► 4 db tools
                                                          │
                                          5 patch+standards
                                                          │
                       6 oracle ──► 7 gate+PR ──► 8 search loop
                                                          │
                                          9 evals ──► 10 production
```

**Sessions 1–2 first** because everything downstream consumes their artifacts. A better
patch agent reasoning over weak evidence is a faster wrong answer.

**Session 3 before 4** because the plan decides *what kind* of fix to attempt; the DB
tools then make that kind of fix testable. Reversed, you'd build tools without knowing
which ones matter.

**Session 6 (oracle) late** because it already works. It's a comprehension session, and
comprehension lands better once you've watched the component reject three of your own
patches.

**Session 7 is the milestone.** First green PR. Everything before it is setup;
everything after is rigor.

**Session 9 is the one to protect.** It is the least fun and the most valuable, and it
is the session that makes the difference between pitching an anecdote and pitching a
result. If the schedule slips, cut scope inside sessions — never cut Session 9.

---

## 7. Rules I will hold myself to

- I read your actual code before every session. No explaining from design memory.
- Every explanation uses real values from your runs, not invented examples.
- I tell you when a component is fine and needs no change, rather than inventing work.
- I flag when a change would weaken an invariant, and refuse rather than quietly comply.
- If a session's plan turns out wrong once I read the code, I say so and re-plan.

## 8. Rules for you

- Read the files I name before Beat 2. The discussion is worthless otherwise.
- Run the command yourself at the end of every session. Watching Claude Code run it
  doesn't count.
- Keep a one-line log per session: what changed, what surprised you.
- Push back. Three of the best decisions this project has made came from you disagreeing
  — the v2-versus-flag question, "why not batch the query", and asking how I could
  explain code I hadn't read.

---

## 9. Where this lands

**The product:** finds service-layer and DB-layer optimizations, validates index and
query changes empirically with EXPLAIN, writes Java to your team's standards, proves
behaviour byte-identical, measures with paired statistics, opens a PR with the evidence
attached — and declines honestly, with a report, when there is nothing safe to win.
Success rate measured, not asserted. Ready to pilot on one Growfin service.

**You:** able to explain and debug every component; fluent in agent loops, tool design,
context management, grounding, verification, guardrails as capability restriction,
evidence discipline, and evaluation. Able to build a different agent from scratch, and —
more usefully — able to tell when someone else's agent is a demo.

---

*Session 1 starts whenever you're ready. I'll read `tools.py`, `agent.py`, `tryllm.py`,
`contracts.py` and the diagnose package first, then explain what's there.*
