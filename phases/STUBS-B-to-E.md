# Phase stubs — Milestones B through E

Written in full one milestone ahead (see `IMPLEMENTATION.md` §5). These stubs fix the
goal, deliverables and dependencies so you can plan; the full prompts get written when
you reach them, informed by what Milestone A actually produced.

Phases marked ⭐ are on the thin thread to a first working version.

---

## Milestone B — evidence, zero AI

### ⭐ P05 — Environment · generate · ~3h

**Goal:** two git worktrees, both apps built and running on separate ports, health
checked.

**Deliverables:** `perfagent/env/worktree.py`, `perfagent/env/runner.py`,
`perfagent/env/health.py`, `tests/test_worktree.py`.

**Spec highlights:** `blue/` pinned at base commit and **never written to**; `green/`
for the agent. Build via a fixed command (no shell tool — an argv list). Identical JVM
flags on both arms, `-Xms == -Xmx`. Boot, poll `/actuator/health`, fail fast with the
app's own stderr on timeout.

**Test:** create worktrees in a temp git repo, assert blue is read-only-enforced,
assert a build failure surfaces compiler output verbatim, assert health-check timeout
produces a useful error.

**Learning:** giving an AI controlled access to an execution environment. The
environment is a *capability you hand out*, not a place the agent lives.

---

### ⭐ P06 — Workload · generate · ~3h

**Goal:** make the endpoint genuinely slow, and define the request corpus.

**Deliverables:** `seed/seed_<domain>.py` (hand-written for the first repo),
`perfagent/workload/corpus.py`, `tests/test_corpus.py`.

**Spec highlights:** fixed random seed — byte-identical data every run, or before/after
numbers are meaningless. Skewed FK distribution (Zipf), not uniform — uniform hides
per-row fan-out. Corpus split: benchmark set + **held-back set** never used for
measurement. Include error-path requests: a 400 becoming a 500 is a behaviour change.
Corpus is content-hashed.

**Learning:** the train/test split for agents. Why realistic data *shape* matters more
than volume.

---

### ⭐ P07 — Benchmark harness · mixed · ~5h

**Goal:** trustworthy latency and SQL evidence. **The highest-risk phase in the
project.**

**Deliverables:** `perfagent/measure/bench.py`, `perfagent/measure/sql.py`,
`perfagent/measure/variance.py`, `tests/test_bench.py`.

**Spec highlights:** warmup pass, discarded. Percentiles from raw samples, not
averages of averages. SQL via MySQL `performance_schema.events_statements_summary_by_digest`
(digests, `COUNT_STAR`, `SUM_TIMER_WAIT`, `SUM_ROWS_EXAMINED`, `SUM_NO_INDEX_USED`)
plus datasource-proxy for per-request attribution. `EXPLAIN ANALYZE` on the top
digests. **Variance gate:** run the baseline twice; if the two runs differ by more
than a threshold, abort with `ENVIRONMENT_INFEASIBLE` rather than produce garbage.

**Test:** synthetic latency arrays → assert percentile maths; a recorded perf_schema
snapshot → assert aggregation; assert the variance gate trips on injected noise.

**Learning:** you cannot evaluate an agent without a trustworthy oracle, and building
oracles is unglamorous. Expect to lose time here. That's the phase working, not
failing.

---

### P08 — Evidence bundle + red flags · **HAND-WRITE** · ~3h

**Goal:** turn 12,000 lines of raw evidence into the ~40 lines a model will see.

**Deliverables:** `perfagent/measure/evidence.py`, `EvidenceBundle.render_for_model()`,
`tests/test_red_flags.py`.

**Spec highlights:** every `RedFlagKind` computed in Python with an explicit threshold
you chose and can defend. `render_for_model()` is capped and contains no raw logs. The
model is told the count; it is never asked to count.

**Test:** hand-built `SqlDigestStat` fixtures → assert each red flag fires exactly when
it should and not otherwise. Assert `render_for_model()` stays under a token budget.

**Learning:** this function decides what the model can possibly conclude. Get it wrong
and no prompt saves you. Deferred out of the thin thread only as a crude inline
version inside P09 — extract it properly right after.

---

## Milestone C — diagnosis and patching

### ⭐ P09 — Diagnosis agent · mixed · ~5h

**Goal:** ranked, cited, falsifiable hypotheses about the real bottleneck.

**Deliverables:** `perfagent/agents/diagnose.py`, `emit_hypotheses` terminal tool,
`perfagent/validate.py` extension for evidence refs, `tests/test_diagnose.py`.

**Spec highlights:** `min_length=3` on hypotheses is already in the contract. Validate:
every evidence ref resolves to a real artifact, and `attribution_valid()` holds — if
the hypotheses claim 4000ms of a 1980ms request, reject and re-ask. Terminal state
`NO_BOTTLENECK_FOUND` when nothing clears the bar.

**Test:** a hand-built `EvidenceBundle` with a planted N+1 signature → assert the
top-ranked hypothesis names it. A clean bundle → assert `NO_BOTTLENECK_FOUND`. A
bundle with one dominant and one minor problem → **assert the ranking is right.**

**Learning:** grounding, forced competition between hypotheses, and the failure mode
where the agent finds *a* real bottleneck, ships a 12% win, and misses the 78% one —
a failure that looks like success.

---

### P10 — `run_probe` · generate · ~3h

**Goal:** let the diagnosis agent run experiments, not just read.

**Spec highlights:** the one open-ended exec tool in the system. Network-isolated,
read-only source mount, timeout-capped, blue arm only. Every probe must declare
`discriminates_between` — a probe that separates nothing is wandering, not science.

**Learning:** agents doing experiments. Where a controlled exception to "no shell" is
justified, and what constraints make it safe.

---

### P11 — Plan + risk tiers · mixed · ~3h

**Goal:** 2–4 candidate fixes, tiered by risk in code, one selected.

**Spec highlights:** `RiskTier` assigned by a **fixed table**, not by the model.
Tier 4 rejected at plan time and re-checked at gate time. `files_to_touch` becomes
P12's write allowlist. Tier 3 sets `requires_flag=True` (feature flag, not `/v2` —
see design doc §7.3).

**Learning:** human/AI boundaries encoded as policy tables. The model proposes; code
decides what's allowed.

---

### ⭐ P12 — Patch loop · mixed · ~5h

**Goal:** the agent writes a diff that builds and passes tests.

**Deliverables:** `perfagent/tools/write.py` (`apply_patch`, `build`, `run_tests`),
`perfagent/agents/implement.py`, `tests/test_apply_patch.py`.

**Spec highlights:** writes confined to the plan's `files_to_touch`. `src/test/**`
**write-denied** — enforced in code, not prompt. Test count may not decrease. Per-
iteration git checkpoint so a bad state is a reset, not a debugging session. Hard
iteration cap. Compiler output fed back **verbatim** — it's the highest-value context
the agent will ever get.

**Test:** attempt a write outside the allowlist → refused + traced. Attempt to touch a
test file → refused + traced. A patch that fails to compile → assert the error text
reaches the model. Assert checkpoint restores cleanly.

**Learning:** autonomous code modification under capability restriction. Errors as
context.

---

## Milestone D — trust and ship

### ⭐ P13 — The oracle · **HAND-WRITE** · ~5h

**Goal:** prove behaviour didn't change. **The trust boundary of the product.**

**Deliverables:** `perfagent/oracle/record.py`, `canonicalise.py`, `compare.py`,
`tests/test_oracle.py`.

**Spec highlights:** record against unmodified code, content-hash, **not in any
agent's tool surface**. Tolerance policy written before the patch exists — writing it
after seeing a diff is rationalising, not verifying. Compare: body (canonicalised),
status, error paths, DB state hash, WireMock journal, and the held-back set. Verdict
`decided_by="code"` — already enforced by the contract.

**Test:** identical responses → pass. One changed field → fail, and the diff names the
exact path. Reordered array with policy → pass; without policy → fail. Timestamp with
policy → pass. **A patch that special-cases benchmark inputs → caught by the held-back
set.** Assert the oracle hash changes if any recording changes.

**Learning:** verification vs evaluation. Why a grader the agent can touch is not a
grader. Expect a hard week here on legitimate-but-flagged diffs — ordering, timestamps,
float formatting. Everyone loses it; it's where the most learning is.

---

### ⭐ P14 — A/B measurement + significance · mixed · ~3h

**Goal:** an improvement number you'd defend in a review.

**Spec highlights:** **interleaved** — blue, green, blue, green — never sequential.
Mann-Whitney U on paired per-request latencies. Report SQL-count and rows-examined
deltas as the **mechanism check**: if the plan predicted "847 queries → 2" and it's
still 847 while latency dropped, the diagnosis was wrong even though the numbers look
great.

**Test:** synthetic paired samples → assert the p-value and deltas. Inject a linear
drift into both arms → assert interleaving cancels it and sequential wouldn't.

**Learning:** objective evaluation of agent outcomes. Why "right answer, wrong reason"
must be detectable.

---

### ⭐ P15 — Gate, PR, report · **HAND-WRITE** (gate) · ~4h

**Goal:** ship it, or refuse honestly.

**Spec highlights:** six gates from design doc §P10, fixed thresholds, no model vote.
PR body **rendered from artifacts** — the renderer must be structurally incapable of
emitting a number that has no `ArtifactRef`. Claude writes narrative prose only.
`gh pr create`.

**Test:** the important one — attempt to render a report containing a number with no
backing artifact and **assert it raises.** That test is the evidence ledger.

**Learning:** provenance as an architectural property rather than a habit.

---

### ⭐ P16 — Orchestrator + trace viewer · mixed · ~4h

**Goal:** `perfagent run` end to end, resumable.

**Spec highlights:** extract the FSM from the `run.py` you've been growing — now that
you know the real shape. Persist `RunState` at every phase boundary; resume from any
completed phase. All eight terminal states reachable and reported honestly.
`perfagent trace <run_id>` renders the run as the narrative in design doc §10.3.

**Learning:** workflow vs agent, made concrete. You'll see exactly why this wasn't
P00 — the abstraction you'd have written then would be wrong.

**→ First working version complete.**

---

## Milestone E — reliability, post-MVP

### P17 — Scenario suite + eval harness · mixed · ~8h

**Goal:** find out how often it actually works.

**Spec highlights:** the ten scenarios from design doc §11.2 with a ground-truth
answer key the agent cannot reach. **Scenarios 9 and 10 are negative controls and are
non-negotiable** — a system that can't say "nothing to do" will patch working code.
Scenario 8 (two problems, 80/20) is the calibration test. Every scenario run ≥3× —
agents are stochastic and one run proves nothing. Metrics: diagnosis accuracy, ranking
accuracy, evidence validity, fix validity, minimality, negative-control precision,
stop discipline, cost, **variance**.

**Learning:** building reliable evaluations — the core AI engineering skill, and the
thing that separates this from a demo.

---

### P18 — Tree-sitter code index · generate · ~5h

**Goal:** replace grep-based exploration with structural queries. **Retrofit, then
measure.**

**Spec highlights:** `tree-sitter-java` → SQLite facts table (symbols, call edges,
annotations, fetch types). New tools `find_symbol`, `get_callers`, `get_callees`.
Then re-run the P04 demo and compare tokens, steps, dollars and accuracy against the
numbers you wrote down in P03/P04.

**Learning:** when *not* to use AI; retrieval without embeddings. The measured delta
is where "tool design is context design" becomes a number instead of a slogan.

---

### P19 — Docker environment + `DataPlan` · generate · ~6h

**Goal:** reproducible measurement anywhere.

**Spec highlights:** compose with pinned CPU/memory, pinned
`innodb_buffer_pool_size`, WireMock. Seeded data **baked into a
`mysql-seeded:<dataset-hash>` image** — MySQL has no template databases and dump
restore is too slow. Claude produces the `DataPlan` (semantics); Python generates rows
(exactness).

**Learning:** reproducibility as an engineering discipline. Why the dataset hash lives
in the image tag.

---

### P20 — Repo profile, feasibility gate, StackPack seam · mixed · ~6h

**Goal:** works on a repo you've never seen.

**Spec highlights:** Claude infers the `RepoProfile` from README/compose/CI/pom;
**code verifies every field by executing it** — build, boot, health-check, one test,
DB connect. Cache as `.perfagent/profile.yml`, invalidate on build-file change.
Feasibility gate reports blockers *before* spending money. `StackPack` protocol with
exactly one implementation behind it.

**Learning:** AI for inference, code for verification — the pattern of the whole
product, applied to onboarding. Build the seam, resist the second implementation until
a real second repo demands it.
