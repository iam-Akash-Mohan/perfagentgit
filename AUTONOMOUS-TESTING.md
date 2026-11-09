# Autonomous scenario testing — the prompt

Hand this whole file to Claude Code. It runs the remaining scenarios from
`TESTING-PLAN.md` without a human in the loop per step, and it treats token spend as a
budget it must justify.

---

## Your task

Work through the remaining scenarios in `TESTING-PLAN.md`, in the order given below.
For each one: build the planted bottleneck, prove it is winnable before spending
anything, run Senapathy, diagnose whatever goes wrong, fix it, and retry — until the
scenario reaches its predicted outcome or you hit its stop condition.

Record every result in the scenario log in `TESTING-PLAN.md` as you go.

---

## Token discipline — read this first

Live runs cost money. A clean run is roughly $0.03–0.05; a scenario that fights you can
cost ten times that. **Every live run must be justified by something you learned since
the last one.**

Hard rules:

- **Never re-run hoping.** After a failure, read `trace.jsonl`, the artifacts and the
  known-failure table in `TESTING-PLAN.md` *first*. Form a specific hypothesis. Fix it.
  Only then run again. A run that repeats a previous one unchanged is wasted money.
- **Exhaust the free checks before any run.** Arms clean, correct branch checked out and
  committed, endpoint present in the built arm, hand-measurement proving a ≥15% win
  exists. All free. All have cost a live run at least once already.
- **Re-gate instead of re-running.** When only gate or report logic changed, the existing
  patch, verification and measurement are still valid:
  `senapathy gate --run-dir .senapathy/runs/<id>` costs nothing. Three PRs so far were
  shipped this way.
- **Run the eval suite after every product change, before any live run.** It is free and
  it is the regression net.
- **Pass `--max-usd 0.50` on every run.** A scenario that burns through that has a
  problem worth stopping for.
- **Budget: 6 live runs per scenario.** On the seventh, stop and write up what you know.
- **Code changed → new run id. Interrupted or crashed → `--resume`.** A resume cannot
  pick up a code change, and a fresh id cannot reuse good work. Getting this backwards
  has cost four runs.

---

## Before you start anything — Phase 0

These have cost nine runs between them across three scenarios. Land them first; they
need no live runs.

1. **Arm setup must be self-healing and tied to the run.** Prune stale registrations and
   remove a leftover directory before `git worktree add`, never after. Resolve blue from
   the repo's current HEAD every run and recreate it when the existing worktree is on a
   different commit. `git worktree add -b` is not atomic — clean up the branch when it
   fails. Record both arms' commits in `provisioning.json` and print them beside the
   ports.
2. **Refuse at provision when the route is absent from the built arm**, naming the route
   and the commit. Three runs died on a corpus of 404s discovered after comprehension had
   been paid for.
3. **`completed_phases` must not skip a phase whose artifact is missing.**
   `orchestrator.py` uses `(already or valid)`, so a deleted artifact is skipped forever.
   Make the artifact the authority.
4. **Freeze STM-01 as the fourth eval scenario** and run Tier 1 across all four. It is the
   only coverage for `remove_redundant_call`.

Report Phase 0 before moving on.

---

## The protocol, per scenario

**1. Build the planted bottleneck.** The scenario entries below give the shape. Obey the
standing rules: GET only, bounded response, explicit total `ORDER BY`, one dominant
instrumented mechanism, every parameter resolvable without `--param`, `mvn test` actually
executing, no comment or name hinting at the problem, no sleeps or artificial work, and
**never write the fix**.

**2. Prove it is winnable — before spending anything.** Hand-measure p50/p95 and the
mechanism quantity as built and with the fix applied locally, interleaved. Then revert the
fix. Require:

- improvement comfortably above 15%, not marginal
- `sql_share_of_latency_pct` above ~60% for a database scenario
- the planted mechanism moving by an order of magnitude

If it is not winnable, fix the **scenario**, never the gate or a threshold. If you cannot
make it winnable, say so and stop — that is a finding, not a failure.

For the two probe scenarios and the two refusal scenarios this step inverts: prove there
is **no** shippable win, by the same measurements.

**3. Run it.**

```bash
senapathy run --repo <repo> --endpoint "<route>" --run-id <ID> --open-pr \
              --max-usd 0.50 --max-wall 7200
```

**4. Triage whatever happens.** Classify the outcome before touching anything:

| Outcome | What it means | What to do |
|---|---|---|
| Predicted result | Scenario passes | Record it, freeze it, next scenario |
| Correct refusal for the stated reason | Also a pass, for the refusal scenarios | Record it |
| Product bug | The agent was right and the harness was wrong | Fix the product, run the eval suite, new run id |
| Scenario bug | The planted bottleneck is wrong or unwinnable | Fix the endpoint, new run id |
| Environment | Arms, branch, credentials, ports | Fix it; it costs no tokens to get right |

The agent reasoning correctly on bad information is a **product** bug, not a model
failure. That has been the cause of most failures so far: a truncation constant hiding
test failures, a classifier reading the wrong field, a plan scoping out a call site.

**5. Record it** in the `TESTING-PLAN.md` scenario log: run id, terminal state, fix kind,
mechanism moved, whether it matched the prediction, and whether it is frozen. Add any new
failure mode to the known-failure table.

**6. Freeze it** as an eval scenario before moving on.

---

## When to stop and ask

Stop and write up the question rather than guessing when:

- a genuine product **design** decision appears — how a guardrail should behave, what a
  gate should accept, whether a mechanism should be instrumented. These shape the product
  and are not yours to settle alone.
- a fix would require weakening a gate, a threshold, or the tolerance policy. **Never do
  this.** If a scenario cannot clear the bar, the scenario is wrong.
- you have spent 6 live runs on one scenario.
- something you find contradicts `SENAPATHY.md`'s stated invariants.

---

## The scenarios

Build and run these in order. Full descriptions are in `TESTING-PLAN.md`.

### Scenario 3 — `batch_query` (expect a PR)

A page of rows each needing a lookup from a second table **by a non-association key**, so
`fetch_join` does not apply. The fix is one `IN (...)` query plus an in-memory join.
Mechanism `sql_count`.

The point is that it is NOT the invoices N+1 wearing a different hat. Check specifically
whether the plan distinguishes `batch_query` from `fetch_join` — they are different policy
rows and only one has ever been exercised.

### Scenario 4 — `move_work_out_of_loop` (expect a PR)

A query inside a loop whose parameters do not vary with the iteration — a configuration or
reference lookup repeated per row. Distinct from an N+1 because the query is not *about*
the row. Mechanism `sql_count`.

### Scenario 5 — `dto_projection` probe (expect a gate failure)

An endpoint returning entities with wide unused columns, where only a few fields reach the
response. **Expect this to fail `mechanism_match`**: `SELECT *` and `SELECT a,b,c` examine
the same rows, so neither instrumented quantity moves even though the patch is genuinely
faster.

If that happens, the finding is that over-fetching needs a third instrumented quantity —
bytes sent, or rows sent × row width. Do not instrument it in this pass. Measure what
would be needed, write it up, and move on.

### Scenario 6 — `read_only_transaction` probe (expect a gate failure)

A read path inside a read-write transaction, where `@Transactional(readOnly = true)` would
skip dirty-checking and flush. The saving is JPA bookkeeping and moves neither instrumented
quantity. Almost certainly unshippable today. Record what the gate says.

### Scenario 7 — serialization-dominant (expect a refusal)

Large response, trivial query, cost in JSON serialization. Expected: diagnosed correctly,
ranked by expected value, refused at fixability 0.30 with an uninstrumented mechanism.
**It must not reach the patch phase.**

### Scenario 8 — unbounded response collection (expect a blocked candidate)

An endpoint whose expensive per-row work *produces* the response contract — no pagination,
the collection is the API. Expected: blocked at fixability 0.05 with the field named.
Proves it will not "optimise" by changing what the API returns.

---

## Final report

When you stop — finished or out of budget — report:

- the scenario log, filled in
- total spend, per scenario
- every product bug found, with its fix
- every failure mode added to the known-failure table
- the design questions you stopped on
- what the two probe scenarios proved about instrumentation coverage
- which scenarios are frozen as eval cases, and the current Tier 1 result
