# Senapathy eval suite

Three completed scenarios, frozen, replayed two ways. Each scenario is a
directory plus two JSON files you can read without running anything.

```
eval/
  scenarios/
    INV-04/   GET /api/customers/{id}/invoices  -- an N+1, non-SQL cost
    COL-06/   GET /api/reports/collections      -- a non-sargable full scan
    PAY-03/   GET /api/payments/recent          -- deliberately optimal
  tier1.py    reasoning replay   -- fast, offline, no database
  tier2.py    full pipeline      -- slow, needs Docker + a seeded database
  live_freeze.py, redo_plan.py   -- how a scenario gets (re-)frozen
  _checkout.py, _extract.py      -- shared helpers, not runnable directly
```

## What's frozen per scenario

```
eval/scenarios/<ID>/
  endpoint_model.json   comprehension's map of the endpoint
  corpus.json            the request corpus
  baseline.json          the measured single-arm baseline (p50, SQL share, ...)
  hypotheses.json        the diagnosis this scenario's conversation produced
  plan.json              the plan this scenario's conversation produced
  cassettes/
    diagnose/*.json       one LLM request/response pair per cassette, the
    plan/*.json            diagnose and plan conversations, in full
  tool_responses.json    every run_probe/explain/list_indexes/index_benefit/
                          query_equivalent/get_schema call this scenario's
                          conversation made, with its exact recorded result
  meta.json              route, mini-spring's base commit, and its own tag
  expected.json          the assertions tier1.py checks
```

`endpoint_model.json` / `corpus.json` / `baseline.json` are facts about the
measured system — they don't change when a prompt or a tool schema changes
later, so they're copied straight from the completed run's own artifacts.
`hypotheses.json` / `plan.json` and the cassettes are NOT archived from the
original run as-is: this project's tool schemas (the `mechanism` field,
`scope_reasons`, the fixability table) keep changing, so a conversation
recorded even a few hours ago no longer hashes the same against today's
code. Each scenario was instead produced by `live_freeze.py` — one real,
live diagnose+plan run against the frozen baseline/endpoint_model, with its
own fresh cassette directory — and the resulting conclusion was checked by
hand against what's already known to be true of that endpoint before being
written into `expected.json`. That's what "freezing" means here: not an
archive of history, but a cassette set guaranteed self-consistent with the
code that just produced it.

Every scenario's `mini_spring_commit` has a matching tag in the mini-spring
repo (`eval/INV-04`, `eval/COL-06`, `eval/PAY-03`). Both tiers resolve the
tag and compare it to the recorded commit before doing anything else; a
missing or moved tag is a named failure, not a silently-different test.

## Tier 1 — reasoning replay

```
.venv/bin/python eval/tier1.py              # all three
.venv/bin/python eval/tier1.py COL-06       # just one
```

Runs in well under a second, makes no network call, needs no database. For
each scenario it:

1. Resolves and verifies the scenario's git tag against a cached,
   read-only `git archive` extraction (`eval/_checkouts/<commit>/` —
   never the mini-spring repo's own working copy, so Tier 1 never cares
   what branch you happen to have checked out there).
2. Replays the diagnose conversation: the real `run_agent` loop, the real
   `validate_hypothesis_set`, an LLM client pointed at the scenario's own
   `cassettes/diagnose/` with `offline=True`, and `run_probe` served from
   `tool_responses.json` by exact argument match.
3. Replays the plan conversation the same way: the real `rank_hypotheses`,
   the real `make_submit_plan` (file-existence checks run against the
   pinned checkout, for real), `read_file`/`search_code`/`list_dir`/
   `find_route` served for real against that checkout, and `explain`/
   `list_indexes`/`index_benefit`/`query_equivalent`/`get_schema` served
   from `tool_responses.json`.
4. Checks the result against `expected.json`: every hypothesis layer
   required/forbidden, which layer is dominant (by `attributed_ms`), the
   selected candidate's `mechanism`, and — for a `rows_examined` candidate —
   the measured EXPLAIN ANALYZE delta against `min_rows_examined_improvement_pct`
   (replayed from the real number `live_freeze.py` measured with a live
   database, via a frozen `verify_sql_candidate` stub keyed on the
   candidate's own `sql_before`/`sql_after`/`params` — see
   `frozen_sql_verification_table` in `tier1.py`). Or, for PAY-03, that
   nothing was selected and why.

   `approach` (which named technique won — `fetch_join` vs `batch_query` vs
   `dto_projection`, say) is recorded as `approach_context`, never gated: for
   several of these scenarios more than one approach is a genuinely correct
   fix for the same underlying bug, and re-sampling the live conversation at
   re-freeze time can legitimately draw a different one each time (INV-04
   drew three different valid N+1 remedies in three re-freezes). Asserting a
   specific approach meant widening the accepted list on every re-sample,
   which is how an eval ends up asserting nothing. `mechanism` plus its
   measured effect is the actual claim a scenario makes; `approach` is how
   the model happened to get there this time.

A cassette miss or a frozen-tool-call miss **raises**, in both the LLM
client and the tool stub; neither ever falls through to a live call. That
is deliberate: an eval that can quietly cost money on a miss is not an eval.

This is the tier that would have caught, had it existed before today:
- the mechanism classifier picking `sql_count` because a candidate's
  *description* mentioned a query, when its stated mechanism was
  `rows examined` (COL-06) — `expected.json` pins `mechanism_any_of:
  ["rows_examined"]` on the selected candidate
- a diagnosis that ignores the measured SQL share of latency — enforced
  structurally, because `validate_hypothesis_set` itself is what runs
  during replay, not a mock of it
- a hypothesis that falls through the fixability table UNCLASSIFIED on a
  slow query and gets sorted out of consideration despite the highest
  attributed cost — `required_layers`/`dominant_layer_any_of` would catch
  the resulting wrong (or missing) dominant hypothesis

Exit code is non-zero if any scenario fails, so this can gate a commit.

### Quarantine

A scenario that fails for a reason unrelated to the product code it exists
to test (an isolation gap in the freeze/replay harness itself, not a bug in
`senapathy`) can be quarantined rather than left permanently red: a
one-scenario-forever failure stops being a signal — "N-1/N" reads as
"basically green" and a real regression hides in the one row nobody expects
to change. Add `eval/scenarios/<ID>/quarantine.json`:

```json
{"since": "2026-11-08", "reason": "why, and what's been ruled out"}
```

Both fields are required — a quarantine with no date or no reason is the
silent-permanent case this exists to prevent. A quarantined scenario is
still replayed every run, just reported separately (`4/4 passed, 1
quarantined`) and excluded from the pass/fail denominator and the exit
code — *unless* it starts passing, in which case the table flags it loudly
(`Q-PASSING ... <-- now passing; remove quarantine.json`) and the exit code
goes non-zero anyway: a quarantine excuses a scenario from the count, never
from being looked at again.

## Tier 2 — full pipeline

```
.venv/bin/python eval/tier2.py
```

Needs Docker (mini-spring's MySQL container) and takes minutes, not
seconds. Runs the real `senapathy run` pipeline — understand, baseline,
diagnose, plan, patch, measure, gate, ship `--dry-run` — against mini-spring
pinned at `eval/INV-04`'s own tag, re-seeded small (2,000 customers / 20,000
invoices by default, not the frozen scenario's 50k/1.2M) so a release-gate
run stays practical.

INV-04 was chosen because it's the smallest of the three: no
`customer_payment`/`payment_allocation` seed required, and the N+1 it
exercises is cheap to reproduce at a small scale.

This is the tier that catches:
- the blue-arm-reused-instead-of-rebuilt class of bug
- a green worktree left behind, or not cleaned up, after a run
- a migration Flyway refuses to apply even though the patch builds
- an oracle comparison that passes informally but fails byte-exact against
  a frozen recording (timezone, locale, serialization order)
- anything timing-dependent in the real interleaved A/B measurement

**Say this plainly: Tier 1 structurally cannot catch any of the above.**
It never creates a worktree, never builds mini-spring, never starts a
second arm, never touches MySQL. A bug in any of those mechanics can pass
Tier 1 every time and only ever show up here.

Tier 2 is a release gate, not a commit gate — run it before shipping, not
on every change, the same way you wouldn't re-seed 1.5M rows to check a
one-line diff.

*As delivered in this change, `tier2.py` is wired against the real
`run_pipeline` orchestrator and its own preconditions (Docker reachable,
tag resolves) are verified to work — but a full live run has not been
executed as part of building this suite. Run it once before trusting it as
a gate; a full pass has not yet been recorded.*

## Adding a scenario

A directory plus two files — `meta.json` and `expected.json` — no code.
Concretely:

1. Have a completed run with `endpoint_model.json`, `corpus.json`,
   `baseline.json` on disk, and a tag in mini-spring at its `base_commit`:
   `git -C <mini-spring> tag eval/<ID> <base_commit>`.
2. Check out that tag, bring mini-spring up, and run:
   `.venv/bin/python eval/live_freeze.py <ID> --run-dir <run_dir>`
   (needs `SENAPATHY_MYSQL_*` pointed at a privileged user — provisioning
   connects as `root`, not the application's own `mini` user, which lacks
   `performance_schema` access `run_probe` needs).
3. Read the `hypotheses.json`/`plan.json` it just wrote, confirm they say
   what you already know to be true of the endpoint, then write
   `expected.json` by hand (see any existing scenario for the shape).
4. `.venv/bin/python eval/tier1.py <ID>` to confirm it replays clean.

If only the plan phase needs re-freezing (diagnose's cassette is still
good), `redo_plan.py <ID>` does just that half, without needing a live app.

## What neither tier catches

- **Model non-determinism across fresh attempts.** A scenario is one
  pinned, verified-correct conversation. Re-running `live_freeze.py` on the
  identical inputs does not reliably reproduce it — while building PAY-03,
  one attempt proposed and selected a spurious "fix" for a provably optimal
  endpoint, and another correctly rejected everything. Freezing the good
  one and asserting against it is exactly what makes the eval meaningful,
  but it means neither tier tells you how OFTEN the agent gets it right —
  only whether this specific, already-verified reasoning still holds.
- **`run_probe`'s hardcoded endpoint.** It always fires against
  `/api/customers/search`, regardless of which endpoint is actually being
  diagnosed (a known, separately-reported bug). Both tiers faithfully
  replay or reproduce this — neither asserts the probe hit the right URL,
  because neither was built to check that.
- **Deliberate prompt or schema changes.** A cassette miss and an expected
  prompt change look identical to Tier 1: both are "the hash doesn't
  match." After an intentional prompt edit, re-run `live_freeze.py`/
  `redo_plan.py` and re-verify `expected.json` by hand; Tier 1 cannot tell
  the two apart for you.
- **Anything about a fourth kind of endpoint.** Three scenarios cover three
  fixability shapes (full scan, N+1-shaped non-SQL cost, nothing to fix).
  A new bottleneck shape — a missing index with no join, a cache-coherency
  bug, a serialization blowup — isn't exercised until someone freezes it.
- **Real patch correctness and real performance.** Tier 1 never builds or
  measures anything; only Tier 2 does, and only for INV-04. COL-06's and
  PAY-03's candidates are never actually implemented and measured by this
  suite.
- **Cost and latency regressions in the agent itself** (token spend
  creeping up turn over turn, wall-clock budget behavior) — neither tier
  tracks spend across runs or exercises `LoopBudget` exhaustion paths.
- **Report/PR rendering** (`report.md`, the PR body, the narrative writer)
  — not exercised by either tier.

## Timing (measured)

- Tier 1, all three scenarios: ~0.2s total, no network, no database.
- Tier 2: not yet timed end to end (see note above) — expect minutes,
  dominated by the small reseed and the real Maven builds of both arms.
