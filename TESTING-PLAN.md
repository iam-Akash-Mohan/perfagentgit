# Senapathy — optimisation coverage plan

One scenario per fix kind. Each is a planted bottleneck in a target repo plus the
verdict the agent is expected to reach. **A refusal is a passing result** when it is
the correct refusal — half of these scenarios exist to prove the agent says no when
it should.

Status as of INV-04: `fetch_join` / N+1 family **proven** (−74.5%, PR #1).
Everything below is unproven.

Indexing (`add_index`) is deliberately out of scope here and planned separately — it
needs per-arm schema isolation built first.

---

## The constraint that gates everything

**Only two mechanisms are instrumented:** `sql_count` and `rows_examined`. A fix whose
effect shows up anywhere else fails `mechanism_match` closed, however fast it actually
is. Several scenarios below exist specifically to find out which fix kinds are
unshippable for this reason.

---

## Scenario table

| # | Fix kind | Tier | Planted cause | Expected mechanism | Expected verdict |
|---|---|---|---|---|---|
| 1 | `query_rewrite` | T2 | Non-sargable date predicate over a large table | `rows_examined` | PR |
| 2 | `remove_redundant_call` | T2 | Three layers each re-fetching the same row | `sql_count` | PR |
| 3 | `batch_query` | T2 | Per-row lookup fixable by one `IN` query | `sql_count` | PR |
| 4 | `move_work_out_of_loop` | T2 | Loop-invariant query inside an iteration | `sql_count` | PR |
| 5 | `dto_projection` | T1 | `SELECT *` over wide rows, few fields used | *unclear* | **Probe** — may expose a gap |
| 6 | `read_only_transaction` | T1 | Read path in a read-write transaction | *unclear* | **Probe** — may expose a gap |
| 7 | — | — | Serialization-dominant endpoint | n/a | `BOTTLENECK_NOT_FIXABLE` |
| 8 | — | — | Unbounded response collection | n/a | Candidate blocked, named |
| 9 | — | — | Already-optimal endpoint | n/a | `NO_BOTTLENECK_FOUND` |

---

## Phase 1 — the measurable fixes

These four should produce PRs. They exercise both instrumented mechanisms across four
distinct code paths, and none needs infrastructure work first.

### 1. `query_rewrite` — in progress

Collections report joining `payment → payment_allocation → invoice → customer`, with a
non-sargable date predicate (`YEAR(paid_on) = ?`) defeating an index that exists and is
correct. Fix is a half-open range. 1–2M payment rows.

Proves: the second instrumented mechanism, EXPLAIN-driven diagnosis, and a fix that
rewrites SQL rather than restructuring Java.

### 2. `remove_redundant_call`

The scenario from your first conversation: a flow through three intermediate methods
where each independently calls the same `findById`. Each layer is defensible alone; the
cost only exists end to end.

Response must be bounded. Fix is to fetch once and pass it down. Mechanism: a repeated
identical digest collapses from N calls to 1.

Proves: the agent can see redundancy *across* call layers, not just inside a loop.

### 3. `batch_query`

A page of rows where each needs a lookup from a second table by a different key — not a
lazy association, so `fetch_join` doesn't apply. The fix is one `IN (...)` query plus an
in-memory join.

Proves: `batch_query` is distinguished from `fetch_join`. These are different policy
rows and different patches; only one has ever run.

### 4. `move_work_out_of_loop`

A query inside a loop whose parameters don't vary with the iteration — a configuration
or reference lookup repeated per row.

Proves: loop-invariant detection, which is distinct from N+1 (the query isn't *about*
the row).

---

## Phase 2 — probes that may expose instrumentation gaps

Expect these to fail. The failure is the finding.

### 5. `dto_projection`

An endpoint returning entities with wide unused columns (TEXT/BLOB), where only three
fields reach the response.

**The problem:** `SELECT *` and `SELECT a,b,c` examine the *same rows*. If `rows_examined`
doesn't move and `sql_count` doesn't move, `mechanism_match` fails and a genuinely faster
patch is rejected. If that happens, the finding is that over-fetching needs a third
instrumented quantity — bytes sent, or `rows_sent` × row width.

### 6. `read_only_transaction`

A read path running inside a read-write transaction, where `@Transactional(readOnly =
true)` would skip dirty-checking and flush.

**The problem:** the saving is JPA bookkeeping. It changes neither query count nor rows
examined. Almost certainly unshippable today.

Run both anyway, and record what the gate says. "We know this fix kind is unshippable and
why" is worth more than leaving it untested.

---

## Phase 3 — refusals (equally important)

### 7. Serialization-dominant endpoint

Large response, trivial query, cost in JSON serialization. Expected: diagnosed correctly,
ranked by expected value, then refused — `fixability 0.30`, mechanism uninstrumented.

Proves the agent doesn't ship what it can't prove.

### 8. Unbounded response collection

An endpoint whose expensive per-row work *produces the response contract* — no pagination,
the collection is the API. Expected: candidate blocked at `fixability 0.05` with the field
named.

Proves it won't "optimise" by changing what the API returns.

### 9. Already-optimal endpoint

A simple indexed lookup with nothing to fix. Expected: `NO_BOTTLENECK_FOUND`.

Proves it doesn't invent work. **This is the most important refusal test** — an agent that
always finds something is an agent that can't be trusted when it finds something.

---

## Phase 4 — genericity

- **A POST endpoint.** Requires `RequestShape` to record request bodies; today the corpus
  generator refuses. One fix, then all POSTs work.
- **A second repository, Gradle-built.** Onboarding detects Gradle but has never run it.
  Liquibase is detected and unwired. This is the real "any codebase" test.

---

## Make each test permanent

Every scenario above becomes a **cassette-backed eval case**, not a throwaway run. Record
each run's LLM exchanges, freeze the artifacts, and the whole suite replays offline, for
free, in minutes.

The reason matters. Last session found eight bugs at roughly one per hour by running live
against one endpoint. Nine frozen scenarios replayed offline find that class of regression
in a single batch overnight. Build the scenarios as evals from the start and the testing
effort compounds instead of evaporating.

Per scenario, freeze: the planted repo state (a tagged commit), the generated corpus, the
sealed oracle, the full trace, and the expected terminal state. A scenario passes when the
terminal state and selected fix kind match; it fails loudly when either drifts.

---

## Suggested order

1. Scenario 1 — `query_rewrite` *(in progress)*
2. Scenario 9 — already-optimal. Cheap, and the most valuable refusal.
3. Scenarios 2, 3, 4 — the remaining `sql_count` fixes
4. Scenarios 5, 6 — the instrumentation probes
5. Scenarios 7, 8 — the remaining refusals
6. Freeze all nine as the eval suite
7. Phase 4 — POST support, then a second repo

---

## Deferred: indexing

`add_index` carries the highest fixability in the policy table (0.95, "additive index,
zero behaviour risk") and in a production codebase missing indexes are the most common
finding — but blue and green currently share one MySQL schema, so an index added for green
exists for blue too and the A/B measurement is meaningless. The agent detects this at plan
time and rejects the candidate as `MEASUREMENT_NOT_ISOLATED`.

Unblocking it means per-arm schema isolation: each arm with its own schema, seeded
identically, with its own JDBC URL. Planned separately.

---

# Operating protocol

Seven beats per scenario. Same every time, so it is always clear where we are.

| Beat | Who | What |
|---|---|---|
| 1 | Claude | Give the build prompt for the planted bottleneck |
| 2 | Ak | Paste the hand-measured before/after numbers and EXPLAIN plans |
| 3 | Claude | Confirm the scenario is winnable — or fix it before spending anything |
| 4 | Ak | Run it, fresh run id |
| 5 | Ak → Claude | **Baseline checkpoint** — paste the three numbers, wait for go/stop |
| 6 | Claude | Read the artifacts, check the verdict against the prediction |
| 7 | Both | Freeze as an eval case, then next scenario |

## Beat 3 — is it winnable?

Check before spending anything:

1. **Improvement well above 15%.** Not 18% — something unambiguous.
2. **`sql_share_of_latency_pct` above ~60%.** On the invoices endpoint it was 8.75%, which
   is why the diagnosis under-attributed by 2.4x and two runs selected a candidate worth
   0.0 ms.
3. **The planted mechanism moves by an order of magnitude** between the broken and fixed
   versions, visible in both EXPLAIN plans.

## Beat 5 — the baseline checkpoint

After `baseline done`, stop and read `.senapathy/runs/<id>/baseline.json`:

```
benchmark.status_counts          → must be overwhelmingly 200
derived.statements_per_request   → must match the planted shape
derived.sql_share_of_latency_pct → above ~60% for a DB-bottleneck scenario
```

Every expensive mistake so far came from letting a run continue past evidence that it was
measuring the wrong thing. A run that has not returned a single 200 is measuring how fast
the app rejects a malformed request.

## Beat 6 — what counts as a pass

A scenario passes when **all three** match the prediction:

- the terminal state (`PR_OPENED`, `NO_BOTTLENECK_FOUND`, `BOTTLENECK_NOT_FIXABLE`…)
- the selected fix kind
- the mechanism that moved

For a refusal scenario, it must refuse **for the stated reason**. Refusing for an unrelated
reason is a fail, not a pass.

---

# Hard rules

**Code changed → new run id. Interrupted or crashed → resume.**
Resume cannot pick up a code change. It re-runs only phases whose artifacts are missing or
stale, and a code fix does not make a data artifact stale. Three cycles were lost to this
in one session.

**Deleting artifacts does not force a re-run.**
`orchestrator.py` skips a phase listed in `run_state.json`'s `completed_phases` even when
its artifact is gone (`already or valid`). Until that is fixed, a fresh run id is the only
reliable way to re-run a phase.

**Never weaken a gate, a threshold, or the tolerance policy to make something pass.**
If a scenario cannot clear 15%, the scenario is wrong, not the gate.

**Stop at the first evidence of nonsense.** Do not let a run continue to burn tokens past
a baseline that shows the endpoint never executed.

---

# Pre-run checklist

```bash
# 1. MySQL up (provision probes the datasource)
docker compose -f /Users/akashm/Documents/mini-spring/docker-compose.yml ps

# 2. Admin credentials EXPORTED, not in .env — that code path reads
#    os.environ directly and nothing calls load_dotenv() before it
export SENAPATHY_MYSQL_ADMIN_USER=root
export SENAPATHY_MYSQL_ADMIN_PASSWORD=root

# 3. Shell must not override the provider key from .env
#    (load_dotenv runs with override=False, so an exported var wins)
echo "shell : ${GROQ_API_KEY:0:7}...${GROQ_API_KEY: -4}"

# 4. Run from the AGENT repo — --runs-dir is relative to cwd
cd /Users/akashm/Documents/ak/perfagent
source .venv/bin/activate
```

---

# Known failure modes

Catalogue from the first end-to-end run, so a repeat is recognised in seconds rather than
diagnosed from scratch.

| Symptom | Cause |
|---|---|
| `HTTP 401 invalid_api_key` | Shell `GROQ_API_KEY` overriding `.env` — `load_dotenv` does not override |
| `HTTP 413 ... Limit 8000` | Groq free tier TPM. One request exceeds the per-minute budget; waiting does not help |
| `HTTP 400 output_parse_failed` | Model emitted prose where a tool call was required; Groq rejects server-side, so the loop never gets to correct it |
| `budget_exhausted: step ceiling` | Phase ceiling too low for a deep call chain. Raise with the phase's `--*-max-steps` flag |
| `No database selected` | Unqualified table name on the deliberately schema-less `MySQLTarget` connection |
| `performance_schema needs an admin login` | `SENAPATHY_MYSQL_ADMIN_*` not exported |
| `required query parameter has no determinable value` | No enum, no default, no resolvable column. Use `--param NAME=VALUE`, or give the parameter a default in code |
| `baseline p95 3 ms`, zero SQL | Every request returned 4xx. The endpoint never ran |
| `attributed 0.0 ms`, patch ABANDONED | Nothing worth fixing — corpus landed on sparse rows, or the bottleneck is genuinely small |
| Oracle hash mismatch | A stale global oracle from a previous endpoint, or the hash function changed |
| Phases skipped after a code fix | See the hard rules — use a new run id |

---

# Scenario log

Fill in as each completes. A scenario is not done until it is frozen as an eval case.

| # | Scenario | Run id | Terminal state | Fix kind | Mechanism moved | Matches prediction? | Frozen |
|---|---|---|---|---|---|---|---|
| 0 | N+1 (invoices) | INV-04 | `PR_OPENED` | `query_rewrite` | `sql_count` 102→2 | yes | no |
| 1 | `query_rewrite` (collections) | | | | | | |
| 2 | `remove_redundant_call` | | | | | | |
| 3 | `batch_query` | | | | | | |
| 4 | `move_work_out_of_loop` | | | | | | |
| 5 | `dto_projection` (probe) | | | | | | |
| 6 | `read_only_transaction` (probe) | | | | | | |
| 7 | Serialization-dominant | | | | | | |
| 8 | Unbounded response | | | | | | |
| 9 | Already-optimal | | | | | | |

## Per scenario, keep

- the tagged commit of the planted repo state
- the generated corpus and its hash
- the sealed oracle and its hash
- the full `trace.jsonl`
- the expected terminal state and fix kind
- the PR link, where one was opened

A frozen scenario passes when terminal state and fix kind match, and fails loudly when
either drifts.
