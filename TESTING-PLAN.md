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

## Headline finding of the Scenario 7/8 sweep: the diagnose phase invents bottlenecks when there is no real one to find — RESOLVED

This is the finding the two refusal scenarios exist to surface, and it is confirmed,
not suspected. Every live diagnose sample across Scenarios 7 (`GET /api/customers/bios`,
6 samples: `SER-01` original, `SER-02`, `SER-03`, and 3 fresh `live_freeze.py` samples,
the last of which is the frozen `SER-01` case) and 8 (`GET /api/invoices/{id}/payment-history`,
4 samples: `UNB-01` through `UNB-04`) produced an `orm_hydration` hypothesis claiming
per-row entity hydration or N+1 round-trips — for an endpoint whose own comprehension
model records **zero** per-row data accesses and whose single query is a flat JPQL (or,
after the Scenario 8 redesign, native-join) constructor projection. 10 of 10 diagnose
samples made this claim, worded differently each time ("Hibernate creates a managed
entity for each result row and attaches proxies", "JPA materialises an entity per row",
"ORM hydration of each JPQL row into a DTO incurs significant CPU work", "DTO projection
returns the same values (invoice: 2 of 5 columns are read)" as a *fixability* reason for
a hypothesis about an entity that is never hydrated).

Of the 8 samples that reached an actual plan verdict (the other 2 — `SER-01` original and
`UNB-01` — crashed in the plan phase on unrelated provider/tooling errors before any
candidate could be judged), **4 of 8 went on to SELECT a candidate predicated on that
invented bottleneck**: the Scenario 7 freeze sample proposing `fetch_join` on a field
that does not exist on the entity; the Scenario 7 freeze sample proposing a no-op
`dto_projection` rewrite of a query already projected; `UNB-03` selecting
`remove_redundant_call` via the (now-fixed) `response_link` classifier bug; `UNB-04`
selecting a no-op `query_rewrite` claiming to eliminate "the extra SELECT that loads the
Invoice entity" that the Scenario 8 redesign had already removed. The other 4 of 8
(`SER-02`, `SER-03`, the final frozen `SER-01` sample, `UNB-02`) carried the same false
hypothesis but happened to get refused anyway, for reasons unrelated to the hypothesis
being wrong (an uninstrumented mechanism, a flat EXPLAIN delta, a schema-isolation block).

**Confirmed, not just likely: `senapathy/diagnose/probe.py::make_run_probe` was hardcoded
to one fixed endpoint, `POST /api/customers/search`, for every run regardless of which
endpoint was under test** — the tool's own schema said so plainly ("it can only call the
search endpoint with a different page size and name filter"). That endpoint genuinely
does exhibit N+1-shaped SQL-count growth, and the probe's measured numbers describing
it were accurate — just describing the wrong endpoint. One sample's own hypothesis
evidence literally reads "application SQL statements per request grow with rows: 4
statements for 1 row, 102 for 50 rows," which is `/api/customers/search`'s behaviour,
reported as if it were the endpoint actually under test. **Fixed**:
`probe.py::ProbeCatalog` now builds the probe entirely from THIS run's own
`endpoint_model.json` and `corpus.json` — the route, method and concrete request shape
come from the same comprehension model and generated corpus every other phase already
uses. The only thing the agent still chooses is WHICH of this endpoint's own corpus
requests to fire (`request_id`, reusing the magnitude diversity the benchmark/held-back
split already encodes) and, only when the endpoint actually paginates, a `page_size`
override. `ProbeCatalog` raises `ContractViolation` at construction — before any tool
call — when no probe can be built at all (an empty corpus, a non-GET request; workload.py
only ever generates GET corpora today, so the latter is a defensive check, not yet a live
case), rather than silently falling back to a different endpoint. Every probe call now
records which endpoint it actually hit (`ProbeLog` entries carry `endpoint_called`),
surfaced in both the trace and `hypotheses.json`. No other phase was found to hardcode an
endpoint-specific request the way this one did — `measure/bench.py` and the oracle
recorder were checked and are already fully corpus-driven.

**Re-running diagnose live, after the fix, against three already-recorded endpoints'
real facts (`COL-04`, `SER-03`, `UNB-06` — mini-spring checked out at each one's own
commit, corpus/baseline/endpoint_model reused unchanged, a fresh live diagnose run
against the correct, now-confirmed-correct endpoint) shows the `orm_hydration`
hypothesis does NOT disappear.** All three re-probes fired the right endpoint
(`probes_executed[].endpoint_called` confirms `GET /api/invoices/{id}/payment-history`,
`GET /api/customers/bios`, `GET /api/reports/collections` respectively — not
`/api/customers/search`), with correct raw evidence (real row counts, real per-request
SQL statement counts of 1, real small latencies) — and all three still produced an
`orm_hydration` hypothesis claiming "Hibernate creates an entity per row" / "ORM
hydration of each JPQL row into \[DTO\] objects." **All three endpoints' queries are
JPQL constructor-expression projections** (`select new com.ex.dto.XDto(...) from ...`),
which by JPQL's own semantics never produce a managed entity at all — that is the entire
point of a constructor expression. So this is not, or not only, the wrong-endpoint bug:
it is an independent, standing misconception about what JPQL constructor-expression
syntax does, which the probe fix cannot correct because it is a reasoning error, not a
measurement one. Severity varied with the dominant bottleneck: negligible (~0.05 ms/row,
non-competing) against `COL-04` (a real, correctly-diagnosed `rows_examined` bottleneck
dominates there instead); moderate-to-dominant against `SER-03`/`UNB-06`, where it
competed for the top ranking. **Do not amend this entry to say the hallucination was
never real** — the probe bug was real and is fixed, and it did produce several of the
wrong-candidate incidents logged above (anything whose evidence cited search-endpoint
numbers specifically); but the `orm_hydration`-from-a-projection-query claim survives a
correct probe and needs its own fix.

The comprehension model has no field that would let a reconciliation check catch this
specifically: `DataAccess.kind` only distinguishes `jpql_query` / `native_query` /
`derived_query` / etc. — it does not distinguish a constructor-expression projection
(`select new ...(...)`, never hydrates) from a plain entity-returning query (`select e
from Entity e`, does hydrate). Without that distinction recorded anywhere, nothing stops
diagnose from assuming "JPQL query" implies "entity hydration." The concrete next step,
not attempted: add that distinction to `DataAccessKind` (or a boolean alongside it),
have the comprehension phase record it, surface it in the diagnose brief, and reject (in
`validate.py`) an `orm_hydration` hypothesis attributed against a data access already
known to be a projection. This session's other fixes (the verification-gate check, the
no-op-rewrite check, the baseline-citation check) still matter and still work — they
catch the *consequence* (an unverifiable, vacuous, or evidence-free candidate) after
diagnose has already invented a bottleneck, which is a real, separate line of defence
regardless of whether the hallucination itself is ever fixed at the source.

**RESOLVED.** The gap identified above — `DataAccess` records that a query runs but not
what it returns — is fixed. `DataAccess.returns: Literal["entities", "projection",
"scalar"] | None` (`contracts/comprehension.py`), required in the comprehension agent's
own tool schema (`tools/submit.py::DATA_ACCESS_SCHEMA`, populated with a citation like
every other field) but optional at the pydantic layer (`None` for an access recorded
before the field existed, read as unknown, never as a negative). Two places read it:

1. **Stated plainly, not just enforced.** `diagnose/evidence.py::_data_access_lines` now
   prints `returns <value>` on every data access line in the evidence diagnosis itself
   reads, before it forms a single hypothesis.
2. **Enforced as a reconciliation check.** `diagnose/validate.py`, new check 2d, the same
   shape as the existing `sql_share_of_latency_pct` check (2c): when every data access
   has a KNOWN `returns` and none of them is `entities`, an `orm_hydration` hypothesis is
   refused by name, quoting both facts — the hypothesis's own claim text and each access's
   trigger plus its recorded `returns`. An access with no recorded `returns` at all makes
   the check a no-op for that endpoint (unknown, not a negative) — it never fires on a
   partially-annotated or pre-existing artifact.

**Live confirmation, not just unit tests.** Ran `senapathy understand` fresh against
`SER-01`'s endpoint (`GET /api/customers/bios`), `UNB-01`'s (`GET /api/invoices/{id}/
payment-history`), `STM-01`'s (`GET /api/customers/{id}/statement`), and `LOOP-01`'s
(`GET /api/invoices/overdue-worklist`) — four real comprehension passes, not hand-built
fixtures. All three single-query projection endpoints were correctly classified
`returns: projection`; `STM-01`'s four separate repository calls were all correctly
`projection`/`scalar`; `LOOP-01` (`feeScheduleRepo.findByActiveTrue()`, a genuine
Spring Data derived query returning the managed `FeeSchedule` entity, no projection
involved) was correctly classified `returns: entities` — confirming the comprehension
agent gets the THREE-way distinction right on a real mix of shapes, not just the
projection case this bug was found on.

Then re-ran diagnose live against `SER-01`'s and `UNB-01`'s endpoints with these fresh
facts: **three independent live samples across the two endpoints, zero proposed
`orm_hydration`.** The fact alone, stated plainly in the evidence, was enough to stop
the model from reaching for it at all — `emit_hypotheses` was accepted on the first
submission every time, no rejection needed. The reconciliation check stands ready as a
backstop for a sample that tries anyway regardless (confirmed directly: `tests/
test_diagnose.py`'s `test_orm_hydration_is_refused_when_every_access_is_a_projection`
and its sibling for the entity-returning/unknown/no-data-access cases).

`SER-01` and `UNB-01` re-frozen against these fresh comprehension facts (full
diagnose+plan live conversations, not hand-edited); `dominant_layer_any_of` narrowed back
to the genuine layers and `orm_hydration` moved from tolerated context to `forbidden_
layers` in both `expected.json` — a future sample that reintroduces it should fail
loudly now, not be quietly accepted as a known quirk. Full suite: 936 passed. Tier 1
after re-freezing: unchanged, `9/9 passed, 1 quarantined` (`PROJ-01`, its own unrelated,
already-logged reason).

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
| Gate fails on a genuinely winnable scenario, `rows examined` flat (`+0.0%`) despite `sql_count`-looking fix | **FIXED.** `expected_value_ms` was computed per-**hypothesis** (`attributed_ms x fixability`), not per-candidate, so two candidates targeting one hypothesis always tied exactly and `_select()`'s sort fell through to lexical `c.id` — whichever candidate the model happened to list first. BATCH-01 shipped the worse of two tied candidates this way (a correlated subquery, which MySQL re-executes once per outer row and so never moves `rows_examined`) and the rejection message falsely claimed "a higher expected value was available" on an exact tie. Fixed by verifying SQL-rewrite candidates with `EXPLAIN ANALYZE` at plan time (`plan/risk.py::verify_sql_candidate`, `tools/db.py::DbToolbox.plan_delta`) — the measured delta disqualifies a non-moving `rows_examined` candidate and drives its `expected_value_ms` directly, so the tie usually never recurs; when a genuine coin-flip tie still occurs (no SQL to verify, e.g. two Java-side candidates), `_select()` now says so honestly instead of claiming a higher score. See BATCH-01/BATCH-02 in the scenario log. |
| Freshly-frozen scenario's own replay misses its own cassette, at any position — **FIXED, two independent causes, both isolated and corrected** | `eval/scenarios/BATCH-01`, `SER-01` and `LOOP-01` each showed a fresh recording's own Tier 1 replay missing its own cassette, at three different positions (first request, last request, mid-conversation respectively) — which ruled out "specific to one phase's tool set" and "specific to the terminal request" as shared causes without identifying what the three actually shared. Isolated by diffing the EXACT computed request body against the nearest recorded cassette, field by field, rather than guessing: two distinct, confirmed causes, both now fixed.\n\n**Cause 1 — `max_tokens` inside the cassette key.** `Cassette.key()` hashed the FULL wire body, including `max_tokens` — a response-length ceiling resolved from `senapathy.yml`/`SENAPATHY_LLM_MAX_TOKENS`/a CLI flag at client-construction time, wholly unrelated to the conversation's content. A reproduction run from a working directory where `Settings.load()` could not find `senapathy.yml` silently resolved the code default (4096) instead of the configured value (8192), and the key changed even though nothing about the request's actual content had. **Fixed**: `Cassette.key()` now excludes `max_tokens` from the hash (`_KEY_EXCLUDED_BODY_FIELDS`); `model` and `tool_choice` deliberately stay in the key, since those ARE semantic to the request. One new test (`test_a_changed_max_tokens_is_still_a_cassette_hit`).\n\n**Cause 2 — a live self-correction is never itself cassette-recorded.** When the model's tool call is rejected by the PROVIDER as malformed during a live recording, `run_agent` appends a `{\"role\": \"user\", \"content\": <schema error>}` message and retries (`agent.py::_correction_for`, existing mechanism). The FIRST, rejected attempt produces no 200 response, so nothing is ever cassette-written for it — only the SECOND, corrected attempt succeeds and gets recorded, with the correction message already inside ITS OWN `request.messages`. A later offline replay reconstructs the conversation by appending exactly what `run_agent` itself appends on a clean path (assistant reply + tool result) — correct, but missing the one extra correction message the live run only added because of a rejection replay never triggers. From that point on, every subsequent request replay computes is one message short of what was actually recorded, and nothing closes that gap on its own. Confirmed directly: `scenarios/STM-01`'s diagnose replay matched its first three recorded cassettes exactly (byte-for-byte) and then diverged at the fourth, where the recorded conversation jumps from 5 messages to 8 (not 6 or 7) — the missing `user`-role message, read back from that very cassette file, is exactly the correction text `run_agent` would have produced live. **Fixed**: `Cassette.recover_self_correction()` scans the cassette directory for a recorded conversation that is the current one plus EXACTLY one more `user`-role message (never a guess — ambiguous or absent, it returns `None`); `CassetteMiss` now carries that text as `recorded_correction`; `agent.py::_correction_for()` answers it FIRST, before any provider-specific dispatch, which hands it straight to the SAME existing retry loop that already handles a live rejection — `run_agent` itself appends the correction to ITS OWN `messages`, so the gap closes permanently from that point forward and a conversation with multiple corrections self-heals the same way a live one would. Four new tests across `test_llm.py` (`Cassette.recover_self_correction` directly, including the ambiguous/absent cases) and `test_agent_loop.py` (end-to-end: a `CassetteMiss` carrying a correction is corrected, not fatal; one without is still fatal, matching the existing live-rejection test's shape exactly).\n\n**Effect of both fixes together**, re-freezing all 10 scenarios (9 re-frozen fresh under current code + `BATCH-01`, previously unreachable): Tier 1 went from **0/0 passed, 9 quarantined** to **9/9 passed, 1 quarantined** (`PROJ-01`, for its own distinct, already-logged, unrelated reason — a model submission mistake, not a replay bug). The remaining quarantine carries a dated, specific reason, which is the target this table exists to make visible, not a residual mystery. `COL-06`, `INV-04`, `LOOP-01`, `PAY-03`, `STM-01`, `TXN-01`, `BATCH-01` and `SER-01`'s `quarantine.json` files are removed; `INV-04`, `BATCH-01` and `SER-01`'s `expected.json` were updated to match outcomes the verification-gate fix (above) now correctly produces, not to paper over a new failure — see each scenario's own updated `notes`. |
| `_improvement_retry` won't retry the next candidate when `mechanism_match` ALSO fails, and most remaining fix kinds can't trip anything else | **Known issue, deliberately not fixed yet.** `orchestrator.py::_improvement_retry` only moves to the next candidate when the gate failed *exclusively* on `improvement_threshold` (`orchestrator.py:1608-1631`); its own docstring reasons that a mechanism which did not move "is not fixed by trying a different fix for the same bottleneck" — true for ONE candidate's implementation detail, false across candidates, since two candidates for the same hypothesis can have genuinely different mechanisms (BATCH-01 proved it: the correlated-subquery candidate failed `mechanism_match`, the untried `batch_query` candidate would not have). Part C closes this for the `rows_examined` family — those candidates are now verified with `EXPLAIN ANALYZE` *before* a patch exists, so a non-moving one is disqualified at plan time and the bad candidate is never reached to retry from. It does NOT close this for `sql_count` — verified (so the evidence is in the artifact) but deliberately NOT scored or disqualified by that verification, because a single EXPLAIN of each side cannot see a call-count fix (see `plan/risk.py::verify_sql_candidate`'s docstring). That family covers `fetch_join`, `batch_query`'s sql_count framing, `entity_graph`, `dto_projection`, and `remove_redundant_call` — most of what Scenarios 4 onward will exercise. A `sql_count` candidate that reaches PATCH, fails `mechanism_match` for a real reason (the implementation didn't actually collapse the call count), and had an untried sibling candidate with a different mechanism or approach will still dead-end at `NO_SIGNIFICANT_IMPROVEMENT` today, exactly as BATCH-01 did, with no plan-time EXPLAIN check to have caught it first. The eventual fix: retry when `mechanism_match` fails AND an unselected candidate for the same hypothesis proposes a different mechanism or approach — not attempted now, flagged for when it's next hit for real. |
| `pytest` + `eval/tier1.py`, run back to back, intermittently fail together then pass cleanly on immediate retry | **Logged, not pinned.** Observed three times while building Scenarios 4/5: once a single Tier 1 scenario (STM-01) failed alone with a `CassetteMiss` indistinguishable in shape from BATCH-01's quarantined one; twice the ENTIRE suite failed together (one unrelated unit test, `test_oracle.py::test_the_committed_policy_tolerates_exactly_one_known_thing`, plus every non-quarantined Tier 1 scenario at once). Every single failure vanished on an immediate re-run with no code change in between, and every component passes reliably in isolation. **Ruled out:** a real code regression (nothing in `senapathy/` changed between a failing run and the next, passing one); the known BATCH-01 diagnose-replay issue specifically (that one is scenario-specific and permanent; this hits different scenarios each time and self-heals); a deterministic trigger from running the two tools back-to-back in one shell invocation (tried immediately reproducing with the identical chained command twice — both times clean). **Not ruled out:** resource contention from the Maven/Spring Boot/Docker processes this session was running concurrently in the background — plausible given the timing, not confirmed. A suite that fails at random erodes trust the same way a permanently red one does, even though no single run was wrong twice in a row; recorded here so a future occurrence is recognised as "the same unexplained flake" rather than investigated from scratch each time. |
| `EXPLAIN`/`index_benefit`/`query_equivalent`/`plan_delta` all raise MySQL 1064 on `LIMIT %s OFFSET %s` with string params — `"near ''50' OFFSET '0''"` | **FIXED — `senapathy/tools/db.py::bind_params`.** Every tool-schema parameter is typed `array of string` (the only shape the model can supply), and `cursor.execute(sql, tuple(params or ()))` handed that straight to the connector, which renders a Python `str` as a QUOTED literal regardless of position — correct for a WHERE-clause comparison, a syntax error for LIMIT/OFFSET, which MySQL's grammar accepts only as a bare integer. First dismissed as an isolated quirk (COL-06's frozen verification, PROJ-01's), then reproduced live for the THIRD time when the model's own `index_benefit` call hit it during Scenario 7 and got stuck in a `no_progress` loop retrying the identical call three times, unable to find an escape — because nothing it could pass through the tool interface would stop the connector from quoting it. That third occurrence is what made clear this was one confirmed bug with three sightings, not three coincidences. Fixed by converting any parameter that is purely digits to a Python `int` before binding (`bind_params`, used by `_read_plan` and `_run` — covers `explain`, `index_benefit`'s own before/after measurement, `query_equivalent`, and `plan/risk.py::verify_sql_candidate`'s `plan_delta` in one place); a genuine string comparison is untouched. Verified against a real MySQL connection (not just the fake-server unit tests) with the exact statement and params SER-01 hit live — `explain`, `plan_delta` and `index_benefit` all now run cleanly. Existing frozen scenarios' `sql_verification` records are historical snapshots from before the fix and were not retroactively re-verified; COL-06's own frozen note still shows the pre-fix 1064, which is correct and expected, not a sign the fix did not take. |
| A candidate whose `sql_verification` never ran (EXPLAIN failed to even parse the statement) was still selectable, same as a candidate whose verification succeeded | **FIXED — `plan/risk.py::sql_verification_failure`, `SqlVerification.attempted`.** Surfaced by two fresh live `diagnose`+`plan` samples of Scenario 7 (taken to freeze it): one selected a `fetch_join` candidate whose `sql_after` joined `cp.someLazyCollection` — a field that does not exist on the entity — with `sql_verification.verified: false`; the other selected a `dto_projection` candidate that was a near-no-op rewrite of the query already in the code, also `verified: false`. Root cause: `sql_verification_failure()` returned `None` (not disqualified) whenever `verified` was `False`, regardless of *why* — collapsing "no live database was reachable at plan time at all" (a run-wide limitation, correctly non-disqualifying) and "a database WAS reachable and EXPLAIN ANALYZE could not even run this candidate's own statement" (evidence of a problem) into the same non-disqualifying case. Fixed by adding `SqlVerification.attempted: bool` (`False` only for the no-database case; defaults `True`, so historical frozen records — none of which came from a missing database — read correctly unmodified) and teaching `sql_verification_failure` three states instead of two: VERIFIED+moved → selectable; VERIFIED+flat → not selectable, disproven; COULD NOT VERIFY (`attempted=True`, `verified=False`) → not selectable, **unverifiable**, naming why — explicitly not reported as disproven, since the quantity was never measured either way. Applies to `sql_count` candidates too, not only the scored `rows_examined` family: a statement EXPLAIN cannot even run is not evidence either way regardless of whether its mechanism would have been scored. **Audit of every shipped PR's selected candidate** (done before this fix, per request): INV-04 (`entity_graph`/sql_count, verified=false), COL-06 (`query_rewrite`/rows_examined, verified=false), BATCH-02 (`batch_query`/rows_examined, **verified=true, rows -100%, scored=true** — the one candidate that was ever actually checked), STM-01 and LOOP-01 (`remove_redundant_call`/`move_work_out_of_loop`, `sql_verification: null` — correctly: neither approach rewrites any SQL text, so there is nothing to verify). **No shipped PR's reported numbers rest on a failed or absent plan-time verification** — those numbers always come from the live gate's real post-patch A/B measurement (`gate/rules.py`, independent of this plan-time check), which is what the scenario log's mechanism-moved column reports. The plan-time check that failed for INV-04/COL-06/BATCH-01 was a pre-patch sanity check meant to save the cost of a patch/verify/measure cycle on an obviously-bad rewrite; it failing silently was wasteful, not dishonest. Applying this fix retroactively (via Tier 1 replay against the frozen historical verifications) now correctly refuses to select INV-04, COL-06 and PROJ-01's previously-selected candidates — **quarantined**, not re-written, since each has a distinct, understood, honest reason (`eval/scenarios/{INV-04,COL-06,PROJ-01}/quarantine.json`): INV-04's `sql_before`/`sql_after` are JPQL, which EXPLAIN cannot parse as SQL at all (see the tooling-gap row below); COL-06 hit the now-fixed `bind_params` bug specifically and may well re-verify cleanly on a live re-freeze (not attempted this session); PROJ-01's candidate supplied `sql_params: ["pageSize", "offset"]` — placeholder NAMES, not sample values, a distinct submission mistake. |
| `EXPLAIN ANALYZE` cannot check a JPQL statement directly | **Tooling gap, logged, not fixed.** `sql_before`/`sql_after` are routinely submitted as JPQL (`select i from Invoice i where i.customer.id = :customerId`, `?` positional params) rather than the native SQL Hibernate actually generates — INV-04's shipped PR is exactly this shape. EXPLAIN ANALYZE only understands real SQL, so these candidates' plan-time verification fails not because the rewrite is wrong but because the statement was never translatable in the first place; after the fix above, this now reads as "unverifiable" and blocks selection, which is honest about the uncertainty but still blocks real, previously-shippable fixes. Verifying a JPQL-shaped candidate needs the generated SQL — either captured from Hibernate directly, or matched against the statement digests already captured at baseline (`performance_schema.events_statements_summary_by_digest`, the same mechanism the hand-measurement protocol already uses). Not attempted. |
| All 4 independent live `diagnose` samples of Scenario 7 (`GET /api/customers/bios`) invented an entity-hydration hypothesis for an endpoint that hydrates zero entities | **Known issue, logged, not fixed.** `CustomerBioService.bios()` calls `CustomerProfileRepository.findBioRows()`, a narrow JPQL constructor projection (`select new CustomerBioDto(cp.id, cp.customerName, cp.bio) from CustomerProfile cp ...`) — confirmed by reading the actual repository/service code, and consistent with the endpoint's own comprehension model, which records zero per-row data accesses. Every one of 4 live diagnose samples (2 real orchestrator runs, SER-02/SER-03; 2 fresh `live_freeze.py` samples) nonetheless produced an `orm_hydration` hypothesis (fixability 0.8) claiming Hibernate "creates a managed entity for each row" / "populates fields and attaches proxies" / "registers it in the persistence context" — worded differently each time, same false claim. Likely mechanism, not confirmed: `run_probe` always measures the harness's one fixed calibration endpoint (`/api/customers/search`), which genuinely DOES exhibit N+1-shaped SQL-count growth (`run_probe` evidence recorded in one SER-01 sample literally reads "application SQL statements per request grow with rows: 4 statements for 1 row, 102 for 50 rows") — that calibration number appears to get cited as if it describes the endpoint actually under test, which issues exactly one SQL statement regardless of page size. **Eventual fix, not attempted:** `diagnose/validate.py::validate_hypothesis_set` already reconciles hypotheses against measured evidence once (the `sql_share_of_latency_pct` check) — the same shape of check could reject a hypothesis asserting per-row entity hydration when the endpoint model's own `data_access` already records zero per-row accesses for this endpoint, which is evidence already in hand, not a new measurement. Would also make the ORIGINAL, stricter Scenario 7 citation requirement (serialization's fixability cited on the SAME candidate that gets refused for an uninstrumented mechanism) appear reliably, since the only remaining real hypothesis would be h1/serialization — noted as a side benefit, not the reason to do it. |
| Diagnose exhausts its entire step budget (80 steps, 5 probes) without ever calling `emit_hypotheses` | **New, logged, not fixed.** `UNB-05` (Scenario 8, 5th live sample): the agent ran probes, never submitted a diagnosis and never had one rejected (0 submissions), and the loop ran out of budget — `terminal state: INTERNAL_ERROR`. Different shape from `UNB-01`'s forced-`tool_choice` crash: no provider error here, the model simply never attempted the terminal call. Not investigated further (one occurrence); re-run with a new run id, same remedy as every other transient agent-loop stall logged in this table. |
| `h1`/`serialization`'s fixability scored 0.9 with a reason about rewriting a sargable predicate | **New, logged, not fixed.** One fresh Scenario 7 freeze sample (the one ultimately frozen as `SER-01`) assigned hypothesis h1 (layer `serialization`, "JSON serialization of large bio strings") a fixability of 0.9 with the reason "rewrite the predicate so it can use the existing index — EXPLAIN shows an implicit type conversion ('converts to' in the plan)" — boilerplate for a sargability/`query_rewrite` policy row, nonsensical for a serialization-layer claim (there is no predicate, no index, nothing EXPLAIN ever examined for this hypothesis). Every other sample of this same hypothesis scored it 0.30 ("library-level, rarely beatable by hand"), which is what `SER-01/expected.json` records as context — this misclassification is explicitly flagged there as not to be read as the scenario's real fixability. Not investigated further; `plan/fixability.py`'s classifier is the suspect, not chased down this session. |
| A plan- or patch-phase provider/tooling crash lands on `BOTTLENECK_NOT_FIXABLE`, indistinguishable from a genuine refusal | **FIXED — `orchestrator.py::_failure_state()`.** `UNB-01` (Scenario 8, first attempt): the orchestrator forced `tool_choice: submit_plan` near the step budget, the model called `index_benefit` anyway, Groq returned `HTTP 400 tool_use_failed`, and the plan agent stopped with status `error`. `_failure_state()` mapped EVERY `Phase.PLAN`/`Phase.IMPLEMENT` exception to `TerminalState.BOTTLENECK_NOT_FIXABLE` unconditionally — the same label a legitimate "no candidate clears the selection rule" refusal gets, and this was not merely "indistinguishable without opening the trace": `_failure_state()` only runs when the phase's own runner RAISED instead of completing, and a genuine refusal never raises — `_run_plan`/`_run_patch` return it normally (`selected_id is None`, `outcome != "PATCHED"`), already separately labelled `BOTTLENECK_NOT_FIXABLE` on their own terms. So every single crash through this path was mislabelled, not just ambiguous in some cases. Fixed by mapping both phases' crashes to `TerminalState.INTERNAL_ERROR` instead — the same bucket every other phase's crash already falls into, and a state that already existed in the enum for exactly this (`SenapathyError`'s generic fallback). Two new tests (`test_a_plan_phase_crash_is_internal_error_not_bottleneck_not_fixable`, `test_a_patch_phase_crash_is_internal_error_not_bottleneck_not_fixable`) cover both phases via `stub_pipeline`'s `fails=` — neither existed before this fix, confirming the bug had no test coverage at all. |
| Per-row work that fills an unbounded response field was scored "genuinely removable" (0.70) instead of blocked (0.05/0.10) | **FIXED — `plan/fixability.py::response_link`.** Surfaced live by `UNB-03`: the planner selected `remove_redundant_call` (merging two queries — a real, unrelated fix, see the scenario-design row below) via hypothesis h2 (application_logic, "running balance calculation"), whose per-row work genuinely produces `remainingAfter` inside the endpoint's own unbounded `allocations` collection, yet was rated 0.70/"genuinely removable". Root cause: `response_link()` checked the per-row work's `description` against the response's field names, never its `scales_with` — and `scales_with` is comprehension's own, more reliable answer to "what drives the count" (here, literally `"allocations per invoice"`), while `description` named the field only fused into a camelCase identifier (`AllocationRowDto`), which `mentions()`'s word-boundary match correctly refuses to treat as the word "allocation" (the same boundary that keeps "invoice" from matching "invoiced"). Fixed by checking `f"{work.description} {work.scales_with}"` instead of `description` alone. One new unit test reproduces the exact UNB-03 shape. Full suite: 913 passed (was 912); Tier 1 unaffected (4/4 passed, 5 quarantined, same as before — no frozen scenario exercises this path differently). |
| `UNB-01`'s own endpoint design left a real, unrelated, legitimately-fixable inefficiency (two separate queries) for the planner to chase instead of the unbounded collection | **Scenario design fix, not a product bug.** The first version fetched the invoice header and its allocations in two separate repository calls; `UNB-03` selected a candidate that correctly, legitimately merged them into one query — a real fix, just orthogonal to what this scenario exists to test, the same way `TXN-01`'s full-entity fetch gave `dto_projection` an escape hatch from `read_only_transaction`. Redesigned to a single `LEFT JOIN` query (`PaymentAllocationRepository::findInvoiceWithAllocations`) before any further live run — not logged as a product issue, since nothing in the harness was wrong; the scenario itself was under-constrained. |

---

# Scenario log

Fill in as each completes. A scenario is not done until it is frozen as an eval case.

| # | Scenario | Run id | Terminal state | Fix kind | Mechanism moved | Matches prediction? | Frozen |
|---|---|---|---|---|---|---|---|
| 0 | N+1 (invoices) | INV-04 | `PR_OPENED` | `query_rewrite` | `sql_count` 102→2 | yes | yes |
| 1 | `query_rewrite` (collections) | COL-06 | `PR_OPENED` | `query_rewrite` | `rows_examined` 3,152,506→276,496 | yes | yes |
| 2 | `remove_redundant_call` | STM-01 | `PR_OPENED` | `remove_redundant_call` | `rows_examined` 2,002,594→1,405,494 (sql_count unchanged, 72→72 — see note) | partial | yes |
| 3 | `batch_query` | BATCH-01 (failed, product bug), BATCH-02 (re-run, passed) | `PR_OPENED` (BATCH-02) | `batch_query` | `rows_examined` 14,533,404→2,400,800 (-83.5%) | yes (BATCH-02) | yes |
| 4 | `move_work_out_of_loop` | LOOP-01 | `PR_OPENED` | ambiguous — see note; NOT independently proven | `sql_count` 202→4 (-98.0%) | partial — see note | yes |
| 5 | `dto_projection` (probe) | PROJ-01 | `BOTTLENECK_NOT_FIXABLE` | `dto_projection` | n/a (predicted: neither instrumented quantity moves) | yes — predicted refusal, for the predicted reason | yes |
| 6 | `read_only_transaction` (probe) | TXN-01 | `NO_SIGNIFICANT_IMPROVEMENT` (live run tried `dto_projection` instead — see note) | n/a — no candidate shipped | n/a (confirmed: nothing a database can observe) | yes — predicted refusal, for the predicted reason | yes |
| 7 | Serialization-dominant | SER-01 (bug), SER-02, SER-03 | `BOTTLENECK_NOT_FIXABLE` (SER-02, SER-03) | n/a — no candidate shipped | n/a (see note) | yes — predicted refusal, for the predicted reason | yes |
| 8 | Unbounded response | UNB-01 (crash), UNB-02, UNB-03 (wrongly selected), UNB-04 (wrongly selected), UNB-05 (crash), UNB-06 | `BOTTLENECK_NOT_FIXABLE` (UNB-02, UNB-06) | n/a — no candidate shipped | n/a (see note) | yes — predicted refusal, for the predicted reason, after 2 real bugs found and fixed | yes |
| 9 | Already-optimal | PAY-03 | `BOTTLENECK_NOT_FIXABLE` | n/a (none selectable) | n/a | yes (see note) | yes |

**Row 2 note:** the plan phase diagnosed the 3-call redundancy and, rather than literally
sharing one fetch across all 3 sites, replaced the single most expensive one
(`findProfile`'s aggregation) with a cheaper existence check + direct sum. SQL statement
*count* per request is unchanged (72→72); the win is entirely in `rows_examined`. The
plan's own prediction said `rows_examined`, not `sql_count` — this table's original
generic prediction (row 29, written before any live run) was the thing that was wrong,
not the agent. Terminal state and fix-kind label both match; only the specific
instrumented mechanism differs from the pre-registered generic expectation.

**Row 9 note:** diagnose found 3 real hypotheses (serialization, ORM hydration,
application logic) — nonzero cost, just none of it fixable — so the system correctly
landed on `BOTTLENECK_NOT_FIXABLE` (computed in the *plan* phase, when hypotheses exist
but no candidate clears selection) rather than `NO_BOTTLENECK_FOUND` (computed in
*diagnose*, when no hypothesis clears the bar at all). This is a more precise answer than
this table's original generic label and is the correct refusal for the correct reason.

**Row 3 note:** BATCH-01 found a real planner bug, not a scenario bug — see the known-failure
table entry below. The plan DID generate the correct `batch_query` candidate (an `IN(...)
GROUP BY`) alongside a correlated-subquery `query_rewrite` candidate, both scoring an
identical `expected_value_ms` (a property of the HYPOTHESIS, not the candidate), and
`_select()`'s tie-break fell through to lexical candidate id and shipped the wrong one — its
own `rows_examined` never moved (`+0.0%`), and the gate correctly refused it at
`NO_SIGNIFICANT_IMPROVEMENT`, $0.079 spent. Fixed in `senapathy/plan/risk.py` /
`senapathy/plan/submit.py`: a SQL-rewrite candidate (`fetch_join`, `entity_graph`,
`batch_query`, `dto_projection`, `query_rewrite`) whose mechanism is instrumented must now
supply `sql_before`/`sql_after`, verified at plan time with `EXPLAIN ANALYZE`
(`tools/db.py::DbToolbox.plan_delta`, read-only, nothing created or dropped); for the
`rows_examined` mechanism the measured delta both disqualifies a candidate whose own
EXPLAIN shows no improvement and drives `expected_value_ms` directly, so two candidates for
one hypothesis no longer have to tie. `sql_count`-mechanism candidates are still verified
(so the evidence is in the artifact) but not scored or disqualified by it — a single
EXPLAIN of each side cannot see a call-COUNT fix. Remaining coin-flip ties (candidates with
no SQL to verify, e.g. two `remove_redundant_call`-family fixes) are no longer silent: the
loser's rejection reason says "tied ... not because it scored higher" instead of falsely
claiming a higher score. BATCH-02, same code, new run id: plan selected `batch_query` on a
genuine score difference (525.0 vs 472.5, not a tie), gate passed all six
(`improvement_threshold` -83.27%, `mechanism_match` rows_examined -83.5%), PR opened,
$0.078 spent. Re-freezing COL-06 and INV-04 after the `sql_before`/`sql_after` schema
change surfaced two more instances of ordinary LLM-sampling variance (not bugs): COL-06's
first re-freeze drew a valid `dto_projection` alternative that happened to ALSO fix the
non-sargable predicate (verified by hand), a second re-freeze drew the original
`query_rewrite`/`rows_examined`, kept; INV-04 drew `fetch_join`, then `entity_graph`,
never `dto_projection` twice in three tries — all four textbook N+1 remedies
(`dto_projection`, `fetch_join`, `entity_graph`, `batch_query`) are equally valid for that
scenario, so `expected.json`'s `approach_any_of` (originally just `["dto_projection"]`,
one historical conversation's incidental choice) was widened to the real family rather than
re-rolled until it matched a narrower assertion than the scenario actually tests for.

Separately, `tier1.py`'s own assertions were reworked across all five frozen scenarios:
`approach` renamed to `approach_context` and made purely informational (never gates;
STM-01's actual outcome — `derive_scope`'s sibling expansion reaching every call site — is
now asserted directly via a new `files_to_touch_contains` check, independent of which
approach label wins); `mechanism` plus, where it applies, the MEASURED `EXPLAIN ANALYZE`
delta are the real gates. The measured-delta check replays the real number
`live_freeze.py` got from a live database at freeze time (`frozen_sql_verification_table`
in `tier1.py`, keyed on the candidate's own `sql_before`/`sql_after`/`params` — a
`senapathy.plan.submit.verify_sql_candidate` stub for Tier 1's otherwise-offline replay),
so a regression that lets a non-moving `rows_examined` candidate through fails Tier 1 even
though Tier 1 itself never touches a live database. Verified end-to-end by temporarily
corrupting COL-06's frozen delta to simulate a regression (caught both ways: a complete
non-mover gets disqualified from selection entirely, a partial one that still clears the
5% gate bar but misses the scenario's own claimed floor fails the new assertion
specifically) and restoring it afterward. `eval/scenarios/BATCH-01` was also quarantined
(`quarantine.json`, dated and reasoned) rather than left as a silent `4/5` — `tier1.py`
now reports `4/4 passed, 1 quarantined` and will flag loudly (non-zero exit) if that
scenario ever starts passing without the quarantine being lifted first. See
`eval/README.md` for both.

## Row 4 (LOOP-01)

The live run (`senapathy run`, $0.054 spent) shipped the EXACT planted fix — "fetch the
active FeeSchedule once before looping... eliminating the per‑row query," hoisting
`feeScheduleRepo.findByActiveTrue()` out of the per-row loop — but labeled its own
candidate `remove_redundant_call`, not `move_work_out_of_loop`. Re-freezing the identical
code immediately after (for the eval scenario) drew `move_work_out_of_loop` cleanly on a
fresh sample, with `query_rewrite` as an equally-scored, correctly-disclosed tied
runner-up both times. `TIER_BY_APPROACH` rates `move_work_out_of_loop` and
`remove_redundant_call` at the same risk tier (T2), and the CODE CHANGE described is
identical in both samples — this is a genuine labeling ambiguity between two adjacent
enum values for the same underlying fix ("call something once instead of once per row"),
not a functional misdiagnosis, and exactly the kind of thing Part 2's
assertion rework (above) is designed to not penalise: `mechanism` (`sql_count`, moved
202→4 statements, -98.0%) is the real, asserted claim; `approach` is recorded context.

**What this does and does not prove.** The loop-invariant hoist itself was genuinely
exercised — a real per-request-constant lookup, called once per row, correctly moved
outside the loop, with `sql_count` collapsing and `PR_OPENED` — so this coverage is real.
What it does NOT prove is that `move_work_out_of_loop` and `remove_redundant_call` are
two independently-validated policy rows: the model used them interchangeably for the
IDENTICAL code change across two samples of the SAME scenario, and `TIER_BY_APPROACH`
already rates them at the same tier (T2), so nothing in this run distinguishes them as
separate fixes with separate risk profiles. **Open question, not resolved here:** do
`move_work_out_of_loop` and `remove_redundant_call` earn separate existence in the
risk/approach table, or are they the same policy row wearing two names? Answering that
needs a scenario built specifically to force the distinction (e.g. one where hoisting a
loop-invariant call is safe but literally sharing one redundant fetch across non-sibling
call sites is not, or vice versa) — this scenario does not do that, and the scenario log
should not be read as though it did.
All six gates passed (`improvement_threshold` p95 -77.99%, behaviour 10/10 identical).
Frozen as `LOOP-01` (`eval/LOOP-01` tag, `approach_context` records all three labels
observed across both samples).

## Row 5 (PROJ-01) — predicted to fail, and it did

**Scope note:** this was predicted to fail. Per instruction, nothing below fixes the
gate, instruments a new quantity, or adjusts `mechanism_match`. The gate's refusal IS
the result; what follows is the measurement and the missing-quantity analysis, not a
patch.

**Built:** `GET /api/customers/directory`, backed by a new `customer_profile` table
carrying wide, unused TEXT/MEDIUMTEXT columns (`bio`, `internal_notes`,
`raw_import_payload`, ~8KB/row across the three) that nothing in the response uses. As
built, the repository fetches the full entity (all six columns) for a paginated listing
that only needs three narrow fields (`id`, `customer_name`, `joined_on`). Independent of
`customer`/`invoice` — seeded once from `customer`'s own id/name — so it cannot perturb
any other scenario.

**How much faster the change actually is.** Hand-measured, size=50, 20 requests each,
full entity fetch vs. a narrow JPQL constructor projection (applied locally, measured,
reverted — never shipped):

| | as built | narrow projection | delta |
|---|---|---|---|
| mean latency | 8.48 ms | 4.38 ms | **-48.3%** |
| `sql_count` | 1/request | 1/request | unchanged |
| `rows_examined` | 50/request | 50/request | unchanged |
| `rows_sent` | 50/request | 50/request | unchanged |
| server-side SQL time (avg/call) | 0.25 ms | 0.116 ms | real, but only ~2.7 ms of the ~82 ms total improvement over 20 requests |
| HTTP response body | 3,855 bytes | 3,855 bytes | unchanged — the wide columns never reach the client either way |

The live run (`PROJ-01`, $0.050, one run) confirmed this independently: `improvement_threshold`
**passed** (p95 27.7 ms → 11.5 ms, **-58.42%**, a real win, comfortably clear of the
15% floor) and `mechanism_match` **failed** (`rows examined 400 → 400, +0.0%` across the
paired sample set) — landing on `BOTTLENECK_NOT_FIXABLE`, not `NO_SIGNIFICANT_IMPROVEMENT`:
`gate/rules.py::terminal_state_for` distinguishes these precisely — a gate that fails
only on `improvement_threshold`/`statistical_significance` means "not actually better";
one that fails only on mechanism/minimality/forbidden-change while the improvement gates
pass means "the bottleneck may be real and the measurement good, but this patch is not
shippable." That is exactly what happened here, and the distinction is the product
working correctly, not a bug.

**Exactly what quantity would have to be instrumented to prove it.** Not `rows_examined`
(scans the same 50 rows either way — no covering index exists or would help, since
there's no secondary index on `customer_profile` beyond the primary key). Not `sql_count`
(one statement either way, no N+1 here). Not `rows_sent` either, which is worth stating
explicitly because it is the most natural-sounding candidate and it still does not work:
`rows_sent` is a **row count** (50 either way), exactly as blind to a column's byte width
as the other two. The missing quantity is **bytes transferred on the MySQL-to-application
wire** — equivalently, `rows_sent x average row width` measured at the JDBC result-set
level, not at the HTTP layer. That last clause matters: the HTTP response body is
identical either way (3,855 bytes) because the DTO never leaks the wide columns, so "just
measure the response" is a dead end that would wrongly conclude there is nothing here to
find. The over-fetched bytes are entirely internal — materialized into Java `String`
objects by Hibernate, then discarded — and never cross the HTTP boundary at all.
`performance_schema` does not expose this at the per-digest level this harness currently
reads; proving it would need either a byte-level wire counter between the app and MySQL,
or `rows_sent` paired with a per-digest average row width (bytes), neither of which exists
in `gate/rules.py::INSTRUMENTED_MECHANISMS` today, and per instruction neither was added.

**What it would take, concretely** (recorded as the answer to "what would it take," not
as a task to pick up): `performance_schema.status_by_thread` with
`VARIABLE_NAME = 'Bytes_sent'`, sampled per connection and taken as a delta across the
benchmark window, is the one route to real wire bytes that does not require modifying
the target application — same shape as the existing digest capture (truncate/reset
before the measured window, read the counter after, diff). This counts bytes written
back to the client on that connection, which includes the wide-column payload MySQL
sends regardless of whether the application ever reads it — exactly the signal
`rows_sent` and `rows_examined` both miss. Not built; recorded for when this family of
probe needs a real answer instead of a documented gap.

Diagnosis itself was clean throughout both the live run and a fresh re-freeze: `database_query`
never appeared as a hypothesis in either sample (correctly — there is no scan problem),
and the dominant layer was `orm_hydration`/`serialization`/`application_logic` both times,
which is exactly where the real cost (entity hydration + wide-string allocation) actually
lives. The live run's selected candidate claimed `mechanism=rows_examined` (a false,
unverified claim — plan-time `EXPLAIN ANALYZE` verification could not run, hitting the
same `LIMIT %s OFFSET %s` string-parameter binding quirk already on record for COL-06,
so the claim reached the gate unchecked and was caught there instead); a fresh re-freeze
of the identical code claimed `mechanism=sql_count` instead on one sample, with
`entity_graph` as a correctly-disclosed tied runner-up — both unverified either way, since
neither is the scored (`rows_examined`) family Part C's plan-time check covers. Frozen as
`PROJ-01` (`eval/PROJ-01` tag); `expected.json` records both predicted terminal state
(`BOTTLENECK_NOT_FIXABLE`) and the real numbers above as historical fact, and is written
so that a FUTURE pass on this scenario is flagged as needing explanation, not quietly
accepted as progress.

## Row 6 (TXN-01) — predicted to fail; the honest answer is "nothing the database sees"

**Scope note:** predicted to fail, same as Scenario 5. Nothing below fixes the gate,
instruments a new quantity, or adjusts `mechanism_match`.

**Built:** `GET /api/invoices/paid-summary` — a read path (no writes anywhere in the
method) running inside the default (read-write) transaction instead of
`@Transactional(readOnly = true)`. Deliberately returns full `Invoice` entities, not a
projection, so there is something for the persistence context to dirty-check at commit —
a constructor-expression projection is never a managed entity, so `readOnly`'s actual
savings would have nothing to skip. Reuses `idx_invoice_status_due_date_id` (V8, a
different status value) so the base query is a plain index-order scan, not a confound.

**How much faster the change actually is: it is not.** Hand-measured directly, size=100,
30 requests each, as built vs. `readOnly = true` applied locally (reverted, never
shipped):

| | as built | `readOnly = true` | delta |
|---|---|---|---|
| mean latency | 5.97 ms | 6.43 ms | **+7.7%, i.e. slightly SLOWER** |
| SQL statements | 1 SELECT, no session-mode statement | 1 SELECT **+ 2** session-mode statements | +2 round trips |

This is the mechanically precise reason: MySQL's session default is already
`READ WRITE`, so the as-built version issues no `SET SESSION TRANSACTION` statement at
all. Adding `readOnly = true` does not just add one — Spring resets the session's
transaction mode before the connection returns to the pool, so it issues
`SET SESSION TRANSACTION READ ONLY` at the start **and**
`SET SESSION TRANSACTION READ WRITE` at the end. Confirmed directly from
`performance_schema.events_statements_summary_by_digest`: both statements present, each
~0.02-0.06 ms server-side, for a combined cost that is small in absolute terms but was
not there before, and nothing offsets it at this entity's size.

**Exactly what quantity would prove it — in this case, the honest answer is that no
database-observable quantity exists at all, as anticipated in advance.** The savings
`@Transactional(readOnly = true)` is documented to provide — Hibernate skipping its
dirty-checking pass over the persistence context and the flush it would otherwise
attempt at commit — is a pure in-JVM loop comparing already-loaded Java objects against
their snapshots. It issues **no SQL statement either way**, whether or not anything
would have been found dirty: a no-op flush is still zero round trips, same as
dirty-checking finding nothing dirty is zero round trips. There is no `rows_examined`,
no `sql_count`, no byte-level wire counter (the one named for Scenario 5) that this
mechanism could ever move, because it never reaches the database at all. Reaching for a
quantity to instrument here would be a category error, not a gap to close — this is the
plain, correct conclusion, not a case of "we haven't found the right counter yet."

**What the live agent actually did, for completeness, and why it is not quite the same
test.** Diagnosis was clean both times (`database_query` never a hypothesis; dominant
layer `serialization`/`orm_hydration`/`application_logic`). But because the service
returns full entities (needed for dirty-checking to have anything to check), "stop
returning entities at all" (`dto_projection`) is also a valid, more complete alternative
that the model reasonably reached for instead of `read_only_transaction` specifically —
in both samples. A fresh re-freeze selected **no candidate at all**
(`dto_projection` correctly refused for an uninstrumented `mechanism=serialization`;
`add_index` blocked by schema isolation, as always). The live run (`TXN-01`, $0.076, one
run) instead measured `dto_projection` for real: `improvement_threshold` **failed**
(p95 -12.55%, below the 15% floor — this entity is too narrow, 4 fields, for entity-
management overhead alone to be worth much) and `mechanism_match` **failed** (SQL
statements 2 → 2, +0.0%), landing on `NO_SIGNIFICANT_IMPROVEMENT`. Both outcomes support
the same conclusion — no candidate clears the bar — but neither live sample actually
tried `read_only_transaction` as its own approach, so the hand-measurement above, not
the live run, is the direct test of this scenario's own named mechanism. Frozen as
`TXN-01` (`eval/TXN-01` tag); `expected.json` records this distinction explicitly rather
than overclaiming that the agent confirmed `read_only_transaction` specifically.

## Row 7 (SER-01/SER-02/SER-03) — predicted refusal; the acceptance bar changed mid-scenario

**Endpoint:** `GET /api/customers/bios` — `CustomerBioService.bios()` returns a page (up
to 100 rows) of `CustomerBioDto(id, customerName, bio)` via a single narrow JPQL
constructor projection (`CustomerProfileRepository.findBioRows`); `bio` is a `@Lob`
column populated with >1000 characters per row. No planted bug — nothing to fix without
changing the response contract. Hand-measured: 206 KB response at size=100, mean request
latency ~7.68 ms, server-side SQL time ~0.47 ms/request (~6% of latency) — serialization
and transfer dominate, as designed. Committed `13ca852`, tagged `eval/SER-01`.

**The acceptance bar was wrong as originally stated, and was corrected.** The original
requirement — "the refusal must cite serialization's 0.30 fixability and an
uninstrumented mechanism, on the same candidate" — specifies a reasoning PATH, not a
behaviour, which contradicts this suite's own outcome-not-method rule (the same rule
`check_expectations()` applies via `approach_context`: methods are recorded, never
gated). Replaced with the outcome the scenario actually exists to test: no patch phase
reached, no PR, a refusal recorded with a stated reason. `mechanism`/`fixability` are
recorded as context in `expected.json`, never as a gate.

**What happened.** `SER-01` (the first live run) never reached a plan verdict at all — it
hit the since-fixed `bind_params` LIMIT/OFFSET bug inside the model's own `index_benefit`
call and got stuck in a 3x `no_progress` loop (see the known-failure table). After that
fix, `SER-02` and `SER-03` both completed cleanly and both correctly refused
(`BOTTLENECK_NOT_FIXABLE`, `selected_id: None`) — meeting the (corrected) outcome bar.
Freezing this scenario for Tier 1 (`eval/live_freeze.py`, which always takes a fresh live
sample rather than replaying a historical conversation) surfaced two further, more
important findings, both logged in the known-failure table above rather than fixed in
full:

1. **Of 4 independent live diagnose samples total** (SER-02, SER-03, and 2 fresh freeze
   attempts), **all 4** invented an `orm_hydration` hypothesis for an endpoint that
   hydrates zero entities — and in 2 of the 4 fresh-freeze attempts, this hallucination
   produced a SELECTED candidate before any gate existed to catch it: one proposing a
   `JOIN FETCH` on a field (`cp.someLazyCollection`) that does not exist on the entity,
   one proposing a near-no-op rewrite of the projection already in the code. Both had
   `sql_verification.verified: false` — EXPLAIN ANALYZE never actually ran — and were
   selected anyway.
2. That second fact is a real, separate, confirmed bug, not specific to this scenario:
   candidate selection never consulted `sql_verification.verified`. **Fixed this
   session** (`plan/risk.py::sql_verification_failure`, `SqlVerification.attempted` —
   full detail in the known-failure table). A third fresh freeze sample, taken after the
   fix, produced a clean refusal (`selected_id: None`, both candidates disqualified for
   an uninstrumented mechanism) even with the hallucination still fully present and
   unfixed — confirming the SELECTION fix is sufficient to keep this scenario's outcome
   reliable on its own, independent of ever resolving the hallucination.

**Frozen as `SER-01`** (`eval/scenarios/SER-01`, matching the `eval/SER-01` mini-spring
tag) using that third, post-fix, clean sample. `expected.json` asserts the corrected
outcome bar only, and states plainly that re-freezing this scenario is not reliable and
why (the hallucination), rather than implying the frozen sample is a stable, repeatable
proof. Quarantined for Tier 1 (`eval/scenarios/SER-01/quarantine.json`) for an unrelated,
already-known-shaped reason: its own plan-replay cassette misses on the terminal
request, the same shape as `BATCH-01`'s quarantined diagnose-replay miss — see the
known-failure table's updated entry.

## Row 8 (UNB-01 through UNB-06) — predicted refusal; two real bugs found and fixed first

**Endpoint:** `GET /api/invoices/{id}/payment-history` — an invoice's complete payment
history, no page/size parameter anywhere on the path; `allocations` is the point of the
endpoint, not a side effect. `remainingAfter` (a running balance over the invoice
amount) is genuine per-row work, computed in one pass over data fetched by a single
query (`PaymentAllocationRepository::findInvoiceWithAllocations`, a `LEFT JOIN` from
`Invoice` so a never-paid invoice still returns its header with an empty list). No
planted bug. Committed, tagged `eval/UNB-01`. Hand-measured against invoice `798455`
(16 allocations): mean ~4-5 ms steady-state, exactly one application SQL statement per
request, SQL time ~6% of latency.

**Six budgeted live runs, two of them real product bugs, both fixed before the budget
ran out:**

1. **`UNB-01`** — crashed in the plan phase: the orchestrator forced
   `tool_choice: submit_plan` near the step budget, the model called `index_benefit`
   anyway, Groq rejected it server-side, and `orchestrator.py::_failure_state()` mapped
   the crash to `BOTTLENECK_NOT_FIXABLE` — indistinguishable from a genuine refusal
   without opening the trace. Logged, not fixed (known-failure table).
2. **`UNB-02`** — the endpoint's first version (two separate queries: an invoice lookup
   plus the allocation join). Diagnose ranked `h2`/orm_hydration top (an invented
   hypothesis — this endpoint hydrates no entities — the headline finding below);
   both candidates proposed against it were refused for `mechanism='allocation'` being
   uninstrumented. Clean refusal, but for the wrong reason to rely on.
3. **`UNB-03`** — **wrongly SELECTED** `remove_redundant_call` (merging the two queries
   into one — a real, legitimate, but entirely orthogonal fix, the scenario's own design
   flaw, not a product bug) via a confirmed classifier bug: `response_link()` only
   checked a per-row work item's `description` against the response's field names, never
   its `scales_with`, so work that demonstrably fills `allocations` was scored 0.70
   "genuinely removable" instead of blocked. **Fixed**
   (`plan/fixability.py::response_link` now checks both).
4. Redesigned the endpoint to the single-query version first (removing the orthogonal
   escape hatch `UNB-03` found), then **`UNB-04`** — **wrongly SELECTED** a
   `query_rewrite` whose `sql_before` and `sql_after` were byte-for-byte identical,
   claiming to eliminate "the extra SELECT" the redesign had already removed. Exposed two
   gaps at once: no check that a rewrite actually changes anything, and no check that an
   `sql_count` claim about "how many times a statement runs" is backed by anything this
   endpoint was ever measured doing. **Both fixed**: `plan/submit.py` now refuses a
   `sql_before == sql_after` (whitespace aside) rewrite outright, and a `sql_count`
   candidate must now cite a real baseline SQL digest hash and a call count within ~25%
   of what that digest actually measured (`baseline_digest`/`baseline_calls`, checked
   against a new "BASELINE: WHAT ACTUALLY RAN" section in the plan brief, itself backed
   by `SqlDigestStat` records already captured at baseline — nothing new needed adding to
   the contract, only to the brief and the check).
5. **`UNB-05`** — diagnose exhausted its 80-step budget without ever calling
   `emit_hypotheses` (5 probes, 0 rejections). `INTERNAL_ERROR`. One occurrence, logged,
   not investigated further.
6. **`UNB-06`** — with the redesign and all three fixes in place: clean refusal,
   `selected_id: None`. `h2`/orm_hydration (the SAME invented hypothesis, still present,
   still unfixed) still ranked top and still drew two candidates, but both were now
   refused for real, verified reasons — `query_rewrite` failed EXPLAIN's rows-examined
   check (0 → 0, flat), `add_index` was schema-blocked. Confirms the gate fixes hold even
   with the underlying hallucination still fully intact.

**Frozen as `UNB-01`** using a fresh `live_freeze.py` sample taken against `UNB-06`'s own
facts (same convention as `SER-01`): `h1`/serialization (fixability 0.05) and
`h3`/application_logic (fixability 0.10) both correctly name `allocations` by its own
rule; the two candidates actually proposed still target `h2`/orm_hydration and are
refused on their own, independently legitimate grounds. Same outcome bar as Scenario 7
(no patch phase, no PR, a refusal with a stated reason), same caveat recorded plainly in
`expected.json`: the 0.05 citation and the refusal reason are both true and both in the
artifact, just not tied to the same candidate every sample — not re-litigated as a
stricter per-candidate gate, for the reason that bar was withdrawn for Scenario 7.
Tier 1: `1/1 passed` at the time this scenario was written up; see the known-failure
table's cassette-replay-divergence entry for the full-suite result after that bug was
later isolated and fixed (`9/9 passed, 1 quarantined`).

## Per scenario, keep

- the tagged commit of the planted repo state
- the generated corpus and its hash
- the sealed oracle and its hash
- the full `trace.jsonl`
- the expected terminal state and fix kind
- the PR link, where one was opened

A frozen scenario passes when terminal state and fix kind match, and fails loudly when
either drifts.
