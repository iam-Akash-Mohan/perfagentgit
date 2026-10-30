# PerfAgent — Finish Line

Four sessions from here to a merged-quality PR. Run them in order. Each is one Claude
Code prompt; paste it, let it finish, run the verification command at the end, move on.

**State at the start of this document**

- Comprehension works and the map is complete (`per_row_work`, `response_shape`,
  `pagination`, `base_commit` all populated on mini-spring)
- Evidence, oracle, verify, measure, gate, report, ship, orchestrator all built and
  tested
- **Missing:** the plan phase, and any ability to reason about indexes empirically
- **Consequence:** diagnosis ranks by raw milliseconds, so every run attacks the
  unfixable serialization cost and the gate correctly rejects it. Three runs, three
  rejections, zero PRs.

**What these four sessions do**

| | | Outcome |
|---|---|---|
| F1 | Plan phase — rank by fixability, not size | It stops attacking the unwinnable problem |
| F2 | DB toolkit — EXPLAIN, index benefit, query equivalence | It can propose an index and *prove* it |
| F3 | Patch agent — candidates, DB fixes, migrations, Java standards | It writes the winning fix |
| F4 | End to end | **A PR** |

---

# F1 — The plan phase

Read `CLAUDE.md` first.

## The problem

`hypotheses.json` from the last full run ranked:

```
h1  serialization / response size   1400ms (78%)   ← chosen, attacked 3 times, lost 3 times
h2  ORM hydration + per-row work     200ms (11%)   ← fixable, never attempted
h3  network                          100ms  (5%)
```

h1 is large and **unfixable**: the response body is the API contract, and the frozen
oracle rejects any change to it. h2 is smaller and has a clean fix. Ranking by
`attributed_ms` alone cannot see that difference, so the agent picks the biggest number
and the gate throws it away.

The map now carries the facts needed to tell them apart —
`response_shape.unbounded_collections`, `per_row_work[].bounded`,
`tables[].existing_indexes` — and nothing reads them.

## Build

### 1 · `perfagent/plan/fixability.py` — deterministic, no model

```python
def assess_fixability(hypothesis: Hypothesis,
                      model: EndpointModel) -> FixabilityAssessment
```

A **fixed policy table**, not model judgement. Returns a probability that a
behaviour-preserving fix exists, plus the reason.

| Condition | Fixability | Reason |
|---|---|---|
| Hypothesis names a field in `response_shape.unbounded_collections` | **0.05** | response size is the API contract; the oracle rejects any change to it |
| `layer == "database_query"` and a `data_access` with `executes_per_row=True` matches | **0.90** | batchable: fetch join, `@EntityGraph`, or an `IN` query |
| A table on the path has `no_index_used` in the evidence and no `existing_index` covering the filtered columns | **0.95** | additive index, zero behaviour risk |
| `layer == "orm_hydration"` and `columns_read` is much smaller than `all_columns` | **0.80** | DTO projection returns the same values |
| `per_row_work` entry with `bounded=False` whose output feeds an unbounded response collection | **0.10** | the work produces the contract |
| `per_row_work` entry with `bounded=False` whose output does **not** feed the response | **0.70** | genuinely removable |
| `layer == "serialization"` with no unbounded collection involved | **0.30** | library-level, rarely beatable by hand |
| `layer == "network"` | **0.15** | out of scope for a code change |
| anything else | **0.40** | unknown; neither favoured nor excluded |

```python
class FixabilityAssessment(BaseModel):
    hypothesis_id: str
    fixability: float = Field(ge=0.0, le=1.0)
    reason: str
    blocked_by: str | None = None     # set when fixability < 0.2
    assessed_by: Literal["code"] = "code"
```

`assessed_by` is a `Literal` for the same reason `EquivalenceVerdict.decided_by` is:
this must be impossible to attribute to a model.

### 2 · Expected-value ranking

```python
expected_value_ms = attributed_ms * fixability
```

On the current artifact that gives:

```
h2  ORM / per-row queries    200ms × 0.90 = 180ms   ← ranked first
h1  serialization            1400ms × 0.05 =  70ms
h3  network                   100ms × 0.15 =  15ms
```

h2 wins despite being seven times smaller. **That single reordering is the session.**

### 3 · `perfagent/plan/agent.py` — the candidate generator

One agent loop over the top 2 hypotheses by expected value. Tools: the existing read
tools plus `get_schema`. Terminal tool `submit_plan`.

It proposes **2 to 4 candidates**, each:

```python
class CandidateFix(BaseModel):
    id: str
    targets_hypothesis: str
    approach: Literal["fetch_join", "entity_graph", "batch_query", "dto_projection",
                      "add_index", "query_rewrite", "remove_redundant_call",
                      "move_work_out_of_loop", "read_only_transaction", "other"]
    description: str
    mechanism_changed: str          # "SQL statements per request"
    predicted_effect: str           # "queries per request drop from 6 to 2"
    files_to_touch: list[str]       # becomes the patch agent's write allowlist
    sql_change: str | None = None   # the index or rewritten query, if any
    what_could_break: list[str]
    preserves_response_bytes: bool  # the model's claim; verified later by the oracle
```

System prompt requirements:

- The byte-identical constraint, stated plainly, with
  `response_shape.unbounded_collections` quoted into the prompt so it knows which fields
  it may not touch
- The fixability assessment **and its reason** for each hypothesis, so a 0.05 is
  explained rather than mysterious
- `tables[].existing_indexes`, so it never proposes an index that exists
- It must produce at least one candidate for the top-ranked hypothesis and at least one
  alternative approach
- It may not propose caching, transaction-boundary changes, API changes, or schema
  changes beyond an additive index

### 4 · Risk tiers — code, from a fixed table

```
T1  add_index · entity_graph · fetch_join · read_only_transaction · dto_projection
T2  batch_query · query_rewrite · remove_redundant_call · move_work_out_of_loop
T3  anything touching transaction boundaries or adding caching
T4  forbidden: schema beyond additive index, API contract, auth, tests
```

T4 is rejected at plan time. T3 sets `requires_flag=True` and is deprioritised.

### 5 · Selection

Pick the highest expected value whose risk tier is ≤ T2 and whose
`preserves_response_bytes` is true. Record the full ranked list in the artifact — the
alternatives that lost are part of the PR body later.

### 6 · `perfagent/plan/cli.py` + wire into `cli.py` and the orchestrator

```
perfagent plan [--run-dir DIR] [--provider X] [--model Y]
```

Reads `hypotheses.json` and `endpoint_model.json`, writes `plan.json`. Add `plan` to the
orchestrator between `diagnose` and `patch`, with a transition guard: **the patch phase
may not be entered without a plan artifact.**

Delete `plan_from_patch` in `gate/rules.py` — the reconstruction hack that existed
because this phase didn't.

## Acceptance

1. `perfagent plan` on the current artifacts ranks the ORM/per-row hypothesis **above**
   serialization, and prints the fixability reason for each.
2. The serialization hypothesis shows `blocked_by` naming `tags` as an unbounded
   response collection.
3. At least two candidates are produced, at least one of them `add_index` or a
   batching approach.
4. No candidate proposes an index that already exists on `customer` or `invoice`.
5. `perfagent patch` without a plan artifact is refused by the orchestrator guard.
6. Existing tests stay green.

```bash
perfagent plan && cat .perfagent/runs/*/plan.json | python -m json.tool | head -60
```

---

# F2 — The database toolkit

Read `CLAUDE.md` first. This is the capability that makes PerfAgent a database
optimizer rather than a Java refactorer.

## Build

`perfagent/tools/db.py`, four tools. All read the MySQL connection from config, the same
one `measure/sql.py` uses.

### `explain(sql: str, params: list[str] | None = None) -> str`

`EXPLAIN ANALYZE` the statement. Return the plan with actual rows and actual timings,
plus a short computed summary: access type per table, rows examined versus rows
returned, whether any index was used.

Reject anything that is not a `SELECT`. Statement timeout 10s.

### `list_indexes(table: str) -> str`

Name, columns in order, uniqueness, cardinality, and the index size. Column **order**
matters for composites and must be preserved in the output.

### `index_benefit(table: str, columns: list[str], sql: str, params: list[str] | None = None) -> str`

**The important one.** It measures instead of guessing:

1. Refuse if an existing index already covers these columns as a prefix — return which
   one, so the agent stops.
2. `EXPLAIN ANALYZE` the statement; record rows examined and actual time.
3. Create `perfagent_tmp_<hash>` on those columns. Time the creation.
4. `EXPLAIN ANALYZE` again.
5. **Drop the index in a `finally` block.** It must never survive the call, including on
   timeout or exception.
6. Return a before/after comparison: access type, rows examined, actual time, plus the
   index creation time and estimated size as the cost side of the trade.

```
index_benefit('invoice', ['customer_id', 'status'], '<the query>')

  BEFORE   type=ALL    rows_examined=1,120,858   actual=1,642ms
  AFTER    type=ref    rows_examined=42          actual=0.8ms
  COST     creation 11.2s · estimated size 38MB
  VERDICT  rows examined reduced 99.996%
```

Guardrails, each enforced in code:

- Additive indexes only. No `DROP`, no `ALTER` of existing indexes, no column changes.
- Only tables present in `endpoint_model.tables` — pass that list into the tool factory.
- Name prefix `perfagent_tmp_` always, so a leaked index is identifiable.
- A startup sweep that drops any orphaned `perfagent_tmp_*` from a crashed run.
- Hard timeout; on timeout, still drop.
- **Refuse entirely unless the connection is the configured measurement database.** This
  tool creates indexes on a live schema; it must be impossible to point at anything else.

### `query_equivalent(original_sql: str, candidate_sql: str, params: list[str] | None = None) -> str`

Run both, compare result sets exactly — same rows, same values, order-insensitive unless
the original has an `ORDER BY`. Report the first difference if any, plus row counts and
timings for both.

A rewrite that returns different rows is not a rewrite, and this is how the agent finds
that out before the oracle does.

### Migration generation — `perfagent/plan/migration.py`

```python
def write_index_migration(worktree: Path, table: str, columns: list[str],
                          rationale: str) -> Path
```

Finds the Flyway migration directory, computes the next version number, writes:

```sql
-- PerfAgent: <rationale>
-- Evidence: rows examined 1,120,858 → 42 (see run <id>)
CREATE INDEX idx_invoice_customer_id_status ON invoice (customer_id, status);
```

Conventional index naming. Idempotent — never overwrites an existing migration file.

## Wiring

- Plan agent gets `explain`, `list_indexes`, `index_benefit` — so a proposed index is
  measured before it is selected, not after.
- Patch agent gets all four.
- Diagnosis gets `explain` and `list_indexes` only. **Not** `index_benefit`: diagnosis
  explains what is, it does not try fixes.

## Acceptance

1. `explain` on the `sumOpenAmountByCustomer` query returns a real plan with actual rows.
2. `index_benefit('invoice', ['customer_id','status'], ...)` shows a large drop in rows
   examined, and the index is gone afterwards — verify with `SHOW INDEX FROM invoice`.
3. `index_benefit` on a column that is already indexed refuses and names the index.
4. `index_benefit` on a table not in the endpoint model is refused.
5. Killing the process mid-call leaves a `perfagent_tmp_*` index; the next call sweeps it.
6. `query_equivalent` detects a deliberately wrong rewrite and names the difference.
7. A generated migration file is valid SQL and Flyway applies it cleanly.

---

# F3 — The patch agent

Read `CLAUDE.md` first.

## What changes

The patch agent currently reads the top hypothesis and attempts one fix in one worktree.
It needs to read a **plan**, implement a **named candidate**, use the DB tools, follow
Java standards, and know whether its own change helped.

### 1 · `standards/java.md` — new file, editable without touching Python

```markdown
# Java standards for generated patches

## Required
- Constructor injection only. No field @Autowired.
- @Transactional(readOnly = true) on read-only service methods.
- New repository query methods carry a Javadoc line stating what they return.
- Projections are interfaces or records, not entity subclasses.
- New query methods are named for what they return, not how they work.

## Forbidden
- No new static mutable state.
- No `new` on a Spring-managed type.
- No raw JDBC outside the repository layer.
- No changes to public method signatures on controllers.

## Style
- Prefer Optional<T> over null returns.
- Streams where they read more clearly than an index loop; not otherwise.
- Keep imports explicit; no wildcard imports.
```

Loaded into the patch prompt at runtime. The mechanically checkable rules — field
`@Autowired`, wildcard imports, `new` on a `@Service`/`@Component` type, controller
signature changes — become a **lint pass over the diff** that fails the phase. The rest
are prompt guidance. Say in `docs/` which rules are enforced and which are advisory.

### 2 · Input is a candidate, not a hypothesis

```python
def implement(*, candidate: CandidateFix, model: EndpointModel,
              evidence: EvidenceBundle, ...) -> Patch
```

The prompt carries: the candidate's approach and predicted effect, its
`files_to_touch` as the write allowlist, the Java standards, the byte-identical
constraint with the unbounded collections named, and the fixability reason.

### 3 · SQL and migrations are in scope

Extend the write allowlist to include `src/main/resources/db/migration/**`, **for
additive index migrations only** — enforced by parsing the generated SQL, not by
trusting the model. Any statement that is not `CREATE INDEX` is refused.

### 4 · Measure inside the loop

After the build and tests pass, before `emit_patch`, run a **quick** benchmark — 20
requests, no interleaving — and give the agent its own delta:

```
your change: p50 1022ms → heuristic 640ms (-37%), SQL/request 6.0 → 2.0
```

If latency got worse, say so and let it reconsider rather than emitting a patch it
already knows is bad. This is a heuristic signal, not a verdict — the real measurement
is still P14's interleaved A/B, and the prompt must say so.

### 5 · Fix the vacuous test guardrail

`mvn test` runs zero tests in mini-spring — surefire's includes don't match
`CustomerSearchIT`. So "existing tests pass" currently proves nothing and the
test-count guardrail can never fire. Fix the surefire configuration in mini-spring so
the integration test actually runs, and make the guardrail real.

## Acceptance

1. `perfagent patch` implements the plan's selected candidate, not the top hypothesis.
2. An attempted write outside `files_to_touch` is refused and traced.
3. A migration containing anything other than `CREATE INDEX` is refused.
4. A patch with a field `@Autowired` fails the standards lint, and the message names the
   rule.
5. `mvn test` runs a non-zero number of tests, and deleting one fails the guardrail.
6. The in-loop benchmark reports a delta, and a regression is visible to the agent
   before it emits.

---

# F4 — End to end, and the PR

Read `CLAUDE.md` first. No new capability — wiring, verification, and the run.

## Build

1. **Orchestrator:** insert `plan` between `diagnose` and `patch`. Guard: no patch
   without a plan. Guard: no plan without hypotheses.

2. **Retry on rejection.** If verification fails, return to the patch phase with the
   oracle's diff as new context, up to 2 attempts. If it fails again, move to the next
   candidate in the plan. If candidates are exhausted, terminate honestly as
   `BEHAVIOUR_NOT_PRESERVED`.

3. **Retry on no improvement.** If the gate fails only on
   `improvement_threshold`, try the next candidate rather than terminating — the plan
   has alternatives and they cost one build each.

4. **`p_value_paired`** as a real field on `PerfReport`. The gate reads Wilcoxon; today
   it is passed as a loose keyword argument.

5. **PR body** rendered from artifacts, containing: endpoint, bottleneck with evidence,
   **why the alternatives were rejected** (the fixability reasons — this is what makes
   the PR reviewable), the fix, the EXPLAIN plan before and after for any index, the
   oracle verdict with request counts, before/after latency with methodology and
   p-value, the mechanism confirmation, and files changed. Every number from the
   evidence ledger.

6. **Cost reporting** — set the real token prices in `pricing.py` so `perfagent trace`
   stops printing `$0.00`.

## Run it

```bash
cd ../mini-spring && docker compose up -d && mvn -q -DskipTests package
java -jar target/*.jar --spring.profiles.active=perfagent &

cd ../perfagent
perfagent run --repo ../mini-spring --endpoint "POST /api/customers/search"
```

## Acceptance

1. The full pipeline runs: understand → workload → oracle → baseline → diagnose →
   **plan** → patch → verify → measure → gate → ship.
2. **A PR is opened.** `ship.py` has never executed; this is the first time.
3. Verification passed on both the benchmark and held-back sets.
4. The improvement is statistically significant on the paired test.
5. The mechanism check confirms the predicted change — if the plan predicted fewer
   queries, SQL per request actually dropped.
6. The PR body contains no number without an artifact behind it.
7. `perfagent trace <run_id>` renders the whole run including the plan phase, with real
   dollar costs.

## Then — the second run, which is the other half of the pitch

```bash
perfagent run --repo ../mini-spring --endpoint "GET /api/customers/{id}"
```

An endpoint with no real bottleneck should terminate `NO_BOTTLENECK_FOUND` and open
nothing.

**Those two traces side by side are the pitch.** One where it shipped with proof, one
where it declined with reasons. A tool that always produces a patch is a tool nobody
can trust; showing both is what makes the first one credible.

---

# After this

The product works. What remains is knowing how *well* it works:

- **Eval suite** — ten scenarios with a ground-truth answer key, three runs each. Turns
  "it worked" into "8 of 10, both negative controls clean, $0.40 a run". This is what a
  sharp manager will ask for.
- **Candidate search** — implement several candidates, measure each, keep the winner,
  try stacking. Turns "an optimization" into "the best one I could find".
- **Code index** — tree-sitter symbols and call graph. Needed before pointing this at a
  3,000-file Growfin service.
- **Repo profile and feasibility gate** — infer how to build and run a repo, verify by
  executing. What makes onboarding a new service fifteen automated minutes instead of a
  manual afternoon.
