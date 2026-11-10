# Shipping a schema change, and what happens when the repo cannot take one

## The limitation, stated plainly

**A target repository with no migration tool cannot receive a schema change
from this agent.** That is a real limitation, not a bug, and the right
behaviour is to say so and stop — not to skip the index quietly.

`index_benefit` can prove an index is worth having: on mini-spring it measured
rows examined falling 2,645 → 571 on the per-customer aggregate, 23× faster,
for 37.6 MB and 1.5 s of build time. None of that matters if there is nowhere
to put the `CREATE INDEX`. An index applied by hand to a database, with nothing
in the diff, produces the exact failure this product exists to prevent: a PR
whose measured improvement cannot be reproduced from the code it contains.

So `senapathy/plan/migration.py::write_index_migration` refuses when it finds
no Flyway directory and no Flyway in the build file, and names what it looked
for. The patch agent surfaces that refusal rather than working around it.

**F3's patch agent reports this as a blocker.** `add_migration` returns the
refusal to the agent as a tool result; if the candidate's whole value was the
index, the run should be abandoned honestly rather than emitting a patch that
claims an index nobody will apply.

## What mini-spring looked like before

Its schema was `docker/mysql/init/01-schema.sql`, mounted into the MySQL
container's entrypoint directory. That file runs **once**, on the first boot of
an empty volume. It is not versioned, not applied to an existing database, and
not something a diff can change. Three copies of the schema existed in
practice: the init script, `seed/seed.py`'s own `CREATE TABLE`, and whatever
the running database actually had.

## What it looks like now

* `src/main/resources/db/migration/V1__initial_schema.sql` is the single
  source of truth.
* Flyway (`flyway-core` + `flyway-mysql`, versions from the Spring Boot parent)
  runs on application startup with `baseline-on-migrate: true` and
  `baseline-version: 0`, so it adopts the already-seeded 1.2M-row volume
  instead of refusing to start against it, and applies V1 normally to an empty
  schema.
* The docker entrypoint mount is **gone**. `MYSQL_DATABASE` still creates the
  empty schema; Flyway creates the tables.
* `seed/seed.py` reads V1 from disk rather than carrying its own copy, so
  seeding a fresh volume still works without booting the app first.
* `flyway-maven-plugin` is configured, so `mvn flyway:info` and
  `mvn flyway:migrate` work without starting the application — which is how a
  reviewer checks a generated migration.

## The hazard this introduces, which is NOT yet solved

Blue and green share one database.

An index is a property of the database, not of a worktree. The moment green's
boot applies a migration that adds one, **blue is running against an indexed
table too** — so an interleaved A/B measurement of an index change compares two
arms that both have the index, and reports no improvement for a change that is
genuinely worth 23×.

### What the harness does about it now

It refuses to measure, and says so in its own terminal state.

Before the measure phase starts, `senapathy/measure/isolation.py` asks two
questions. Does this patch touch the schema — because the plan chose
`add_index`, or because the diff contains a migration, or because the patch
reports one among its changed files? And do the two arms' resolved JDBC URLs
name the same vendor, host, port and schema, read from each worktree's own
`application-*.yml` rather than from the harness's idea of them?

If both are true the run stops with **`MEASUREMENT_NOT_ISOLATED`**, which is a
distinct terminal state on purpose. `NO_SIGNIFICANT_IMPROVEMENT` means "we
measured it and it did not help". This means "we could not find out". A report
in which those two look alike teaches the reader the wrong lesson, and the
lesson is expensive: the first says the idea was wrong, the second says the
instrument was.

The refusal names the shared schema and says in as many words that it is not a
verdict on the patch. A Java-only patch is never affected — `touches_schema`
returns false and the guard does nothing.

### What per-arm isolation would actually require

Four things, in rough order of effort:

1. **A second schema.** `mini_blue` and `mini_green`, both seeded from the same
   deterministic script, with `CHECKSUM TABLE ... EXTENDED` compared before any
   measurement — if the two datasets are not byte-identical the comparison is
   measuring the data, not the patch.
2. **Per-arm datasource configuration.** Each arm's JDBC URL would have to be
   injected at start (`-Dspring.datasource.url=...`) rather than read from the
   shared profile. `arms.py::arm_argv` is the single place that would change,
   and `assert_flags_identical` would need to learn that the datasource URL is
   now an expected difference rather than a disqualifying one.
3. **Migrations applied per arm.** Flyway on boot already does the right thing
   here — each arm would migrate its own schema — but the blue arm must NOT
   apply green's migration, which means blue runs from the base commit's
   migration directory and the isolation check becomes a check that it did.
4. **Double the buffer pool, or halve it per arm.** Two schemas of 1.2M rows
   share one InnoDB buffer pool. Whichever arm runs second benefits from the
   first arm's cache warming unless the pool is sized to hold both, or flushed
   between arms — and a flush between arms reintroduces the cold-start bias
   that interleaving exists to remove.

Point 4 is the one that makes this more than a configuration change, and it is
why the guard ships before the fix: refusing to measure is correct today, and
a wrong number would have been worse than no number.

## The schema-candidate gate path (built)

An index candidate is now shippable without per-arm schema isolation — the
gap this document used to describe as unbuilt. The insight that makes it
possible: an index is proven by the database, not by a stopwatch. Plan shape
and rows examined are deterministic and repeatable on ANY schema state,
including one blue and green share; only an endpoint-level wall-clock claim
needs isolated arms, and this path simply never makes that claim.

**Proving the index** (`senapathy/audit/candidates.py::propose_index_candidate`,
`tools/db.py::index_benefit_measured`). The SAME create/EXPLAIN/drop lifecycle
this document originally described now ALSO runs an interleaved
direct-statement benchmark while the candidate index is alive — `SELECT ...
IGNORE INDEX (senapathy_tmp_x)` against the plain statement, A/B/A/B, never
sequential, so neither side is credited with the other's buffer-pool warming
— before dropping the index in the same `finally` as before. The resulting
`IndexCandidate` (`contracts/audit.py`) carries the EXPLAIN plan before and
after, rows examined before and after, the column order and why, the
interleaved benchmark's own paired samples and Wilcoxon p-value, the
candidate's size, and the table's write volume
(`table_io_waits_summary_by_table`) — everything a reviewer needs to judge an
index's cost, not just its benefit.

**Citing it at plan time** (`plan/risk.py::audit_proof_for`,
`plan/submit.py`). An `add_index` candidate names `audit_digest`, the digest
hash of the MISSING_INDEX statement in `sql_audit.json` it fixes.
`audit_proof_for` checks that digest's own measured `index_candidate` names
the SAME table and columns as the candidate's `sql_change`, and on a match
copies that measurement onto `CandidateFix.schema_proof` — code-assigned,
never in the model's tool schema, same discipline as `risk_tier`. A
candidate with `schema_proof` set skips `schema_change_blocked`'s
unconditional refusal entirely; one without it (a hand-written index with no
citation) is blocked exactly as before.

**The gate** (`gate.rules.evaluate_schema_gates`). `schema_proof`'s presence
is the switch every downstream phase reads. MEASURE skips the endpoint A/B
entirely for this candidate — there is no "without the index" state left to
re-measure live once green's migration has applied it to the shared schema —
and instead confirms the shipped migration creates the exact index the proof
measured, writing `schema_measurement.json`. GATE evaluates six checks instead
of the usual six: `behaviour_preserved` and `no_forbidden_change` reused
unchanged (both are schema-agnostic facts), plus `plan_improved` (EXPLAIN's
access type ranks strictly better), `scanning_reduced` (rows examined fell by
at least 50%), `statement_faster` (the interleaved benchmark's own Wilcoxon
p < 0.05), and `index_not_redundant` (no existing index already covers these
columns). `schema_terminal_state_for` mirrors `terminal_state_for`'s own
ordering over this set. The rendered report (`render_schema_report`) states
outright that the claim is statement-level and that no endpoint p95 figure is
asserted — and because that template simply never names an endpoint-level
slot, the evidence ledger's existing missing-slot refusal is what makes that
durable rather than a convention someone could forget to keep following.

`MEASUREMENT_NOT_ISOLATED` is narrower now, not gone: it means a schema
change someone tried to prove by an endpoint-level A/B instead of by citing
the audit's own statement-level measurement. A candidate with a matching
`audit_digest` never reaches it.

## What per-arm isolation would additionally buy (report only — not built)

The schema-candidate gate path above proves an index at the statement level
and deliberately asserts nothing about the endpoint. Per-arm isolation is
the other way to close that gap: build it, and a schema-touching candidate
gets the SAME `IMPROVEMENT_THRESHOLD`/`STATISTICAL_SIGNIFICANCE` endpoint
p95 claim every other candidate already gets, instead of the narrower
statement-level one. Whether that is worth building is a cost question, not
a capability question — the capability already exists in `evaluate_schema_
gates`; the question is whether a SECOND kind of proof is worth the
infrastructure.

**What it would buy.** An endpoint p95 number for a schema candidate, in the
SAME units and SAME gate as a Java-side fix — directly comparable across a
PR history instead of two different kinds of evidence. It would also prove
something the statement-level path structurally cannot: that the endpoint's
OTHER costs (serialization, per-row work, the rest of the call chain) do not
eat the index's own gain. A 23x reduction in one statement's rows examined
that sits behind 200ms of unrelated per-row allocation is real, but the
endpoint might barely move — `statement_faster` would still pass, correctly,
because its claim was always about the statement, never the endpoint; only
an endpoint-level measurement could show that composition.

**What it would cost**, beyond the four requirements already listed above
(a second schema, per-arm datasource configuration, per-arm migrations, and
buffer-pool sizing):

- **Two schemas, one MySQL instance.** The cheaper option, and the one the
  four requirements above assume — `mini_blue` and `mini_green` in the same
  `mysqld`. It still shares ONE buffer pool between them. Point 4 above is
  this cost stated plainly: size the pool to hold both datasets twice over
  (double the memory a single schema needed, since each arm's own working
  set now competes for the same cache), or flush between arms and reintroduce
  the exact cold-start bias interleaving exists to remove. Measuring 100 MB
  of working set becomes measuring 200 MB competing for whatever RAM the
  pool is given — on a laptop or a shared CI box, that is the difference
  between "fits" and "thrashes."
- **Two MySQL instances.** Genuinely isolated — no shared buffer pool, no
  cross-arm cache interference, the clean answer to point 4. The cost moves
  from "shared and contended" to "doubled and idle half the time": two
  `mysqld` processes, each with its own buffer pool sized for one arm (so
  the MEMORY cost is roughly the same total as the single-instance option,
  just partitioned instead of shared-and-doubled) — but also two sets of
  connections, two ports to provision, and critically, double the SEED
  TIME: `seed/seed.py`'s own deterministic population run happens twice,
  serially or in parallel competing for the same disk I/O, on every run that
  touches schema. On mini-spring's own 1.2M-row seed, that is minutes added
  per run, every run, whether or not the candidate being measured is a
  schema change at all — unless seeding is itself made incremental or
  cached, which is a THIRD piece of infrastructure this estimate does not
  include.

Both options still need points 2 and 3 from the existing list (per-arm
datasource configuration, per-arm migrations) regardless of one instance or
two — those costs are identical either way and are not repeated here.

**The decision this document cannot make:** whether an endpoint-level p95
claim for schema candidates is worth EITHER cost, given that the
statement-level path already ships the common case (an additive index) with
neither. The schema-candidate gate path was the cheap, high-value 80% —
whether the remaining 20% (endpoint composition effects, a uniform p95
figure across every candidate type) justifies doubled memory or doubled seed
time is a call to make with real numbers from a real CI budget, not
instinct — which is what this section is for.

## Whether the audit should ever seed data (report only — not built)

`index_benefit_measured`'s EXPLAIN ANALYZE is only as informative as the data
it runs against. A table with ten rows gives every access type the same
`rows_examined`, index or no index — the measurement would be real but
meaningless, which is a worse failure shape than refusing to measure at all,
because it looks exactly as credible as a measurement against a realistic
table.

**The case for seeding:** a fresh target repository's database — onboarded for
the first time, or a CI container spun up empty — would otherwise make the
audit phase (and `index_benefit` generally) silently useless on exactly the
runs where a human has the least other evidence to fall back on. A
deterministic seed (the same discipline `seed/seed.py` already uses: a fixed
row count, a fixed distribution, no `RAND()`) would make the audit phase work
the first time a repo is onboarded, not only on a database someone has
already been exercising for a while.

**The case against:** writing to the target database is a mutation this
product has never performed, anywhere, for any phase. Every existing read
against MySQL — `explain`, `list_indexes`, `index_benefit`'s own temporary
index, `query_equivalent`'s `SELECT`-only guardrail — either reads, or writes
something additive and self-cleaning (`senapathy_tmp_*`, dropped in
`finally`). Seeding is categorically different: it writes rows a human did not
put there, into a database this product does not own, and there is no
`finally` that un-seeds a row once something else — a migration, a foreign
key, a later seed run — has come to depend on it existing. It would also
produce a second, silent source of truth about "how much data is in this
table," alongside whatever the real application put there, with no recorded
link between the two; `CLAUDE.md`'s invariant that the oracle is frozen and
unmodified by any agent code path is a close cousin of the same worry applied
to the application's recordings, and seed data sitting permanently in a
table a later run's oracle capture also reads is the same shape of risk.

Left unbuilt. If it is ever built, the seed should be sized to the SMALLEST
row count that makes EXPLAIN ANALYZE's numbers meaningful (not maximised for
realism), should run in a check-and-skip mode that refuses when the table
already holds real rows (never adding to an already-populated table), and
should record what it added as its own artifact — the same "no number without
an artifact" discipline this project applies to every other measurement — so
a later run can tell seeded rows from real ones rather than guessing from row
count alone.
