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
