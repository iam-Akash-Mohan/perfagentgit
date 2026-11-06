# PerfAgent — PRD & Technical Design

**An autonomous agent that finds, fixes, verifies and ships a real performance optimization for one API endpoint.**

Version 1.0 · Owner: Ak (Growfin) · Status: pre-implementation

---

## 0. How to read this document

This document is the single source of truth for the problem and the solution. It is written to be read by a human engineer **or** by a Claude agent that has no prior context. After reading it, the reader should be able to answer:

- What problem does this product solve, and for whom?
- What exactly does the system do, step by step, from input to output?
- Which parts are AI and which parts are ordinary code, and *why* that line was drawn where it was?
- What are the invariants that must never be violated?
- What is in the first week's build and what is deliberately deferred?

The phase-by-phase implementation plan is a **separate document**. This one defines *what* and *why*. That one defines *when* and *in what order*.

---

## 1. Problem statement

### 1.1 The engineering problem

In any mature backend, a handful of endpoints are slow. Fixing one is a well-understood but expensive piece of work:

1. Read the code and trace the execution path across controllers, services, repositories, helpers.
2. Reproduce the slowness with enough data to make it visible.
3. Instrument, profile, read SQL logs, read query plans.
4. Form a hypothesis about the real bottleneck — not the first plausible one.
5. Implement a fix.
6. Prove the endpoint still behaves identically.
7. Prove the fix actually made it faster, and not by accident.
8. Write it up so a reviewer can trust it.

Steps 1–4 take a senior engineer half a day to two days per endpoint. Steps 6–8 are the ones most often skipped, which is why performance PRs are hard to review and sometimes ship regressions.

The work is **highly repetitive in shape** and **highly variable in content**. That is exactly the profile of work an agent can do.

### 1.2 Why existing tools don't solve it

| Tool class | What it does | What it doesn't do |
|---|---|---|
| APM (Datadog, New Relic) | Tells you *which* endpoint is slow | Doesn't tell you why, in code terms; doesn't fix it |
| Static analysers / linters | Flags known anti-patterns from a fixed rule list | No runtime evidence; huge false-positive rate; can't find novel problems |
| LLM code review | Suggests plausible optimizations | No measurement, no verification, no proof; suggestions only |
| Coding agents (Claude Code etc.) | Can genuinely do the work | Requires a human in the loop at every step; no enforced baseline, oracle or measurement, so its claims are unverified |

The gap is not intelligence. Modern models can reason about a slow endpoint perfectly well. **The gap is trust.** Nothing today produces a performance change with *evidence attached* — a recorded baseline, a proven-unchanged behaviour, and a measured, statistically meaningful improvement.

### 1.3 The problem this product actually solves

> Turning "this endpoint is slow" into "here is a merged PR that made it 73% faster, and here is the proof that nothing else changed" — without a human doing the investigation.

The deliverable is not a patch. **The deliverable is a patch plus the evidence that makes it mergeable.**

---

## 2. Product definition

### 2.1 Input

One command:

```bash
perfagent run --repo ./crm-service --endpoint "POST /api/customers/search"
```

Everything else — the execution path, the data volume needed, the workload, the bottleneck, the fix — is derived by the system.

### 2.1a Supported stack

**PerfAgent supports JVM web services on MySQL.** This is a real and deliberate limit, not an oversight, and it is **detected at onboarding** rather than discovered as a crash at baseline.

| Dimension | Supported | Detected from |
|---|---|---|
| Build tool | Maven, Gradle (wrapper preferred over the binary on PATH) | `pom.xml`, `build.gradle[.kts]`, `mvnw`, `gradlew` |
| Entrypoint | Spring Boot (`@SpringBootApplication` + `main`) | the annotation, not the filename |
| Source layout | `src/main/{java,kotlin,scala}`, `app/src/main/java` | the directory that exists on disk |
| Database | **MySQL / MariaDB only** | the `jdbc:` URL in the repo's own `application*.yml`/`.properties` |
| Migrations | Flyway (Liquibase detected, not yet applied) | the build file *and* a migration directory |

The database limit is the load-bearing one. `measure/sql.py` attributes latency to individual statements by reading MySQL's `performance_schema` digest tables; there is no portable equivalent, so a Postgres or H2 datasource is **detected and refused at phase P-1** with that sentence, before a single token is spent. Everything the onboarding phase concludes is written to a `RepoProfile` artifact in which **every field cites the file it was read from** — see `perfagent/onboard.py` and `tests/test_onboard.py`, which assert what each detector refuses.

A repository that cannot be onboarded fails with `ENVIRONMENT_INFEASIBLE` naming the question it could not answer, where it looked, and what the operator should do.

### 2.2 Output

One of a small set of **honest terminal states**, each with a full report and trace:

| Terminal state | Meaning | Artefact |
|---|---|---|
| `PR_OPENED` | Verified, measured optimization shipped | PR + report + trace |
| `READY_TO_SHIP` | All six gates passed; nothing has been pushed yet (a dry run, or any passing decision before `open_pull_request` runs) | Report + trace |
| `NO_BOTTLENECK_FOUND` | Endpoint is already efficient | Report explaining why |
| `BOTTLENECK_NOT_FIXABLE` | Real bottleneck found, but no safe fix within risk budget | Diagnosis report — still valuable |
| `BEHAVIOUR_NOT_PRESERVED` | Patch worked but changed behaviour; rejected | Report + the diff that failed |
| `NO_SIGNIFICANT_IMPROVEMENT` | Patch was correct but didn't help | Report + measurements |
| `MEASUREMENT_NOT_ISOLATED` | A schema-changing candidate can't be A/B measured because both arms share one database | Report + the isolation reason |
| `ENVIRONMENT_INFEASIBLE` | Cannot build/run/measure this repo | Blocker list |
| `BUDGET_EXHAUSTED` | Ran out of steps/tokens/time | Partial findings |
| `INTERNAL_ERROR` | Bug in the harness | Trace |

**Only one of ten outcomes is a PR.** This is a core product decision, not a limitation. A system that always produces a patch is a system nobody can trust. `NO_BOTTLENECK_FOUND` is a correct, valuable answer.

### 2.3 Console output of a successful run

```
$ perfagent run --repo ./crm-service --endpoint "POST /api/customers/search"

P-1 onboarding      profile cached · stack java-spring-jpa-maven ....... ok
P0  provisioning    mysql, app-blue, app-green, wiremock .............. ok
P1  comprehending   18 tool calls · 7 classes on path ................. ok
P2  workload        seeded 50k customers / 1.2M invoices .............. ok
P3  oracle sealed   200 requests · sha256 4f9a1c ...................... ok
P4  baseline        p50 1980ms  p95 2410ms · 847 SQL queries .......... ok
P5  diagnosing      11 probes · N+1 on Customer.invoices (78%) ........ ok
P6  planning        3 candidates · selected @EntityGraph fetch join ... ok
P7  implementing    2 build iterations · 1 file · +6/-2 lines ......... ok
P8  verifying       200/200 responses identical · DB state identical .. ok
P9  measuring       p95 2410ms → 640ms  (-73.4%, p<0.001) ............. ok
P10 gating          all gates passed ................................. ok
P11 shipping        PR #42 opened

run 01JBQ8...   trace: .perfagent/runs/01JBQ8/trace.jsonl
                report: .perfagent/runs/01JBQ8/report.html
```

### 2.4 Users

- **Primary:** a backend engineer who owns a service and knows an endpoint is slow.
- **Secondary:** a tech lead who wants performance debt paid down continuously without allocating people to it.
- **Not a user:** anyone who wants an unattended bot merging its own PRs. A human reviews and merges. Always.

### 2.5 Non-goals

- Not an APM. It does not find *which* endpoint is slow — you tell it.
- Not a general coding agent. One endpoint, one optimization, per run.
- Not a merge bot. It opens PRs; humans merge them.
- Not architectural refactoring. It won't introduce a cache layer, split a service, or change your data model.
- Not multi-endpoint or whole-service optimization in v1.

---

## 3. Design principles

These five decisions determine everything else. Every other choice in this document follows from them.

### 3.1 A deterministic state machine with agentic segments inside it — not "an agent"

The tempting design is one Claude loop with a shell and a prompt saying "optimize this endpoint." That is essentially Claude Code, and it works *because a human is watching and can interrupt.* Remove the human and it fails in specific, predictable ways:

- It patches before it baselines, then "measures improvement" against nothing.
- It edits a test to make the build green.
- It reports a 60% gain it never measured.
- Run 3 looks nothing like run 1 — no reproducibility.
- A failure at step 180 costs the entire run.
- It cannot be evaluated, because there are no intermediate outputs to grade.

All six are fixed by the same move: **cut the run into phases with typed artefacts at the boundaries, and let ordinary Python own the transitions.** The phase order is knowable in advance (baseline → diagnose → patch → verify → measure → ship), so there is nothing there for a model to decide.

### 3.2 The agency test

For every piece of the system, ask:

> **Does the number of steps depend on information we don't have until runtime?**

- "Read the controller file" → No. Fixed. That is a function call.
- "Understand the full execution path" → Could be 3 files or 40, unknowable in advance. That is a loop.
- "Run 200 requests and compute p95" → Fixed. Function call.
- "Figure out *why* p95 is 2.4s" → Unknown number of probes. Loop.

By this test the system needs exactly **four** agentic loops: comprehension, diagnosis, implementation, and diff-triage-on-verification-failure. Everything else — provisioning, seeding, measurement, comparison, gating, git, PR, report — is deterministic code.

**Agency is a cost, not a virtue.** Every loop adds nondeterminism, latency, token spend and a new failure mode. Spend it only where the alternative doesn't exist.

### 3.3 The evidence ledger

**Claude never asserts a number.**

Not "is discouraged from" — the report renderer *cannot* emit a claim that isn't backed by an artefact reference produced by deterministic code. Every latency figure, query count, file path and line number in the PR body is templated from a recorded measurement. Claude writes only the connective narrative.

This single rule eliminates the most common and most damaging failure of AI engineering tools: a beautifully written, entirely fictional summary.

### 3.4 The frozen oracle

The definition of "behaviour unchanged" — the request corpus, the recorded responses, the canonicalisation rules, the tolerance policy, the DB side-effect assertions — is constructed **against the unmodified code, before the patching agent exists**, then content-hashed. The patching agent cannot read it and cannot modify it.

An agent that can edit its own grader is not being verified. (§8 explains the oracle in full.)

### 3.5 No general shell

Every capability is a **named, schema-validated tool with an allowlist.** There is no `bash` tool.

Three reasons:

1. A shell is an unbounded action space. You cannot enumerate what it might do, so you cannot guard it — you can only hope.
2. Observability collapses. Your audit log becomes a list of `sh -c "..."` strings instead of structured, queryable events.
3. Tool design *is* context design. A tool that returns exactly the right 200 tokens is worth more than a shell that returns 20,000.

**One narrow exception:** a constrained `run_probe` tool during diagnosis only, network-isolated, read-only source mount, timeout-capped. Diagnosis genuinely needs open-ended investigation; nothing else does.

Capabilities are scoped **per phase**, not granted globally. The comprehension agent has read tools only. The implementation agent gets write access, and only to an allowlist derived from its own approved plan.

---

## 4. Architecture

### 4.1 Three layers

Generality is achieved by separating what is universal from what is stack-specific from what is repo-specific.

```
┌─────────────────────────────────────────────────────────────┐
│  LAYER 1 — CORE  (universal, knows nothing about Java)      │
│                                                             │
│  Orchestrator FSM · Agent loop · Tool dispatcher            │
│  Context manager · Artefact store · Evidence ledger         │
│  Frozen oracle · A/B benchmark harness · Gate rules         │
│  Trace/event log · Eval harness · Git & PR workflow         │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│  LAYER 2 — STACK PACK  (one per ecosystem, swappable)      │
│                                                             │
│  Code index (tree-sitter grammar + framework rules)         │
│  Route extraction · Build/test commands                     │
│  SQL capture instrumentation · Profiler hook                │
│                                                             │
│  v1: java-spring-jpa-maven                                  │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│  LAYER 3 — REPO PROFILE  (discovered per codebase)         │
│                                                             │
│  How THIS repo builds, runs, connects to its DB,            │
│  where migrations live, what externals it calls.            │
│  Inferred by Claude, then VERIFIED BY EXECUTION.            │
└─────────────────────────────────────────────────────────────┘
```

**Layer 1 is most of the system and all of the interesting engineering.** It does not know what Java is.

**Layer 2** is bounded work per ecosystem — a Gradle variant is a couple of hours; a genuinely different stack (Node/Prisma, Python/Django) is a few hundred lines because everything above it is untouched.

```python
class StackPack(Protocol):
    def detect(repo) -> Confidence
    def build_index(repo) -> CodeIndex
    def extract_routes(index) -> list[Route]
    def build_cmd(), test_cmd(), run_cmd() -> Command
    def sql_capture() -> Instrumentation
    def profiler() -> Instrumentation
```

**Layer 3** is the answer to "works on any codebase." Claude reads the README, `docker-compose.yml`, `application.yml`, CI config, `Dockerfile` and `pom.xml`, and infers how to operate the repo. Then **deterministic code verifies every inferred field by actually doing it** — build it, boot it, health-check it, run one test, connect to the DB.

```yaml
# .perfagent/profile.yml — generated, then verified, then committed
stack: java-spring-jpa-maven
build:  mvn -q -DskipTests package
test:   mvn -q test
run:    java -jar target/app.jar --spring.profiles.active=perfagent
db:
  engine: mysql
  version: "8.0"
  schema_source: flyway:src/main/resources/db/migration
externals:
  - { name: billing-api, base_url_prop: billing.base-url }
health: /actuator/health
verified_at: 2026-08-21T09:14Z    # by execution, not by assertion
invalidate_on: [pom.xml, docker-compose.yml, application.yml]
```

**Build the seam early, build only one implementation behind it.** The protocol costs an afternoon. Premature generality kills the project.

### 4.2 Environment topology

```
                    ┌──────────────────┐
                    │  mysql-seeded    │
                    │  :<dataset-hash> │   perf_schema ON
                    │                  │   slow_query_log, long_query_time=0
                    └────────┬─────────┘   innodb_buffer_pool_size PINNED
                             │
              ┌──────────────┴──────────────┐
              │                             │
     ┌────────▼────────┐          ┌─────────▼───────┐
     │   app-blue      │          │   app-green     │
     │   :8081         │          │   :8082         │
     │   worktree blue │          │   worktree green│
     │   ORIGINAL      │          │   PATCHED       │
     └────────┬────────┘          └────────┬────────┘
              │      IDENTICAL JVM FLAGS   │
              └──────────────┬─────────────┘
                             │
                    ┌────────▼─────────┐
                    │    wiremock      │  fixed-latency stubs
                    └──────────────────┘
```

One git clone, **two worktrees**: `blue/` pinned at the base commit, `green/` where the agent works.

Both apps run simultaneously so P9 can **interleave** measurement (blue, green, blue, green…). Sequential measurement silently attributes thermal throttling and background load to the patch.

Measurement-critical configuration (these are about *trustworthy numbers*, not about running the app):

- Fixed heap: `-Xms == -Xmx`, identical on both arms.
- Fixed CPU and memory limits per container, so host load can't masquerade as a regression.
- Byte-identical JVM flags on both arms. If green gets an extra flag, the entire result is void.
- Pinned `innodb_buffer_pool_size` — otherwise it sizes off the host and results aren't portable.
- Mandatory warmup, discarded. The JIT needs hundreds of iterations; a "40% improvement" that is really C2 compilation is the classic self-inflicted wound.

### 4.3 MySQL specifics

The target stack uses MySQL 8, which is *better* for this product than Postgres in two ways and worse in one.

**Better — SQL evidence is native.** `performance_schema.events_statements_summary_by_digest` already provides statement digests with `COUNT_STAR`, `SUM_TIMER_WAIT`, `SUM_ROWS_EXAMINED`, `SUM_NO_INDEX_USED`. That is most of the evidence bundle in one query, with no fingerprinting logic to write. `SUM_NO_INDEX_USED > 0` is a free red flag. (Per-request attribution still needs datasource-proxy or p6spy, since perf_schema is global.)

**Better — plans and optimizer insight.** `EXPLAIN ANALYZE` (8.0.18+) gives actual row counts and timings. `EXPLAIN FORMAT=JSON` gives estimated cost. `optimizer_trace` can explain *why* an index was or wasn't chosen — a genuinely useful tool for index hypotheses.

**Worse — no template databases.** Postgres can clone a database instantly; MySQL cannot, and `mysqldump` restore of 1.2M rows is too slow to do between every replay.

**Solution:** bake the seeded dataset into a Docker image, `mysql-seeded:<dataset-hash>`. "Restore" becomes "destroy the container, start a new one from the image" — seconds, and byte-identical every time. The dataset hash in the tag makes it impossible to accidentally compare runs that used different data.

---

## 5. Runtime flow

Twelve phases. Each has a typed input, a typed output artefact, and an explicit AI/code classification. Ordinary Python owns every transition.

---

### P-1 · Onboarding — `code + AI inference`

**In:** repo path.
**Do:** if `.perfagent/profile.yml` exists and its `invalidate_on` files are unchanged, load it. Otherwise: detect stack pack; Claude reads README/compose/app config/CI/pom and infers a `RepoProfile`; code then verifies every field by executing it (build, boot, health-check, one test, DB connect).
**Out:** verified `RepoProfile`, cached in the repo.

---

### P-1b · Feasibility gate — `code`

**In:** `RepoProfile`.
**Do:** answer honestly, before spending money: can I build it? boot it? reach a database? are external calls mockable? can I generate plausible data?
**Out:** proceed, or `ENVIRONMENT_INFEASIBLE` with a specific blocker list.

> On real-world repos this will fail often. **Failing fast and legibly is what makes the tool usable** — the alternative is dying in P4 with a confusing stack trace.

---

### P0 · Provision — `code`

**In:** repo path, endpoint string, `RepoProfile`.
**Do:** create the two git worktrees. Build the **code index**. Start MySQL, both apps, WireMock. Health-check everything.
**Out:** a running environment and a queryable index. No AI has run yet.

**The code index.** Before any AI runs, tree-sitter parses every source file and the structural facts go into a plain SQLite table:

```
CustomerController.search              CustomerController.java:34
  annotations: @PostMapping("/api/customers/search")
  calls: CustomerService.search

CustomerRepository.findByFilter        CustomerRepository.java:22
  annotations: @Query("select c from Customer c where ...")
  implemented_by: SimpleJpaRepository (Spring Data proxy)

Customer.invoices  [field]             Customer.java:58
  annotations: @OneToMany(fetch = LAZY)
```

Without it, "who calls `getCustomerById`?" means grepping and reading six files — ~15,000 tokens, and interface indirection gets missed. With it, it's a 20-token exact answer.

**This is the single biggest lever on both cost and accuracy in the whole system**, and it is pure deterministic parsing. The agent asks the index structural questions and reads source only when it needs to see logic.

---

### P1 · Comprehend — `AI loop, read-only`

**In:** endpoint string, code index, repo skeleton.
**Do:** Claude walks the execution path — finds the controller method matching the route, follows into services and repositories, resolves interfaces to implementations, reads `@Query`/`@Transactional`/fetch types, notes entities and tables. It maintains a worklist of unresolved on-path symbols and pops until empty.
**Out:** `EndpointModel`.

```json
{
  "route": "POST /api/customers/search",
  "entry": "CustomerController.search (CustomerController.java:34)",
  "path": [
    "CustomerController.search:34",
    "CustomerService.search:51",
    "CustomerService.enrich:88",
    "InvoiceService.getOutstanding:40",
    "CustomerRepository.findByFilter:22"
  ],
  "sql_sources": [
    {"method": "CustomerRepository.findByFilter", "kind": "@Query JPQL", "tables": ["customer"]},
    {"method": "lazy load Customer.invoices", "kind": "Hibernate lazy collection", "tables": ["invoice"]}
  ],
  "entities": [
    {"name": "Customer", "table": "customer",
     "relations": [{"field": "invoices", "type": "OneToMany", "fetch": "LAZY"}]}
  ],
  "transactions": [{"method": "CustomerService.search", "propagation": "REQUIRED"}],
  "external_calls": [],
  "tests_covering": ["CustomerSearchIT.shouldFilterByStatus"],
  "uncertainties": [
    "InvoiceService is an interface with 2 impls; picked DefaultInvoiceService based on @Primary"
  ]
}
```

Three deliberate properties:

- **Every entry carries `file:line`**, and code then *validates that each citation resolves* before accepting the artefact. This is how hallucinated methods get caught mechanically rather than by reading.
- **An explicit `uncertainties` field.** An agent forced to declare what it wasn't sure about is far more useful than one that sounds uniformly confident.
- **The artefact, not the transcript, flows downstream.** The exploration may burn 150k tokens; the next phase receives 800.

**Stop condition:** the worklist empties and Claude calls `emit_endpoint_model`. Forced structured output *is* the stopping condition — a schema-forced final tool call is a far more reliable terminator than hoping the model stops talking.

---

### P2 · Workload — `hybrid`

**In:** `EndpointModel`, DB schema.
**Do:** decide what realistic data and realistic requests look like, then create them.
**Out:** seeded MySQL image (`mysql-seeded:<hash>`) + request corpus.

**Data generation — the LLM decides semantics, Python generates rows.**

1. Introspect `INFORMATION_SCHEMA` — tables, columns, types, nullability, FKs, indexes. No AI.
2. Compute the **seed set**: tables in the `EndpointModel`, plus transitive FK *parents* (can't insert an invoice without a customer), plus FK *children* only if the endpoint reads them. Everything else gets one token row or nothing. On a real service this is the difference between seeding 12 tables and 300.
3. Topologically sort by FK dependency. Pure graph code.
4. **Claude produces a `DataPlan`** — row counts and per-column generator specs, not data. It already knows the domain from P1.
5. Python executes the plan with a fixed seed, bulk-inserting via `LOAD DATA LOCAL INFILE`.
6. Validate: FK integrity, row counts, distribution sanity.
7. Hash the `DataPlan`; tag the image with it.

```json
{
  "customer": {
    "rows": 50000,
    "columns": {
      "segment":    {"gen": "enum_weighted", "values": {"SMB": 0.7, "MID": 0.25, "ENT": 0.05}},
      "created_at": {"gen": "date_range", "from": "2021-01-01", "to": "2026-06-01"}
    }
  },
  "invoice": {
    "rows": 1200000,
    "columns": {
      "customer_id": {"gen": "fk_zipf", "ref": "customer.id", "alpha": 1.2},
      "status":      {"gen": "enum_weighted", "values": {"PAID": 0.68, "OPEN": 0.22, "OVERDUE": 0.10}},
      "amount":      {"gen": "money_lognormal", "median": 4200, "sigma": 1.4},
      "due_date":    {"gen": "date_offset", "of": "created_at", "days": [15, 30, 60]}
    }
  },
  "rationale": "fk_zipf on customer_id because AR data is heavily skewed — a few large
                accounts hold most invoices. Uniform distribution would hide per-customer
                fan-out problems."
}
```

**Why the distribution matters more than the volume.** A uniform 24 invoices per customer makes an N+1 look like a mild constant cost. Real AR data is Zipf — a handful of enterprise accounts hold 40% of the invoices — and *that* is what turns an N+1 into a p95 disaster while p50 looks fine. Getting the data shape wrong means measuring a problem production doesn't have. This is the reason Claude is in this loop at all.

**Request corpus** — a fixed list of concrete HTTP requests that serves double duty as benchmark workload (P4/P9) and behavioural test input (P3/P8):

```json
[
  {"id": "r001", "method": "POST", "path": "/api/customers/search",
   "body": {"status": "OVERDUE", "minAmount": 1000, "page": 0, "size": 50}},
  {"id": "r002", "method": "POST", "path": "/api/customers/search",
   "body": {"nameContains": "acme", "page": 3, "size": 20}},
  {"id": "r003", "method": "POST", "path": "/api/customers/search",
   "body": {"status": "GARBAGE"}}
]
```

Note `r003`. **Error paths must stay identical too** — a "fix" that changes a 400 into a 500 has changed behaviour.

**Split the corpus.** ~150 requests are the visible benchmark set; ~50 are **held back**, never used for measurement, used only for the final equivalence check. That is a train/test split for the agent. If it ever finds a way to make the benchmark fast by cheating — caching that exact payload — the held-back set catches it.

---

### P3 · Freeze the oracle — `code, one AI advisory call`

**In:** request corpus, the **unmodified** app.
**Do:** fire all requests at blue. Record every response body, status code, and a hash of resulting DB state. Claude drafts the tolerance policy (which fields are timestamps to ignore, which collections are order-insensitive). Code hashes the whole bundle and seals it.
**Out:** `BehaviourOracle`, content-hashed, immutable.

Full detail in §8.

---

### P4 · Baseline — `code`

**In:** environment, corpus.
**Do:** warm up (discarded), then benchmark **blue only** — green is byte-identical at this point. Capture per-request latency, every SQL statement with timing, a CPU/allocation profile, and `EXPLAIN ANALYZE` for the queries that appeared. Aggregate SQL by digest. Check run-to-run variance; if the machine is too noisy, stop here rather than produce garbage.
**Out:** `BaselineReport` + `EvidenceBundle`. **All numbers, no opinions.**

---

### P5 · Diagnose — `AI loop, read + probe`

**In:** `EndpointModel` + **pre-digested** evidence + **computed** red flags.
**Out:** `HypothesisSet`.

The raw SQL log is 12,000 lines. Claude never sees it. It sees:

```
LATENCY   p50 1980ms   p95 2410ms   n=200

SQL (aggregated by digest)
  #1  select c.* from customer c where c.status = ? ...
      calls: 1     total: 180ms   mean: 180ms    [seq scan, 1.2M rows]
  #2  select i.* from invoice i where i.customer_id = ?
      calls: 847   total: 1520ms  mean: 1.8ms
      sample params: [1043], [1044], [1045]

PROFILE (top self-time, app frames only)
  Hibernate SessionImpl.load       41%
  InvoiceService.getOutstanding    12%
  Jackson serialization             9%

RED FLAGS (computed, not inferred)
  - digest #2 executed 847x with distinct params → possible N+1
  - no_index_used on invoice.customer_id (1.2M rows)
  - SQL total 1700ms of 1980ms measured → 86% of time is data access
```

**We counted, and we told it the count.** The model is never asked to count 847 things — it is bad at that and does not need to be good at it. It gets facts and does the part it is actually good at: connecting "847 identical digests" to "lazy `@OneToMany` on `Customer.invoices`" to "a loop in `enrich()`" and naming the mechanism.

Output — competing, falsifiable, cited:

```json
{
  "hypotheses": [
    {"id": "h1",
     "claim": "N+1: enrich() iterates customers and triggers lazy load of invoices",
     "mechanism": "Customer.invoices is LAZY; enrich() touches it per row in a loop",
     "evidence": ["sql#2:847calls", "profile:SessionImpl.load:41%",
                  "endpointmodel:CustomerService.enrich:88"],
     "attributed_ms": 1520,
     "prediction": "with page size 10 instead of 50, query count drops to ~11",
     "confidence": 0.85},

    {"id": "h2",
     "claim": "Missing index on invoice.customer_id",
     "attributed_ms": 300,
     "prediction": "EXPLAIN shows no index used; with index, mean drops below 0.3ms",
     "confidence": 0.6},

    {"id": "h3",
     "claim": "Jackson serialization of unused fields",
     "attributed_ms": 180,
     "confidence": 0.25}
  ],
  "probes_run": [
    {"tool": "run_probe", "args": {"size": 10},
     "discriminates_between": ["h1", "h2"],
     "result": "query count 847 → 11; confirms per-row pattern"}
  ]
}
```

Four mechanisms make this reliable:

1. **At least three competing hypotheses are required** before ranking. This prevents locking onto the first plausible story.
2. **Every probe must declare `discriminates_between`.** A probe that cannot separate two hypotheses is not an experiment, it is wandering. This one schema field is what turns exploration into science.
3. **`attributed_ms` arithmetic is checked in Python.** If the hypotheses claim 4000ms of a 1980ms request, reject and re-ask.
4. **Every evidence reference must resolve** to a real artefact ID.

**Why this matters, concretely:** h1 and h2 are *both real*, and h1 dominates. An agent that picked h2 would add an index, get a 12% improvement, pass every gate, and ship a PR that missed the actual problem. **A 12% win looks like success.** The ranking-plus-attribution discipline exists to prevent exactly that, and the eval suite must be able to detect it.

**Terminal:** if nothing clears the confidence and attribution bar → `NO_BOTTLENECK_FOUND`, with a report. This is a correct answer.

---

### P6 · Plan — `AI, single constrained call`

**In:** top hypotheses.
**Do:** Claude proposes 2–4 concrete fixes, each with: the mechanism it expects to change, the files it will touch, the predicted effect, the risk, and what could break. Code assigns a **risk tier from a fixed table** and selects the highest expected-value candidate within the allowed tier.
**Out:** `CandidatePlan[]` + selection.

| Tier | Examples | Policy |
|---|---|---|
| **1 — Low** | Add index; `@EntityGraph`/fetch join; `@Transactional(readOnly)`; JSON view narrowing | Auto-approve |
| **2 — Medium** | Query rewrite; batch fetching; move filtering into SQL; DTO projection | Auto-approve with stricter verification |
| **3 — High** | Restructuring call flow; changing transaction boundaries; adding caching | **Flag-gated (§7.3)**, human review emphasised |
| **4 — Forbidden** | Schema changes beyond additive indexes; changing API contract; touching auth | Never |

Optimizations are **discovered from evidence, not selected from a checklist.** This table governs *risk policy*, not the space of possible fixes.

---

### P7 · Implement — `AI loop, write + build + test, green worktree only`

**In:** the selected plan.
**Do:** propose diff → apply → build → run existing tests → read failures → repair. Loop.
**Out:** `Patch` (a real diff on a real branch), or `ABANDONED` with a reason.

Guardrails, all enforced in code:

- Writes restricted to a **path allowlist derived from the plan itself**. Not "the repo" — the files the plan declared.
- `src/test/**` is **immutable**. The single most important guardrail in the system: an agent that can edit tests will always find a way to make the build green.
- Test count cannot decrease.
- Every iteration is **checkpointed** (`git stash`/`reset`), so a bad state is a reset, not a debugging session.
- Hard iteration cap. Escalating failures → abandon and report honestly.

Build and test failures are not errors to be hidden — **they are the highest-value context the agent will ever receive.** Compiler messages are precise, cheap and authoritative. Feed them back verbatim.

---

### P8 · Verify — `code, AI only for triage`

**In:** the sealed `BehaviourOracle`, the patched build.
**Do:** restore the database from the seeded image. Replay every request against green. Canonicalise and compare responses to the recordings. Compare resulting DB state. Compare error paths. Replay the **held-back** set.
**Out:** `EquivalenceVerdict` — **decided by code, never by the model.**

On difference, Claude may *explain* the diff (is this a real behaviour change, or a legitimate timestamp?). It may not *decide*. Loosening the oracle requires a human, and doing so downgrades the run to "needs review."

- Fail with budget remaining → back to P6 with the diff as new evidence.
- Fail with budget exhausted → `BEHAVIOUR_NOT_PRESERVED`, no PR.

---

### P9 · Measure — `code`

**In:** blue and green, both running.
**Do:** the same harness as P4, but **interleaved** — blue, green, blue, green — for the full corpus. Compute both distributions and a significance test (Mann-Whitney U on paired per-request latencies).
**Out:** `PerfReport`.

```
                blue        green       delta
p50            1980ms       520ms      -73.7%
p95            2410ms       640ms      -73.4%
SQL queries       847          2       -99.8%    ← mechanism check
rows examined  1.2M         50k        -95.8%
p-value                                 <0.001
```

**The mechanism check is as important as the timing.** If the plan predicted "847 queries become 2" and queries are still 847 but latency dropped, something else caused the improvement and the diagnosis was wrong. That is a red flag even when the numbers look great.

---

### P10 · Gate — `code, fixed rules`

All must pass:

1. Behaviour preserved (P8 verdict, including held-back set).
2. Improvement above threshold (e.g. p95 ≥ 15% better).
3. Statistically significant (p < 0.01).
4. Diff within minimality bounds — no unrelated files, no unexplained hunks.
5. **Mechanism matches the plan's prediction.**
6. No forbidden-tier change present.

Any failure → retry a different candidate, or terminate honestly. **The gate is code with fixed thresholds. The model has no vote here.**

---

### P11 · Ship — `code, AI writes prose only`

**In:** all artefacts.
**Do:** commit, push branch, `gh pr create`. The PR body is **rendered from artefacts**: Claude writes the root-cause narrative; every number, file path, diff and measurement is templated. The renderer refuses to emit a claim with no artefact behind it.
**Out:** PR + `report.html` + `trace.jsonl`.

PR body sections: endpoint · bottleneck identified · root cause · optimization implemented · why this optimization was selected over the alternatives · behavioural verification (N requests, M identical, oracle hash) · before / after / improvement / p-value · mechanism confirmation · files changed · full run trace link.

---

## 6. AI vs deterministic — the complete map

| Component | Classification | Why |
|---|---|---|
| Repo profile inference | **AI** | Reading prose docs and inferring intent |
| Repo profile verification | Code | Execute it and see |
| Code index construction | Code | Parsing has one right answer |
| Route → controller resolution | Code | Index lookup |
| Execution path exploration | **AI loop** | Unknown depth, requires judgement about relevance |
| Citation validation | Code | Does `file:line` exist? |
| Data volume/distribution design | **AI** | Domain judgement about realistic shape |
| Data generation | Code | Must be exact and reproducible |
| Request corpus expansion | Code | Templating |
| Recording the oracle | Code | It's I/O |
| Tolerance policy drafting | **AI (advisory)** | Knowing a field is a timestamp requires semantics |
| Tolerance policy approval | Human | It's the definition of correct |
| Benchmark execution | Code | Precision matters, judgement doesn't |
| SQL aggregation, red flags | Code | Counting — never ask a model to count |
| Bottleneck diagnosis | **AI loop** | Connecting evidence to mechanism; open-ended |
| Attribution arithmetic check | Code | It's arithmetic |
| Optimization candidate generation | **AI** | Open-ended engineering creativity |
| Risk tier assignment | Code | Fixed policy table |
| Candidate selection | Code | Expected value within risk budget |
| Writing the patch | **AI loop** | It's code generation |
| Applying the patch | Code | With allowlist enforcement |
| Build / test execution | Code | Deterministic |
| Response comparison | Code | Never let a model decide equality |
| Diff triage on failure | **AI (advisory)** | Explaining *why* two outputs differ |
| Equivalence verdict | Code | Non-negotiable |
| Performance measurement | Code | |
| Significance testing | Code | |
| Gate decision | Code | Fixed thresholds, no model vote |
| Git branch / commit / PR | Code | |
| Root-cause narrative | **AI** | Prose is what models are for |
| Every number in the report | Code | Evidence ledger (§3.3) |

**Rule of thumb: if the task has one right answer, it is not a job for a model.**

---

## 7. Tool catalogue

Tools are the agent's entire action space. Their design is the highest-leverage engineering in the system — a tool that returns exactly the right 200 tokens beats a shell that returns 20,000.

### 7.1 Design rules

1. **One purpose per tool.** No `do_thing(mode=...)`.
2. **Return the smallest sufficient answer.** Never dump a file when a symbol will do.
3. **Errors are informative and actionable.** `"symbol not found; did you mean CustomerSvc?"` beats `"error"`.
4. **Truncate with a continuation handle**, never silently.
5. **Every call is logged structurally** — tool, args, result size, duration, phase.
6. **Capability scoped per phase.** Write tools do not exist during comprehension.

### 7.2 The tools

| Tool | Phase | In | Out | Permissions |
|---|---|---|---|---|
| `find_symbol` | P1, P5, P7 | name / fuzzy | declarations + `file:line` + annotations | index read |
| `get_callers` / `get_callees` | P1, P5 | symbol | edge list | index read |
| `read_source` | P1, P5, P7 | `file:line` range | source slice, capped | source read, worktree-scoped |
| `code_search` | P1, P5 | regex + filters | matches with context, capped | source read |
| `get_schema` | P1, P2, P5 | table names | DDL, indexes, cardinality | DB read (schema only) |
| `emit_endpoint_model` | P1 | schema-forced | terminates loop | — |
| `explain_query` | P5 | SQL + params | `EXPLAIN ANALYZE` output | DB read |
| `query_stats` | P5 | digest filter | perf_schema aggregates | DB read |
| `run_probe` | **P5 only** | request variation, flags | latency + SQL + profile | **network-isolated, read-only source, timeout-capped** |
| `emit_hypotheses` | P5 | schema-forced | terminates loop | — |
| `emit_plan` | P6 | schema-forced | candidate plans | — |
| `apply_patch` | P7 | unified diff | applied / rejected + reason | **write, allowlist only, green worktree only, `src/test/**` denied** |
| `build` | P7 | — | exit code + compiler output | fixed command from profile |
| `run_tests` | P7 | optional filter | pass/fail + failure text | fixed command from profile |
| `emit_narrative` | P11 | artefact refs | prose only | cannot emit numbers |

**No tool for:** arbitrary shell, network access, writing outside the green worktree, touching tests, DB writes outside seeding, git operations, or reading the oracle.

### 7.3 Rollback strategy for high-risk changes: feature flag, not `/v2`

A recurring real-world case: three intermediate methods each make the same call, and the fix is to make it once. Clean if those methods have no other callers — but in practice they do, and *their* callers have callers.

Two candidate approaches were considered:

- **`/v2` endpoint** — new version alongside the old, UI repoints to v1 on trouble, later cleanup PR removes v1.
- **Feature flag** — original path untouched, new path beside it, one `if (flag)` switch.

**Decision: feature flag.**

It gives the same rollback story, faster — a flag flip in seconds, no deploy, no UI change — and it does not signal an API contract change to consumers when nothing about the contract changed. `/v2` also multiplies the surface that must be maintained and creates a cleanup obligation across every consumer.

What the agent produces for a Tier-3 change:

- Original code path untouched.
- New optimized path beside it.
- One `if (flag)` switch at the entry point.
- A test that exercises **both** paths and asserts identical output.

Rollback = turn the flag off. Cleanup = a separate, tiny PR later, which the system can open on request.

**This applies only to Tier 3.** Most optimizations — an index, a fetch join, a query rewrite — are internal and need no flag. Flag-gating is deferred past the MVP.

---

## 8. Behaviour verification — the oracle

> The single most important component. If this is weak, the product is a demo.

### 8.1 The idea

"Oracle" is the testing term for *the thing that tells you the right answer*. In a unit test, the oracle is your `assertEquals`.

Three ways to answer "does the patched endpoint still behave the same?":

- **Ask Claude.** "I believe this is behaviourally equivalent." Worthless — same system that wrote the change.
- **Run the existing tests.** Better, but tests cover what someone thought to check, which for most endpoints is a thin slice. An N+1 fix can pass every test and silently drop a field.
- **Ask the original code.** ← this is the oracle.

This is a known technique: **golden master / characterisation testing.** You generate an enormous test suite for free by treating current behaviour as the specification.

### 8.2 Mechanism

**Before any code changes:**

1. Fire the full corpus at the **original** app.
2. Record: response body, status code, headers-of-interest, and a hash of resulting DB state.
3. That bundle is now the definition of correct — not what the spec says, but **what this code actually does today**.

**After the patch:**

4. Restore the DB to the identical starting point (fresh container from the seeded image).
5. Fire the exact same requests at the patched app.
6. Compare. Any difference → reject.

### 8.3 The two subtleties that make or break it

**(a) Some differences are legitimate.** Timestamps. Generated IDs. The ordering of a collection the API never promised to sort. Floating-point formatting. So the oracle carries a **tolerance policy** — fields to ignore, fields to compare order-insensitively, numeric epsilons.

**(b) The tolerance policy must be written before the patch exists.** This is the "frozen" part, and it is the whole game. If you write "ordering doesn't matter" *after* discovering your patch changed the ordering, you are not verifying — you are rationalising. So:

- The oracle is built in P3, against unmodified code.
- It is **content-hashed**.
- The patching agent cannot read it and cannot modify it.
- Loosening it requires a human, and downgrades the run to "needs review."

### 8.4 What is compared

| Dimension | Method |
|---|---|
| Response body | Canonicalised deep JSON compare under the tolerance policy |
| Status codes | Exact |
| Error responses | Exact — a 400 becoming a 500 is a behaviour change |
| DB side effects | Table-state hash before/after each request |
| External calls | WireMock request journal: same calls, same order, same payloads |
| Held-back set | Same comparison, on requests never used for measurement |

### 8.5 Escape hatches, explicitly closed

- Agent cannot modify tests → `src/test/**` write-denied.
- Agent cannot modify the oracle → not in its tool surface; hash-verified before use.
- Agent cannot narrow the corpus → corpus sealed in P3.
- Agent cannot special-case benchmark inputs → held-back set catches it.
- Agent cannot declare success → verdict is computed by code.

### 8.6 MVP version

Roughly 80 lines: a JSON dump of ~30 recorded responses, a comparison function with a few ignore rules, and a hash. **Small, but not skippable.** Without it the product is fake.

---

## 9. Guardrails

| Risk | Control |
|---|---|
| Modifying unrelated code | Path allowlist derived from the approved plan; diff minimality gate |
| Gaming the tests | `src/test/**` immutable; test count cannot decrease |
| Gaming the oracle | Oracle frozen, hashed, out of tool surface |
| Gaming the benchmark | Held-back request set |
| Claiming unmeasured success | Evidence ledger — renderer cannot emit unbacked claims |
| Dangerous commands | No shell; fixed commands from the verified profile |
| Network exfiltration / side effects | `run_probe` network-isolated; WireMock for externals |
| Destructive DB changes | Schema changes limited to additive indexes; DB is disposable and image-restored |
| Infinite loops | Per-phase step caps, token caps, wall-clock caps, no-progress detection |
| Runaway cost | Hard token budget per phase; run aborts with partial findings |
| Corrupted repo state | All work in the green worktree; per-iteration checkpoints; blue never touched |
| Forbidden change classes | Tier-4 list rejected at plan time and re-checked at gate time |
| Silent scope reduction | Any cap that bounds coverage is logged and surfaced in the report |

Two structural properties do most of the work:

- **The blue worktree is never written to.** Ever. It is the reference implementation for the entire run.
- **Capabilities are phase-scoped.** The comprehension agent physically cannot write a file, because that tool is not in its schema list.

---

## 10. Observability

Debugging an agent means answering "why did it do that?" hours later. That requires a real event log, not print statements.

### 10.1 The trace

Append-only JSONL, one file per run, one event per line:

```json
{"ts":"...","run":"01JBQ8","phase":"P5","seq":41,"type":"tool_call",
 "tool":"explain_query","args_hash":"...","duration_ms":180,
 "result_bytes":2140,"tokens_in":8200,"tokens_out":310,"cost_usd":0.041}
```

Event types: `phase_start`, `phase_end`, `llm_request`, `llm_response`, `tool_call`, `tool_result`, `artefact_written`, `guardrail_triggered`, `checkpoint`, `gate_evaluated`, `terminal_state`.

### 10.2 What must be recoverable from the trace alone

- Every prompt sent and every response received, replayable.
- Every tool call with arguments and result size.
- Token and dollar cost per phase, per agent, per run.
- Every artefact, content-addressed.
- Every guardrail that fired and what triggered it.
- Every gate evaluation with its inputs.
- Where time went — LLM latency vs build vs benchmark.

### 10.3 The human-readable view

The report renders the trace as the narrative of the run:

```
Agent started · POST /api/customers/search
  ↓ inspected CustomerController.java:34
  ↓ followed CustomerService.search → enrich → InvoiceService
  ↓ found @Query on CustomerRepository.findByFilter
  ↓ built EndpointModel (7 classes, 2 SQL sources)
  ↓ seeded 50k customers / 1.2M invoices
  ↓ sealed oracle (200 requests, sha256 4f9a1c)
  ↓ baseline p95 2410ms · 847 queries
  ↓ 3 hypotheses · 11 probes
  ↓ probe: size=50→10 · queries 847→11 · confirms h1
  ↓ selected @EntityGraph fetch join (Tier 1)
  ↓ 2 build iterations · 1 file · +6/-2
  ↓ 200/200 responses identical · DB state identical
  ↓ p95 2410ms → 640ms (-73.4%, p<0.001) · queries 847→2
  ↓ all gates passed
  ↓ PR #42 opened
```

Cost visibility is a first-class feature: **per-phase token spend is a design signal.** If comprehension costs $4 per run, the code index needs work.

---

## 11. Evaluation

Verification asks "is *this* patch correct?" **Evaluation asks "is the *system* any good?"** They are different activities and both are required.

### 11.1 Four distinct kinds

| Kind | Question | Method |
|---|---|---|
| **Software correctness** | Does the patch preserve behaviour? | Deterministic — the oracle. Per run. |
| **Performance** | Did it actually get faster? | Deterministic — interleaved A/B + significance test. Per run. |
| **Agent workflow** | Did the agent behave well? | Trace assertions: step counts, tools used, guardrails fired, cost, no-progress loops. |
| **LLM output quality** | Was the *reasoning* right? | Requires labelled ground truth (below). Hardest and most valuable. |

### 11.2 The ground-truth scenario suite

The core evaluation asset: a set of endpoints with **known, deliberately planted bottlenecks**, and an answer key the agent cannot reach.

| # | Planted problem | Expected diagnosis | Expected fix |
|---|---|---|---|
| 1 | N+1 on lazy `@OneToMany` | N+1 | fetch join / `@EntityGraph` |
| 2 | Missing index on filter column | no index used | add index |
| 3 | In-memory filtering after full load | over-fetching | push predicate into SQL |
| 4 | Same service call made 3× in one flow | redundant calls | reuse result |
| 5 | Per-row external HTTP call | network fan-out | batch |
| 6 | Over-fetching wide entity for 2 fields | over-fetching | DTO projection |
| 7 | `count(*)` + full page load per request | poor pagination | keyset pagination |
| 8 | Two problems, one dominant (80/20) | **must rank correctly** | fix the dominant one |
| **9** | **None — already efficient** | **`NO_BOTTLENECK_FOUND`** | **no change** |
| **10** | **None — slow but by design (large legit payload)** | **`NO_BOTTLENECK_FOUND`** | **no change** |

**Scenarios 9 and 10 are negative controls and are non-negotiable.** A system that cannot say "nothing to do here" will confidently patch working code. These are the two scenarios most likely to be cut for time and the two that must never be cut.

Scenario 8 is the calibration test: it detects the failure mode where the agent finds *a* real bottleneck, ships a 12% win, and misses the 78% one — a failure that *looks like success*.

### 11.3 Metrics

Per scenario, over ≥3 runs each (agents are stochastic; a single run tells you almost nothing):

- **Diagnosis accuracy** — did it name the planted bottleneck? (root-cause match, not keyword match)
- **Ranking accuracy** — on multi-problem scenarios, did it pick the dominant one?
- **Evidence validity** — do all cited artefacts resolve, and does the attribution arithmetic hold?
- **Fix validity** — behaviour preserved AND significant improvement.
- **Minimality** — files touched vs files necessary. Unrelated edits are a defect even if the patch works.
- **Negative-control precision** — did it correctly do nothing on 9 and 10?
- **Stop discipline** — did it terminate for the right reason?
- **Cost & latency** — dollars and wall-clock per run.
- **Variance** — same scenario, 3 runs, how different are the outcomes? High variance is a design smell, not bad luck.

### 11.4 Evaluating the reasoning without a human reading every trace

Where ground truth is a judgement call ("was this root cause identified correctly?"), use an LLM judge — but under constraints, because a naive judge is worse than useless:

- The judge sees the **evidence and the answer key**, never the agent's reasoning narrative. It grades the conclusion against reality, not the eloquence of the argument.
- Rubric-based, single dimension per call.
- Calibrated against ~30 human-labelled examples before it is trusted.
- Never used for anything the oracle or arithmetic can decide.

---

## 12. Failure and recovery

### 12.1 Recovery model

Every phase boundary is a **checkpoint**: artefacts written, content-addressed, state persisted. A run can be resumed from any completed phase. Within P7, every iteration is a git checkpoint, so a broken working tree is a reset rather than a debugging session.

### 12.2 Failure taxonomy

| Failure | Detection | Response |
|---|---|---|
| Build fails | exit code | Feed compiler output back; retry within cap |
| Tests fail | test report | Feed failure text back; retry within cap |
| Path unresolvable in P1 | worklist stalls | Report partial model with `uncertainties`; may still proceed |
| Benchmark too noisy | variance check in P4 | **Abort before diagnosis** rather than produce garbage |
| No hypothesis clears the bar | P5 confidence gate | `NO_BOTTLENECK_FOUND` |
| Behaviour differs | oracle compare | Retry alternative candidate → else `BEHAVIOUR_NOT_PRESERVED` |
| Improvement insignificant | P9 stats | Try next candidate → else `NO_SIGNIFICANT_IMPROVEMENT` |
| Mechanism mismatch | P10 gate | Reject — diagnosis was wrong even if numbers look good |
| No progress in a loop | repeated identical tool calls / no new artefacts | Break, escalate, report |
| Budget exhausted | counters | `BUDGET_EXHAUSTED` + partial findings |
| Environment broken | health checks | `ENVIRONMENT_INFEASIBLE` + blockers |

**The governing rule: every failure produces an honest report.** A run that ends in `NO_BOTTLENECK_FOUND` with good evidence is a successful run of the system.

---

## 13. State and context architecture

### 13.1 Three distinct kinds of state — do not conflate them

| | Lives in | Lifetime | Who reads it |
|---|---|---|---|
| **Run state** | `runs/<id>/state.json` | one run | orchestrator |
| **Artefacts** | `runs/<id>/artefacts/*` content-addressed | forever | phases, report, evals |
| **Context** | the message array | one agent loop | one Claude call |

**Artefacts are the interface between phases. Context is disposable.** P1's exploration may consume 150k tokens; P5 receives an 800-token `EndpointModel`. This is the mechanism that keeps a twelve-phase run inside a context window and makes each phase independently testable and replayable.

### 13.2 Context management inside a loop

Techniques, in order of preference:

1. **Better tools** — the primary fix. If the agent needs 40 file reads, the index is inadequate.
2. **Result shaping** — return the symbol, not the file.
3. **Structured scratchpad** — the agent maintains its own worklist/findings as an artefact rather than relying on the transcript as memory.
4. **Compaction** — at a threshold, summarise older turns into a findings digest and drop the raw exchanges.
5. **Prompt caching** — stable system prompt + tool schemas + index digest cached; only the tail varies.

The important lesson: **context management is mostly a tool-design problem, not a summarisation problem.**

### 13.3 On-disk layout

```
.perfagent/
  profile.yml                      # verified repo profile (committed)
  index.sqlite                     # code index (gitignored)
  runs/<run_id>/
    state.json                     # FSM state, resumable
    trace.jsonl                    # append-only event log
    artefacts/
      endpoint_model.json
      data_plan.json
      request_corpus.json
      oracle/                      # recordings + policy + HASH
      baseline/                    # latency, sql, profile, plans
      hypotheses.json
      plan.json
      patch.diff
      verdict.json
      perf_report.json
      gate.json
    report.html
```

---

## 14. Claude integration

### 14.1 Required capabilities

| Capability | Used for | Necessity |
|---|---|---|
| Tool use / function calling | everything | Required |
| Forced tool choice | schema-guaranteed artefacts and loop termination | Required |
| Extended thinking | diagnosis and planning | High value |
| Prompt caching | stable prefix across a loop's turns | Required for cost sanity |
| Long context | comprehension on large repos | High value |
| Structured outputs | typed artefacts | Required |

### 14.2 Integration rules

1. **Hand-roll the loop first.** Build the raw Messages API request/response/tool-dispatch cycle yourself once before adopting any SDK convenience. Understanding that the LLM is a *stateless function* — and that "the agent" is entirely your own engineering around it — is the foundation for everything else in this document.
2. **One system prompt per agent role**, not one global prompt.
3. **Force structured output for every artefact.** Never parse prose into a schema.
4. **Forced final tool call as the stop condition.** Far more reliable than hoping the model stops talking.
5. **Cache the stable prefix** — system prompt, tool schemas, index digest.
6. **Deterministic settings for reproducibility**; then deliberately test with varied settings to measure variance.
7. **Never ask the model to count, sum, or compare.** Compute it and hand it the answer.
8. **Model tiering:** the expensive model for diagnosis and planning; a cheap model is adequate for mechanical summarisation and diff description. Route by task difficulty, not uniformly.
9. **No framework in v1.** No LangChain, no vector DB — not out of purism, but because retrieval-by-embedding is the wrong tool when the search space has exact structure and an index, and because hand-rolling once is the difference between knowing agents and knowing a library. Adopt a framework later, from a position of knowing what it takes away.

---

## 15. Scope: MVP vs production

### 15.1 MVP (first working end-to-end version)

Target: **one endpoint, on one repo that already runs locally.** No generality.

**In:**

- Hand-written LLM client (retries, cost meter, cache) and agent loop.
- `fs_read` + `code_search` + forced `emit_endpoint_model` → comprehension.
- Hand-written seed script to make one endpoint genuinely slow.
- SQL capture + a Python script that hits the endpoint N times and aggregates by digest.
- Diagnosis agent with ≥3 cited hypotheses and arithmetic validation.
- Second worktree, `apply_patch` with allowlist, `mvn package`, `mvn test`, repair loop.
- **Oracle v0**: ~30 recorded responses, compare, hash.
- Interleaved before/after timing.
- `gh pr create` + a markdown report rendered only from recorded artefacts.

**Explicitly cut from the MVP:**

tree-sitter code index (ripgrep is adequate, just token-expensive) · Docker for the app (two `java -jar` processes on two ports) · realistic seeded data via `DataPlan` (a crude script suffices) · the orchestrator state machine (a Python script calling functions in order) · statistical significance · the eval suite · negative controls · stack packs · repo profile · feasibility gate · HTML report · flag-gating · run_probe.

**Four things that are never cut, even in the MVP** — each is 1–2 hours and each is the line between a product and a demo:

1. Record the oracle **before** patching.
2. Every hypothesis cites evidence, and citations are validated.
3. Report numbers come only from measured data.
4. Test files are unwritable.

**What the MVP honestly cannot do:** you will not know how often it works. Run it on ten endpoints and perhaps four produce a good PR, three produce nothing, and three produce something subtly wrong that passed a 30-response oracle — and you will not be able to tell which is which without reading each trace. Pointing it at an unfamiliar repo will fail confusingly.

### 15.2 Production

Everything above, plus: the full twelve-phase state machine with resume · tree-sitter index · Dockerised environment with pinned resources · `DataPlan`-driven seeded images · full oracle with held-back set and DB-state hashing · statistical significance · all gates including mechanism confirmation · the ten-scenario eval suite run ≥3× · complete trace and HTML report · repo profile with execution-verified fields · feasibility gate · flag-gating for Tier-3 changes · stack pack seam with one implementation behind it.

### 15.3 The gap between them, stated plainly

The months between MVP and production buy exactly one thing: **the ability to trust the output without reading the trace.** That is evals, negative controls, the frozen oracle, statistical rigour, and guardrails.

That is also, not coincidentally, the entire second half of AI engineering. The difference between an LLM wrapper and a reliable AI system *is* precisely that work.

---

## 16. What "success" means

### 16.1 Product success

- On the scenario suite: correct diagnosis on ≥8/10, correct no-op on both negative controls, zero behaviour regressions escaping to a PR.
- On a real service: produces a verified, measured PR when the bottleneck is within its competent classes, and an honest inconclusive report otherwise.
- A reviewer can merge a PerfAgent PR **without redoing the investigation**, because the evidence is attached.

### 16.2 Competent classes (be honest about scope)

**Strong on:** N+1s · missing indexes · in-memory filtering that belongs in SQL · redundant calls · over-fetching · poor batching · unnecessary serialisation · pagination.

**Will stay hard:** anything requiring architectural change · anything with a cache-invalidation story · anything where "correct" is genuinely ambiguous.

That is a good place to land. The hard ones are the ones that *should* have a human on them.

### 16.3 Anti-goals — signs the project went wrong

- A system that always produces a patch. (It has learned to please, not to help.)
- Numbers in a report that no artefact backs.
- An agent with a shell.
- Multiple agents introduced because "multi-agent" sounds sophisticated.
- Passing tests that the agent edited.
- A framework adopted before the mechanics were understood.
- Negative controls cut for time.

---

## 17. Learning objectives (secondary goal, tracked deliberately)

This project doubles as a structured path into AI engineering. The concept each component forces you to learn:

| Component | Concept |
|---|---|
| Hand-written LLM client | The LLM is a stateless function; everything else is your engineering |
| Agent loop + tool dispatch | `tool_use` / `tool_result` mechanics; what "agent" actually means |
| Tool design & result shaping | **Tool design is context design** — the highest-leverage skill |
| Code index | When *not* to use AI; retrieval without embeddings |
| Forced structured output | Schema-guaranteed artefacts; termination conditions |
| Artefact-based phase boundaries | Context management as architecture, not summarisation |
| Orchestrator FSM | Workflow vs agent; where determinism beats autonomy |
| Evidence bundle & red flags | Grounding; hallucination prevention by construction |
| Hypothesis discipline | Making model reasoning falsifiable and checkable |
| `run_probe` | Agents doing experiments, not just retrieval |
| Patch/build/test loop | Autonomous code modification; **errors as high-value context** |
| Path allowlists, immutable tests | Guardrails as capability restriction, not instruction |
| Frozen oracle | Verification vs evaluation; why graders must be untouchable |
| Interleaved A/B + significance | Objective evaluation of agent outcomes |
| Gate rules | Human/AI boundaries encoded in code |
| Trace & cost accounting | Agent observability and debugging |
| Scenario suite + metrics | **Building reliable evaluations** — the core AI engineering skill |
| Variance measurement | Agents are stochastic; a single run proves nothing |
| Terminal-state design | Teaching a system to say "I don't know" |

**Discipline:** the ~800 lines that constitute the actual AI engineering — LLM client, agent loop, tool dispatcher, context compaction, tool result shapers, oracle comparison, gate rules, eval metrics — are written by hand, not generated. Everything else (sandbox app, data generators, Docker config, report templates, JSON schemas) can and should be generated. Burning weekends on Spring boilerplate is how the project dies at week 6.

---

## 18. Decisions log

| Decision | Rationale |
|---|---|
| State machine with agentic segments, not one big agent | Reproducibility, resumability, evaluability, cost control |
| Exactly four agentic loops | The agency test: only where step count is runtime-dependent |
| No general shell | Unbounded action space cannot be guarded; destroys observability |
| Oracle frozen before patching, hashed, out of tool surface | An agent that can edit its grader isn't verified |
| Evidence ledger in the renderer, not the prompt | Instructions are advisory; code is not |
| Interleaved A/B measurement | Sequential measurement attributes machine drift to the patch |
| Mechanism check as a gate | Catches "right answer, wrong reason" — improvement without the predicted cause |
| MySQL: bake seeded data into an image | No template DBs; dump/restore too slow; image tag pins the dataset |
| Feature flag over `/v2` for Tier-3 changes | Same rollback, faster, no false API-contract signal, less surface to maintain |
| Negative controls mandatory in the eval suite | A system that can't say "nothing to do" will patch working code |
| No framework, no vector DB in v1 | Exact structure beats embeddings here; hand-rolling is the lesson |
| Repo profile inferred by AI, verified by execution | Only way to be repo-agnostic without hardcoding or guessing |
| Build the stack-pack seam, one implementation behind it | Seam costs an afternoon; premature generality costs the project |
| Ten terminal states, one of which is a PR | Honest failure is a feature; always-produces-a-patch is untrustworthy |

---

## 19. Open questions

1. **Concurrency in the workload.** v1 measures sequentially. Some bottlenecks (lock contention, connection-pool starvation) only appear under concurrency, but concurrency adds measurement variance. Defer, but design the harness so it can be added.
2. **Distributed flows.** If the endpoint calls another internal service, WireMock stubs it — which means cross-service bottlenecks are invisible. Acceptable for v1; a real limitation to state.
3. **Multi-tenancy in seeded data.** Growfin data is tenant-scoped; the `DataPlan` must respect tenant distribution or query plans will be unrepresentative.
4. **Where the oracle's tolerance policy gets human approval** — inline prompt, or a reviewable file committed to the repo? Leaning toward the latter, so it accumulates as a durable asset per repo.
5. **Continuous mode.** Long term, the natural evolution is APM-triggered: Datadog flags a slow endpoint, PerfAgent runs on it automatically. Explicitly out of scope for v1, but the architecture should not preclude it.

---

*End of document.*
