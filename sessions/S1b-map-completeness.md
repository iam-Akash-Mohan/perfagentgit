# Session 1b — Completing the endpoint map

Read `CLAUDE.md` first. This is a focused follow-up to Session 1, not a new phase.

**The problem this fixes.** On the target endpoint, 93% of latency is non-SQL work —
`enrich()` builds one string per invoice, producing a 142 KB response body. The
`EndpointModel` has no field that can express that. `data_access`, `sql_sources`,
`entities`, `transactions` and `external_calls` are all about the database. So the
diagnosis phase had to invent "serialization dominates" as a hypothesis with no
structural support, and the patch agent then attacked Jackson — because the map told it
where the queries were, and when the queries turned out not to matter it had nothing
else to reason over.

**Scope:** `contracts.py`, `tools/submit.py`, `understand/validate.py`,
`prompts/understand.*.md`, `understand/agent.py`, `render/endpoint.py`, tests. Nothing
else. Do not touch `diagnose/`, `patch/`, `oracle/`, `gate/`, `measure/`.

---

## Part A — Contract additions

Add to `contracts.py`, in the comprehension section.

### A1 · PerRowWork — the one that matters

```python
class PerRowWork(BaseModel):
    """Work that executes once per RESULT ROW and is not a database access.

    This is the field the target endpoint needed and did not have. A loop that
    builds one string per invoice does not appear in data_access, does not appear
    in sql_sources, and was 93% of the measured latency. Every phase downstream
    was reasoning about queries because queries were the only thing the map could
    describe.
    """
    description: str        # "appends one 'overdue-{id}' string per invoice"
    file: str
    line: int
    kind: Literal["string_building", "collection_processing", "object_creation",
                  "computation", "serialization", "external_call", "other"]
    scales_with: str        # what drives the count: "invoices per customer"
    bounded: bool           # is there ANY limit on how much work this does?
    bound: str | None = None    # if bounded, what bounds it: "page size, max 100"

    def citation(self) -> "Citation":
        return Citation(file=self.file, line=self.line, symbol=None)
```

`bounded=False` with `scales_with="invoices per customer"` on a skewed relation *is* the
finding. Nothing else in the artifact can state it.

### A2 · ResponseShape

Responses in the target codebase are DTO-based with no `@JsonView` and no dynamic
shaping, so keep this simple. Do not add `json_view` or `ignored_fields`.

```python
class ResponseShape(BaseModel):
    """What the endpoint returns, and which parts of it are unbounded.

    `unbounded_collections` exists so that a later phase learns BEFORE writing a
    patch that the response size is part of the API contract. The first patch
    attempt spent four build iterations discovering that by hand.
    """
    dto: str | None = None
    fields_returned: list[str] = []
    computed_fields: list[str] = []       # getters that do real work, not accessors
    unbounded_collections: list[str] = [] # fields whose size scales with data
    citation: Citation | None = None
```

### A3 · PaginationInfo

```python
class PaginationInfo(BaseModel):
    """How the endpoint pages, and what that costs.

    The measured baseline showed SELECT COUNT(id) FROM customer at 1,640ms with
    no_index_used — a real finding with nowhere to be recorded.
    """
    mechanism: Literal["offset", "keyset", "none", "unbounded"]
    fires_count_query: bool = False
    max_page_size: int | None = None      # None means uncapped
    citation: Citation | None = None
```

### A4 · Recorded if present, not emphasised

The target entities are plain, so these will usually be empty. Add them anyway — the
product must run on repos where they are not empty, and an empty list is honest while a
missing field is a blind spot.

```python
class CacheInfo(BaseModel):
    kind: Literal["spring_cacheable", "hibernate_second_level",
                  "query_cache", "manual", "other"]
    on: str                               # method or entity
    cache_name: str | None = None
    citation: Citation


class HiddenQueryTrigger(BaseModel):
    """SQL that fires from somewhere other than the call site.

    @Formula, @PostLoad, entity listeners, Hibernate filters. Invisible when
    reading the service layer, and a classic source of "where did that query
    come from".
    """
    kind: Literal["formula", "post_load", "pre_persist", "entity_listener",
                  "hibernate_filter", "audit", "other"]
    on: str                               # entity or field
    detail: str
    citation: Citation
```

Extend `EntityRelation` with two optional fields. **Facts only, no judgement** — whether
an already-batched relation still needs work is diagnosis's call, not comprehension's.
Do not add an `already_optimized` flag.

```python
    batch_size: int | None = None                              # @BatchSize
    fetch_mode: Literal["SELECT", "JOIN", "SUBSELECT"] | None = None   # @Fetch
```

### A5 · EndpointModel

```python
    per_row_work: list[PerRowWork] = []
    response_shape: ResponseShape | None = None
    pagination: PaginationInfo | None = None
    caching: list[CacheInfo] = []
    hidden_query_triggers: list[HiddenQueryTrigger] = []
    base_commit: str | None = None
```

`base_commit` is the git sha of the target repo at analysis time — filled by
deterministic code, never by the model. The oracle already records this; without it here,
there is no way to tell a stale artifact from a current one, and the orchestrator's
resume logic is guessing.

**Add one computed helper** — computed in Python, never counted by a model (invariant 5):

```python
    def unbounded_per_row_work(self) -> list[PerRowWork]:
        return [w for w in self.per_row_work if not w.bounded]
```

Extend `all_citations()` to cover the new citation-bearing types.

---

## Part B — The submit tool

Add the new shapes to `submit_endpoint_analysis`'s JSON schema, with the field
descriptions carrying their reasoning — the schema is read by the model and is part of
the prompt.

**Required in the payload:** `per_row_work` and `response_shape`. An endpoint with
genuinely no per-row work must submit an empty array explicitly, so that "none found" is
a statement rather than an omission.

**Optional:** `pagination`, `caching`, `hidden_query_triggers`.

**Never in the payload:** `base_commit` — the handler fills it from `git rev-parse HEAD`
in the source root. A model-supplied commit sha is a claim; a code-supplied one is a
fact.

### Validation additions in `understand/validate.py`

1. Every `PerRowWork` citation resolves, same rules as existing citations.
2. `bounded=True` requires a non-empty `bound`. Claiming a limit without naming it is
   not a claim — reject it with a message saying so.
3. If any `DataAccess` has `executes_per_row=True` **and** `per_row_work` is empty,
   return a *warning* on the artifact (not a rejection): a loop that makes a query per
   row almost always does other work per row too, and an empty list is suspicious. Put
   it in `uncertainties` rather than blocking.

That third rule is a heuristic, so it must not fail the submission. Say so in the code
comment.

---

## Part C — Prompt changes

`prompts/understand.system.md`.

### C1 · Delete the phantom tools

Lines 8-12 currently advertise `get_callees`, `get_callers` and `find_symbol` with
"(Not yet available; use search_code until they are.)" That text was mine and it was a
mistake — telling a model about tools it cannot call invites it to try. Remove all three
entries and renumber. They come back when the code index does.

### C2 · New section, after "What counts as database access"

```markdown
## Per-row work that is not a database access

A query that runs per row is one kind of per-row cost. It is often not the
expensive one.

Look for work inside any loop, stream, or .map over a result set:

- building strings, especially one per element of a collection
- iterating a collection to filter, sum or transform it
- creating objects per element
- computation repeated per row that could be done once
- anything appended to a list that is then returned in the response

For each one, answer the question that matters: **is the amount of work
bounded?**

- BOUNDED means something limits it — a page size, a hard cap, a fixed-size
  collection. Name the limit.
- UNBOUNDED means it scales with data and nothing caps it. A loop over
  `customer.getInvoices()` is unbounded: a customer with 30,000 invoices does
  30,000 iterations, and real data is skewed.

Unbounded per-row work is frequently the single largest cost in an endpoint,
and it is invisible in the SQL logs. Report it even when you are unsure of its
cost — say so in uncertainties instead of leaving it out.

## What the endpoint returns

Record the response DTO, the fields it returns, which getters do real work
rather than just returning a field, and — most importantly — which returned
collections are UNBOUNDED in size.

This matters because a later phase must know whether the response can be made
smaller. If a returned collection grows with the data, its size is part of the
API contract and cannot be trimmed without changing behaviour.

## Pagination

State the mechanism: offset-based, keyset, or none. Say whether a separate
count query fires, and whether the page size is capped. An uncapped page size
on an endpoint with unbounded per-row work is a compounding problem.
```

### C3 · One line in the finishing section

```markdown
`per_row_work` and `response_shape` are required. If an endpoint genuinely has
no per-row work, submit an empty array — that is a finding, not an omission.
```

Do **not** add prompt sections for caching, hidden query triggers, `@BatchSize` or
`@Fetch`. The schema descriptions are sufficient; the target codebase has none of them
and prompt weight spent on absent features is prompt weight taken from the ones that
matter.

---

## Part D — The four issues from Session 1 review

1. **`MySQLTarget.from_env()` in `understand/agent.py:83`.** The comment claims
   "constructed, not connected", but verify: with MySQL environment variables unset, the
   phase must still run and complete, with `get_schema` returning a readable error to
   the model and `tables` left empty. If `from_env()` raises, wrap it — the phase can do
   90% of its job without a database and must not die at the door. Add a test that
   proves it.

2. **`build_registry` docstring** claims tools are "ordered as the system prompt
   describes them", but `source` is added before `schema` while the prompt lists
   `get_schema` before `search_code`. Either fix the order or fix the docstring. A
   docstring making a claim the code does not keep is worse than no docstring.

3. **`.gitignore`** — add `Claude outputs/`.

4. **Deprecated `max_steps`** on `run_agent` — leave it, but add a `DeprecationWarning`
   so it does not survive indefinitely by accident.

---

## Part E — Rendering

`render/endpoint.py` must show the new fields, and show them in the order that reflects
what usually matters:

```
POST /api/customers/search                    base a3f91c2

CALL CHAIN
  CustomerController.search           CustomerController.java:19
  └ CustomerService.search            CustomerService.java:27
    ├ CustomerRepository.findByFilter  CustomerRepository.java:12   1 query/request
    └ CustomerService.enrich           CustomerService.java:36     per row

PER-ROW WORK
  ⚠ UNBOUNDED  builds one "overdue-{id}" string per invoice
               CustomerService.java:39 · scales with invoices per customer
    bounded    maps each Customer to a CustomerDto
               CustomerService.java:36 · scales with page size (max 100)

DATA ACCESS
  1×/request   findByFilter          jpql_query      customer
  per row      sumOpenAmountByCustomer  jpql_query   invoice
  per row      lazy load Customer.invoices           invoice

RESPONSE          CustomerDto
  unbounded       tags            ← size is part of the API contract
  computed        outstanding

PAGINATION        offset · count query fires · page size uncapped

TABLES
  customer   1 index (PRIMARY)      ~50,000 rows    reads 2 of 8 columns
  invoice    1 index (PRIMARY)   ~1,200,000 rows    reads 3 of 22 columns

UNCERTAINTIES
  · ...
```

The `⚠ UNBOUNDED` marker is the point of the whole session. A human skimming this output
should see the endpoint's real problem in the first five seconds.

---

## Quality bar

The Session 1 standing bar applies. In addition:

- No module over ~400 lines; `contracts.py` is approaching it — if it passes, split the
  comprehension shapes into `contracts/comprehension.py` with re-exports so no import
  path changes.
- Existing tests stay green. `EndpointModel` gains only optional fields and defaults, so
  every existing artifact must still load — add a test that loads the current
  `.senapathy/endpoint_model.json` unchanged.
- `docs/understand.md` updated with the new fields and what each is for.

---

## Acceptance criteria

The first is the session. The rest support it.

1. **On mini-spring, `per_row_work` is populated and contains the `tags` loop with
   `bounded=false` and a `scales_with` naming invoices per customer.** An empty
   `per_row_work` on this endpoint is a failed session — the field exists precisely
   because that loop is 93% of the cost. If the model will not fill it, iterate the
   prompt and tell me what it took.

2. `response_shape.unbounded_collections` contains `tags`.

3. `pagination.fires_count_query` is `true` — the baseline proves the count query runs.

4. `bounded=true` with no `bound` is rejected, and the message says what is missing.

5. `base_commit` matches `git rev-parse HEAD` in mini-spring, and is not settable from
   the payload.

6. The phase completes with MySQL environment variables unset: `tables` empty,
   `get_schema` error visible to the model, no crash.

7. The existing `endpoint_model.json` still loads under the new schema.

8. `senapathy understand --verbose` renders the new sections, with the unbounded marker
   visible.

## Report back

- Whether the model populated `per_row_work` honestly on the first attempt, and what the
  prompt iteration looked like if not. That is the real result of this session.
- Anything in this spec you think is wrong.
- Any field you added or removed against the spec, and why.