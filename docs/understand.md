# `senapathy understand` — the comprehension phase

Takes one HTTP endpoint and a Spring Boot repository, and produces a validated
`EndpointModel`: the execution path, every database access with a per-request /
per-result-row classification, **the per-row work that is not a database
access**, what the endpoint returns and which parts of it are unbounded, how it
pages, the tables on that path with the indexes that already exist on them, and
whatever the agent could not determine.

It is the first phase of the pipeline and the only useful thing here that
stands on its own — an endpoint map is worth showing a team with no
optimisation attached.

---

## What it does

1. Builds a `SourceRoot` for the repository and a tool registry bound to it.
2. Loads `prompts/understand.system.md` and `prompts/understand.user.md`,
   substitutes `repo_root`, `stack` and `endpoint`, and records both files'
   content hashes in the trace.
3. Runs the agent loop with `terminal_tool="submit_endpoint_analysis"`.
4. On a successful submission, validates every citation against the
   filesystem, merges the *measured* table facts in, writes
   `endpoint_model.json` to the run directory, and returns the model.

The transcript does not leave the phase. The `EndpointModel` is the entire
output — that boundary is what keeps a ten-phase run inside a context window.

## Tools it offers

| tool | what it is for |
|---|---|
| `find_route` | Resolve `METHOD path` to the controller method. Class-level `@RequestMapping` and method-level `@GetMapping`/`@PostMapping`/… are composed for you, from the AST rather than from grep. |
| `read_file` | 400 numbered lines at a time, with an explicit "call me again with `start_line=N`" when truncated. |
| `search_code` | ripgrep, capped per file, with the cap stated in the output. `--sort path`, so the same search always returns the same order. |
| `list_dir` | For when the layout is unknown. |
| `get_schema` | Columns, approximate row counts, foreign keys and **the indexes that already exist**, read live from MySQL. |
| `submit_endpoint_analysis` | The terminal tool. It can and does reject. |

There is no shell tool and nothing that could become one.

## Inputs

```
senapathy understand --repo PATH --endpoint "POST /api/customers/search"
                     [--provider cerebras] [--model X]
                     [--max-usd 2.0] [--max-steps 30] [--max-wall S]
                     [--offline] [--verbose] [--json]
                     [--run-dir DIR] [--runs-dir DIR]
```

Provider and model default to `senapathy.yml`:

```yaml
llm:
  provider: cerebras
  model: gpt-oss-120b
  max_tokens: 8192
  timeout_s: 120
  cassette_dir: .senapathy/cassettes
  offline: false
```

Everything in that block can be overridden by `SENAPATHY_LLM_*` environment
variables, and those by explicit flags. Keys are per provider and resolved by
the factory: `CEREBRAS_API_KEY`, `GROQ_API_KEY`, `ANTHROPIC_API_KEY`; Ollama
needs none.

The database connection is the same one `measure/sql.py` uses:
`SENAPATHY_MYSQL_HOST` / `_PORT` / `_USER` / `_PASSWORD` / `_DATABASE`.

## Outputs

- `<run-dir>/endpoint_model.json` — the artifact, an `EndpointModel`
- `<run-dir>/trace.jsonl` — every LLM request/response, tool call, tool result
  and guardrail event, with tokens, cost and duration

`--verbose` renders the conversation live from those same trace events, so
what you watch and what you read back cannot disagree.

## The fields, and what each is for

| field | what it is for |
|---|---|
| `call_chain` | The methods executed, in order. Order of execution, not nesting depth — the artifact does not record depth and the renderer does not pretend it does. |
| `data_access` | Every way this endpoint reaches the database, each marked per-request or per-result-row. |
| `per_row_work` | **The one that matters.** Work that runs once per result row and is NOT a database access: string building, collection iteration, object creation, repeated computation. Each item says what it `scales_with` and whether it is `bounded`. On the endpoint this product was built against, an unbounded loop building one string per invoice was 93% of the latency and appeared in no other field. |
| `response_shape` | The DTO, its fields, which are computed, and which returned collections are **unbounded**. A collection whose size grows with the data is part of the API contract and cannot be trimmed without changing behaviour — a later phase needs to know that before it writes a patch, not after four build iterations. |
| `pagination` | Mechanism (offset / keyset / none / unbounded), whether a count query fires, whether the page size is capped. An uncapped page size on top of unbounded per-row work is a compounding problem. |
| `tables` | Columns, approximate row counts, foreign keys and the indexes that already exist. Filled from the database, never from the model. |
| `caching`, `hidden_query_triggers` | Usually empty. Recorded anyway: an empty list is honest, a missing field is a blind spot. |
| `entities[].relations[].batch_size` / `.fetch_mode` | `@BatchSize` and `@Fetch` as **facts**. Whether an already-batched relation still needs work is diagnosis's call, so there is no `already_optimized` flag. |
| `base_commit` | The target repo's git sha at analysis time, with `+dirty` when the worktree is not clean. Filled by code from `git rev-parse HEAD`; a model-supplied sha is discarded and the model is told so. Without it there is no way to tell a stale artifact from a current one. |
| `prompt_hash` | Which prompt version produced this. |
| `uncertainties` | What the agent could not determine — plus any `AUTOMATED CHECK` lines the handler added. |

`per_row_work` and `response_shape` are **required**. An endpoint with genuinely
no per-row work submits an empty array, so that "none found" is a statement the
model had to make rather than a field it left out. The requirement is enforced
in the handler, not only in the JSON schema, because `required` in a tool schema
is a request that some providers honour and some ignore.

## What is checked, and by what

| claim | checked by |
|---|---|
| the payload's shape | Pydantic, against `EndpointModel` |
| every file cited exists | `understand/validate.py` |
| every line is inside its file | `understand/validate.py` |
| every symbol is within 2 lines of the cited line | `understand/validate.py` |
| how many accesses run per row | `EndpointModel.per_row_accesses()`, in Python |
| how much per-row work is unbounded | `EndpointModel.unbounded_per_row_work()`, in Python |
| `bounded=true` names its `bound` | `validate_claims()` — a rejection |
| per-row queries with empty `per_row_work` | `completeness_warnings()` — a note in `uncertainties`, never a rejection |
| column types, row counts, index cardinalities | `information_schema`, never the model |
| the base commit | `git rev-parse HEAD`, never the model |

A rejection lists **every** problem at once, with what was found on the cited
line and the nearest line that really does contain the symbol, so one retry
can fix everything. The agent gets the rejection as a tool result and keeps
going; rejection is not the end of the phase.

The model supplies judgement — which table, which entity, which columns this
endpoint reads. Everything measurable is templated in from the database.

## Cassettes

Every request is keyed on the content hash of the full body (system prompt,
messages, tool schemas) and recorded under `llm.cassette_dir`. A hit replays
with no HTTP and `from_cassette=True`; `--offline` turns a miss into an error
that names the missing hash.

Record and replay one scenario into its own directory, not the shared one:

```
SENAPATHY_LLM_CASSETTE_DIR=.senapathy/cassettes/search \
  senapathy understand --repo ../mini-spring --endpoint "POST /api/customers/search"

SENAPATHY_LLM_CASSETTE_DIR=.senapathy/cassettes/search \
  senapathy understand --repo ../mini-spring --endpoint "POST /api/customers/search" --offline
```

A shared directory holding cassettes from several runs of a
non-deterministic model replays a path that never happened: each individual
request hits, but the walk through them is a mixture of runs and eventually
reaches a history nobody recorded.

## The three most common failure modes

**1. The model writes its analysis as prose instead of calling
`submit_endpoint_analysis`.**
Weaker models do this routinely. The loop notices, appends a pointed
instruction, and re-asks with the provider's own forced-tool-choice mechanism
— up to three times. If it still refuses, the phase raises and includes the
model's final text so you can see what it wanted to say. Fixes: a stronger
model, or `--max-steps` room for the forced retries.

**2. Citations are rejected, repeatedly.**
Normal on the first submission and usually fixed on the second. If it loops,
look at the rejection in the trace: `'sumOpenAmountByCustomer' does not appear
within 2 lines of line 10` means the model cited the interface declaration
instead of the method. Repeated identical submissions end the loop with
`NO_PROGRESS` rather than burning the budget.

**3. `get_schema` cannot reach the database.**
It raises a `ToolError` naming the host, port and user. It does **not** fall
back to guessing indexes from `@Table` annotations — a guessed index list is
worse than none, because everything downstream would treat it as measured.
The phase can still complete without it; `tables` will simply carry no
measured facts. Start the database (`docker compose up -d mysql` in the target
repo) and re-run.

If no target can even be *described* — a garbled `SENAPATHY_MYSQL_PORT`, say —
the tool is not registered at all, a note goes into the trace, and the phase
runs without it. The phase does 90% of its job from source alone and must not
die at the door.

**A fourth: `per_row_work` comes back empty on an endpoint that clearly has
some.** The handler does not reject it — the check is a heuristic and an agent
right about an unusual endpoint must not be blocked by a rule of thumb — but it
appends an `AUTOMATED CHECK` line to `uncertainties` and the renderer prints
"treat that as suspect". If it happens repeatedly, the prompt needs work, not
the code.

A fifth, rarer one: a reasoning model can spend its entire output token budget
thinking and return nothing. That is a truncated reply, not a finished one; the
loop says so and re-asks, and gives up with an error naming `llm.max_tokens`
after three in a row.

## Where it sits in the pipeline

`senapathy run --endpoint "..."` runs this phase first. The phase is skipped
only if `<run-dir>/endpoint_model.json` exists **and its `route` matches the
requested endpoint** — a cached model for a different route does not satisfy
it. The diagnosis phase then reads that same file, not a global one.

Note that the standalone `senapathy diagnose` command still defaults to
`.senapathy/endpoint_model.json`; pass `--endpoint-model <run-dir>/endpoint_model.json`
to point it at a specific run.
