# Session 1 — Comprehension phase, production-grade

Read `CLAUDE.md` first. Then execute everything below.

This session turns the throwaway comprehension script into a real phase, and fixes the
LLM client, the agent loop and the tool layer along the way. It is the foundation every
other phase sits on, so quality here matters more than speed.

**Scope boundary:** do not touch `diagnose/`, `patch/`, `oracle/`, `gate/`, `report/`,
`measure/` except where explicitly listed. Do not build a code index beyond the two
tools specified in Part C — the full symbol/call-graph index is a later session.

---

## Part A — LLM client: provider abstraction

`senapathy/llm.py` today is a class called `GroqClient` that posts to Cerebras, with a
hardcoded URL, no retries, and a `print()` on non-200 followed by parsing the error body
as a completion. Replace it.

**New layout:**

```
senapathy/llm/__init__.py        make_client() factory, re-exports
senapathy/llm/base.py            LLMClient protocol, shared retry/cost/cassette logic
senapathy/llm/openai_compat.py   Cerebras, Groq, Ollama — all OpenAI-shaped
senapathy/llm/anthropic.py       Anthropic Messages API
senapathy/llm/pricing.py         per-model token prices
```

**Requirements:**

1. **One factory, one switch.**
   ```python
   make_client(provider: str | None = None, model: str | None = None) -> LLMClient
   ```
   Resolution order: explicit argument → `senapathy.yml` → environment → built-in
   default. Provider names: `cerebras`, `groq`, `anthropic`, `ollama`.

2. **Keys resolved by the factory, per provider.** `CEREBRAS_API_KEY`, `GROQ_API_KEY`,
   `ANTHROPIC_API_KEY`; Ollama needs none. A missing key raises a clear error naming the
   variable and the provider. No key or URL literal outside `llm/`.

3. **Config.** Extend `senapathy/config.py`:
   ```yaml
   llm:
     provider: cerebras
     model: qwen-3-coder-480b
     max_tokens: 4096
     timeout_s: 120
     cassette_dir: .senapathy/cassettes
     offline: false
   ```
   Every phase may override the model — diagnosis and planning will want a stronger
   model than mechanical summarisation.

4. **`system` parameter** — the current client has none, so prompts and evidence travel
   together in the user message. Add it:
   ```python
   def complete(self, *, system: str | list[dict] | None = None,
                messages: list[dict], tools: list[dict] | None = None,
                tool_choice: dict | None = None,
                max_tokens: int | None = None) -> LLMResponse: ...
   ```
   OpenAI-shaped providers get it prepended as a `{"role": "system"}` message; Anthropic
   gets it as the top-level `system` field.

5. **Retries.** Retry on 429, 500, 502, 503, 529 and on timeout/connect errors. Honour
   `Retry-After`; otherwise exponential backoff with jitter, capped at 60s, max 5
   attempts. **Never retry 400 or 401** — those are our bug, and the API's own error
   message must be surfaced verbatim in the exception.

6. **Non-200 must not be parsed as success.** Today it prints and then calls `r.json()`
   anyway, producing `KeyError: 'choices'`. A non-200 that survives retries raises
   `SenapathyError` carrying status and the response body.

7. **Cost accounting.** Price table in `pricing.py`; per-call and cumulative
   `Usage.usd`. Anthropic's cache-read and cache-write tokens are priced separately.
   An unknown model logs a warning and prices at zero rather than crashing — but says so.

8. **Cassettes.** Key on `content_hash` of the full request body. Hit → return it with
   `from_cassette=True` and no HTTP. Miss with `offline=True` → raise naming the missing
   hash. Otherwise call, write the cassette, return. This is what makes agent tests
   deterministic and free.

9. **Prompt caching (Anthropic).** `cache_control: ephemeral` on the last system block
   and the last tool. The system prompt and tool schemas are identical across every turn
   of a loop; not caching them is the single biggest avoidable cost.

10. **Statelessness.** `complete()` is a pure function of its arguments. The current
    client stores `self.last_raw_message`, which is both stateful and duplicated with
    the field on `LLMResponse`. Remove the attribute; keep `LLMResponse.raw_message` as
    the single source of the provider-shaped assistant turn for the loop to echo back.

11. **Omit empty fields.** `"tools": null` is currently sent when there are no tools.
    Omit the key.

Keep `GroqClient` as a deprecated alias for one release so nothing breaks mid-refactor.

---

## Part B — Agent loop: make it reusable and safe

`senapathy/agent.py` has four problems. Fix all four while keeping its shape — this is
the file the human wrote and understands, so favour clarity over cleverness.

1. **`time.sleep(20)` on line 112 — delete it.** It was a Groq free-tier workaround and
   it costs ~5.7 minutes of an 8-minute run. Rate limiting belongs in the client's retry
   logic, where it can be conditional.

2. **Fix the latent crash.** Line 89 looks up `handlers[tool["name"]]` *outside* the
   `try`, then line 93 looks it up again inside. A hallucinated tool name raises
   `KeyError` at line 89, where the `except KeyError` cannot catch it, and kills the run.
   One lookup, inside the guard. Delete the dead `if callable(handler)` branch.

3. **Parameterise the terminal tool.** Line 86 hardcodes `emit_endpoint_model`, which is
   why the loop only works for one agent. Signature:
   ```python
   def run_agent(*, client, task, tools, handlers, system=None,
                 terminal_tool: str | None = None,
                 budget: LoopBudget | None = None,
                 trace: Trace | None = None) -> LoopOutcome: ...
   ```
   When the terminal tool's handler succeeds, return. When it raises `ToolError`, feed
   the error back as a tool result and continue looping — a rejected artifact must be
   retryable.

4. **Return a typed outcome, not `str | dict`:**
   ```python
   class LoopStatus(StrEnum):
       COMPLETED, BUDGET_EXHAUSTED, NO_PROGRESS, ERROR

   @dataclass
   class LoopOutcome:
       status: LoopStatus
       terminal_payload: dict | None
       final_text: str
       steps: int
       usage: Usage
       messages: list[dict]
   ```

**Also add:**

- **`LoopBudget`** — `max_steps`, `max_usd`, `max_wall_s`. Checked before each call.
  Exceeding one is a normal outcome with partial results, never an exception. Treat
  `None` as unlimited and `0` as zero (the orchestrator already hit that bug once).
- **No-progress detection** — hash `(tool_name, canonical_json(args))`; three identical
  dispatches in a row ends the loop with `NO_PROGRESS`.
- **Trace events instead of print.** `LLM_REQUEST`, `LLM_RESPONSE`, `TOOL_CALL`,
  `TOOL_RESULT`, `GUARDRAIL_TRIGGERED`, with tokens, cost, duration and result size.
  Move the transcript rendering out of `agent.py` into `senapathy/render/transcript.py`,
  driven by trace events and enabled by `--verbose`. The pretty output is genuinely
  useful for learning — keep it, just don't put it in library code.
- **Guardrail events are distinct from errors.** A `ToolError` raised by a *containment*
  check (path escape, denied write) emits `GUARDRAIL_TRIGGERED`. Today both arrive as
  `except Exception` and are indistinguishable in the trace.

---

## Part C — Tools

`senapathy/tools.py` becomes a package. `REPO_ROOT` as a module constant is the single
reason the product cannot run on another repo.

```
senapathy/tools/__init__.py      registry, ToolSpec
senapathy/tools/source.py        read_file, search_code, list_dir
senapathy/tools/routes.py        find_route          ← new
senapathy/tools/schema.py        get_schema          ← new
senapathy/tools/submit.py        submit_endpoint_analysis
```

**1. Inject the root.** A `SourceRoot` object constructed per run and passed to the tool
factory. No module-level path. Two repos must be usable in one process, and tests must
work against fixtures.

**2. `_resolve` raises `ToolError`, not `ValueError`,** and containment failures emit
`GUARDRAIL_TRIGGERED`. Keep the current logic — `resolve()` before the containment check
is correct and the reasoning in the comment is right.

**3. `search_code` — reconsider `--max-count 5`.** It currently caps at 5 matches *per
file*, which silently hides matches and is probably not what was wanted. Make it a
parameter defaulting to 3, and state the per-file cap in the returned text so the model
knows results were limited per file rather than overall.

**4. `find_route(method: str, path: str)` — new.**

Parse Java controllers with tree-sitter (`tree-sitter-java`) and build a route table by
composing class-level `@RequestMapping` with method-level
`@GetMapping`/`@PostMapping`/`@PutMapping`/`@DeleteMapping`/`@PatchMapping`/
`@RequestMapping(method=...)`.

- Handle `value=`, `path=`, and bare-string forms; handle multiple paths on one mapping
- Match path templates: `/api/customers/{id}` matches a request for that template, and
  also matches a concrete `/api/customers/2` — report which interpretation was used
- Return: controller class, method name, `file:line`, the composed path, the HTTP method,
  parameter types, and the return type
- Ambiguous match → return all candidates with a note, do not pick one silently
- No match → an error listing the closest routes found, so the model can correct itself
- Cache the route table per repo, invalidated on the mtime of any controller file

This tool exists to eliminate the search-and-guess step that currently opens every run.

**5. `get_schema(tables: list[str])` — new.**

Introspect the live MySQL database (connection from config, same as `measure/sql.py`):

- Columns with types and nullability
- **Existing indexes** — name, columns in order, uniqueness, cardinality
- Approximate row count from `information_schema.TABLES`
- Foreign keys in and out

Returning existing indexes is the point: Session 3 must be able to tell a missing index
from one that already exists, and must never propose a duplicate.

If the database is unreachable, return a clear error saying so — do not fabricate, and
do not fall back to guessing from entity annotations.

**6. `submit_endpoint_analysis` — renamed, and it validates.**

Rename from `emit_endpoint_model`. "Submit" is honest: the call can be rejected.

The handler must:

1. Parse the payload into the Pydantic `EndpointModel` from `contracts.py`. A schema
   violation is a `ToolError` quoting Pydantic's own message.
2. Validate **every** citation:
   - the file exists under the source root
   - the line is within the file
   - if a symbol is given, it appears on that line **or within 2 lines either side** —
     Java puts annotations above methods and wraps long signatures, so a strict
     single-line check produces false rejections
3. On failure, raise `ToolError` listing **every** problem at once, each with what was
   found instead and, where possible, the nearest line that does contain the symbol.
   One retry should be able to fix everything.
4. On success, store the validated model and return a short confirmation.

Add `senapathy/understand/validate.py` with `validate_citations(model, root) ->
list[CitationProblem]`, tested independently of any model call.

---

## Part D — Prompts as data

Prompts will be tuned dozens of times. They must not be Python string literals.

```
prompts/understand.system.md
prompts/understand.user.md
```

Loader in `senapathy/prompts.py`: reads the file, substitutes `{placeholders}`, and
**records the prompt file's content hash in the trace**, so a run is reproducible and you
can tell which prompt version produced which artifact. A missing placeholder is an error,
not a silent blank.

### `prompts/understand.system.md` — use this content

```markdown
You trace the complete execution path of one HTTP endpoint through a Spring Boot
codebase, and report what you found as structured data.

## Tools, in order of preference

1. find_route — start here. Give it the method and path; it returns the exact
   controller method, with class-level and method-level mappings already composed.
2. get_callees / get_callers — follow the path. These give exact call edges.
   Prefer them over searching. (Not yet available; use search_code until they are.)
3. find_symbol — resolve a type to its declaration. Use it when an interface has
   multiple implementations; it reports annotations like @Primary.
   (Not yet available; use search_code until it is.)
4. read_file — read actual logic. Use it when you need to see what a method *does*,
   not just that it exists: loops, conditionals, field access, string building.
5. get_schema — tables, columns and EXISTING INDEXES for every table on the path.
   Call it for every table you touch. Knowing which indexes already exist matters
   as much as knowing which are missing.
6. search_code — fallback, for text the index cannot model: string literals, SQL in
   comments, configuration values.

## What counts as database access

EVERY interaction with the database, not only explicit queries. Include:

- repository methods and @Query annotations
- lazy loading of a collection or association, at the point where code touches it
- pagination count queries
- anything triggered inside a loop, a stream, or a .map over a result set

A lazy load inside a per-row loop is one of the most important things you can find.
Implicit queries are not out of scope — they are frequently the whole problem.

For each one, decide: does it execute ONCE per request, or ONCE PER RESULT ROW?
Say which. If you cannot tell from the code, put it in uncertainties rather than
guessing.

## Citations

Every claim carries a file path and line number. These are verified against the
filesystem after you submit — a guessed line number will be rejected and cost you a
turn. Re-read the file if you are unsure.

## Scope

Trace only this endpoint. Do not explore:
- other endpoints or unrelated controllers
- framework or library source (Spring, Hibernate, Jackson)
- generated code
- test code, beyond identifying which tests cover this route

## Finishing

Call submit_endpoint_analysis. That call is your entire output — do not write a prose
summary. If your submission is rejected, fix exactly what the error names and submit
again.
```

### `prompts/understand.user.md` — use this content

```markdown
Repository: {repo_root}
Stack: {stack}
Endpoint: {endpoint}

Trace it end to end and submit your analysis.
```

Note what was removed from the old prompt and why — put this in the git commit message:
the phrase *"down to the SQL or database access"* caused the model to classify
lazy-loaded per-row queries as out of scope, which hid the endpoint's real bottleneck on
the first run.

---

## Part E — Contract change: schema on the EndpointModel

Session 3 must reason about indexes. It can only do that if this phase captures them.
This is a deliberate, reviewed addition to `contracts.py` — the only one this session.

```python
class IndexInfo(BaseModel):
    name: str
    columns: list[str]          # in index order — order matters for composites
    unique: bool
    cardinality: int | None = None


class TableInfo(BaseModel):
    table: str
    entity: str | None = None
    columns_read: list[str] = []      # columns this endpoint actually reads
    all_columns: list[str] = []
    existing_indexes: list[IndexInfo] = []
    approx_rows: int | None = None
    foreign_keys: list[dict[str, str]] = []
```

Add to `EndpointModel`:

```python
tables: list[TableInfo] = []
prompt_hash: str | None = None      # which prompt version produced this
```

`columns_read` is the field that makes over-fetching detectable later — an entity with 30
columns where the endpoint reads 3 is a projection opportunity, and nothing downstream
can see that unless this phase records it.

Keep everything else in `EndpointModel` as it is. Update `tests/test_contracts.py`.

---

## Part F — The phase itself

```
senapathy/understand/__init__.py
senapathy/understand/agent.py     the agent: builds tools, runs the loop
senapathy/understand/validate.py  citation validation
senapathy/understand/cli.py       the understand command
```

**1. The runner:**

```python
def understand(*, repo: Path, endpoint: str, client: LLMClient,
               trace: Trace, budget: LoopBudget,
               run_dir: Path) -> EndpointModel: ...
```

- builds a `SourceRoot` from `repo` and a tool registry bound to it
- loads the prompts, substitutes `repo_root`, `stack`, `endpoint`
- runs the loop with `terminal_tool="submit_endpoint_analysis"`
- `COMPLETED` with a payload → write `endpoint_model.json` to the run directory, return
  the model
- `COMPLETED` without one (the model gave up) → raise, including its final text
- `BUDGET_EXHAUSTED` / `NO_PROGRESS` → raise with the trace reference

**Do not pass the transcript out of this function.** The `EndpointModel` is the entire
output of the phase. That boundary is what keeps a ten-phase run inside a context window.

**2. Replace the CLI stub.** `cli.py:17-20` currently takes no arguments and raises
`NotImplementedError`. Implement it in `understand/cli.py` and register it in `cli.py`
the same way `baseline` is registered — as an imported function, not defined inline.
Standardise on that pattern; this session moves one command, later sessions move theirs.

```
senapathy understand --repo PATH --endpoint "POST /api/customers/search"
                     [--provider cerebras] [--model X] [--max-usd 2.0]
                     [--max-steps 30] [--offline] [--verbose] [--json]
```

Human-readable by default: the call chain as a tree, data accesses with per-row flags
called out, tables with their existing indexes, uncertainties listed plainly. `--json`
dumps the artifact.

**3. Wire the orchestrator.** This is the bug that made the whole flag decorative:
`senapathy run --endpoint "..."` accepts the value, the orchestrator carries it, and the
understand phase never receives it — it only checks whether an artifact file exists and
skips.

Fix so the endpoint string reaches the phase. And make the skip condition honest: skip
only if an artifact exists **whose `route` matches the requested endpoint**. A cached
model for a different endpoint must not satisfy the phase. The orchestrator already
learned this lesson once when a stub `{"phase": "understand"}` passed a
parseability-only validator.

---

## Part G — Cleanup

- Delete `senapathy/tryllm.py`, `senapathy/test.py`, `senapathy/Untitled`
- Move the transcript printing out of `agent.py` per Part B
- `.senapathy/endpoint_model.json` carries a `PROVENANCE_WARNING` from a hand-rebuild.
  Once `senapathy understand` works, regenerate it for real and drop the warning.

---

## Quality bar

- No module over ~400 lines. Split by responsibility.
- Full type annotations; `mypy --strict` clean on everything touched.
- Artifacts are Pydantic models from `contracts.py`; no shapes invented inline.
- `ToolError` for expected failures. Tool handlers never raise into the loop. Error text
  is written for the model to act on, with a concrete next step.
- No `print()` in library code. Trace events; rendering in `render/`.
- Tests: no network, no database. Use the cassette mechanism and a fixture repo. The
  whole suite stays under 30 seconds. Existing 300 tests must remain green.
- `docs/understand.md`: what the phase does, inputs, outputs, how to run it, its three
  most common failure modes.

---

## Acceptance criteria

Verify each yourself and show me the evidence.

1. `senapathy understand --repo ../mini-spring --endpoint "POST /api/customers/search"`
   produces a validated `EndpointModel` including `tables` with existing indexes.
2. The same command with a **different** endpoint in mini-spring also works — proving the
   endpoint genuinely flows from the CLI and nothing is hardcoded.
3. `find_route` resolves a route whose path is split across a class-level
   `@RequestMapping` and a method-level mapping, and reports the composed path.
4. `get_schema` returns the real indexes on `customer` and `invoice` from the live MySQL
   on port 3307.
5. A deliberately bad citation — patch the payload in a test — is rejected, **all**
   problems are listed at once, and the agent's retry path is exercised.
6. A path-escape attempt (`../../etc/passwd`) is refused, and a `GUARDRAIL_TRIGGERED`
   event appears in the trace, distinguishable from an ordinary error.
7. Switching provider changes nothing but config: run once against `cerebras`, once
   against `ollama` (or a cassette-backed `anthropic`), with no code edit.
8. `--offline` runs entirely from cassettes with no network.
9. `senapathy run --endpoint "..."` reaches the understand phase **with that endpoint**,
   and does not skip on a cached artifact for a different route.
10. `time.sleep` no longer appears in `agent.py`, and a run that previously took ~8
    minutes is materially faster. Report the before and after wall clock.

## Report back

- What you changed, file by file
- Anything in this spec you think is wrong, with your reasoning — you have caught real
  design errors before and I would rather hear it than have it quietly implemented
- Any invariant in `CLAUDE.md` this work came close to weakening
- The wall-clock delta from removing the sleep