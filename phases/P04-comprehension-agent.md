# PHASE P04 — Comprehension agent

**Mode:** mixed — you write the prompt and the validator, Claude may write the
plumbing · **Est:** 4h · **Preconditions:** P00–P03 complete

---

## 1. GOAL

The first real agent: given a route string, explore a repo and emit a validated
`EndpointModel`. Ship it as `senapathy understand`.

## 2. WHY THIS PHASE

Milestone A ends here, and this is the first thing that's useful on its own — an
agent that maps any endpoint's full execution path is worth showing your team even
with no optimization attached.

It's also where three ideas become concrete: forced structured output as a stop
condition, mechanical validation of model claims, and the artifact-not-transcript
boundary between phases.

## 3. PRECONDITIONS

P02 loop, P03 read tools, `EndpointModel` in contracts, mini-spring fixture.

## 4. DELIVERABLES

```
senapathy/agents/__init__.py
senapathy/agents/comprehend.py          # prompt, emit tool, runner
senapathy/validate.py                   # citation validation
senapathy/cli.py                        # implement `understand`
tests/test_citation_validation.py       # from section 6
tests/test_comprehend.py                # from section 6
tests/cassettes/comprehend_mini/        # recorded, committed
```

## 5. SPEC

### 5.1 `emit_endpoint_model` — the terminal tool

Register with `phases={"P1"}`, `terminal=True`, and `input_schema` generated from the
Pydantic model:

```python
schema = EndpointModel.model_json_schema()
```

The handler:

1. Parses args into `EndpointModel` (Pydantic validation).
2. Runs `validate_citations(model, source_root)`.
3. On failure → `ToolError` listing each bad citation and what was found instead.
   The loop turns that into an `is_error` result and the agent gets to fix it.
4. On success → stash the model and return `"accepted"`.

**Termination is this tool being called successfully.** Not `end_turn`, not a step
count. This is the mechanism that guarantees the phase produces a typed artifact or
an honest failure — never prose you have to parse.

### 5.2 `validate.py` — mechanical checking of model claims

```python
class CitationProblem(BaseModel):
    citation: Citation
    problem: Literal["file_missing", "line_out_of_range", "symbol_not_on_line"]
    found_instead: str | None
    suggestion: str | None

def validate_citations(model: EndpointModel,
                       root: SourceRoot) -> list[CitationProblem]: ...
```

For every citation from `model.all_citations()`:

- File exists under the root? Else `file_missing`, suggest near-matching filenames.
- Line within the file's length? Else `line_out_of_range`, report the real length.
- If `symbol` is set, does that line (or the two lines either side, to allow for
  annotations and wrapped signatures) contain it? Else `symbol_not_on_line`, and
  include the actual line content plus the nearest line that *does* contain the
  symbol.

The tolerance window matters: a strict single-line check produces false rejections on
Java, where `@Transactional` sits above the method and long signatures wrap.

### 5.3 The system prompt

You write this. Requirements, not wording:

- State the job: trace one route to every piece of code that runs, and the data
  access it performs.
- Name the tools and when each is the right choice.
- Demand a `file:line` for every claim, and state that citations are machine-verified
  — so a guess will be rejected and cost a turn.
- Instruct it to maintain an explicit worklist of unresolved symbols and to keep
  going until nothing on the path is unresolved.
- Demand `uncertainties` be populated for anything ambiguous — especially interfaces
  with multiple implementations, reflection, conditional beans, AOP.
- Tell it to stop exploring when the path is closed, and that emitting the model is
  how it finishes.
- Tell it what *not* to explore: unrelated endpoints, framework internals, generated
  code, test code beyond identifying which tests cover this route.

That last bullet is load-bearing. Without a scope boundary the agent wanders into
Spring's own source and burns your budget.

### 5.4 The runner

```python
def comprehend(*, repo: Path, endpoint: str, llm: LLMClient,
               trace: Trace, budget: LoopBudget) -> EndpointModel: ...
```

Builds the registry (read tools + emit tool), runs the loop with `phase="P1"`, and:

- `COMPLETED` with terminal args → return the model, write the artifact.
- `COMPLETED` without terminal args (model gave up) → raise, with the final text.
- `BUDGET_EXHAUSTED` / `NO_PROGRESS` → raise with the partial transcript reference.

Note what the runner does *not* do: it does not pass the transcript onward. The
`EndpointModel` artifact is the entire output. The exploration may have cost 100k
tokens; the next phase receives ~800.

### 5.5 `senapathy understand`

```
senapathy understand --repo PATH --endpoint "POST /api/customers/search"
                     [--max-usd 2.0] [--json] [--offline]
```

Human-readable by default: the path as a tree, SQL sources, entities with fetch
types, uncertainties called out. `--json` dumps the artifact. `--offline` forces
cassette-only, for tests and demos.

## 6. ACCEPTANCE TEST

### `tests/test_citation_validation.py`

Pure functions, no model. This is where you prove the validator actually catches
hallucination.

```python
from __future__ import annotations
from pathlib import Path
from senapathy.contracts import Citation, EndpointModel
from senapathy.tools.read import SourceRoot
from senapathy.validate import validate_citations

FIX = Path(__file__).parent / "fixtures" / "mini-spring"
ROOT = SourceRoot(FIX)
CTRL = "src/main/java/com/ex/web/CustomerController.java"


def _model(cits):
    return EndpointModel(route="POST /api/customers/search",
                         entry=cits[0], path=cits)


def _line_of(rel, needle):
    for i, line in enumerate((FIX / rel).read_text().splitlines(), 1):
        if needle in line:
            return i
    raise AssertionError(needle)


def test_valid_citation_passes():
    ln = _line_of(CTRL, "class CustomerController")
    problems = validate_citations(
        _model([Citation(file=CTRL, line=ln, symbol="CustomerController")]), ROOT)
    assert problems == []


def test_missing_file_is_caught_and_suggested():
    p = validate_citations(
        _model([Citation(file="src/main/java/com/ex/web/Ctrl.java", line=1)]), ROOT)
    assert len(p) == 1 and p[0].problem == "file_missing"
    assert p[0].suggestion and "CustomerController" in p[0].suggestion


def test_line_out_of_range_is_caught_with_real_length():
    p = validate_citations(_model([Citation(file=CTRL, line=99_999)]), ROOT)
    assert p[0].problem == "line_out_of_range"
    assert p[0].found_instead is not None       # must state the real length


def test_symbol_not_on_line_is_caught():
    p = validate_citations(
        _model([Citation(file=CTRL, line=1, symbol="totallyMadeUpMethod")]), ROOT)
    assert p[0].problem == "symbol_not_on_line"


def test_symbol_within_tolerance_window_passes():
    """Annotations sit above methods; a strict single-line check false-rejects."""
    ln = _line_of(CTRL, "@PostMapping")
    p = validate_citations(
        _model([Citation(file=CTRL, line=ln + 1, symbol="@PostMapping")]), ROOT)
    assert p == []


def test_all_bad_citations_reported_not_just_first():
    """The agent should be able to fix everything in one retry."""
    p = validate_citations(_model([
        Citation(file="nope/A.java", line=1),
        Citation(file=CTRL, line=99_999),
        Citation(file=CTRL, line=1, symbol="nonsense"),
    ]), ROOT)
    assert len(p) == 3
```

### `tests/test_comprehend.py`

Runs the **whole agent** against the fixture with cassettes — deterministic, offline,
free. Record the cassettes once with a live run, then commit them.

```python
from __future__ import annotations
import pytest
from pathlib import Path
from senapathy.llm import AnthropicClient
from senapathy.agents.comprehend import comprehend
from senapathy.agent import LoopBudget
from senapathy.trace import Trace

FIX = Path(__file__).parent / "fixtures" / "mini-spring"
CASS = Path(__file__).parent / "cassettes" / "comprehend_mini"


@pytest.fixture
def offline_llm():
    return AnthropicClient(api_key="unused", model="claude-opus-4-5",
                           cassette_dir=CASS, offline=True)


def test_comprehend_finds_the_full_path(offline_llm, tmp_path):
    m = comprehend(repo=FIX, endpoint="POST /api/customers/search",
                   llm=offline_llm, trace=Trace(tmp_path, "t"),
                   budget=LoopBudget(max_steps=25))
    files = " ".join(c.file for c in m.all_citations())
    for expected in ("CustomerController", "CustomerService",
                     "InvoiceService", "CustomerRepository"):
        assert expected in files, f"missed {expected} on the path"


def test_comprehend_identifies_lazy_relation(offline_llm, tmp_path):
    """The planted N+1 depends on this being spotted."""
    m = comprehend(repo=FIX, endpoint="POST /api/customers/search",
                   llm=offline_llm, trace=Trace(tmp_path, "t"),
                   budget=LoopBudget(max_steps=25))
    lazy = [r for e in m.entities for r in e.relations if r.fetch == "LAZY"]
    assert any(r.field == "invoices" for r in lazy)


def test_comprehend_finds_sql_sources(offline_llm, tmp_path):
    m = comprehend(repo=FIX, endpoint="POST /api/customers/search",
                   llm=offline_llm, trace=Trace(tmp_path, "t"),
                   budget=LoopBudget(max_steps=25))
    assert m.sql_sources
    assert any("customer" in s.tables for s in m.sql_sources)


def test_every_citation_in_the_output_is_real(offline_llm, tmp_path):
    """The artifact contract: an accepted EndpointModel has zero bad citations."""
    from senapathy.validate import validate_citations
    from senapathy.tools.read import SourceRoot
    m = comprehend(repo=FIX, endpoint="POST /api/customers/search",
                   llm=offline_llm, trace=Trace(tmp_path, "t"),
                   budget=LoopBudget(max_steps=25))
    assert validate_citations(m, SourceRoot(FIX)) == []


def test_run_is_fully_traced(offline_llm, tmp_path):
    trace = Trace(tmp_path, "t")
    comprehend(repo=FIX, endpoint="POST /api/customers/search",
               llm=offline_llm, trace=trace, budget=LoopBudget(max_steps=25))
    events = trace.read_all()
    assert any(e["type"] == "TOOL_CALL" for e in events)
    assert any(e["type"] == "LLM_RESPONSE" for e in events)


@pytest.mark.live
def test_live_smoke(tmp_path):
    """Only run deliberately: pytest -m live. Costs money."""
    import os
    llm = AnthropicClient(api_key=os.environ["ANTHROPIC_API_KEY"],
                          model="claude-opus-4-5", cassette_dir=tmp_path / "c")
    m = comprehend(repo=FIX, endpoint="POST /api/customers/search", llm=llm,
                   trace=Trace(tmp_path, "t"), budget=LoopBudget(max_steps=25))
    assert m.route
```

Register the `live` marker in `pyproject.toml` and default-exclude it.
`pytest tests/ -q` must pass offline with no API key.

## 7. OUT OF SCOPE

No benchmarking, no database, no SQL execution, no Docker, no diagnosis, no patching.
No code index. No `run_probe`. Do not add fields to `EndpointModel`. Do not pass the
transcript out of the runner.

## 8. DEMO

```bash
senapathy understand --repo tests/fixtures/mini-spring \
                     --endpoint "POST /api/customers/search"
```

Then the real thing:

```bash
senapathy understand --repo ../crm-service \
                     --endpoint "POST /api/customers/search" --max-usd 3
```

Read the output next to the actual code. Where is it right? Where is it vague? Did it
populate `uncertainties` honestly, or claim certainty it shouldn't have?

## 9. LEARNING

**Concepts: forced structured output, mechanical verification of model claims, and
the artifact boundary.**

**Forced structured output as termination.** You didn't ask the model to stop and hope
it did. You gave it exactly one way to declare completion, and that way produces a
schema-valid object. Compare with the alternative — parsing prose for a conclusion —
and notice you've eliminated an entire class of failure rather than mitigating it.

**Verification beats trust.** The citation validator is a small thing that changes
the character of the system. The model can hallucinate freely; hallucinations don't
survive contact with the filesystem. That's the pattern to internalise:
**don't try to make the model not lie — make lies detectable.** It recurs at every
scale in this product, all the way up to the frozen oracle.

**The artifact boundary.** A 100k-token exploration becomes an 800-token artifact.
That's what keeps a twelve-phase run inside a context window, and it's why P09 can be
tested with a hand-written `EndpointModel` and no P04 at all. Phase independence
comes from artifacts, not from good intentions.

**Cassettes.** You just tested a non-deterministic system deterministically, offline,
for free. Almost nobody does this, and it's why most people's agent work is
untestable. Note the tradeoff you accepted: cassette tests verify *your plumbing*,
not the model's continued competence. That's what the `live` marker and, later, the
eval suite are for.

### Experiments to run

1. **Hand it a lie.** Temporarily make `emit_endpoint_model` skip validation, then
   feed the model a prompt encouraging speed over care ("be fast, don't re-read
   files"). Inspect the citations. Count the fabricated ones. Re-enable validation
   and watch the retry loop fix them. You now know the failure rate you're preventing.

2. **Break the scope boundary.** Remove the "don't explore framework internals" line
   from the prompt and run against the real repo. Watch it read Spring source. Note
   the token cost of one missing sentence.

3. **Compare `uncertainties` against ground truth.** On your real repo, is the
   `uncertainties` list honest? An agent that says "InvoiceService has two impls, I
   assumed the `@Primary` one" is doing something genuinely hard. If it's always
   empty, your prompt isn't asking properly.

4. **Cost accounting.** From the trace, compute dollars per endpoint understood.
   Write it down. In P18 you'll add the code index and cut it; that "before" number
   is the only thing that makes the improvement real.

5. **Vary the model.** Run with a cheap model. Does it still find the whole path?
   Where does it fail first? Model routing decisions should come from measurements
   like this, not vibes.

### Before moving to Milestone B, you should be able to answer

- Why is a forced terminal tool a better stop condition than a step limit or
  `end_turn`?
- What does citation validation catch that a better prompt cannot?
- Why does the next phase get the artifact instead of the transcript?
- What do cassette tests prove, and what do they *not* prove?
- Your agent produced a plausible but incomplete `EndpointModel` — it missed a
  service two hops down. Rank your options: better prompt, better tools, bigger
  budget, better validation. Justify the order.

---

## Milestone A complete

You have an agent you wrote yourself that reads an unfamiliar codebase and produces a
verified structural model of an endpoint. Ship `senapathy understand` to your team.

Next: **Milestone B — evidence, with zero AI in it.** Environment, workload,
benchmark harness. You cannot evaluate an agent's diagnosis without a trustworthy
measurement first, which is why the next four phases contain no model calls at all.
