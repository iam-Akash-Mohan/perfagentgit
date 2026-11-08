# PHASE P00 — Scaffold, contracts, trace, artifact store

**Mode:** generate · **Est:** 2h · **Preconditions:** none

---

## 1. GOAL

Create the repo skeleton and **freeze every artifact schema and protocol** so that
all later phases plug into stable types.

## 2. WHY THIS PHASE

Artifacts are the interface between phases (design doc §13.1). If `EvidenceBundle`
gains a field in P09, every phase from P04 onward needs rework. Defining all of them
once, first, is what makes the phases independent and lets a phase be tested in
isolation.

The trace and artifact store come now for the same reason: every later phase writes
to them, so they must exist before anything else does.

## 3. PRECONDITIONS

Python 3.12 available. Nothing else.

## 4. DELIVERABLES

```
pyproject.toml
CLAUDE.md                          # already provided — do not modify
senapathy/__init__.py
senapathy/contracts.py             # ← the whole point of this phase
senapathy/config.py
senapathy/ids.py
senapathy/artifacts.py
senapathy/trace.py
senapathy/errors.py
senapathy/cli.py
tests/__init__.py
tests/test_contracts.py            # from section 6
tests/test_artifacts.py            # from section 6
```

## 5. SPEC

### 5.1 `pyproject.toml`

Package `senapathy`. Deps: `pydantic>=2.7`, `typer>=0.12`, `httpx>=0.27`,
`python-ulid`, `numpy`, `scipy`, `pyyaml`, `mysql-connector-python`.
Dev deps: `pytest`, `pytest-cov`. Console script `senapathy = senapathy.cli:app`.

### 5.2 `senapathy/errors.py`

```python
class SenapathyError(Exception): ...
class ToolError(SenapathyError):
    """Expected tool failure. The agent loop turns this into an is_error
    tool_result and continues. Message must be actionable for the model."""
    def __init__(self, message: str, suggestion: str | None = None): ...
class BudgetExhausted(SenapathyError): ...
class ContractViolation(SenapathyError):
    """An artifact failed validation, or a guardrail was breached."""
```

### 5.3 `senapathy/ids.py`

`new_run_id() -> str` returning a ULID string. `content_hash(obj) -> str` returning
`sha256` of canonical JSON (sorted keys, no whitespace), first 16 hex chars.

### 5.4 `senapathy/config.py`

Pydantic `Settings` loaded from env then optional `senapathy.yml`:
`anthropic_api_key`, `model` (default `claude-opus-4-5`), `cheap_model`,
`runs_dir` (default `.senapathy/runs`), `cassette_dir`, `max_usd_per_run` (default
`5.0`), `offline` (bool — when true, LLM calls must hit a cassette or raise).

### 5.5 `senapathy/contracts.py`

**This is the deliverable.** Write it exactly as given below. Later phases import
from here and must never redefine these shapes.

```python
from __future__ import annotations
from enum import Enum
from typing import Any, Literal, Protocol, runtime_checkable
from pydantic import BaseModel, Field


# ─────────────────────────────  primitives  ─────────────────────────────

class Citation(BaseModel):
    """A pointer into source. Every claim an agent makes carries one of these,
    and every one is validated against the filesystem before acceptance."""
    file: str
    line: int
    symbol: str | None = None

    def ref(self) -> str:
        return f"{self.file}:{self.line}"


class ArtifactRef(BaseModel):
    """Reference to a stored artifact. The evidence ledger works by requiring
    these on every quantitative claim."""
    kind: str
    hash: str
    path: str


# ─────────────────────────  P04 comprehension  ──────────────────────────

class SqlSource(BaseModel):
    method: str
    kind: Literal["@Query JPQL", "@Query native", "derived query",
                  "lazy collection", "criteria", "jdbc template", "other"]
    tables: list[str] = []
    citation: Citation


class EntityRelation(BaseModel):
    field: str
    type: Literal["OneToMany", "ManyToOne", "OneToOne", "ManyToMany"]
    fetch: Literal["LAZY", "EAGER"]
    target: str


class EntityInfo(BaseModel):
    name: str
    table: str
    citation: Citation
    relations: list[EntityRelation] = []


class TransactionInfo(BaseModel):
    method: str
    propagation: str
    read_only: bool = False
    citation: Citation


class ExternalCall(BaseModel):
    method: str
    target: str
    kind: Literal["http", "grpc", "queue", "other"]
    citation: Citation


class EndpointModel(BaseModel):
    route: str
    entry: Citation
    path: list[Citation]
    sql_sources: list[SqlSource] = []
    entities: list[EntityInfo] = []
    transactions: list[TransactionInfo] = []
    external_calls: list[ExternalCall] = []
    tests_covering: list[str] = []
    uncertainties: list[str] = Field(
        default_factory=list,
        description="What the agent was NOT sure about. An agent forced to "
                    "declare uncertainty is more useful than a confident one.")

    def all_citations(self) -> list[Citation]: ...


# ────────────────────────────  P06 workload  ─────────────────────────────

class RequestSpec(BaseModel):
    id: str
    method: str
    path: str
    headers: dict[str, str] = {}
    body: dict[str, Any] | None = None
    note: str | None = None


class RequestCorpus(BaseModel):
    benchmark: list[RequestSpec]
    held_back: list[RequestSpec] = Field(
        default_factory=list,
        description="Never used for measurement. Only for final equivalence. "
                    "This is the train/test split that catches benchmark gaming.")
    hash: str


# ─────────────────────────────  P13 oracle  ──────────────────────────────

class RecordedResponse(BaseModel):
    request_id: str
    status: int
    body: Any
    headers_of_interest: dict[str, str] = {}
    db_state_hash: str | None = None
    external_calls_made: list[dict[str, Any]] = []


class TolerancePolicy(BaseModel):
    """Written BEFORE the patch exists. This is the frozen part."""
    ignore_paths: list[str] = []          # JSONPath-ish: fields to skip entirely
    order_insensitive_paths: list[str] = []
    numeric_epsilon: float = 0.0
    rationale: dict[str, str] = {}        # path -> why it's tolerated


class BehaviourOracle(BaseModel):
    corpus_hash: str
    recorded: list[RecordedResponse]
    policy: TolerancePolicy
    base_commit: str
    hash: str                              # content hash of everything above


class FieldDiff(BaseModel):
    request_id: str
    path: str
    before: Any
    after: Any


class EquivalenceVerdict(BaseModel):
    passed: bool
    compared: int
    identical: int
    diffs: list[FieldDiff] = []
    db_state_matched: bool
    external_calls_matched: bool
    held_back_passed: bool
    oracle_hash: str
    decided_by: Literal["code"] = "code"   # never "model"


# ─────────────────────  P07/P08 evidence & baseline  ────────────────────

class LatencyStats(BaseModel):
    n: int
    p50_ms: float
    p95_ms: float
    p99_ms: float
    mean_ms: float
    stdev_ms: float


class SqlDigestStat(BaseModel):
    digest: str
    sample_sql: str
    calls: int
    total_ms: float
    mean_ms: float
    rows_examined: int = 0
    rows_sent: int = 0
    no_index_used: bool = False
    sample_params: list[str] = []


class ProfileFrame(BaseModel):
    frame: str
    self_pct: float


class QueryPlan(BaseModel):
    digest: str
    explain_analyze: str
    est_cost: float | None = None


class RedFlagKind(str, Enum):
    REPEATED_DIGEST = "repeated_digest"
    NO_INDEX_USED = "no_index_used"
    FULL_SCAN_LARGE_TABLE = "full_scan_large_table"
    DATA_ACCESS_DOMINATES = "data_access_dominates"
    HIGH_ROWS_EXAMINED_RATIO = "high_rows_examined_ratio"
    SERIALIZATION_DOMINATES = "serialization_dominates"
    EXTERNAL_CALL_FANOUT = "external_call_fanout"
    ALLOCATION_PRESSURE = "allocation_pressure"


class RedFlag(BaseModel):
    """COMPUTED by Python, never inferred by a model. The model is told the
    count; it is never asked to count."""
    kind: RedFlagKind
    detail: str
    magnitude: float
    evidence: list[str]


class EvidenceBundle(BaseModel):
    latency: LatencyStats
    sql: list[SqlDigestStat]
    profile: list[ProfileFrame]
    plans: list[QueryPlan] = []
    red_flags: list[RedFlag] = []
    total_sql_ms: float
    sql_share_of_latency: float
    variance_acceptable: bool
    raw_refs: list[ArtifactRef] = []

    def render_for_model(self) -> str:
        """The ONLY thing a model ever sees of this. Digested, capped, no raw logs."""
        ...


class BaselineReport(BaseModel):
    arm: Literal["blue", "green"]
    latency: LatencyStats
    sql_query_count: int
    evidence: EvidenceBundle
    dataset_hash: str
    corpus_hash: str
    commit: str


# ────────────────────────────  P09 diagnosis  ────────────────────────────

class Probe(BaseModel):
    tool: str
    args: dict[str, Any]
    discriminates_between: list[str] = Field(
        ...,
        description="Hypothesis IDs this probe separates. A probe that cannot "
                    "separate anything is not an experiment.")
    result: str


class Hypothesis(BaseModel):
    id: str
    claim: str
    mechanism: str
    evidence: list[str]                    # must resolve to real artifact refs
    attributed_ms: float
    prediction: str
    confidence: float = Field(ge=0.0, le=1.0)


class HypothesisSet(BaseModel):
    hypotheses: list[Hypothesis] = Field(min_length=3)   # forced competition
    probes_run: list[Probe] = []
    ranked_ids: list[str]
    measured_total_ms: float

    def attribution_valid(self) -> bool:
        """Sum of attributed_ms must not exceed measured latency. Checked in
        Python — the model does not get to do arithmetic."""
        ...


# ──────────────────────────  P11 plan & risk  ────────────────────────────

class RiskTier(int, Enum):
    LOW = 1          # index, @EntityGraph, readOnly tx, view narrowing
    MEDIUM = 2       # query rewrite, batching, push predicate to SQL, projection
    HIGH = 3         # flow restructuring, tx boundaries, caching — flag-gated
    FORBIDDEN = 4    # schema beyond additive index, API contract, auth


class CandidatePlan(BaseModel):
    id: str
    targets_hypothesis: str
    description: str
    mechanism_changed: str
    predicted_effect: str
    files_to_touch: list[str]              # becomes the write allowlist
    risk_tier: RiskTier                    # assigned by CODE, not the model
    what_could_break: list[str]
    requires_flag: bool = False


# ──────────────────────────  P12 implementation  ─────────────────────────

class Patch(BaseModel):
    plan_id: str
    branch: str
    diff: str
    files_changed: list[str]
    lines_added: int
    lines_removed: int
    build_iterations: int
    tests_passed: int
    tests_before: int


# ────────────────────────────  P14 measure  ──────────────────────────────

class PerfReport(BaseModel):
    before: LatencyStats
    after: LatencyStats
    p50_delta_pct: float
    p95_delta_pct: float
    sql_count_before: int
    sql_count_after: int
    rows_examined_before: int
    rows_examined_after: int
    p_value: float
    interleaved: Literal[True] = True
    mechanism_confirmed: bool
    mechanism_note: str


# ──────────────────────────  P15 gate & terminal  ───────────────────────

class GateName(str, Enum):
    BEHAVIOUR_PRESERVED = "behaviour_preserved"
    IMPROVEMENT_THRESHOLD = "improvement_threshold"
    STATISTICAL_SIGNIFICANCE = "statistical_significance"
    DIFF_MINIMALITY = "diff_minimality"
    MECHANISM_MATCH = "mechanism_match"
    NO_FORBIDDEN_CHANGE = "no_forbidden_change"


class GateResult(BaseModel):
    name: GateName
    passed: bool
    detail: str
    threshold: str | None = None
    actual: str | None = None


class GateDecision(BaseModel):
    results: list[GateResult]
    passed: bool
    decided_by: Literal["code"] = "code"


class TerminalState(str, Enum):
    PR_OPENED = "PR_OPENED"
    NO_BOTTLENECK_FOUND = "NO_BOTTLENECK_FOUND"
    BOTTLENECK_NOT_FIXABLE = "BOTTLENECK_NOT_FIXABLE"
    BEHAVIOUR_NOT_PRESERVED = "BEHAVIOUR_NOT_PRESERVED"
    NO_SIGNIFICANT_IMPROVEMENT = "NO_SIGNIFICANT_IMPROVEMENT"
    ENVIRONMENT_INFEASIBLE = "ENVIRONMENT_INFEASIBLE"
    BUDGET_EXHAUSTED = "BUDGET_EXHAUSTED"
    INTERNAL_ERROR = "INTERNAL_ERROR"


# ────────────────────────────  run state  ───────────────────────────────

class Phase(str, Enum):
    ONBOARD = "P-1"; PROVISION = "P0"; COMPREHEND = "P1"; WORKLOAD = "P2"
    ORACLE = "P3"; BASELINE = "P4"; DIAGNOSE = "P5"; PLAN = "P6"
    IMPLEMENT = "P7"; VERIFY = "P8"; MEASURE = "P9"; GATE = "P10"; SHIP = "P11"


class RunState(BaseModel):
    run_id: str
    repo: str
    endpoint: str
    phase: Phase
    completed_phases: list[Phase] = []
    artifacts: dict[str, ArtifactRef] = {}
    terminal: TerminalState | None = None
    usd_spent: float = 0.0
    tokens_in: int = 0
    tokens_out: int = 0


# ────────────────────────────  protocols  ───────────────────────────────

class Usage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    usd: float = 0.0


class LLMResponse(BaseModel):
    content: list[dict[str, Any]]
    stop_reason: str | None
    usage: Usage
    latency_ms: float
    from_cassette: bool = False

    def tool_uses(self) -> list[dict[str, Any]]: ...
    def text(self) -> str: ...


@runtime_checkable
class LLMClient(Protocol):
    """Deliberately stateless. There is no conversation memory here — the
    caller owns the message list. This is the central fact about LLMs."""
    def complete(self, *, system: str | list[dict], messages: list[dict],
                 tools: list[dict] | None = None,
                 tool_choice: dict | None = None,
                 max_tokens: int = 8192,
                 thinking_budget: int | None = None) -> LLMResponse: ...


class ToolResult(BaseModel):
    content: str
    is_error: bool = False
    bytes_returned: int = 0
    truncated: bool = False
    continuation: str | None = None


@runtime_checkable
class StackPack(Protocol):
    def detect(self, repo: str) -> float: ...
    def extract_routes(self, repo: str) -> list[dict]: ...
    def build_cmd(self) -> list[str]: ...
    def test_cmd(self) -> list[str]: ...
    def run_cmd(self, port: int) -> list[str]: ...
```

### 5.6 `senapathy/artifacts.py`

```python
class ArtifactStore:
    def __init__(self, run_dir: Path): ...
    def write(self, kind: str, model: BaseModel) -> ArtifactRef:
        """Serialise, content-hash, write to run_dir/artifacts/{kind}.json,
        return ref. Writing the same content twice is idempotent."""
    def read(self, ref: ArtifactRef, model_cls: type[T]) -> T:
        """Read and validate. Raise ContractViolation if the hash doesn't match
        the content — artifacts are immutable once written."""
    def exists(self, kind: str) -> bool: ...
```

### 5.7 `senapathy/trace.py`

```python
class EventType(str, Enum):
    PHASE_START; PHASE_END; LLM_REQUEST; LLM_RESPONSE; TOOL_CALL; TOOL_RESULT
    ARTIFACT_WRITTEN; GUARDRAIL_TRIGGERED; CHECKPOINT; GATE_EVALUATED
    TERMINAL_STATE; NOTE

class Trace:
    def __init__(self, run_dir: Path, run_id: str): ...
    def emit(self, type: EventType, phase: str, **fields) -> None:
        """Append one JSON line. Auto-adds ts, run, seq. Never raises —
        a broken trace must not kill a run."""
    def read_all(self) -> list[dict]: ...
    def cost_by_phase(self) -> dict[str, float]: ...
```

### 5.8 `senapathy/cli.py`

Typer app with subcommands registered but each raising
`NotImplementedError("phase Pnn")`: `understand`, `baseline`, `diagnose`, `patch`,
`verify`, `run`, `trace`.

## 6. ACCEPTANCE TEST

Create `tests/test_contracts.py`:

```python
from __future__ import annotations
import pytest
from pydantic import ValidationError
from senapathy.contracts import (
    Citation, EndpointModel, HypothesisSet, Hypothesis,
    RiskTier, TerminalState, EquivalenceVerdict, RequestCorpus, RequestSpec,
)


def test_citation_ref():
    assert Citation(file="a/B.java", line=34).ref() == "a/B.java:34"


def test_hypothesis_set_requires_three():
    """Forced competition: fewer than 3 hypotheses is a contract violation,
    because a single hypothesis means the model locked onto its first idea."""
    one = Hypothesis(id="h1", claim="c", mechanism="m", evidence=["e"],
                     attributed_ms=100, prediction="p", confidence=0.5)
    with pytest.raises(ValidationError):
        HypothesisSet(hypotheses=[one], ranked_ids=["h1"], measured_total_ms=200)


def test_attribution_validity():
    def h(i, ms):
        return Hypothesis(id=f"h{i}", claim="c", mechanism="m", evidence=["e"],
                          attributed_ms=ms, prediction="p", confidence=0.5)
    ok = HypothesisSet(hypotheses=[h(1, 100), h(2, 50), h(3, 20)],
                       ranked_ids=["h1", "h2", "h3"], measured_total_ms=200)
    assert ok.attribution_valid()
    bad = HypothesisSet(hypotheses=[h(1, 5000), h(2, 50), h(3, 20)],
                        ranked_ids=["h1", "h2", "h3"], measured_total_ms=200)
    assert not bad.attribution_valid()


def test_confidence_bounded():
    with pytest.raises(ValidationError):
        Hypothesis(id="h", claim="c", mechanism="m", evidence=[],
                   attributed_ms=1, prediction="p", confidence=1.4)


def test_verdict_decided_by_code_only():
    """A verdict must never be attributable to the model."""
    v = EquivalenceVerdict(passed=True, compared=10, identical=10,
                           db_state_matched=True, external_calls_matched=True,
                           held_back_passed=True, oracle_hash="abc")
    assert v.decided_by == "code"
    with pytest.raises(ValidationError):
        EquivalenceVerdict(passed=True, compared=1, identical=1,
                           db_state_matched=True, external_calls_matched=True,
                           held_back_passed=True, oracle_hash="a",
                           decided_by="model")


def test_endpoint_model_collects_citations():
    m = EndpointModel(route="POST /x",
                      entry=Citation(file="C.java", line=1),
                      path=[Citation(file="C.java", line=1),
                            Citation(file="S.java", line=9)])
    assert len(m.all_citations()) >= 2
    assert m.uncertainties == []


def test_corpus_held_back_defaults_empty_but_present():
    c = RequestCorpus(benchmark=[RequestSpec(id="r1", method="GET", path="/x")],
                      hash="h")
    assert c.held_back == []


def test_terminal_states_cover_all_eight():
    assert len(TerminalState) == 8
    assert TerminalState.NO_BOTTLENECK_FOUND in TerminalState


def test_risk_tiers_ordered():
    assert RiskTier.LOW < RiskTier.HIGH < RiskTier.FORBIDDEN
```

Create `tests/test_artifacts.py`:

```python
from __future__ import annotations
import pytest
from senapathy.artifacts import ArtifactStore
from senapathy.trace import Trace, EventType
from senapathy.contracts import Citation, EndpointModel
from senapathy.errors import ContractViolation


def _model():
    return EndpointModel(route="POST /x", entry=Citation(file="C.java", line=1),
                         path=[Citation(file="C.java", line=1)])


def test_write_read_roundtrip(tmp_path):
    store = ArtifactStore(tmp_path)
    ref = store.write("endpoint_model", _model())
    assert store.exists("endpoint_model")
    back = store.read(ref, EndpointModel)
    assert back.route == "POST /x"


def test_write_is_content_addressed_and_idempotent(tmp_path):
    store = ArtifactStore(tmp_path)
    a = store.write("endpoint_model", _model())
    b = store.write("endpoint_model", _model())
    assert a.hash == b.hash


def test_tampered_artifact_is_rejected(tmp_path):
    """Artifacts are immutable. If the bytes change, reading must fail —
    this is what makes the evidence ledger trustworthy."""
    store = ArtifactStore(tmp_path)
    ref = store.write("endpoint_model", _model())
    p = tmp_path / "artifacts" / "endpoint_model.json"
    p.write_text(p.read_text().replace("POST /x", "POST /tampered"))
    with pytest.raises(ContractViolation):
        store.read(ref, EndpointModel)


def test_trace_appends_and_never_raises(tmp_path):
    t = Trace(tmp_path, "run1")
    t.emit(EventType.PHASE_START, "P1")
    t.emit(EventType.LLM_RESPONSE, "P1", usd=0.02, tokens_in=100)
    t.emit(EventType.PHASE_END, "P1")
    events = t.read_all()
    assert len(events) == 3
    assert [e["seq"] for e in events] == [0, 1, 2]
    assert all(e["run"] == "run1" for e in events)


def test_trace_cost_by_phase(tmp_path):
    t = Trace(tmp_path, "run1")
    t.emit(EventType.LLM_RESPONSE, "P1", usd=0.02)
    t.emit(EventType.LLM_RESPONSE, "P1", usd=0.03)
    t.emit(EventType.LLM_RESPONSE, "P5", usd=0.10)
    costs = t.cost_by_phase()
    assert round(costs["P1"], 4) == 0.05
    assert round(costs["P5"], 4) == 0.10


def test_trace_survives_unserialisable_field(tmp_path):
    """A broken trace must never kill a run."""
    t = Trace(tmp_path, "run1")
    t.emit(EventType.NOTE, "P1", weird=object())
    assert len(t.read_all()) == 1
```

Both files must pass: `pytest tests/ -q`

## 7. OUT OF SCOPE

Do **not** create: any LLM client, any agent loop, any tool, any Docker file, any
StackPack implementation, any orchestrator. Do not add fields to the contracts beyond
what section 5.5 specifies. Do not implement `render_for_model()` beyond a stub that
raises `NotImplementedError` — that belongs to P08.

## 8. DEMO

```bash
pytest tests/ -q                 # all green
senapathy --help                 # subcommands listed
senapathy understand --help      # exists, raises NotImplementedError when run
```

## 9. LEARNING

**Concept: artifacts as the interface between phases.**

The thing to notice here is that you've just written down the entire data flow of the
product before writing any logic. Every phase's contract with every other phase is
visible in one file. That's not bureaucracy — it's what lets you test P09 without
having built P07, by constructing an `EvidenceBundle` by hand in a test.

**Notice what's encoded as a type rather than a rule:**

- `HypothesisSet.hypotheses` has `min_length=3`. "Generate competing hypotheses"
  isn't a prompt instruction that the model may ignore — it's a validation error.
- `EquivalenceVerdict.decided_by` is `Literal["code"]`. It is *impossible* to
  construct a verdict attributed to the model.
- `RedFlag` carries a docstring saying it's computed. `attribution_valid()` lives on
  the model, not in a prompt.

This is the difference between a guardrail and a wish. **Instructions are advisory;
types are enforced.** Every time you're tempted to write "the agent should always..."
in a prompt, ask whether it can be a type constraint or a validation instead.

**Experiment before moving on:** try to construct an `EquivalenceVerdict` that claims
a model decided it. You can't. Now try to write a prompt that guarantees a model
won't claim it decided. You can't. Sit with that difference — it's most of what
separates reliable AI systems from LLM wrappers.
