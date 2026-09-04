from __future__ import annotations
from enum import Enum
from typing import Any, Literal, Protocol, cast, runtime_checkable
from pydantic import BaseModel, Field, model_validator


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

class CallHop(BaseModel):
    """One method on the execution path, with what it actually does.

    Flat rather than wrapping a Citation because this is the shape the
    diagnosis phase already reads off disk, and because `does` is not a
    property of the pointer — it is the finding.
    """
    method: str                     # e.g. CustomerService.search
    file: str
    line: int
    does: str

    def citation(self) -> "Citation":
        return Citation(file=self.file, line=self.line, symbol=self.method)


DataAccessKind = Literal[
    "jpql_query", "native_query", "derived_query",
    "lazy_collection", "lazy_association", "criteria", "jdbc_template", "other",
]

# How a data access maps onto the older SqlSource vocabulary. Derived in code
# so the model is never asked to fill in two overlapping fields.
_SQL_SOURCE_KIND: dict[str, str] = {
    "jpql_query": "@Query JPQL",
    "native_query": "@Query native",
    "derived_query": "derived query",
    "lazy_collection": "lazy collection",
    "lazy_association": "lazy collection",
    "criteria": "criteria",
    "jdbc_template": "jdbc template",
    "other": "other",
}


class DataAccess(BaseModel):
    """One way this endpoint can reach the database.

    `executes_per_row` is the field the whole phase exists for: a query that
    runs once per request and the identical query run once per result row are
    the same SQL and a completely different bottleneck.
    """
    trigger: str                    # "repo.findByFilter", "lazy load of Customer.invoices"
    kind: DataAccessKind
    file: str
    line: int
    sql: str | None = None
    tables: list[str] = []
    executes_per_row: bool
    note: str | None = None

    def citation(self) -> "Citation":
        return Citation(file=self.file, line=self.line, symbol=None)


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
    """The entire output of the comprehension phase.

    `call_chain` / `data_access` are what an agent submits. `entry`, `path`
    and `sql_sources` are DERIVED from them in `_derive` below rather than
    asked for separately: two fields describing the same fact are two fields
    that can disagree, and the model does not get to decide which is right.
    """
    route: str
    entry: Citation
    path: list[Citation]
    call_chain: list[CallHop] = []
    data_access: list[DataAccess] = []
    sql_sources: list[SqlSource] = []
    entities: list[EntityInfo] = []
    transactions: list[TransactionInfo] = []
    external_calls: list[ExternalCall] = []
    tables: list[TableInfo] = []
    tests_covering: list[str] = []
    uncertainties: list[str] = Field(
        default_factory=list,
        description="What the agent was NOT sure about. An agent forced to "
                    "declare uncertainty is more useful than a confident one.")
    prompt_hash: str | None = None      # which prompt version produced this

    @model_validator(mode="before")
    @classmethod
    def _derive(cls, data: Any) -> Any:
        """Fill entry / path / sql_sources from call_chain / data_access.

        Only fills what is absent, so hand-built models that supply the older
        fields directly keep working.
        """
        if not isinstance(data, dict):
            return data
        out = dict(data)

        chain = out.get("call_chain") or []
        hops = [h if isinstance(h, CallHop) else CallHop.model_validate(h)
                for h in chain]
        if not out.get("entry") and hops:
            out["entry"] = hops[0].citation()
        if not out.get("path") and hops:
            out["path"] = [h.citation() for h in hops]

        access = out.get("data_access") or []
        if not out.get("sql_sources") and access:
            reads = [a if isinstance(a, DataAccess) else DataAccess.model_validate(a)
                     for a in access]
            out["sql_sources"] = [
                SqlSource(method=a.trigger,
                          kind=cast(Any, _SQL_SOURCE_KIND.get(a.kind, "other")),
                          tables=list(a.tables),
                          citation=a.citation())
                for a in reads
            ]
        return out

    def per_row_accesses(self) -> list[DataAccess]:
        """Computed here, never counted by a model (invariant 5)."""
        return [a for a in self.data_access if a.executes_per_row]

    def all_citations(self) -> list[Citation]:
        out: list[Citation] = [self.entry, *self.path]
        out.extend(s.citation for s in self.sql_sources)
        out.extend(e.citation for e in self.entities)
        out.extend(t.citation for t in self.transactions)
        out.extend(c.citation for c in self.external_calls)
        return out


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
        raise NotImplementedError("phase P08")


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


class Layer(str, Enum):
    """Where in the stack the time is claimed to go.

    Diagnosis exists to stop the first plausible story from winning. Forcing
    every hypothesis to name a layer is what makes "the SQL is slow" and "the
    response is enormous" compete instead of blur together — and it lets code
    check that the set actually spans more than one layer.
    """
    DATABASE_QUERY = "database_query"      # time inside the server executing SQL
    ORM_HYDRATION = "orm_hydration"        # building entities from result sets
    APPLICATION_LOGIC = "application_logic"  # our own loops and allocation
    SERIALIZATION = "serialization"        # object graph -> JSON bytes
    NETWORK = "network"                    # transfer and socket time
    OTHER = "other"


class Hypothesis(BaseModel):
    id: str
    claim: str
    mechanism: str
    layer: Layer
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
        return sum(h.attributed_ms for h in self.hypotheses) <= self.measured_total_ms


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
    raw_message: dict[str, Any] | None = Field(
        default=None,
        description="The provider-shaped assistant turn, verbatim. The loop "
                    "echoes this back in the next request. This field is the "
                    "single source of it — the client stores no such state.")
    model: str | None = None

    
    def tool_uses(self) -> list[dict[str, Any]]:
        return [b for b in self.content if b.get("type") == "tool_use"]

    def text(self) -> str:
        return "".join(b.get("text", "") for b in self.content
                       if b.get("type") == "text")


@runtime_checkable
class LLMClient(Protocol):
    """Deliberately stateless. There is no conversation memory here — the
    caller owns the message list. This is the central fact about LLMs."""
    def complete(self, *, system: str | list[dict[str, Any]] | None = None,
                 messages: list[dict[str, Any]],
                 tools: list[dict[str, Any]] | None = None,
                 tool_choice: dict[str, Any] | None = None,
                 max_tokens: int | None = None) -> LLMResponse: ...

    def tool_result_messages(
        self, results: list["ToolCallResult"]
    ) -> list[dict[str, Any]]:
        """Provider-shaped messages carrying tool results back.

        The wire shape differs per provider — OpenAI wants one message per
        call, Anthropic wants one user message holding every tool_result — so
        the client owns it and the loop stays provider-agnostic."""
        ...


class ToolResult(BaseModel):
    content: str
    is_error: bool = False
    bytes_returned: int = 0
    truncated: bool = False
    continuation: str | None = None


class ToolCallResult(BaseModel):
    """One dispatched tool call and what came back, before wire formatting."""
    id: str
    name: str
    content: str
    is_error: bool = False


@runtime_checkable
class StackPack(Protocol):
    def detect(self, repo: str) -> float: ...
    def extract_routes(self, repo: str) -> list[dict[str, Any]]: ...
    def build_cmd(self) -> list[str]: ...
    def test_cmd(self) -> list[str]: ...
    def run_cmd(self, port: int) -> list[str]: ...
