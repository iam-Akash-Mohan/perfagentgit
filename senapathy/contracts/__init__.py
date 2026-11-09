"""Every artifact this product produces, as a Pydantic model.

The comprehension shapes live in `comprehension.py` and the two pointers they
are built from in `primitives.py`; both are re-exported here, so
`from senapathy.contracts import X` keeps working for every X.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, Field

from .comprehension import (CacheInfo, CallHop, DataAccess, DataAccessKind,
                            EndpointModel, EntityInfo, EntityRelation,
                            ExternalCall, HiddenQueryTrigger, IndexInfo,
                            PaginationInfo, PerRowWork, RequestParameter,
                            RequestShape, ResponseShape, SqlSource, TableInfo,
                            TransactionInfo)
from .plan import (APPROACHES, BLOCKED_BELOW, MECHANISMS, Approach,
                   CandidateFix, FixabilityAssessment, Mechanism, Plan,
                   RankedHypothesis, Rejection, RiskTier, SqlVerification)
from .primitives import ArtifactRef, Citation

__all__ = [
    # primitives
    "Citation", "ArtifactRef",
    # comprehension
    "CacheInfo", "CallHop", "DataAccess", "DataAccessKind", "EndpointModel",
    "EntityInfo", "EntityRelation", "ExternalCall", "HiddenQueryTrigger",
    "IndexInfo", "PaginationInfo", "PerRowWork", "RequestParameter",
    "RequestShape", "ResponseShape", "SqlSource", "TableInfo",
    "TransactionInfo",
    # plan
    "APPROACHES", "BLOCKED_BELOW", "MECHANISMS", "Approach", "CandidateFix",
    "FixabilityAssessment", "Mechanism", "Plan", "RankedHypothesis",
    "Rejection", "SqlVerification",
    # everything defined below
    "RequestSpec", "RequestCorpus", "RecordedResponse", "TolerancePolicy",
    "BehaviourOracle", "FieldDiff", "EquivalenceVerdict", "LatencyStats",
    "SqlDigestStat", "ProfileFrame", "QueryPlan", "RedFlagKind", "RedFlag",
    "EvidenceBundle", "BaselineReport", "Probe", "Layer", "Hypothesis",
    "HypothesisSet", "RiskTier", "CandidatePlan", "Patch", "PerfReport",
    "GateName", "GateResult", "GateDecision", "TerminalState", "Phase",
    "RunState", "Usage", "LLMResponse", "LLMClient", "ToolResult",
    "ToolCallResult", "StackPack",
]


# ────────────────────────────  P06 workload  ─────────────────────────────

class RequestSpec(BaseModel):
    id: str
    method: str
    path: str
    headers: dict[str, str] = {}
    body: dict[str, Any] | None = None
    note: str | None = None
    expected_status: int | None = Field(
        default=None,
        description="What this exact request actually returned when the "
                    "corpus was PROBED against the blue arm, before anything "
                    "was sealed or measured. None until probed. Not a claim "
                    "about correctness — a deliberate error-path request is "
                    "legitimate and keeps its observed 4xx/5xx here — only a "
                    "record that the request was fired and this is what came "
                    "back, so a corpus that never ran cannot be mistaken for "
                    "one that did.")


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
    # Which endpoint this recording is OF. The oracle used to carry only a
    # corpus_hash, so the only link back to an endpoint ran through a corpus
    # file you had to still possess; a stale global oracle for another route
    # therefore validated cleanly and the mismatch surfaced much later, as a
    # hash error, if at all.
    endpoint: str = ""
    corpus_hash: str
    recorded: list[RecordedResponse]
    policy: TolerancePolicy
    base_commit: str
    excluded_tables: list[str] = Field(
        default_factory=list,
        description="Schema-management bookkeeping tables (Flyway, Liquibase, "
                    "...) found in the schema at record time and left out of "
                    "every db-state comparison. Sealed in here, exactly like "
                    "the policy, so a later run cannot silently exclude more "
                    "than this one did. Empty when capture_db was off.")
    status_counts: dict[str, int] = Field(
        default_factory=dict,
        description="HTTP status code -> how many of `recorded` carried it. "
                    "Computed from `recorded`, never asserted — so a later "
                    "phase can see at a glance whether this recording ever "
                    "exercised the endpoint at all, without re-deriving it "
                    "from the raw list every time.")
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
# `RiskTier`, `CandidateFix`, `FixabilityAssessment` and `Plan` live in
# `plan.py` and are re-exported above. `CandidatePlan` below is the older
# shape the gate reads; a selected `CandidateFix` converts into one.

class CandidatePlan(BaseModel):
    id: str
    targets_hypothesis: str
    description: str
    mechanism: Mechanism | None = None     # None only for pre-existing plans
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
    granted_files: list[str] = Field(
        default_factory=list,
        description="Files request_file_access granted during patching, "
                    "beyond the plan's own files_to_touch. DIFF_MINIMALITY "
                    "checks the diff against files_to_touch PLUS this list "
                    "— the effective scope the patch actually used, not "
                    "the narrower one the plan alone declared. Never set "
                    "by the model; copied from PatchTools' own grant log.")


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
    p_value: float                      # Mann-Whitney U, unpaired
    p_value_paired: float | None = Field(
        default=None,
        description="Wilcoxon signed-rank, computed over the interleaved "
                    "pairs. THIS is what the significance gate reads: the "
                    "samples are paired by design, and on this corpus the "
                    "per-request spread is two orders of magnitude larger "
                    "than the between-arm difference, so the unpaired test "
                    "reports p=0.36 where the paired one reports p=0.0015. "
                    "Optional only so that artifacts recorded before the "
                    "field existed still validate.")
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
    # All six gates passed and nothing has been pushed yet -- the state a
    # dry run, or any passing decision not yet shipped, actually reached.
    # Distinct from PR_OPENED (which requires shipped=True) and from
    # BOTTLENECK_NOT_FIXABLE (which means a gate FAILED).
    READY_TO_SHIP = "READY_TO_SHIP"
    NO_BOTTLENECK_FOUND = "NO_BOTTLENECK_FOUND"
    BOTTLENECK_NOT_FIXABLE = "BOTTLENECK_NOT_FIXABLE"
    BEHAVIOUR_NOT_PRESERVED = "BEHAVIOUR_NOT_PRESERVED"
    NO_SIGNIFICANT_IMPROVEMENT = "NO_SIGNIFICANT_IMPROVEMENT"
    # "we could not measure this" is a different fact from "this did not
    # help", and a report in which they look alike is a report that teaches
    # the reader the wrong lesson. Raised when a schema change cannot be A/B
    # measured because both arms share one database.
    MEASUREMENT_NOT_ISOLATED = "MEASUREMENT_NOT_ISOLATED"
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
