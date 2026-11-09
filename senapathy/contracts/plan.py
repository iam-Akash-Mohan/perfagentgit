"""The plan artifact: what is worth fixing, and which fix is selected.

The shape this file exists for is `FixabilityAssessment`. Ranking bottlenecks
by `attributed_ms` alone cannot tell a 1400 ms cost that IS the API contract
from a 200 ms cost with a clean fix, so the agent spent three runs attacking
the unwinnable one. Fixability is the second axis, it is computed from a fixed
policy table, and `assessed_by` is a `Literal["code"]` for the same reason
`EquivalenceVerdict.decided_by` is: it must be impossible to attribute to a
model.

`CandidateFix` mixes two kinds of field on purpose, and the split is worth
knowing when reading one:

  * The model supplies the judgement — approach, description, predicted
    effect, what could break, whether it believes the response bytes survive.
  * Code supplies `risk_tier`, `requires_flag` and `expected_value_ms`. None
    of the three is in the tool schema, so a model cannot assign itself a low
    risk tier or a high expected value.
"""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


class RiskTier(int, Enum):
    """How much behaviour risk an approach carries. Assigned by CODE.

    Moved here from the package root when the plan phase was built: the tier
    table and the shape it stamps belong together, and `senapathy.contracts`
    re-exports it, so no import path changed.
    """

    LOW = 1          # index, @EntityGraph, readOnly tx, view narrowing
    MEDIUM = 2       # query rewrite, batching, push predicate to SQL, projection
    HIGH = 3         # flow restructuring, tx boundaries, caching — flag-gated
    FORBIDDEN = 4    # schema beyond additive index, API contract, auth, tests


Approach = Literal[
    "fetch_join", "entity_graph", "batch_query", "dto_projection",
    "add_index", "query_rewrite", "remove_redundant_call",
    "move_work_out_of_loop", "read_only_transaction", "other",
]

APPROACHES: tuple[str, ...] = (
    "fetch_join", "entity_graph", "batch_query", "dto_projection",
    "add_index", "query_rewrite", "remove_redundant_call",
    "move_work_out_of_loop", "read_only_transaction", "other",
)

# The measurable quantity a candidate claims to move, typed rather than
# classified from prose (COL-06: a candidate whose mechanism_changed said
# "rows examined" was classified sql_count because its description also
# mentioned a query — nearly every description does). Only sql_count and
# rows_examined are actually instrumented (gate.rules.INSTRUMENTED_
# MECHANISMS, read from PerfReport); the others are here so a candidate can
# state one honestly and be ruled out at plan time instead of failing
# MECHANISM_MATCH after a patch, a verify and a measurement have been paid
# for.
Mechanism = Literal["sql_count", "rows_examined", "serialization", "allocation", "other"]

MECHANISMS: tuple[str, ...] = (
    "sql_count", "rows_examined", "serialization", "allocation", "other",
)

# Below this, a hypothesis is not worth attacking and `blocked_by` says why.
BLOCKED_BELOW = 0.2


class FixabilityAssessment(BaseModel):
    """How likely a BEHAVIOUR-PRESERVING fix exists, and why.

    Not a confidence, and not a judgement about whether the hypothesis is
    true — diagnosis already decided that. This answers the separate question
    the gate keeps asking: if we fix this, will the oracle let it through?
    """

    hypothesis_id: str
    fixability: float = Field(ge=0.0, le=1.0)
    reason: str
    blocked_by: str | None = None      # set when fixability < BLOCKED_BELOW
    assessed_by: Literal["code"] = "code"   # never "model"
    classified: bool = Field(
        default=True,
        description="False when NO policy row matched and the default was "
                    "used. The default is a placeholder, not a finding: it "
                    "means the table could not describe this hypothesis, and "
                    "an unclassified entry must never outrank a classified "
                    "one no matter how large its expected value.")

    @property
    def blocked(self) -> bool:
        return self.fixability < BLOCKED_BELOW


class RankedHypothesis(BaseModel):
    """One hypothesis with its fixability and the product of the two.

    `expected_value_ms = attributed_ms * fixability` is the whole point of the
    phase: it is what puts a 200 ms fixable cause above a 1400 ms unfixable
    one. Computed here, never by a model (invariant 5).
    """

    hypothesis_id: str
    claim: str
    layer: str
    attributed_ms: float
    fixability: FixabilityAssessment
    expected_value_ms: float


class SqlVerification(BaseModel):
    """What EXPLAIN ANALYZE said about a candidate's own `sql_before`/
    `sql_after`, measured at PLAN time, before anything is patched.

    `assessed_by` mirrors `FixabilityAssessment`: this is a measurement, never
    a model's claim about its own fix. `verified=False` covers two different
    situations that must not be collapsed into one:

    - `attempted=False` — no live database was reachable at plan time AT ALL.
      Every candidate in this run is equally unable to be checked, so this is
      a planning limitation, not something this candidate did wrong; it falls
      back to the hypothesis-level `expected_value_ms`, same as before this
      field existed.
    - `attempted=True` — a database WAS reachable and EXPLAIN ANALYZE was
      actually run against this candidate's own statements, and it could not
      complete (a parse error, a parameter mismatch, a reference EXPLAIN
      rejects). This is NOT the same as `rows_delta_pct` clearing or missing
      the mechanism-match bar — it means the claim was never checked at all.
      An instrumented mechanism has not earned selection without evidence, so
      this disqualifies the candidate (see `sql_verification_failure`) — the
      unknown is reported as unverifiable, never as disproven.
    """

    verified: bool
    attempted: bool = Field(
        default=True,
        description="False only when there was no live database connection "
                    "at all at plan time — a run-wide limitation. True for "
                    "every other case, including when the statement itself "
                    "could not be run; defaults True so historical records "
                    "(all of which came from a reachable database) read "
                    "correctly without being rewritten.")
    rows_examined_before: int = 0
    rows_examined_after: int = 0
    rows_delta_pct: float = 0.0
    actual_ms_before: float = 0.0
    actual_ms_after: float = 0.0
    scored: bool = Field(
        default=False,
        description="True only when this measurement was actually used to "
                    "set expected_value_ms — currently just the "
                    "rows_examined mechanism, and only once verified=True. A "
                    "sql_count candidate is still verified and reported "
                    "here, but a single EXPLAIN ANALYZE of one call does not "
                    "capture a call-COUNT fix, so it is not used to score.")
    note: str = ""
    assessed_by: Literal["code"] = "code"


class CandidateFix(BaseModel):
    """One proposed fix. The model's plan, with code's risk assessment on it."""

    id: str
    targets_hypothesis: str
    approach: Approach
    description: str
    mechanism: Mechanism | None = Field(
        default=None,
        description="The measurable quantity this fix moves. None only for "
                    "a candidate recorded before this field existed — the "
                    "plan agent must always set it; classify_mechanism's "
                    "text fallback exists solely to read those older "
                    "artifacts.")
    mechanism_changed: str          # "SQL statements per request"
    predicted_effect: str           # "queries per request drop from 6 to 2"
    files_to_touch: list[str]       # becomes the patch agent's write allowlist
    sql_change: str | None = None   # the additive CREATE INDEX, if any — NOT
                                    # a rewritten query; see sql_before/sql_after
    sql_before: str | None = Field(
        default=None,
        description="The query that runs TODAY, in runnable form with %s "
                    "placeholders — required alongside sql_after for any "
                    "SQL-rewrite approach whose mechanism is instrumented, so "
                    "EXPLAIN ANALYZE has a baseline to compare against.")
    sql_after: str | None = Field(
        default=None,
        description="The query this candidate replaces it with. Same "
                    "placeholder convention as sql_before and the explain "
                    "tool. This is checked with EXPLAIN ANALYZE before the "
                    "candidate can be selected — a claim about rows examined "
                    "is a measurement here, not a prediction left for the "
                    "gate to discover after a patch, a verify and a "
                    "measurement have already been paid for.")
    sql_params: list[str] = Field(
        default_factory=list,
        description="Values for the %s placeholders shared by sql_before "
                    "and sql_after, in order — same binding convention as "
                    "the explain tool.")
    baseline_digest: str | None = Field(
        default=None,
        description="The digest hash, from the brief's baseline evidence, "
                    "of the statement a 'sql_count' candidate claims to "
                    "reduce the call count of. Checked against that same "
                    "evidence, not trusted — see "
                    "submit.py::_baseline_citation_failure.")
    baseline_calls: int | None = Field(
        default=None,
        description="The call count this candidate claims that baseline "
                    "digest runs today, copied from the same evidence.")
    what_could_break: list[str]
    preserves_response_bytes: bool  # the model's claim; verified by the oracle

    # ── assigned by code, never present in the tool schema ───────────────
    risk_tier: RiskTier
    requires_flag: bool = False
    expected_value_ms: float = 0.0
    sql_verification: SqlVerification | None = Field(
        default=None,
        description="Set at PLAN time by running EXPLAIN ANALYZE on "
                    "sql_before/sql_after and comparing. None for a candidate "
                    "with no SQL to verify — a Java-side call-structure "
                    "change like remove_redundant_call or "
                    "move_work_out_of_loop keeps hypothesis-level scoring. "
                    "Never set by the model.")
    not_selectable_because: str | None = Field(
        default=None,
        description="Set at PLAN time when this candidate changes the schema "
                    "and the two arms would not be isolated — measuring it "
                    "would compare two arms that both have the change and "
                    "report a false negative. The candidate is NOT deleted: "
                    "'we could not verify this here' is itself a finding, and "
                    "the argument for building per-arm schemas. A model never "
                    "sets this field; only code does.")
    scope_reasons: dict[str, str] = Field(
        default_factory=dict,
        description="Files CODE added to files_to_touch beyond what the "
                    "model declared, each with why: it sits on the "
                    "endpoint's own traced call path between the entry "
                    "point and a file the candidate already names, so a "
                    "signature change downstream has a caller to update. "
                    "Never set by the model.")


class Rejection(BaseModel):
    """A candidate that lost, and the reason. Part of the PR body later."""

    candidate_id: str
    reason: str


class Plan(BaseModel):
    """The whole output of the plan phase.

    `selected_id` is chosen by code from the ranked list — highest expected
    value whose risk tier is low enough and whose author claims the response
    bytes survive. The losers stay in `candidates`, because "why not the
    obvious fix" is the part of a PR that makes it reviewable.
    """

    endpoint: str
    measured_total_ms: float
    ranked: list[RankedHypothesis]
    considered: list[str] = []          # hypothesis ids put to the agent
    candidates: list[CandidateFix] = []
    selected_id: str | None = None
    selection_reason: str = ""
    not_selected: list[Rejection] = []
    prompt_hash: str | None = None
    base_commit: str | None = None
    decided_by: Literal["code"] = "code"

    def selected(self) -> CandidateFix | None:
        return next((c for c in self.candidates if c.id == self.selected_id), None)

    def assessment_for(self, hypothesis_id: str) -> FixabilityAssessment | None:
        for entry in self.ranked:
            if entry.hypothesis_id == hypothesis_id:
                return entry.fixability
        return None
