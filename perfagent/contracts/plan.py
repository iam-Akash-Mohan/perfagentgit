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
    table and the shape it stamps belong together, and `perfagent.contracts`
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


class CandidateFix(BaseModel):
    """One proposed fix. The model's plan, with code's risk assessment on it."""

    id: str
    targets_hypothesis: str
    approach: Approach
    description: str
    mechanism_changed: str          # "SQL statements per request"
    predicted_effect: str           # "queries per request drop from 6 to 2"
    files_to_touch: list[str]       # becomes the patch agent's write allowlist
    sql_change: str | None = None   # the index or rewritten query, if any
    what_could_break: list[str]
    preserves_response_bytes: bool  # the model's claim; verified by the oracle

    # ── assigned by code, never present in the tool schema ───────────────
    risk_tier: RiskTier
    requires_flag: bool = False
    expected_value_ms: float = 0.0


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
