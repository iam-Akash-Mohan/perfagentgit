from __future__ import annotations

from typing import Any

from pydantic import ValidationError

from ..contracts import HypothesisSet, Layer
from ..errors import ToolError
from .probe import ProbeLog

# How far over measured p50 the attributed milliseconds may sum.
# Some overlap between hypotheses is legitimate — ORM hydration and
# serialization both touch the same objects — so the sum is allowed to exceed
# p50 a little. Allowing it to exceed p50 by 3x would let the model attribute
# time that does not exist.
ATTRIBUTION_TOLERANCE = 0.20
MIN_HYPOTHESES = 3
MIN_LAYERS = 2

# The other direction: a diagnosis that explains only a sliver of the
# measured latency has not found the cause, whatever it found instead.
# COL-04 attributed 11 ms of a 267 ms endpoint (4%) and passed the OLD check
# cleanly, because that check only ever caught OVER-attribution. Half is a
# low bar on purpose -- legitimate overlap between hypotheses already eats
# into the other direction's tolerance, so this is "explains less than half
# of what was measured", not "explains it exactly."
MIN_ATTRIBUTION_SHARE = 0.5

# sql_share_of_latency_pct thresholds for the evidence/diagnosis contradiction
# check. Matched to the same "well above 60%" bar this project already uses
# elsewhere (the collections endpoint was deliberately seeded to clear it) --
# not a new number invented for this check alone.
HIGH_SQL_SHARE_PCT = 60.0
LOW_SQL_SHARE_PCT = 20.0
# How much of measured p50 a database_query hypothesis must claim before a
# LOW sql_share makes that claim a contradiction, not just an overstatement.
MAJORITY_ATTRIBUTION_SHARE = 0.5


def validate_hypothesis_set(
    payload: dict[str, Any],
    measured_p50_ms: float,
    probe_log: ProbeLog | None = None,
    *,
    sql_share_of_latency_pct: float | None = None,
) -> HypothesisSet:
    """Accept or reject the agent's diagnosis. Code decides, not the model.

    Every failure raises ToolError with the specific problem, because the
    message goes straight back to the agent as a tool result and is the only
    thing it has to work from. "Invalid hypotheses" teaches it nothing;
    "attributed_ms sums to 4200 but measured p50 is 1780" tells it exactly
    which number to change.

    `sql_share_of_latency_pct` is read from the recorded baseline, same as
    `measured_p50_ms` — never asked of the model. Optional only so that a
    caller without a baseline handy (a hand-built test, an older artifact)
    still gets every other check; passing it is what makes 2c reachable.
    """
    if not isinstance(payload, dict):
        raise ToolError(f"emit_hypotheses expects an object, got {type(payload).__name__}")

    # measured_total_ms is injected from the recorded baseline, never taken
    # from the model — invariant 4: no number without an artifact.
    enriched = dict(payload)
    enriched["measured_total_ms"] = measured_p50_ms

    try:
        result = HypothesisSet.model_validate(enriched)
    except ValidationError as exc:
        raise ToolError(
            "the hypotheses did not match the required schema:\n"
            + _explain_validation_error(exc),
            suggestion="Fix the listed fields and call emit_hypotheses again.",
        ) from exc

    # ── 1. forced competition ───────────────────────────────────────────
    if len(result.hypotheses) < MIN_HYPOTHESES:
        raise ToolError(
            f"only {len(result.hypotheses)} hypotheses; at least {MIN_HYPOTHESES} are "
            "required so that explanations compete instead of the first one winning",
            suggestion="Add hypotheses that would each be ruled out by a different "
                       "measurement.",
        )

    # ── 2. attribution cannot exceed the time that exists ───────────────
    attributed = sum(h.attributed_ms for h in result.hypotheses)
    ceiling = measured_p50_ms * (1.0 + ATTRIBUTION_TOLERANCE)
    if attributed > ceiling:
        raise ToolError(
            f"attributed_ms sums to {attributed:.1f} ms but measured p50 is "
            f"{measured_p50_ms:.1f} ms (ceiling {ceiling:.1f} ms, "
            f"{int(ATTRIBUTION_TOLERANCE * 100)}% over). "
            "You have attributed more time than the endpoint actually takes.",
            suggestion="Reduce attributed_ms so the total fits inside measured p50. "
                       "Overlapping causes should not each claim the full duration.",
        )

    # ── 2b. ...nor account for only a sliver of it ──────────────────────
    # The symmetric case the check above never covered: a diagnosis that
    # explains 11 ms of a 267 ms endpoint sailed through unchallenged
    # because nothing checked the floor, only the ceiling.
    floor = measured_p50_ms * MIN_ATTRIBUTION_SHARE
    if measured_p50_ms > 0 and attributed < floor:
        raise ToolError(
            f"attributed_ms sums to {attributed:.1f} ms, only "
            f"{100.0 * attributed / measured_p50_ms:.1f}% of the measured p50 "
            f"of {measured_p50_ms:.1f} ms. A diagnosis that explains a small "
            "fraction of what was actually measured has not found the cause, "
            "whatever it found instead.",
            suggestion="Keep looking: run_probe again with a different shape, "
                       "or check the database tools (explain, list_indexes) "
                       "for a cost none of these hypotheses names yet.",
        )

    # ── 2c. the evidence and the diagnosis must agree on WHERE the time is ──
    # A high SQL share with no database hypothesis, or a low SQL share with
    # a database hypothesis claiming most of p50, are the same mistake in
    # opposite directions: the diagnosis disagrees with the measurement it
    # is supposed to explain.
    if sql_share_of_latency_pct is not None:
        has_database_hypothesis = any(
            h.layer == Layer.DATABASE_QUERY for h in result.hypotheses)
        if sql_share_of_latency_pct >= HIGH_SQL_SHARE_PCT and not has_database_hypothesis:
            raise ToolError(
                f"the measured baseline shows {sql_share_of_latency_pct:.1f}% "
                "of latency inside SQL, but none of your hypotheses names the "
                "database_query layer. That is a contradiction between the "
                "evidence and the diagnosis, not a legitimate competing theory.",
                suggestion="Name the database cost directly: which query, why "
                           "it is slow (no_index_used? rows examined?), and "
                           "how many of the measured ms it accounts for. Use "
                           "explain/list_indexes on the busiest digest if you "
                           "have not already.",
            )
        database_ms = sum(
            h.attributed_ms for h in result.hypotheses
            if h.layer == Layer.DATABASE_QUERY)
        if (sql_share_of_latency_pct <= LOW_SQL_SHARE_PCT
                and measured_p50_ms > 0
                and database_ms >= measured_p50_ms * MAJORITY_ATTRIBUTION_SHARE):
            raise ToolError(
                f"a database_query hypothesis attributes {database_ms:.1f} ms "
                f"of the measured {measured_p50_ms:.1f} ms p50, but the "
                f"measured baseline shows only {sql_share_of_latency_pct:.1f}% "
                "of latency inside SQL. The diagnosis claims the database "
                "dominates where the measurement shows it does not.",
                suggestion="Reduce the database_query attribution to match "
                           "what sql_share_of_latency_pct actually shows, and "
                           "look elsewhere for the rest.",
            )

    # ── 3. the ranking must refer to hypotheses that exist ──────────────
    known = {h.id for h in result.hypotheses}
    unknown = [i for i in result.ranked_ids if i not in known]
    if unknown:
        raise ToolError(
            f"ranked_ids references unknown hypothesis id(s): {', '.join(unknown)}. "
            f"Declared ids are: {', '.join(sorted(known))}",
            suggestion="Use exactly the ids you gave the hypotheses.",
        )
    missing = sorted(known - set(result.ranked_ids))
    if missing:
        raise ToolError(
            f"ranked_ids omits hypothesis id(s): {', '.join(missing)}. "
            "Every hypothesis must be ranked.",
            suggestion="Include all ids in ranked_ids, most significant first.",
        )

    # ── 4. a probe that separates nothing is not an experiment ──────────
    if not result.probes_run:
        raise ToolError(
            "probes_run is empty. You must run at least one probe with run_probe "
            "and report it before ranking hypotheses.",
            suggestion="Call run_probe to sweep page_size, then report each probe "
                       "with the hypothesis ids it separates.",
        )

    discriminating = [p for p in result.probes_run if p.discriminates_between]
    if not discriminating:
        raise ToolError(
            f"none of the {len(result.probes_run)} reported probes declares "
            "discriminates_between. A probe that cannot separate two hypotheses is "
            "not an experiment.",
            suggestion="For each probe, list the hypothesis ids whose predictions it "
                       "distinguishes.",
        )

    bad_refs = sorted(
        {
            hid
            for p in result.probes_run
            for hid in p.discriminates_between
            if hid not in known
        }
    )
    if bad_refs:
        raise ToolError(
            f"probes_run.discriminates_between references unknown hypothesis id(s): "
            f"{', '.join(bad_refs)}",
            suggestion="Reference only ids you declared in hypotheses.",
        )

    # A self-reported probe list is a claim. This is the check against what
    # actually ran, so the agent cannot describe an experiment it never did.
    if probe_log is not None and len(probe_log) == 0:
        raise ToolError(
            f"you reported {len(result.probes_run)} probe(s), but no probe was actually "
            "executed against the app. Reported probes must be real.",
            suggestion="Call run_probe for real before reporting probes_run.",
        )

    # ── 5. the set must span layers ─────────────────────────────────────
    layers = {h.layer for h in result.hypotheses}
    if len(layers) < MIN_LAYERS:
        only = ", ".join(sorted(l.value for l in layers))
        raise ToolError(
            f"all {len(result.hypotheses)} hypotheses sit in the same layer ({only}). "
            f"At least {MIN_LAYERS} different layers are required — otherwise you have "
            "one theory stated three ways, not competing explanations.",
            suggestion="Available layers: database_query, orm_hydration, "
                       "application_logic, serialization, network, other.",
        )

    return result


def _explain_validation_error(exc: ValidationError) -> str:
    """Turn pydantic's error list into instructions a model can act on."""
    lines: list[str] = []
    for err in exc.errors()[:12]:
        loc = ".".join(str(part) for part in err["loc"])
        lines.append(f"  - {loc}: {err['msg']}")
    return "\n".join(lines)
