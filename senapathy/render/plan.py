"""The plan artifact, rendered for a person.

Two things have to be legible at a glance: that the ranking changed, and why.
So the expected-value arithmetic is printed in full — attributed milliseconds,
fixability, product — and every fixability reason is printed next to the
hypothesis it belongs to. A 0.05 that arrives without its reason looks like the
tool being timid; with the reason it reads as the finding it is.
"""

from __future__ import annotations

from ..contracts import Plan, RankedHypothesis

BLOCKED = "BLOCKED"
UNCLASSIFIED = "NOT CLASSIFIED"


def render_plan(plan: Plan) -> str:
    out: list[str] = []
    out.extend(_header(plan))
    out.append("")
    out.extend(_ranking(plan))
    if plan.candidates:
        out.append("")
        out.extend(_candidates(plan))
        out.append("")
        out.extend(_selection(plan))
    return "\n".join(out)


def render_ranking(plan: Plan) -> str:
    """Just the ranking. What `--rank-only` prints, with no model involved."""
    return "\n".join([*_header(plan), "", *_ranking(plan)])


def _header(plan: Plan) -> list[str]:
    return [
        "",
        f"PLAN  ·  {plan.endpoint}",
        f"  measured p50   {plan.measured_total_ms:.1f} ms",
        f"  ranked by      attributed_ms × fixability  (decided by "
        f"{plan.decided_by})",
    ]


def _ranking(plan: Plan) -> list[str]:
    out = ["RANKED HYPOTHESES"]
    if not plan.ranked:
        out.append("  none")
        return out
    out.append("")
    for position, entry in enumerate(plan.ranked, start=1):
        out.extend(_one_hypothesis(position, entry, plan.considered))
    return out


def _one_hypothesis(position: int, entry: RankedHypothesis,
                    considered: list[str]) -> list[str]:
    mark = " ← planned" if entry.hypothesis_id in considered else ""
    if not entry.fixability.classified:
        mark = f"  [{UNCLASSIFIED}]{mark}"
    out = [f"  {position}. [{entry.hypothesis_id}] {entry.claim}{mark}",
           f"     layer         {entry.layer}"]

    if entry.fixability.classified:
        out.append(f"     attributed    {entry.attributed_ms:.1f} ms"
                   f"   ×  fixability {entry.fixability.fixability:.2f}"
                   f"   =  expected value {entry.expected_value_ms:.1f} ms")
    else:
        # Deliberately NOT the same line shape. An unassessed hypothesis
        # printed as "0.40 × 200 ms = 80 ms" reads as a finding, and that is
        # how one outranked a classified 0.90 with a known fix.
        out.append(f"     attributed    {entry.attributed_ms:.1f} ms"
                   f"   ×  placeholder {entry.fixability.fixability:.2f}"
                   f"   →  {entry.expected_value_ms:.1f} ms NOTIONAL")
        out.append(f"     {UNCLASSIFIED}  no policy row matched this "
                   f"hypothesis, so it is ranked BELOW every classified one "
                   f"regardless of that number")

    out.append(f"     why           {entry.fixability.reason}")
    if entry.fixability.blocked_by:
        out.append(f"     {BLOCKED} BY    {entry.fixability.blocked_by}")
    out.append("")
    return out


def _candidates(plan: Plan) -> list[str]:
    out = [f"CANDIDATE FIXES ({len(plan.candidates)})", ""]
    for candidate in plan.candidates:
        chosen = " ← SELECTED" if candidate.id == plan.selected_id else ""
        flag = "  (requires a flag)" if candidate.requires_flag else ""
        out.append(f"  [{candidate.id}] {candidate.approach}  →  "
                   f"{candidate.targets_hypothesis}{chosen}")
        out.append(f"     risk          T{candidate.risk_tier.value}{flag}"
                   f"   ·  expected value {candidate.expected_value_ms:.1f} ms"
                   f"   ·  response bytes preserved: "
                   f"{'claimed' if candidate.preserves_response_bytes else 'NO'}")
        out.append(f"     change        {candidate.description}")
        out.append(f"     mechanism     {candidate.mechanism_changed}")
        out.append(f"     predicts      {candidate.predicted_effect}")
        if candidate.sql_change:
            out.append(f"     sql           {candidate.sql_change.strip()}")
        out.append(f"     files         {', '.join(candidate.files_to_touch)}")
        for risk in candidate.what_could_break:
            out.append(f"     could break   - {risk}")
        if candidate.not_selectable_because:
            out.append("     NOT SELECTABLE  the arms cannot be isolated "
                       "for this schema change:")
            for line in candidate.not_selectable_because.splitlines():
                out.append(f"       {line}")
        out.append("")
    return out


def _selection(plan: Plan) -> list[str]:
    out = ["SELECTION"]
    if plan.selected_id is None:
        out.append("  nothing selected")
    else:
        out.append(f"  selected      {plan.selected_id}")
    out.append(f"  because       {plan.selection_reason}")
    if plan.not_selected:
        out.append("")
        out.append("  not selected:")
        for rejection in plan.not_selected:
            out.append(f"    [{rejection.candidate_id}] {rejection.reason}")
    return out
