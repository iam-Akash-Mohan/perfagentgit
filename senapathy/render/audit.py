"""The SQL audit artifact, rendered for a person.

One line per statement, worst-ratio first, with the specific finding and the
proven candidate if one exists — the same discipline as the plan render: a
classification that arrives without a number next to it reads as an opinion,
not a finding.
"""

from __future__ import annotations

from ..contracts.audit import AuditedStatement, SqlAudit


def render_audit(audit: SqlAudit) -> str:
    out: list[str] = [
        "",
        f"SQL AUDIT  ·  {audit.endpoint}",
        f"  measured total  {audit.measured_total_ms:.1f} ms",
        f"  statements      {len(audit.statements)} "
        f"({len(audit.non_optimal())} not optimal)",
        "",
    ]
    ranked = sorted(
        audit.statements,
        key=lambda s: (s.rows_examined / s.rows_returned if s.rows_returned
                       else float(s.rows_examined)),
        reverse=True,
    )
    for statement in ranked:
        out.extend(_statement(statement))
        out.append("")
    return "\n".join(out)


def _statement(statement: AuditedStatement) -> list[str]:
    lines = [
        f"[{statement.classification}]  {statement.calls}× calls, "
        f"{statement.total_ms:.1f} ms total",
        f"  {statement.digest_text}",
    ]
    if statement.limit_offset_source is not None:
        request = (f", request {statement.limit_offset_request_id}"
                   if statement.limit_offset_request_id else "")
        caveat = (" -- a guess, not a measured page depth"
                  if statement.limit_offset_source == "fallback" else "")
        lines.append(
            f"  LIMIT/OFFSET source: {statement.limit_offset_source}{request}"
            f"{caveat}"
        )
    if statement.classification == "UNANALYSABLE":
        lines.append(f"  reason: {statement.unanalysable_reason}")
        return lines

    ratio = (statement.rows_examined / statement.rows_returned
             if statement.rows_returned else statement.rows_examined)
    lines.append(
        f"  rows examined {statement.rows_examined:,}  ->  "
        f"rows returned {statement.rows_returned:,}  "
        f"({ratio:,.0f}x)  actual {statement.actual_ms:.1f} ms"
    )
    if statement.non_sargable_marker:
        lines.append(f"  marker: {statement.non_sargable_marker}")
    if statement.uses_filesort:
        lines.append("  uses filesort")
    if statement.uses_temporary:
        lines.append("  uses a temporary table")

    if statement.index_candidate is not None:
        candidate = statement.index_candidate
        if candidate.already_covered_by is not None:
            lines.append(
                f"  INDEX CANDIDATE: already covered by "
                f"{candidate.already_covered_by} — not actually missing"
            )
        else:
            lines.append(
                f"  INDEX CANDIDATE: {candidate.table}"
                f"({', '.join(candidate.columns)})  "
                f"rows examined {candidate.rows_examined_before:,} -> "
                f"{candidate.rows_examined_after:,}, "
                f"{candidate.actual_ms_before:.1f} -> "
                f"{candidate.actual_ms_after:.1f} ms"
            )
    if statement.rewrite_candidate is not None:
        candidate = statement.rewrite_candidate
        verdict = "verified" if candidate.equivalence_verified else "NOT verified"
        improvement = (f"{candidate.improvement_pct:+.1f}%"
                       if candidate.improvement_pct is not None else "no improvement")
        lines.append(
            f"  REWRITE CANDIDATE ({verdict}, {improvement}): "
            f"{candidate.sql_after}"
        )
    return lines
