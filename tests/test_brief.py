"""The plan brief's baseline section, extended by the SQL audit.

`_baseline_sql` is tested directly rather than through `render_brief`'s full
signature: the audit extension touches exactly this function, and building a
full `RankedHypothesis`/`EndpointModel` fixture just to reach it would test
unrelated rendering this change does not touch.
"""

from __future__ import annotations

from senapathy.contracts import SqlDigestStat
from senapathy.contracts.audit import (AuditedStatement, IndexCandidate,
                                       RewriteCandidate, SqlAudit)
from senapathy.plan.brief import _baseline_sql

DIGEST = SqlDigestStat(
    digest="d1", sample_sql="select * from invoice where customer_id = ?",
    calls=5, total_ms=900.0, mean_ms=180.0, rows_examined=1_500_000,
    rows_sent=3,
)


def _audit(statement: AuditedStatement) -> SqlAudit:
    return SqlAudit(endpoint="GET /x", measured_total_ms=1000.0,
                    statements=[statement])


def test_baseline_section_is_unchanged_when_no_audit_is_given() -> None:
    lines = _baseline_sql([DIGEST], None)
    assert any("digest d1" in line for line in lines)
    assert not any("audit" in line for line in lines)


def test_a_missing_index_finding_is_rendered_with_its_numbers() -> None:
    statement = AuditedStatement(
        digest="d1", digest_text=DIGEST.sample_sql, calls=5, total_ms=900.0,
        classification="MISSING_INDEX", rows_examined=1_500_000,
        rows_returned=3, actual_ms=900.0,
    )
    lines = _baseline_sql([DIGEST], _audit(statement))
    text = "\n".join(lines)
    assert "[MISSING_INDEX]" in text
    assert "1,500,000" in text
    assert "500,000x" in text  # 1,500,000 / 3, rounded


def test_an_unanalysable_statement_shows_its_reason_not_numbers() -> None:
    statement = AuditedStatement(
        digest="d1", digest_text=DIGEST.sample_sql, calls=5, total_ms=900.0,
        classification="UNANALYSABLE",
        unanalysable_reason="no real value could be sampled",
    )
    lines = _baseline_sql([DIGEST], _audit(statement))
    text = "\n".join(lines)
    assert "UNANALYSABLE" in text
    assert "no real value could be sampled" in text


def test_an_index_candidate_already_covered_is_reported_as_a_refusal() -> None:
    statement = AuditedStatement(
        digest="d1", digest_text=DIGEST.sample_sql, calls=5, total_ms=900.0,
        classification="MISSING_INDEX", rows_examined=1_500_000,
        rows_returned=3, actual_ms=900.0,
        index_candidate=IndexCandidate(
            table="invoice", columns=["customer_id"],
            rows_examined_before=1_500_000, rows_examined_after=1_500_000,
            actual_ms_before=900.0, actual_ms_after=900.0,
            already_covered_by="idx_invoice_customer_id_status",
        ),
    )
    lines = _baseline_sql([DIGEST], _audit(statement))
    text = "\n".join(lines)
    assert "already covered by idx_invoice_customer_id_status" in text
    assert "not actually missing" in text


def test_a_measured_index_candidate_is_rendered_with_before_and_after() -> None:
    statement = AuditedStatement(
        digest="d1", digest_text=DIGEST.sample_sql, calls=5, total_ms=900.0,
        classification="MISSING_INDEX", rows_examined=1_500_000,
        rows_returned=3, actual_ms=900.0,
        index_candidate=IndexCandidate(
            table="invoice", columns=["customer_id"],
            rows_examined_before=1_500_000, rows_examined_after=3,
            actual_ms_before=900.0, actual_ms_after=0.9,
        ),
    )
    lines = _baseline_sql([DIGEST], _audit(statement))
    text = "\n".join(lines)
    assert "MEASURED index candidate: invoice(customer_id)" in text
    assert "1,500,000 -> 3" in text
    # The actionable instruction a model needs to find in order to cite
    # this finding at all -- without it, add_index has no selectable path.
    assert 'audit_digest="d1"' in text
    assert "invoice(customer_id)" in text
    assert "ONLY path to selection" in text


def test_a_rewrite_candidate_is_rendered_with_its_verified_sql() -> None:
    statement = AuditedStatement(
        digest="d1", digest_text="select * from invoice where DATE(created_at) >= ?",
        calls=5, total_ms=900.0, classification="NON_SARGABLE",
        rows_examined=1_500_000, rows_returned=3, actual_ms=900.0,
        rewrite_candidate=RewriteCandidate(
            sql_before="select * from invoice where DATE(created_at) >= %s",
            sql_after="select * from invoice where created_at >= %s",
            equivalence_verified=True, equivalence_note="EQUIVALENT",
            rows_examined_before=1_500_000, rows_examined_after=3,
            benchmark_rounds=5, original_ms_samples=[20.0] * 5,
            candidate_ms_samples=[2.0] * 5, improvement_pct=90.0,
        ),
    )
    lines = _baseline_sql([DIGEST], _audit(statement))
    text = "\n".join(lines)
    assert "MEASURED rewrite candidate (equivalence verified)" in text
    assert "select * from invoice where created_at >= %s" in text


def test_a_digest_with_no_corresponding_audit_entry_is_unannotated() -> None:
    """Two digests at baseline, one audited -- only that one gets the
    extra lines; the other's section looks exactly as it did before."""
    other = SqlDigestStat(
        digest="d2", sample_sql="select 1", calls=1, total_ms=1.0, mean_ms=1.0,
    )
    statement = AuditedStatement(
        digest="d1", digest_text=DIGEST.sample_sql, calls=5, total_ms=900.0,
        classification="OPTIMAL", rows_examined=10, rows_returned=5,
    )
    lines = _baseline_sql([DIGEST, other], _audit(statement))
    text = "\n".join(lines)
    assert text.count("digest d") == 2
    assert text.count("audit") == 1
