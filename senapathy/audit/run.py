"""The SQL audit sweep — ordinary code, not an agent.

Every application digest baseline captured gets reconstructed, EXPLAIN
ANALYZE'd, classified, and (for MISSING_INDEX / the one rewritable
NON_SARGABLE shape) given a measured candidate — every time, unconditionally,
rather than only when a diagnosis agent happens to call `explain` on it.
COL-04 is the proof this phase exists for: a full scan of 1.5M rows sat in
the baseline's own digest evidence while diagnosis theorised about ORM
hydration. This sweep would have found it without anyone asking.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..contracts import AuditedStatement, RequestCorpus, SqlAudit, SqlDigestStat
from ..measure.baseline import is_session_control
from ..measure.sql import MySQLTarget
from ..tools.db import DbToolbox
from ..tools.sample import ValueSampler
from .candidates import propose_index_candidate, propose_rewrite_candidate
from .classify import classify
from .reconstruct import resolve_statement

ARTIFACT = "sql_audit.json"


def run_sql_audit(
    *, endpoint: str, measured_total_ms: float,
    digests: list[SqlDigestStat], target: MySQLTarget,
    allowed_tables: list[str], run_dir: Path,
    baseline_hash: str | None = None,
    corpus: RequestCorpus | None = None,
) -> SqlAudit:
    """Audit every application digest `baseline` captured, write
    `sql_audit.json`, return the artifact.

    `digests` is `baseline.json`'s own `sql_digests` — the same evidence
    the plan brief's "BASELINE: WHAT ACTUALLY RAN" section already reads,
    extended here rather than re-measured from scratch.

    `corpus` is this run's own `corpus.json` — read for the real page/size
    every benchmark request actually used, so a paginated statement's
    LIMIT/OFFSET comes from a real measured page depth rather than a fixed
    guess whenever one is available (`reconstruct.py::_resolve_limit_offset`).
    """
    box = DbToolbox(target, allowed_tables)
    sampler = ValueSampler(target, allowed_tables)

    statements: list[AuditedStatement] = []
    for digest in digests:
        if is_session_control(digest):
            continue
        statements.append(_audit_one(box, sampler, target, digest, corpus))

    audit = SqlAudit(endpoint=endpoint, measured_total_ms=measured_total_ms,
                     baseline_hash=baseline_hash, statements=statements)
    path = Path(run_dir) / ARTIFACT
    path.write_text(
        json.dumps(audit.model_dump(mode="json"), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return audit


def load_sql_audit(path: str | Path) -> SqlAudit | None:
    """None, not an error, when no audit was ever written — a caller
    reconciling against it (diagnose, the plan brief) treats that as
    nothing to reconcile against, same discipline as an older baseline
    with no `sql_share_of_latency_pct`."""
    p = Path(path)
    if not p.exists():
        return None
    return SqlAudit.model_validate(json.loads(p.read_text(encoding="utf-8")))


def _audit_one(box: DbToolbox, sampler: Any, target: MySQLTarget,
              digest: SqlDigestStat,
              corpus: RequestCorpus | None = None) -> AuditedStatement:
    executable, params, note, limit_offset_source, limit_offset_request_id = \
        resolve_statement(target, digest.digest, digest.sample_sql, sampler,
                          corpus=corpus)

    if executable is None:
        return AuditedStatement(
            digest=digest.digest, digest_text=digest.sample_sql,
            calls=digest.calls, total_ms=digest.total_ms,
            classification="UNANALYSABLE", unanalysable_reason=note,
        )

    try:
        reading = box.explain_measured(executable, params)
    except Exception as exc:  # noqa: BLE001 — a bad reconstruction is a finding
        return AuditedStatement(
            digest=digest.digest, digest_text=digest.sample_sql,
            executable_sql=executable, resolved_params=list(params),
            calls=digest.calls, total_ms=digest.total_ms,
            classification="UNANALYSABLE",
            unanalysable_reason=f"reconstructed ({note}) but EXPLAIN ANALYZE "
                                f"could not run it: {exc}",
            limit_offset_source=limit_offset_source,
            limit_offset_request_id=limit_offset_request_id,
        )

    classification, marker = classify(reading, executable)
    statement = AuditedStatement(
        digest=digest.digest, digest_text=digest.sample_sql,
        executable_sql=executable, resolved_params=list(params),
        calls=digest.calls, total_ms=digest.total_ms,
        classification=classification, non_sargable_marker=marker,
        access_types=reading.access_types, keys_used=reading.keys_used,
        rows_examined=reading.rows_examined, rows_returned=reading.rows_returned,
        actual_ms=reading.actual_ms, uses_filesort=reading.uses_filesort,
        uses_temporary=reading.uses_temporary,
        limit_offset_source=limit_offset_source,
        limit_offset_request_id=limit_offset_request_id,
    )

    if classification == "MISSING_INDEX":
        statement.index_candidate = propose_index_candidate(box, statement)
    elif classification == "NON_SARGABLE":
        statement.rewrite_candidate = propose_rewrite_candidate(box, statement)

    return statement
