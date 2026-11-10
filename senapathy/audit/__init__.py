"""The SQL audit sweep: ordinary code, not an agent.

Every application digest captured at BASELINE is reconstructed into an
executable statement, EXPLAIN ANALYZE'd, and classified — unconditionally,
every run, rather than only when a diagnosis agent happens to call
`explain`. DIAGNOSE and the plan brief reconcile against what this phase
found; neither re-measures it.
"""

from __future__ import annotations

from .candidates import propose_index_candidate, propose_rewrite_candidate
from .classify import OPTIMAL_RATIO, classify
from .reconstruct import (UnresolvableStatement, extract_equality_columns,
                          extract_tables, resolve_statement,
                          table_alias_map, table_reference_end)
from .run import ARTIFACT, load_sql_audit, run_sql_audit

__all__ = [
    "ARTIFACT",
    "OPTIMAL_RATIO",
    "UnresolvableStatement",
    "classify",
    "extract_equality_columns",
    "extract_tables",
    "load_sql_audit",
    "propose_index_candidate",
    "propose_rewrite_candidate",
    "resolve_statement",
    "run_sql_audit",
    "table_alias_map",
    "table_reference_end",
]
