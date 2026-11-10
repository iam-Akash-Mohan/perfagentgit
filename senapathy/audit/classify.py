"""Classifying one statement's own EXPLAIN ANALYZE plan.

The rows-examined to rows-returned ratio is the primary signal; the
specific syntax EXPLAIN prints decides WHICH of the non-optimal buckets it
falls into. Reuses the plan phase's own detectors
(`plan/fixability.py::non_sargable_marker`, `join_product_marker`) rather
than inventing a second copy that could disagree with the first.
"""

from __future__ import annotations

from ..contracts import AuditClassification
from ..plan.fixability import join_product_marker, non_sargable_marker
from ..tools.db import PlanReading

# Below this ratio, a statement examines roughly what it returns -- not
# OPTIMAL by some stricter bound no query ever clears, just not a finding
# this audit should raise. 3x gives room for an index that is selective
# but not perfectly so (a composite index's leading column, say) without
# excusing the ACTUAL full-scan case this audit exists to catch, where the
# ratio is routinely in the thousands or millions (COL-04: 1.5M rows
# examined for single-digit rows returned).
OPTIMAL_RATIO = 3.0

# `type=ALL` (full table scan) and `type=index` (full scan, in some index's
# own order rather than storage order) are BOTH genuine scans -- confirmed
# live, not theoretical: an InnoDB table is always clustered by its primary
# key, so a predicate no index serves can still report `type=index,
# key=PRIMARY, possible_keys=NULL` under an ORDER BY + LIMIT the optimizer
# finds attractive (walk the PK's own order, filter as it goes, stop once
# enough matches are found). `PlanReading.any_index_used` alone cannot tell
# that apart from a real, selective lookup — it only checks whether SOME key
# name is present, and PRIMARY is a key name. mini-spring's own `payment`
# table (no index on `paid_date`) produces exactly this shape for `WHERE
# paid_date = ? ORDER BY id LIMIT ?`, which `any_index_used` alone classified
# OVER_FETCHING -- a materially different finding from the MISSING_INDEX it
# actually is, with no index_candidate ever proposed for it as a result.
_SCAN_ACCESS_TYPES = frozenset({"all", "index"})


def _is_a_scan(reading: PlanReading, table: str) -> bool:
    """True when `table`'s own access in this plan is a scan: no key used
    at all, OR a key is reported but the access TYPE is still ALL/index —
    the clustered-primary-key case above."""
    if not reading.keys_used.get(table):
        return True
    return reading.access_types.get(table, "").lower() in _SCAN_ACCESS_TYPES


def classify(reading: PlanReading, sql_text: str) -> tuple[
    AuditClassification, str | None
]:
    """(classification, marker). `marker` is the specific EXPLAIN-plan
    syntax that justifies a non-OPTIMAL verdict, when there is one --
    `non_sargable_marker`'s own text for NON_SARGABLE, `join_product_
    marker`'s for BAD_JOIN_ORDER, None otherwise (MISSING_INDEX and
    OVER_FETCHING are read directly off the numbers, not a plan phrase)."""
    ratio = (reading.rows_examined / reading.rows_returned
            if reading.rows_returned else
            (float(reading.rows_examined) if reading.rows_examined else 0.0))

    if ratio <= OPTIMAL_RATIO:
        return "OPTIMAL", None

    # Checked FIRST and against the statement's own SQL text: a
    # function-wrapped column or leading wildcard is the single most
    # actionable finding this audit can make (a mechanical, proven rewrite
    # exists for at least one shape of it -- see candidates.py), and it is
    # checked whether or not an index is currently used at all, because the
    # whole point of this classification is "an index COULD serve this
    # predicate but the predicate's own shape defeats it."
    sargable_marker = non_sargable_marker(sql_text)
    if sargable_marker is not None:
        return "NON_SARGABLE", sargable_marker

    scanning = [t for t in reading.access_types if _is_a_scan(reading, t)]
    multi_table = len(reading.access_types) > 1
    join_marker = join_product_marker(reading.tree) if multi_table else None
    if join_marker is not None:
        return "BAD_JOIN_ORDER", join_marker
    if multi_table and scanning:
        return "BAD_JOIN_ORDER", None

    if scanning:
        return "MISSING_INDEX", None

    # Every table's access is a genuine seek, the predicate is sargable,
    # this is not a bad join -- and the ratio is still bad. The index
    # exists but is not selective enough for this particular filter.
    return "OVER_FETCHING", None
