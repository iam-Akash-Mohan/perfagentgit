"""Can this bottleneck be fixed without changing behaviour? Code decides.

The run this phase exists to stop went like this: diagnosis attributed 1400 ms
to response serialization and 200 ms to ORM hydration, the patch agent took the
bigger number three times, and the oracle rejected all three — because the
response body IS the API contract and the only way to serialize less is to
return less. Ranking by `attributed_ms` cannot see that. Nothing read
`response_shape.unbounded_collections`, which says it in one field.

So every hypothesis gets a second number: the probability that a
behaviour-preserving fix exists. It comes from a FIXED POLICY TABLE, in the
order written below, FIRST MATCH WINS. No model is asked, which is why
`FixabilityAssessment.assessed_by` is a `Literal["code"]`.

One deviation from the table as written in `sessions/FINISH-LINE.md`, flagged
rather than silent. The batchable row is specified as `layer ==
"database_query"` and a per-row data access; the same document's worked example
expects 0.90 for a hypothesis whose layer is `orm_hydration` ("ORM hydration
and per-row enrichment, including lazy loading of invoices"). Both cannot hold.
A lazy collection load per row IS a database access — the layer label only
records where the milliseconds landed, and a fetch join fixes it either way —
so the rule accepts both layers and still requires a real per-row access to
match. The alternative reading ranks the same way here but for a worse reason:
the ORM hypothesis would land on the 0.40 default.
"""

from __future__ import annotations

import re
from typing import Iterable

from ..contracts import (DataAccess, EndpointModel, FixabilityAssessment,
                         Hypothesis, HypothesisSet, Layer, PerRowWork,
                         RankedHypothesis, ResponseShape, TableInfo)

# A hypothesis about the response body, in a layer where that is what is being
# described. Used only after the unbounded-collection list is known non-empty.
_RESPONSE_WORDS = (
    "response", "body", "payload", "bytes", "dto", "json", "serial",
    "response size", "message converter",
)

# The signature of work that happens once per result row, in prose. A
# hypothesis that says this is talking about the per-row path even when it
# names no symbol from the map.
_PER_ROW_WORDS = (
    "per row", "per-row", "per result row", "each row", "each result",
    "once per row", "n+1", "n + 1", "lazy", "lazily", "per customer",
    "for each",
)

# What "the index is missing" sounds like in evidence the diagnosis quoted.
_NO_INDEX_WORDS = (
    "no_index_used", "no index used", "no index", "without an index",
    "full scan", "full table scan", "table scan", "type=all", "rows examined",
)

# A projection only pays when most of the row is read and thrown away.
PROJECTION_RATIO = 0.5

# How many distinctive words a hypothesis and a per-row work item must share
# before the item is treated as what the hypothesis is about. One shared word
# is a coincidence; the description and the claim are both prose.
MIN_WORK_OVERLAP = 2

_STOPWORDS = frozenset("""
a an and are as at be by for from has have in into is it its of on or that the
then this to with each one per row rows work does done over new not but which
""".split())

_WORD = re.compile(r"[a-z][a-z0-9_]{3,}")

# Hypothesis prose arrives with typographic dashes and narrow spaces in it —
# the recorded artifact spells "per‑row" with U+2011 — and a keyword table that
# does not normalise them silently matches nothing.
_PUNCTUATION = {
    "\u2010": "-", "\u2011": "-", "\u2012": "-", "\u2013": "-",
    "\u2014": "-", "\u2015": "-", "\u2212": "-",
    "\u00a0": " ", "\u202f": " ", "\u2009": " ",
}


# ──────────────────────────────  text helpers  ───────────────────────────


def normalise(text: str) -> str:
    for odd, plain in _PUNCTUATION.items():
        text = text.replace(odd, plain)
    return text.lower()


def hypothesis_text(hypothesis: Hypothesis) -> str:
    """Everything the model said about this hypothesis, as one haystack."""
    parts = [hypothesis.claim, hypothesis.mechanism, hypothesis.prediction,
             *hypothesis.evidence]
    return normalise(" ".join(parts))


def mentions(text: str, term: str) -> bool:
    """Whole-word match, tolerating a plural `s`.

    `invoice` has to match "the invoices collection" — the map names tables in
    the singular and prose uses the plural — but must not match `invoiced` or
    a longer identifier that merely contains it.
    """
    term = normalise(term).strip()
    if not term:
        return False
    return re.search(rf"\b{re.escape(term)}s?\b", text) is not None


def _any(text: str, terms: Iterable[str]) -> bool:
    return any(term in text for term in terms)


def _significant(*phrases: str) -> set[str]:
    words = set()
    for phrase in phrases:
        for word in _WORD.findall(normalise(phrase or "")):
            if word not in _STOPWORDS:
                words.add(word.rstrip("s"))
    return words


# ─────────────────────────────  map accessors  ───────────────────────────


def unbounded_fields(shape: ResponseShape | None) -> list[str]:
    """Field names from `response_shape.unbounded_collections`, cleaned up.

    The schema asks for a bare field name, and models sometimes send
    "content (list of CustomerDto)" anyway. The parenthetical is dropped here
    rather than being allowed to poison every later match.
    """
    if shape is None:
        return []
    out: list[str] = []
    for entry in shape.unbounded_collections:
        name = str(entry).split("(")[0].strip().strip("`'\" ")
        if name:
            out.append(name)
    return out


def matching_per_row_access(hypothesis: Hypothesis,
                            model: EndpointModel) -> DataAccess | None:
    """The per-row database access this hypothesis is about, if any.

    Named match first — the trigger's last segment, or a table it reads —
    because that is evidence the hypothesis really is about this access. The
    per-row-prose fallback is second, and still requires the map to contain a
    per-row access: with no such access there is nothing to batch.
    """
    accesses = model.per_row_accesses()
    if not accesses:
        return None
    text = hypothesis_text(hypothesis)

    for access in accesses:
        symbol = access.trigger.split(".")[-1].split("(")[0].strip()
        if symbol and mentions(text, symbol):
            return access
        if any(mentions(text, table) for table in access.tables):
            return access

    if _any(text, _PER_ROW_WORDS):
        return accesses[0]
    return None


def uncovered_columns(table: TableInfo) -> list[str]:
    """Columns this endpoint reads that no existing index leads with.

    Leading column only, because that is what MySQL can use: an index on
    (customer_id, status) does nothing for a filter on `status` alone.
    """
    leading = {index.columns[0] for index in table.existing_indexes
               if index.columns}
    return [column for column in table.columns_read if column not in leading]


def covering_index(table: TableInfo, columns: list[str]) -> str | None:
    """The existing index that already covers `columns` as a prefix, if any."""
    wanted = [c.strip().lower() for c in columns if c and c.strip()]
    if not wanted:
        return None
    for index in table.existing_indexes:
        have = [c.lower() for c in index.columns]
        if have[:len(wanted)] == wanted:
            return index.name
    return None


def matching_per_row_work(hypothesis: Hypothesis,
                          model: EndpointModel) -> PerRowWork | None:
    """The unbounded per-row work item this hypothesis is about, if any."""
    text = hypothesis_text(hypothesis)
    best: tuple[int, PerRowWork] | None = None
    for work in model.unbounded_per_row_work():
        words = _significant(work.description, work.scales_with)
        hits = sum(1 for word in words if mentions(text, word))
        if hits >= MIN_WORK_OVERLAP and (best is None or hits > best[0]):
            best = (hits, work)
    return best[1] if best else None


def response_link(work: PerRowWork,
                  shape: ResponseShape | None) -> tuple[str, str] | None:
    """How this per-row work relates to the response, if at all.

    Three answers, because the policy table needs three. ("unbounded", field)
    means the work fills a collection whose size is the contract — removing the
    work removes bytes. ("response", name) means it produces something returned
    but not unbounded; that is neither free to delete nor obviously the
    contract, so it falls through to the default rather than being called
    "genuinely removable". None means the work's output never reaches the
    caller.
    """
    if shape is None:
        return None
    text = normalise(work.description)
    for field in unbounded_fields(shape):
        if mentions(text, field.rstrip("s")):
            return ("unbounded", field)
    for name in [shape.dto or "", *shape.fields_returned]:
        if name and mentions(text, name.rstrip("s")):
            return ("response", name)
    return None


# ───────────────────────────  the policy table  ──────────────────────────


def assess_fixability(hypothesis: Hypothesis,
                      model: EndpointModel) -> FixabilityAssessment:
    """How likely a behaviour-preserving fix exists. Fixed table, first match."""
    text = hypothesis_text(hypothesis)
    shape = model.response_shape
    fields = unbounded_fields(shape)

    def verdict(value: float, reason: str,
                blocked_by: str | None = None) -> FixabilityAssessment:
        return FixabilityAssessment(
            hypothesis_id=hypothesis.id, fixability=value, reason=reason,
            blocked_by=blocked_by,
        )

    # 1 ─ the response body is the contract ───────────────────────────────
    if fields:
        named = [f for f in fields if mentions(text, f)]
        about_body = (hypothesis.layer is Layer.SERIALIZATION
                      and _any(text, _RESPONSE_WORDS))
        if named or about_body:
            implicated = named or fields
            return verdict(
                0.05,
                "response size is the API contract; the oracle rejects any "
                f"change to it (unbounded response collection: "
                f"{', '.join(implicated)})",
                blocked_by=(
                    f"{', '.join(implicated)} "
                    f"{'is an unbounded response collection' if len(implicated) == 1 else 'are unbounded response collections'}"
                    f" in response_shape.unbounded_collections; the frozen "
                    f"oracle compares the response body byte for byte"
                ),
            )

    # 2 ─ a query that runs per row is batchable ──────────────────────────
    if hypothesis.layer in (Layer.DATABASE_QUERY, Layer.ORM_HYDRATION):
        access = matching_per_row_access(hypothesis, model)
        if access is not None:
            return verdict(
                0.90,
                "batchable: fetch join, @EntityGraph, or an IN query "
                f"({access.trigger} executes once per result row, "
                f"{access.file}:{access.line})",
            )

    # 3 ─ a missing index is additive ─────────────────────────────────────
    if _any(text, _NO_INDEX_WORDS):
        for table in model.tables:
            gaps = uncovered_columns(table)
            if gaps:
                return verdict(
                    0.95,
                    "additive index, zero behaviour risk "
                    f"({table.table} is read on {', '.join(gaps)} and no "
                    f"existing index leads with "
                    f"{'those columns' if len(gaps) > 1 else 'that column'})",
                )

    # 4 ─ a projection returns the same values ────────────────────────────
    if hypothesis.layer is Layer.ORM_HYDRATION:
        for table in model.tables:
            read, total = len(table.columns_read), len(table.all_columns)
            if total and read and read <= total * PROJECTION_RATIO:
                return verdict(
                    0.80,
                    "DTO projection returns the same values "
                    f"({table.table}: {read} of {total} columns are read)",
                )

    # 5/6 ─ unbounded per-row work, split by where its output goes ────────
    work = matching_per_row_work(hypothesis, model)
    if work is not None:
        link = response_link(work, shape)
        if link is not None and link[0] == "unbounded":
            field = link[1]
            return verdict(
                0.10,
                f"the work produces the contract ({work.description} feeds "
                f"the unbounded response collection '{field}')",
                blocked_by=(
                    f"'{field}' is an unbounded response collection and this "
                    f"work is what fills it ({work.file}:{work.line}); "
                    f"removing the work removes response bytes"
                ),
            )
        if link is None:
            return verdict(
                0.70,
                f"genuinely removable ({work.description} is unbounded per-row "
                f"work at {work.file}:{work.line} and does not feed the "
                f"response)",
            )

    # 7 ─ serialization with nothing unbounded in the response ────────────
    if hypothesis.layer is Layer.SERIALIZATION:
        return verdict(
            0.30,
            "library-level, rarely beatable by hand (no unbounded response "
            "collection is implicated)",
        )

    # 8 ─ the network is not a code change ────────────────────────────────
    if hypothesis.layer is Layer.NETWORK:
        return verdict(
            0.15,
            "out of scope for a code change",
            blocked_by="transfer time is not changed by a patch to this "
                       "endpoint's code",
        )

    # 9 ─ unknown ─────────────────────────────────────────────────────────
    return verdict(0.40, "unknown; neither favoured nor excluded")


# ────────────────────────────  expected value  ───────────────────────────


def rank_hypotheses(hypotheses: HypothesisSet,
                    model: EndpointModel) -> list[RankedHypothesis]:
    """Hypotheses by `attributed_ms * fixability`, largest first.

    This single multiplication is the session: it puts a 200 ms cause with a
    clean fix above a 1400 ms cause that is the API contract. Ties break on
    attributed milliseconds so a genuinely bigger cost still wins when the
    expected values are equal.
    """
    ranked: list[RankedHypothesis] = []
    for h in hypotheses.hypotheses:
        assessment = assess_fixability(h, model)
        ranked.append(RankedHypothesis(
            hypothesis_id=h.id,
            claim=h.claim,
            layer=h.layer.value,
            attributed_ms=h.attributed_ms,
            fixability=assessment,
            expected_value_ms=round(h.attributed_ms * assessment.fixability, 3),
        ))
    ranked.sort(key=lambda r: (r.expected_value_ms, r.attributed_ms),
                reverse=True)
    return ranked
