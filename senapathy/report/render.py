"""The evidence ledger: a renderer that cannot state an unbacked number.

Invariant 4 says every latency, count, percentage and path in a report is
templated from a recorded measurement. That is easy to agree with and easy to
violate by accident — one f-string with a hand-typed number and the report is
lying. So the mechanism is structural rather than procedural:

1. A value reaches a template only as `Backed(value, ref)`, where `ref` is an
   `ArtifactRef` that resolves to a file on disk whose content still hashes to
   `ref.hash`. A stale or invented ref is refused.
2. The template's own literal text may not contain a numeral. If it could,
   a hardcoded "40% faster" would sail past every check on the slots. Digits
   inside identifiers (`p95`, `h1`, `CustomerDto2`) are allowed, because those
   are labels, not claims.

Both refusals raise ContractViolation. `tests/test_gate.py` asserts them; that
test is the ledger.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..contracts import ArtifactRef
from ..errors import ContractViolation
from ..ids import content_hash

# A numeral that is NOT part of an identifier: not preceded by a letter,
# underscore or digit-in-a-word. `p95` and `h1` pass; `95` and `1417.2` do not.
_BARE_NUMERAL = re.compile(r"(?<![A-Za-z0-9_.])\d")

_SLOT = re.compile(r"\{([a-zA-Z_][a-zA-Z0-9_]*)\}")

# Ledger keys with a mandatory source. `terminal_state` may only be backed
# by the GateDecision that computed it — backing it with any other artifact
# (the perf report, say — a document the gate decision is DERIVED from, not
# the other way around) is how a report ends up stating a terminal state its
# own citation never produced. The confidence-band keys carry the same
# requirement for the same reason: the band is computed from the gate's own
# results plus a fixed rule table, never narrated, so it may only point back
# at the decision that produced it.
_RESTRICTED_SOURCES: dict[str, str] = {
    "terminal_state": "gate_decision",
    "confidence_level": "gate_decision",
    "confidence_reason": "gate_decision",
    "confidence_inputs": "gate_decision",
    "confidence_would_raise": "gate_decision",
}


@dataclass(frozen=True)
class Backed:
    """A value and the artifact it came from. The only thing a slot accepts."""

    value: Any
    ref: ArtifactRef

    def __str__(self) -> str:  # pragma: no cover - rendering goes through format
        return str(self.value)


class EvidenceLedger:
    """Holds backed values and renders templates that can only use them."""

    def __init__(self, *, verify_refs: bool = True) -> None:
        self._entries: dict[str, Backed] = {}
        self._verify_refs = verify_refs

    # ─────────────────────────────  recording  ───────────────────────────

    def record(self, key: str, value: Any, ref: ArtifactRef) -> Backed:
        """Register a value together with the artifact that produced it."""
        required_kind = _RESTRICTED_SOURCES.get(key)
        if required_kind is not None and ref.kind != required_kind:
            raise ContractViolation(
                f"'{key}' must be backed by the '{required_kind}' artifact "
                f"— the GateDecision that actually determined it — not "
                f"'{ref.kind}'. Citing a different artifact here is exactly "
                "the bug that let a rendered report show a state or a "
                "confidence band its own citation never produced."
            )
        if self._verify_refs:
            self._assert_ref_resolves(key, ref)
        backed = Backed(value=value, ref=ref)
        self._entries[key] = backed
        return backed

    def record_from(
        self, ref: ArtifactRef, payload: dict[str, Any], mapping: dict[str, str]
    ) -> None:
        """Record several values out of one artifact's payload.

        `mapping` is {ledger key: dotted path into the payload}. Pulling values
        by path from the artifact we just hashed means the recorded number and
        the artifact cannot drift apart in between.
        """
        for key, dotted in mapping.items():
            self.record(key, _dig(payload, dotted, key), ref)

    def _assert_ref_resolves(self, key: str, ref: ArtifactRef) -> None:
        path = Path(ref.path)
        if not path.exists():
            raise ContractViolation(
                f"the value for '{key}' cites artifact {ref.kind} at {ref.path}, "
                "which does not exist. A citation that does not resolve is not "
                "evidence."
            )
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ContractViolation(
                f"the artifact backing '{key}' ({ref.path}) is not valid JSON: {exc}"
            ) from exc
        digest = content_hash(payload)
        if digest != ref.hash:
            raise ContractViolation(
                f"the artifact backing '{key}' ({ref.path}) has changed since it "
                f"was cited: ref says {ref.hash}, content hashes to {digest}. The "
                "report would quote a number that no longer exists."
            )

    # ─────────────────────────────  rendering  ───────────────────────────

    def render(self, template: str, **extra: Any) -> str:
        """Fill `{slot}` placeholders from the ledger. Refuse anything else.

        Two refusals, and they are the whole point:
          * a slot with no backed value behind it
          * a template whose own prose contains a bare numeral
        """
        self._assert_no_literal_numbers(template)

        available: dict[str, Backed] = {**self._entries}
        for key, value in extra.items():
            if not isinstance(value, Backed):
                raise ContractViolation(
                    f"slot '{key}' was passed as a bare {type(value).__name__} "
                    f"({value!r}). Every value in a report must arrive as "
                    "Backed(value, artifact_ref) so the number can be traced to "
                    "the measurement that produced it."
                )
            if self._verify_refs:
                self._assert_ref_resolves(key, value.ref)
            available[key] = value

        missing = [
            name for name in dict.fromkeys(_SLOT.findall(template))
            if name not in available
        ]
        if missing:
            raise ContractViolation(
                f"the template has slot(s) with no artifact behind them: "
                f"{', '.join(missing)}. Record them on the ledger with the "
                "artifact they came from, or remove the claim."
            )

        return _SLOT.sub(lambda m: str(available[m.group(1)].value), template)

    def _assert_no_literal_numbers(self, template: str) -> None:
        """A numeral in the template's own text bypasses every slot check."""
        literal = _SLOT.sub("", template)
        offenders = [
            line.strip() for line in literal.splitlines()
            if _BARE_NUMERAL.search(line)
        ]
        if offenders:
            raise ContractViolation(
                "the template contains hardcoded numeral(s), which would appear "
                "in the report with no artifact behind them:\n  "
                + "\n  ".join(offenders[:5])
                + "\nPut the number on the ledger and reference it as a {slot}."
            )

    # ──────────────────────────────  reading  ────────────────────────────

    def get(self, key: str) -> Backed:
        if key not in self._entries:
            raise ContractViolation(f"'{key}' is not on the evidence ledger")
        return self._entries[key]

    def provenance(self) -> list[dict[str, str]]:
        """Every claim and the artifact behind it, for the report's footer."""
        return [
            {
                "claim": key,
                "value": str(backed.value),
                "artifact": backed.ref.kind,
                "hash": backed.ref.hash,
                "path": backed.ref.path,
            }
            for key, backed in sorted(self._entries.items())
        ]

    def __contains__(self, key: str) -> bool:
        return key in self._entries

    def __len__(self) -> int:
        return len(self._entries)


def _dig(payload: dict[str, Any], dotted: str, key: str) -> Any:
    node: Any = payload
    for part in dotted.split("."):
        if isinstance(node, list):
            node = node[int(part)]
        elif isinstance(node, dict) and part in node:
            node = node[part]
        else:
            raise ContractViolation(
                f"'{key}' cannot be recorded: the artifact has no value at "
                f"'{dotted}'"
            )
    return node


# ────────────────────────────  the report body  ──────────────────────────
#
# Note every numeric slot below is a {placeholder}. There is not one digit in
# the literal text of these templates, and `_assert_no_literal_numbers` is what
# keeps it that way as they are edited.

_HEADER = """# Performance report: {endpoint}

**Terminal state:** `{terminal_state}`
**Confidence:** **{confidence_level}** — {confidence_reason}
**Gate:** {gate_summary}

<details>
<summary>Confidence inputs</summary>

{confidence_inputs}

{confidence_would_raise}
</details>

| | |
|---|---|
| Endpoint | `{endpoint}` |
| Branch | `{branch}` |
| Base commit | `{base_commit}` |
| Oracle | `{oracle_hash}` |
| Corpus | `{corpus_hash}` |

## What a human must check before merging

{fixability_check}

**Risk tier:** {risk_tier_check}

**What the oracle did not cover:** it proves blue and green are
byte-identical on every request in the benchmark and held-back sets counted
below, under Behaviour — nothing else. Parameter values outside those sets,
error paths, and concurrent access were not generated, not compared, and are
not covered by this verdict.

**Repository test suite:** {test_suite_check}

**Pre-existing risks** (found while tracing the endpoint, independent of
this patch — properties of the application today, on unmodified code):

{pre_existing_risks}

## The bottleneck

**Diagnosed cause:** {bottleneck_claim}

*Attributed:* {attributed_ms} ms of {baseline_p50} ms measured by the
BASELINE phase — a single-arm probe taken before any patch existed, and a
different measurement from the paired MEASURE-phase numbers in Before and
after below.

### Root cause

{narrative}

### Why not the bigger number

Causes are ranked by `attributed_ms x fixability` — the measured cost times the
probability that a fix exists which does not change behaviour. Fixability is
computed by code from a fixed policy table, never by a model. This is the
ranking that produced the fix below, and the reason each alternative lost:

{ranked_alternatives}

## What was tried

{patch_summary}

**Stated mechanism:** {mechanism_changed}

**Predicted effect:** {expected_effect}

**Approach:** `{approach}` (risk tier {risk_tier}, chosen over
{candidates_considered} candidate(s) in the plan)

### Candidates not selected

{rejected_candidates}

### Schema change

{index_evidence}

## Before and after

MEASURE-phase numbers: paired, interleaved (blue, green, blue, green) across
the benchmark set — a separate measurement from the BASELINE-phase p50 cited
above, which is why blue p50 here need not match it exactly.

| Measurement | Blue (base) | Green (patched) | Delta |
|---|---|---|---|
| p50 | {blue_p50} ms | {green_p50} ms | {p50_delta} |
| p95 | {blue_p95} ms | {green_p95} ms | {p95_delta} |
| p99 | {blue_p99} ms | {green_p99} ms | |
| SQL statements | {blue_sql} | {green_sql} | |
| Rows examined | {blue_rows} | {green_rows} | |

Paired samples per arm: {pair_count}.

Wilcoxon signed-rank p: {paired_p}. Mann-Whitney U p: {unpaired_p}.

## Reconciliation: predicted vs. delivered

The diagnosis attributed **{attributed_ms} ms** to the selected cause. The
patch delivered **{delivered_ms} ms** (p50 {blue_p50} ms -> {green_p50} ms,
measure phase) — {delivery_ratio}.

{reconciliation_note}

## Did the predicted mechanism actually move?

{mechanism_verdict}

{mechanism_note}

## Gates

| Gate | Result | Threshold | Actual |
|---|---|---|---|
{gate_rows}

## Files changed

{file_list}

Lines added: {lines_added}. Lines removed: {lines_removed}.

## Behaviour

Requests compared: {compared}. Identical: {identical}. Differing paths:
{diff_count}. Held-back requests: {held_back_count}.

Tables excluded from the database-state comparison as schema-management
bookkeeping, not application data: {excluded_tables}.

## Provenance

{provenance_rows}
"""


def render_report(ledger: EvidenceLedger, **slots: Backed) -> str:
    """Render report.md. Every quantity comes off the ledger."""
    return ledger.render(_HEADER, **slots)


# ─────────────────────────  the schema-candidate report  ─────────────────
#
# A SEPARATE template, not a branch inside `_HEADER`: the two reports make
# different KINDS of claim, and sharing one template is how an endpoint-
# level slot (`{blue_p50}`, `{p95_delta}`, ...) would end up either present
# with nothing behind it, or quietly omitted in a way a reader could miss.
# This template simply never names an endpoint-level slot at all — the
# ledger's own missing-slot refusal is what makes that durable rather than
# a convention someone could forget.

_SCHEMA_HEADER = """# Performance report: {endpoint}

**Terminal state:** `{terminal_state}`
**Gate:** {gate_summary}

This is a SCHEMA-CANDIDATE report. The claim below is STATEMENT-level — an
additive index, proven by the database itself (EXPLAIN's own plan, rows
examined, an interleaved direct-statement benchmark) — never an
endpoint-level one. Blue and green share one database, so an endpoint p95
figure could not be measured here without comparing two arms that both
have the index; no such figure is asserted anywhere in this report.

| | |
|---|---|
| Endpoint | `{endpoint}` |
| Branch | `{branch}` |
| Base commit | `{base_commit}` |
| Oracle | `{oracle_hash}` |
| Corpus | `{corpus_hash}` |

## What a human must check before merging

**Risk tier:** {risk_tier_check}

**What the oracle did not cover:** it proves blue and green are
byte-identical on every request in the benchmark and held-back sets counted
below, under Behaviour — nothing else. An additive index cannot change a
query's RESULT, only its cost, which is what makes this check sufficient
for a schema candidate without needing per-arm isolation.

**Repository test suite:** {test_suite_check}

**Pre-existing risks** (found while tracing the endpoint, independent of
this patch):

{pre_existing_risks}

## The bottleneck

**Diagnosed cause:** {bottleneck_claim}

### Root cause

{narrative}

## The statement-level proof

**Table and columns:** `{index_table}({index_columns})`

**Why this column order:** {column_rationale}

**EXPLAIN plan, before and after the candidate index:**

```
{plan_before}
{plan_after}
```

**Rows examined:** {rows_examined_before} -> {rows_examined_after}
({rows_examined_delta})

**Interleaved direct-statement benchmark** (with the index, without, with,
without, ... — never sequential, so neither side is credited with the
other's buffer-pool warming): {benchmark_rounds} round(s), Wilcoxon
signed-rank p = {benchmark_p_value}, median improvement {improvement_pct}.

**Redundancy check:** {redundancy_check}

**Write cost** (table_io_waits_summary_by_table, cumulative since the
server's statistics were last reset, not a rate): {table_writes}

**Index size:** {index_size}

## The migration

{migration_evidence}

## Gates

| Gate | Result | Threshold | Actual |
|---|---|---|---|
{gate_rows}

## Files changed

{file_list}

Lines added: {lines_added}. Lines removed: {lines_removed}.

## Behaviour

Requests compared: {compared}. Identical: {identical}. Differing paths:
{diff_count}. Held-back requests: {held_back_count}.

Tables excluded from the database-state comparison as schema-management
bookkeeping, not application data: {excluded_tables}.

## Provenance

{provenance_rows}
"""


def render_schema_report(ledger: EvidenceLedger, **slots: Backed) -> str:
    """Render report.md for a schema candidate. Every quantity comes off
    the ledger, same discipline as `render_report` — this template just
    never names an endpoint-level slot."""
    return ledger.render(_SCHEMA_HEADER, **slots)
