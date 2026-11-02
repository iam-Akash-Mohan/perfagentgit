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
**Gate:** {gate_summary}

| | |
|---|---|
| Endpoint | `{endpoint}` |
| Branch | `{branch}` |
| Base commit | `{base_commit}` |
| Oracle | `{oracle_hash}` |
| Corpus | `{corpus_hash}` |

## The bottleneck

**Diagnosed cause:** {bottleneck_claim}

*Attributed:* {attributed_ms} ms of a measured p50 of {baseline_p50} ms.

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

| Measurement | Blue (base) | Green (patched) | Delta |
|---|---|---|---|
| p50 | {blue_p50} ms | {green_p50} ms | {p50_delta} |
| p95 | {blue_p95} ms | {green_p95} ms | {p95_delta} |
| p99 | {blue_p99} ms | {green_p99} ms | |
| SQL statements | {blue_sql} | {green_sql} | |
| Rows examined | {blue_rows} | {green_rows} | |

Paired samples per arm: {pair_count}. Measurement was interleaved
(blue, green, blue, green) across the benchmark set.

Wilcoxon signed-rank p: {paired_p}. Mann-Whitney U p: {unpaired_p}.

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

## Provenance

Every number above is templated from a recorded artifact. The renderer refuses
a slot with no `ArtifactRef` behind it, and refuses a template containing a
bare numeral.

{provenance_rows}
"""


def render_report(ledger: EvidenceLedger, **slots: Backed) -> str:
    """Render report.md. Every quantity comes off the ledger."""
    return ledger.render(_HEADER, **slots)
