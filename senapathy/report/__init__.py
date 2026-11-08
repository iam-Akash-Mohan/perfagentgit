"""P15 reporting: an evidence ledger, and one LLM call for prose only.

INVARIANT 4 (CLAUDE.md): no number without an artifact. The renderer here is
structurally incapable of emitting a quantity that has no `ArtifactRef` behind
it — not by convention, but because `render()` refuses a slot that is not a
`Backed` value and refuses a template that contains a bare numeral.
"""

from __future__ import annotations

from .render import Backed, EvidenceLedger, render_report

__all__ = ["Backed", "EvidenceLedger", "render_report"]
