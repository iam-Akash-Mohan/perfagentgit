"""P15 gate: six deterministic checks, no model vote.

INVARIANT 6 (CLAUDE.md): gate decisions are computed. `GateDecision.decided_by`
is a `Literal["code"]`, so the contract itself refuses any other author, and
every threshold in `rules.py` is a module constant rather than a parameter —
there is no argument a caller can pass that loosens a gate.
"""

from __future__ import annotations

from .rules import evaluate_gates, terminal_state_for

__all__ = ["evaluate_gates", "terminal_state_for"]
