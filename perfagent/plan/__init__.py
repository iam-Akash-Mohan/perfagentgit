"""The plan phase: rank bottlenecks by fixability, propose candidate fixes.

`fixability.py` and `risk.py` are pure code with no model in them. `agent.py`
is the one place a model is asked anything, and what it may answer is bounded
by `submit.py`.
"""

from __future__ import annotations

from .fixability import assess_fixability, rank_hypotheses
from .risk import assess_risk, forbidden_reason, selection_problem

__all__ = [
    "assess_fixability", "rank_hypotheses",
    "assess_risk", "forbidden_reason", "selection_problem",
]
