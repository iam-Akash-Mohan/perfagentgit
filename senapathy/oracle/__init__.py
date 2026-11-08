"""The behaviour oracle: what the endpoint returned on unmodified code.

INVARIANT 2 (CLAUDE.md): the oracle is frozen and is **not in any agent's tool
surface**. Nothing in `senapathy/tools.py` or `senapathy/diagnose/` may import
this package, and no tool schema may expose a path under `.senapathy/oracle*`.
`tests/test_oracle.py` asserts that mechanically rather than trusting review.

INVARIANT 6: the equivalence verdict is computed here in Python. The model has
no vote and no way to widen the tolerance policy.
"""

from __future__ import annotations

# `canonicalise` and `compare` are NOT re-exported here: a function of the same
# name as its module shadows the module, so `from senapathy.oracle import
# compare` would silently hand back the function. Import those two from their
# modules directly.
from .canonicalise import load_policy, unmatched_policy_paths
from .record import load_oracle, record_oracle, replay_live, write_oracle

__all__ = [
    "load_oracle",
    "load_policy",
    "record_oracle",
    "replay_live",
    "unmatched_policy_paths",
    "write_oracle",
]
