"""P12 patch loop: the agent writes a real diff, builds it, and repairs itself.

INVARIANT 1 (CLAUDE.md): the agent gets three named tools with JSON schemas —
`apply_patch`, `build`, `run_tests` — and no shell. `build` and `run_tests`
each run ONE fixed argv list; neither takes an argument from the model, so
there is no string the model can supply that becomes part of a command.

INVARIANT 3: `src/test/**` is write-denied in code, and the test count may not
decrease. Both are checked in `tools.py`, not asked for in the prompt.

INVARIANT 7: the blue working tree is never written. All writes go to the green
worktree under `.perfagent/green`.
"""

from __future__ import annotations

from .worktree import GREEN_DIR, GreenWorktree

__all__ = ["GREEN_DIR", "GreenWorktree"]
