"""The comprehension phase.

Deliberately free of eager imports: `tools/submit.py` reaches into
`understand.validate`, so importing the runner here would close a cycle.
Import `perfagent.understand.agent` directly.
"""

from __future__ import annotations

__all__ = ["agent", "cli", "validate"]
