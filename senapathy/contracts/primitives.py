"""The two pointers every other artifact is built out of.

Split out of `contracts.py` so that the comprehension shapes can import them
without importing the rest of the package back — everything else in
`contracts` is downstream of these, and a cycle here would be a cycle
everywhere.
"""

from __future__ import annotations

from pydantic import BaseModel


class Citation(BaseModel):
    """A pointer into source. Every claim an agent makes carries one of these,
    and every one is validated against the filesystem before acceptance."""
    file: str
    line: int
    symbol: str | None = None

    def ref(self) -> str:
        return f"{self.file}:{self.line}"


class ArtifactRef(BaseModel):
    """Reference to a stored artifact. The evidence ledger works by requiring
    these on every quantitative claim."""
    kind: str
    hash: str
    path: str
