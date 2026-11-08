"""The tool surface: a registry, a schema, and an injected source root.

`REPO_ROOT` used to be a module constant pointing at one developer's clone of
one fixture repo. That single line was the reason the product could not run
against a second repository, could not be tested against a fixture, and could
not have two repos open in one process. It is now a `SourceRoot` constructed
per run and passed in.

Every capability in here is a named tool with a JSON schema and an allowlist.
There is no shell tool and nothing that could become one (invariant 1).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

from ..errors import GuardrailError


@dataclass(frozen=True)
class SourceRoot:
    """A repository an agent may read, and the containment rule for it."""

    path: Path

    @classmethod
    def of(cls, path: str | Path) -> "SourceRoot":
        resolved = Path(path).expanduser().resolve()
        if not resolved.is_dir():
            raise GuardrailError(
                f"source root {resolved} is not a directory",
                suggestion="Pass --repo pointing at a checked-out repository.",
            )
        return cls(resolved)

    def resolve(self, rel: str) -> Path:
        """Turn an agent-supplied path into a real path, or refuse.

        resolve() collapses `..` and follows symlinks, and it happens BEFORE
        the check, because "src/../../../etc/passwd" looks innocent until
        resolved. `is_relative_to` is the actual containment test: string
        matching on ".." is not enough, since a symlink inside the repo can
        point anywhere.
        """
        target = (self.path / rel).resolve()
        if not target.is_relative_to(self.path):
            raise GuardrailError(
                f"path '{rel}' resolves outside the repository "
                f"({target}) and was refused.",
                suggestion=f"Use a path relative to the repository root "
                           f"{self.path.name}/, e.g. src/main/java/...",
            )
        return target

    def relative(self, target: Path) -> str:
        try:
            return str(Path(target).resolve().relative_to(self.path))
        except ValueError:
            return str(target)

    def __str__(self) -> str:
        return str(self.path)


@dataclass(frozen=True)
class ToolSpec:
    """One tool: what the model is told, and what actually runs.

    The description IS a prompt. It is where the model learns when to reach
    for this tool and what it gets back, so it lives next to the handler
    rather than in a prompt file that can drift away from the code.
    """

    name: str
    description: str
    parameters: dict[str, Any]
    handler: Callable[..., Any]
    terminal: bool = False

    def schema(self) -> dict[str, Any]:
        """The neutral shape. Clients translate it to their wire format."""
        return {"name": self.name, "description": self.description,
                "parameters": self.parameters}

    def openai_schema(self) -> dict[str, Any]:
        return {"type": "function", "function": self.schema()}


@dataclass
class ToolRegistry:
    """The tools one phase offers, and the allowlist that follows from it."""

    specs: list[ToolSpec] = field(default_factory=list)

    def add(self, *specs: ToolSpec) -> "ToolRegistry":
        self.specs.extend(specs)
        return self

    def extend(self, specs: Iterable[ToolSpec]) -> "ToolRegistry":
        self.specs.extend(specs)
        return self

    def schemas(self) -> list[dict[str, Any]]:
        return [spec.schema() for spec in self.specs]

    def handlers(self) -> dict[str, Callable[..., Any]]:
        return {spec.name: spec.handler for spec in self.specs}

    def names(self) -> list[str]:
        return [spec.name for spec in self.specs]

    def terminal_name(self) -> str | None:
        for spec in self.specs:
            if spec.terminal:
                return spec.name
        return None


__all__ = ["SourceRoot", "ToolSpec", "ToolRegistry"]
