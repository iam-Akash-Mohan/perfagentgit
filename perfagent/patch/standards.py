"""Java standards: the document the agent reads, and the lint that enforces it.

`standards/java.md` is prose for a model. Prose is advice, and a patch agent
under build pressure takes advice the way anyone does — when it is convenient.
So the four rules that can be checked mechanically are checked here, against
the DIFF, and a violation fails the phase.

Four, and not more, on purpose. "Prefer Optional over null" and "streams where
they read more clearly" need taste; a regex that tried would reject correct
code, and a lint nobody trusts gets switched off. `docs/patch-standards.md`
records which rule is which so that nobody has to read this file to find out.

Checked against ADDED lines, because the standard governs what the patch
writes, not what the repository already contained. The one exception is the
controller-signature rule, which by definition is about what was removed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

STANDARDS_DIR = Path(__file__).resolve().parent.parent.parent / "standards"
DEFAULT_STANDARDS = "java.md"

# Spring stereotypes. A type carrying one of these is built by the container,
# and `new`-ing it produces an object with every dependency null.
STEREOTYPES = ("@Service", "@Component", "@Repository", "@Controller",
               "@RestController", "@Configuration")

_AUTOWIRED = re.compile(r"^\s*@Autowired\b")
_FIELD_DECLARATION = re.compile(
    r"^\s*(private|protected|public|final|static|\w)[\w\s<>,.\[\]]*\s+\w+\s*;\s*$")
_WILDCARD_IMPORT = re.compile(r"^\s*import\s+(static\s+)?[\w.]+\.\*\s*;")
_CLASS_NAME = re.compile(r"\b(?:class|record|interface)\s+(\w+)")
_PUBLIC_METHOD = re.compile(
    r"^\s*(public|protected)\s+[\w<>,\[\]\s.]+\s+(?P<name>\w+)\s*\(")
_MAPPING = re.compile(r"@(Request|Get|Post|Put|Delete|Patch)Mapping")

# A controller is a file that carries a mapping annotation, not a file whose
# name ends in Controller. The agent could rename one.
_CONTROLLER_MARK = re.compile(r"@(Rest)?Controller\b")


@dataclass(frozen=True)
class Violation:
    """One broken rule, with the line that broke it."""

    rule: str
    file: str
    line: str
    detail: str

    def render(self) -> str:
        return (f"{self.file}: {self.rule}\n"
                f"    {self.line.strip()[:120]}\n"
                f"    {self.detail}")


def load_standards(directory: Path | str | None = None,
                   name: str = DEFAULT_STANDARDS) -> str:
    """The standards document, verbatim, for the prompt.

    Missing file is not fatal: the four enforced rules are in code and still
    apply. The agent simply loses the advisory half, and the caller is told.
    """
    path = Path(directory or STANDARDS_DIR) / name
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8").strip()


# ─────────────────────────────  diff reading  ────────────────────────────


@dataclass
class FileDiff:
    path: str
    added: list[str]
    removed: list[str]


def split_diff(diff: str) -> list[FileDiff]:
    """A unified diff, per file, as added and removed content lines."""
    files: list[FileDiff] = []
    current: FileDiff | None = None
    for line in (diff or "").splitlines():
        if line.startswith("+++ b/"):
            current = FileDiff(line[len("+++ b/"):].strip(), [], [])
            files.append(current)
            continue
        if current is None:
            continue
        if line.startswith("+") and not line.startswith("+++"):
            current.added.append(line[1:])
        elif line.startswith("-") and not line.startswith("---"):
            current.removed.append(line[1:])
    return files


def spring_managed_types(repo: Path) -> set[str]:
    """Every type in the repository the Spring container builds.

    Read from the source rather than assumed from a naming convention: a
    `@Component` called `Clock` is container-managed and `new Clock()` is a
    bug, while a plain `OrderCalculator` is not.
    """
    found: set[str] = set()
    root = Path(repo) / "src/main/java"
    if not root.is_dir():
        return found
    for path in root.rglob("*.java"):
        try:
            body = path.read_text(encoding="utf-8")
        except OSError:                     # pragma: no cover
            continue
        if not any(mark in body for mark in STEREOTYPES):
            continue
        match = _CLASS_NAME.search(body)
        if match:
            found.add(match.group(1))
    return found


# ──────────────────────────────  the rules  ──────────────────────────────


def _field_autowired(file: FileDiff) -> list[Violation]:
    out: list[Violation] = []
    for position, line in enumerate(file.added):
        if not _AUTOWIRED.match(line):
            continue
        following = next((l for l in file.added[position + 1:] if l.strip()), "")
        # A constructor or setter carries `(`; a field declaration ends in `;`.
        if "(" in following:
            continue
        if _FIELD_DECLARATION.match(following):
            out.append(Violation(
                rule="Required: constructor injection only. No field @Autowired.",
                file=file.path, line=following,
                detail="Take the dependency as a constructor parameter and "
                       "assign it to a final field. Field injection hides the "
                       "dependency from every caller and from the test.",
            ))
    return out


def _wildcard_imports(file: FileDiff) -> list[Violation]:
    return [
        Violation(
            rule="Style: keep imports explicit; no wildcard imports.",
            file=file.path, line=line,
            detail="Import the types you use by name.",
        )
        for line in file.added if _WILDCARD_IMPORT.match(line)
    ]


def _new_on_managed_type(file: FileDiff, managed: Iterable[str]) -> list[Violation]:
    out: list[Violation] = []
    for name in managed:
        pattern = re.compile(rf"\bnew\s+{re.escape(name)}\s*\(")
        for line in file.added:
            if pattern.search(line):
                out.append(Violation(
                    rule="Forbidden: no `new` on a Spring-managed type.",
                    file=file.path, line=line,
                    detail=f"{name} is a Spring bean. Constructing it by hand "
                           f"produces an instance with null dependencies and "
                           f"no proxy — no transaction, no caching, no "
                           f"advice. Inject it instead.",
                ))
    return out


def _controller_signature(file: FileDiff) -> list[Violation]:
    """A removed public method or mapping in a controller is a contract change."""
    if not (_CONTROLLER_MARK.search("\n".join(file.added + file.removed))
            or _MAPPING.search("\n".join(file.added + file.removed))
            or file.path.endswith("Controller.java")):
        return []

    removed_methods = {m.group("name") for line in file.removed
                       if (m := _PUBLIC_METHOD.match(line))}
    added_methods = {m.group("name") for line in file.added
                     if (m := _PUBLIC_METHOD.match(line))}
    gone = sorted(removed_methods - added_methods)

    out = [
        Violation(
            rule="Forbidden: no changes to public method signatures on "
                 "controllers.",
            file=file.path, line=f"removed: {name}(...)",
            detail="The controller's public surface is the API. Changing it "
                   "changes the contract, and the frozen oracle compares "
                   "responses byte for byte.",
        )
        for name in gone
    ]
    # A method kept but re-signed counts too: same name, different parameters.
    for line in file.removed:
        match = _PUBLIC_METHOD.match(line)
        if match and match.group("name") in added_methods:
            signature = line.split("(", 1)[1] if "(" in line else ""
            if not any(signature in added for added in file.added):
                out.append(Violation(
                    rule="Forbidden: no changes to public method signatures on "
                         "controllers.",
                    file=file.path, line=line,
                    detail=f"{match.group('name')} kept its name but changed "
                           f"its parameters.",
                ))
    return out


def lint_diff(diff: str, *, repo: Path | None = None,
              managed_types: Sequence[str] | None = None) -> list[Violation]:
    """Every enforced standard violated by this diff.

    `managed_types` is injectable so the rule can be tested without a
    repository on disk; in a real run it is read from the green worktree.
    """
    managed = (set(managed_types) if managed_types is not None
               else spring_managed_types(repo) if repo is not None else set())

    violations: list[Violation] = []
    for file in split_diff(diff):
        if not file.path.endswith(".java"):
            continue
        violations.extend(_field_autowired(file))
        violations.extend(_wildcard_imports(file))
        violations.extend(_new_on_managed_type(file, managed))
        violations.extend(_controller_signature(file))
    return violations


def render_violations(violations: Sequence[Violation]) -> str:
    """What the agent is told. Names the rule, because "fix the lint" is not
    actionable and "constructor injection only" is."""
    if not violations:
        return ""
    head = (f"STANDARDS VIOLATION: {len(violations)} rule(s) from "
            f"standards/{DEFAULT_STANDARDS} were broken by this diff. These "
            f"are enforced in code, not advice.\n")
    return head + "\n".join(f"  {v.render()}" for v in violations)
