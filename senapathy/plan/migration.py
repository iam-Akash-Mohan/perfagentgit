"""Write an index as a Flyway migration the target repo will actually apply.

The hard part is not the SQL, it is the filename. A migration whose version
number collides with an existing one makes Flyway refuse to start, and a
migration written into a directory Flyway does not scan is a file that silently
never runs — which is worse, because the patch then claims an index that does
not exist and the measurement disagrees with the schema.

So this module discovers the directory rather than assuming it, refuses when
the repository has no Flyway at all, and never overwrites a file it did not
write in this call.
"""

from __future__ import annotations

import re
from pathlib import Path

from ..errors import ToolError
from ..tools.db import TEMP_PREFIX

# Where Flyway looks by default, and the two spellings projects actually use.
# Checked in order; the first that exists wins.
CONVENTIONAL_DIRS: tuple[str, ...] = (
    "src/main/resources/db/migration",
    "src/main/resources/db/migrations",
    "src/main/resources/migration",
    "migrations",
)

# `V12__add_invoice_index.sql`
VERSIONED = re.compile(r"^V(?P<version>\d+(?:[._]\d+)*)__(?P<name>.+)\.sql$")

MAX_NAME_LENGTH = 60


def find_migration_dir(worktree: Path) -> Path | None:
    """The directory Flyway scans in this repository, or None.

    Any directory with a `V<n>__*.sql` in it beats a conventional path that
    happens to exist but is empty: the files are the evidence, the convention
    is only a guess.
    """
    root = Path(worktree)
    for relative in CONVENTIONAL_DIRS:
        candidate = root / relative
        if candidate.is_dir() and any(
                VERSIONED.match(p.name) for p in candidate.glob("*.sql")):
            return candidate
    for relative in CONVENTIONAL_DIRS:
        candidate = root / relative
        if candidate.is_dir():
            return candidate
    return None


def uses_flyway(worktree: Path) -> bool:
    """Does this repository actually run Flyway?

    Read from the build file, because the answer decides whether a migration
    is a change or a decoration. A repo whose schema is owned by a docker
    entrypoint script — which is how the mini-spring sandbox is built — gets a
    refusal rather than a file nobody will ever apply.
    """
    root = Path(worktree)
    for name in ("pom.xml", "build.gradle", "build.gradle.kts"):
        build = root / name
        if build.exists():
            try:
                if "flyway" in build.read_text(encoding="utf-8").lower():
                    return True
            except OSError:         # pragma: no cover - unreadable build file
                continue
    return find_migration_dir(root) is not None


def next_version(directory: Path) -> int:
    """One past the highest existing version. Starts at 1 in an empty tree."""
    highest = 0
    for path in Path(directory).glob("*.sql"):
        match = VERSIONED.match(path.name)
        if match is None:
            continue
        leading = re.split(r"[._]", match.group("version"))[0]
        try:
            highest = max(highest, int(leading))
        except ValueError:          # pragma: no cover - V1_2__x.sql style
            continue
    return highest + 1


def index_name(table: str, columns: list[str]) -> str:
    """`idx_invoice_customer_id_status` — the conventional spelling.

    Truncated on length rather than hashed: a reviewer has to be able to read
    what the index is for, and MySQL's identifier ceiling is 64 characters.
    """
    name = "_".join(["idx", table, *columns])
    return name[:64].rstrip("_")


def existing_migration_for(directory: Path, index: str) -> Path | None:
    """A migration in this directory that already creates `index`, if any."""
    pattern = re.compile(
        r"create\s+(unique\s+)?index\s+[`\"]?" + re.escape(index) + r"[`\"]?\b",
        re.I)
    for path in sorted(Path(directory).glob("*.sql")):
        try:
            if pattern.search(path.read_text(encoding="utf-8")):
                return path
        except OSError:             # pragma: no cover - unreadable sibling
            continue
    return None


def write_index_migration(worktree: Path, table: str, columns: list[str],
                          rationale: str, *, evidence: str | None = None,
                          run_id: str | None = None) -> Path:
    """Write `V<n>__add_<table>_<columns>_index.sql` and return its path.

    `evidence` and `run_id` are optional only so that a caller with no
    measurement can still produce a file; when they are supplied they are
    templated verbatim into the comment, never summarised (invariant 4).
    """
    root = Path(worktree)
    if not root.is_dir():
        raise ToolError(f"{root} is not a directory.")

    clean = [c.strip().strip("`") for c in columns if c and c.strip()]
    if not clean:
        raise ToolError("a migration needs at least one column to index.")
    for column in [table, *clean]:
        if not re.fullmatch(r"\w+", column):
            raise ToolError(f"{column!r} is not a usable identifier.")

    directory = find_migration_dir(root)
    if directory is None:
        if not uses_flyway(root):
            raise ToolError(
                f"{root} has no Flyway migration directory and its build file "
                f"does not mention Flyway, so a migration written here would "
                f"never be applied. Looked in: "
                f"{', '.join(CONVENTIONAL_DIRS)}.",
                suggestion="Add Flyway to the project, or apply the index "
                           "through whatever owns the schema in this repo.",
            )
        directory = root / CONVENTIONAL_DIRS[0]
    directory.mkdir(parents=True, exist_ok=True)

    name = index_name(table, clean)
    if name.startswith(TEMP_PREFIX):        # unreachable by construction
        raise ToolError(
            f"a migration may not create a temporary index ({TEMP_PREFIX}*); "
            f"those exist only inside index_benefit.")

    # Idempotent in the sense that matters: the same index twice is not two
    # migrations. A second call returns the file the first one wrote. Writing
    # V4 alongside V3 for the same index passes "never overwrites" and still
    # breaks the deploy — Flyway applies both and MySQL rejects the duplicate
    # key name.
    already = existing_migration_for(directory, name)
    if already is not None:
        return already

    version = next_version(directory)
    slug = f"add_{table}_{'_'.join(clean)}_index"[:MAX_NAME_LENGTH].rstrip("_")
    path = directory / f"V{version}__{slug}.sql"
    if path.exists():
        # Idempotent, and deliberately not "overwrite": a migration already on
        # disk may have been applied, and rewriting its contents changes a
        # checksum Flyway will refuse to start against.
        raise ToolError(
            f"{path} already exists. A migration is never rewritten — its "
            f"checksum is recorded once it has been applied.",
            suggestion="Delete the file if it was never applied, or write the "
                       "next version by hand.",
        )

    lines = [f"-- Senapathy: {rationale.strip()}"]
    if evidence:
        reference = f" (see run {run_id})" if run_id else ""
        lines.append(f"-- Evidence: {evidence.strip()}{reference}")
    lines.append(
        f"CREATE INDEX {name} ON {table} ({', '.join(clean)});")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
