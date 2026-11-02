"""Are the two arms actually independent? For a schema change, usually not.

An index is a property of the DATABASE, not of a worktree. Blue and green are
two checkouts of the same repository pointed at one MySQL, which is fine for
every change that lives in Java and wrong for every change that does not: the
moment green's migration adds an index, blue is querying an indexed table too.

The A/B then compares two arms that both have the index and reports no
improvement — a FALSE NEGATIVE that rejects a good patch while looking exactly
like the system working correctly. That is the worst failure shape available to
this product, because nothing about it is visibly broken.

Per-arm schemas are the real fix. Until then the minimum that keeps the product
honest is to detect the condition and refuse to measure, with a terminal state
that says "could not measure" rather than "did not help" — those are different
facts and a report must not make them look alike.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

# `jdbc:mysql://localhost:3307/mini?useSSL=false&...`
_JDBC = re.compile(
    r"jdbc:(?P<vendor>\w+)://(?P<host>[\w.\-]+)(?::(?P<port>\d+))?/(?P<schema>\w+)")

# `spring.datasource.url` in a Spring YAML, without pulling in a YAML parser
# for one line — the value may carry `${SERVER_PORT:18080}`-style placeholders
# that a strict parse would not help with anyway.
_URL_LINE = re.compile(r"^\s*url:\s*(?P<url>\S+)", re.M)

# Checked in order; the profile file wins because that is what the arms run.
CONFIG_FILES = (
    "src/main/resources/application-perfagent.yml",
    "src/main/resources/application.yml",
    "src/main/resources/application.properties",
)

MIGRATION_MARK = "src/main/resources/db/migration"


@dataclass(frozen=True)
class Datasource:
    """Where one arm's JDBC URL actually points."""

    url: str
    vendor: str = ""
    host: str = ""
    port: str = ""
    schema: str = ""

    @property
    def identity(self) -> tuple[str, str, str, str]:
        """What makes two datasources the same database."""
        return (self.vendor, self.host, self.port, self.schema)

    def render(self) -> str:
        if not self.vendor:
            return self.url or "(no datasource url found)"
        port = f":{self.port}" if self.port else ""
        return f"{self.vendor}://{self.host}{port}/{self.schema}"


def resolve_datasource(worktree: Path) -> Datasource:
    """The datasource an arm started from this worktree would connect to.

    Read from the arm's own configuration rather than assumed from the
    harness's: the point of the check is to compare what the two arms will
    really do, and green is a different checkout that could in principle carry
    a different config.
    """
    root = Path(worktree)
    for relative in CONFIG_FILES:
        path = root / relative
        if not path.exists():
            continue
        try:
            body = path.read_text(encoding="utf-8")
        except OSError:                     # pragma: no cover
            continue
        match = _URL_LINE.search(body) or _JDBC.search(body)
        if match is None:
            continue
        url = match.groupdict().get("url") or match.group(0)
        parsed = _JDBC.search(url)
        if parsed is None:
            return Datasource(url=url)
        return Datasource(url=url, vendor=parsed.group("vendor"),
                          host=parsed.group("host"),
                          port=parsed.group("port") or "",
                          schema=parsed.group("schema"))
    return Datasource(url="")


def touches_schema(*, approach: str = "", diff: str = "",
                   files_changed: list[str] | None = None) -> bool:
    """Does this patch change the database rather than the code?

    Three signals, any of which is enough: the plan said `add_index`, the diff
    contains a migration file, or the patch reports one among its files. A
    schema change that arrived by a route nobody predicted is still a schema
    change.
    """
    if approach == "add_index":
        return True
    if MIGRATION_MARK in (diff or ""):
        return True
    return any(MIGRATION_MARK in f for f in (files_changed or []))


def arms_share_one_schema(blue: Path, green: Path) -> str | None:
    """Would these two arms' resolved datasources collide, or None.

    Pure comparison, no opinion on whether the candidate under test even
    touches the schema — that is `touches_schema`'s job, deliberately kept
    separate. The split exists because a caller at PLAN TIME has no `diff` or
    `green` worktree yet (nothing has been written) and needs to ask "if this
    ran today, would it be measurable" without a patch to point at; a caller
    at MEASURE TIME has a real diff and a real green checkout. Both go through
    this one comparison so "are the arms isolated" has one definition.
    """
    blue_source = resolve_datasource(blue)
    green_source = resolve_datasource(green)
    if blue_source.identity != green_source.identity:
        return None
    if not blue_source.vendor:
        return (
            "this patch changes the database schema, and neither arm's "
            "configuration names a JDBC URL that could be compared. Whether "
            "the arms are isolated is unknown, and an unknown is not an "
            "isolation."
        )

    return (
        f"this patch changes the database schema, and both arms resolve to the "
        f"SAME schema: {blue_source.render()}.\n"
        f"An index is a property of the database, not of a worktree. Once "
        f"green's migration applies it, blue is querying an indexed table too, "
        f"so an interleaved A/B would compare two arms that both have the "
        f"index and report no improvement for a change that may be worth a "
        f"great deal. That is a false negative, and it looks exactly like the "
        f"system working.\n"
        f"The measurement was NOT run. This is not a statement that the patch "
        f"does not help — it is a statement that this harness cannot currently "
        f"find out. Give each arm its own schema and run again; "
        f"docs/schema-changes.md says what that requires."
    )


def isolation_problem(blue: Path, green: Path, *, approach: str = "",
                      diff: str = "",
                      files_changed: list[str] | None = None) -> str | None:
    """Why this patch cannot be A/B measured here, or None.

    Returns None for every patch that does not touch the schema — which is
    most of them — so the common path is unaffected. The MEASURE-time entry
    point: called with a real diff and a real green worktree, right before
    firing the A/B. `perfagent/plan/risk.py` calls `arms_share_one_schema`
    directly at SELECTION time, earlier, before either exists.
    """
    if not touches_schema(approach=approach, diff=diff,
                          files_changed=files_changed):
        return None
    return arms_share_one_schema(blue, green)
