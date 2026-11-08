"""What onboarding learned about a repository, and what provisioning built.

Every value here was DETECTED by reading the repository — a file that exists,
a key in a config the repo itself ships. Nothing is guessed, and nothing
carries a default that happens to match the one repo this was developed
against. A field that could not be detected is `None` and the phase refuses
with `ENVIRONMENT_INFEASIBLE` naming it, because a profile that quietly
invents `src/main/java` is a profile that silently runs against the wrong
tree.

`evidence` is not decoration. Every detected value names the file it came
from, so an operator reading a refusal can check the detector's reasoning
rather than trusting it.
"""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator


class BuildTool(str, Enum):
    MAVEN = "maven"
    GRADLE = "gradle"


class MigrationTool(str, Enum):
    FLYWAY = "flyway"
    LIQUIBASE = "liquibase"
    NONE = "none"


class DatabaseKind(str, Enum):
    """What the repo's own configuration says it talks to.

    POSTGRES and H2 are DETECTABLE but not SUPPORTED — see
    `DatabaseProfile.supported`. Detecting them is the point: the harness
    refuses at onboarding with the reason, instead of discovering it as a
    crash in `measure/sql.py` three phases later.
    """

    MYSQL = "mysql"
    POSTGRES = "postgres"
    H2 = "h2"
    OTHER = "other"
    NONE = "none"


# `measure/sql.py` reads MySQL's performance_schema digest tables to attribute
# latency to statements. There is no portable equivalent, so this is the
# supported set — stated here, enforced at onboarding, and documented in
# SENAPATHY.md.
INSTRUMENTABLE_DATABASES: frozenset[DatabaseKind] = frozenset({DatabaseKind.MYSQL})


class Evidence(BaseModel):
    """One detected fact and where it was read from."""

    field: str
    value: str
    source: str = Field(description="Repo-relative file the value was read "
                                    "from, or the probe that found it.")


class BuildPlan(BaseModel):
    """How to build, test, and get a runnable classpath out of this repo.

    `invocation` is a list, not a string, because a wrapper script
    (`./mvnw`, `./gradlew`) and a binary on PATH (`mvn`) are the same
    concept with different argv, and quoting one into a shell string is how
    a path with a space becomes a bug.
    """

    tool: BuildTool
    invocation: list[str] = Field(min_length=1)
    build_args: list[str]
    test_args: list[str]
    classpath_args: list[str] = []
    classpath_file: str | None = None
    classes_dir: str
    # Untracked files under these prefixes are carried into the green
    # worktree; everything else is recorded as skipped. Build-tool specific:
    # a Gradle repo has no pom.xml and a Maven repo has no settings.gradle.
    build_inputs: list[str] = Field(min_length=1)


class Layout(BaseModel):
    """Where the source, tests and resources actually live."""

    source_root: str
    test_root: str
    resource_roots: list[str] = []


class DatabaseProfile(BaseModel):
    kind: DatabaseKind
    jdbc_url: str | None = None
    host: str | None = None
    port: int | None = None
    user: str | None = None
    password: str | None = None
    database: str | None = None
    # Whether `measure/sql.py` can attribute latency to statements here.
    # Computed by code from `kind`, never asserted by a detector.
    supported: bool = False
    unsupported_reason: str | None = None

    @model_validator(mode="after")
    def _support_follows_kind(self) -> "DatabaseProfile":
        expected = self.kind in INSTRUMENTABLE_DATABASES
        if self.supported != expected:
            object.__setattr__(self, "supported", expected)
        if not expected and not self.unsupported_reason:
            object.__setattr__(
                self, "unsupported_reason",
                f"{self.kind.value} is not instrumentable: SQL attribution "
                f"reads MySQL's performance_schema digest tables and there is "
                f"no portable equivalent.",
            )
        return self


class MigrationProfile(BaseModel):
    tool: MigrationTool = MigrationTool.NONE
    directory: str | None = None

    @model_validator(mode="after")
    def _directory_required_when_a_tool_is_present(self) -> "MigrationProfile":
        if self.tool is not MigrationTool.NONE and not self.directory:
            raise ValueError(
                f"{self.tool.value} was detected but no migration directory "
                f"was found. A migration tool with nowhere to write a "
                f"migration cannot ship a schema change."
            )
        return self


# A write allowlist entry that resolves to the repository root defeats the
# guardrail it is supposed to be. These are the spellings that do.
_ROOT_SPELLINGS = frozenset({"", ".", "/", "./", "*", "**"})


class RepoProfile(BaseModel):
    """Everything the pipeline needs to know about a repository it has never
    seen, detected by reading that repository.

    This replaces fourteen module-level constants that named one developer's
    laptop, one build tool, one source layout and one main class. See
    `senapathy/onboard.py` for the detectors and `tests/test_onboard.py` for
    what each one refuses.
    """

    repo: str
    build: BuildPlan
    layout: Layout
    entrypoint: str = Field(description="Fully-qualified main class.")
    run_profile: str | None = Field(
        default=None,
        description="Config profile to run the arms under, or None when the "
                    "repo's default profile is already usable.")
    database: DatabaseProfile
    migrations: MigrationProfile = Field(default_factory=MigrationProfile)
    health_path: str = "/actuator/health"
    # Derived from `layout.source_root`, never configured directly: a profile
    # that can widen its own write allowlist is not a guardrail.
    write_allowlist: list[str] = Field(min_length=1)
    evidence: list[Evidence] = []
    decided_by: Literal["code"] = "code"

    @field_validator("write_allowlist")
    @classmethod
    def _allowlist_cannot_be_the_repo_root(cls, value: list[str]) -> list[str]:
        for entry in value:
            normalised = entry.strip().rstrip("/") or entry.strip()
            if entry.strip() in _ROOT_SPELLINGS or normalised in _ROOT_SPELLINGS:
                raise ValueError(
                    f"write_allowlist entry {entry!r} resolves to the "
                    f"repository root. The allowlist exists so a patch cannot "
                    f"reach the build files, the tests or the migrations it "
                    f"was not planned to touch; one that spans the whole repo "
                    f"is not an allowlist."
                )
            parts = tuple(p for p in entry.replace("\\", "/").split("/") if p)
            if entry.startswith("/") or ".." in parts:
                raise ValueError(
                    f"write_allowlist entry {entry!r} is absolute or escapes "
                    f"the repository. Entries are repo-relative prefixes."
                )
        return value

    @model_validator(mode="after")
    def _allowlist_entries_must_be_derived(self) -> "RepoProfile":
        """Every entry must be one of the two paths onboarding DERIVES.

        The source root is where application code lives. The migration
        directory is the one place outside it a patch may create a file, and
        only through `add_migration`, which parses the SQL before writing.
        Those two are the whole vocabulary: an entry that is neither is a
        widening, whoever wrote it.

        This replaces a stricter rule -- "every entry must sit under the
        source root" -- which was wrong: on a conventional Maven layout the
        migration directory is under `src/main/resources`, a sibling of
        `src/main/java`, so that rule refused the very path the migration
        tool needs.
        """
        source_root = self.layout.source_root.rstrip("/")
        migration_dir = (self.migrations.directory or "").rstrip("/")
        for entry in self.write_allowlist:
            stem = entry.rstrip("/")
            if stem.startswith(source_root):
                continue
            if migration_dir and stem == migration_dir:
                continue
            raise ValueError(
                f"write_allowlist entry {entry!r} is neither under the "
                f"detected source root {source_root!r} nor the detected "
                f"migration directory {migration_dir or '(none)'!r}. The "
                f"allowlist is DERIVED from those two; it cannot be widened "
                f"to anything else."
            )
        return self

    def evidence_for(self, field: str) -> str | None:
        for item in self.evidence:
            if item.field == field:
                return item.source
        return None


# ────────────────────────────  provisioning  ─────────────────────────────


class ArmEndpoint(BaseModel):
    name: Literal["blue", "green"]
    path: str
    port: int
    base_url: str


class Provisioning(BaseModel):
    """Two runnable arms, on ports chosen at runtime.

    The ports are chosen by binding port 0 and reading back what the kernel
    gave us, rather than defaulting to 18080/8081 — those were one repo's
    `application-perfagent.yml` leaking into the harness, and on a machine
    where something else holds 8081 the green arm silently answered 401.
    """

    blue: ArmEndpoint
    green: ArmEndpoint
    database_ready: bool = False
    migrations_applied: list[str] = []
    health_confirmed: list[str] = []
    notes: list[str] = []
