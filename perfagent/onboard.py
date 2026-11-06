"""Phase P-1: look at a repository you have never seen and write it down.

Every detector here answers one question by READING THE REPOSITORY, and
records the file it read. None of them has a fallback that happens to be
correct for the repo this project was developed against — that fallback is
the bug this phase exists to remove. A detector that cannot answer returns
`None` and `onboard()` refuses with `ENVIRONMENT_INFEASIBLE` naming the
question, the files it looked in, and what the operator should do.

The supported stack is JVM web services on MySQL. That is a real limit, not
an oversight: `measure/sql.py` attributes latency to statements by reading
MySQL's `performance_schema` digest tables, and nothing portable does the
same job. Detecting Postgres here and refusing with that sentence is the
whole point — the alternative is discovering it as a driver crash four
phases later, after the oracle has been recorded.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .contracts.repo import (BuildPlan, BuildTool, DatabaseKind,
                             DatabaseProfile, Evidence, Layout,
                             MigrationProfile, MigrationTool, RepoProfile)
from .errors import ContractViolation

# Where a JVM project's configuration conventionally lives. Each is TRIED;
# none is assumed. The order matters only for which file is cited as the
# source when several agree.
_APP_CONFIG_GLOBS = (
    "src/main/resources/application*.yml",
    "src/main/resources/application*.yaml",
    "src/main/resources/application*.properties",
    "app/src/main/resources/application*.yml",
    "src/main/resources/config/application*.yml",
)

_SOURCE_ROOT_CANDIDATES = (
    "src/main/java", "src/main/kotlin", "app/src/main/java", "src/main/scala",
)
_TEST_ROOT_CANDIDATES = (
    "src/test/java", "src/test/kotlin", "app/src/test/java", "src/test/scala",
)
_RESOURCE_ROOT_CANDIDATES = (
    "src/main/resources", "app/src/main/resources",
)

_MIGRATION_DIR_CANDIDATES = (
    "src/main/resources/db/migration",
    "src/main/resources/db/migrations",
    "src/main/resources/migration",
    "src/main/resources/db/changelog",
    "migrations",
)

# jdbc:mysql://host:port/database?params
_JDBC = re.compile(
    r"jdbc:(?P<vendor>\w+)://(?P<host>[\w.\-]+)(?::(?P<port>\d+))?/(?P<db>[\w\-]+)"
)

_VENDOR_TO_KIND = {
    "mysql": DatabaseKind.MYSQL,
    "mariadb": DatabaseKind.MYSQL,     # wire-compatible, same digest tables
    "postgresql": DatabaseKind.POSTGRES,
    "postgres": DatabaseKind.POSTGRES,
    "h2": DatabaseKind.H2,
}

# A Spring Boot application class. Deliberately a parser of the ANNOTATION
# rather than of the filename: "Application.java" is a convention, and
# @SpringBootApplication is the fact.
_SPRING_BOOT_APP = re.compile(r"@SpringBootApplication\b")
_PACKAGE = re.compile(r"^\s*package\s+([\w.]+)\s*;", re.M)
_MAIN_METHOD = re.compile(r"\bpublic\s+static\s+void\s+main\s*\(")


class OnboardingRefusal(ContractViolation):
    """A repository this harness cannot onboard, and exactly why.

    A subclass so the orchestrator can map it to ENVIRONMENT_INFEASIBLE
    without string-matching the message — the mistake that made a classpath
    mismatch look like a behaviour change for three runs.
    """

    def __init__(self, question: str, looked_in: list[str], advice: str) -> None:
        self.question = question
        self.looked_in = list(looked_in)
        self.advice = advice
        looked = "\n".join(f"    {p}" for p in looked_in) or "    (nothing)"
        super().__init__(
            f"cannot onboard this repository: {question}\n"
            f"  looked in:\n{looked}\n"
            f"  what to do: {advice}"
        )


# ──────────────────────────────  detectors  ──────────────────────────────


def detect_build(repo: Path) -> tuple[BuildPlan, list[Evidence]]:
    """Maven or Gradle, wrapper preferred over whatever is on PATH.

    The wrapper is preferred because it pins the build tool's own version:
    a repo that ships `mvnw` has said which Maven it expects, and using the
    machine's `mvn` instead is how a build reproduces differently on two
    laptops.
    """
    evidence: list[Evidence] = []

    if (repo / "pom.xml").exists():
        wrapper = repo / "mvnw"
        invocation = ["./mvnw"] if wrapper.exists() else ["mvn"]
        evidence.append(Evidence(
            field="build.tool", value="maven", source="pom.xml"))
        evidence.append(Evidence(
            field="build.invocation", value=" ".join(invocation),
            source="mvnw" if wrapper.exists() else "mvn (on PATH; no mvnw in repo)"))
        return BuildPlan(
            tool=BuildTool.MAVEN,
            invocation=invocation,
            build_args=["-q", "-DskipTests", "package"],
            test_args=["-q", "test"],
            classpath_args=["-q", "dependency:build-classpath",
                            "-DincludeScope=runtime"],
            classpath_file="{out}",
            classes_dir="target/classes",
            build_inputs=["src/", "pom.xml", "mvnw", "mvnw.cmd", ".mvn/"],
        ), evidence

    gradle_files = [n for n in ("build.gradle", "build.gradle.kts")
                    if (repo / n).exists()]
    if gradle_files:
        wrapper = repo / "gradlew"
        invocation = ["./gradlew"] if wrapper.exists() else ["gradle"]
        evidence.append(Evidence(
            field="build.tool", value="gradle", source=gradle_files[0]))
        evidence.append(Evidence(
            field="build.invocation", value=" ".join(invocation),
            source="gradlew" if wrapper.exists()
                   else "gradle (on PATH; no gradlew in repo)"))
        return BuildPlan(
            tool=BuildTool.GRADLE,
            invocation=invocation,
            build_args=["-q", "assemble", "-x", "test"],
            test_args=["-q", "test"],
            classpath_args=[],
            classpath_file=None,
            classes_dir="build/classes/java/main",
            build_inputs=["src/", "build.gradle", "build.gradle.kts",
                          "settings.gradle", "settings.gradle.kts",
                          "gradlew", "gradlew.bat", "gradle/"],
        ), evidence

    raise OnboardingRefusal(
        question="which build tool builds this repository?",
        looked_in=[str(repo / n) for n in
                   ("pom.xml", "build.gradle", "build.gradle.kts")],
        advice="PerfAgent supports Maven and Gradle. Add the build file, or "
               "point --repo at the module root rather than a parent "
               "directory.",
    )


def detect_layout(repo: Path) -> tuple[Layout, list[Evidence]]:
    """Source, test and resource roots — found on disk, never assumed."""
    source = next((c for c in _SOURCE_ROOT_CANDIDATES if (repo / c).is_dir()), None)
    if source is None:
        raise OnboardingRefusal(
            question="where does the application source live?",
            looked_in=[str(repo / c) for c in _SOURCE_ROOT_CANDIDATES],
            advice="PerfAgent derives the patch agent's write allowlist from "
                   "the source root, so it will not proceed without one. A "
                   "non-standard layout needs a candidate added to "
                   "_SOURCE_ROOT_CANDIDATES in perfagent/onboard.py.",
        )

    test = next((c for c in _TEST_ROOT_CANDIDATES if (repo / c).is_dir()), None)
    if test is None:
        raise OnboardingRefusal(
            question="where do the tests live?",
            looked_in=[str(repo / c) for c in _TEST_ROOT_CANDIDATES],
            advice="The test tree is write-DENIED and its test count is a "
                   "guardrail (invariant 3). Without it the harness cannot "
                   "tell a patch that deleted a test from one that did not, "
                   "so it refuses rather than running unguarded.",
        )

    resources = [c for c in _RESOURCE_ROOT_CANDIDATES if (repo / c).is_dir()]
    return Layout(source_root=source, test_root=test,
                  resource_roots=resources), [
        Evidence(field="layout.source_root", value=source, source=source),
        Evidence(field="layout.test_root", value=test, source=test),
    ]


def _config_files(repo: Path) -> list[Path]:
    found: list[Path] = []
    for pattern in _APP_CONFIG_GLOBS:
        found.extend(sorted(repo.glob(pattern)))
    # Deduplicate while keeping order.
    seen: set[Path] = set()
    out: list[Path] = []
    for path in found:
        if path not in seen:
            seen.add(path)
            out.append(path)
    return out


def detect_database(repo: Path) -> tuple[DatabaseProfile, list[Evidence]]:
    """The datasource the repo's OWN config names.

    Reads every application config it can find and takes the first JDBC URL.
    A repo with no datasource at all is legal — not every service has a
    database — and yields `DatabaseKind.NONE`; what is NOT legal is guessing
    one.
    """
    configs = _config_files(repo)
    for path in configs:
        text = path.read_text(encoding="utf-8", errors="replace")
        match = _JDBC.search(text)
        if not match:
            continue
        vendor = match.group("vendor").lower()
        kind = _VENDOR_TO_KIND.get(vendor, DatabaseKind.OTHER)
        rel = str(path.relative_to(repo))
        port = int(match.group("port")) if match.group("port") else None
        profile = DatabaseProfile(
            kind=kind,
            jdbc_url=match.group(0),
            host=match.group("host"),
            port=port,
            database=match.group("db"),
            user=_scalar(text, ("username", "spring.datasource.username")),
            password=_scalar(text, ("password", "spring.datasource.password")),
        )
        return profile, [
            Evidence(field="database.kind", value=kind.value, source=rel),
            Evidence(field="database.jdbc_url", value=match.group(0), source=rel),
        ]

    return DatabaseProfile(kind=DatabaseKind.NONE), [
        Evidence(field="database.kind", value="none",
                 source=f"no jdbc: URL in {len(configs)} config file(s)")
    ]


def _scalar(text: str, keys: tuple[str, ...]) -> str | None:
    """A `key: value` or `key=value` scalar out of YAML or properties text.

    Deliberately not a YAML parse: these files routinely contain Spring
    placeholder syntax (`${SERVER_PORT:18080}`) and profile documents
    separated by `---`, and a strict parse of a file we only want two
    strings from turns a readable config into an onboarding refusal.
    """
    for key in keys:
        leaf = key.rsplit(".", 1)[-1]
        pattern = re.compile(
            rf"^\s*{re.escape(leaf)}\s*[:=]\s*(?P<value>\S+)\s*$", re.M)
        match = pattern.search(text)
        if match:
            value = match.group("value").strip().strip('"').strip("'")
            if value and not value.startswith("${"):
                return value
    return None


def detect_entrypoint(repo: Path, layout: Layout) -> tuple[str, list[Evidence]]:
    """The fully-qualified main class, by finding @SpringBootApplication.

    Searches the detected source root for the annotation rather than for a
    file called `Application.java`: the filename is a convention and the
    annotation is the fact. A repo with several is ambiguous and is refused
    rather than having one picked for it.
    """
    root = repo / layout.source_root
    found: list[tuple[str, str]] = []
    for path in sorted(root.rglob("*.java")):
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if not _SPRING_BOOT_APP.search(text) or not _MAIN_METHOD.search(text):
            continue
        package = _PACKAGE.search(text)
        if not package:
            continue
        fqcn = f"{package.group(1)}.{path.stem}"
        found.append((fqcn, str(path.relative_to(repo))))

    if not found:
        raise OnboardingRefusal(
            question="which class starts the application?",
            looked_in=[f"{layout.source_root}/**/*.java "
                       f"(searching for @SpringBootApplication + main)"],
            advice="PerfAgent launches each arm as `java -cp ... <MainClass>`, "
                   "so it needs the entrypoint. Only Spring Boot entrypoints "
                   "are auto-detected today; a plain-Java main needs the "
                   "detector extended.",
        )
    if len(found) > 1:
        raise OnboardingRefusal(
            question="which class starts the application? (found several)",
            looked_in=[source for _, source in found],
            advice="A repository with more than one @SpringBootApplication is "
                   "ambiguous to this harness. Point --repo at the single "
                   "module under test.",
        )
    fqcn, source = found[0]
    return fqcn, [Evidence(field="entrypoint", value=fqcn, source=source)]


def detect_run_profile(repo: Path) -> tuple[str | None, list[Evidence]]:
    """Which config profile to launch the arms under.

    A file named `application-<profile>.yml` declares that profile exists.
    Returning None is a real answer: it means the default profile is already
    usable, not that detection failed.
    """
    for path in _config_files(repo):
        match = re.fullmatch(r"application-([\w\-]+)\.(ya?ml|properties)",
                             path.name)
        if match:
            name = match.group(1)
            return name, [Evidence(field="run_profile", value=name,
                                   source=str(path.relative_to(repo)))]
    return None, [Evidence(field="run_profile", value="(repo default)",
                           source="no application-<profile> file found")]


def detect_migrations(repo: Path) -> tuple[MigrationProfile, list[Evidence]]:
    """Flyway, Liquibase, or neither — from the build file AND a directory.

    Both halves are required. A dependency with nowhere to put a migration
    cannot ship a schema change, and a directory with no tool to apply it is
    a directory.
    """
    build_text = ""
    build_source = ""
    for name in ("pom.xml", "build.gradle", "build.gradle.kts"):
        path = repo / name
        if path.exists():
            build_text += path.read_text(encoding="utf-8", errors="replace")
            build_source = build_source or name

    tool = MigrationTool.NONE
    if "flyway" in build_text.lower():
        tool = MigrationTool.FLYWAY
    elif "liquibase" in build_text.lower():
        tool = MigrationTool.LIQUIBASE

    if tool is MigrationTool.NONE:
        return MigrationProfile(), [Evidence(
            field="migrations.tool", value="none",
            source=build_source or "(no build file read)")]

    directory = next((c for c in _MIGRATION_DIR_CANDIDATES
                      if (repo / c).is_dir()), None)
    if directory is None:
        raise OnboardingRefusal(
            question=f"{tool.value} is a dependency, but where do its "
                     f"migrations live?",
            looked_in=[str(repo / c) for c in _MIGRATION_DIR_CANDIDATES],
            advice="A schema-change candidate needs somewhere to write a "
                   "migration. Create the conventional directory, or remove "
                   "the dependency if the schema is managed elsewhere.",
        )
    return MigrationProfile(tool=tool, directory=directory), [
        Evidence(field="migrations.tool", value=tool.value, source=build_source),
        Evidence(field="migrations.directory", value=directory, source=directory),
    ]


# ───────────────────────────────  the phase  ─────────────────────────────


def onboard(repo: str | Path) -> RepoProfile:
    """Inspect a repository and produce its profile, or refuse saying why."""
    root = Path(repo).expanduser().resolve()
    if not root.is_dir():
        raise OnboardingRefusal(
            question=f"is {root} a directory?",
            looked_in=[str(root)],
            advice="Pass --repo <path-to-repository>.",
        )
    if not (root / ".git").exists():
        raise OnboardingRefusal(
            question=f"is {root} a git repository?",
            looked_in=[str(root / ".git")],
            advice="PerfAgent works in git worktrees: blue is read-only and "
                   "every write goes to a green worktree branched from a "
                   "commit. A directory with no history cannot be branched.",
        )

    evidence: list[Evidence] = []
    build, build_ev = detect_build(root)
    evidence += build_ev
    layout, layout_ev = detect_layout(root)
    evidence += layout_ev
    entrypoint, entry_ev = detect_entrypoint(root, layout)
    evidence += entry_ev
    run_profile, profile_ev = detect_run_profile(root)
    evidence += profile_ev
    database, db_ev = detect_database(root)
    evidence += db_ev
    migrations, migration_ev = detect_migrations(root)
    evidence += migration_ev

    # The write allowlist is DERIVED. There is no argument to this function
    # that widens it, and RepoProfile's own validator refuses an entry that
    # is not under the source root.
    allowlist = [layout.source_root]
    if migrations.directory:
        # A migration is written by `add_migration`, which parses the SQL
        # first; the allowlist still has to admit the path.
        allowlist.append(migrations.directory)

    profile = RepoProfile(
        repo=str(root),
        build=build,
        layout=layout,
        entrypoint=entrypoint,
        run_profile=run_profile,
        database=database,
        migrations=migrations,
        write_allowlist=allowlist,
        evidence=evidence,
    )
    return profile


def assert_instrumentable(profile: RepoProfile) -> None:
    """Refuse a database the measurement phase cannot read.

    Called at ONBOARDING, not at baseline. The whole reason this phase runs
    first is so that an unsupported stack costs zero tokens instead of being
    discovered as a driver import error after diagnosis.
    """
    if profile.database.kind is DatabaseKind.NONE:
        return
    if profile.database.supported:
        return
    raise OnboardingRefusal(
        question=f"can SQL be attributed on a {profile.database.kind.value} "
                 f"database?",
        looked_in=[profile.evidence_for("database.kind") or "(unknown)"],
        advice=f"{profile.database.unsupported_reason} PerfAgent's supported "
               f"stack is JVM web services on MySQL; see PERFAGENT.md. Point "
               f"it at a MySQL-backed service, or run with the database "
               f"phases disabled once that is supported.",
    )


# ──────────────────────────────  persistence  ────────────────────────────


def save_profile(profile: RepoProfile, path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(profile.model_dump(mode="json"), indent=2,
                   ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return out


def load_profile(path: str | Path) -> RepoProfile:
    source = Path(path)
    if not source.exists():
        raise ContractViolation(
            f"no repo profile at {source}. Run the onboard phase first — "
            f"every repo-specific path the pipeline uses comes from it."
        )
    payload: Any = json.loads(source.read_text(encoding="utf-8"))
    return RepoProfile.model_validate(payload)


def render_profile(profile: RepoProfile) -> str:
    """The profile, for a person. Every line cites where it was read."""
    out: list[str] = ["", f"REPO PROFILE  ·  {profile.repo}",
                      f"  decided by     {profile.decided_by}", ""]

    def row(label: str, value: str, field: str) -> None:
        source = profile.evidence_for(field)
        suffix = f"   ← {source}" if source else ""
        out.append(f"  {label:<16}{value}{suffix}")

    row("build tool", profile.build.tool.value, "build.tool")
    row("build cmd", " ".join(profile.build.invocation
                              + profile.build.build_args), "build.invocation")
    out.append(f"  {'test cmd':<16}"
               + " ".join(profile.build.invocation + profile.build.test_args))
    out.append(f"  {'classes dir':<16}{profile.build.classes_dir}")
    row("entrypoint", profile.entrypoint, "entrypoint")
    row("run profile", profile.run_profile or "(repo default)", "run_profile")
    out.append("")
    row("source root", profile.layout.source_root, "layout.source_root")
    row("test root", profile.layout.test_root, "layout.test_root")
    out.append(f"  {'resources':<16}"
               + (", ".join(profile.layout.resource_roots) or "(none)"))
    out.append("")
    row("database", profile.database.kind.value, "database.kind")
    if profile.database.jdbc_url:
        row("jdbc", profile.database.jdbc_url, "database.jdbc_url")
    out.append(f"  {'instrumentable':<16}"
               f"{'yes' if profile.database.supported else 'NO'}")
    if profile.database.unsupported_reason:
        out.append(f"      {profile.database.unsupported_reason}")
    out.append("")
    row("migrations", profile.migrations.tool.value, "migrations.tool")
    if profile.migrations.directory:
        row("migration dir", profile.migrations.directory,
            "migrations.directory")
    out.append("")
    out.append(f"  {'write allowlist':<16}"
               + ", ".join(profile.write_allowlist))
    out.append(f"  {'':16}(derived from the source root; cannot be widened)")
    return "\n".join(out)
