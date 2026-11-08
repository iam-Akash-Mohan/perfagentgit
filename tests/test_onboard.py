"""P-1 tests: what onboarding detects, and what it refuses.

Every test builds a throwaway repository on disk. None of them reads the
machine's real target repo -- a detector tested only against the one repo it
was written for is the hardcoding this phase exists to delete.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from senapathy.contracts.repo import (BuildTool, DatabaseKind, Layout,
                                      MigrationTool, RepoProfile)
from senapathy.onboard import (OnboardingRefusal, assert_instrumentable,
                               load_profile, onboard, save_profile)

APP = """package com.example.demo;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;

@SpringBootApplication
public class DemoApplication {
    public static void main(String[] args) {
        SpringApplication.run(DemoApplication.class, args);
    }
}
"""


def _repo(tmp_path: Path, *, build: str = "maven", db: str = "mysql",
          flyway: bool = True, source: bool = True, tests: bool = True,
          app: bool = True, profile: str | None = "perfagent") -> Path:
    root = tmp_path / "repo"
    (root / "src/main/resources").mkdir(parents=True)

    if build == "maven":
        pom = "<project><artifactId>demo</artifactId>"
        if flyway:
            pom += "<dependency><artifactId>flyway-core</artifactId></dependency>"
        pom += "</project>\n"
        (root / "pom.xml").write_text(pom)
    elif build == "gradle":
        body = "plugins { id 'java' }\n"
        if flyway:
            body += "dependencies { implementation 'org.flywaydb:flyway-core' }\n"
        (root / "build.gradle").write_text(body)

    if source:
        (root / "src/main/java/com/example/demo").mkdir(parents=True)
        if app:
            (root / "src/main/java/com/example/demo/DemoApplication.java"
             ).write_text(APP)
    if tests:
        (root / "src/test/java").mkdir(parents=True)

    if flyway:
        (root / "src/main/resources/db/migration").mkdir(parents=True)

    urls = {
        "mysql": "jdbc:mysql://localhost:3307/demo?useSSL=false",
        "postgres": "jdbc:postgresql://localhost:5432/demo",
        "none": None,
    }
    config = "spring:\n  datasource:\n"
    if urls[db]:
        config += f"    url: {urls[db]}\n    username: demo\n    password: demo\n"
    name = f"application-{profile}.yml" if profile else "application.yml"
    (root / "src/main/resources" / name).write_text(config)

    subprocess.run(["git", "init", "-q", str(root)], check=True,
                   capture_output=True)
    return root


# ────────────────────────────  what it detects  ──────────────────────────


def test_a_maven_spring_repo_is_fully_detected(tmp_path: Path) -> None:
    profile = onboard(_repo(tmp_path))

    assert profile.build.tool is BuildTool.MAVEN
    assert profile.build.invocation == ["mvn"]          # no mvnw in this repo
    assert profile.build.classes_dir == "target/classes"
    assert profile.layout.source_root == "src/main/java"
    assert profile.layout.test_root == "src/test/java"
    # The FQCN, from the package statement plus the file name -- not the
    # filename alone, and not a constant.
    assert profile.entrypoint == "com.example.demo.DemoApplication"
    assert profile.run_profile == "perfagent"
    assert profile.database.kind is DatabaseKind.MYSQL
    assert profile.database.port == 3307
    assert profile.database.database == "demo"
    assert profile.database.user == "demo"
    assert profile.migrations.tool is MigrationTool.FLYWAY
    assert profile.migrations.directory == "src/main/resources/db/migration"


def test_the_wrapper_is_preferred_over_the_binary_on_path(tmp_path: Path) -> None:
    """A repo that ships mvnw has stated which Maven it expects."""
    root = _repo(tmp_path)
    (root / "mvnw").write_text("#!/bin/sh\n")
    assert onboard(root).build.invocation == ["./mvnw"]


def test_gradle_is_detected_with_its_own_layout_and_inputs(tmp_path: Path) -> None:
    profile = onboard(_repo(tmp_path, build="gradle"))
    assert profile.build.tool is BuildTool.GRADLE
    assert profile.build.classes_dir == "build/classes/java/main"
    # The build inputs carried into the green worktree are build-tool
    # specific: a Gradle repo has no pom.xml to carry.
    assert "build.gradle" in profile.build.build_inputs
    assert "pom.xml" not in profile.build.build_inputs


def test_every_detected_value_cites_the_file_it_came_from(tmp_path: Path) -> None:
    profile = onboard(_repo(tmp_path))
    for field in ("build.tool", "layout.source_root", "entrypoint",
                  "database.kind", "migrations.tool"):
        assert profile.evidence_for(field), f"{field} has no evidence"
    assert profile.evidence_for("entrypoint").endswith("DemoApplication.java")


def test_a_repo_with_no_datasource_is_legal(tmp_path: Path) -> None:
    """Not every service has a database. That is an answer, not a failure."""
    profile = onboard(_repo(tmp_path, db="none"))
    assert profile.database.kind is DatabaseKind.NONE
    assert_instrumentable(profile)      # must not raise


# ────────────────────────────  what it refuses  ──────────────────────────


def test_postgres_is_detected_and_refused_at_onboarding(tmp_path: Path) -> None:
    """The supported stack is MySQL. Refusing HERE is the point: the
    alternative is a driver crash after the oracle has been recorded."""
    profile = onboard(_repo(tmp_path, db="postgres"))
    assert profile.database.kind is DatabaseKind.POSTGRES
    assert profile.database.supported is False

    with pytest.raises(OnboardingRefusal) as exc:
        assert_instrumentable(profile)
    message = str(exc.value)
    assert "performance_schema" in message
    assert "MySQL" in message


def test_no_build_file_is_refused_naming_what_was_looked_for(tmp_path: Path) -> None:
    root = tmp_path / "bare"
    root.mkdir()
    subprocess.run(["git", "init", "-q", str(root)], check=True,
                   capture_output=True)
    with pytest.raises(OnboardingRefusal) as exc:
        onboard(root)
    message = str(exc.value)
    assert "which build tool" in message
    assert "pom.xml" in message and "build.gradle" in message


def test_a_missing_test_root_is_refused_because_the_guardrail_needs_it(
    tmp_path: Path
) -> None:
    with pytest.raises(OnboardingRefusal) as exc:
        onboard(_repo(tmp_path, tests=False))
    assert "tests live" in str(exc.value)
    assert "invariant 3" in str(exc.value)


def test_no_entrypoint_is_refused(tmp_path: Path) -> None:
    with pytest.raises(OnboardingRefusal) as exc:
        onboard(_repo(tmp_path, app=False))
    assert "starts the application" in str(exc.value)


def test_two_entrypoints_are_ambiguous_and_refused(tmp_path: Path) -> None:
    """Picking one for the operator is how the wrong arm gets launched."""
    root = _repo(tmp_path)
    second = root / "src/main/java/com/example/other"
    second.mkdir(parents=True)
    (second / "OtherApplication.java").write_text(
        APP.replace("com.example.demo", "com.example.other")
           .replace("DemoApplication", "OtherApplication"))
    with pytest.raises(OnboardingRefusal) as exc:
        onboard(root)
    assert "found several" in str(exc.value)


def test_flyway_without_a_migration_directory_is_refused(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    import shutil
    shutil.rmtree(root / "src/main/resources/db/migration")
    with pytest.raises(OnboardingRefusal) as exc:
        onboard(root)
    assert "migrations live" in str(exc.value)


def test_a_directory_with_no_git_history_is_refused(tmp_path: Path) -> None:
    root = tmp_path / "nogit"
    (root / "src/main/java").mkdir(parents=True)
    (root / "pom.xml").write_text("<project/>")
    with pytest.raises(OnboardingRefusal) as exc:
        onboard(root)
    assert "git repository" in str(exc.value)


# ───────────────────  the allowlist cannot be widened  ───────────────────


def _profile_with_allowlist(entries: list[str]) -> RepoProfile:
    base = {
        "repo": "/tmp/x",
        "build": {"tool": "maven", "invocation": ["mvn"],
                  "build_args": [], "test_args": [],
                  "classes_dir": "target/classes", "build_inputs": ["src/"]},
        "layout": {"source_root": "src/main/java", "test_root": "src/test/java"},
        "entrypoint": "com.example.App",
        "database": {"kind": "none"},
        "write_allowlist": entries,
    }
    return RepoProfile.model_validate(base)


@pytest.mark.parametrize("entry", ["", ".", "/", "./", "*", "**"])
def test_an_allowlist_that_is_the_repo_root_is_refused(entry: str) -> None:
    with pytest.raises(ValueError, match="repository root"):
        _profile_with_allowlist([entry])


def test_an_allowlist_entry_outside_the_derived_paths_is_refused() -> None:
    """A profile cannot widen its own allowlist -- not to the build file,
    not to the test tree, not anywhere onboarding did not derive."""
    with pytest.raises(ValueError, match="neither under the detected"):
        _profile_with_allowlist(["src/main/java", "src/test/java"])
    with pytest.raises(ValueError, match="neither under the detected"):
        _profile_with_allowlist(["src/main/java", "pom.xml"])


def test_an_absolute_or_escaping_entry_is_refused() -> None:
    with pytest.raises(ValueError, match="absolute or escapes"):
        _profile_with_allowlist(["/etc"])
    with pytest.raises(ValueError, match="absolute or escapes"):
        _profile_with_allowlist(["src/main/java/../../.."])


def test_the_migration_directory_is_admitted_even_though_it_is_a_sibling(
    tmp_path: Path
) -> None:
    """On a conventional Maven layout the migration dir is under
    src/main/resources, a sibling of src/main/java. It is a DERIVED path and
    must be admitted; an earlier, stricter rule refused it."""
    profile = onboard(_repo(tmp_path))
    assert "src/main/resources/db/migration" in profile.write_allowlist


# ────────────────────────────  round-tripping  ───────────────────────────


def test_a_profile_survives_save_and_load(tmp_path: Path) -> None:
    profile = onboard(_repo(tmp_path))
    path = save_profile(profile, tmp_path / "run" / "repo_profile.json")
    assert json.loads(path.read_text())["decided_by"] == "code"
    assert load_profile(path) == profile
