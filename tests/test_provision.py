"""Provisioning resolves blue's own worktree and records both arms' commits.

No database and no build: `database.kind=NONE` and `require_database=False`
keep these hermetic, since the worktree-resolution behaviour under test
does not depend on either.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from senapathy.contracts.repo import (BuildPlan, BuildTool, DatabaseKind,
                                      DatabaseProfile, Layout, RepoProfile)
from senapathy.provision import provision


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args], capture_output=True, text=True, check=True)
    return result.stdout


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    (root / "src/main/java").mkdir(parents=True)
    (root / "pom.xml").write_text("<project/>\n")
    _git(root, "init", "-q")
    _git(root, "config", "user.email", "t@t")
    _git(root, "config", "user.name", "t")
    _git(root, "add", "-A")
    _git(root, "commit", "-qm", "init")
    return root


def _profile(repo: Path) -> RepoProfile:
    return RepoProfile(
        repo=str(repo),
        build=BuildPlan(tool=BuildTool.MAVEN, invocation=["mvn"],
                        build_args=["-q", "package"], test_args=["-q", "test"],
                        classes_dir="target/classes",
                        build_inputs=["src/", "pom.xml"]),
        layout=Layout(source_root="src/main/java", test_root="src/test/java"),
        entrypoint="com.ex.Application",
        database=DatabaseProfile(kind=DatabaseKind.NONE),
        write_allowlist=["src/main/java"],
    )


def test_provision_records_both_arms_commits(repo: Path, tmp_path: Path) -> None:
    head = _git(repo, "rev-parse", "HEAD").strip()
    record = provision(
        _profile(repo), repo=repo, run_id="r1",
        blue_dir=tmp_path / "blue-arm", require_database=False,
    )
    assert record.blue.commit == head
    assert record.green.commit == head


def test_provision_reuses_a_current_blue(repo: Path, tmp_path: Path) -> None:
    blue_dir = tmp_path / "blue-arm"
    first = provision(_profile(repo), repo=repo, run_id="r1",
                      blue_dir=blue_dir, require_database=False)
    second = provision(_profile(repo), repo=repo, run_id="r2",
                       blue_dir=blue_dir, require_database=False)
    assert first.blue.commit == second.blue.commit


def test_provision_recreates_a_stale_blue(repo: Path, tmp_path: Path) -> None:
    blue_dir = tmp_path / "blue-arm"
    first = provision(_profile(repo), repo=repo, run_id="r1",
                      blue_dir=blue_dir, require_database=False)

    (repo / "src/main/java/New.java").write_text("class New {}\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "a commit blue has not seen")

    second = provision(_profile(repo), repo=repo, run_id="r2",
                       blue_dir=blue_dir, require_database=False)

    assert second.blue.commit != first.blue.commit
    assert (blue_dir / "src/main/java/New.java").exists()
