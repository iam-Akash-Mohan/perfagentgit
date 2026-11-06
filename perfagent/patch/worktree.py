"""The green worktree: an isolated checkout the agent is allowed to write to.

Blue — the original working copy at TARGET_REPO — is never written. `git
worktree add` does record metadata under blue's `.git/worktrees/`, which is
unavoidable and is how the feature works; what matters, and what the acceptance
check verifies, is that blue's working tree and index are untouched, so
`git status` in blue reports exactly what it reported before.

One wrinkle this had to solve for real: a worktree is checked out from a
COMMIT, so anything uncommitted in blue does not exist in green. In this repo
that includes `application-perfagent.yml` (the whole runtime profile) and the
`pom.xml` additions for the actuator and the MySQL driver. A green worktree
without them is not the application under test, so `carry_over_uncommitted()`
copies them across and records exactly what it copied. Copying into green never
writes to blue.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from ..errors import ContractViolation, ToolError

GREEN_DIR = Path(".perfagent/green")

# Never carried into green: build output and vendored dependencies. Both are
# large, both are regenerated, and copying them would make the diff unreadable.
_NEVER_CARRY = ("target/", "node_modules/", ".git/")

# The fallback set of untracked-file prefixes to carry into green, used ONLY
# when no RepoProfile is supplied. The real list is `profile.build
# .build_inputs`, which is build-tool specific — a Gradle repo has no pom.xml
# and these five prefixes would drop its `settings.gradle` on the floor.
_BUILD_INPUTS = ("src/", "pom.xml", "mvnw", "mvnw.cmd", ".mvn/",
                 "build.gradle", "build.gradle.kts", "settings.gradle",
                 "settings.gradle.kts", "gradlew", "gradle/")

_GIT_TIMEOUT_S = 120


def target_repo(explicit: str | Path | None = None) -> Path:
    """Where blue lives.

    There is no built-in default any more. It used to be one developer's
    home directory, which meant every command in this package silently
    worked on exactly one machine and failed confusingly everywhere else.
    A caller must say which repository it means, or set
    PERFAGENT_TARGET_REPO.
    """
    if explicit is not None:
        return Path(explicit).resolve()
    from_env = os.environ.get("PERFAGENT_TARGET_REPO")
    if from_env:
        return Path(from_env).resolve()
    raise ContractViolation(
        "no target repository. Pass --repo, or set PERFAGENT_TARGET_REPO. "
        "This used to default to a path on one developer's laptop, which is "
        "why nothing said so when it was wrong."
    )


def _git(repo: Path, *args: str, check: bool = True) -> str:
    """Run one git command as an argv list. Never a shell string."""
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        text=True,
        timeout=_GIT_TIMEOUT_S,
    )
    if check and result.returncode != 0:
        raise ContractViolation(
            f"git {' '.join(args)} failed in {repo}: "
            f"{result.stderr.strip() or result.stdout.strip()}"
        )
    return result.stdout


@dataclass
class GreenWorktree:
    """A writable checkout of blue, on its own branch, with a reset point."""

    path: Path
    branch: str
    base_commit: str
    repo: Path
    carried_over: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    checkpoints: list[str] = field(default_factory=list)

    # ─────────────────────────────  lifecycle  ───────────────────────────

    @classmethod
    def create(
        cls,
        run_id: str,
        *,
        repo: str | Path | None = None,
        green: str | Path = GREEN_DIR,
        carry_uncommitted: bool = True,
    ) -> "GreenWorktree":
        """Add a worktree at `green` on a new branch `perfagent/<run_id>`."""
        blue = target_repo(repo)
        if not (blue / ".git").exists():
            raise ContractViolation(f"{blue} is not a git repository")

        green_path = Path(green).resolve()
        if green_path.exists():
            raise ContractViolation(
                f"{green_path} already exists. Remove the previous worktree first: "
                f"git -C {blue} worktree remove --force {green_path}"
            )

        base = _git(blue, "rev-parse", "HEAD").strip()
        branch = f"perfagent/{run_id}"

        existing = _git(blue, "branch", "--list", branch).strip()
        if existing:
            raise ContractViolation(
                f"branch {branch} already exists in {blue}. Runs are identified by "
                "run_id; this one has been used."
            )

        green_path.parent.mkdir(parents=True, exist_ok=True)
        _git(blue, "worktree", "add", "-b", branch, str(green_path), base)

        worktree = cls(
            path=green_path, branch=branch, base_commit=base, repo=blue
        )
        if carry_uncommitted:
            worktree.carry_over_uncommitted()
        # The reset point is taken AFTER the carry-over, so a reset returns to a
        # buildable tree rather than to one missing the runtime profile.
        worktree.checkpoint("baseline: pristine green worktree")
        return worktree

    def carry_over_uncommitted(self) -> list[str]:
        """Copy blue's uncommitted BUILD INPUTS into green.

        A worktree is checked out from a commit, so without this green is not
        the code that is actually running: in this repo the uncommitted set
        includes `application-perfagent.yml` (the whole runtime profile) and the
        `pom.xml` additions for the actuator and the MySQL driver.

        Two deliberate narrowings, both recorded in `skipped` rather than done
        silently:

        * A MODIFIED TRACKED file is always carried — its content differs from
          the commit green was checked out from, so its absence changes the
          build. An UNTRACKED file is carried only under `_BUILD_INPUTS`.
        * Symlinks are never carried. `shutil.copy2` follows them, so copying
          `src/main/resources/escape-link.txt` — which points at
          `/Users/akashm/Documents/mini-spring-OUTSIDE.txt` — would import
          content from outside the repository into the tree the agent edits.

        Reads blue, writes only green.
        """
        copied: list[str] = []
        skipped: list[str] = []
        # -uall so untracked directories are listed as their individual files,
        # rather than as a directory this code would have to walk itself.
        status = _git(self.repo, "status", "--porcelain", "-uall")

        for line in status.splitlines():
            if not line.strip():
                continue
            code, rel = line[:2], line[3:].strip()
            if "D" in code:
                continue  # a deletion in blue is not something to copy
            if rel.startswith(_NEVER_CARRY):
                continue

            untracked = code == "??"
            if untracked and not rel.startswith(_BUILD_INPUTS):
                skipped.append(f"{rel} (untracked, not a build input)")
                continue

            source = self.repo / rel
            if source.is_symlink():
                skipped.append(f"{rel} (symlink — would import content from outside)")
                continue
            if not source.is_file():
                continue
            self._copy_one(rel, copied)

        self.carried_over = copied
        self.skipped = skipped
        return copied

    def _copy_one(self, rel: str, copied: list[str]) -> None:
        destination = self.path / rel
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(self.repo / rel, destination)
        copied.append(rel)

    def remove(self) -> None:
        """Drop the worktree and its branch. Blue's working tree is untouched."""
        _git(self.repo, "worktree", "remove", "--force", str(self.path), check=False)
        _git(self.repo, "branch", "-D", self.branch, check=False)

    # ─────────────────────────────  checkpoints  ─────────────────────────

    def checkpoint(self, label: str) -> str:
        """Commit the current state of green so a bad iteration is a reset.

        `--allow-empty` on purpose: an iteration that changed nothing still gets
        a reset point, so the checkpoint list and the iteration list stay in
        step and there is never an off-by-one when resetting.
        """
        _git(self.path, "add", "-A")
        _git(
            self.path,
            "-c", "user.email=perfagent@local",
            "-c", "user.name=perfagent",
            "commit", "--allow-empty", "-q", "-m", label,
        )
        sha = _git(self.path, "rev-parse", "HEAD").strip()
        self.checkpoints.append(sha)
        return sha

    def reset_to_checkpoint(self, sha: str | None = None) -> str:
        """Hard-reset green to a checkpoint and delete untracked leftovers.

        A bad state is a reset, not a debugging session — an agent that has to
        unpick its own half-applied edit spends its remaining iterations on
        bookkeeping instead of on the problem.
        """
        target = sha or (self.checkpoints[0] if self.checkpoints else self.base_commit)
        _git(self.path, "reset", "--hard", "-q", target)
        _git(self.path, "clean", "-qfd")
        return target

    # ──────────────────────────────  reading  ────────────────────────────

    def diff(self, against: str | None = None) -> str:
        """Cumulative diff of green versus its first checkpoint."""
        base = against or (self.checkpoints[0] if self.checkpoints else self.base_commit)
        _git(self.path, "add", "-A")
        return _git(self.path, "diff", base, "--")

    def source_diff(self, against: str | None = None) -> str:
        """Diff restricted to what can change behaviour.

        `diff()` stages everything, so a build artifact dropped into the tree by
        an arm build shows up as part of the patch. The patch identity that
        `measure` checks against its verification has to be stable across
        builds, so it is computed from source paths only.
        """
        base = against or (self.checkpoints[0] if self.checkpoints else self.base_commit)
        _git(self.path, "add", "-A")
        return _git(self.path, "diff", base, "--", "src", "pom.xml")

    def diff_stat(self, against: str | None = None) -> tuple[list[str], int, int]:
        """(files changed, lines added, lines removed) — computed, never counted
        by a model."""
        base = against or (self.checkpoints[0] if self.checkpoints else self.base_commit)
        _git(self.path, "add", "-A")
        raw = _git(self.path, "diff", "--numstat", base, "--")
        files: list[str] = []
        added = removed = 0
        for line in raw.splitlines():
            parts = line.split("\t")
            if len(parts) != 3:
                continue
            plus, minus, name = parts
            files.append(name)
            added += int(plus) if plus.isdigit() else 0
            removed += int(minus) if minus.isdigit() else 0
        return files, added, removed

    def blue_status(self) -> str:
        """Blue's porcelain status, for the untouched-working-copy check."""
        return _git(self.repo, "status", "--porcelain")


def existing_green_branch(path: str | Path, *, repo: str | Path | None = None) -> str:
    """The branch a worktree on disk is currently on."""
    directory = Path(path).resolve()
    if not (directory / ".git").exists():
        return "(not a worktree)"
    return _git(directory, "rev-parse", "--abbrev-ref", "HEAD").strip()


def resolve_java_home(explicit: str | Path | None = None) -> str | None:
    """Find a JDK the target's build can actually use.

    Not incidental: this machine's default `java` is 11, mini-spring targets 21,
    and spring-boot-maven-plugin 3.2.0 is compiled for class file 61. Without
    this, every build in every iteration fails with "has been compiled by a more
    recent version of the Java Runtime" and the agent spends its whole budget
    trying to fix source code that was never wrong.
    """
    if explicit:
        return str(explicit)
    from_env = os.environ.get("PERFAGENT_JAVA_HOME")
    if from_env:
        return from_env
    try:
        found = subprocess.run(
            ["/usr/libexec/java_home", "-v", "21"],
            capture_output=True, text=True, timeout=15,
        )
        if found.returncode == 0 and found.stdout.strip():
            return found.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return None


def build_env(java_home: str | Path | None = None) -> dict[str, str]:
    """Environment for maven: the resolved JDK first on PATH."""
    env = dict(os.environ)
    resolved = resolve_java_home(java_home)
    if resolved:
        env["JAVA_HOME"] = resolved
        env["PATH"] = f"{Path(resolved) / 'bin'}{os.pathsep}{env.get('PATH', '')}"
    return env
