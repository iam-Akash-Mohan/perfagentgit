"""Two arms of the same application, running at once with identical JVM flags.

The comparison is only worth anything if the two processes differ in exactly two
respects: the source they were built from, and the port they listen on. So both
arms are started from ONE argv template here, and `assert_flags_identical`
checks the two command lines differ only where they are allowed to. A heap size
that differs between arms, or a JIT flag present on one side, converts a
measurement into a coincidence.

Neither arm is the developer's working copy. Blue is its own worktree pinned at
the base commit, so the original checkout is never built into, never started,
and never written — invariant 7 — and both arms are built the same way from the
same kind of tree.
"""

from __future__ import annotations

import subprocess
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import httpx

from ..errors import ContractViolation, ToolError
from ..patch.worktree import GreenWorktree, build_env, target_repo

BLUE_DIR = Path(".perfagent/blue")
GREEN_DIR = Path(".perfagent/green")

# mini-spring's perfagent profile pins ${SERVER_PORT:18080}; 8080 on this
# machine is a different application entirely and answers 401.
BLUE_PORT = 18080
GREEN_PORT = 8081

SPRING_PROFILE = "perfagent"
MAIN_CLASS = "com.ex.Application"

# Identical on both arms, and that is the whole point.
#   -Xms == -Xmx        : no arm pays for heap growth the other avoided
#   AlwaysPreTouch      : the heap is committed at startup, so neither arm
#                         takes page faults during the measured window
#   TieredStopAtLevel   : NOT set — we want the real C2 JIT, since warmup is
#                         discarded and steady-state is what is being compared
JVM_FLAGS: tuple[str, ...] = (
    "-Xms1g",
    "-Xmx1g",
    "-XX:+UseG1GC",
    "-XX:+AlwaysPreTouch",
    "-XX:+PerfDisableSharedMem",
    "-Dfile.encoding=UTF-8",
    "-Duser.timezone=UTC",
)

HEALTH_TIMEOUT_S = 180
HEALTH_POLL_S = 1.0
SHUTDOWN_GRACE_S = 15
BUILD_TIMEOUT_S = 900
# Written OUTSIDE the worktree. A build artifact inside green ends up in
# `git add -A`, and then the patch identity changes every time an arm is built
# — which made `measure` refuse a patch it had just verified.
CLASSPATH_DIR = Path(".perfagent/arms")


@dataclass
class Arm:
    name: str
    path: Path
    port: int
    argv: list[str]
    process: subprocess.Popen | None = None
    log_path: Path | None = None

    @property
    def base_url(self) -> str:
        return f"http://localhost:{self.port}"

    def health_url(self) -> str:
        return f"{self.base_url}/actuator/health"

    def tail_log(self, lines: int = 25) -> str:
        """The app's OWN stderr. A boot failure must surface the app's words,
        not ours."""
        if self.log_path is None or not self.log_path.exists():
            return "(no log captured)"
        return "\n".join(self.log_path.read_text(errors="replace").splitlines()[-lines:])


def port_is_free(port: int) -> bool:
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.settimeout(1.0)
        return probe.connect_ex(("127.0.0.1", port)) != 0


def prepare_blue_worktree(
    run_id: str,
    *,
    repo: str | Path | None = None,
    blue: str | Path = BLUE_DIR,
) -> GreenWorktree:
    """A worktree pinned at the base commit, with the same carry-over as green.

    Blue has to be a worktree rather than the developer's checkout for two
    reasons: invariant 7 keeps the original tree unwritten, and an arm built
    from a tree with uncommitted edits is not the baseline the oracle was
    recorded against.
    """
    path = Path(blue)
    if path.exists():
        return existing_worktree(path, repo=repo)
    return GreenWorktree.create(
        f"blue-{run_id}", repo=repo, green=path, carry_uncommitted=True
    )


def existing_worktree(path: str | Path, *, repo: str | Path | None = None) -> GreenWorktree:
    """Adopt a worktree that is already on disk (green, from P12)."""
    directory = Path(path).resolve()
    if not (directory / ".git").exists():
        raise ContractViolation(
            f"{directory} is not a git worktree. Run `perfagent patch` first to "
            "create the green worktree."
        )
    blue_repo = target_repo(repo)
    branch = subprocess.run(
        ["git", "-C", str(directory), "rev-parse", "--abbrev-ref", "HEAD"],
        capture_output=True, text=True, timeout=30,
    ).stdout.strip()
    base = subprocess.run(
        ["git", "-C", str(directory), "rev-parse", "HEAD"],
        capture_output=True, text=True, timeout=30,
    ).stdout.strip()
    return GreenWorktree(path=directory, branch=branch, base_commit=base, repo=blue_repo)


def classpath_file(path: Path, classpath_dir: Path = CLASSPATH_DIR) -> Path:
    """Where an arm's runtime classpath is materialised, keyed by worktree."""
    return (Path(classpath_dir) / f"{Path(path).name}-classpath.txt").resolve()


def build_arm(
    path: Path, *, java_home: str | None = None, classpath_dir: Path = CLASSPATH_DIR
) -> None:
    """Compile, then materialise the runtime classpath.

    `mvn` reads JAVA_HOME rather than PATH, which is why this goes through
    build_env: without it the compiler reports "release version 21 not
    supported" on a machine whose default java is 11, and the failure looks
    like a source problem.
    """
    env = build_env(java_home)
    out = classpath_file(path, classpath_dir)
    out.parent.mkdir(parents=True, exist_ok=True)
    for args, label in (
        (["-q", "-DskipTests", "package"], "build"),
        (
            [
                "-q",
                "dependency:build-classpath",
                f"-Dmdep.outputFile={out}",
                "-DincludeScope=runtime",
            ],
            "classpath",
        ),
    ):
        completed = subprocess.run(
            ["mvn", *args], cwd=str(path), capture_output=True, text=True,
            env=env, timeout=BUILD_TIMEOUT_S,
        )
        if completed.returncode != 0:
            output = (completed.stdout or "") + (completed.stderr or "")
            raise ToolError(
                f"{label} failed in {path} (exit {completed.returncode}):\n"
                + "\n".join(output.splitlines()[:40])
            )


def assert_build_current(path: Path) -> None:
    """Refuse to run an arm whose compiled classes are older than its source.

    This bit for real: `--no-build` after restoring a source file meant the arm
    still served the previous bytecode, so `verify` reported a behaviour change
    for code that was no longer there — while the patch identity, computed from
    source, said the patch was fine. A verdict about different code than the
    identity records is worse than no verdict.
    """
    classes = path / "target" / "classes"
    if not classes.exists():
        raise ContractViolation(
            f"{classes} does not exist, so this arm has never been built. "
            "Drop --no-build."
        )
    newest_class = max(
        (f.stat().st_mtime for f in classes.rglob("*.class")), default=0.0
    )
    sources = list((path / "src" / "main").rglob("*"))
    stale = [
        f for f in sources
        if f.is_file() and f.stat().st_mtime > newest_class
    ]
    if stale:
        listed = ", ".join(str(f.relative_to(path)) for f in stale[:5])
        raise ContractViolation(
            f"{path.name}: {len(stale)} source file(s) are newer than the "
            f"compiled classes ({listed}). The arm would serve stale bytecode "
            "and the verdict would describe code that is no longer there. "
            "Re-run without --no-build."
        )


def arm_argv(
    path: Path, port: int, *, java_home: str | None = None,
    classpath_dir: Path = CLASSPATH_DIR,
) -> list[str]:
    """One template, both arms. Only -Dserver.port and the classpath root differ."""
    cp_file = classpath_file(path, classpath_dir)
    if not cp_file.exists():
        raise ContractViolation(
            f"{cp_file} is missing — build_arm() has not run for {path}"
        )
    dependencies = cp_file.read_text(encoding="utf-8").strip()
    classpath = ":".join(
        [str(path / "target" / "classes"), str(path / "src" / "main" / "resources"),
         dependencies]
    )
    java = str(Path(build_env(java_home)["JAVA_HOME"]) / "bin" / "java")
    return [
        java,
        *JVM_FLAGS,
        "-cp", classpath,
        f"-Dserver.port={port}",
        f"-Dspring.profiles.active={SPRING_PROFILE}",
        MAIN_CLASS,
    ]


def assert_flags_identical(blue: Arm, green: Arm) -> None:
    """The two command lines may differ ONLY in port and classpath root.

    Checked rather than trusted: this is the one precondition whose failure
    would silently invalidate every number the phase produces, and it is
    cheaper to compare two lists than to explain a bogus 30% win later.
    """
    def normalise(arm: Arm) -> list[str]:
        out: list[str] = []
        for token in arm.argv:
            if token.startswith("-Dserver.port="):
                out.append("-Dserver.port=<port>")
            elif str(arm.path) in token:
                out.append(token.replace(str(arm.path), "<arm>"))
            else:
                out.append(token)
        return out

    blue_flags, green_flags = normalise(blue), normalise(green)
    if blue_flags != green_flags:
        difference = [
            f"  blue : {b}\n  green: {g}"
            for b, g in zip(blue_flags, green_flags) if b != g
        ] or [f"  blue has {len(blue_flags)} tokens, green has {len(green_flags)}"]
        raise ContractViolation(
            "the two arms would run with DIFFERENT flags, which invalidates any "
            "comparison between them:\n" + "\n".join(difference)
        )


def start_arm(arm: Arm, log_dir: Path) -> Arm:
    log_dir.mkdir(parents=True, exist_ok=True)
    arm.log_path = log_dir / f"{arm.name}.log"
    handle = arm.log_path.open("w", encoding="utf-8")
    arm.process = subprocess.Popen(
        arm.argv, cwd=str(arm.path), stdout=handle, stderr=subprocess.STDOUT
    )
    return arm


def await_health(arm: Arm, timeout_s: float = HEALTH_TIMEOUT_S) -> dict:
    """Poll until UP, or fail with the app's own output."""
    deadline = time.monotonic() + timeout_s
    last: str = "no response"
    with httpx.Client(timeout=5.0) as client:
        while time.monotonic() < deadline:
            if arm.process is not None and arm.process.poll() is not None:
                raise ToolError(
                    f"the {arm.name} arm exited during startup with code "
                    f"{arm.process.returncode}. Its own output:\n{arm.tail_log()}"
                )
            try:
                response = client.get(arm.health_url())
                if response.status_code == 200:
                    payload = response.json()
                    if payload.get("status") == "UP":
                        return payload
                    last = f"health reported {payload.get('status')}"
                else:
                    last = f"HTTP {response.status_code}"
            except httpx.HTTPError as exc:
                last = f"{type(exc).__name__}"
            time.sleep(HEALTH_POLL_S)

    raise ToolError(
        f"the {arm.name} arm on port {arm.port} never became healthy within "
        f"{timeout_s:.0f}s (last: {last}). Its own output:\n{arm.tail_log()}"
    )


def stop_arm(arm: Arm) -> None:
    """SIGTERM, then SIGKILL. Always called, including on failure."""
    if arm.process is None or arm.process.poll() is not None:
        return
    arm.process.terminate()
    try:
        arm.process.wait(timeout=SHUTDOWN_GRACE_S)
    except subprocess.TimeoutExpired:
        arm.process.kill()
        arm.process.wait(timeout=SHUTDOWN_GRACE_S)


@dataclass
class ArmPair:
    blue: Arm
    green: Arm
    health: dict[str, dict] = field(default_factory=dict)

    def argv_record(self) -> dict[str, list[str]]:
        """Both command lines, for the report. Auditable, not asserted-and-lost."""
        return {"blue": list(self.blue.argv), "green": list(self.green.argv)}


@contextmanager
def single_arm(
    run_id: str,
    *,
    which: str = "blue",
    repo: str | Path | None = None,
    path: str | Path | None = None,
    port: int = BLUE_PORT,
    java_home: str | None = None,
    log_dir: Path = Path(".perfagent/arms"),
    build: bool = True,
) -> Iterator[Arm]:
    """Run ONE arm — what the oracle, baseline and diagnosis phases need.

    Those three phases run before a patch exists, so there is no green tree to
    pair with. They go through the same build, argv template and health check as
    `both_arms` so that a recording made here and a measurement made later are
    made against a process started the same way; a different launcher would put
    a JVM difference inside the before/after comparison.
    """
    if not port_is_free(port):
        raise ContractViolation(
            f"port {port} is already in use, so the {which} arm cannot start "
            f"there. Stop whatever is listening on {port} first."
        )

    if path is not None:
        worktree = existing_worktree(path, repo=repo)
    elif which == "blue":
        worktree = prepare_blue_worktree(run_id, repo=repo)
    else:
        worktree = existing_worktree(GREEN_DIR, repo=repo)

    if build:
        build_arm(worktree.path, java_home=java_home)
    else:
        assert_build_current(worktree.path)

    arm = Arm(which, worktree.path, port,
              arm_argv(worktree.path, port, java_home=java_home))
    try:
        start_arm(arm, log_dir)
        await_health(arm)
        yield arm
    finally:
        stop_arm(arm)


@contextmanager
def both_arms(
    run_id: str,
    *,
    repo: str | Path | None = None,
    blue_dir: str | Path = BLUE_DIR,
    green_dir: str | Path = GREEN_DIR,
    blue_port: int = BLUE_PORT,
    green_port: int = GREEN_PORT,
    java_home: str | None = None,
    log_dir: Path = Path(".perfagent/arms"),
    build: bool = True,
) -> Iterator[ArmPair]:
    """Build and run both arms simultaneously; shut both down on the way out."""
    for name, port in (("blue", blue_port), ("green", green_port)):
        if not port_is_free(port):
            raise ContractViolation(
                f"port {port} is already in use, so the {name} arm cannot start "
                f"there. Stop whatever is listening on {port} first — a stale "
                "app would be measured instead of the one built here."
            )

    green_worktree = existing_worktree(green_dir, repo=repo)
    blue_worktree = prepare_blue_worktree(run_id, repo=repo, blue=blue_dir)

    if build:
        build_arm(blue_worktree.path, java_home=java_home)
        build_arm(green_worktree.path, java_home=java_home)
    else:
        # Skipping the build is allowed, serving stale bytecode is not.
        assert_build_current(blue_worktree.path)
        assert_build_current(green_worktree.path)

    blue = Arm("blue", blue_worktree.path, blue_port,
               arm_argv(blue_worktree.path, blue_port, java_home=java_home))
    green = Arm("green", green_worktree.path, green_port,
                arm_argv(green_worktree.path, green_port, java_home=java_home))
    assert_flags_identical(blue, green)

    pair = ArmPair(blue=blue, green=green)
    try:
        start_arm(blue, log_dir)
        start_arm(green, log_dir)
        # Started together, then both awaited: they warm up under the same
        # machine conditions rather than one before the other.
        pair.health["blue"] = await_health(blue)
        pair.health["green"] = await_health(green)
        yield pair
    finally:
        stop_arm(green)
        stop_arm(blue)
