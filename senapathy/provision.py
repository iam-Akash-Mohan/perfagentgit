"""Phase P0: turn a RepoProfile into two arms that actually answer.

Onboarding reads the repository. Provisioning proves the reading was right:
it picks a free port pair, brings the database up if the repo describes one,
applies the migrations, builds both arms and waits for each to answer its
health check. A repository that onboards cleanly but will not boot is still
a repository this harness cannot work on, and finding that out here costs
one build instead of the whole pipeline.

Every refusal is `ENVIRONMENT_INFEASIBLE` and names the specific thing that
would not start.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from .contracts.repo import (ArmEndpoint, DatabaseKind, MigrationTool,
                             Provisioning, RepoProfile)
from .errors import ContractViolation

DB_READY_TIMEOUT_S = 120.0
DB_POLL_S = 2.0


class ProvisioningRefusal(ContractViolation):
    """Something the repository described would not start, and which."""

    def __init__(self, what: str, detail: str, advice: str) -> None:
        self.what = what
        super().__init__(
            f"cannot provision: {what}\n"
            f"  detail: {detail}\n"
            f"  what to do: {advice}"
        )


def choose_ports(blue: int | None = None,
                 green: int | None = None) -> tuple[int, int]:
    """A free port pair, unless the operator pinned one.

    Pinned ports are still CHECKED — an operator who asks for a port
    something else holds should be told, not silently moved.
    """
    from .measure.arms import free_port_pair, port_is_free

    chosen_blue, chosen_green = free_port_pair()
    blue = blue if blue is not None else chosen_blue
    green = green if green is not None else chosen_green
    if blue == green:
        raise ProvisioningRefusal(
            what="the two arms were given the same port",
            detail=f"blue and green both on {blue}",
            advice="Pass different --blue-port/--green-port, or omit both and "
                   "let the harness choose a free pair.",
        )
    for name, port in (("blue", blue), ("green", green)):
        if not port_is_free(port):
            raise ProvisioningRefusal(
                what=f"port {port} for the {name} arm is already in use",
                detail=f"something is listening on 127.0.0.1:{port}",
                advice=f"Stop it, or omit --{name}-port and let the harness "
                       f"choose a free one. A stale app on that port would be "
                       f"measured instead of the one built here.",
            )
    return blue, green


def database_reachable(profile: RepoProfile) -> tuple[bool, str]:
    """Can we actually connect to the datasource the repo describes?

    Returns (ready, note) rather than raising, so the caller decides whether
    a database is required for this run. A repo with no datasource is ready
    by definition.
    """
    db = profile.database
    if db.kind is DatabaseKind.NONE:
        return True, "repo declares no datasource"
    if db.kind is not DatabaseKind.MYSQL:
        return False, (f"{db.kind.value} is detected but not supported; "
                       f"{db.unsupported_reason}")

    from .measure.sql import MySQLTarget, connect

    try:
        target = MySQLTarget.from_profile(db)
    except Exception as exc:  # noqa: BLE001 — any build failure is "not ready"
        return False, f"no target could be built from the detected profile: {exc}"
    try:
        # `connect` is a @contextmanager; entering it is what actually opens
        # the socket, so the probe has to go through `with`.
        with connect(target):
            pass
    except Exception as exc:  # noqa: BLE001 — any driver error is "not ready"
        return False, f"{target.dsn()} did not accept a connection: {exc}"
    return True, f"connected to {target.dsn()}"


def apply_migrations(profile: RepoProfile, repo: Path,
                     java_home: str | None = None) -> list[str]:
    """Run the repo's own migration tool, through its own build tool.

    Deliberately the repo's tool and not a reimplementation: the point of
    detecting Flyway is to use it, and a harness that applied migrations its
    own way would be testing a schema the application never sees.
    """
    if profile.migrations.tool is MigrationTool.NONE:
        return []
    if profile.migrations.tool is not MigrationTool.FLYWAY:
        # Liquibase is detected but its Maven goal differs; saying so is
        # better than running the wrong goal.
        raise ProvisioningRefusal(
            what=f"{profile.migrations.tool.value} migrations cannot be "
                 f"applied by this harness yet",
            detail="only Flyway's maven/gradle goal is wired up",
            advice="Apply the migrations yourself before the run, or use "
                   "Flyway.",
        )

    from .patch.worktree import build_env

    goal = ("flyway:migrate" if profile.build.tool.value == "maven"
            else "flywayMigrate")
    completed = subprocess.run(
        [*profile.build.invocation, "-q", goal],
        cwd=str(repo), capture_output=True, text=True,
        env=build_env(java_home), timeout=300,
    )
    if completed.returncode != 0:
        output = (completed.stdout or "") + (completed.stderr or "")
        raise ProvisioningRefusal(
            what="the migration tool failed",
            detail=output.strip()[-1200:] or f"exit {completed.returncode}",
            advice="Fix the migration, or bring the database up first — a "
                   "run against an un-migrated schema measures the wrong "
                   "thing.",
        )
    return [goal]


def provision(
    profile: RepoProfile,
    *,
    repo: Path | None = None,
    blue_port: int | None = None,
    green_port: int | None = None,
    java_home: str | None = None,
    require_database: bool = True,
    run_migrations: bool = True,
    run_id: str | None = None,
    blue_dir: str | Path | None = None,
) -> Provisioning:
    """Ports, database, migrations, and blue's own worktree. Does not
    build or boot either arm.

    Building and booting are left to the phases that need a running arm
    (`measure.arms.single_arm` / `both_arms`), because they already know
    how and doing it twice would double the slowest step in the pipeline.
    But resolving blue's WORKTREE — cheap, git-only, no compile and no JVM
    — belongs here: four runs (COL-01, PAY-01, PAY-02) failed deep into
    the pipeline, after understand had already spent real tokens, because
    a blue left over from an earlier run was pinned at a commit the
    repository had since moved past, and corpus generation 404'd on every
    probe against a route that did not exist yet there. Resolving blue
    from the repo's current HEAD here means that failure shows up as a
    provisioning note before a single token is spent, not as a baffling
    wall of 404s two phases later. `prepare_blue_worktree` recreates it,
    rather than reusing it, whenever it is on a different commit.
    """
    root = Path(repo) if repo is not None else Path(profile.repo)
    notes: list[str] = []

    blue, green = choose_ports(blue_port, green_port)
    notes.append(f"ports chosen at runtime: blue={blue} green={green}")

    ready, note = database_reachable(profile)
    notes.append(note)
    if require_database and not ready:
        raise ProvisioningRefusal(
            what="the database the repository describes is not reachable",
            detail=note,
            advice="Start it (a docker-compose.yml in the target repo is the "
                   "usual way), or set SENAPATHY_MYSQL_* to point at a "
                   "running instance.",
        )

    applied: list[str] = []
    if ready and run_migrations:
        applied = apply_migrations(profile, root, java_home)
        if applied:
            notes.append(f"migrations applied via {', '.join(applied)}")

    from .measure.arms import BLUE_DIR, prepare_blue_worktree

    blue_worktree = prepare_blue_worktree(
        run_id or "provision", repo=root, blue=blue_dir or BLUE_DIR)
    commit = blue_worktree.base_commit
    notes.append(f"blue resolved at {commit[:12]} ({root})")

    return Provisioning(
        blue=ArmEndpoint(name="blue", path=".senapathy/blue", port=blue,
                         base_url=f"http://localhost:{blue}", commit=commit),
        # Green does not exist yet — the patch phase creates it, from the
        # same commit blue was just resolved to. Recorded here as the
        # commit it WILL start from, not a claim that it already has.
        green=ArmEndpoint(name="green", path=".senapathy/green", port=green,
                          base_url=f"http://localhost:{green}", commit=commit),
        database_ready=ready,
        migrations_applied=applied,
        notes=notes,
    )


def save_provisioning(provisioning: Provisioning, path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(provisioning.model_dump(mode="json"), indent=2,
                   ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return out


def load_provisioning(path: str | Path) -> Provisioning:
    source = Path(path)
    if not source.exists():
        raise ContractViolation(
            f"no provisioning record at {source}. Run the provision phase — "
            f"the arm ports come from it, and they are chosen at runtime "
            f"rather than fixed."
        )
    return Provisioning.model_validate(
        json.loads(source.read_text(encoding="utf-8")))


def render_provisioning(provisioning: Provisioning) -> str:
    out = ["", "PROVISIONING", ""]
    for arm in (provisioning.blue, provisioning.green):
        commit = f"  commit {arm.commit[:12]}" if arm.commit else ""
        out.append(f"  {arm.name:<8}{arm.base_url}   ({arm.path}){commit}")
    out.append("")
    out.append(f"  database  {'ready' if provisioning.database_ready else 'NOT READY'}")
    if provisioning.migrations_applied:
        out.append(f"  migrations {', '.join(provisioning.migrations_applied)}")
    for note in provisioning.notes:
        out.append(f"    · {note}")
    return "\n".join(out)
