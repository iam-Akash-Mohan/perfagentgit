from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import typer

from .errors import ContractViolation, ToolError
from .measure.baseline import baseline_command
from .plan.cli import plan_command
from .understand.cli import understand_command

app = typer.Typer(
    name="perfagent",
    help="Autonomous performance-bottleneck agent for Spring Boot endpoints.",
    no_args_is_help=True,
)

# Commands are imported functions, not bodies defined here. `understand` and
# `baseline` have moved; the rest follow in their own sessions.
app.command(name="understand")(understand_command)
app.command(name="baseline")(baseline_command)
app.command(name="plan")(plan_command)


@app.command()
def diagnose(
    url: str | None = typer.Option(None, "--url",
                            help="Base URL of the running app (probes hit this)."),
    run_dir: Path | None = typer.Option(
        None, "--run-dir", help="Run directory holding baseline.json. Defaults to latest."),
    endpoint_model: Path = typer.Option(
        Path(".perfagent/endpoint_model.json"), "--endpoint-model",
        help="Comprehension artifact from the understand phase."),
    max_steps: int = typer.Option(20, "--max-steps", help="Agent turn budget."),
    evidence_only: bool = typer.Option(
        False, "--evidence-only",
        help="Render the evidence the model would see and exit. No LLM call."),
) -> None:
    """Generate and rank competing bottleneck hypotheses."""
    from .diagnose.agent import render_diagnosis, run_diagnosis
    from .diagnose.evidence import (latest_run_dir, load_baseline,
                                    load_endpoint_model, render_evidence)

    try:
        directory = run_dir if run_dir is not None else latest_run_dir()
        if evidence_only:
            typer.echo(render_evidence(
                load_baseline(directory / "baseline.json"),
                load_endpoint_model(endpoint_model),
            ))
            return

        hypothesis_set, probe_log, path = run_diagnosis(
            base_url=url,
            run_dir=directory,
            endpoint_model_path=endpoint_model,
            max_steps=max_steps,
        )
    except (ToolError, ContractViolation) as exc:
        typer.secho(f"diagnose failed: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc

    # The artifact is the source of truth for what happened, including a loop
    # failure that occurred after the diagnosis was already accepted.
    import json as _json

    loop_error = _json.loads(path.read_text()).get("loop_error_after_acceptance")
    typer.echo(render_diagnosis(hypothesis_set, probe_log, loop_error))
    typer.echo(f"written: {path}")


@app.command()
def patch(
    run_dir: Path | None = typer.Option(
        None, "--run-dir", help="Run directory holding hypotheses.json. Defaults to latest."),
    run_id: str | None = typer.Option(
        None, "--run-id", help="Branch suffix for the green worktree. Defaults to a fresh ULID."),
    repo: Path | None = typer.Option(
        None, "--repo", help="Target repo. Env: PERFAGENT_TARGET_REPO."),
    green: Path = typer.Option(
        Path(".perfagent/green"), "--green", help="Where the green worktree is created."),
    max_iterations: int = typer.Option(
        8, "--max-iterations", help="Build/test budget before abandoning."),
    max_steps: int = typer.Option(24, "--max-steps", help="Agent turn budget."),
    java_home: str | None = typer.Option(
        None, "--java-home", help="JDK for maven. Env: PERFAGENT_JAVA_HOME."),
    keep_worktree: bool = typer.Option(
        True, "--keep-worktree/--remove-worktree",
        help="Leave the green worktree in place for inspection."),
    prompt_only: bool = typer.Option(
        False, "--prompt-only",
        help="Create green, print the task the model would see, and exit. No LLM call."),
) -> None:
    """Write, build and test a real fix on the green worktree."""
    from .ids import new_run_id
    from .patch.agent import (build_task, load_hypotheses, load_plan_inputs,
                              render_patch, resolve_run_dir, run_patch)
    from .patch.worktree import GreenWorktree

    identifier = run_id or new_run_id()

    try:
        directory = run_dir if run_dir is not None else resolve_run_dir()
        if prompt_only:
            # Useful before spending tokens: the worktree and the prompt are the
            # two things most likely to be wrong. Load the diagnosis BEFORE
            # creating a worktree, so a missing one leaves nothing to clean up.
            hypotheses = load_hypotheses(directory)
            candidate, model, plan = load_plan_inputs(directory)
            worktree = GreenWorktree.create(identifier, repo=repo, green=green)
            from .patch.agent import _evidence_block, _fixability_reason
            from .patch.standards import load_standards

            typer.echo(build_task(
                candidate=candidate, model=model, green=worktree.path,
                fixability_reason=_fixability_reason(plan, candidate),
                evidence=_evidence_block(hypotheses, candidate),
                standards=load_standards(),
                measured_p50_ms=hypotheses.get("measured_total_ms"),
            ))
            typer.echo("")
            typer.secho(
                f"green worktree at {worktree.path} on {worktree.branch}; "
                f"carried over {len(worktree.carried_over)} uncommitted file(s). "
                f"Remove it with: git -C {worktree.repo} worktree remove --force "
                f"{worktree.path}",
                fg=typer.colors.YELLOW,
            )
            return

        document = run_patch(
            directory,
            run_id=identifier,
            repo=repo,
            green_dir=green,
            max_steps=max_steps,
            max_iterations=max_iterations,
            java_home=java_home,
            keep_worktree=keep_worktree,
        )
    except (ToolError, ContractViolation) as exc:
        typer.secho(f"patch failed: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc

    typer.echo(render_patch(document))
    typer.echo("")
    typer.echo(f"written: {directory / 'patch.json'}")
    typer.echo(f"written: {directory / 'patch.diff'}")

    if document["blue_status_at_end"] != document["blue_status_at_start"]:
        typer.secho(
            "ALARM: the original working copy changed during this run. That must "
            "never happen — investigate before trusting anything here.",
            fg=typer.colors.RED, err=True,
        )
        raise typer.Exit(code=1)

    if document["outcome"] != "PATCHED":
        raise typer.Exit(code=1)


# ─────────────────────  P14 verification & A/B measure  ──────────────────
#
# Both commands start BOTH arms themselves. Reusing an already-running app
# would mean measuring a process whose JVM flags nobody recorded.

def _arm_options(function: Any) -> Any:
    return function


@app.command()
def verify(
    run_dir: Path | None = typer.Option(
        None, "--run-dir", help="Run directory. Defaults to the latest with a patch."),
    repo: Path | None = typer.Option(
        None, "--repo", help="Target repo. Env: PERFAGENT_TARGET_REPO."),
    blue_port: int | None = typer.Option(None, "--blue-port", help="Port for the blue arm. Default: a free port chosen at runtime."),
    green_port: int | None = typer.Option(None, "--green-port", help="Port for the green arm. Default: a free port chosen at runtime."),
    oracle_path: Path = typer.Option(
        Path(".perfagent/oracle.json"), "--oracle", help="The frozen bundle."),
    corpus: Path | None = typer.Option(None, "--corpus", help="Request corpus JSON."),
    java_home: str | None = typer.Option(None, "--java-home", help="JDK for maven."),
    no_build: bool = typer.Option(False, "--no-build", help="Skip the maven build."),
    no_db: bool = typer.Option(False, "--no-db", help="Skip database state capture."),
    all_diffs: bool = typer.Option(
        False, "--all-diffs",
        help="Print every leaf difference, not just the shapes and examples."),
) -> None:
    """Prove the green patch did not change behaviour. Exit 1 if it did."""
    from .measure.arms import both_arms
    from .measure.verify import render_verdict, run_verification
    from .patch.agent import resolve_run_dir
    from .measure.arms import existing_worktree

    try:
        directory = run_dir if run_dir is not None else resolve_run_dir()
        green = existing_worktree(".perfagent/green", repo=repo)
        green_diff = green.source_diff()

        # Green alone is enough to verify against a RECORDED oracle — blue's
        # bytes are already in the bundle. Both arms are started anyway so that
        # verify and measure exercise the identical environment, and so a green
        # arm that only boots when blue does not is caught here.
        with both_arms(
            f"verify-{directory.name}", repo=repo,
            blue_port=blue_port, green_port=green_port,
            java_home=java_home, build=not no_build,
        ) as pair:
            verdict, path = run_verification(
                pair.green.base_url,
                run_dir=directory,
                green_diff=green_diff,
                oracle_path=oracle_path,
                corpus_path=corpus,
                capture_db=not no_db,
            )
    except (ToolError, ContractViolation) as exc:
        typer.secho(f"verify failed: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc

    import json as _json

    document = _json.loads(path.read_text(encoding="utf-8"))
    typer.echo(render_verdict(verdict, document, all_diffs=all_diffs))
    typer.echo("")
    typer.echo(f"written: {path}")

    if not verdict.passed:
        raise typer.Exit(code=1)


@app.command()
def measure(
    run_dir: Path | None = typer.Option(
        None, "--run-dir", help="Run directory. Defaults to the latest with a patch."),
    repo: Path | None = typer.Option(
        None, "--repo", help="Target repo. Env: PERFAGENT_TARGET_REPO."),
    blue_port: int | None = typer.Option(None, "--blue-port", help="Port for the blue arm. Default: a free port chosen at runtime."),
    green_port: int | None = typer.Option(None, "--green-port", help="Port for the green arm. Default: a free port chosen at runtime."),
    warmup: int = typer.Option(10, "--warmup", help="Discarded warmup rounds."),
    iterations: int = typer.Option(10, "--iterations", "-n", help="Measured rounds."),
    corpus: Path | None = typer.Option(None, "--corpus", help="Request corpus JSON."),
    oracle_path: Path = typer.Option(
        Path(".perfagent/oracle.json"), "--oracle", help="The frozen bundle."),
    java_home: str | None = typer.Option(None, "--java-home", help="JDK for maven."),
    no_build: bool = typer.Option(False, "--no-build", help="Skip the maven build."),
    no_sql: bool = typer.Option(
        False, "--no-sql", help="Skip the SQL mechanism check."),
    order_lines: int = typer.Option(
        24, "--order-lines", help="Firing-order lines to print."),
) -> None:
    """Measure blue against green, interleaved. Refuses unless verify passed."""
    import json as _json
    from datetime import datetime, timezone

    from .measure.ab import (build_report, measure_sql_for_arm, render_report,
                             run_interleaved, wilcoxon_p, ArmSql)
    from .measure.arms import both_arms, existing_worktree
    from .measure.corpus import load_corpus
    from .measure.sql import MySQLTarget
    from .measure.verify import require_verified
    from .patch.agent import resolve_run_dir

    try:
        directory = run_dir if run_dir is not None else resolve_run_dir()
        green = existing_worktree(".perfagent/green", repo=repo)

        # The refusal, before anything is built or started. A number for a
        # patch that changed behaviour is worse than no number.
        verification = require_verified(
            directory, green_diff=green.source_diff(),
            oracle_path=oracle_path, corpus_path=corpus,
        )
        request_corpus = load_corpus(corpus)

        with both_arms(
            f"measure-{directory.name}", repo=repo,
            blue_port=blue_port, green_port=green_port,
            java_home=java_home, build=not no_build,
        ) as pair:
            run = run_interleaved(
                pair.blue.base_url, pair.green.base_url, request_corpus,
                warmup=warmup, iterations=iterations,
            )
            if no_sql:
                empty = ArmSql(statements=0, rows_examined=0, digests=[])
                blue_sql, green_sql = empty, empty
            else:
                target = MySQLTarget.from_env()
                blue_sql = measure_sql_for_arm(
                    pair.blue.base_url, request_corpus, target)
                green_sql = measure_sql_for_arm(
                    pair.green.base_url, request_corpus, target)

            report = build_report(run, blue_sql, green_sql)
            argv = pair.argv_record()
    except (ToolError, ContractViolation) as exc:
        typer.secho(f"measure failed: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc

    document = {
        "report": report.model_dump(mode="json"),
        "blue_url": f"http://localhost:{blue_port}",
        "green_url": f"http://localhost:{green_port}",
        "iterations": iterations,
        "warmup_rounds": warmup,
        "wilcoxon_p": wilcoxon_p(run.pairs),
        "verified_against": {
            "oracle_hash": verification["oracle_hash"],
            "corpus_hash": verification["corpus_hash"],
            "patch_identity": verification["patch_identity"],
            "verified_at": verification["verified_at"],
        },
        "arm_argv": argv,
        "jvm_flags_identical": True,
        "firing_order": [
            {"seq": s.seq, "arm": s.arm, "request_id": s.request_id,
             "status": s.status, "latency_ms": round(s.latency_ms, 3)}
            for s in run.samples
        ],
        "pairs": [
            {"iteration": p.iteration, "request_id": p.request_id,
             "blue_ms": round(p.blue_ms, 3), "green_ms": round(p.green_ms, 3),
             "delta_ms": round(p.delta_ms, 3)}
            for p in run.pairs
        ],
        "status_counts": run.status_counts,
        "sql_digests": {
            "blue": [d.model_dump(mode="json") for d in blue_sql.digests[:20]],
            "green": [d.model_dump(mode="json") for d in green_sql.digests[:20]],
        },
        "measured_at": datetime.now(timezone.utc).isoformat(),
    }

    out_path = directory / "perf_report.json"
    out_path.write_text(
        _json.dumps(document, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    typer.echo(render_report(report, run, document, order_lines=order_lines))
    typer.echo("")
    typer.echo(f"written: {out_path}")


# ─────────────────────────────  P13 oracle  ──────────────────────────────
#
# A sub-app rather than two flat commands, because `record` and `check` share a
# subject and must never be confused for each other: one freezes behaviour, the
# other tests against it.

oracle_app = typer.Typer(
    name="oracle",
    help="Freeze and check endpoint behaviour. The oracle is recorded against "
         "UNMODIFIED code and is not in any agent's tool surface.",
    no_args_is_help=True,
)
app.add_typer(oracle_app)


def _render_verdict(
    verdict: Any, oracle: Any, held_back_count: int,
    unmatched_paths: Sequence[str] = (), max_diffs: int = 20,
) -> str:
    """Render the verdict. Every number here comes off the verdict model."""
    # "nothing was checked" and "something differed" are different facts and
    # must not print the same way.
    if verdict.db_state_matched:
        db_state_label = "matched"
    elif all(r.db_state_hash is None for r in oracle.recorded):
        db_state_label = "NOT CHECKED (recorded with --no-db)"
    else:
        db_state_label = "DIFFERS"

    lines = [
        "PASSED — behaviour is equivalent" if verdict.passed
        else "FAILED — behaviour changed",
        f"  decided by     : {verdict.decided_by}",
        f"  oracle hash    : {verdict.oracle_hash}",
        f"  base commit    : {oracle.base_commit}",
        f"  compared       : {verdict.compared}",
        f"  identical      : {verdict.identical}",
        f"  db state       : {db_state_label}",
        f"  held-back set  : {'passed' if verdict.held_back_passed else 'FAILED'}"
        f" ({held_back_count} request(s))",
        f"  external calls : "
        f"{'matched' if verdict.external_calls_matched else 'DIFFER'}"
        f" (no journal recorded until P19)",
        f"  excluded       : "
        f"{', '.join(oracle.excluded_tables) if oracle.excluded_tables else '(none)'}"
        f" (schema-management bookkeeping, sealed in at record time)",
    ]
    if unmatched_paths:
        lines.append(
            "  WARNING: these tolerance paths matched nothing in any response, "
            "so they are not doing what they look like they are doing: "
            + ", ".join(unmatched_paths)
        )
    if held_back_count == 0:
        lines.append(
            "  WARNING: the held-back set is empty, so a patch that special-cases "
            "benchmark inputs would NOT be caught. Populate `held_back` in the "
            "corpus (P06) to arm it."
        )
    if verdict.diffs:
        lines.append(f"  {len(verdict.diffs)} difference(s):")
        for diff in verdict.diffs[:max_diffs]:
            lines.append(f"    {diff.request_id}  {diff.path}")
            lines.append(f"      before: {diff.before!r}")
            lines.append(f"      after : {diff.after!r}")
        if len(verdict.diffs) > max_diffs:
            lines.append(f"    ... {len(verdict.diffs) - max_diffs} more")
    return "\n".join(lines)


@oracle_app.command("record")
def oracle_record(
    url: str | None = typer.Option(None, "--url",
                            help="Base URL of the running, UNMODIFIED app."),
    corpus: Path | None = typer.Option(
        None, "--corpus", help="Request corpus JSON. Defaults to the shipped corpus."),
    oracle_path: Path = typer.Option(
        Path(".perfagent/oracle.json"), "--oracle", help="Where the bundle is written."),
    tolerance: Path | None = typer.Option(
        None, "--tolerance", help="Tolerance policy. Defaults to .perfagent/tolerance.json."),
    repo: Path | None = typer.Option(
        None, "--repo", help="Target repo, for the base commit. Env: PERFAGENT_TARGET_REPO."),
    no_db: bool = typer.Option(
        False, "--no-db", help="Skip database state capture. Weakens the oracle."),
    timeout: float = typer.Option(120.0, "--timeout", help="Per-request timeout (s)."),
    force: bool = typer.Option(
        False, "--force", help="Archive and replace an existing oracle. Think first."),
) -> None:
    """Record behaviour on unmodified code and freeze it."""
    from .measure.corpus import load_corpus
    from .oracle.canonicalise import load_policy
    from .oracle.record import record_oracle

    try:
        request_corpus = load_corpus(corpus)
        policy = load_policy(tolerance)
        from .measure.sql import require_base_url

        bundle = record_oracle(
            require_base_url(url),
            request_corpus,
            policy=policy,
            repo_path=repo,
            capture_db=not no_db,
            timeout_s=timeout,
            path=oracle_path,
            force=force,
        )
    except (ToolError, ContractViolation) as exc:
        typer.secho(f"oracle record failed: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc

    recorded = bundle["recorded"]
    typer.echo(f"recorded {len(recorded)} response(s)")
    typer.echo(f"  corpus hash : {bundle['corpus_hash']}")
    typer.echo(f"  base commit : {bundle['base_commit']}")
    typer.echo(f"  oracle hash : {bundle['hash']}")
    excluded = bundle.get("excluded_tables") or []
    if excluded:
        typer.echo(
            f"  excluded    : {', '.join(excluded)} (schema-management "
            f"bookkeeping, not application state — sealed into this bundle, "
            f"never compared)"
        )
    if "+dirty" in bundle["base_commit"]:
        typer.secho(
            "  WARNING: the target repo has uncommitted changes, so the "
            "recording is stamped '+dirty'. An oracle recorded against "
            "modified code proves nothing.",
            fg=typer.colors.YELLOW,
        )
    if no_db:
        typer.secho(
            "  WARNING: --no-db, so a patch that writes to the database will "
            "not be caught.",
            fg=typer.colors.YELLOW,
        )
    if not request_corpus.held_back:
        typer.secho(
            "  WARNING: the corpus has no held-back requests, so benchmark "
            "gaming will not be caught.",
            fg=typer.colors.YELLOW,
        )
    typer.echo(f"written: {oracle_path}")


@oracle_app.command("check")
def oracle_check(
    url: str | None = typer.Option(None, "--url",
                            help="Base URL of the app under test."),
    corpus: Path | None = typer.Option(
        None, "--corpus", help="Request corpus JSON. Must be the recorded one."),
    oracle_path: Path = typer.Option(
        Path(".perfagent/oracle.json"), "--oracle", help="The frozen bundle."),
    no_db: bool = typer.Option(
        False, "--no-db", help="Skip database state capture."),
    timeout: float = typer.Option(120.0, "--timeout", help="Per-request timeout (s)."),
) -> None:
    """Replay the corpus and compare against the frozen oracle."""
    from .measure.corpus import load_corpus
    from .measure.sql import MySQLTarget
    from .oracle.canonicalise import unmatched_policy_paths
    from .oracle.compare import compare
    from .oracle.record import (db_state_drift, db_table_snapshot, load_oracle,
                                recorded_initial_db_state, replay_live)

    try:
        request_corpus = load_corpus(corpus)
        oracle = load_oracle(oracle_path)

        # Pre-flight: refuse to compare against a database that is not where the
        # recording started. Every per-request hash is cumulative, so a drifted
        # starting state fails the whole check for a reason that is not the patch.
        if not no_db:
            live_snapshot, live_excluded = db_table_snapshot(MySQLTarget.from_env())
            drift = db_state_drift(
                recorded_initial_db_state(oracle_path),
                live_snapshot,
            )
            if drift:
                raise ContractViolation(
                    "the database is not in the state the oracle was recorded "
                    f"against: {'; '.join(drift)}. Re-seed to the recorded "
                    "dataset before checking; comparing bodies computed from "
                    "different data proves nothing."
                )
            if set(live_excluded) != set(oracle.excluded_tables):
                typer.secho(
                    f"  NOTE: this database excludes {sorted(live_excluded) or '(none)'} "
                    f"as schema-management bookkeeping; the oracle recorded "
                    f"excluding {sorted(oracle.excluded_tables) or '(none)'}. "
                    "A comparison is still run, but this is worth a look.",
                    fg=typer.colors.YELLOW,
                )

        from .measure.sql import require_base_url

        live = replay_live(
            require_base_url(url), request_corpus, capture_db=not no_db, timeout_s=timeout
        )
        verdict = compare(oracle, live, request_corpus)
    except NotImplementedError as exc:
        typer.secho(f"oracle check unavailable: {exc}", fg=typer.colors.YELLOW, err=True)
        raise typer.Exit(code=2) from exc
    except (ToolError, ContractViolation) as exc:
        typer.secho(f"oracle check failed: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc

    unmatched = unmatched_policy_paths(
        [r.body for r in oracle.recorded], oracle.policy
    )
    typer.echo(
        _render_verdict(verdict, oracle, len(request_corpus.held_back), unmatched)
    )
    if not verdict.passed:
        raise typer.Exit(code=1)


# ───────────────────────  P15 gate, report, ship  ────────────────────────

@app.command()
def gate(
    run_dir: Path | None = typer.Option(
        None, "--run-dir", help="Run directory. Defaults to the latest with a patch."),
) -> None:
    """Evaluate the six gates. No LLM call, no PR. Exit 1 if the gate failed."""
    from .gate.rules import render_gate_table
    from .patch.agent import resolve_run_dir
    from .ship import run_gate

    try:
        directory = run_dir if run_dir is not None else resolve_run_dir()
        gated = run_gate(directory)
    except (ToolError, ContractViolation) as exc:
        typer.secho(f"gate failed: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc

    typer.echo(render_gate_table(gated["decision"], gated["terminal"]))
    if not gated["decision"].passed:
        raise typer.Exit(code=1)


@app.command()
def ship(
    run_dir: Path | None = typer.Option(
        None, "--run-dir", help="Run directory. Defaults to the latest with a patch."),
    green: Path = typer.Option(
        Path(".perfagent/green"), "--green", help="Green worktree to push from."),
    dry_run: bool = typer.Option(
        False, "--dry-run",
        help="Commit locally but do not push or call gh. Prints what it would run."),
    show_report: bool = typer.Option(
        False, "--show-report", help="Print the rendered report.md as well."),
) -> None:
    """Gate, write the report, and open a PR — or refuse and say why."""
    from .patch.agent import resolve_run_dir
    from .ship import render_ship_summary, run_ship

    try:
        directory = run_dir if run_dir is not None else resolve_run_dir()
        result = run_ship(directory, green_path=green, dry_run=dry_run)
    except (ToolError, ContractViolation) as exc:
        typer.secho(f"ship failed: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc

    typer.echo(render_ship_summary(result))
    typer.echo("")
    typer.echo(f"written: {result['report_path']}")
    typer.echo(f"written: {result['gate_path']}")

    if show_report:
        typer.echo("")
        typer.echo(result["report"])

    if not result["decision"].passed:
        raise typer.Exit(code=1)


# ─────────────────────────  P16 orchestrator  ────────────────────────────

@app.command()
def run(
    repo: Path | None = typer.Option(
        None, "--repo", help="Target repo. Env: PERFAGENT_TARGET_REPO."),
    endpoint: str = typer.Option(
        "POST /api/customers/search", "--endpoint",
        help='The endpoint under test, e.g. "POST /api/customers/search".'),
    resume: str | None = typer.Option(
        None, "--resume", help="Run id to resume from its last completed phase."),
    run_id: str | None = typer.Option(
        None, "--run-id", help="Use this run id instead of a fresh ULID."),
    runs_dir: Path = typer.Option(
        Path(".perfagent/runs"), "--runs-dir", help="Where run directories live."),
    max_usd: float = typer.Option(5.0, "--max-usd", help="Run-level spend ceiling."),
    max_wall: float = typer.Option(
        3600.0, "--max-wall", help="Run-level wall-clock ceiling (seconds)."),
    blue_port: int | None = typer.Option(None, "--blue-port", help="Port for blue. Default: a free port chosen at runtime."),
    green_port: int | None = typer.Option(None, "--green-port", help="Port for green. Default: a free port chosen at runtime."),
    java_home: str | None = typer.Option(None, "--java-home", help="JDK for maven."),
    baseline_iterations: int = typer.Option(
        100, "--baseline-iterations", help="Measured baseline requests."),
    baseline_warmup: int = typer.Option(
        20, "--baseline-warmup", help="Discarded baseline requests."),
    ab_iterations: int = typer.Option(
        8, "--ab-iterations", help="Measured A/B rounds."),
    ab_warmup: int = typer.Option(3, "--ab-warmup", help="Discarded A/B rounds."),
    no_build: bool = typer.Option(
        False, "--no-build", help="Skip maven builds (arms must be current)."),
    open_pr: bool = typer.Option(
        False, "--open-pr",
        help="Actually push and call gh if the gate passes. Default is a dry run."),
) -> None:
    """Run the whole pipeline in order, resumable."""
    from .orchestrator import run_pipeline

    def show(event: dict[str, Any]) -> None:
        kind = event["kind"]
        if kind == "run_start":
            label = "resumed" if event["resumed"] else "started"
            typer.secho(f"run {event['run_id']} {label} · {event['endpoint']}",
                        fg=typer.colors.CYAN)
        elif kind == "phase_start":
            typer.echo(f"  {event['phase']:<11} running   ({event['describe']})")
        elif kind == "phase_skipped":
            typer.secho(f"  {event['phase']:<11} skipped   ({event['reason']})",
                        fg=typer.colors.BRIGHT_BLACK)
        elif kind == "phase_end":
            typer.secho(
                f"  {event['phase']:<11} done      {event['summary']} "
                f"[{event['duration_ms'] / 1000:.1f}s]",
                fg=typer.colors.GREEN,
            )
        elif kind == "phase_failed":
            typer.secho(f"  {event['phase']:<11} FAILED    {event['error']}",
                        fg=typer.colors.RED)
        elif kind == "blocked":
            typer.secho(f"  {event['phase']:<11} BLOCKED   {event['reason']}",
                        fg=typer.colors.YELLOW)
        elif kind == "terminal":
            colour = (typer.colors.GREEN if event["terminal"] == "PR_OPENED"
                      else typer.colors.YELLOW)
            typer.secho(f"terminal state: {event['terminal']}", fg=colour)
            if event.get("reason"):
                typer.echo(f"  {event['reason']}")

    try:
        state = run_pipeline(
            run_id=run_id,
            repo=repo, endpoint=endpoint, resume=resume, runs_dir=runs_dir,
            max_usd=max_usd, max_wall_s=max_wall, blue_port=blue_port,
            green_port=green_port, java_home=java_home,
            options={
                "ab_iterations": ab_iterations, "ab_warmup": ab_warmup,
                "iterations": baseline_iterations, "warmup": baseline_warmup,
                "build": not no_build, "dry_run": not open_pr,
            },
            on_event=show,
        )
    except (ToolError, ContractViolation) as exc:
        typer.secho(f"run failed: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc

    typer.echo("")
    typer.echo(f"run state: {runs_dir / state.run_id / 'run_state.json'}")
    typer.echo(f"narrative: perfagent trace {state.run_id}")

    from .contracts import TerminalState

    if state.terminal != TerminalState.PR_OPENED:
        raise typer.Exit(code=1)


@app.command()
def trace(
    run_id: str | None = typer.Argument(
        None, help="Run id. Defaults to the most recent run."),
    runs_dir: Path = typer.Option(
        Path(".perfagent/runs"), "--runs-dir", help="Where run directories live."),
) -> None:
    """Render a run as the narrative of what happened, plus its cost."""
    from .trace_view import render_trace

    try:
        typer.echo(render_trace(run_id, runs_dir))
    except ContractViolation as exc:
        typer.secho(f"trace failed: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc


if __name__ == "__main__":
    app()
