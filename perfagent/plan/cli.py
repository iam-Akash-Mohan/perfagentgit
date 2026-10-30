"""`perfagent plan` — rank by expected value, then propose candidate fixes.

`--rank-only` exists for the same reason `diagnose --evidence-only` and
`patch --prompt-only` do: the half of this phase that matters most is computed
by code, and being able to see it without spending a token is what makes it
reviewable. It prints exactly the ranking the agent would be handed.
"""

from __future__ import annotations

import json
from pathlib import Path

import typer

from ..agent import LoopBudget
from ..errors import ContractViolation, PerfAgentError, ToolError
from ..ids import new_run_id
from ..render.plan import render_plan, render_ranking
from ..render.transcript import attach_transcript
from ..trace import Trace

DEFAULT_RUNS_DIR = Path(".perfagent/runs")


def plan_command(
    run_dir: Path | None = typer.Option(
        None, "--run-dir",
        help="Run directory holding hypotheses.json. Defaults to the newest."),
    endpoint_model: Path | None = typer.Option(
        None, "--endpoint-model",
        help="Comprehension artifact. Defaults to <run-dir>/endpoint_model.json."),
    repo: Path | None = typer.Option(
        None, "--repo", help="Target repo. Env: PERFAGENT_TARGET_REPO."),
    endpoint: str | None = typer.Option(
        None, "--endpoint",
        help="The endpoint. Defaults to the route in the comprehension model."),
    provider: str | None = typer.Option(
        None, "--provider",
        help="cerebras | groq | gemini | anthropic | ollama. Default: perfagent.yml."),
    model: str | None = typer.Option(
        None, "--model", help="Model id. Default: perfagent.yml."),
    max_usd: float | None = typer.Option(
        2.0, "--max-usd", help="Spend ceiling for this phase. 0 means zero."),
    max_steps: int = typer.Option(24, "--max-steps", help="Agent turn budget."),
    max_wall_s: float | None = typer.Option(
        None, "--max-wall", help="Wall-clock ceiling in seconds."),
    offline: bool = typer.Option(
        False, "--offline", help="Replay recorded cassettes only. No network."),
    rank_only: bool = typer.Option(
        False, "--rank-only",
        help="Print the fixability ranking and exit. No LLM call."),
    runs_dir: Path = typer.Option(
        DEFAULT_RUNS_DIR, "--runs-dir", help="Parent of run directories."),
    verbose: bool = typer.Option(
        False, "--verbose", help="Print the conversation as it happens."),
    as_json: bool = typer.Option(
        False, "--json", help="Dump the artifact instead of rendering it."),
) -> None:
    """Rank hypotheses by expected value and plan candidate fixes."""
    from ..contracts import Plan
    from ..llm import make_client
    from ..patch.worktree import target_repo
    from ..plan.agent import load_inputs, plan, resolve_run_dir
    from ..plan.fixability import rank_hypotheses

    try:
        directory = (Path(run_dir) if run_dir is not None
                     else resolve_run_dir(Path(runs_dir)))
        hypotheses, comprehension = load_inputs(directory, endpoint_model)
        route = endpoint or comprehension.route

        if rank_only:
            ranked = rank_hypotheses(hypotheses, comprehension)
            preview = Plan(
                endpoint=route,
                measured_total_ms=hypotheses.measured_total_ms,
                ranked=ranked,
                considered=[r.hypothesis_id for r in ranked[:2]],
                selection_reason="not selected: --rank-only makes no LLM call",
                base_commit=comprehension.base_commit,
            )
            typer.echo(render_ranking(preview))
            typer.echo("")
            typer.secho("--rank-only: no candidates were generated and no "
                        "plan.json was written.", fg=typer.colors.YELLOW)
            return

        run_id = new_run_id()
        trace = Trace(directory, run_id)
        if verbose:
            attach_transcript(trace, typer.echo)

        result = plan(
            repo=target_repo(repo),
            endpoint=route,
            hypotheses=hypotheses,
            model=comprehension,
            client=make_client(provider, model, offline=offline),
            trace=trace,
            budget=LoopBudget(max_steps=max_steps, max_usd=max_usd,
                              max_wall_s=max_wall_s),
            run_dir=directory,
        )
    except (ToolError, ContractViolation) as exc:
        typer.secho(f"plan failed: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc
    except PerfAgentError as exc:
        typer.secho(f"plan failed: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc

    if as_json:
        typer.echo(json.dumps(result.model_dump(mode="json"), indent=2,
                              ensure_ascii=False))
    else:
        typer.echo(render_plan(result))

    typer.echo("")
    typer.echo(f"artifact: {directory / 'plan.json'}")
    typer.echo(f"trace:    {trace.path}")

    if result.selected_id is None:
        typer.secho(
            "no candidate was selectable; the patch phase has nothing to "
            "implement.", fg=typer.colors.YELLOW,
        )
