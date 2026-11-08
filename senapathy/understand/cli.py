"""`senapathy understand` — the comprehension phase as a command.

Registered in `senapathy/cli.py` as an imported function rather than defined
inline. Every command moves to this pattern eventually; a 700-line cli.py that
imports the whole product to render `--help` is the thing being unwound.
"""

from __future__ import annotations

import json
from pathlib import Path

import typer

from ..agent import LoopBudget
from ..errors import ContractViolation, SenapathyError, ToolError
from ..ids import new_run_id
from ..render.endpoint import render_endpoint_model
from ..render.transcript import attach_transcript
from ..trace import Trace

DEFAULT_RUNS_DIR = Path(".senapathy/runs")


def understand_command(
    repo: Path = typer.Option(..., "--repo",
                              help="Target repository to trace."),
    endpoint: str = typer.Option(
        ..., "--endpoint",
        help='The endpoint, e.g. "POST /api/customers/search".'),
    provider: str | None = typer.Option(
        None, "--provider",
        help="cerebras | groq | gemini | anthropic | ollama. Default: senapathy.yml."),
    model: str | None = typer.Option(
        None, "--model", help="Model id. Default: senapathy.yml."),
    max_usd: float | None = typer.Option(
        2.0, "--max-usd", help="Spend ceiling for this phase. 0 means zero."),
    max_steps: int = typer.Option(120, "--max-steps", help="Agent turn budget."),
    max_wall_s: float | None = typer.Option(
        None, "--max-wall", help="Wall-clock ceiling in seconds."),
    offline: bool = typer.Option(
        False, "--offline",
        help="Replay recorded cassettes only. No network."),
    run_dir: Path | None = typer.Option(
        None, "--run-dir", help="Where the artifact and trace are written."),
    runs_dir: Path = typer.Option(
        DEFAULT_RUNS_DIR, "--runs-dir", help="Parent of run directories."),
    verbose: bool = typer.Option(
        False, "--verbose", help="Print the conversation as it happens."),
    as_json: bool = typer.Option(
        False, "--json", help="Dump the artifact instead of rendering it."),
) -> None:
    """Build the endpoint comprehension model."""
    from ..llm import make_client
    from ..understand.agent import understand

    run_id = new_run_id()
    directory = Path(run_dir) if run_dir is not None else Path(runs_dir) / run_id
    trace = Trace(directory, run_id)
    if verbose:
        attach_transcript(trace, typer.echo)

    try:
        client = make_client(provider, model, offline=offline)
        result = understand(
            repo=Path(repo),
            endpoint=endpoint,
            client=client,
            trace=trace,
            budget=LoopBudget(max_steps=max_steps, max_usd=max_usd,
                              max_wall_s=max_wall_s, step_flag="--max-steps"),
            run_dir=directory,
        )
    except (ToolError, ContractViolation) as exc:
        typer.secho(f"understand failed: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc
    except SenapathyError as exc:
        typer.secho(f"understand failed: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc

    if as_json:
        typer.echo(json.dumps(result.model_dump(mode="json"), indent=2,
                              ensure_ascii=False))
    else:
        typer.echo("")
        typer.echo(render_endpoint_model(result))

    typer.echo("")
    typer.echo(f"artifact: {directory / 'endpoint_model.json'}")
    typer.echo(f"trace:    {trace.path}")
