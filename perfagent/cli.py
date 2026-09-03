from __future__ import annotations

from pathlib import Path

import typer

from .errors import ContractViolation, ToolError
from .measure.baseline import baseline_command

app = typer.Typer(
    name="perfagent",
    help="Autonomous performance-bottleneck agent for Spring Boot endpoints.",
    no_args_is_help=True,
)


@app.command()
def understand() -> None:
    """Build the endpoint comprehension model."""
    raise NotImplementedError("phase P04")


app.command(name="baseline")(baseline_command)


@app.command()
def diagnose(
    url: str = typer.Option("http://localhost:18080", "--url",
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
def patch() -> None:
    """Implement the selected candidate plan on the green worktree."""
    raise NotImplementedError("phase P12")


@app.command()
def verify() -> None:
    """Check behaviour equivalence against the frozen oracle."""
    raise NotImplementedError("phase P13")


@app.command()
def run() -> None:
    """Run the full pipeline end to end."""
    raise NotImplementedError("phase P16")


@app.command()
def trace() -> None:
    """Render the event trace for a run."""
    raise NotImplementedError("phase P17")


if __name__ == "__main__":
    app()
