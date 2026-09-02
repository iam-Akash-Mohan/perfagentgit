from __future__ import annotations

import typer

app = typer.Typer(
    name="perfagent",
    help="Autonomous performance-bottleneck agent for Spring Boot endpoints.",
    no_args_is_help=True,
)


@app.command()
def understand() -> None:
    """Build the endpoint comprehension model."""
    raise NotImplementedError("phase P04")


@app.command()
def baseline() -> None:
    """Measure the blue-arm baseline and collect evidence."""
    raise NotImplementedError("phase P08")


@app.command()
def diagnose() -> None:
    """Generate and rank competing bottleneck hypotheses."""
    raise NotImplementedError("phase P09")


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
