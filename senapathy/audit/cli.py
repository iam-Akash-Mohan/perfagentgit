"""`senapathy audit` — the SQL audit sweep as a standalone command.

Reads a run directory's own `baseline.json` and `endpoint_model.json`
(already on disk from an earlier `senapathy baseline` / `senapathy
understand`), audits every application digest, and writes `sql_audit.json`
next to them — no live app, no orchestrator run required.
"""

from __future__ import annotations

import json
from pathlib import Path

import typer

from ..contracts import SqlDigestStat
from ..errors import ContractViolation, SenapathyError, ToolError
from ..measure.baseline import baseline_hash
from ..measure.corpus import load_corpus
from ..measure.sql import MySQLTarget
from ..render.audit import render_audit
from ..understand.agent import load_model
from .run import run_sql_audit


def audit_command(
    run_dir: Path = typer.Option(
        ..., "--run-dir",
        help="Run directory containing baseline.json and endpoint_model.json."),
    endpoint: str | None = typer.Option(
        None, "--endpoint",
        help="Endpoint label for the artifact. Default: the endpoint model's own route."),
    as_json: bool = typer.Option(
        False, "--json", help="Dump the artifact instead of rendering it."),
) -> None:
    """Audit every SQL digest a baseline captured."""
    try:
        model = load_model(run_dir / "endpoint_model.json")
        document = json.loads((run_dir / "baseline.json").read_text(encoding="utf-8"))
        digests = [SqlDigestStat.model_validate(d) for d in document["sql_digests"]]
        corpus_path = run_dir / "corpus.json"
        corpus = load_corpus(corpus_path) if corpus_path.exists() else None
        audit = run_sql_audit(
            endpoint=endpoint or model.route,
            measured_total_ms=document["derived"]["measured_latency_ms"],
            digests=digests, target=MySQLTarget.from_env(),
            allowed_tables=[t.table for t in model.tables],
            run_dir=run_dir, baseline_hash=baseline_hash(document),
            corpus=corpus,
        )
    except (ToolError, ContractViolation) as exc:
        typer.secho(f"audit failed: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc
    except SenapathyError as exc:
        typer.secho(f"audit failed: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc

    if as_json:
        typer.echo(json.dumps(audit.model_dump(mode="json"), indent=2,
                              ensure_ascii=False))
    else:
        typer.echo(render_audit(audit))

    typer.echo("")
    typer.echo(f"artifact: {run_dir / 'sql_audit.json'}")
