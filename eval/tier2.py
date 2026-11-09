"""Tier 2: full pipeline. Slow, needs Docker and a seeded database.

Runs the real pipeline end to end -- understand, baseline, diagnose, plan,
patch, measure, gate, ship --dry-run -- against mini-spring pinned at the
INV-04 scenario's own git tag. This is the one thing Tier 1 cannot touch:
Tier 1 replays recorded reasoning against frozen facts; this exercises the
actual worktree creation, the actual Maven builds of both arms, the actual
frozen-oracle comparison, the actual interleaved A/B measurement, and the
actual six gates -- against a live app and a live, seeded MySQL.

INV-04 ("GET /api/customers/{id}/invoices") was chosen deliberately as the
smallest of the three scenarios: no customer_payment/payment_allocation
seed is needed, just customer+invoice, and --seed-customers/--seed-invoices
below default to a size small enough to run in minutes rather than the
~12 seconds-at-1.5M-rows production seed shape the frozen scenarios used.

What this tier catches that Tier 1 structurally cannot:
  - the blue-arm-reused-instead-of-rebuilt class of bug (worktree hygiene)
  - a green worktree left behind, or not cleaned up, after a run
  - a patch that builds but a migration Flyway refuses to apply
  - an oracle comparison that passes locally but fails against a byte-exact
    frozen recording (timezone/locale/serialization drift)
  - anything timing-dependent in the real A/B measurement (gate thresholds,
    paired-test behavior under real noise)

What it does NOT re-check: the reasoning tested by Tier 1. A passing Tier 2
run with a stale understanding of "the right diagnosis" would still pass --
that drift is Tier 1's job, which is why both tiers exist and neither
replaces the other.

This script does not execute by default when Tier 1 runs. Treat it as a
release gate: run it by hand (or in a pre-release CI job) before shipping,
not on every commit.

Usage:
    .venv/bin/python eval/tier2.py
    .venv/bin/python eval/tier2.py --open-pr      # actually push + gh pr (don't, casually)
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from _checkout import CheckoutError, ensure_checkout  # noqa: E402

SCENARIO_ID = "INV-04"
MINI_SPRING_REPO = Path("/Users/akashm/Documents/mini-spring")
SEED_CUSTOMERS_DEFAULT = 2000
SEED_INVOICES_DEFAULT = 20000


class PreconditionFailed(Exception):
    """A named, specific reason Tier 2 cannot even start."""


def check_docker() -> None:
    result = subprocess.run(["docker", "info"], capture_output=True, text=True)
    if result.returncode != 0:
        raise PreconditionFailed(
            "docker is not reachable (`docker info` failed) -- Tier 2 needs "
            "a running MySQL container. `docker compose up -d` in the "
            "mini-spring checkout, then retry."
        )


def check_tag() -> Path:
    meta = json.loads(
        (Path(__file__).resolve().parent / "scenarios" / SCENARIO_ID
         / "meta.json").read_text())
    try:
        return ensure_checkout(
            MINI_SPRING_REPO, meta["mini_spring_tag"], meta["mini_spring_commit"])
    except CheckoutError as exc:
        raise PreconditionFailed(str(exc)) from exc


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--open-pr", action="store_true",
                        help="Actually push and call gh if the gate passes. "
                             "Default is a dry run, same as `senapathy run`.")
    parser.add_argument("--seed-customers", type=int, default=SEED_CUSTOMERS_DEFAULT)
    parser.add_argument("--seed-invoices", type=int, default=SEED_INVOICES_DEFAULT)
    parser.add_argument("--run-id", default=None)
    args = parser.parse_args()

    try:
        check_docker()
        checkout = check_tag()
    except PreconditionFailed as exc:
        print(f"TIER 2 CANNOT START: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    print(f"mini-spring pinned at {checkout} (tag eval/{SCENARIO_ID})")
    print(f"seeding {args.seed_customers} customers / {args.seed_invoices} "
         f"invoices -- smaller than the frozen scenario's own 50k/1.2M, "
         f"small enough to run this on every release without it becoming "
         f"the thing nobody runs")
    seed = subprocess.run(
        [sys.executable, "seed/seed.py",
         "--customers", str(args.seed_customers),
         "--invoices", str(args.seed_invoices)],
        cwd=checkout,
    )
    if seed.returncode != 0:
        raise SystemExit(f"seeding failed (exit {seed.returncode})")

    from senapathy.orchestrator import run_pipeline

    def show(event: dict) -> None:
        kind = event["kind"]
        if kind == "phase_end":
            print(f"  {event['phase']:<11} done   {event['summary']} "
                 f"[{event['duration_ms'] / 1000:.1f}s]")
        elif kind == "phase_failed":
            print(f"  {event['phase']:<11} FAILED {event['error']}")
        elif kind == "terminal":
            print(f"terminal state: {event['terminal']}")

    state = run_pipeline(
        run_id=args.run_id,
        repo=checkout,
        endpoint="GET /api/customers/{id}/invoices",
        options={"dry_run": not args.open_pr},
        on_event=show,
    )

    from senapathy.contracts import TerminalState

    print(f"\nrun state: .senapathy/runs/{state.run_id}/run_state.json")
    if state.terminal not in (TerminalState.READY_TO_SHIP, TerminalState.PR_OPENED):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
