"""A read-only, cached extraction of mini-spring at one commit.

Shared by live_freeze.py (which must use the SAME path convention it writes
into a scenario's frozen prompt text) and tier1.py (which replays that exact
text later). Using a stable `eval/_checkouts/<commit>/` path for both, rather
than whatever branch happens to be checked out in the repo's own working
copy, is what makes `repo_root: {path}` in the plan prompt hash the same
both times.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

CHECKOUTS_DIR = Path(__file__).resolve().parent / "_checkouts"


class CheckoutError(Exception):
    """A named, specific reason the pinned checkout could not be built."""


def ensure_checkout(repo: Path, tag: str, commit: str) -> Path:
    """Verify `tag` resolves to `commit` in `repo`, then return a cached,
    read-only extraction of the tree at that commit."""
    resolved = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", f"{tag}^{{commit}}"],
        capture_output=True, text=True,
    )
    if resolved.returncode != 0:
        raise CheckoutError(
            f"git tag {tag!r} does not exist in {repo} -- cannot pin this "
            f"scenario's repo state ({resolved.stderr.strip()})"
        )
    actual = resolved.stdout.strip()
    if actual != commit:
        raise CheckoutError(
            f"tag {tag!r} resolves to {actual}, but this scenario was frozen "
            f"against {commit} -- the tag has moved since."
        )

    checkout = CHECKOUTS_DIR / commit
    if not (checkout / "pom.xml").exists():
        checkout.mkdir(parents=True, exist_ok=True)
        git = subprocess.Popen(
            ["git", "-C", str(repo), "archive", commit], stdout=subprocess.PIPE)
        tar = subprocess.Popen(["tar", "-x", "-C", str(checkout)], stdin=git.stdout)
        assert git.stdout is not None
        git.stdout.close()
        tar.communicate()
        git.wait()
        if git.returncode != 0 or tar.returncode != 0:
            raise CheckoutError(f"git archive {commit} | tar -x failed")
    return checkout
