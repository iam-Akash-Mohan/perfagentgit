"""Prove behaviour is unchanged on the green arm, and record that it was proven.

This is the gate in front of measurement. A performance number for a patch that
changed behaviour is worse than no number: it is a number someone will quote.

The verdict is `compare()` from P13, unchanged and un-widened — the frozen
tolerance policy travels inside the oracle bundle, so nothing here can loosen
it. All this module does is fire the corpus at green, hand the results to
`compare()`, and persist what code decided along with enough identity for
`measure` to check the verdict still applies.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..contracts import EquivalenceVerdict, FieldDiff
from ..errors import ContractViolation
from ..ids import content_hash
from ..measure.corpus import load_corpus
from ..oracle.canonicalise import unmatched_policy_paths
from ..oracle.compare import compare
from ..oracle.record import ORACLE_PATH, load_oracle, replay_live

VERIFICATION_FILE = "verification.json"

# One changed string in a per-row loop produced 111,407 leaf diffs across 10
# requests — all of a single shape, `$.content[*].tags[*]` — and a 17 MB
# artifact. Neither a human nor the patch agent can use that. What both need is
# the SHAPE and one example; the raw leaves add nothing after the first few.
MAX_PRINTED_EXAMPLES = 6
MAX_PERSISTED_DIFFS = 200
_INDEX = re.compile(r"\[\d+\]")


def path_shape(path: str) -> str:
    """`$.content[3].tags[7]` -> `$.content[*].tags[*]`."""
    return _INDEX.sub("[*]", path)


def summarise_diffs(diffs: list[FieldDiff]) -> list[dict[str, Any]]:
    """Collapse leaf diffs to distinct path shapes, commonest first.

    This is the actionable form: "one shape, 111407 occurrences, 'overdue' ->
    'OVERDUE'" tells you what to fix; 111407 individual paths do not, and will
    not fit in a model's context either.
    """
    grouped: dict[str, dict[str, Any]] = {}
    for diff in diffs:
        shape = path_shape(diff.path)
        entry = grouped.setdefault(
            shape,
            {
                "shape": shape,
                "occurrences": 0,
                "requests": [],
                "example_path": diff.path,
                "example_before": diff.before,
                "example_after": diff.after,
            },
        )
        entry["occurrences"] += 1
        if diff.request_id not in entry["requests"]:
            entry["requests"].append(diff.request_id)
    return sorted(grouped.values(), key=lambda e: -e["occurrences"])


def patch_identity(green_diff: str) -> str:
    """Content hash of the patch under test.

    `measure` re-computes this. Verifying, then editing green, then measuring
    would otherwise attribute a fresh patch's numbers to an old patch's
    verdict.
    """
    return content_hash({"diff": green_diff})


def run_verification(
    green_url: str,
    *,
    run_dir: Path,
    green_diff: str,
    oracle_path: Path = ORACLE_PATH,
    corpus_path: Path | None = None,
    capture_db: bool = True,
    timeout_s: float = 300.0,
) -> tuple[EquivalenceVerdict, Path]:
    """Replay the FULL corpus — benchmark and held-back — against green.

    `replay_live` fires `benchmark + held_back` in recorded order, which is what
    makes the held-back arm of the verdict meaningful: those requests were never
    used for measurement, so a patch tuned to the measured set has never seen
    them.
    """
    corpus = load_corpus(corpus_path)
    oracle = load_oracle(oracle_path)

    live = replay_live(green_url, corpus, capture_db=capture_db, timeout_s=timeout_s)
    verdict = compare(oracle, live, corpus)

    summary = summarise_diffs(verdict.diffs)
    capped = verdict.model_dump(mode="json")
    capped["diffs"] = capped["diffs"][:MAX_PERSISTED_DIFFS]

    document: dict[str, Any] = {
        "verdict": capped,
        "diffs_total": len(verdict.diffs),
        "diffs_persisted": len(capped["diffs"]),
        "diffs_truncated": len(verdict.diffs) > MAX_PERSISTED_DIFFS,
        "diff_shapes": summary,
        "green_url": green_url,
        "oracle_hash": oracle.hash,
        "oracle_base_commit": oracle.base_commit,
        "corpus_hash": corpus.hash,
        "benchmark_requests": [s.id for s in corpus.benchmark],
        "held_back_requests": [s.id for s in corpus.held_back],
        "patch_identity": patch_identity(green_diff),
        "unmatched_policy_paths": unmatched_policy_paths(
            [r.body for r in oracle.recorded], oracle.policy
        ),
        # Reported, not silent — an exclusion nobody can see is indistinguishable
        # from a bug. Sealed inside the oracle at record time; carried here
        # verbatim so the CLI, the trace and the PR body can all quote it.
        "excluded_tables": oracle.excluded_tables,
        "verified_at": datetime.now(timezone.utc).isoformat(),
    }

    directory = Path(run_dir)
    directory.mkdir(parents=True, exist_ok=True)
    out_path = directory / VERIFICATION_FILE
    out_path.write_text(
        json.dumps(document, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return verdict, out_path


def load_verification(run_dir: Path) -> dict[str, Any]:
    path = Path(run_dir) / VERIFICATION_FILE
    if not path.exists():
        raise ContractViolation(
            f"no verification at {path}. Run `perfagent verify` first: a "
            "performance number for a patch whose behaviour was never checked "
            "is worse than no number."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def require_verified(
    run_dir: Path,
    *,
    green_diff: str,
    oracle_path: Path = ORACLE_PATH,
    corpus_path: Path | None = None,
) -> dict[str, Any]:
    """Refuse to proceed unless a PASSING verdict still applies to this patch.

    Four separate ways a stale verdict could be waved through, all checked:
    the verdict itself failed; a different oracle; a different corpus; or the
    patch changed after it was verified.
    """
    document = load_verification(run_dir)
    verdict = EquivalenceVerdict.model_validate(document["verdict"])

    if not verdict.passed:
        # `verdict.diffs` on a loaded document is the CAPPED list, so the count
        # has to come from the recorded total or the message understates it.
        total = document.get("diffs_total", len(verdict.diffs))
        shapes = document.get("diff_shapes") or []
        shape_note = (
            " Distinct path shape(s): "
            + ", ".join(entry["shape"] for entry in shapes[:5])
            if shapes else ""
        )
        raise ContractViolation(
            f"verification FAILED for this patch ({total} differing leaf path(s) "
            f"across {len(shapes)} shape(s)).{shape_note} Measurement is refused: "
            "a performance number for a patch that changed behaviour is worse "
            "than no number. Run `perfagent verify` for detail, and fix the "
            "patch before measuring it."
        )

    oracle = load_oracle(oracle_path)
    if document.get("oracle_hash") != oracle.hash:
        raise ContractViolation(
            f"the verification was made against oracle {document.get('oracle_hash')} "
            f"but the oracle on disk is {oracle.hash}. Re-run `perfagent verify`."
        )

    corpus = load_corpus(corpus_path)
    if document.get("corpus_hash") != corpus.hash:
        raise ContractViolation(
            f"the verification used corpus {document.get('corpus_hash')} but the "
            f"corpus on disk is {corpus.hash}. Re-run `perfagent verify`."
        )

    current = patch_identity(green_diff)
    if document.get("patch_identity") != current:
        raise ContractViolation(
            "the green worktree has changed since it was verified "
            f"({document.get('patch_identity')} -> {current}). The passing "
            "verdict describes a different patch. Re-run `perfagent verify`."
        )
    return document


def render_verdict(
    verdict: EquivalenceVerdict,
    document: dict[str, Any],
    *,
    all_diffs: bool = False,
) -> str:
    """On failure this output is what a retry is built from, so it leads with
    the distinct path SHAPES and then shows a few concrete examples."""
    out: list[str] = [""]
    out.append("PASSED — behaviour is unchanged on the green arm" if verdict.passed
               else "FAILED — the patch changed behaviour")
    out.append(f"  decided by     : {verdict.decided_by}")
    out.append(f"  green arm      : {document['green_url']}")
    out.append(f"  oracle hash    : {verdict.oracle_hash}")
    out.append(f"  oracle recorded: {document['oracle_base_commit']}")
    out.append(f"  patch identity : {document['patch_identity']}")
    out.append(f"  compared       : {verdict.compared}  "
               f"identical: {verdict.identical}")
    out.append(f"  benchmark set  : {len(document['benchmark_requests'])} request(s)")
    out.append(f"  held-back set  : {len(document['held_back_requests'])} request(s) — "
               f"{'passed' if verdict.held_back_passed else 'FAILED'}")
    out.append(f"  db state       : "
               f"{'matched' if verdict.db_state_matched else 'DIFFERS'}")
    out.append(f"  external calls : "
               f"{'matched' if verdict.external_calls_matched else 'DIFFER'}")
    excluded = document.get("excluded_tables") or []
    out.append(f"  excluded       : "
               f"{', '.join(excluded) if excluded else '(none)'} "
               f"(schema-management bookkeeping, sealed in at record time)")
    if document.get("unmatched_policy_paths"):
        out.append("  WARNING: tolerance paths matching nothing: "
                   + ", ".join(document["unmatched_policy_paths"]))
    if not document["held_back_requests"]:
        out.append("  WARNING: the held-back set is empty, so benchmark gaming "
                   "would not be caught.")

    shapes = document.get("diff_shapes") or summarise_diffs(verdict.diffs)
    total = document.get("diffs_total", len(verdict.diffs))
    if not total:
        return "\n".join(out)

    out.append("")
    out.append(f"WHAT CHANGED — {len(shapes)} distinct path shape(s), "
               f"{total} leaf difference(s)")
    for entry in shapes:
        requests = entry["requests"]
        shown = ", ".join(requests[:4]) + (
            f" (+{len(requests) - 4} more)" if len(requests) > 4 else ""
        )
        out.append(f"  {entry['shape']}")
        out.append(f"    {entry['occurrences']} occurrence(s) across "
                   f"{len(requests)} request(s): {shown}")
        out.append(f"    e.g. {entry['example_path']}")
        out.append(f"         before {entry['example_before']!r}")
        out.append(f"         after  {entry['example_after']!r}")

    listed = verdict.diffs if all_diffs else verdict.diffs[:MAX_PRINTED_EXAMPLES]
    if listed:
        out.append("")
        out.append(f"CONCRETE DIFFERENCES ({len(listed)} of {total})")
        for diff in listed:
            out.append(f"  {diff.request_id}  {diff.path}")
            out.append(f"    before {diff.before!r}")
            out.append(f"    after  {diff.after!r}")
    if not all_diffs and total > len(listed):
        out.append(f"  ... {total - len(listed)} further leaf difference(s) not "
                   f"printed. Re-run with --all-diffs to list them; the first "
                   f"{document.get('diffs_persisted', 0)} are in "
                   f"{VERIFICATION_FILE}.")
    return "\n".join(out)
