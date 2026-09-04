"""The equivalence verdict. `passed` is computed here, in Python, or nowhere.

INVARIANT 6 (CLAUDE.md): behaviour equivalence is decided by code. There is no
argument, prompt, tool or config that lets a model set `passed`, and
`EquivalenceVerdict.decided_by` is a `Literal["code"]` so the contract itself
refuses any other answer.

`resolve_policy`, `verify_corpus` and `align` are the integrity guards;
`diff_bodies` and `compare` are the comparison itself. Nothing here reads the
recording except through the `oracle` argument, and nothing can widen the
frozen tolerance policy.
"""

from __future__ import annotations

from typing import Any

from ..contracts import (BehaviourOracle, EquivalenceVerdict, FieldDiff,
                         RecordedResponse, RequestCorpus, TolerancePolicy)
from ..errors import ContractViolation
from .canonicalise import canonicalise

# Sentinel for "this key was not present on this side". Distinct from None,
# because a field explicitly set to null and a field that is absent are
# different behaviours and must produce different diffs.
MISSING = "<absent>"


def resolve_policy(
    oracle: BehaviourOracle, policy: TolerancePolicy | None = None
) -> TolerancePolicy:
    """Return the policy frozen inside the oracle.

    The `policy` argument exists ONLY so a caller can state what it believes
    the policy to be and be told if it is wrong. It cannot widen anything: a
    policy that differs from the recorded one is a ContractViolation, not an
    override. The frozen policy is inside the content-hashed bundle, so
    "frozen" is checkable rather than aspirational.

    This is the hole the phase spec had: a comparison whose tolerance comes
    from the caller is a comparison the caller can win.
    """
    if policy is None:
        return oracle.policy
    if policy.model_dump(mode="json") != oracle.policy.model_dump(mode="json"):
        raise ContractViolation(
            "the tolerance policy passed to compare() differs from the one "
            "frozen in the oracle. The recorded policy is the only policy. "
            "If a tolerance genuinely needs to change, edit "
            ".perfagent/tolerance.json and re-record against unmodified code — "
            "changing it now would be rationalising a diff, not verifying one."
        )
    return oracle.policy


def verify_corpus(oracle: BehaviourOracle, corpus: RequestCorpus) -> None:
    """Refuse to compare against a corpus the oracle was not recorded from."""
    if corpus.hash != oracle.corpus_hash:
        raise ContractViolation(
            f"corpus hash {corpus.hash} does not match the oracle's recorded "
            f"corpus hash {oracle.corpus_hash}. The request set changed since "
            "the oracle was recorded, so the two are not comparable."
        )


def align(
    oracle: BehaviourOracle, live: list[RecordedResponse]
) -> tuple[list[tuple[RecordedResponse, RecordedResponse]], list[str], list[str]]:
    """Pair recorded and live responses by request id.

    Returns `(pairs, missing, unexpected)` in recorded order. `missing` is a
    recorded request the replay did not produce; `unexpected` is a replay
    response with no recording. Both are failures — a request that vanished is
    not a request that passed.

    Bookkeeping in code, per invariant 5: nothing downstream is ever asked to
    count or match these up.
    """
    by_id: dict[str, RecordedResponse] = {}
    for response in live:
        if response.request_id in by_id:
            raise ContractViolation(
                f"replay produced two responses for request {response.request_id}"
            )
        by_id[response.request_id] = response

    pairs: list[tuple[RecordedResponse, RecordedResponse]] = []
    missing: list[str] = []
    for recorded in oracle.recorded:
        found = by_id.pop(recorded.request_id, None)
        if found is None:
            missing.append(recorded.request_id)
        else:
            pairs.append((recorded, found))
    return pairs, missing, sorted(by_id)


def _kind(value: Any) -> str:
    """The over-the-wire type class of a JSON value.

    `bool` before `int` because bool subclasses int, and `int` apart from
    `float` because `1` and `1.0` are equal in Python but are a Long and a
    Double on the wire. Both distinctions are behaviour changes that plain `==`
    would swallow.
    """
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "list"
    if isinstance(value, dict):
        return "object"
    return type(value).__name__


def _diff(
    request_id: str,
    before: Any,
    after: Any,
    policy: TolerancePolicy,
    path: str,
    out: list[FieldDiff],
) -> None:
    before_kind, after_kind = _kind(before), _kind(after)
    if before_kind != after_kind:
        # A type change is decisive: descending further would report every leaf
        # of a subtree that simply became a different shape.
        out.append(FieldDiff(request_id=request_id, path=path, before=before, after=after))
        return

    if before_kind == "object":
        for key in sorted(set(before) | set(after)):
            in_before, in_after = key in before, key in after
            if in_before and in_after:
                _diff(request_id, before[key], after[key], policy, f"{path}.{key}", out)
            else:
                # Absence by key membership, never by comparing to MISSING —
                # a body whose value literally IS the sentinel string must not
                # read as absent.
                out.append(
                    FieldDiff(
                        request_id=request_id,
                        path=f"{path}.{key}",
                        before=before[key] if in_before else MISSING,
                        after=after[key] if in_after else MISSING,
                    )
                )
        return

    if before_kind == "list":
        if len(before) != len(after):
            # One diff at the list itself. Recursing would report every element
            # from the first insertion onwards as changed, which is true and
            # useless.
            out.append(
                FieldDiff(
                    request_id=request_id,
                    path=path,
                    before=f"<list of {len(before)}>",
                    after=f"<list of {len(after)}>",
                )
            )
            return
        for index, (b, a) in enumerate(zip(before, after)):
            _diff(request_id, b, a, policy, f"{path}[{index}]", out)
        return

    if before_kind == "float" and policy.numeric_epsilon > 0:
        # Epsilon applies to floats only. An int change is a row count, a page
        # size or an id, and rounding those is never a tolerance.
        if abs(before - after) <= policy.numeric_epsilon:
            return

    if before != after:
        out.append(FieldDiff(request_id=request_id, path=path, before=before, after=after))


def diff_bodies(
    request_id: str, before: Any, after: Any, policy: TolerancePolicy, path: str = "$"
) -> list[FieldDiff]:
    """Deep-compare two ALREADY-CANONICALISED bodies, reporting exact paths.

    Inputs must already be through `canonicalise()`; canonicalising here as well
    would process ignored paths twice and drift the `[*]` wildcards.

    Returns `[]` for identical. The caller counts — this function never reports
    a total, per invariant 5.
    """
    diffs: list[FieldDiff] = []
    _diff(request_id, before, after, policy, path, diffs)
    return diffs


def _header_diffs(
    request_id: str, before: dict[str, str], after: dict[str, str]
) -> list[FieldDiff]:
    """Compare the behavioural header allowlist.

    A response that stopped being `application/json` changed behaviour even if
    the bytes happen to parse the same, and a vanished `Location` breaks every
    client that followed it.
    """
    diffs: list[FieldDiff] = []
    for name in sorted(set(before) | set(after)):
        was, now = before.get(name, MISSING), after.get(name, MISSING)
        if was != now:
            diffs.append(
                FieldDiff(request_id=request_id, path=f"$headers.{name}", before=was, after=now)
            )
    return diffs


def compare(
    oracle: BehaviourOracle,
    live: list[RecordedResponse],
    corpus: RequestCorpus,
    policy: TolerancePolicy | None = None,
) -> EquivalenceVerdict:
    """Decide whether behaviour changed. `passed` is computed at the end, here.

    The signature deviates from the phase spec in two ways, both deliberate:
    `corpus` is required, because it is the only way to know which request ids
    are held back without inventing a field on `RecordedResponse`; and `policy`
    is verify-only through `resolve_policy`, because a comparison whose
    tolerance comes from the caller is a comparison the caller can win.

    Non-body differences use pseudo-paths — `$status`, `$headers.<name>`,
    `$db_state`, `$external_calls` — with no dot after `$`, so they are
    distinguishable at a glance from real body paths like `$.content[3].total`.
    """
    # Integrity before comparison. Comparing against the wrong corpus, or with a
    # swapped policy, produces a confident and meaningless verdict.
    verify_corpus(oracle, corpus)
    effective_policy = resolve_policy(oracle, policy)

    pairs, missing, unexpected = align(oracle, live)

    diffs: list[FieldDiff] = []
    identical = 0
    db_state_compared = False
    db_state_matched = True
    external_calls_matched = True

    for recorded, replayed in pairs:
        request_id = recorded.request_id
        request_diffs: list[FieldDiff] = []

        if recorded.status != replayed.status:
            # Always a failure, no tolerance, and the body is not compared: a
            # 400 that became a 500 with an identical body still changed
            # behaviour, and the bodies of two different error paths tell you
            # nothing useful.
            request_diffs.append(
                FieldDiff(
                    request_id=request_id,
                    path="$status",
                    before=recorded.status,
                    after=replayed.status,
                )
            )
        else:
            request_diffs.extend(
                _header_diffs(
                    request_id, recorded.headers_of_interest, replayed.headers_of_interest
                )
            )
            request_diffs.extend(
                diff_bodies(
                    request_id,
                    canonicalise(recorded.body, effective_policy),
                    canonicalise(replayed.body, effective_policy),
                    effective_policy,
                )
            )

        if recorded.db_state_hash is not None or replayed.db_state_hash is not None:
            db_state_compared = True
            if recorded.db_state_hash != replayed.db_state_hash:
                db_state_matched = False
                # Note these accumulate: the hashes are cumulative, so one stray
                # write makes every later request diverge too. The first one is
                # the lead; the rest are its shadow.
                request_diffs.append(
                    FieldDiff(
                        request_id=request_id,
                        path="$db_state",
                        before=recorded.db_state_hash,
                        after=replayed.db_state_hash,
                    )
                )

        # Compared for real rather than stubbed, so this arms itself the moment
        # P19 starts populating the WireMock journal. Both sides are empty by
        # construction today, so it passes trivially and silently.
        if recorded.external_calls_made != replayed.external_calls_made:
            external_calls_matched = False
            request_diffs.append(
                FieldDiff(
                    request_id=request_id,
                    path="$external_calls",
                    before=recorded.external_calls_made,
                    after=replayed.external_calls_made,
                )
            )

        if not request_diffs:
            identical += 1
        diffs.extend(request_diffs)

    # A request that vanished is not a request that passed.
    for request_id in missing:
        diffs.append(
            FieldDiff(
                request_id=request_id,
                path="$",
                before="<recorded>",
                after=MISSING,
            )
        )
    for request_id in unexpected:
        diffs.append(
            FieldDiff(
                request_id=request_id,
                path="$",
                before=MISSING,
                after="<replayed>",
            )
        )

    if not db_state_compared:
        # Recorded with --no-db: nothing was checked. Reporting True here would
        # be a clean bill of health for a check that never looked, so --no-db
        # deliberately cannot produce a passing verdict.
        db_state_matched = False

    held_back_ids = {spec.id for spec in corpus.held_back}
    compared_ids = {recorded.request_id for recorded, _ in pairs}
    failed_ids = {diff.request_id for diff in diffs}
    held_back_passed = (
        held_back_ids <= compared_ids  # every held-back request actually ran
        and not (held_back_ids & failed_ids)
    )
    # An empty held-back set passes trivially. That is honest — there is nothing
    # to fail — but it means the anti-gaming net is NOT armed, which is why the
    # CLI says so out loud whenever the set is empty.

    passed = (
        not diffs
        and not missing
        and not unexpected
        and db_state_matched
        and held_back_passed
        and external_calls_matched
    )

    return EquivalenceVerdict(
        passed=passed,
        compared=len(pairs),
        identical=identical,
        diffs=diffs,
        db_state_matched=db_state_matched,
        external_calls_matched=external_calls_matched,
        held_back_passed=held_back_passed,
        oracle_hash=oracle.hash,
    )
