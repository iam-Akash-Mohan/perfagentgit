"""Tolerance policy loading (plumbing) and body canonicalisation (your part).

Canonicalisation is a *structural* transform: it drops fields the frozen policy
says to ignore and sorts arrays the policy says are order-insensitive, so that
two bodies which are behaviourally identical compare equal with `==`.

`numeric_epsilon` is deliberately NOT applied here. It is a leaf-comparison
rule and belongs in `compare.py`. Rounding during canonicalisation is a trap:
with an epsilon of 0.001, 1.0049 and 1.0051 round to different values yet are
within tolerance of each other.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable

from ..contracts import TolerancePolicy
from ..errors import ContractViolation

DEFAULT_TOLERANCE_PATH = Path(".perfagent/tolerance.json")

# The supported path subset. Deliberately small: `$` root, dotted object keys,
# `[*]` for "every element of this array", `[3]` for one index.
#
#   $.meta.generatedAt
#   $.items[*].updatedAt
#   $.content[0].id
#
# Anything else is rejected at load time rather than silently matching nothing.
# A policy entry that matches nothing is the dangerous failure mode: it looks
# like a tolerance is in force when in fact comparison is still strict, or
# (worse) the author believes a real difference is being tolerated.
_SEGMENT = r"[A-Za-z_][A-Za-z0-9_-]*"
_PATH_RE = re.compile(rf"^\$(?:\.{_SEGMENT}|\[(?:\*|\d+)\])+$")


def parse_path(path: str) -> list[str | int]:
    """`$.items[*].updatedAt` -> ['items', '*', 'updatedAt'].

    Integer indices come back as ints, `[*]` as the literal string '*'.
    Raises ContractViolation on anything outside the supported subset.
    """
    if not _PATH_RE.match(path):
        raise ContractViolation(
            f"unsupported tolerance path {path!r}. Supported form: "
            r"$.key, $.key[*].other, $.key[0].other"
        )
    segments: list[str | int] = []
    for token in re.finditer(rf"\.({_SEGMENT})|\[(\*|\d+)\]", path):
        key, index = token.group(1), token.group(2)
        if key is not None:
            segments.append(key)
        elif index == "*":
            segments.append("*")
        else:
            segments.append(int(index))
    return segments


def load_policy(path: str | Path | None = None) -> TolerancePolicy:
    """Load the frozen tolerance policy from disk.

    There is NO argument here that widens the policy, and no `save_policy`.
    Changing what is tolerated is a deliberate human edit of
    `.perfagent/tolerance.json`, reviewed like any other change. That is the
    whole point: a tolerance written after seeing a diff is rationalising, not
    verifying.

    Every tolerated path must carry a `rationale` entry. An exception without a
    written reason is indistinguishable from an accident six months later.
    """
    policy_path = Path(path) if path is not None else DEFAULT_TOLERANCE_PATH
    if not policy_path.exists():
        raise ContractViolation(
            f"tolerance policy not found: {policy_path}. Create it with "
            '{"ignore_paths": [], "order_insensitive_paths": [], '
            '"numeric_epsilon": 0.0, "rationale": {}} — an empty policy is a '
            "decision, a missing one is an omission."
        )

    try:
        payload = json.loads(policy_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ContractViolation(f"{policy_path} is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ContractViolation(f"{policy_path} must contain a JSON object")

    # Reject unknown keys. Pydantic ignores extras by default, which would let
    # `ignore_path` (singular typo) load as a silently empty policy.
    unknown = sorted(set(payload) - set(TolerancePolicy.model_fields))
    if unknown:
        allowed = ", ".join(sorted(TolerancePolicy.model_fields))
        raise ContractViolation(
            f"{policy_path} has unknown key(s): {', '.join(unknown)}. "
            f"Allowed keys: {allowed}"
        )

    policy = TolerancePolicy.model_validate(payload)

    if policy.numeric_epsilon < 0:
        raise ContractViolation("numeric_epsilon must be >= 0")

    tolerated = [*policy.ignore_paths, *policy.order_insensitive_paths]
    for entry in tolerated:
        parse_path(entry)  # syntax check, raises with the offending path

    duplicates = sorted({p for p in tolerated if tolerated.count(p) > 1})
    if duplicates:
        raise ContractViolation(
            f"{policy_path} lists the same path twice: {', '.join(duplicates)}"
        )

    # An index underneath an order-insensitive array is a contradiction:
    # sorting moves the element that `[0]` refers to, so canonicalise() would
    # stop being idempotent and the same body would compare differently
    # depending on how often it had been canonicalised.
    for sorted_path in policy.order_insensitive_paths:
        prefix = parse_path(sorted_path)
        for other in tolerated:
            spec = parse_path(other)
            if len(spec) > len(prefix) and spec[: len(prefix)] == prefix:
                if any(isinstance(segment, int) for segment in spec[len(prefix):]):
                    raise ContractViolation(
                        f"{policy_path}: {other!r} uses a fixed index inside "
                        f"{sorted_path!r}, which is order-insensitive. Sorting "
                        "moves the element that index refers to. Use [*] or "
                        "drop the order tolerance."
                    )

    needs_reason = [*tolerated]
    if policy.numeric_epsilon > 0:
        needs_reason.append("numeric_epsilon")
    missing = [key for key in needs_reason if not policy.rationale.get(key, "").strip()]
    if missing:
        raise ContractViolation(
            f"{policy_path} tolerates {', '.join(missing)} with no rationale. "
            "Add a `rationale` entry for each, explaining why that difference "
            "is not a behaviour change."
        )

    return policy


def _matches(segment: str | int, key: str | int) -> bool:
    """Does one path segment match one dict key or list index?

    `[*]` comes out of `parse_path` as the literal string '*' and matches list
    indices only. Object keys are strings and list indices are ints, so the
    types keep the two apart without a separate flag.
    """
    if segment == "*":
        return isinstance(key, int)
    if isinstance(segment, int):
        return isinstance(key, int) and key == segment
    return isinstance(key, str) and key == segment


def _descend(specs: list[list[str | int]], key: str | int) -> list[list[str | int]]:
    """Advance every spec that matches `key` by one segment."""
    return [spec[1:] for spec in specs if spec and _matches(spec[0], key)]


def _targets_here(specs: list[list[str | int]]) -> bool:
    """True when some spec has been fully consumed — this node is the target."""
    return any(not spec for spec in specs)


def _sort_key(element: Any) -> str:
    """Total, type-stable order over arbitrary JSON.

    Sorting by the element itself raises TypeError the moment an array holds
    both a dict and an int, and that crash would surface as a mysterious oracle
    failure rather than as the policy mistake it is. Canonical JSON of the
    already-canonicalised element is comparable for every JSON value.
    """
    return json.dumps(element, sort_keys=True, ensure_ascii=False, default=str)


def _walk(
    node: Any,
    ignore_specs: list[list[str | int]],
    order_specs: list[list[str | int]],
) -> Any:
    sort_here = _targets_here(order_specs)

    if isinstance(node, dict):
        out: dict[str, Any] = {}
        for key, value in node.items():
            child_ignore = _descend(ignore_specs, key)
            if _targets_here(child_ignore):
                # Dropped, not set to None: "the patch removed this field" and
                # "policy ignores this field" must not become the same shape.
                continue
            out[key] = _walk(value, child_ignore, _descend(order_specs, key))
        return out

    if isinstance(node, list):
        items: list[Any] = []
        for index, value in enumerate(node):
            child_ignore = _descend(ignore_specs, index)
            if _targets_here(child_ignore):
                continue
            items.append(_walk(value, child_ignore, _descend(order_specs, index)))
        if sort_here:
            # Elements are already canonicalised by the recursion above, so two
            # elements differing only in an ignored field sort together instead
            # of flapping and producing phantom diffs.
            items.sort(key=_sort_key)
        return items

    # Scalars are immutable, so returning them directly keeps this pure.
    return node


def canonicalise(body: Any, policy: TolerancePolicy) -> Any:
    """Return `body` with ignored paths removed and tolerated arrays sorted.

    Pure: `body` is never mutated, because the caller still holds the recorded
    oracle body and has to be able to render it after a comparison.

    Idempotent: `canonicalise(canonicalise(x, p), p) == canonicalise(x, p)`.
    That is what makes comparison independent of how many times a body has been
    through here. `load_policy` refuses an indexed path underneath an
    order-insensitive array precisely because that combination would break it —
    sorting moves the element that `[0]` refers to.

    `policy.numeric_epsilon` is deliberately NOT applied; see the module
    docstring. A bare string, number or None passes through untouched, and a 204
    with no body stays None.
    """
    ignore_specs = [parse_path(p) for p in policy.ignore_paths]
    order_specs = [parse_path(p) for p in policy.order_insensitive_paths]
    if not ignore_specs and not order_specs:
        # The starter policy tolerates nothing, which is the common case: skip
        # the walk entirely rather than deep-copying every body for no reason.
        return body
    return _walk(body, ignore_specs, order_specs)


def _resolves(node: Any, spec: list[str | int]) -> bool:
    """Does `spec` reach an existing node inside `node`?"""
    if not spec:
        return True
    head, rest = spec[0], spec[1:]
    if isinstance(node, dict):
        return any(_matches(head, k) and _resolves(v, rest) for k, v in node.items())
    if isinstance(node, list):
        return any(_matches(head, i) and _resolves(v, rest) for i, v in enumerate(node))
    return False


def unmatched_policy_paths(bodies: Iterable[Any], policy: TolerancePolicy) -> list[str]:
    """Policy paths that matched nothing in any of `bodies`.

    A tolerance that matches nothing is the dangerous failure mode: it reads as
    "this difference is accounted for" while comparison is in fact still strict
    on that field, or — worse — while the field has been renamed and the real
    difference is now unguarded. Advisory only; it never changes a verdict, it
    just refuses to let a stale policy sit there looking effective.
    """
    materialised = list(bodies)
    unmatched: list[str] = []
    for path in [*policy.ignore_paths, *policy.order_insensitive_paths]:
        spec = parse_path(path)
        if not any(_resolves(body, spec) for body in materialised):
            unmatched.append(path)
    return unmatched
